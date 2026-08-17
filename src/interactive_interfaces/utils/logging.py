"""Run logging - see the "Logging and experiment tracking" section of PLAN.md.

Principle: every run and every model call must be reproducible from logs
alone. If it isn't written to a file, it didn't happen.

A ``Run`` owns one ``logs/runs/<run_id>/`` directory. It writes a manifest,
tees stdout/stderr to files, and records one JSON event per pipeline call via
``log_call()``. Full prompt/response payloads go to ``logs/raw/<run_id>/``.

Usage::

    with Run(command="decompose_steps", argv=sys.argv) as run:
        run.log_call(stage="decompose", adapter="mock_llm", ...)
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import subprocess
import sys
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, TextIO

RAW_KINDS = ("llm", "image_edit", "critic")


@dataclass(frozen=True)
class CallContext:
    """Pipeline-stage metadata: set by the pipeline, read by adapters.

    Adapters call :meth:`Run.log_call` themselves, but cannot know which
    pipeline stage or step they were invoked for. The pipeline marks that with
    :func:`call_context`; the adapter reads it back via :func:`get_call_context`.
    """

    stage: str
    task_id: str | None = None
    step_index: int | None = None


_current_run: contextvars.ContextVar["Run | None"] = contextvars.ContextVar(
    "interactive_interfaces_current_run", default=None
)
_call_ctx: contextvars.ContextVar[CallContext | None] = contextvars.ContextVar(
    "interactive_interfaces_call_ctx", default=None
)


def get_current_run() -> "Run | None":
    """Return the Run for the active ``with Run(...)`` block, or None."""
    return _current_run.get()


def get_call_context() -> CallContext | None:
    """Return the active :class:`CallContext`, or None outside a stage."""
    return _call_ctx.get()


@contextmanager
def call_context(
    stage: str, *, task_id: str | None = None, step_index: int | None = None
) -> Iterator[CallContext]:
    """Mark the current pipeline stage so adapter log events are labelled."""
    ctx = CallContext(stage, task_id, step_index)
    token = _call_ctx.set(ctx)
    try:
        yield ctx
    finally:
        _call_ctx.reset(token)


def _utc_compact() -> str:
    """UTC timestamp as a filesystem-safe compact string."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path | str) -> str:
    """Hex SHA-256 of a file's contents, read in chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_text(text: str) -> str:
    """Hex SHA-256 of a UTF-8 string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _git_sha() -> str | None:
    """Current commit SHA, or None if not in a git repo / git unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


class _Tee:
    """File-like wrapper that writes to the original stream and a mirror file."""

    def __init__(self, stream: TextIO, mirror: TextIO) -> None:
        self._stream = stream
        self._mirror = mirror

    def write(self, data: str) -> int:
        self._mirror.write(data)
        return self._stream.write(data)

    def flush(self) -> None:
        self._mirror.flush()
        self._stream.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


class Run:
    """One logged run. Use as a context manager.

    Adapters receive the active ``Run`` and call ``log_call()`` themselves, so
    pipeline code stays free of logging boilerplate.
    """

    def __init__(
        self,
        command: str,
        *,
        argv: list[str] | None = None,
        run_name: str | None = None,
        logs_root: Path | str = "logs",
        adapters: dict[str, str] | None = None,
        env_var_names: list[str] | None = None,
        input_files: list[Path | str] | None = None,
    ) -> None:
        self.command = command
        self.argv = list(argv) if argv is not None else list(sys.argv)
        self.run_name = run_name
        self.logs_root = Path(logs_root)
        self.adapters = adapters or {}
        self.env_var_names = env_var_names or []  # names only, never values
        self.input_files = [Path(p) for p in (input_files or [])]

        suffix = f"-{run_name}" if run_name else ""
        self.run_id = f"{_utc_compact()}-{uuid.uuid4().hex[:8]}{suffix}"
        self.run_dir = self.logs_root / "runs" / self.run_id
        self.raw_dir = self.logs_root / "raw" / self.run_id

        self._call_counter = 0
        self._events_fp: TextIO | None = None
        self._stdout_fp: TextIO | None = None
        self._stderr_fp: TextIO | None = None
        self._orig_stdout: TextIO | None = None
        self._orig_stderr: TextIO | None = None
        self._run_token: contextvars.Token | None = None

    # -- lifecycle ---------------------------------------------------------

    def __enter__(self) -> "Run":
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self._write_manifest()
        self._events_fp = open(self.run_dir / "run.jsonl", "a", encoding="utf-8")
        self._stdout_fp = open(self.run_dir / "stdout.log", "a", encoding="utf-8")
        self._stderr_fp = open(self.run_dir / "stderr.log", "a", encoding="utf-8")
        self._orig_stdout, self._orig_stderr = sys.stdout, sys.stderr
        sys.stdout = _Tee(self._orig_stdout, self._stdout_fp)
        sys.stderr = _Tee(self._orig_stderr, self._stderr_fp)
        self._run_token = _current_run.set(self)
        return self

    def __exit__(self, exc_type: object, exc: BaseException | None, tb: object) -> bool:
        if exc is not None:
            self.log_call(stage="__run__", ok=False, error=repr(exc))
        if self._orig_stdout is not None:
            sys.stdout = self._orig_stdout
        if self._orig_stderr is not None:
            sys.stderr = self._orig_stderr
        if self._run_token is not None:
            _current_run.reset(self._run_token)
            self._run_token = None
        for fp in (self._events_fp, self._stdout_fp, self._stderr_fp):
            if fp is not None:
                fp.flush()
                fp.close()
        return False  # never suppress exceptions

    # -- manifest ----------------------------------------------------------

    def _write_manifest(self) -> None:
        manifest = {
            "run_id": self.run_id,
            "timestamp": _now_iso(),
            "git_sha": _git_sha(),
            "python_version": sys.version.split()[0],
            "command": self.command,
            "argv": self.argv,
            "adapters": self.adapters,
            "env_var_names": self.env_var_names,  # names only, never values
            "input_file_hashes": {
                str(p): (file_sha256(p) if p.exists() else None)
                for p in self.input_files
            },
        }
        (self.run_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )

    # -- per-call logging --------------------------------------------------

    def log_call(
        self,
        stage: str,
        *,
        task_id: str | None = None,
        step_index: int | None = None,
        adapter: str | None = None,
        input_hash: str | None = None,
        output_hash: str | None = None,
        latency_ms: int | None = None,
        ok: bool = True,
        error: str | None = None,
        raw: dict[str, Any] | None = None,
        raw_kind: str | None = None,
    ) -> int:
        """Record one pipeline call as a ``run.jsonl`` event. Returns its call_id.

        If ``raw`` is given, the full payload (prompt, response, image paths,
        params, latency, token counts) is written to
        ``logs/raw/<run_id>/<raw_kind>/<call_id>.json`` and referenced from the
        event. ``raw_kind`` must be one of :data:`RAW_KINDS`.
        """
        if self._events_fp is None:
            raise RuntimeError("log_call() used outside a `with Run(...)` block")

        self._call_counter += 1
        call_id = self._call_counter

        raw_path: str | None = None
        if raw is not None:
            if raw_kind not in RAW_KINDS:
                raise ValueError(
                    f"raw_kind must be one of {RAW_KINDS}, got {raw_kind!r}"
                )
            kind_dir = self.raw_dir / raw_kind
            kind_dir.mkdir(parents=True, exist_ok=True)
            raw_file = kind_dir / f"{call_id:04d}.json"
            raw_file.write_text(
                json.dumps(raw, indent=2, default=str) + "\n", encoding="utf-8"
            )
            raw_path = str(raw_file)

        event = {
            "ts": _now_iso(),
            "run_id": self.run_id,
            "call_id": call_id,
            "stage": stage,
            "task_id": task_id,
            "step_index": step_index,
            "adapter": adapter,
            "input_hash": input_hash,
            "output_hash": output_hash,
            "latency_ms": latency_ms,
            "ok": ok,
            "error": error,
            "raw_path": raw_path,
        }
        self._events_fp.write(json.dumps(event) + "\n")
        self._events_fp.flush()
        return call_id
