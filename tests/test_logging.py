"""Stage 0 logging tests - verifies the Run context manager and log_call."""

import json

import pytest

from interactive_interfaces.utils.logging import Run, hash_text


def test_run_creates_populated_run_dir(tmp_path):
    with Run(command="test", argv=["test"], logs_root=tmp_path / "logs") as run:
        run.log_call(
            stage="unit",
            adapter="none",
            input_hash=hash_text("a"),
            output_hash=hash_text("b"),
            latency_ms=1,
        )
        run_dir = run.run_dir

    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "run.jsonl").exists()
    assert (run_dir / "stdout.log").exists()
    assert (run_dir / "stderr.log").exists()

    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["run_id"] == run.run_id
    assert manifest["command"] == "test"
    assert "python_version" in manifest

    events = (run_dir / "run.jsonl").read_text().splitlines()
    assert len(events) == 1
    event = json.loads(events[0])
    assert event["stage"] == "unit"
    assert event["call_id"] == 1
    assert event["ok"] is True


def test_call_ids_are_monotonic(tmp_path):
    with Run(command="test", logs_root=tmp_path / "logs") as run:
        first = run.log_call(stage="a")
        second = run.log_call(stage="b")
    assert (first, second) == (1, 2)


def test_raw_payload_is_persisted(tmp_path):
    with Run(command="test", logs_root=tmp_path / "logs") as run:
        call_id = run.log_call(
            stage="decompose",
            adapter="mock_llm",
            raw_kind="llm",
            raw={"prompt": "hi", "response": "ok"},
        )
        raw_file = run.raw_dir / "llm" / f"{call_id:04d}.json"

    assert raw_file.exists()
    payload = json.loads(raw_file.read_text())
    assert payload == {"prompt": "hi", "response": "ok"}


def test_bad_raw_kind_is_rejected(tmp_path):
    with Run(command="test", logs_root=tmp_path / "logs") as run:
        with pytest.raises(ValueError):
            run.log_call(stage="x", raw={"k": "v"}, raw_kind="not_a_kind")


def test_exception_is_logged_and_not_suppressed(tmp_path):
    with pytest.raises(RuntimeError):
        with Run(command="test", logs_root=tmp_path / "logs") as run:
            raise RuntimeError("boom")

    events = (run.run_dir / "run.jsonl").read_text().splitlines()
    last = json.loads(events[-1])
    assert last["ok"] is False
    assert "boom" in last["error"]
