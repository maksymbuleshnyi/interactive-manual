"""Stage 0 smoke CLI: open a Run, log one call, exit.

Exists to satisfy the Stage 0 acceptance check "running any CLI script writes
a populated logs/runs/<run_id>/". The real pipeline CLIs arrive in Stage 1.

    python scripts/check_logging.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from a checkout without `pip install -e .`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from interactive_interfaces.utils.logging import Run, hash_text  # noqa: E402


def main() -> None:
    with Run(command="check_logging", argv=sys.argv, run_name="smoke") as run:
        print(f"run_id = {run.run_id}")
        run.log_call(
            stage="smoke",
            adapter="none",
            input_hash=hash_text("hello"),
            output_hash=hash_text("world"),
            latency_ms=0,
        )
        print(f"wrote {run.run_dir}/ (manifest.json, run.jsonl, std*.log)")


if __name__ == "__main__":
    main()
