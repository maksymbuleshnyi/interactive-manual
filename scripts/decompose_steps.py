"""CLI: fill a demo's steps[] with atomic visual steps (no images yet).

    python scripts/decompose_steps.py --demo outputs/wiper-001/demo.json
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import typer  # noqa: E402

from interactive_interfaces.models.registry import get_llm  # noqa: E402
from interactive_interfaces.pipeline.decompose import decompose  # noqa: E402
from interactive_interfaces.utils.io import load_demo, save_demo  # noqa: E402
from interactive_interfaces.utils.logging import Run  # noqa: E402


def main(
    demo: Path = typer.Option(..., help="Path to demo.json."),
    max_steps: int = typer.Option(8, help="Maximum atomic steps to keep."),
    llm: str = typer.Option("mock", help="LLM adapter name."),
    run_name: str | None = typer.Option(None, "--run-name", help="Run id suffix."),
) -> None:
    with Run(
        command="decompose_steps",
        argv=sys.argv,
        run_name=run_name,
        adapters={"llm": llm},
        input_files=[demo],
    ) as run:
        d = decompose(load_demo(demo), llm=get_llm(llm), max_steps=max_steps)
        d.run_log_path = run.run_dir
        save_demo(d, demo)
        flagged = sum(1 for s in d.steps if s.failure_modes)
        print(f"decomposed into {len(d.steps)} steps ({flagged} lint-flagged) -> {demo}")
        print(f"run_id = {run.run_id}")


if __name__ == "__main__":
    typer.run(main)
