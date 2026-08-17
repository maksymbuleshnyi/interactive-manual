"""CLI: task spec -> outputs/<task>/demo.json with an LLM-written procedure.

    python scripts/create_demo.py --task data/tasks/wiper_blade.json --out outputs/wiper-001
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import typer  # noqa: E402

from interactive_interfaces.models.registry import get_llm  # noqa: E402
from interactive_interfaces.pipeline.create_demo import create_demo  # noqa: E402
from interactive_interfaces.utils.io import load_task, save_demo  # noqa: E402
from interactive_interfaces.utils.logging import Run  # noqa: E402


def main(
    task: Path = typer.Option(..., help="Task spec JSON under data/tasks/."),
    out: Path = typer.Option(..., help="Output directory for the demonstration."),
    llm: str = typer.Option("mock", help="LLM adapter name."),
    run_name: str | None = typer.Option(None, "--run-name", help="Run id suffix."),
) -> None:
    with Run(
        command="create_demo",
        argv=sys.argv,
        run_name=run_name,
        adapters={"llm": llm},
        input_files=[task],
    ) as run:
        demo = create_demo(load_task(task), out_dir=out, llm=get_llm(llm))
        demo.run_log_path = run.run_dir
        demo_path = save_demo(demo, Path(out) / "demo.json")
        print(f"wrote {demo_path}  ({len(demo.steps)} steps)")
        print(f"run_id = {run.run_id}")


if __name__ == "__main__":
    typer.run(main)
