"""CLI: run the critic over every generated step of a demo.

    python scripts/critique_demo.py --demo outputs/wiper-001/demo.json
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import typer  # noqa: E402

from interactive_interfaces.models.registry import get_critic  # noqa: E402
from interactive_interfaces.pipeline.critique import critique_demo  # noqa: E402
from interactive_interfaces.utils.io import load_demo, save_demo  # noqa: E402
from interactive_interfaces.utils.logging import Run  # noqa: E402


def main(
    demo: Path = typer.Option(..., help="Path to demo.json."),
    critic: str = typer.Option("mock", help="Critic adapter name."),
    run_name: str | None = typer.Option(None, "--run-name", help="Run id suffix."),
) -> None:
    with Run(
        command="critique_demo",
        argv=sys.argv,
        run_name=run_name,
        adapters={"critic": critic},
        input_files=[demo],
    ) as run:
        d = critique_demo(load_demo(demo), critic=get_critic(critic))
        d.run_log_path = run.run_dir
        save_demo(d, demo)
        print(f"critiqued {len(d.steps)} steps; overall = {d.overall_quality_score}")
        print(f"run_id = {run.run_id}")


if __name__ == "__main__":
    typer.run(main)
