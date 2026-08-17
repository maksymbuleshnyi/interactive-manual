"""CLI: generate images for every step, feeding each output forward.

    python scripts/chain_steps.py --demo outputs/wiper-001/demo.json --max-steps 5
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import typer  # noqa: E402

from interactive_interfaces.models.registry import get_image_editor, get_llm  # noqa: E402
from interactive_interfaces.pipeline.chain import chain  # noqa: E402
from interactive_interfaces.utils.io import load_demo, save_demo  # noqa: E402
from interactive_interfaces.utils.logging import Run  # noqa: E402


def main(
    demo: Path = typer.Option(..., help="Path to demo.json."),
    max_steps: int = typer.Option(5, help="Maximum steps to chain."),
    llm: str = typer.Option("mock", help="LLM adapter name."),
    editor: str = typer.Option("mock", help="Image-edit adapter name."),
    run_name: str | None = typer.Option(None, "--run-name", help="Run id suffix."),
) -> None:
    with Run(
        command="chain_steps",
        argv=sys.argv,
        run_name=run_name,
        adapters={"llm": llm, "editor": editor},
        input_files=[demo],
    ) as run:
        d = chain(
            load_demo(demo),
            llm=get_llm(llm),
            editor=get_image_editor(editor),
            out_dir=Path(demo).parent,
            max_steps=max_steps,
        )
        d.run_log_path = run.run_dir
        save_demo(d, demo)
        done = sum(1 for s in d.steps if s.generated_next_image_path is not None)
        print(f"chained {done}/{len(d.steps)} steps -> {demo}")
        print(f"run_id = {run.run_id}")


if __name__ == "__main__":
    typer.run(main)
