"""CLI: generate the next-state image for one step.

    python scripts/generate_next_image.py --demo outputs/wiper-001/demo.json --step 0
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import typer  # noqa: E402

from interactive_interfaces.models.registry import get_image_editor, get_llm  # noqa: E402
from interactive_interfaces.pipeline.generate_image import generate_next_image  # noqa: E402
from interactive_interfaces.utils.io import load_demo, save_demo  # noqa: E402
from interactive_interfaces.utils.logging import Run  # noqa: E402


def main(
    demo: Path = typer.Option(..., help="Path to demo.json."),
    step: int = typer.Option(..., help="Step index to generate."),
    llm: str = typer.Option("mock", help="LLM adapter name."),
    editor: str = typer.Option("mock", help="Image-edit adapter name."),
    run_name: str | None = typer.Option(None, "--run-name", help="Run id suffix."),
) -> None:
    with Run(
        command="generate_next_image",
        argv=sys.argv,
        run_name=run_name,
        adapters={"llm": llm, "editor": editor},
        input_files=[demo],
    ) as run:
        d = generate_next_image(
            load_demo(demo),
            step,
            llm=get_llm(llm),
            editor=get_image_editor(editor),
            out_dir=Path(demo).parent,
        )
        d.run_log_path = run.run_dir
        save_demo(d, demo)
        print(f"step {step} -> {d.steps[step].generated_next_image_path}")
        print(f"run_id = {run.run_id}")


if __name__ == "__main__":
    typer.run(main)
