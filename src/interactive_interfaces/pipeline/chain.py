"""Stage: chain generate_next_image across steps, feeding each image forward.

This recursive feed (each generated image becomes the next step's input) is
what Experiment 3 measures for drift. In Stage 1 the mock editor is a no-op on
content, so the chain is a structural dry run.
"""

from __future__ import annotations

from pathlib import Path

from interactive_interfaces.models.base import ImageEditClient, LLMClient
from interactive_interfaces.pipeline.generate_image import generate_next_image
from interactive_interfaces.schemas.demonstration import Demonstration


def chain(
    demo: Demonstration,
    *,
    llm: LLMClient,
    editor: ImageEditClient,
    out_dir: Path | str,
    max_steps: int = 5,
) -> Demonstration:
    """Generate images for steps 0..min(len(steps), max_steps), in order."""
    count = min(len(demo.steps), max_steps)
    for i in range(count):
        generate_next_image(demo, i, llm=llm, editor=editor, out_dir=out_dir)
    return demo
