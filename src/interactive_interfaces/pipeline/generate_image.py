"""Stage: one atomic step -> the generated next-state image.

Two model calls: the LLM writes a minimal-change edit prompt, then the image
editor applies it. The step's ``current_image_path`` is resolved here - the
initial image for step 0, the prior step's generated image otherwise.
"""

from __future__ import annotations

from pathlib import Path

from interactive_interfaces.models.base import ImageEditClient, LLMClient
from interactive_interfaces.schemas.demonstration import Demonstration
from interactive_interfaces.utils.logging import call_context
from interactive_interfaces.utils.prompts import render


def generate_next_image(
    demo: Demonstration,
    step_index: int,
    *,
    llm: LLMClient,
    editor: ImageEditClient,
    out_dir: Path | str,
) -> Demonstration:
    """Generate ``step_index``'s next image. Mutates and returns ``demo``."""
    out_dir = Path(out_dir)
    step = demo.steps[step_index]

    if step_index == 0:
        current = demo.initial_image_path
    else:
        current = demo.steps[step_index - 1].generated_next_image_path
    if current is None:
        raise ValueError(
            f"step {step_index}: no current image - run earlier steps first"
        )
    step.current_image_path = current

    edit_prompt = render(
        "image_edit_prompt",
        current_state_description=step.current_state_description,
        next_state_description=step.next_state_description or "",
        natural_language_instruction=step.natural_language_instruction,
    )
    with call_context("image_edit_prompt", task_id=demo.task_id, step_index=step_index):
        step.image_generation_prompt = llm.generate(prompt=edit_prompt)

    output_path = out_dir / f"step_{step_index}.png"
    with call_context("generate_image", task_id=demo.task_id, step_index=step_index):
        editor.generate_next_image(
            input_image=Path(current),
            instruction=step.image_generation_prompt,
            output_path=output_path,
        )
    step.generated_next_image_path = output_path
    step.image_model = editor.name
    return demo
