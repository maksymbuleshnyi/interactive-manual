"""Stage: run the critic over every step that has a generated image."""

from __future__ import annotations

from interactive_interfaces.models.base import CriticClient
from interactive_interfaces.schemas.demonstration import Demonstration
from interactive_interfaces.utils.logging import call_context


def critique_demo(demo: Demonstration, *, critic: CriticClient) -> Demonstration:
    """Score each generated step and set the demo's overall quality score."""
    overall_scores: list[int] = []
    for step in demo.steps:
        if step.generated_next_image_path is None:
            continue
        with call_context("critique", task_id=demo.task_id, step_index=step.step_index):
            result = critic.evaluate(
                instruction=step.natural_language_instruction,
                before_image=step.current_image_path,
                after_image=step.generated_next_image_path,
                expected_next_state=step.next_state_description or "",
            )
        step.critique = result
        step.quality_score = float(result.overall)
        overall_scores.append(result.overall)

    if overall_scores:
        demo.overall_quality_score = sum(overall_scores) / len(overall_scores)
    return demo
