"""Stage: procedure text -> atomic visual Steps.

Also hosts ``atomic_lint`` - the programmatic check of the atomic-step rules
(PLAN.md). Stage 1 records violations as ``failure_modes``; the violation ->
LLM-retry loop is enabled in Stage 2, where a real LLM makes a retry meaningful
(a mock LLM is deterministic, so retrying it is a no-op).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from interactive_interfaces.models.base import LLMClient
from interactive_interfaces.schemas.demonstration import Demonstration, Step
from interactive_interfaces.utils.io import parse_json_response
from interactive_interfaces.utils.logging import call_context
from interactive_interfaces.utils.prompts import render

# Non-visual verbs (rule R4) and action-joining conjunctions (rule R5).
_MENTAL_VERBS = ("decide", "verify", "remember", "wait", "think", "consider")
_CONJUNCTIONS = (" and ", " then ", " while ")


@dataclass(frozen=True)
class Violation:
    """One atomic-step rule failure found by :func:`atomic_lint`."""

    rule: str
    message: str


def atomic_lint(step: Step) -> list[Violation]:
    """Check the mechanically-decidable subset of the atomic-step rules.

    Semantic rules (one visible change, identifiable subject, blind-viewer
    test) need a model and are left to the critic; this catches the textual
    ones: joined actions (R5) and non-visual verbs (R4).
    """
    text = step.natural_language_instruction.lower()
    violations: list[Violation] = []
    for conj in _CONJUNCTIONS:
        if conj in text:
            violations.append(
                Violation("R5", f"instruction joins actions with '{conj.strip()}'")
            )
    for verb in _MENTAL_VERBS:
        if re.search(rf"\b{verb}\b", text):
            violations.append(
                Violation("R4", f"instruction uses non-visual verb '{verb}'")
            )
    return violations


def decompose(
    demo: Demonstration, *, llm: LLMClient, max_steps: int = 8
) -> Demonstration:
    """Fill ``demo.steps`` with atomic visual steps. Mutates and returns ``demo``."""
    procedure = [line for line in (demo.procedure_text or "").splitlines() if line.strip()]
    prompt = render("decompose_to_atomic_steps", procedure_json=json.dumps(procedure))

    with call_context("decompose", task_id=demo.task_id):
        raw = llm.generate(prompt=prompt)
    parsed = parse_json_response(raw)[:max_steps]

    steps: list[Step] = []
    for i, item in enumerate(parsed):
        step = Step(
            step_index=i,
            # current_image_path is provisional; generate_image fixes it to the
            # real prior image once images exist.
            current_image_path=demo.initial_image_path or "UNSET",
            current_state_description=item.get("current_state_description", ""),
            natural_language_instruction=item.get("natural_language_instruction", ""),
            expected_user_action=item.get("expected_user_action", ""),
            image_generation_prompt="",  # filled by generate_image
            next_state_description=item.get("next_state_description"),
            safety_notes=item.get("safety_notes", []),
        )
        violations = atomic_lint(step)
        if violations:
            step.failure_modes = [
                f"atomic_lint {v.rule}: {v.message}" for v in violations
            ]
        steps.append(step)

    demo.steps = steps
    return demo
