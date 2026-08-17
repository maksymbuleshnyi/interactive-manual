"""Stage 0 schema tests - see PLAN.md section E acceptance check."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from interactive_interfaces.schemas.demonstration import (
    CritiqueResult,
    Demonstration,
    Step,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
WIPER_DEMO = REPO_ROOT / "data" / "examples" / "wiper-001" / "demo.json"


def test_wiper_demo_loads_and_validates():
    demo = Demonstration.model_validate_json(WIPER_DEMO.read_text())
    assert demo.task_id == "wiper-001"
    assert demo.domain == "physical_repair"
    assert demo.schema_version == "0.1"
    assert demo.source_type == "manually_written"
    assert len(demo.steps) >= 1
    assert demo.steps[0].generated_next_image_path is not None


def test_demo_round_trips_through_json():
    demo = Demonstration.model_validate_json(WIPER_DEMO.read_text())
    again = Demonstration.model_validate_json(demo.model_dump_json())
    assert again == demo


def test_minimal_demonstration_constructs():
    demo = Demonstration(
        task_id="t1",
        domain="other",
        user_goal="do a thing",
        source_type="manually_written",
        provenance={"author": "tester", "date": "2026-05-17"},
    )
    assert demo.steps == []
    assert demo.overall_review_status == "pending"
    assert demo.provenance.license == "CC0-1.0"


def test_critique_scores_must_be_within_1_to_5():
    with pytest.raises(ValidationError):
        CritiqueResult(
            instruction_clarity=6,  # out of range
            visual_correctness=3,
            image_faithfulness=3,
            illustrates_next_step=3,
            irrelevant_detail_preservation=3,
            safety=3,
            overall=3,
            rationale="x",
            model="mock",
        )


def test_step_requires_core_fields():
    with pytest.raises(ValidationError):
        Step(step_index=0)  # missing required fields


def test_unknown_domain_is_rejected():
    with pytest.raises(ValidationError):
        Demonstration(
            task_id="t2",
            domain="cooking",  # not in the Domain literal
            user_goal="g",
            source_type="manually_written",
            provenance={"author": "tester", "date": "2026-05-17"},
        )
