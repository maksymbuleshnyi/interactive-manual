"""Stage 1 acceptance tests: the mock pipeline runs end-to-end."""

import json

import pytest

from interactive_interfaces.models.registry import (
    get_critic,
    get_image_editor,
    get_llm,
)
from interactive_interfaces.pipeline.chain import chain
from interactive_interfaces.pipeline.create_demo import create_demo
from interactive_interfaces.pipeline.critique import critique_demo
from interactive_interfaces.pipeline.decompose import Violation, atomic_lint, decompose
from interactive_interfaces.schemas.demonstration import Step
from interactive_interfaces.utils.logging import Run

TASK = {
    "task_id": "test-bike-001",
    "domain": "physical_repair",
    "user_goal": "Replace a bike inner tube.",
    "source_type": "manually_written",
}


def test_mock_pipeline_end_to_end(tmp_path):
    out_dir = tmp_path / "out"
    with Run(command="test", logs_root=tmp_path / "logs") as run:
        demo = create_demo(TASK, out_dir=out_dir, llm=get_llm("mock"))
        demo = decompose(demo, llm=get_llm("mock"))
        demo = chain(
            demo,
            llm=get_llm("mock"),
            editor=get_image_editor("mock"),
            out_dir=out_dir,
            max_steps=5,
        )
        demo = critique_demo(demo, critic=get_critic("mock"))
        run_dir = run.run_dir

    # Acceptance: >=3 steps, every step has a generated image on disk.
    assert len(demo.steps) >= 3
    for step in demo.steps:
        assert step.generated_next_image_path is not None
        assert step.generated_next_image_path.exists()
        assert step.critique is not None
    assert demo.overall_quality_score == 3.0

    # Acceptance: run.jsonl has one event per pipeline (model) call.
    events = [json.loads(x) for x in (run_dir / "run.jsonl").read_text().splitlines()]
    n = len(demo.steps)
    expected = 1 + 1 + 2 * n + n  # create + decompose + (llm+editor)/step + critic/step
    assert len(events) == expected
    assert all(e["ok"] for e in events)
    assert {e["stage"] for e in events} >= {
        "create_demo",
        "decompose",
        "image_edit_prompt",
        "generate_image",
        "critique",
    }


def test_step_current_image_chains_forward(tmp_path):
    out_dir = tmp_path / "out"
    with Run(command="test", logs_root=tmp_path / "logs"):
        demo = create_demo(TASK, out_dir=out_dir, llm=get_llm("mock"))
        demo = decompose(demo, llm=get_llm("mock"))
        demo = chain(
            demo,
            llm=get_llm("mock"),
            editor=get_image_editor("mock"),
            out_dir=out_dir,
            max_steps=5,
        )
    # Step k's input image is step k-1's generated output.
    for i in range(1, len(demo.steps)):
        assert demo.steps[i].current_image_path == demo.steps[i - 1].generated_next_image_path


def test_atomic_lint_flags_conjunctions_and_mental_verbs():
    bad = Step(
        step_index=0,
        current_image_path="x.png",
        current_state_description="s",
        natural_language_instruction="Lift the arm and decide what to do",
        expected_user_action="lift",
        image_generation_prompt="",
    )
    violations = atomic_lint(bad)
    assert {v.rule for v in violations} == {"R4", "R5"}
    assert all(isinstance(v, Violation) for v in violations)


def test_atomic_lint_passes_a_clean_instruction():
    good = Step(
        step_index=0,
        current_image_path="x.png",
        current_state_description="s",
        natural_language_instruction="Raise the wiper arm to the upright position",
        expected_user_action="raise the arm",
        image_generation_prompt="",
    )
    assert atomic_lint(good) == []


def test_unknown_adapter_name_is_rejected():
    with pytest.raises(ValueError):
        get_llm("gpt-nonexistent")
