"""Demonstration schema v0.1 - see PLAN.md section C.

A *demonstration* is one task turned into an ordered list of atomic visual
steps. It is persisted as ``demo.json``. A flattened JSONL manifest (one row
per step) is built from these for batch evaluation in later stages.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = "0.1"

Domain = Literal["software_ui", "physical_repair", "assembly", "education", "other"]
SourceType = Literal[
    "manually_written",
    "model_generated",
    "user_uploaded",
    "public_task_description",
    "synthetic",
]


class Provenance(BaseModel):
    """Where a demonstration came from and under what terms it can be used."""

    author: str
    date: str  # ISO date the demonstration was authored
    license: str = "CC0-1.0"
    notes: str = ""


class CritiqueResult(BaseModel):
    """Rubric scores for one generated next-step image.

    The same fields are produced by the LLM critic prompt and by the human
    review CLI, so machine and human scores are directly comparable.
    All criterion scores are on a 1-5 scale (5 best).
    """

    instruction_clarity: int = Field(ge=1, le=5)
    visual_correctness: int = Field(ge=1, le=5)
    image_faithfulness: int = Field(ge=1, le=5)
    illustrates_next_step: int = Field(ge=1, le=5)
    irrelevant_detail_preservation: int = Field(ge=1, le=5)
    safety: int = Field(ge=1, le=5)
    overall: int = Field(ge=1, le=5)
    rationale: str
    model: str


class Step(BaseModel):
    """One atomic visual step: a single observable state change."""

    step_index: int = Field(ge=0)
    current_image_path: Path
    current_state_description: str
    natural_language_instruction: str  # shown to the user
    expected_user_action: str  # observable user action, verb phrase
    image_generation_prompt: str  # prompt fed to the image editor
    image_model: str | None = None
    image_model_params: dict = Field(default_factory=dict)
    generated_next_image_path: Path | None = None
    next_state_description: str | None = None
    safety_notes: list[str] = Field(default_factory=list)
    critique: CritiqueResult | None = None
    failure_modes: list[str] = Field(default_factory=list)
    human_review_status: Literal["pending", "approved", "rejected", "edited"] = "pending"
    quality_score: float | None = Field(default=None, ge=0.0, le=5.0)


class Demonstration(BaseModel):
    """One task's full demonstration. Persisted as ``demo.json``."""

    schema_version: Literal["0.1"] = "0.1"
    task_id: str  # slug, unique
    domain: Domain
    user_goal: str
    initial_image_path: Path | None = None
    initial_state_description: str | None = None
    procedure_text: str | None = None  # full LLM narrative
    steps: list[Step] = Field(default_factory=list)
    source_type: SourceType
    provenance: Provenance
    safety_notes: list[str] = Field(default_factory=list)
    overall_quality_score: float | None = Field(default=None, ge=0.0, le=5.0)
    overall_review_status: Literal["pending", "approved", "rejected"] = "pending"
    tags: list[str] = Field(default_factory=list)
    run_log_path: Path | None = None  # pointer to logs/runs/<run_id>/
