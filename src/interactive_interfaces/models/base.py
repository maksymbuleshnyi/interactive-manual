"""Vendor-agnostic model adapter protocols (PLAN.md section "Model adapter
interfaces"). All model access in the pipeline goes through these, so swapping
a mock for a real vendor is a single line in ``registry.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from interactive_interfaces.schemas.demonstration import CritiqueResult


@runtime_checkable
class LLMClient(Protocol):
    """Text-in, text-out language model."""

    name: str

    def generate(
        self,
        *,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.2,
        **kwargs: object,
    ) -> str: ...


@runtime_checkable
class ImageEditClient(Protocol):
    """Edits a current image into the next-state image."""

    name: str

    def generate_next_image(
        self,
        *,
        input_image: Path,
        instruction: str,
        output_path: Path,
        **kwargs: object,
    ) -> Path: ...


@runtime_checkable
class CriticClient(Protocol):
    """Scores a generated next-step image against the rubric."""

    name: str

    def evaluate(
        self,
        *,
        instruction: str,
        before_image: Path,
        after_image: Path,
        expected_next_state: str,
    ) -> CritiqueResult: ...
