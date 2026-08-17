"""Mock critic adapter: returns a fixed neutral rubric score.

Stands in for the LLM critic. Every criterion is scored 3/5 so downstream
aggregation code has real CritiqueResult objects to work with, without any
claim of real evaluation.
"""

from __future__ import annotations

import time
from pathlib import Path

from interactive_interfaces.schemas.demonstration import CritiqueResult
from interactive_interfaces.utils.logging import (
    file_sha256,
    get_call_context,
    get_current_run,
    hash_text,
)


class MockCritic:
    """Deterministic stand-in for a CriticClient."""

    name = "mock_critic"
    FIXED_SCORE = 3

    def evaluate(
        self,
        *,
        instruction: str,
        before_image: Path,
        after_image: Path,
        expected_next_state: str,
    ) -> CritiqueResult:
        t0 = time.perf_counter()
        score = self.FIXED_SCORE
        result = CritiqueResult(
            instruction_clarity=score,
            visual_correctness=score,
            image_faithfulness=score,
            illustrates_next_step=score,
            irrelevant_detail_preservation=score,
            safety=score,
            overall=score,
            rationale="Mock critic: fixed neutral score; no real evaluation performed.",
            model=self.name,
        )
        latency_ms = int((time.perf_counter() - t0) * 1000)
        self._log(before_image, after_image, instruction, result, latency_ms)
        return result

    def _log(
        self,
        before_image: Path,
        after_image: Path,
        instruction: str,
        result: CritiqueResult,
        latency_ms: int,
    ) -> None:
        run = get_current_run()
        if run is None:
            return
        ctx = get_call_context()
        before_image, after_image = Path(before_image), Path(after_image)
        run.log_call(
            stage=ctx.stage if ctx else "critic",
            task_id=ctx.task_id if ctx else None,
            step_index=ctx.step_index if ctx else None,
            adapter=self.name,
            input_hash=hash_text(instruction),
            output_hash=hash_text(result.model_dump_json()),
            latency_ms=latency_ms,
            raw_kind="critic",
            raw={
                "model": self.name,
                "instruction": instruction,
                "before_image": str(before_image),
                "before_hash": file_sha256(before_image) if before_image.exists() else None,
                "after_image": str(after_image),
                "after_hash": file_sha256(after_image) if after_image.exists() else None,
                "critique": result.model_dump(),
            },
        )
