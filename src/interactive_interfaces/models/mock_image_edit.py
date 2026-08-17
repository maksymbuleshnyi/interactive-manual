"""Mock image-edit adapter: copies the input image and stamps a caption.

Stands in for a real image editor. It deliberately does not change scene
content - it just annotates - so the pipeline runs offline and any real drift
study (Stage 4) is comparing against a known no-op baseline.
"""

from __future__ import annotations

import time
from pathlib import Path

from PIL import Image, ImageDraw

from interactive_interfaces.utils.logging import (
    file_sha256,
    get_call_context,
    get_current_run,
)


class MockImageEditor:
    """Deterministic stand-in for an ImageEditClient."""

    name = "mock_image_edit"

    def generate_next_image(
        self,
        *,
        input_image: Path,
        instruction: str,
        output_path: Path,
        **kwargs: object,
    ) -> Path:
        t0 = time.perf_counter()
        input_image = Path(input_image)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        img = Image.open(input_image).convert("RGB")
        draw = ImageDraw.Draw(img)
        band_h = 64
        draw.rectangle(
            [0, img.height - band_h, img.width, img.height], fill=(18, 18, 18)
        )
        draw.text(
            (12, img.height - band_h + 8),
            self._fit(instruction, 86),
            fill=(255, 255, 255),
        )
        draw.text((12, 10), "synthetic / research", fill=(255, 210, 120))
        img.save(output_path)

        latency_ms = int((time.perf_counter() - t0) * 1000)
        self._log(input_image, output_path, instruction, latency_ms)
        return output_path

    @staticmethod
    def _fit(text: str, width: int) -> str:
        text = " ".join(text.split())
        return text if len(text) <= width else text[: width - 1] + "…"

    def _log(
        self,
        input_image: Path,
        output_path: Path,
        instruction: str,
        latency_ms: int,
    ) -> None:
        run = get_current_run()
        if run is None:
            return
        ctx = get_call_context()
        run.log_call(
            stage=ctx.stage if ctx else "image_edit",
            task_id=ctx.task_id if ctx else None,
            step_index=ctx.step_index if ctx else None,
            adapter=self.name,
            input_hash=file_sha256(input_image),
            output_hash=file_sha256(output_path),
            latency_ms=latency_ms,
            raw_kind="image_edit",
            raw={
                "model": self.name,
                "instruction": instruction,
                "input_image": str(input_image),
                "output_image": str(output_path),
            },
        )
