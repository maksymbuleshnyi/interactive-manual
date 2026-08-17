"""IO helpers: load task specs, load/save demonstrations, placeholder images."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

from interactive_interfaces.schemas.demonstration import Demonstration


def load_task(path: Path | str) -> dict:
    """Load a task spec JSON from ``data/tasks/``."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_json_response(text: str) -> object:
    """Parse JSON from an LLM response, tolerating a ```` ```json ```` fence.

    The prompts and system prompt forbid markdown fences, but real models
    occasionally add them anyway. Strip one wrapping fence if present, then
    parse strictly - a genuinely malformed response still raises.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        newline = stripped.find("\n")
        if newline != -1:
            stripped = stripped[newline + 1 :]
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[:-3]
        stripped = stripped.strip()
    return json.loads(stripped)


def load_demo(path: Path | str) -> Demonstration:
    """Load and validate a ``demo.json``."""
    return Demonstration.model_validate_json(Path(path).read_text(encoding="utf-8"))


def save_demo(demo: Demonstration, path: Path | str) -> Path:
    """Write a demonstration to ``demo.json`` (pretty-printed). Returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(demo.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def make_placeholder_image(
    path: Path | str,
    label: str,
    sub: str = "",
    bg: tuple[int, int, int] = (54, 78, 110),
) -> Path:
    """Generate a synthetic placeholder image. Used when no seed photo exists.

    Clearly marked so it can never be mistaken for a real photo (see the
    legal/safety section of PLAN.md).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (640, 480), bg)
    d = ImageDraw.Draw(img)
    d.rectangle([8, 8, 631, 471], outline=(255, 255, 255), width=3)
    d.text((24, 210), label, fill=(255, 255, 255))
    if sub:
        d.text((24, 240), sub[:90], fill=(220, 220, 220))
    d.text(
        (24, 446),
        "SYNTHETIC PLACEHOLDER - research / not a real photo",
        fill=(255, 210, 120),
    )
    img.save(path)
    return path
