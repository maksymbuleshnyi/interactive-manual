"""Minimal iFixit ingestion pilot.

Pulls a small number of published guides from the iFixit public REST API and
converts each into a `Demonstration` (demo.json + downloaded step images) using
the project's schema. iFixit content is CC BY-NC-SA - recorded in provenance.

The point: try ~5-10 real guides and confirm the shape works before scaling.
Since iFixit gives real before/after photos, no image generation is needed -
`image_generation_prompt` stays empty for these rows.

    python scripts/ingest_ifixit.py --count 5
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import typer  # noqa: E402

from interactive_interfaces.schemas.demonstration import (  # noqa: E402
    Demonstration, Provenance, Step,
)
from interactive_interfaces.utils.io import load_demo, save_demo  # noqa: E402

_API = "https://www.ifixit.com/api/2.0"
_HEADERS = {
    "User-Agent": "interactive-interfaces-research/0.1",
    "X-App-Id": "interactive-interfaces-research",
}
_LICENSE = "CC BY-NC-SA"
_LICENSE_URL = "https://creativecommons.org/licenses/by-nc-sa/3.0/"


def _get(url: str) -> object:
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def _list_guides(limit: int) -> list[dict]:
    return _get(f"{_API}/guides?limit={limit}&offset=0")  # type: ignore[return-value]


def _fetch_guide(guide_id: int) -> dict:
    return _get(f"{_API}/guides/{guide_id}")  # type: ignore[return-value]


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=60) as resp:
        dest.write_bytes(resp.read())
    return dest


def _step_text(step: dict) -> str:
    return "\n".join(
        ln.get("text_raw", "").strip()
        for ln in step.get("lines", [])
        if ln.get("text_raw", "").strip()
    )


def _step_image_url(step: dict) -> str | None:
    images = step.get("images") or []
    img = images[0] if images else None
    if img is None:
        media_data = (step.get("media") or {}).get("data") or []
        img = media_data[0] if media_data else None
    if not img:
        return None
    return (
        img.get("standard")
        or img.get("large")
        or img.get("original")
        or img.get("medium")
    )


def _convert(guide: dict, out_root: Path) -> Path | None:
    raw_steps = guide.get("steps") or []
    usable = []
    for s in raw_steps:
        url = _step_image_url(s)
        txt = _step_text(s)
        if url and txt:
            usable.append((s, url, txt))
    if len(usable) < 2:
        return None  # too thin to be useful

    guide_id = guide["guideid"]
    out_dir = out_root / f"ifixit-{guide_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    guide_img = guide.get("image") or {}
    initial_url = guide_img.get("standard") or guide_img.get("large") or usable[0][1]
    initial_path = _download(initial_url, out_dir / "initial.jpg")

    step_imgs: list[Path] = []
    for i, (_, url, _) in enumerate(usable):
        step_imgs.append(_download(url, out_dir / f"step_{i}.jpg"))
        time.sleep(0.2)

    steps: list[Step] = []
    for i, (s, _, txt) in enumerate(usable):
        prev_img = initial_path if i == 0 else step_imgs[i - 1]
        steps.append(
            Step(
                step_index=i,
                current_image_path=prev_img,
                current_state_description="",
                natural_language_instruction=txt,
                expected_user_action=(s.get("title") or "").strip(),
                # No edit needed - we already have the real next-state photo.
                image_generation_prompt="",
                image_model=None,
                generated_next_image_path=step_imgs[i],
                next_state_description=(s.get("title") or None),
                safety_notes=[],
            )
        )

    demo = Demonstration(
        task_id=f"ifixit-{guide_id}",
        domain="physical_repair",
        user_goal=(guide.get("title") or f"iFixit guide {guide_id}").strip(),
        initial_image_path=initial_path,
        initial_state_description=guide.get("summary") or None,
        procedure_text="\n".join((s.get("title") or "").strip() for s, _, _ in usable),
        steps=steps,
        source_type="public_task_description",
        provenance=Provenance(
            author="iFixit contributors (via public API)",
            date=date.today().isoformat(),
            license=_LICENSE,
            notes=(
                f"Source: {guide.get('url', '')} (guide_id={guide_id}). "
                f"License: {_LICENSE} ({_LICENSE_URL}). "
                "Step images are real photos from the iFixit guide; no AI "
                "edits applied."
            ),
        ),
        tags=["physical_repair", "ifixit", "real-photos"],
    )
    return save_demo(demo, out_dir / "demo.json")


def main(
    count: int = typer.Option(5, help="How many guides to convert."),
    out: Path = typer.Option(Path("outputs/ifixit"), help="Output directory."),
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    print(f"listing up to {count * 3} guide summaries...")
    summaries = _list_guides(count * 3)
    converted = 0
    for summary in summaries:
        if converted >= count:
            break
        gid = summary.get("guideid")
        if not gid:
            continue
        try:
            guide = _fetch_guide(gid)
        except Exception as exc:
            print(f"  guide {gid}: fetch failed: {exc}")
            continue
        try:
            demo_path = _convert(guide, out)
        except Exception as exc:
            print(f"  guide {gid}: convert failed: {exc}")
            continue
        if demo_path is None:
            print(f"  guide {gid}: skipped (thin / no usable steps)")
            continue
        loaded = load_demo(demo_path)  # schema validation
        converted += 1
        print(
            f"  [{converted}/{count}] {loaded.task_id}: "
            f"{len(loaded.steps)} steps -> {demo_path}"
        )
        time.sleep(0.3)
    print(f"done: {converted} demonstrations written to {out}")


if __name__ == "__main__":
    typer.run(main)
