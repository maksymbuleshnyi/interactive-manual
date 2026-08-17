"""Load prompt templates from ``prompts/`` and render them with variables.

Templates use Jinja-style ``{{var}}`` interpolation (see PLAN.md section D).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from jinja2 import Template

# repo_root/src/interactive_interfaces/utils/prompts.py -> repo_root/prompts
_PROMPTS_DIR = Path(__file__).resolve().parents[3] / "prompts"


@lru_cache(maxsize=None)
def load_prompt(name: str) -> str:
    """Return the raw text of a prompt template. ``.md`` suffix is optional."""
    path = _PROMPTS_DIR / name
    if not path.suffix:
        path = path.with_suffix(".md")
    return path.read_text(encoding="utf-8")


def render(name: str, /, **variables: object) -> str:
    """Load template ``name`` and render it with the given variables."""
    return Template(load_prompt(name)).render(**variables)
