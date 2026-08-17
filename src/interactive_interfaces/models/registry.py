"""Adapter registry: resolve an adapter name to a client instance.

Mock-only in Stage 1. Real vendors are added here, one line each, in Stages
2-3; record the choice in docs/DECISIONS.md.
"""

from __future__ import annotations

from collections.abc import Callable

from interactive_interfaces.models.base import CriticClient, ImageEditClient, LLMClient
from interactive_interfaces.models.mock_critic import MockCritic
from interactive_interfaces.models.mock_image_edit import MockImageEditor
from interactive_interfaces.models.mock_llm import MockLLM


def _make_claude() -> LLMClient:
    """Construct the Claude adapter. Imported lazily so the core package does
    not depend on the ``anthropic`` SDK unless the adapter is actually used."""
    from interactive_interfaces.models.claude_llm import ClaudeLLM

    return ClaudeLLM()


_LLMS: dict[str, Callable[[], LLMClient]] = {"mock": MockLLM, "claude": _make_claude}
_EDITORS: dict[str, Callable[[], ImageEditClient]] = {"mock": MockImageEditor}
_CRITICS: dict[str, Callable[[], CriticClient]] = {"mock": MockCritic}


def _resolve(table: dict, kind: str, name: str):
    try:
        return table[name]()
    except KeyError:
        raise ValueError(
            f"unknown {kind} adapter {name!r}; available: {sorted(table)}"
        ) from None


def get_llm(name: str = "mock") -> LLMClient:
    return _resolve(_LLMS, "llm", name)


def get_image_editor(name: str = "mock") -> ImageEditClient:
    return _resolve(_EDITORS, "image editor", name)


def get_critic(name: str = "mock") -> CriticClient:
    return _resolve(_CRITICS, "critic", name)
