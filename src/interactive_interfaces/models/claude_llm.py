"""Real LLM adapter: Claude via the official Anthropic SDK.

Stage 2 of PLAN.md. Implements the ``LLMClient`` protocol so it drops into the
pipeline behind ``registry.get_llm("claude")`` with no other code changes.

Defaults to Claude Opus 4.7 with adaptive thinking - the decomposition stage
(procedure -> atomic visual steps under 7 rules) is genuine multi-step
reasoning. Streams the response and uses ``get_final_message()`` so large or
slow generations don't hit request timeouts.

Requires ``ANTHROPIC_API_KEY`` in the environment and the optional dependency:
``pip install -e ".[claude]"``.
"""

from __future__ import annotations

import os
import time

import anthropic

from interactive_interfaces.utils.logging import (
    get_call_context,
    get_current_run,
    hash_text,
)

_DEFAULT_MODEL = "claude-opus-4-7"

# Stable across every call, so it can carry a cache breakpoint. (At the current
# prompt sizes this is well below Opus 4.7's ~4096-token minimum cacheable
# prefix, so it is a documented no-op until prompts grow - it costs nothing and
# becomes a real cache hit if the system prompt is ever expanded.)
_SYSTEM = (
    "You are a precise assistant in a research pipeline that turns tasks into "
    "step-by-step visual guidance. Follow the output format in each prompt "
    "exactly. When a prompt asks for JSON, return only valid JSON: no prose, "
    "no markdown code fences, no preamble, no trailing commentary."
)


class ClaudeLLM:
    """``LLMClient`` backed by Anthropic's Claude."""

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 16000,
        effort: str | None = None,
        thinking: bool = True,
    ) -> None:
        self.model = model or os.environ.get("II_LLM_MODEL", _DEFAULT_MODEL)
        self.effort = effort or os.environ.get("II_LLM_EFFORT", "high")
        self.max_tokens = max_tokens
        self.thinking = thinking
        self.name = f"claude:{self.model}"
        # api_key=None lets the SDK read ANTHROPIC_API_KEY from the environment.
        self._client = anthropic.Anthropic(api_key=api_key)

    def generate(
        self,
        *,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.2,
        **kwargs: object,
    ) -> str:
        # `temperature` is part of the LLMClient protocol, but Opus 4.7 removed
        # sampling parameters (sending one is a 400). It is intentionally not
        # forwarded; behaviour is steered by the prompt and `effort` instead.
        system_text = system or _SYSTEM
        t0 = time.perf_counter()
        try:
            with self._client.messages.stream(
                model=self.model,
                max_tokens=self.max_tokens,
                system=[
                    {
                        "type": "text",
                        "text": system_text,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": prompt}],
                output_config={"effort": self.effort},
                thinking={"type": "adaptive"} if self.thinking else {"type": "disabled"},
            ) as stream:
                message = stream.get_final_message()
        except anthropic.APIError as exc:
            latency_ms = int((time.perf_counter() - t0) * 1000)
            self._log(prompt, "", system_text, None, latency_ms, ok=False, error=repr(exc))
            raise

        latency_ms = int((time.perf_counter() - t0) * 1000)
        response = "".join(b.text for b in message.content if b.type == "text")
        self._log(prompt, response, system_text, message, latency_ms, ok=True, error=None)
        return response

    def _log(
        self,
        prompt: str,
        response: str,
        system: str,
        message: object | None,
        latency_ms: int,
        *,
        ok: bool,
        error: str | None,
    ) -> None:
        run = get_current_run()
        if run is None:
            return
        ctx = get_call_context()
        usage = None
        msg_usage = getattr(message, "usage", None)
        if msg_usage is not None:
            usage = {
                "input_tokens": getattr(msg_usage, "input_tokens", None),
                "output_tokens": getattr(msg_usage, "output_tokens", None),
                "cache_read_input_tokens": getattr(
                    msg_usage, "cache_read_input_tokens", None
                ),
                "cache_creation_input_tokens": getattr(
                    msg_usage, "cache_creation_input_tokens", None
                ),
            }
        run.log_call(
            stage=ctx.stage if ctx else "llm",
            task_id=ctx.task_id if ctx else None,
            step_index=ctx.step_index if ctx else None,
            adapter=self.name,
            input_hash=hash_text(prompt),
            output_hash=hash_text(response),
            latency_ms=latency_ms,
            ok=ok,
            error=error,
            raw_kind="llm",
            raw={
                "model": self.model,
                "effort": self.effort,
                "thinking": self.thinking,
                "system": system,
                "prompt": prompt,
                "response": response,
                "usage": usage,
            },
        )
