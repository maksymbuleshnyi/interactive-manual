"""Mock LLM adapter: returns templated, task-flavored text. No network calls.

It inspects the rendered prompt to decide which pipeline stage it is serving
and returns a plausibly-shaped response (procedure JSON, atomic-steps JSON, or
an image-edit paragraph), so the whole pipeline runs end-to-end offline.
"""

from __future__ import annotations

import json
import re
import time

from interactive_interfaces.utils.logging import (
    get_call_context,
    get_current_run,
    hash_text,
)


class MockLLM:
    """Deterministic stand-in for a real LLMClient."""

    name = "mock_llm"

    def generate(
        self,
        *,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.2,
        **kwargs: object,
    ) -> str:
        t0 = time.perf_counter()
        response = self._respond(prompt)
        latency_ms = int((time.perf_counter() - t0) * 1000)
        self._log(prompt, response, system, temperature, latency_ms)
        return response

    # -- response templates ------------------------------------------------

    def _respond(self, prompt: str) -> str:
        if "procedural guides" in prompt:
            return self._procedure(prompt)
        if "ATOMIC VISUAL STEPS" in prompt:
            return self._atomic_steps(prompt)
        if "image editing prompt" in prompt:
            return self._edit_prompt(prompt)
        return "MOCK_LLM: " + prompt.strip()[:120]

    def _procedure(self, prompt: str) -> str:
        goal = self._field(prompt, "Task") or "the task"
        domain = self._field(prompt, "Domain") or "other"
        procedure = [
            f"Prepare the workspace and gather what is needed for: {goal}",
            "Position the main object so it is clearly visible in frame",
            "Apply the first observable change toward the goal",
            "Apply the second observable change toward the goal",
            "Confirm the final visible state matches the goal",
        ]
        return json.dumps(
            {
                "safety": [f"Mock safety note: review risks before starting ({domain})."],
                "tools": ["mock tool A", "mock tool B"],
                "procedure": procedure,
            },
            indent=2,
        )

    def _atomic_steps(self, prompt: str) -> str:
        procedure = self._extract_json_array(prompt, "Procedure:")
        steps = []
        for i, item in enumerate(procedure):
            steps.append(
                {
                    "expected_user_action": f"complete action {i + 1}",
                    "natural_language_instruction": str(item),
                    "current_state_description": f"The scene before step {i + 1}.",
                    "next_state_description": f"The scene after step {i + 1}: {item}",
                    "safety_notes": [],
                }
            )
        return json.dumps(steps, indent=2)

    def _edit_prompt(self, prompt: str) -> str:
        nxt = self._field(prompt, "Next state description") or "the next state"
        instr = self._field(prompt, "User instruction") or "the instruction"
        return (
            f"Modify the current image with the minimum change needed to show: "
            f"{nxt} Keep the background, lighting, and camera angle unchanged; "
            f"apply only the change implied by '{instr}'."
        )

    # -- prompt parsing helpers -------------------------------------------

    @staticmethod
    def _field(prompt: str, label: str) -> str:
        match = re.search(rf"^{re.escape(label)}:\s*(.+)$", prompt, re.MULTILINE)
        return match.group(1).strip() if match else ""

    @staticmethod
    def _extract_json_array(prompt: str, marker: str) -> list:
        """Pull the first ``[...]`` JSON array appearing after ``marker``.

        Mock-only: bracket matching ignores strings, which is safe because the
        mock never emits brackets inside procedure text.
        """
        start = prompt.find(marker)
        if start == -1:
            return []
        open_idx = prompt.find("[", start)
        if open_idx == -1:
            return []
        depth = 0
        for i in range(open_idx, len(prompt)):
            if prompt[i] == "[":
                depth += 1
            elif prompt[i] == "]":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(prompt[open_idx : i + 1])
                    except json.JSONDecodeError:
                        return []
        return []

    # -- logging -----------------------------------------------------------

    def _log(
        self,
        prompt: str,
        response: str,
        system: str | None,
        temperature: float,
        latency_ms: int,
    ) -> None:
        run = get_current_run()
        if run is None:
            return
        ctx = get_call_context()
        run.log_call(
            stage=ctx.stage if ctx else "llm",
            task_id=ctx.task_id if ctx else None,
            step_index=ctx.step_index if ctx else None,
            adapter=self.name,
            input_hash=hash_text(prompt),
            output_hash=hash_text(response),
            latency_ms=latency_ms,
            raw_kind="llm",
            raw={
                "model": self.name,
                "system": system,
                "temperature": temperature,
                "prompt": prompt,
                "response": response,
            },
        )
