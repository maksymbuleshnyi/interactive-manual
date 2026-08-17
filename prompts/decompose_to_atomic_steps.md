You are converting a procedure into ATOMIC VISUAL STEPS suitable for image
generation. Each atomic step MUST satisfy ALL rules:

R1. Exactly one visible state change.
R2. The change is observable in a static image of the same viewpoint.
R3. The subject of the change is unambiguously identifiable.
R4. No mental or invisible state changes (e.g., "decide", "remember").
R5. No multi-clause actions joined by "and" or "then".
R6. If a viewpoint change is needed, emit a separate {"type": "reframe"} step.
R7. If the step has physical risk, populate `safety_notes`.

Procedure:
{{procedure_json}}

Return JSON list of:
{
  "expected_user_action": "...",            // verb phrase
  "natural_language_instruction": "...",    // shown to the user
  "current_state_description": "...",
  "next_state_description": "...",
  "safety_notes": ["..."]
}
