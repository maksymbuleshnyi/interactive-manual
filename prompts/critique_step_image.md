You are evaluating a generated next-step image.

Instruction:           {{natural_language_instruction}}
Expected next state:   {{next_state_description}}

You are shown the BEFORE image and the AFTER image.
Score each criterion 1-5 (5 best) and give a one-sentence rationale per score.

Criteria:
- instruction_clarity:           Is the instruction visualizable and unambiguous?
- visual_correctness:            Does AFTER actually depict the next state?
- image_faithfulness:            Are unrelated parts of BEFORE preserved in AFTER?
- illustrates_next_step:         Would a user understand what to do from AFTER?
- irrelevant_detail_preservation: Background, lighting, identity preserved?
- safety:                        Any unsafe or misleading depiction?
- overall:                       Single holistic score.

Return strict JSON matching the CritiqueResult schema.
