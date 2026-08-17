# Experiments log

Append-only. One entry per experiment. See the "Logging and experiment
tracking" and "First three experiments" sections of `PLAN.md`.

Each entry records: date, experiment id, what changed, dataset, command run,
run_id, headline numbers, qualitative notes, link to artifacts.

---

### 2026-05-17 - Stage 0: no experiments yet

The skeleton is in place (schema, run logging, seed `demo.json`). No model
calls have been made and no demonstrations generated.

The first entry is **E0** below.

---

### 2026-05-17 - E0: mock pipeline end-to-end smoke run

- **What changed:** First full pipeline run. All adapters are mocks
  (`mock_llm`, `mock_image_edit`, `mock_critic`); no network calls.
- **Dataset:** `data/tasks/wiper_blade.json` (physical_repair),
  `data/tasks/outlook_dark_mode.json` (software_ui). Neither has a seed photo
  yet, so each demo's `initial.png` is a synthesized placeholder.
- **Commands run (per task):** `create_demo.py` -> `decompose_steps.py` ->
  `chain_steps.py --max-steps 5` -> `critique_demo.py`.
- **run_ids:** wiper - `20260517T001809Z-{eb49c728,48364bca,6ea3e235,0d9681fa}-wiper`;
  outlook - `20260517T0018{33,34}Z-{7c432abf,c33b2090,04319629,0f165386}-outlook`.
- **Headline numbers:** both demos produced 5 atomic steps; all 5 steps per
  demo have a generated `step_N.png`; mock critic `overall` = 3/5 fixed, so
  `overall_quality_score` = 3.0 for both. `atomic_lint` flagged 1/5 steps in
  each demo (rule R5 - the procedure's first item joins clauses with "and").
- **Qualitative notes:** The pipeline wiring is sound end-to-end: stage
  hand-off, image chaining (step k's input = step k-1's output), per-call
  logging, and raw payload capture all work. The lint flag is expected and
  informative - the mock LLM echoes procedure phrasing verbatim into
  instructions, so non-atomic wording survives; a real decomposition LLM
  (Stage 2) plus the violation->retry loop should drive this down. No quality
  signal is real yet: the mock editor only annotates images and the mock
  critic returns a constant. E0 validates plumbing, not model quality.
- **Artifacts:** `outputs/wiper-001/`, `outputs/outlook-darkmode-001/`,
  and the `logs/runs/<run_id>/` directories above.
