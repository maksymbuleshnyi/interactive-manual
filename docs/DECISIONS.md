# Decisions log

Append-only. One entry per decision. Newest at the bottom. See the "Logging
and experiment tracking" section of `PLAN.md`.

Format: **date** - decision - options considered - rationale - who.

---

### 2026-05-17 - D0: Adopt the PLAN.md section B repo layout

- **Decision:** Stage 0 builds the skeleton from `PLAN.md` section B. Only the
  directories needed by Stage 0 (`schemas/`, `utils/`) are populated;
  `models/`, `pipeline/`, `eval/`, and `notebooks/` are created in their
  respective later stages, not pre-stubbed.
- **Options considered:** (a) scaffold every directory now with empty stubs;
  (b) create only what Stage 0 needs.
- **Rationale:** Empty stubs for later stages add noise and invite drift from
  the plan. Stages are gated; build each stage's files when that stage runs.
- **Who:** Maksym Buleshnyi.

### 2026-05-17 - D1: Schema implemented exactly as the v0.1 sketch

- **Decision:** `schemas/demonstration.py` matches the `PLAN.md` section C
  sketch field-for-field. No extra fields (e.g. a `Step.step_type` for the
  `reframe` case from the decomposition prompt) were added.
- **Options considered:** (a) add a `step_type` discriminator now; (b) keep
  v0.1 faithful and revisit when `decompose.py` lands.
- **Rationale:** The schema version is pinned to "0.1"; reframe-step handling
  is a Stage 1 concern and should be designed alongside `decompose.py`.
- **Who:** Maksym Buleshnyi.

### 2026-05-17 - D2: wiper-001 images are synthetic placeholders, not a real photo

- **Decision:** `data/examples/wiper-001/initial.jpg` and `step_0.jpg` are
  generated programmatically (solid background + label text), not photographs.
- **Options considered:** (a) block Stage 0 until a real wiper photo is
  captured; (b) ship clearly-marked placeholders now.
- **Rationale:** The Stage 0 acceptance check only requires a schema-valid
  `demo.json`. A real seed photo is an Experiment-1 input, not a skeleton
  blocker. The placeholder status is recorded in the demo's `provenance.notes`.
- **Who:** Maksym Buleshnyi.

### 2026-05-17 - D3: pytest is a dev-only optional dependency

- **Decision:** `pytest` lives in `[project.optional-dependencies].dev`, not in
  runtime `dependencies`. Develop with `pip install -e ".[dev]"`.
- **Options considered:** (a) list pytest among runtime deps as the plan's
  Stage 0 bullet reads literally; (b) scope it to a `dev` extra.
- **Rationale:** Keeps the runtime install lean; test tooling is not needed to
  run the pipeline. The plan's intent ("deps include pytest") is preserved.
- **Who:** Maksym Buleshnyi.

### 2026-05-17 - D4: Prompt templates materialized verbatim in Stage 0

- **Decision:** The four `prompts/*.md` templates are created now, copied
  verbatim from `PLAN.md` section D.
- **Options considered:** (a) defer prompt files to Stage 1 when the pipeline
  uses them; (b) materialize them with the Stage 0 skeleton.
- **Rationale:** They are static text fully specified by the plan, with no
  logic and no gating risk. Having them present makes the skeleton complete.
- **Who:** Maksym Buleshnyi.

### 2026-05-17 - D5: Adapters log calls; pipeline marks the stage via a contextvar

- **Decision:** Adapters call `Run.log_call()` themselves (per the plan's
  logging implementation note), but the pipeline stage / task_id / step_index
  they cannot know is supplied through a `call_context()` contextvar that the
  pipeline sets. This yields exactly one `run.jsonl` event per model call with
  a correct `stage` label.
- **Options considered:** (a) log from the pipeline (clean stage labels, but
  the plan explicitly wants adapter-side logging); (b) adapters log with a
  generic stage; (c) adapters log, pipeline supplies stage via contextvar.
- **Rationale:** (c) honors the plan's note and keeps the pipeline free of
  logging boilerplate (just a `with call_context(...)`), while still labelling
  events correctly for evaluation.
- **Who:** Maksym Buleshnyi.

### 2026-05-17 - D6: atomic_lint runs in Stage 1; the violation->retry loop waits for Stage 2

- **Decision:** `atomic_lint` runs over every decomposed step and records
  violations as `Step.failure_modes`. The "feed violations back to the LLM for
  one retry" loop is deferred to Stage 2.
- **Options considered:** (a) implement the retry now; (b) lint-only now.
- **Rationale:** A retry against the deterministic mock LLM produces identical
  output - a no-op. The retry becomes meaningful only with a real LLM, so it
  is built alongside the Stage 2 adapter.
- **Who:** Maksym Buleshnyi.

### 2026-05-17 - D7: Pipeline-generated images are PNG

- **Decision:** `create_demo` writes `initial.png` and the image editor writes
  `step_N.png`. `PLAN.md` uses `step_0.jpg` only as an illustrative filename.
- **Options considered:** (a) JPEG per the plan's example; (b) PNG.
- **Rationale:** Mock images are synthetic and text-heavy; PNG avoids JPEG
  ringing on the caption/overlay text. Revisit if a real editor needs JPEG.
- **Who:** Maksym Buleshnyi.

### 2026-05-17 - D8: Stage 2 LLM vendor is Claude (Anthropic)

- **Decision:** The real `LLMClient` is Claude, via the official `anthropic`
  SDK; default model `claude-opus-4-7`.
- **Options considered:** (a) Claude (Anthropic); (b) OpenAI (GPT).
- **Rationale:** Chosen by the user. Native to the working environment; built
  against current Anthropic SDK guidance (correct model IDs, adaptive thinking,
  prompt caching). The adapter still sits behind the `LLMClient` protocol, so
  the choice is reversible.
- **Who:** Maksym Buleshnyi (via an explicit choice prompt).

### 2026-05-17 - D9: `anthropic` is an optional extra, lazy-imported

- **Decision:** `anthropic` lives in the `[claude]` optional-dependency extra,
  not core `dependencies`. `registry.py` imports `ClaudeLLM` lazily, only when
  the `claude` adapter is requested.
- **Options considered:** (a) core dependency; (b) optional extra + lazy import.
- **Rationale:** Keeps the core package vendor-agnostic (the plan's vendor
  lock-in mitigation); mock-only users need not install a vendor SDK.
- **Who:** Maksym Buleshnyi.

### 2026-05-17 - D10: Claude adapter request configuration

- **Decision:** Opus 4.7 with adaptive thinking + `effort: "high"`, streamed
  (`get_final_message()`). The protocol's `temperature` argument is accepted
  but not forwarded; the system prompt carries a `cache_control` breakpoint.
- **Options considered:** thinking on/off; effort level; whether to forward
  `temperature`.
- **Rationale:** Decomposition is multi-step reasoning, which suits adaptive
  thinking; streaming avoids request timeouts. Opus 4.7 removed sampling
  parameters - forwarding `temperature` would return a 400.
- **Who:** Maksym Buleshnyi.

### 2026-05-17 - D11: Tolerant JSON parsing; atomic-lint retry deferred

- **Decision:** Pipeline parses LLM JSON via `parse_json_response` (strips one
  wrapping markdown fence). The `atomic_lint` violation->retry loop is still
  not implemented.
- **Options considered:** (a) implement the retry now; (b) measure first.
- **Rationale:** Measure Experiment 1's real violation rate before adding the
  retry; add it only if the rate exceeds the < 30% acceptance bar. Matches the
  plan's measure-then-decide discipline. Supersedes the deferral note in D6.
- **Who:** Maksym Buleshnyi.

### 2026-05-17 - D12: UI / product pivot deferred - finish the gated study first

- **Decision:** Declined to build a UI or pivot to a product now. Stages 2-5
  will be completed in gated order; a UI is reconsidered only afterward.
- **Options considered:** (a) finish the research study first; (b) build a UI
  over the current pipeline; (c) pivot fully to a product.
- **Rationale:** A live UI is an explicit MVP non-goal (`PLAN.md` section A).
  Building one over an unvalidated pipeline (Experiment 1 unrun, image
  faithfulness unmeasured) would ship polished but unverified guidance. Validate
  the pipeline first; a UI is a sensible follow-on project if the findings are
  positive.
- **Who:** Maksym Buleshnyi (via an explicit choice prompt).
