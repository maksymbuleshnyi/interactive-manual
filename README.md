# interactive-interfaces

Research prototype: **From Text Answers to Interactive Interfaces.**

Long-term goal: a data-generation pipeline for chatbot-generated *interactive
visual guidance*. A user provides a task (and optionally a photo of their
current situation); the system generates step-by-step instructions and uses
an image generation / editing model to synthesize what the next step should
look like.

This repo is the MVP for that data-generation pipeline. It is exploratory
research code, not a product.

## Start here

1. Read [`PLAN.md`](PLAN.md). It is the single source of truth for the
   research and engineering plan, the data schema, the prompt templates,
   the experiments, and the staged implementation checklist.
2. Read `docs/DECISIONS.md` and `docs/EXPERIMENTS.md` (will exist once
   Stage 0 lands) to catch up on decisions and results so far.
3. When you make a design choice or run an experiment, **log it**. Every
   model call must produce a raw log under `logs/`. See the "Logging and
   experiment tracking" section of `PLAN.md`.

## Status

Planning. No code in the repo yet. Stage 0 of `PLAN.md` is the next step.
