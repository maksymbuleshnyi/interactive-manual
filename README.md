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
2. Read `docs/DECISIONS.md` and `docs/EXPERIMENTS.md` to catch up on the
   decisions made and the experiments run so far.
3. When you make a design choice or run an experiment, **log it**. Every
   model call must produce a raw log under `logs/`. See the "Logging and
   experiment tracking" section of `PLAN.md`.

## Status

Stages 0 and 1 are complete: the schema, run logging, and the mock pipeline
run end-to-end (see `docs/EXPERIMENTS.md`, entry E0). Stage 2 — the real Claude
LLM adapter — is implemented and tested; its acceptance check, running
Experiment 1 live, is still pending. Stages 3-5 have not started.

## Prototypes in this repo

### F1 assembly manual with a learned renderer ([`f1_assembly/`](f1_assembly/))

A working end to end instance of the idea above, overfit to one object. An
agent builds a 3D assembly manual from crude primitives, boxes and cylinders
with no CAD, and a fine-tuned diffusion renderer turns that blockout into a
photographic frame in real time at about 8 fps, grounded on one reference
photo of the parts on a bench.

![blockout on the left, neural render on the right](f1_assembly/docs/demo.gif)

See [`f1_assembly/README.md`](f1_assembly/README.md) for the design, the
conditioning layout (depth, flat per-part colour, memory picture) and how to
capture data, train and serve.
