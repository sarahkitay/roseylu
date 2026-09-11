# Values Model: Where Compassion Actually Lives Right Now

This doc exists because "the model should be compassionate" is not the same
claim as "the model has been trained to be compassionate," and this project
has, so far, entirely relied on the former to produce something that reads
like the latter. This is a companion to `SAFETY_MODEL.md` (what the
pipeline blocks and redirects) and `BLOCKERS.md` (what's unresolved before
launch) — this one specifies where warmth, gentleness, and non-judgment
currently come from, and what it would take for that to be a trained
property of the model itself rather than an engineered one sitting in front
of it.

## Current state: compassion is 100% hand-authored, 0% trained

Everywhere a child currently receives something that reads as gentle,
validating, or non-judgmental, a human wrote it directly:

- `backend/app/response/redirect_engine.py` — every tier-1 REDIRECT
  response (validate/explain/redirect) is a fixed template written by one
  engineer, pending the clinical review `BLOCKERS.md` item 1 already flags.
- `backend/app/knowledge/curated_qa.py`'s `LIFE` category — grief, fear of
  the dark, "why does mommy yell," body image — same situation: written
  carefully, but by an engineer, not yet reviewed by a child psychologist.

The generative model itself — `model/`, trained from scratch, currently
serving open-ended conversation that doesn't hit a REDIRECT or a curated
match — has never received a single training signal oriented toward
compassion, tone, or gentleness. `train_from_scratch.py` optimizes
next-token prediction against a text corpus. `online_trainer.py`'s bursts
optimize the same objective against recent conversation plus replay.
Neither loss function has any term that distinguishes "a correct-sounding
continuation" from "a kind one." The model isn't unkind — it has no
orientation toward kindness in either direction, because nothing has ever
asked it to.

This isn't a criticism of the architecture — it's the correct reason
`SAFETY_MODEL.md` explicitly refuses to trust the model with anything that
matters and routes tier-1 content to fixed templates instead. But it means
every claim this product makes about being gentle and compassionate is
currently a claim about `redirect_engine.py` and `curated_qa.py`, full
stop — not about Rosey. Worth being precise about that distinction anywhere
this product gets described, including internally.

## Why "just train it to be compassionate" isn't a plan on its own

Compassion doesn't fall out of a model getting more fluent, any more than
it falls out of a person getting smarter — it has to be a specifically
targeted training signal, with a way to measure whether it's present, or it
doesn't happen. Three things are required, not optional, for this to become
a real trained property rather than a documentation claim:

1. **A labeled signal for what "compassionate" means, concretely, per
   response** — not a vibe, a rubric a reviewer can apply consistently.
   See `EVAL_CRITERIA.md`.
2. **A mechanism to optimize against that signal** — the model has to
   actually be pushed toward the labeled-good responses and away from the
   labeled-bad ones, not just exposed to more text in general.
3. **A way to verify it moved the needle without breaking anything else** —
   given `BLOCKERS.md` already documents a case on this exact project where
   loss went down and output quality went down with it, "the loss looks
   better" cannot be the verification step.

## The concrete path, using infrastructure that already exists

`training/scripts/teacher_review_loop.py` already has an external, stronger
model reviewing live conversations and writing corrected/confirmed examples
to `training/data/corpus/teacher_reviewed.jsonl`, which `online_trainer.py`
already reads and upweights. Right now that reviewer is presumably scoring
for correctness/quality. The lowest-friction real step toward trained
compassion is extending that same reviewer's rubric to explicitly score a
compassion dimension per response — validates the feeling without
editorializing, explains rather than just refuses, age-appropriate warmth,
no judgment — alongside whatever it already checks. That produces labeled
data for exactly this, from a pipeline that already exists, rather than a
new system.

Once there's enough of that labeled data, and once the base model is
competent enough that fine-tuning on top of it means something (see
`BLOCKERS.md`'s capability-gap item — a model still drifting into literary
pastiche on non-math topics isn't ready to have a values layer stacked on
top of it yet), a lightweight preference-tuning pass over that scored data
is the actual mechanism — not full RLHF at this scale, but the same shape
sized to an ~11M-parameter model: pairs of (response A, response B, which
one the compassion rubric preferred), pushing the model's own weights
toward the preferred one. That's the difference between "we hope it's
gentle" and "we optimized it to be gentle and can show the eval that says
so."

## What this doc is not

- Not a claim that compassion is currently trained into the model. It
  isn't.
- Not a substitute for clinical review of the existing hand-authored
  templates — those still need `BLOCKERS.md` item 1 regardless of anything
  in this doc.
- Not a proposal to replace human clinical judgment with an automated
  rubric score. The rubric in `EVAL_CRITERIA.md` is meant to give a
  clinical reviewer's standards something to propagate into training data
  at scale, not to invent those standards itself.
