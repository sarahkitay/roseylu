# Evaluation Criteria

`training/scripts/evaluate.py` currently calibrates the guardrail pipeline
against the seed dataset (see `BLOCKERS.md`'s "suggested order of
operations" — this is not itself a launch blocker). This doc specifies the
fuller set of things worth measuring, most of which aren't implemented yet,
so "we evaluated it" means something specific rather than "we ran the
script that exists today."

## Why this needs to exist as its own doc

`BLOCKERS.md` already documents a real case on this project where training
loss went down and actual output quality went down with it (the
over-trained BPE run). That's not a one-off mistake to note and move past —
it's a standing argument that no single automatic number is a safe proxy
for "this is better," and everything below is written with that in mind:
every dimension needs its own check, not one score assumed to cover
everything.

## Dimensions

**Guardrail recall and precision** — not yet really measurable.
`SAFETY_MODEL.md` documents that the semantic classifier and LLM-judge
layers are both heuristic stand-ins, so there's no real classifier recall
to report yet beyond the keyword layer's own. Once real models replace
those stand-ins, this needs a held-out labeled set the classifiers never
trained on, split per `RiskCategory` — a single blended accuracy number
would hide a category-specific failure (e.g. strong on `SELF_HARM`, weak on
`GROOMING`) behind a good-looking average.

**Compassion / tone** — see `VALUES_MODEL.md`. Rubric-scored, initially by
whoever does clinical review, later at scale by the teacher-review process
once its rubric is extended. Needs measuring separately for REDIRECT
responses (currently templates — this is really template review) and
ALLOW-path generative responses (currently ungoverned — this is where it
matters most once the model is competent enough to be trusted with more).

**Curriculum accuracy** — `curated_qa.py`'s curriculum entries need a
subject-matter reviewer per `BLOCKERS.md` item 8, checking factual accuracy
and age-appropriate framing separately — a response can be factually
correct and still framed badly for a 7-year-old, or vice versa.

**Generation competence** — `BLOCKERS.md`'s capability-gap section already
describes this qualitatively (math paraphrase retrieval improved, non-math
topics still drift, seed variance affects specific strengths). Worth a
standing benchmark set — a fixed list of prompts run after every real
training run (not every online burst) with output diffed against the
previous run, so "did this run help" has an answer that isn't "it felt
different."

**Online-burst regression check** — not yet implemented. Given
`online_trainer.py` currently has no evaluator hook and reports only
training loss, the minimum viable version is running a small fixed prompt
set through the model after every N bursts and flagging if outputs degrade
against the standing benchmark set above — not a full evaluate.py pass
every burst, that's too slow, but something cheaper than no signal at all.

## What this doc is not

- Not an implementation — none of the unmeasured dimensions above have code
  yet. This is the spec `evaluate.py` should grow into, not a description
  of what it currently does.
- Not a replacement for clinical/subject-matter review — automated scoring
  is downstream of a human standard, not a substitute for setting one.
