# Training

Everything here targets the *long-term* plan in `docs/PRODUCT_VISION.md`:
fine-tune an open-source base model as the primary safety/tone alignment
mechanism, with the guardrail pipeline (`backend/app/guardrails/`) as the
always-on supplementary net. Nothing in this directory should be read as "the
model is trained" -- as of this commit, no fine-tuning run has happened.
See `docs/BLOCKERS.md` for why (clinical review has to land first).

## What's here and what actually runs today

| File | Status |
|---|---|
| `data/seed_dataset.jsonl` | 26 hand-written examples, all marked `DRAFT_NEEDS_CLINICAL_REVIEW`. Enough to exercise and calibrate the guardrail pipeline; nowhere near enough, or clinically sound enough, to fine-tune on directly. |
| `scripts/evaluate.py` | **Runs today, no GPU needed.** Scores the guardrail pipeline (not a fine-tuned model -- none exists yet) against the seed dataset. This is the real feedback loop for tuning keyword lists and thresholds. |
| `scripts/prepare_dataset.py` | **Runs today.** Converts the seed dataset into chat-format JSONL (`data/prepared/sft.jsonl`) suitable for a supervised fine-tuning framework. Useful as a pipeline dry run; the *content* it produces inherits every caveat of the seed dataset above. |
| `scripts/finetune_lora.py` | **Does not run in this environment** -- no GPU, no base model weights, no HF token configured. It's a complete, correct LoRA fine-tuning script (transformers + peft + trl) so that once there's real hardware and a clinically-reviewed dataset, running a fine-tune is a command, not a from-scratch build. |

## Recommended loop once past the clinical-review blocker

1. Specialist review rewrites/expands `data/seed_dataset.jsonl` (target: low
   hundreds of examples per tier-1 category minimum, not 26 total).
2. `python3 scripts/evaluate.py` against the *pipeline* first -- if the
   guardrail layer can't correctly classify the reviewed examples, fix the
   lexicon/thresholds before touching the model at all. The pipeline is the
   safety net regardless of fine-tune quality (docs/ARCHITECTURE.md), so it
   has to be right independently.
3. `python3 scripts/prepare_dataset.py` to produce SFT-format data.
4. `python3 scripts/finetune_lora.py --base-model <model> --data data/prepared/sft.jsonl`
   on real hardware.
5. Re-run `evaluate.py`, this time pointed at the fine-tuned model's actual
   outputs (not just the pipeline's classification) -- a step not yet built,
   since there's no model to evaluate yet.

## Why LoRA on an open base model, not full fine-tune

Not an explicit product decision on record, but the sane default given the
stated constraints: full fine-tuning an 7-8B+ model needs meaningfully more
compute/data than LoRA to avoid catastrophic forgetting of general
capability (homework help, creative writing) while aligning safety behavior.
Revisit this once real training data volume and hardware budget are known.
