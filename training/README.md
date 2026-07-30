# Training

The app's generation backend runs on a language model trained from scratch,
on this machine, with no third-party LLM API involved at any point in the
running app -- not Anthropic, not anyone else. See
`docs/PRODUCT_VISION.md`'s Technical Architecture section and
`docs/ARCHITECTURE.md`'s "design principle" section for why that's a hard
rule here, not a preference.

## What "from scratch" means concretely

- `model/architecture.py` -- a decoder-only transformer (embeddings, causal
  multi-head self-attention, MLP blocks, layernorm) written by hand in plain
  PyTorch. No `transformers` library, no pretrained checkpoint loaded in.
  Every weight starts randomly initialized.
- `model/tokenizer.py` -- a character-level tokenizer, also written from
  scratch (no `tiktoken`/`sentencepiece`/`tokenizers` dependency). The vocab
  is just the unique characters seen in the training corpus.
- `training/scripts/train_from_scratch.py` -- the training loop (AdamW,
  gradient clipping, train/val loss tracking, periodic sampling) that turns
  random weights into a model that's actually learned something from the
  corpus below.

## Where the training data comes from

Three sources, assembled by `training/scripts/build_corpus.py` into
`training/data/corpus/combined.txt`:

1. **Public-domain children's literature** (`training/data/corpus/public_domain/`)
   -- Alice's Adventures in Wonderland, Just So Stories, The Tale of Peter
   Rabbit, The Aesop for Children, pulled from Project Gutenberg. This is
   what teaches the model general English fluency, grammar, and narrative
   structure -- there's no way to get that from a few hundred hand-written
   examples alone.
2. **Hand-authored synthetic dialogue** (`training/data/synthetic_dialogues.py`)
   -- `Child: ... / Rosey: ...` pairs across homework help, science
   curiosity, creative writing, everyday feelings, and persona/identity
   questions. **This is where Claude's role in this project is**: these
   examples were written directly by Claude, offline, as a developer-facing
   drafting task -- not called from any running code, not a runtime
   dependency. This is "Anthropic used for internal training and learning,"
   made literal: knowledge transfer into training data, never a live API
   call the deployed app makes.
3. **Rendered redirect templates** (`backend/app/response/redirect_engine.py`,
   rendered by `build_corpus.py` against example messages from
   `seed_dataset.jsonl`) -- so the model has *some* exposure to its own
   safety voice in training, not just general conversation. The guardrail
   pipeline is still what actually enforces safety at inference time (see
   `docs/ARCHITECTURE.md`) -- this is about tone consistency, not a
   substitute for the pipeline.

Sources 2 and 3 are deliberately duplicated several times relative to source
1 when assembled (see `build_corpus.py`'s docstring) -- otherwise the much
larger literature corpus would dominate and the model would barely learn the
`Child:`/`Rosey:` dialogue format at all.

## Running it

```bash
python3 training/scripts/build_corpus.py
python3 training/scripts/train_from_scratch.py --max-iters 3000
```

Trains on Apple Silicon MPS automatically if available, falls back to CPU.
Checkpoints and the tokenizer vocab land in `training/runs/v0/` (gitignored
-- it's a build artifact, not something to commit). `--resume training/runs/v0/checkpoint.pt`
continues from an existing checkpoint instead of restarting.
`backend/app/generation/local_model.py` picks up the checkpoint automatically
the next time the app starts.

## Honest scope and limits

A ~10M-parameter character-level model trained for 3000 steps on a
~600K-character corpus is a **real, working training pipeline** -- the loss
genuinely dropped from 3.79 to 0.13 over the run -- but it is **not a
conversationally competent assistant** at this scale, and the loss curve
itself tells you why: train and val loss fell together (0.13 / 0.07), which
looks like a good fit, but the "val" split isn't independent evidence here.
`synthetic_dialogues.py` is deliberately duplicated 8x when the corpus is
assembled (see `build_corpus.py`), so a random 90/10 split still puts
near-duplicate copies of the same ~36 examples on both sides.

Concretely, tested against the checkpoint in this repo: prompts that match a
training example (or a close paraphrase, e.g. "8 times 7" against a
"7 times 8" example) get near-verbatim, correct-sounding recall. Prompts
with no match in the ~36 hand-written dialogues ("what's your favorite
animal," "how do birds fly") produce fluent-*looking* English that drifts
into the style of the public-domain literature corpus (Alice in Wonderland's
Gryphon and Dormouse showing up in an answer about a whale, for instance)
without actually answering the question. That's memorization of a small,
duplicated example set, not generalization -- an honest reading of the
result, not a knock on the pipeline, which is doing exactly what training a
tiny model on this little data would predict. Closing that gap needs one of:

1. **More data and compute at the same "from scratch" approach** -- a
   meaningfully larger corpus (the literature/dialogue split here is a
   proof of concept, not a production dataset) and more training steps,
   ideally with a trained BPE vocabulary instead of character-level tokens
   so the model isn't spending capacity re-deriving how English words are
   spelled. Genuinely competitive from-scratch pretraining (GPT-3/4-class)
   costs real money (compute budgets in the hundreds of thousands to
   millions of dollars) and is not a realistic target for a solo project --
   worth being upfront about that ceiling rather than implying otherwise.
2. **Fine-tune an open-source pretrained base model instead of pretraining
   from random init** -- `training/scripts/finetune_lora.py` is a complete,
   ready-to-run LoRA fine-tuning scaffold for this (e.g. against Llama/Qwen
   weights downloaded once and fine-tuned locally or on owned infra). This
   is not "starting from scratch" in the literal sense -- it starts from
   someone else's pretrained weights -- but it is **not** a hosted API call
   either: the weights are downloaded once, fine-tuned and run entirely
   under your control, no different in kind from any other open-source
   dependency in this repo. Worth reconsidering as the primary path once
   the from-scratch model's ceiling becomes the actual bottleneck.

Either way, `training/scripts/evaluate.py` (scores the **guardrail
pipeline**, not model quality) and `training/data/seed_dataset.jsonl` (every
entry marked `DRAFT_NEEDS_CLINICAL_REVIEW`) stay relevant regardless of which
generation path wins -- the safety pipeline's correctness doesn't depend on
how good the underlying model gets.
