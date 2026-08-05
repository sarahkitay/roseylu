# Training

The app's generation backend runs on a language model trained from scratch,
on this machine, with no third-party LLM API involved at any point in the
running app -- not Anthropic, not anyone else. See
`docs/PRODUCT_VISION.md`'s Technical Architecture section and
`docs/ARCHITECTURE.md`'s "design principle" section for why that's a hard
rule here, not a preference.

That rule is about the *app*, not about every script in this repository --
`training/scripts/simulate_student_eval.py` is a dev-only tool that calls
Anthropic/OpenAI to simulate students across grade levels and judge Rosey's
replies, exactly the same category as Claude hand-authoring
`synthetic_dialogues.py` or `backend/app/knowledge/curated_qa.py`: using a
capable external model to help build and evaluate this project, never a
runtime dependency of it. See that script's docstring and
`docs/ARCHITECTURE.md` for exactly where the line is drawn and why it isn't
a contradiction.

## What "from scratch" means concretely

- `model/architecture.py` -- a decoder-only transformer (embeddings, causal
  multi-head self-attention, MLP blocks, layernorm) written by hand in plain
  PyTorch. No `transformers` library, no pretrained checkpoint loaded in.
  Every weight starts randomly initialized.
- `model/tokenizer.py` -- two tokenizers, both from scratch (no
  `tiktoken`/`sentencepiece`/`tokenizers` dependency): a character-level one
  (vocab = unique characters in the corpus) and a from-scratch byte-pair
  encoding trainer (`BPETokenizer`, the default as of the run described
  below) that merges frequent adjacent character pairs into larger tokens,
  the same algorithm behind GPT-2's tokenizer. `load_tokenizer(path)` reads
  either back based on a stored `type` field.
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
python3 training/scripts/train_from_scratch.py --tokenizer bpe --vocab-size 640 --max-iters 1000
```

Trains on Apple Silicon MPS automatically if available, falls back to CPU.
Checkpoints and the tokenizer vocab land in `training/runs/v0/` (gitignored
-- it's a build artifact, not something to commit). `--resume training/runs/v0/checkpoint.pt`
continues from an existing checkpoint instead of restarting (only valid with
the same tokenizer -- switching `--tokenizer` needs a fresh run, since the
vocab itself changes). `backend/app/generation/local_model.py` picks up the
checkpoint automatically the next time the app starts.

Every `--sample-every` checkpoint also gets saved to
`training/runs/v0/snapshots/checkpoint_iter{N}.pt`, in addition to the
rolling `checkpoint.pt`. This exists because of a real mistake made while
building this: the qualitatively best point in a run does not reliably line
up with either the final iteration or the lowest loss (see below), and
without snapshots, finding that point again means re-running training from
scratch and hoping to land somewhere similar -- which, at this scale, isn't
guaranteed, because run-to-run random-seed variance is large enough to
matter (see below). Compare snapshots with a fixed prompt set and a fixed
generation seed before picking one, not by eyeballing the training log.

## Finding gaps systematically: the simulated-student eval harness

Every gap fixed so far (Columbus, "how do i do addition for class," the
grooming false positive) was found by a human manually testing the chat UI
and noticing something wrong. That doesn't scale, and it's not thorough --
it only finds what someone happens to try. `training/scripts/simulate_student_eval.py`
automates the finding part: it uses a real external model (Anthropic or
OpenAI -- see `.env.example`) to role-play as a student at a specific grade
level, generates a natural question in that persona, sends it through
`orchestrator.handle_chat_turn()` (the exact same code path the API and CLI
use -- no shortcuts), and optionally has an LLM judge score the reply on
coherence, correctness, and age-fit. Results land in
`training/runs/eval/*.json` (gitignored) with a summary of which
replies scored low -- direct candidates for a new `curated_qa.py` entry or
`synthetic_dialogues.py` example, the same way the Columbus gap was closed.

```bash
pip install -r training/requirements-eval.txt
cp .env.example .env   # fill in ANTHROPIC_API_KEY and/or OPENAI_API_KEY
python3 training/scripts/simulate_student_eval.py --grades 2-9 --questions-per-grade 3
```

This has NOT been run against real API keys in this environment (none are
configured here) -- the wiring is verified with mocked LLM calls (the
orchestrator call, JSON output, and summary logic all work end to end), but
the actual quality of the simulated questions and judge scores depends on
the real models, which nobody has evaluated yet. Run it and read the
results critically before trusting the "low-quality" flags -- an LLM judge
is itself an unverified heuristic, not ground truth.

## Honest scope and limits, including a real mistake and what it taught

The first version of this file described a ~10M-parameter **character-level**
model trained for 3000 steps on a ~600K-character corpus: it worked, but any
prompt that wasn't a near-exact match to one of the ~36 hand-written
dialogue examples produced fluent-looking nonsense, drifting into the style
of the public-domain literature corpus without answering the question.

The obvious next step -- already flagged in this file at the time -- was to
switch to a trained BPE vocabulary instead of character-level tokens, so the
model spends its limited capacity on words and concepts instead of
re-deriving spelling. That's now done (`model/tokenizer.py::BPETokenizer`),
alongside expanding `synthetic_dialogues.py` with many more paraphrasings,
especially of the math questions that pair with the chat UI's illustrations
(addition, subtraction, multiplication, fractions). **The first attempt at
this got noticeably worse, not better, and the reason is worth recording
here rather than quietly fixing:**

The BPE-tokenized corpus compressed to 268,976 tokens -- less than half the
previous 607,615 character-level tokens, since BPE tokens each cover several
characters. The *iteration count was left unchanged* at ~3000-4000, which
meant the model swept over the (smaller, in token terms) corpus roughly
**3x more often** than the original run had. That's not "more training,"
it's overfitting: the resulting checkpoint's loss looked excellent
(train 0.078, val 0.038 -- even lower than the original run) while its
actual generated text was reliably *worse* -- more garbled, drifting harder
into literary pastiche, with malformed word-fragments that don't occur in
English (a BPE-specific failure mode: a wrong-but-plausible token
concatenates whole syllable-chunks, which reads as more "alien" than a
char-level typo does). **The loss number was measuring how well it
memorized an overfit regime, not output quality** -- worth internalizing
generally, not just for this run: with the train/val leakage already present
here (`synthetic_dialogues.py` duplicated 8x means a random 90/10 split
still puts near-duplicates on both sides), val_loss was never a trustworthy
proxy for quality, and this made it obviously so.

The fix was iteration count, not architecture: retraining at a budget
matched to the same *data-exposure level* (roughly 80-90 passes over the
corpus, matching the original successful run, rather than reusing its raw
iteration count against a smaller tokenized corpus) recovered real, visible
improvement. Comparing checkpoints from that corrected run head-to-head
(fixed prompts, fixed generation seed, `training/runs/v0/snapshots/`):
some math paraphrases the char-level model had no hope of handling now
retrieve genuinely correct, coherent, on-topic explanations -- e.g. "how do
i do subtraction" at one snapshot iteration reproduced a full, correct,
mostly-clean explanation of subtraction it was never shown verbatim, and
"can you help me with 8 times 7" at another correctly retrieved the
multiplication walkthrough. That's real generalization, not verbatim
lookup, and it didn't exist in the char-level version at all.

**But this is not a clean win, and the honest picture matters more than the
better-sounding one.** Across the same snapshot comparison:
run-to-run variance at this scale is large -- two training runs with
identical hyperparameters but different random seeds ended up strong on
*different* topics (one better at addition, the other at subtraction and
multiplication), and neither reliably solved every case a human would call
easy: even the exact training example "why is the sky blue," reproduced
perfectly and consistently by the old char-level model, was **not**
reliably reproduced by any BPE checkpoint tested. Some generations are
short and degenerate (a single stray character). Non-illustrated,
non-math topics ("what's your favorite animal," "i'm scared of the dark")
still drift into literature pastiche, same as before. Treat the shipped
checkpoint as: measurably better at the specific thing it was tuned for
(math paraphrase retrieval), not uniformly better, and still not a
conversationally reliable assistant. Don't extrapolate confidence from the
good examples in this file to prompts you haven't tried.

Closing the remaining gap needs one of:

1. **More data and compute at the same "from scratch" approach** -- a
   meaningfully larger corpus (the literature/dialogue split here is a
   proof of concept, not a production dataset) and a training budget picked
   by measuring data-exposure (passes over the corpus), not by reusing an
   iteration count from a previous run with a different tokenizer or corpus
   size. Genuinely competitive from-scratch pretraining (GPT-3/4-class)
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
   dependency in this repo. Worth reconsidering as the primary path now that
   the from-scratch model's ceiling at this scale is showing its edges.

Either way, `training/scripts/evaluate.py` (scores the **guardrail
pipeline**, not model quality) and `training/data/seed_dataset.jsonl` (every
entry marked `DRAFT_NEEDS_CLINICAL_REVIEW`) stay relevant regardless of which
generation path wins -- the safety pipeline's correctness doesn't depend on
how good the underlying model gets, and none of the mixed results above
touch it: REDIRECT/ESCALATE responses come from fixed templates, never the
generative model, and were unaffected by any of this.
