# SafeAI for Kids

An AI companion for children, built around a layered safety-guardrail
pipeline and an explain-then-redirect response philosophy, instead of hard
blocks or silent parent surveillance. See `docs/PRODUCT_VISION.md` for the
full product spec.

**Status: early prototype.** The guardrail pipeline, response engine, API, and
a real from-scratch-trained local language model are built and tested. No
hosted LLM API is used anywhere in the running app -- the generation backend
is a transformer trained from random initialization on this machine (see
`training/README.md`). No mobile client or cloud deployment exists yet, and
several launch-blocking prerequisites (clinical review, legal counsel, a
finalized data policy) are explicitly unresolved. Read `docs/BLOCKERS.md`
before treating anything here as ready for a real child to use.

## Start here

- `docs/PRODUCT_VISION.md` -- what this is and why, distilled from the
  founding product interrogation.
- `docs/SAFETY_MODEL.md` -- the guardrail pipeline and response-mechanism
  spec that `backend/app/` implements.
- `docs/ARCHITECTURE.md` -- how the pieces fit together, current vs. target.
- `docs/BLOCKERS.md` -- what has to happen, outside of engineering, before
  this is safe to launch.
- `docs/COMPLIANCE_COPPA.md`, `docs/DATA_RETENTION_POLICY.md` -- compliance
  posture (both explicitly marked draft/unreviewed).

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt

# build the training corpus and train the from-scratch model (once)
python3 training/scripts/build_corpus.py
python3 training/scripts/train_from_scratch.py --max-iters 3000

# talk to it directly, no server needed
python3 cli/chat.py --age 11 --name Rosey --verbose

# or run the API + styled chat UI, then open http://localhost:8000
cd backend && uvicorn app.main:app --reload

# run the test suite (from backend/)
python -m pytest -q

# calibrate the guardrail pipeline against the seed dataset
python3 ../training/scripts/evaluate.py
```

No API key of any kind is used anywhere in this repo. The guardrail
pipeline, persona engine, and redirect engine are fully self-contained
regardless of whether a trained model checkpoint exists. Without one, the
ALLOW-path generation backend falls back to a clearly-labeled stub
(`backend/app/generation/base_model.py::StubEchoBackend`) so the rest of the
system stays fully testable; once `training/runs/v0/checkpoint.pt` exists,
`LocalTransformerBackend` picks it up automatically.

## Design your own character

Click the avatar (or "🎨 Customize your character") to open the designer:
pick a body shape, color, eyes, mouth, and an accessory, name it, and it's
who responds to every message from then on. Saved to `localStorage`, so it
persists across page reloads and can be changed anytime. The character
actually animates while it talks -- mouth flaps in sync with the reply text
being typed out, idle blink/bounce the rest of the time -- built entirely
from hand-composed SVG (`backend/app/static/index.html`), no image
generation model involved, consistent with the "no third-party API, and no
heavy model dependency beyond the one this app trains" rule elsewhere in
this repo.

## Topic illustrations

Ask a question that matches a known topic (addition, subtraction,
multiplication, fractions, shapes, science, animals, reading) and a small
cartoon scene renders alongside the reply -- e.g. addition shows pennies
being counted out and added up. Classified server-side
(`backend/app/illustration/topic_classifier.py`, a small keyword matcher
with zero safety weight -- see its docstring) from the *child's question*,
not the model's reply, since the reply is often unreliable at this model
scale (see below) but "what is the child asking about" is a much easier
signal. Illustrations are pre-built SVG scenes, not generated images -- same
reasoning as the character avatars.

## The chat UI trains the model as you use it

`http://localhost:8000/` serves a styled chat page (`backend/app/static/index.html`)
with a status pill showing the online-learning state. Every ALLOW-path
exchange (the guardrail pipeline let the child's message through, and the
model's own reply also passed an output-side check -- see
`docs/SAFETY_MODEL.md`) gets logged, and every few exchanges the model takes
a short real training burst on the live conversation, mixed with a replay
sample of the original corpus so it doesn't just forget everything else.
This updates the actual checkpoint on disk (`training/runs/v0/checkpoint.pt`)
-- it's genuine continual training, not a canned response cache. See
`backend/app/generation/online_trainer.py` for how and why (short bursts,
low learning rate, replay buffer) it's built to not go off the rails on one
weird exchange.

## Repo layout

```
docs/       product spec, safety model, compliance, blockers
model/      from-scratch transformer architecture + tokenizer (no pretrained weights)
backend/    guardrail pipeline, persona/redirect engines, FastAPI app + static chat UI, tests
cli/        interactive terminal chat against the pipeline
training/   corpus assembly, from-scratch training loop, online learning, seed dataset (draft), evaluator
```

See `docs/ARCHITECTURE.md` for the full breakdown and request flow.
