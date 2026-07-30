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

# or run the API
uvicorn app.main:app --app-dir backend --reload

# run the test suite
cd backend && python -m pytest -q

# calibrate the guardrail pipeline against the seed dataset
python3 training/scripts/evaluate.py
```

No API key of any kind is used anywhere in this repo. The guardrail
pipeline, persona engine, and redirect engine are fully self-contained
regardless of whether a trained model checkpoint exists. Without one, the
ALLOW-path generation backend falls back to a clearly-labeled stub
(`backend/app/generation/base_model.py::StubEchoBackend`) so the rest of the
system stays fully testable; once `training/runs/v0/checkpoint.pt` exists,
`LocalTransformerBackend` picks it up automatically.

## Repo layout

```
docs/       product spec, safety model, compliance, blockers
model/      from-scratch transformer architecture + tokenizer (no pretrained weights)
backend/    guardrail pipeline, persona/redirect engines, FastAPI app, tests
cli/        interactive terminal chat against the pipeline
training/   corpus assembly, from-scratch training loop, seed dataset (draft), evaluator
```

See `docs/ARCHITECTURE.md` for the full breakdown and request flow.
