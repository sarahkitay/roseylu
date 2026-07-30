# SafeAI for Kids

An AI companion for children, built around a layered safety-guardrail
pipeline and an explain-then-redirect response philosophy, instead of hard
blocks or silent parent surveillance. See `docs/PRODUCT_VISION.md` for the
full product spec.

**Status: early prototype.** The guardrail pipeline, response engine, and API
are real and tested. No fine-tuned model, mobile client, or cloud deployment
exists yet, and several launch-blocking prerequisites (clinical review, legal
counsel, a finalized data policy) are explicitly unresolved. Read
`docs/BLOCKERS.md` before treating anything here as ready for a real child to
use.

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

# talk to it directly, no server needed
python3 cli/chat.py --age 11 --name Rosey --verbose

# or run the API
uvicorn app.main:app --app-dir backend --reload

# run the test suite
cd backend && python -m pytest -q

# calibrate the guardrail pipeline against the seed dataset
python3 training/scripts/evaluate.py
```

No model API key is required to run any of the above -- the guardrail
pipeline, persona engine, and redirect engine are fully self-contained. Set
`ANTHROPIC_API_KEY` to activate real generation on the ALLOW path and a real
LLM-as-judge on the guardrail's third layer (`backend/app/generation/base_model.py`,
`backend/app/guardrails/llm_judge.py`); without it, both fall back to
clearly-labeled stand-ins so the rest of the system stays fully testable.

## Repo layout

```
docs/       product spec, safety model, compliance, blockers
backend/    guardrail pipeline, persona/redirect engines, FastAPI app, tests
cli/        interactive terminal chat against the pipeline
training/   seed dataset (draft), dataset prep, LoRA fine-tune scaffold, evaluator
```

See `docs/ARCHITECTURE.md` for the full breakdown and request flow.
