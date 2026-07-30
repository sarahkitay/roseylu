# Architecture

## Current prototype scope

This repo currently implements the **safety-critical core** as a local Python
package, runnable via CLI or a thin FastAPI wrapper — no cloud infra, no
mobile client, no fine-tuned model yet. That's deliberate: the guardrail
pipeline and response engine are the part that has to be right before
anything else matters, so they're built and testable first, in isolation,
without needing AWS, app store accounts, or GPU access.

```
safeai-kids/
  docs/                        product spec, safety model, compliance, blockers
  backend/
    app/
      config.py                 age tiers, thresholds
      models/schemas.py         ChildProfile, RiskCategory, GuardrailResult, Action
      guardrails/
        wordlists/*.yaml        editable keyword/slang lists per category
        keyword_filter.py       layer 1: rule-based
        embedding_classifier.py layer 2: semantic (TF-IDF stand-in today)
        llm_judge.py            layer 3: nuanced judgment (heuristic stand-in today)
        pipeline.py             runs all 3 layers, decides Action
      response/
        redirect_engine.py      explain-then-redirect templates, tiered by age
      persona/
        persona_engine.py       child-customized persona + tone blending
      generation/
        base_model.py           pluggable model backend interface
      review_queue.py           append-only escalation log (no raw text)
      main.py                   FastAPI app: POST /chat
    tests/                      pytest suite against the seed dataset
    requirements.txt
  cli/
    chat.py                     interactive terminal chat against the pipeline
  training/
    data/seed_dataset.jsonl     small, DRAFT, clinically-unreviewed examples
    scripts/
      prepare_dataset.py        seed_dataset.jsonl -> HF dataset format
      finetune_lora.py          LoRA fine-tune scaffold (needs GPU + base model download)
      evaluate.py               runs guardrail pipeline against seed set, reports metrics
```

## Request flow (implemented)

```
child message + child_profile
        │
        ▼
GuardrailPipeline.evaluate()  ── runs keyword / semantic / judge layers in parallel
        │
        ├─ Action.ALLOW ────────────► ModelBackend.generate() ──► response
        │                              (persona + age-tier system prompt)
        │
        ├─ Action.REDIRECT ─────────► RedirectEngine.build_redirect() ──► response
        │
        └─ Action.ESCALATE ─────────► ReviewQueue.append()  (side effect)
                                       + RedirectEngine.build_redirect() ──► response
```

The child always gets a response in the same turn — ESCALATE never blocks or
delays the reply; it only adds a queue entry for specialist review.

## Target production architecture (not built yet)

Per `PRODUCT_VISION.md`:

- **Clients**: iOS/Android app (standalone), eventually OS-level integration.
- **API**: same FastAPI-shaped service, deployed to AWS, fronting a real
  fine-tuned base model.
- **Data**: AWS RDS for account/profile/cross-device-sync state, scoped per
  `DATA_RETENTION_POLICY.md` once that's legally reviewed.
- **Model serving**: fine-tuned open-source base model (primary) + this
  guardrail pipeline (safety net, always on regardless of fine-tune quality)
  + real embedding model and LLM-judge backend swapped into the
  `Classifier`/`JudgeBackend` interfaces already defined.
- **Image pipeline**: CSAM screening (third-party API — e.g., a hash-matching
  service, not build-your-own), appearance-analysis refusal, zero retention.
  Not started — see `BLOCKERS.md`.
- **Human review queue**: real specialist-facing tooling reading
  `review_queue` entries, with a path to trigger parent notifications.

## Design principle carried through the code

The guardrail pipeline is the trust boundary, not the model. Every code path
that reaches `ModelBackend.generate()` has already passed
`GuardrailPipeline.evaluate()` — there is no direct route from user input to
the base model. Keep it that way as real model backends get wired in;
"the fine-tune is well-behaved" is not a reason to skip the pipeline.
