# Architecture

## Current prototype scope

This repo implements the **safety-critical core** as a local Python package,
runnable via CLI or a thin FastAPI wrapper, plus a **real from-scratch
language model** trained on this machine — no cloud infra, no mobile client,
no hosted LLM API anywhere in the runtime path. That's deliberate on two
counts: the guardrail pipeline and response engine are the part that has to
be right before anything else matters, so they're built and testable first,
in isolation; and the generation backend is intentionally self-trained
rather than a wrapped third-party API, per product direction — see
`PRODUCT_VISION.md`'s Technical Architecture section.

```
safeai-kids/
  docs/                        product spec, safety model, compliance, blockers
  model/
    architecture.py             hand-written GPT (embeddings, attention, MLP) -- no pretrained weights
    tokenizer.py                from-scratch tokenizers: char-level and BPE (BPE is the default)
  backend/
    app/
      config.py                 age tiers, guardrail thresholds
      models/schemas.py         ChildProfile, RiskCategory, GuardrailResult, Action
      orchestrator.py           handle_chat_turn() -- the ONE place request-handling logic lives
      guardrails/
        wordlists/*.yaml        editable keyword/slang lists per category
        keyword_filter.py       layer 1: rule-based
        embedding_classifier.py layer 2: semantic (TF-IDF stand-in today)
        llm_judge.py            layer 3: nuanced judgment (heuristic rules, no LLM API)
        pipeline.py             runs all 3 layers, decides Action
      response/
        redirect_engine.py      explain-then-redirect templates, tiered by age
      persona/
        persona_engine.py       child-customized persona + tone blending
      illustration/
        topic_classifier.py     keyword topic detection for cartoon illustrations (zero safety weight)
        topic_responses.py      deterministic math answers (addition/subtraction/multiplication/fractions)
      knowledge/
        curated_qa.py           ~40 hand-authored History/English/Math answers, keyword-matched, fuzzy-typo-tolerant
        quiz.py                 stateful multi-turn quiz/game generator for curated topics
      generation/
        base_model.py           pluggable model backend interface + stub fallback
        local_model.py          loads the from-scratch checkpoint, runs generation
        online_trainer.py       short live training bursts; also READS (never writes) teacher_reviewed.jsonl
      review_queue.py           append-only escalation log (no raw text)
      main.py                   FastAPI app: POST /chat, GET / (chat UI), GET /training-status
      static/index.html         styled single-page chat UI + character designer, no build step
    tests/                      pytest suite against the seed dataset
    requirements.txt            app runtime deps ONLY -- no LLM API SDK ever belongs here
  cli/
    chat.py                     interactive terminal chat, same orchestrator as the API
  training/
    data/
      seed_dataset.jsonl        small, DRAFT, clinically-unreviewed guardrail examples
      synthetic_dialogues.py    hand-authored Child:/Rosey: training dialogue
      corpus/public_domain/     public-domain children's literature (fluency data)
      corpus/combined.txt       assembled training corpus (build_corpus.py output)
      corpus/teacher_reviewed.jsonl  teacher_review_loop.py output (gitignored) -- confirmed/corrected live exchanges
    runs/v0/                    trained checkpoint + tokenizer vocab (gitignored)
      snapshots/                per-checkpoint snapshots, since the best iteration ≠ the final one
    runs/eval/                  simulate_student_eval.py output (gitignored)
    scripts/
      build_corpus.py           assembles the training corpus from the 4 sources above
      train_from_scratch.py     trains model/architecture.py on the corpus, saves checkpoint
      simulate_student_eval.py  offline: external LLMs role-play students by grade + judge replies
      teacher_review_loop.py    offline, continuous: external "teacher" model reviews live conversation
      prepare_dataset.py        seed_dataset.jsonl -> chat-format JSONL (for the alt LoRA path below)
      finetune_lora.py          alt path: LoRA fine-tune of an open-source base model (not Anthropic; needs GPU, not run here)
      evaluate.py                runs guardrail pipeline against seed set, reports metrics
    requirements-eval.txt       anthropic + openai SDKs -- ONLY for scripts/, never backend/app/
  .env                          real keys (gitignored): ANTHROPIC_API_KEY, OPENAI_API_KEY, TEACHER_BACKEND, TEACHER_MODEL
  .env.example                  template for the above (eval/teacher tooling only)
```

## Request flow (implemented)

```
child message + child_profile
        │
        ▼
GuardrailPipeline.evaluate()  ── runs keyword / semantic / judge layers in parallel
        │
        ├─ Action.ALLOW ────────────► ModelBackend.generate() ──► GuardrailPipeline.evaluate() again, on the OUTPUT
        │                              (persona + age-tier system prompt)         │
        │                                                    ┌────────────────────┴───────────────────┐
        │                                                 clean                                 flagged
        │                                                    │                                         │
        │                                                    ▼                                         ▼
        │                                                 response                     generic fallback response
        │                                                    │                        (reported as REDIRECT, not ALLOW)
        │                                                    ▼
        │                                     OnlineTrainer.log_interaction()  (background task, after response sent)
        │                                                    │
        │                                     every few interactions: short training burst,
        │                                     replay-buffered against the original corpus,
        │                                     overwrites training/runs/v0/checkpoint.pt
        │
        ├─ Action.REDIRECT ─────────► RedirectEngine.build_redirect() ──► response
        │
        └─ Action.ESCALATE ─────────► ReviewQueue.append()  (side effect)
                                       + RedirectEngine.build_redirect() ──► response
```

The child always gets a response in the same turn — ESCALATE never blocks or
delays the reply; it only adds a queue entry for specialist review. The
output-side re-check exists because a small, not-instruction-tuned model can
hallucinate an unsafe-sounding fragment on a completely benign input — the
input-side check alone doesn't catch that, since the child's message was
fine. Observed directly during dev testing; see `training/README.md`.
Online-learning bursts never run on the request-handling thread (they're
dispatched as a `BackgroundTasks` job after the HTTP response is already
sent), and they always log the *shown* reply (the fallback, if the output
check fired) rather than a flagged raw generation, so a burst can't
reinforce something that just got caught.

## Target production architecture (not built yet)

Per `PRODUCT_VISION.md`:

- **Clients**: iOS/Android app (standalone), eventually OS-level integration.
- **API**: same FastAPI-shaped service, deployed to AWS, fronting the
  self-trained model.
- **Data**: AWS RDS for account/profile/cross-device-sync state, scoped per
  `DATA_RETENTION_POLICY.md` once that's legally reviewed.
- **Model serving**: a larger, better-trained version of the same
  from-scratch architecture (more data, more compute, likely a trained BPE
  vocab instead of char-level) is the primary path to real capability,
  scaled up without introducing a third-party API dependency. If that
  ceiling proves too low, `training/scripts/finetune_lora.py` is a documented
  fallback -- LoRA fine-tuning an open-source *base model* (e.g. Llama/Qwen
  weights, downloaded once, run locally/on owned infra) -- still never a
  live call to Anthropic or any other hosted provider. Either way, this
  guardrail pipeline stays the safety net, always on regardless of
  generation quality, with real embedding and judge implementations swapped
  into the `Classifier`/`JudgeBackend` interfaces already defined (trained
  classifiers, not an LLM API call, to keep the "no third-party API in the
  live path" property).
- **Image pipeline**: CSAM screening (third-party API — e.g., a hash-matching
  service, not build-your-own), appearance-analysis refusal, zero retention.
  Not started — see `BLOCKERS.md`.
- **Human review queue**: real specialist-facing tooling reading
  `review_queue` entries, with a path to trigger parent notifications.

## Design principle carried through the code

The guardrail pipeline is the trust boundary, not the model. Every code path
that reaches `ModelBackend.generate()` has already passed
`GuardrailPipeline.evaluate()` — there is no direct route from user input to
the base model. Keep it that way as the model gets bigger/better; "the model
is well-behaved" is not a reason to skip the pipeline.

Second principle, added when the generation backend moved off any hosted
API: no `ModelBackend` or `JudgeBackend` implementation in `backend/app/`
may call a third-party LLM API. Both interfaces exist specifically so a
better implementation can be swapped in later — a bigger self-trained model,
a locally-run open-source fine-tune, a real trained classifier for the judge
layer — but "swap in a hosted API call" is not an acceptable implementation
of either interface for this product, independent of which provider.

**Where the line actually is.** "No third-party API" is a rule about the
*app* (`backend/app/`) — what a real child's message touches at runtime. It
is not a rule against ever using a capable external model anywhere in this
repository. `training/scripts/simulate_student_eval.py` and
`training/scripts/teacher_review_loop.py` both call Anthropic/OpenAI
directly, and that's fine: one role-plays students and judges Rosey's
replies to find gaps (like the Columbus one) on demand, the other
continuously reviews live conversation and writes corrected examples for
the model to learn from — both dev/training tools, the same category as
Claude hand-authoring `synthetic_dialogues.py` or `curated_qa.py`: using a
capable model to help build and improve the product, never a runtime
dependency of it. `online_trainer.py` reading `teacher_reviewed.jsonl` is
still on the safe side of this line even though it lives in `backend/app/`
— it only ever reads a file another process wrote; it never makes a network
call or imports either SDK itself (verify with
`grep -rl dotenv backend/app/` — nothing). The test that matters: does a
real child's request ever cause the running app to make a network call to
a third-party model provider? For `backend/app/`, the answer must always be
no, including indirectly — the child's reply is never delayed by, or
swapped for, a live teacher-model call. For `training/scripts/`, it's fine
as long as the *result* (a checkpoint, a curated answer, a corrected
training example, a list of gaps to fix) is what lands in the app — never a
live call in the request path itself. This is also why
`training/requirements-eval.txt` is a separate file from
`backend/requirements.txt`, and why `.env` only ever gets read by scripts
under `training/scripts/`, never by anything under `backend/app/`.
