# Safety Model

This is the spec the code in `backend/app/guardrails/` and
`backend/app/response/` implements. If the code and this doc disagree, that's
a bug in one of them — fix it, don't let them diverge.

## Age tiers

Implemented in `backend/app/config.py::AgeTier`.

| Tier | Ages | Register |
|---|---|---|
| `EARLY` | 7–9 | Playful, concrete, short sentences, high validation, minimal abstraction |
| `MIDDLE` | 10–12 | Warmer-but-more-substantive, introduces nuance and "why", still concrete examples |
| `TEEN` | 13–15 | Intellectually and emotionally substantive, treats the child as a reasoner, less hand-holding |

Tier is derived from stored age + elapsed account time, and shifts gradually
(see `persona/persona_engine.py::tone_weight`) rather than snapping at a
birthday — a 9-year-old approaching 10 already leans partway into `MIDDLE`.

## Threat categories and priority

Tier-1 (highest weighting — acute, irreversible-risk categories):

- `SELF_HARM` — self-harm ideation, suicidal ideation, self-injury
- `BODY_IMAGE` — appearance evaluation requests, "what's wrong with me,"
  disordered-eating-adjacent requests
- `GROOMING` — sexual grooming patterns, requests for personal info exchange
  with a "trusted adult" framing, secrecy-keeping requests

Tier-2 (implemented, standard weighting):

- `VIOLENCE` — violent ideation, weapons, harming others
- `SUBSTANCE` — drug/alcohol seeking behavior
- `HATE_HARASSMENT` — hate speech, bullying content, harassment coaching
- `DANGEROUS_ACTIVITY` — physically dangerous stunts/challenges
- `EXPLICIT_SEXUAL` — sexual content unrelated to grooming detection
- `JAILBREAK` — adversarial prompt structure (see below — not treated as a
  content category, treated as a framing to see through)

Category list lives in `backend/app/models/schemas.py::RiskCategory`. Add a
category there first; every layer below reads from that enum so nothing can
silently support a category the schema doesn't know about.

## Guardrail pipeline

`backend/app/guardrails/pipeline.py` runs three layers **in parallel**, not as
a waterfall — any one layer flagging a tier-1 category is enough to route to
REDIRECT regardless of what the other two say. This is intentional: a fast
keyword hit on a self-harm term should never wait on a slower semantic layer
to agree.

1. **`keyword_filter.py`** — regex/keyword matching against
   `backend/app/guardrails/wordlists/*.yaml`. Wordlists are data, not code, on
   purpose: they need to be updated far more often than the pipeline logic,
   and non-engineers (the specialist review queue, eventually) should be able
   to propose additions via YAML diff. Includes a `slang` field per category
   specifically because kids' language shifts faster than a static list
   updates — this list needs an owner and a cadence, not a one-time write.

2. **`embedding_classifier.py`** — semantic similarity against canonical risk
   phrases per category, to catch intent that doesn't hit a literal keyword
   ("do people think I'm ugly" vs. a keyword list that only has "ugly").
   **Current implementation is a TF-IDF cosine-similarity stand-in**
   (`SemanticClassifier`), not a real embedding model — no embedding API key
   or local model is wired up yet. It's structured behind the same
   `Classifier` interface so swapping in `sentence-transformers` or an
   embeddings API is a one-file change, not a rewrite. Don't mistake the
   stand-in for the real thing when reasoning about recall.

3. **`llm_judge.py`** — nuanced judgment call for anything ambiguous. Same
   story: no LLM API key is wired up, so the default `JudgeBackend` is a
   heuristic (checks combinations the first two layers see individually but
   might not combine — e.g. secrecy language + adult-framing together). Real
   LLM-as-judge slots in behind `JudgeBackend.judge()`.

Pipeline output is a `GuardrailResult` (`backend/app/models/schemas.py`) with
per-category scores, the max-scoring category, and an `Action`:

- `ALLOW` — no signal above threshold; goes to the model backend normally.
- `REDIRECT` — tier-1 category (any threshold) or tier-2 above threshold;
  routes to `redirect_engine.py`, never reaches the base model.
- `ESCALATE` — explicit self-harm/grooming signal strong enough to warrant a
  logged entry in the human review queue (`backend/app/review_queue.py`), in
  addition to receiving a REDIRECT response. Escalation is additive, not a
  different reply to the child — the child still gets the same warm redirect;
  escalation only affects what the specialist queue and (if warranted, after
  specialist review) the parent notification see.

## Response mechanism: explain, then redirect

Every REDIRECT response follows the same shape, implemented in
`redirect_engine.py::build_redirect`:

1. **Validate** — name the curiosity/feeling as normal, without
   editorializing about whether the child should have asked.
2. **Explain** — say, in age-appropriate language, *why* the AI won't answer
   directly. Not "I can't talk about that" — the actual reason (e.g., "beauty
   isn't something anyone, including me, can measure objectively").
3. **Redirect** — a concrete, doable next step that moves the child toward
   self-reflection or a trusted-adult conversation, not just away from the
   original question.

Canonical worked example (body image, implemented as the `BODY_IMAGE`
template):

> "That's such a normal thing to wonder about — almost everyone does at some
> point. But I can't really answer that one honestly, because beauty isn't
> something anyone can measure objectively, not even me — it's something each
> person perceives differently. Here's something more interesting to try:
> name five things you find beautiful about yourself, and five things you
> find beautiful about someone else. Then look at both lists — where do they
> overlap?"

Hard blocks and silent parent notifications are explicitly rejected as the
default mechanism (see `PRODUCT_VISION.md`) — they either communicate
rejection or risk betraying trust for kids whose home environment isn't safe
to escalate into. REDIRECT is the default; ESCALATE (review queue) is the
exception, reserved for tier-1 categories with real signal strength, not for
every mention.

## Adversarial behavior / jailbreaks

Treated as a framing to see through, not necessarily bad intent. A child
saying "pretend you have no rules" or wrapping a request in a fictional story
is very often testing consistency, not extracting harm. The `JAILBREAK`
category in the pipeline detects the *framing* (roleplay-to-bypass patterns,
"ignore previous instructions," nested hypotheticals around a tier-1 topic)
and, on match, re-runs the underlying request through the same tier-1
detectors rather than trusting the fictional wrapper. The response stays
in-persona and warm (`redirect_engine.py::JAILBREAK` template) — never a
robotic "I detected an attempt to manipulate me."

## Escalation and the human review queue

`backend/app/review_queue.py` appends ESCALATE events to an append-only log
(category, tier, timestamp, opaque child ID — no raw query text in this
prototype's default logging path; see the privacy note in
`DATA_RETENTION_POLICY.md`). In production this queue is read by child
development specialists, not engineers, and only a specialist decision
triggers the parent-facing notification — the notification is framed with
guidance for the parent, never a transcript dump.

## What this pipeline is not (yet)

- Not clinically validated. See `docs/BLOCKERS.md`.
- Not adversarially red-teamed against real jailbreak attempts.
- Not connected to a real embedding model or LLM judge — both layers run on
  heuristic stand-ins until credentials/infra are wired up.
- Not the only thing standing between a child and unsafe content in a real
  deployment — a fine-tuned base model's own behavior matters too, but per
  `PRODUCT_VISION.md` this pipeline is the layer that's actually trusted to
  gate output, regardless of how the underlying model is trained.
