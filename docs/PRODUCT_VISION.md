# SafeAI for Kids — Product Vision

Source of truth for product decisions. Distilled from the founding pre-production
interrogation (2026-07-30). Update this file when a decision changes — do not let
it drift from what's actually built.

## Product Definition

- **Form factor**: Standalone app first. The long-term goal is to mature into an
  embeddable infrastructure layer / OS-level integration (iOS-first), but the
  safety architecture must be proven in a controlled, single-surface product
  before opening to third-party embedding.
- **Buyer**: Parents. The pitch is a middle ground between "no AI access" (which
  kids route around anyway) and unrestricted access — a guided, developmentally
  appropriate environment.
- **Age range**: Not a single bracket. Three developmental tiers that the product
  grows through automatically as a child ages and accrues time in the product,
  adjusting tone, complexity, permitted topics, and safety weighting. See
  `SAFETY_MODEL.md` for the concrete tier boundaries used in code
  (`backend/app/config.py::AgeTier`).
- **Tiering**: One product, tiered experience — not separate SKUs. A 7-year-old
  and the same user at 12 should feel like they're using a product that grew up
  with them, with no manual reconfiguration.
- **Geography**: US only at launch. COPPA compliance is foundational, not a
  later pass. See `COMPLIANCE_COPPA.md`.

## The Safety Model (summary — full detail in `SAFETY_MODEL.md`)

- **Threat priority**: Every category is implemented, but body image harm,
  self-harm ideation, and sexual grooming carry the highest weighting —
  acute, developmentally sensitive, and irreversible if mishandled.
- **Response mechanism**: Thoughtful redirect, not a hard block and not a
  silent parent notification. Validate the curiosity, explain why the AI won't
  answer directly, then redirect toward genuine self-reflection.
- **Explain vs. redirect**: Both, explanation first. The goal is comprehension,
  not compliance — a refused child hasn't learned anything about why.
- **Parent dashboard**: Minimal, category-level only. No query-level or
  emotional-content access. Genuine safety concerns (post human review) reach
  parents as a carefully framed, non-shaming notification with guidance on how
  to open the conversation with their child.
- **Edge case audit**: Human review queue staffed by child development
  specialists. Automated systems will misclassify edge cases; in this domain
  that has to be caught by a person, not a metric.

## Body Image and Self-Worth

Redirect with validation, never evaluation. The AI never scores or comments on
appearance. Canonical example (implemented verbatim in
`backend/app/response/redirect_engine.py`):

> A child asks the AI to evaluate their appearance / find their flaws. Response
> acknowledges how universal that curiosity is, explains that beauty is
> perceived rather than measured (so the AI answering would just be handing
> back an opinion dressed as fact), then redirects: name five things you find
> beautiful about yourself, five things you find beautiful in someone else —
> do the lists overlap?

Not yet done: formal child psychologist / developmental specialist
consultation. The product philosophy is informed by reading and proximity to a
practicing psychotherapist, but **no training data should be treated as
clinically validated until that consultation happens.** Every seed example in
`training/data/seed_dataset.jsonl` is marked `DRAFT — NEEDS CLINICAL REVIEW`
for this reason.

## Technical Architecture (summary — full detail in `ARCHITECTURE.md`)

- **Model strategy**: Fine-tune an open-source base model as the primary path;
  layered API guardrails as a fallback/supplement, not the other way around.
  In the current build, the guardrail pipeline is the thing that actually
  gates unsafe output — treat it as the safety-critical layer regardless of
  how good the fine-tune gets. A fine-tune should never be trusted alone.
- **Hosting**: Cloud (AWS + RDS), cross-device continuity as a hard UX
  requirement (phone ↔ tablet).
- **Latency budget**: Consumer-AI-comparable (a few seconds). Kids disengage
  from sluggish experiences — this is a product-quality bar, not just an infra
  one.
- **Modality**: Text + image input. Image pipeline needs CSAM screening, a
  strict no-appearance-analysis rule, and no image retention. Not yet built —
  see `BLOCKERS.md`.
- **Team background**: Full-stack engineer, comfortable integrating AI APIs,
  learning into fine-tuning/classifier training.

## Guardrail Layers (full detail in `SAFETY_MODEL.md`)

Three layers in parallel, not in sequence — a single layer flagging risk is
enough to change the outcome:

1. **Rule-based** keyword/regex filtering — fast, expansive, and expected to
   need constant updates as kid slang evolves.
2. **Embedding-based semantic classifier** — catches intent that doesn't match
   any literal keyword.
3. **LLM-as-judge** — nuanced risk assessment for cases the first two layers
   are unsure about.

Adversarial behavior (jailbreaks, "pretend you have no rules," fictional
framing) is treated as a child testing the system's consistency, not
necessarily malicious extraction — the in-character, warm response matters as
much as the detection.

## Accounts

Age-verified accounts with parent-established credentials. Future: a
supervisory layer limiting access to other, unguarded AI surfaces on the same
device — a broader digital-safety environment, not just one app.

## Content and Persona

- **Persona**: Customizable, built collaboratively with the child (prompting +
  visual design). Ownership reduces adversarial behavior — kids are less
  likely to try to break something they built.
- **Permitted subjects**: Homework support, creative exploration, emotional
  processing — guided learning, not answer delivery. The product should teach
  the thinking, not remove the cognitive work.
- **Tone**: Weighted blend by developmental tier — playful/simple for younger
  kids, more intellectually and emotionally substantive for older kids. The
  blend shifts gradually, not as a hard cutover at a birthday.

## Compliance and Distribution (full detail in `COMPLIANCE_COPPA.md`)

- Apple App Store guidelines for under-13 apps reviewed: no behavioral ads, no
  third-party analytics SDKs, no external links without a parental gate.
  Architecture is built around these constraints from day one, not retrofitted.
- COPPA compliance is day one: verifiable parental consent, data minimization,
  deletion on request — structural, not a feature flag.
- Data retention/deletion policy: drafted in `DATA_RETENTION_POLICY.md`, but
  **explicitly unreviewed by counsel or a privacy architect** — do not treat it
  as final. See `BLOCKERS.md`.
