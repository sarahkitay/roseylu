# Compliance: COPPA + App Store (Kids Category)

Status: **engineering checklist, not a legal document.** This is what the
architecture needs to structurally support so that when counsel reviews it,
the answer is "already built this way" rather than "we'll retrofit it." It
does not substitute for legal review — see `BLOCKERS.md`.

## COPPA — day-one requirements

- [ ] **Verifiable parental consent (VPC)** before any collection of a child's
      personal information. Account creation flow must gate on a completed VPC
      step (credit card verification, signed consent form, or another
      FTC-accepted method) before a child profile is usable — not before
      it's *sold*, before it *exists*.
- [ ] **Data minimization** — collect only what's operationally required: age
      (for tiering), a persona config, account credentials. No name, no
      location, no contact info beyond the parent's. Every new field added to
      `ChildProfile` (`backend/app/models/schemas.py`) should be asked "does
      COPPA data minimization allow this, and do we actually need it."
- [ ] **No behavioral advertising.** No ad SDKs in the client at all — this
      isn't a "keep it minimal" preference, it's a hard exclusion.
- [ ] **No third-party analytics SDKs.** First-party, privacy-preserving
      telemetry only if any. Do not add a drop-in analytics package without
      re-checking this line.
- [ ] **No external links without a parental gate.** Any outbound link
      (including in AI-generated text — this matters once the model can
      produce URLs) needs a parental-gate interstitial.
- [ ] **Deletion on request** — a parent can request full deletion of their
      child's data, and the system needs an actual code path for that, not a
      support-ticket promise. `backend/app/review_queue.py` and any future
      chat-log store must support deletion by child ID.
- [ ] **Right to review** — a parent can request what's stored about their
      child. Given the "minimal dashboard" product decision, this is closely
      related to but distinct from the parent dashboard UI — the dashboard is
      what parents see by default; this is what they can request in full.

## Apple App Store — Kids Category

- [ ] No behavioral advertising (same as COPPA, enforced again at the
      platform layer).
- [ ] No third-party analytics SDKs.
- [ ] No external links without a parental gate.
- [ ] Age rating and Kids Category metadata must match the actual permitted
      content — do not under- or over-declare relative to what the guardrail
      pipeline actually allows through.

## Explicitly out of scope for engineering to decide alone

- What counts as acceptable VPC method for this specific product.
- Whether category-level parent dashboard data (vs. query-level) satisfies
  COPPA's parental access rights, or whether it needs to be paired with a
  full-export-on-request path (drafted above as a hedge, but confirm with
  counsel).
- Data retention windows — see `DATA_RETENTION_POLICY.md`, marked draft.

## Non-negotiable per product vision

These aren't up for later relaxation as growth pressure shows up — they were
called out as day-one, not "add before scale":

> "COPPA compliance is not a feature to be added, it is a structural
> requirement. Verifiable parental consent, data minimization, and deletion
> on request need to be built into the architecture before any child data is
> collected." — `PRODUCT_VISION.md`
