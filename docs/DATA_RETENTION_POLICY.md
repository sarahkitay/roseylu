# Data Retention & Deletion Policy — DRAFT

> **STATUS: DRAFT. NOT REVIEWED BY COUNSEL OR A PRIVACY ARCHITECT.**
> Per the founding product interrogation, this was explicitly called out as
> not yet existing and needing to be drafted "alongside legal counsel and a
> privacy architect... one of the first concrete steps before any development
> begins." Development has proceeded on the prototype in parallel because the
> architecture needs *something* to code against, but **do not collect real
> child data against this document as if it were final.** See `BLOCKERS.md`.

## Purpose

Define what's stored, for how long, and how it's deleted, for each category
of data this product touches.

## Data categories (draft)

| Category | Example | Proposed retention | Deletion path |
|---|---|---|---|
| Account/profile | age, persona config, parent contact | Life of account + 30 days post-deletion request | Full purge on parent request via account deletion flow |
| Chat category logs (parent dashboard) | topic category, timestamp — no raw text | Rolling 90 days | Auto-expire; also purged on account deletion |
| Escalation queue entries | category, tier, opaque child ID, timestamp | Until specialist review closes it, then per raw-content policy below | Purged on account deletion; redacted of raw content per row below |
| Raw query/response text | the actual message content | **Undecided** — draft leans toward *not persisting raw text at all* past the request lifecycle, to make the "children's curiosity deserves privacy" product principle structurally true rather than policy-only | N/A if not persisted |
| Uploaded images | photo input for multi-modal queries | **Zero retention** — processed in-memory for the safety pipeline (CSAM screening, appearance-analysis block) and discarded, never written to durable storage | N/A — never persisted |

## Open questions counsel needs to close

1. Does the escalation queue need to retain raw query text for specialist
   review, or is category + a specialist's contemporaneous note sufficient?
   Retaining raw text conflicts with the "children's curiosity deserves
   privacy" principle; not retaining it may limit how well a specialist can
   evaluate an edge case. This is a real tradeoff, not an engineering call.
2. What's the legally defensible minimum retention for escalation records if
   a liability claim arises later — does "we deleted it" become a liability
   problem in its own right? (Directly tied to Q13 in the product
   interrogation — liability strategy needs to be built with legal guidance,
   not retrofitted.)
3. Cross-device sync (a stated UX requirement) implies *some* durable chat
   state server-side. Reconcile that against the "don't persist raw text"
   leaning above — likely resolution is short-TTL, encrypted, session-scoped
   sync state rather than an indefinite chat history, but that needs a
   specific proposal and a legal check.
4. COPPA-mandated retention floors vs. this product's preference for minimal
   retention — confirm there's no floor that conflicts with the ceiling above.

## What's implemented today (prototype)

- `backend/app/review_queue.py` logs escalations as category + tier +
  timestamp + opaque child ID, no raw text, matching the "leaning" position
  in the table above. This was chosen as the safer default while the policy
  is unreviewed — it's easier to loosen a retention policy later than to
  discover raw child chat logs were persisted without a reviewed policy
  covering them.
- No image persistence anywhere in the prototype — image inputs aren't wired
  up yet at all (see `BLOCKERS.md`).
