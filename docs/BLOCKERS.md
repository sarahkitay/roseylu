# Blockers to real production launch

This file exists so "prod ready" doesn't quietly come to mean "the code
runs." For a product that talks to children about self-harm, body image, and
grooming risk, code quality is necessary and nowhere near sufficient. These
are the items the founding product interrogation itself flagged as
unresolved — they are not implementable by writing more Python, and no amount
of engineering polish substitutes for them.

## Hard blockers — must close before any real child uses this

1. **Clinical consultation.** No formal child psychologist / developmental
   specialist review has happened. Every response template in
   `redirect_engine.py` and every example in `training/data/seed_dataset.jsonl`
   is a well-read engineer's best guess, not a validated clinical
   intervention. This is explicitly true of the tier-1 categories
   (self-harm, body image, grooming) where a wrong guess is highest-cost.
2. **Legal counsel engagement.** Liability architecture, terms of service,
   and the guardrail audit trail all need to be built with legal guidance.
   None of that exists yet. This also gates the data retention policy (see
   below) and any real claim about COPPA compliance — the checklist in
   `COMPLIANCE_COPPA.md` is an engineering readiness list, not a compliance
   sign-off.
3. **Data retention & deletion policy finalization.** `DATA_RETENTION_POLICY.md`
   is a draft built to give the code something to target; it has not been
   reviewed by counsel or a privacy architect and should not gate real data
   collection decisions.
4. **Human review queue staffing.** The code writes to
   `backend/app/review_queue.py`; nobody reads it yet. A logged escalation
   with no specialist on the other end is not a functioning safety net.
5. **CSAM screening integration.** Image input is a stated product
   requirement and is **not implemented** in this prototype at all — no image
   endpoint exists yet, specifically because CSAM screening has to be in
   place before image upload ships, not after.
6. **Security audit.** AWS/RDS hosting with cross-device sync of
   child-related data needs a real security audit before it holds real
   accounts. Nothing in this prototype has been audited.
7. **Formal age-verification / parental consent flow.** The prototype's
   `ChildProfile` takes an age as a field; it does not implement verifiable
   parental consent, which is a COPPA requirement, not a nice-to-have.
8. **Curriculum content needs educator/subject-matter review.**
   `backend/app/knowledge/curated_qa.py` has ~40 hand-authored History,
   English, and Math answers, including sensitive topics that are standard
   curriculum but easy to get the framing wrong on (slavery and the Civil
   War, WWII and the Holocaust, Columbus and Indigenous peoples). This was
   written carefully, but by an engineer, not a history/curriculum
   specialist -- the same category of gap as clinical review above, just
   for factual/pedagogical accuracy and age-appropriate framing rather than
   emotional safety. Treat it as a draft pending that review, same as
   everything else touching sensitive content in this repo.
9. **`AgeTier.PRESCHOOL` (ages 3-6) is a product-fit question, not just a
   code question.** The tier exists, is tested, and has its own tone rules
   and safety redirect copy, extending the supported age floor from 7 down
   to 3 on request. But this product is a TEXT chat interface, and most
   3-5 year olds can't read or type independently -- adding the tier makes
   the templates *exist*, it doesn't make a text chatbot a good product fit
   for a non-reading toddler using it unsupervised. Plausible legitimate use
   (a parent reading responses aloud together) is different from the
   product's stated model of a child directly using it, and that gap hasn't
   been resolved, just made visible. Worth a real product decision, not an
   assumption baked in by whoever happened to be writing the code that day.

## Capability gap (not a safety blocker, but don't oversell it)

8. **The from-scratch model is not conversationally competent yet, and
   quality is inconsistent even within what it's tuned for.** It's a real,
   working training pipeline -- see `training/README.md` for the full
   story, including a real mistake (an over-trained BPE run that had a
   much lower loss and much worse actual output than the model it was
   meant to improve on) worth reading before trusting a loss number here.
   The corrected version shows genuine improvement on math paraphrase
   retrieval specifically, but not uniformly: even a training example
   reproduced perfectly by the previous version isn't reliably reproduced
   now, non-math topics still drift into literary pastiche, and two
   identically-configured training runs land on different specific
   strengths due to random-seed variance at this scale. This doesn't block
   the safety architecture (the guardrail pipeline gates output regardless
   of generation quality) but it does block a real product launch on its
   own terms: closing it needs either substantially more training
   data/compute at the same from-scratch approach, or falling back to
   fine-tuning a larger open-source pretrained model
   (`training/scripts/finetune_lora.py`) -- both documented as options in
   `training/README.md`, neither involving a third-party hosted API.

## A new consideration from adding online learning

9. **Continual training on live conversation complicates "clinically
   reviewed" as a one-time gate.** `backend/app/generation/online_trainer.py`
   means the model's weights on disk today are not the same weights that
   existed after the last full training run -- they drift with every dev
   session. That's fine for a solo prototype, but it means a future clinical
   review of model *behavior* would need either (a) online learning turned
   off for anything a reviewer signs off on, or (b) a defined re-validation
   cadence, since a model that keeps training on whatever a developer (or
   later, real conversations) feeds it can drift away from a previously
   reviewed baseline without anyone deciding that should happen. Worth a
   deliberate decision before this pattern goes anywhere near real users,
   not just an engineering default carried over from the dev-preview stage.

## What's reasonable to build now, ahead of those blockers

Everything currently in this repo: the guardrail pipeline architecture, the
response-engine shape, the persona system, the training/evaluation scaffold,
and docs that make the open questions visible instead of hiding them. Getting
the *architecture* right — three-layer guardrails as the trust boundary, a
human review queue as a first-class concept, non-persistence of raw child
chat by default — makes the eventual clinical/legal review faster and gives
reviewers something concrete to react to, which is worth more than a
whiteboard diagram. What isn't reasonable is treating any of it as safe to
put in front of a real child before the blockers above close.

## Suggested order of operations

1. Clinical + legal engagement (parallel, both needed before real data
   collection of any kind).
2. Finalize `DATA_RETENTION_POLICY.md` with counsel.
3. Clinically review and rewrite `redirect_engine.py` templates and
   `training/data/seed_dataset.jsonl` before either is treated as anything
   more than a scaffold.
4. Staff the review queue (even a single specialist, part-time, is better
   than an unread log).
5. Build the CSAM screening integration before any image endpoint ships.
6. Security audit before real accounts / real AWS deployment.
7. VPC-compliant account flow before any child data collection begins.
