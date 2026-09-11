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
- `docs/VALUES_MODEL.md` -- honest accounting of where compassion currently
  comes from (100% hand-authored templates, 0% trained into the model
  itself) and what it would take for that to change.
- `docs/EVAL_CRITERIA.md` -- the fuller set of things worth measuring
  (guardrail recall/precision, tone, curriculum accuracy, generation
  competence, online-burst regressions), most of which aren't implemented
  yet -- the spec `training/scripts/evaluate.py` should grow into.
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

For the four numeric math topics (addition/subtraction/multiplication/
fractions), the *text* reply is also built deterministically from the same
extracted numbers (`backend/app/illustration/topic_responses.py`) instead of
going through the generative model at all -- "3 plus 5" is arithmetic, not
a language-modeling problem, and templating it is strictly more reliable
than hoping an ~11M-parameter model narrates it correctly. This was a direct
response to live testing: longer, natural phrasings ("i need to learn
addition and subtraction but i dont understand") reliably produced
incoherent text from the model even though the topic classifier and
illustration for the same message were both already correct. Every other
topic, and all open-ended conversation, still goes through the generative
model with all of its documented limitations below.

## Curated History, English, Math, Science, Life, and Skills answers

Beyond arithmetic, `backend/app/knowledge/curated_qa.py` is a hand-authored,
keyword-matched lookup of curated questions across six subjects -- checked
before falling back to the generative model, same reasoning as the math
templates: the local model has no reliable general knowledge at its current
training scale, and open-ended factual questions produced fluent-looking
nonsense during live testing. This is **original content Claude wrote
directly, not text copied from any textbook** -- see that file's docstring
for the full reasoning, including why sensitive topics (slavery, WWII) are
included (they're standard elementary/middle-school curriculum) and treated
carefully, and why this is a curated *starting set*, not a claim of
exhaustive K-12 coverage.

- **History, English, Math** -- ~50 common curriculum questions ("how did
  Christopher Columbus come to the Americas," "what is a synonym," "what is
  a decimal," cell biology, the periodic table).
- **Science** -- astronomy (sun/moon/gravity/sky), cells, and basic chemistry,
  split into two illustration styles: astronomy gets the sun-and-planets
  scene, everything else gets a subject-neutral "bubbling beaker" scene --
  found live that reusing the astronomy scene for a water-composition answer
  read as a mismatch.
- **Life** -- everyday emotional and family questions ("why does mommy
  yell," "how do I make friends," grief over a pet, fear of the dark, body
  image) written the way a thoughtful child psychologist would talk, not a
  generic AI assistant. Every entry validates the child's feelings and points
  toward a trusted adult rather than trying to fully resolve it in-app --
  Rosey isn't a therapist and the copy doesn't pretend otherwise. Uses its
  own calm, muted illustration (a soft heart, gentle sparkles) since it plays
  next to grief and fear answers, not just cheerful ones.
- **Skills** -- practical how-to questions distinct from Life's emotional
  content, starting with tying shoes (age-tiered: preschoolers get the
  "bunny ears" method specifically, not just simpler wording).

A recurring bug shape while building this out: the typo-tolerant fuzzy
fallback (below) picks a keyword phrase's *longest* word as the word worth
being typo-tolerant about, which works for rare curriculum terms
("calculus," "photosynthesis") but breaks for everyday words ("friend,"
"makes," "world") that can appear in a totally unrelated message and get
matched via an exact (non-typo) ratio. Fixed with a stoplist
(`_GENERIC_ANCHOR_STOPWORDS`) and a `QAEntry.fuzzy_eligible` flag (off for
the whole LIFE category, where a mismatch is costlier than a missed typo) --
see the file's own comments for the running list of what was found and why.

**Columbus is the flagship example of the fuller experience**, built out
after live feedback on the plain-paragraph version: it's tiered by age
(`QAEntry.answers_by_tier` -- a preschooler and a teenager get genuinely
different stories, not the same paragraph at different reading levels),
ends by asking the child a real question instead of a flat "both sides are
valid" hedge, and pairs with a 3-frame animated illustration
(`MULTI_BEAT_BUILDERS` in `index.html`) that advances in step with how much
of the story has been told -- ships sailing, then landing and claiming the
island, then meeting the people who already lived there. Ask "quiz me about
columbus" (or "make a game about columbus") afterward for a real, stateful
4-question quiz (`backend/app/knowledge/quiz.py`) -- it tracks which
question you're on per child, checks your answer against the actual
question, and gives you a score at the end. Only Columbus has the full
tiered+animated+quiz treatment so far; everything else in `curated_qa.py`
still uses a single register and a static illustration, extending the same
pattern to more topics is straightforward but not yet done everywhere.

Age support also now goes down to `AgeTier.PRESCHOOL` (3-6), added on
request -- see `docs/BLOCKERS.md` for the honest caveat that a text chat
interface has real limits for children who can't read or type yet.

## A small hand-authored haiku generator

Ask "write a haiku about the sun" (or moon, rain, ocean, snow, stars,
flowers) and `backend/app/knowledge/haiku.py` returns a real, hand-verified
5-7-5 haiku instead of routing to curated_qa's "what is a haiku" *definition*
entry, which is what used to happen (a fuzzy match on the word "haiku" answered
what a haiku *is*, not the actual request to write one). This is genuine
creative generation, which the small local model has no reliable way to do,
so a handful of classic topics are hand-authored instead of generated --
deliberately narrow, same "curated set, not exhaustive" honesty as
`curated_qa.py`. A topic outside the list falls through to the normal
curated/generative flow rather than fabricating something. Note this can't
help with "write a poem about it" (referring to something from a few turns
back) -- the backend has no conversation memory between messages, so there's
no way to resolve "it" to anything; that's a separate, harder architectural
gap, not something this module attempts to solve.

## Literacy mini-games for early readers

Ask "let's play a letter game" or "play a rhyming game" for a short,
stateful, multiple-choice practice round (`backend/app/knowledge/
literacy_games.py`) -- letter-sound matching and rhyming, aimed at the
youngest kids this app supports. Distinct from the Columbus quiz (which
tests recall of something just taught): this is about building a
pre-reading skill through repeated, game-shaped practice, and answers are
graded the same reliable letter-or-text-match way the quiz engine already
uses, since an open-ended phonetic answer ("say the B sound out loud") isn't
something a text-only interface can check. Completing a round triggers the
same sparkle-burst celebration as finishing a quiz. This doesn't remove
`docs/BLOCKERS.md`'s existing caveat about text chat and pre-readers --
it's useful for kids at the edge of that limit, not a replacement for
audio-based phonics instruction.

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
