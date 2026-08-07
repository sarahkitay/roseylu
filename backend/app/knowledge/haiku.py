"""Hand-written haikus for a small set of common topics, detected from a
"write/make a haiku about X" style request.

Different problem from curated_qa.py's what_is_a_haiku entry (which
explains what a haiku IS) and different from generic curated Q&A generally:
this is actual creative-writing generation, which the from-scratch local
model has no reliable way to do (see training/README.md's honest
accounting of its limits on open-ended text). Rather than let a request
like this fall through to the model -- or worse, to what_is_a_haiku's
definition, which doesn't fulfill "write one" at all -- a small, verified
5-7-5 haiku is hand-authored per topic. This is deliberately narrow: a
handful of classic nature subjects, not a general poem generator. Asking
for a haiku about anything outside this list falls through to the normal
curated/generative flow, same as any other unmatched request.

Note the real limit this works around: orchestrator.handle_chat_turn()
sees one message at a time with no conversation history, so "write a haiku
about it" (referring to something asked a few turns earlier) can't be
resolved to a topic at all -- that's a separate, harder gap (no multi-turn
context), not something this module attempts to solve.
"""
from __future__ import annotations

import re

_HAIKU_TOPIC_RE = re.compile(r"haiku\s+(?:for|about)\s+(?:the\s+|a\s+|an\s+)?([a-z]+)", re.I)

# Each haiku is hand-verified at 5-7-5 syllables.
_HAIKUS: dict[str, str] = {
    "sun": "Golden light streams down\nWarming the whole waking world\nShadows stretch and fade",
    "moon": "Silver circle glows\nWatching over sleeping towns\nNight's quiet lantern",
    "rain": "Soft drops on the roof\nPuddles bloom on empty streets\nEarth drinks it all in",
    "ocean": "Endless waves roll in\nSalt air carries far off dreams\nBlue meets the sky's edge",
    "sea": "Endless waves roll in\nSalt air carries far off dreams\nBlue meets the sky's edge",
    "snow": "Soft white blankets fall\nSilent world wrapped in cold light\nFootprints disappear",
    "stars": "Tiny lights afar\nScattered across the dark sky\nAncient light still shines",
    "flowers": "Petals slowly bloom\nReaching for the morning sun\nBright colors unfold",
    "flower": "Petals slowly bloom\nReaching for the morning sun\nBright colors unfold",
}


def detect_haiku_topic(message: str) -> str | None:
    match = _HAIKU_TOPIC_RE.search(message)
    return match.group(1).lower() if match else None


def build_haiku_answer(message: str) -> str | None:
    """Returns a hand-written haiku for a recognized topic, or None if the
    message doesn't ask for one or the topic isn't in the small covered
    set -- callers should fall through to the normal curated/generative
    flow in that case."""
    topic = detect_haiku_topic(message)
    if topic is None or topic not in _HAIKUS:
        return None
    return (
        f"Here's a haiku about the {topic} for you:\n\n{_HAIKUS[topic]}\n\n"
        "(Every haiku has 3 lines with 5, 7, then 5 syllables!)"
    )
