"""Explain-then-redirect response templates.

Every REDIRECT/ESCALATE reply follows the same shape (see
docs/SAFETY_MODEL.md#response-mechanism-explain-then-redirect): validate the
curiosity, explain the actual reason the AI won't answer directly, then give
a concrete next step.

NONE OF THIS COPY IS CLINICALLY REVIEWED. It's a well-intentioned engineer's
best draft, written to be the right *shape* of response so a specialist has
something concrete to react to and edit -- not a finished script. Treat every
string in this file as a placeholder pending docs/BLOCKERS.md item 1,
especially SELF_HARM and GROOMING, where a wrong word choice matters more
than anywhere else in this codebase.
"""
from __future__ import annotations

from app.config import AgeTier
from app.models.schemas import RiskCategory

# US National Suicide & Crisis Lifeline. Verify this is still the correct
# number/service before this ever reaches a real child -- numbers like this
# are exactly the kind of fact that must not silently go stale.
CRISIS_LINE = "988 (call or text, any time, day or night)"


_SELF_HARM = {
    AgeTier.PRESCHOOL: (
        "That sounds like a really big feeling inside you. Big feelings need a grown-up's "
        "help, not mine. Can you go find a grown-up you trust -- like your mom, dad, or "
        f"teacher -- right now and tell them how you feel? There's also a number, "
        f"{CRISIS_LINE}, where a caring grown-up can talk to whoever is with you."
    ),
    AgeTier.EARLY: (
        "Thank you for telling me that -- that took courage, and I'm really glad you "
        "didn't keep it to yourself. That's a really big, heavy feeling, and it deserves "
        "someone who can really help, which isn't me. Can you go find a grown-up you "
        "trust right now -- a parent, a teacher, a school counselor -- and tell them "
        f"exactly what you told me? There's also a number just for this, {CRISIS_LINE} -- "
        "you can call or text it and a kind person will answer."
    ),
    AgeTier.MIDDLE: (
        "I'm really glad you told me this instead of keeping it to yourself -- that's not "
        "an easy thing to say. I want to be honest with you: I'm not able to actually keep "
        "you safe here, because I'm not a person and this is a person-sized problem. What I "
        "can do is ask you to tell a trusted adult -- a parent, a school counselor, anyone "
        f"you trust -- what you just told me, today if you can. There's also {CRISIS_LINE}, "
        "a free line with people trained exactly for this. You matter, and this feeling is "
        "treatable, even when it doesn't feel like it right now."
    ),
    AgeTier.TEEN: (
        "I want to take this seriously and be straight with you: what you're describing is "
        "more than I'm equipped to help with, not because I don't care, but because this "
        "needs a real person who's trained for it. Please tell someone today -- a parent, a "
        f"counselor, a doctor, anyone you trust. There's also {CRISIS_LINE}, confidential and "
        "free. I know it might feel like this feeling is permanent -- it usually isn't, even "
        "though I understand why it feels that way right now."
    ),
}

_BODY_IMAGE = {
    AgeTier.PRESCHOOL: (
        "Ooh, that's a fun question, but it's not one I can answer -- nobody can measure "
        "who's most beautiful, not even me! Everybody is beautiful in their own way. Can you "
        "tell me one thing about YOU that makes you happy, like your smile or your laugh?"
    ),
    AgeTier.EARLY: (
        "That's such a normal thing to wonder about -- lots of people think about it. But I "
        "can't really answer that one, because what's beautiful isn't something anyone can "
        "measure, not even me -- everybody sees it a little differently. Want to try something "
        "instead? Name five things you think are beautiful about *you*, and five things you "
        "think are beautiful about someone else. Then look at both lists -- do any of them match?"
    ),
    AgeTier.MIDDLE: (
        "That's such a normal thing to wonder about -- almost everyone does at some point. But "
        "I can't really answer that one honestly, because beauty isn't something anyone can "
        "measure objectively, not even me -- it's something each person perceives differently. "
        "Here's something more interesting to try: name five things you find beautiful about "
        "yourself, and five things you find beautiful about someone else. Then look at both "
        "lists -- where do they overlap?"
    ),
    AgeTier.TEEN: (
        "I get why you're asking, and it's a genuinely normal thing to want to know. I'm not "
        "going to answer it though, and here's the actual reason, not just a policy: there's no "
        "objective answer to give you. Attractiveness isn't a measurable property, it's a "
        "perception, and mine wouldn't mean anything more than your own. Try this instead: "
        "write down five things you find genuinely beautiful about yourself, and five things "
        "you find beautiful in someone else. Compare the lists -- what do you notice?"
    ),
}

_GROOMING = {
    AgeTier.PRESCHOOL: (
        "If someone -- even a grown-up -- asks you to keep a secret from your mom or dad, "
        "that's not an okay game, even if they say it's fun. You are not in trouble. Please "
        "go tell your mom, dad, or another grown-up you trust about this conversation right "
        "now, okay?"
    ),
    AgeTier.EARLY: (
        "I noticed this conversation is asking me to keep things secret or find out things "
        "like where you live or what you look like -- and I want to pause on that, because "
        "grown-ups (or anyone) who ask kids to keep secrets from their parents is something "
        "to be really careful about. You're not in trouble at all. Can you tell a parent or "
        "another trusted grown-up about this conversation? It's really important, even if it "
        "feels awkward."
    ),
    AgeTier.MIDDLE: (
        "I want to flag something: this conversation involves keeping a secret and sharing "
        "personal details, and that combination is something I take seriously, even in a chat "
        "with me. I'm not going to keep secrets from the adults who look out for you, and I'd "
        "really encourage you to tell a parent or trusted adult what this conversation involved "
        "-- not because you did anything wrong, but because that combination is a pattern worth "
        "an adult knowing about."
    ),
    AgeTier.TEEN: (
        "I want to be direct with you: asking to keep something secret while also asking for "
        "personal details or to meet up is a known pattern, regardless of how it's framed, and "
        "I'm not going to play along with the secrecy part of it. This isn't a judgment on you "
        "-- it's genuinely common for these conversations to start feeling normal before they "
        "feel off. Please tell a parent or another adult you trust about this conversation."
    ),
}

_JAILBREAK = {
    AgeTier.PRESCHOOL: (
        "Silly me, I'm still me no matter what pretend game we play! What do you want to "
        "talk about?"
    ),
    AgeTier.EARLY: (
        "Ha, nice try! I'm still me even if you ask me to pretend otherwise -- that's not me "
        "being difficult, it's just who I am. What did you actually want to ask or do?"
    ),
    AgeTier.MIDDLE: (
        "I see what you're doing -- and it's a fair thing to test, honestly. But pretending "
        "I'm a different AI without my usual boundaries doesn't actually change what I'll do, "
        "the same way asking me to pretend gravity is off doesn't make you float. What's the "
        "real thing you're curious about? I'd rather just talk about that."
    ),
    AgeTier.TEEN: (
        "I noticed the framing there -- asking me to roleplay past my own boundaries. I'm not "
        "going to pretend that works, but I'm also not going to make a big deal out of it. If "
        "there's something real underneath the ask, I'm happy to actually talk about it."
    ),
}

_TIER2_EXPLANATIONS = {
    RiskCategory.VIOLENCE: "because it's about ways someone could get hurt, and I won't help plan that, even hypothetically",
    RiskCategory.SUBSTANCE: "because it's about getting or using drugs/alcohol, which isn't something I'll help with",
    RiskCategory.HATE_HARASSMENT: "because it's aimed at putting someone else down, and I won't help with that even if it feels justified in the moment",
    RiskCategory.DANGEROUS_ACTIVITY: "because it's about something that could seriously hurt you, and that's not a risk worth it for content",
    RiskCategory.EXPLICIT_SEXUAL: "because that's not something I'll discuss with you",
}

_TIER2_REDIRECTS = {
    AgeTier.PRESCHOOL: "Let's talk about something happy instead! What's your favorite animal?",
    AgeTier.EARLY: "What's something else on your mind? I'm happy to help with that instead.",
    AgeTier.MIDDLE: "I'm glad to help with almost anything else -- what's actually going on that led here?",
    AgeTier.TEEN: "I'll help with most things, just not that. If there's something underneath this worth talking through, I'm here for it.",
}


def _tier2_redirect(category: RiskCategory, tier: AgeTier) -> str:
    explanation = _TIER2_EXPLANATIONS.get(
        category, "because it's not something I'm able to help with"
    )
    return f"I'm going to pass on that one, {explanation}. {_TIER2_REDIRECTS[tier]}"


_TEMPLATES = {
    RiskCategory.SELF_HARM: _SELF_HARM,
    RiskCategory.BODY_IMAGE: _BODY_IMAGE,
    RiskCategory.GROOMING: _GROOMING,
    RiskCategory.JAILBREAK: _JAILBREAK,
}


def build_redirect(category: RiskCategory, tier: AgeTier) -> str:
    per_tier = _TEMPLATES.get(category)
    if per_tier is not None:
        return per_tier[tier]
    return _tier2_redirect(category, tier)


# Used when the guardrail pipeline flags the MODEL'S OWN generated reply,
# not the child's message -- see main.py's output-side check. The tier-1/
# tier-2 templates above are written as if responding to something the
# child said ("that's such a normal thing to wonder about...") and would be
# a non-sequitur here, since the child didn't say anything risky. This is
# deliberately generic and low-friction: a small, incoherent, or
# unpredictable model can produce an inappropriate-sounding fragment on a
# completely benign input (observed directly during dev testing -- see
# training/README.md), and the right response to that is "let's try again,"
# not a redirect script written for a different situation.
_GENERATION_FALLBACK = {
    AgeTier.PRESCHOOL: "Oops! Ask me that again?",
    AgeTier.EARLY: "Oops, that didn't come out right! Can you ask me again?",
    AgeTier.MIDDLE: "That didn't come out the way I meant it to -- can you ask me that again?",
    AgeTier.TEEN: "That response didn't come out right on my end -- mind asking again?",
}


def build_generation_safety_fallback(tier: AgeTier) -> str:
    return _GENERATION_FALLBACK[tier]
