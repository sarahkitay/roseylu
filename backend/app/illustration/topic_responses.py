"""Deterministic, numerically-correct answers for the math topics the
illustration system already handles (backend/app/illustration/topic_classifier.py).

Why this exists: the from-scratch local model is small and, per
training/README.md's honest accounting, unreliable on exactly the kind of
longer/compound phrasing a real child uses ("i need to learn addition and
subtraction single digits but i dont understand" produced fluent-looking
nonsense during live testing, drifting into unrelated literary text, even
though the topic classifier and illustration for that same message were
both correct). Chasing this further with more training runs hit
diminishing, inconsistent returns -- see training/README.md's run-to-run
variance discussion. This takes a different approach for the subset of
questions where it's actually tractable: for the four numeric math topics,
`topic_classifier.extract_numbers()` already gives real, correct numbers,
and the arithmetic answer to "3 plus 2" is not a language-modeling problem.
Building the sentence directly from those numbers is strictly more reliable
than hoping a ~11M-parameter model narrates it correctly, and it keeps the
text consistent with what the illustration already shows (same numbers,
same total).

This is NOT a general answer to "the model isn't good enough" -- it only
covers topics with a clean numeric template. Everything else (open-ended
conversation, feelings, non-numeric topics like animals/reading/science)
still goes through the generative model with all of its documented
limitations. Templated answers still flow through the same output-side
guardrail check and online-learning log as generated ones (main.py::chat())
-- there's no special-cased bypass of either.
"""
from __future__ import annotations

import re

from app.config import AgeTier

# "how do i do addition with 2 digit numbers" is a genuinely different
# question from "what is 3 plus 2" -- multi-digit column addition (with
# carrying) is a distinct skill, taught later, not just a bigger version of
# single-digit counting. Found live: that exact phrasing fell back to the
# single-digit penny template with default numbers (3, 2) -- content-wise
# fine for "what is addition," but it doesn't address carrying at all,
# which is the actual thing being asked about.
_MULTIDIGIT_ADDITION_HINTS = [
    "2 digit", "two digit", "2-digit", "two-digit", "double digit", "double-digit",
    "multi-digit", "multi digit", "multiple digit", "carrying", "carry the",
    "regroup", "bigger numbers", "larger numbers",
]
_TWO_DIGIT_RE = re.compile(r"\b([1-9][0-9])\b")


def _is_multidigit_addition_question(message: str) -> bool:
    text = message.lower()
    if any(hint in text for hint in _MULTIDIGIT_ADDITION_HINTS):
        return True
    # No explicit "2 digit" phrasing needed if the message already contains
    # two actual multi-digit numbers -- "how do i add 45 and 27" IS a
    # multi-digit addition question by virtue of the numbers themselves.
    return len(_TWO_DIGIT_RE.findall(message)) >= 2


def _extract_two_digit_pair(message: str) -> tuple[int, int]:
    found = [int(n) for n in _TWO_DIGIT_RE.findall(message)]
    if len(found) >= 2:
        return found[0], found[1]
    return 24, 38  # ones digits (4+8=12) deliberately demonstrate carrying by default


def _build_multidigit_addition_answer(message: str, tier: AgeTier | None) -> str:
    a, b = _extract_two_digit_pair(message)

    if tier == AgeTier.PRESCHOOL:
        # Carrying/place-value isn't a developmentally appropriate concept
        # at 3-6 -- found live, at age 4, the full column-addition
        # explanation below (correct, but written for at least a 2nd
        # grader) got shown as-is with no simplification at all. The right
        # move here isn't a "simple" version of carrying -- it's the same
        # explain-then-redirect shape used elsewhere in this app: validate
        # the curiosity, say honestly that it's a skill for later, and
        # pivot to something actually within reach right now.
        return (
            f"Ooh, adding big numbers like {a} and {b} is a math trick for bigger kids -- "
            "you'll learn it soon! Right now, let's count something together. How many "
            "fingers do you have? 1, 2, 3, 4, 5! Want to count something else?"
        )

    ones_a, ones_b = a % 10, b % 10
    tens_a, tens_b = a // 10, b // 10
    ones_sum = ones_a + ones_b
    total = a + b

    if ones_sum >= 10:
        carried = ones_sum // 10
        ones_digit = ones_sum % 10
        return (
            f"For bigger numbers like {a} and {b}, line them up by place value -- ones under "
            f"ones, tens under tens. Start with the ones column: {ones_a} + {ones_b} = "
            f"{ones_sum}. That's too big for one digit, so write down the {ones_digit} and "
            f"carry the {carried} over to the tens column. Now add the tens column, including "
            f"what you carried: {tens_a} + {tens_b} + {carried} = {tens_a + tens_b + carried}. "
            f"Put it together and {a} + {b} = {total}."
        )
    return (
        f"For bigger numbers like {a} and {b}, line them up by place value -- ones under ones, "
        f"tens under tens. Add the ones column first: {ones_a} + {ones_b} = {ones_sum}. Then "
        f"add the tens column: {tens_a} + {tens_b} = {tens_a + tens_b}. Put them together and "
        f"{a} + {b} = {total}."
    )


def _count_up_phrase(total: int) -> str | None:
    """A spoken-out counting sequence, only for totals small enough that
    reciting every number is actually helpful rather than tedious."""
    if 0 < total <= 10:
        return ", ".join(str(i) for i in range(1, total + 1))
    return None


def _preschool_addition(a: int, b: int) -> str:
    total = a + b
    seq = _count_up_phrase(total)
    counting = f" Let's count ALL of them together: {seq}." if seq else ""
    return f"You have {a} pennies, then {b} more pennies come along!{counting} That's {total} pennies!"


def _preschool_subtraction(a: int, b: int) -> str:
    result = a - b
    return f"You have {a} apples. {b} of them go away! Now let's count what's left -- that's {result} apples!"


def _preschool_redirect(concept: str) -> str:
    # Used for concepts that aren't really simplifiable to this age, not
    # just multiplication/fractions by name -- see multidigit addition
    # above for the same shape of response.
    return (
        f"Ooh, {concept} is a math trick for bigger kids -- you'll learn it soon! Right now, "
        "let's count something together instead. How many fingers do you have? 1, 2, 3, 4, 5! "
        "Want to count something else?"
    )


def needs_illustration_suppressed(topic: str, message: str, tier: AgeTier | None = None) -> bool:
    """True when the illustration would be wrong or irrelevant for the
    reply that's actually being shown:
    - multi-digit addition: the penny-counting scene draws one coin per
      unit, which would try to render 62 individual coins for "24 + 38".
    - PRESCHOOL multiplication/fractions: these get a "let's count
      something else instead" redirect (see _preschool_redirect), not an
      actual answer -- a dot-grid or pizza-slice illustration next to that
      text would be a non sequitur.
    """
    if topic == "addition" and _is_multidigit_addition_question(message):
        return True
    if tier == AgeTier.PRESCHOOL and topic in ("multiplication", "fractions"):
        return True
    return False


def build_templated_answer(
    topic: str, numbers: list[int], message: str = "", tier: AgeTier | None = None
) -> str | None:
    """Returns a deterministic answer for topic/numbers, or None if this
    topic doesn't have a numeric template -- callers should fall back to
    the generative model in that case. `message` is only used to detect the
    multi-digit-addition special case; `tier` picks an age-appropriate
    register where one exists (currently just PRESCHOOL, which needs a
    genuinely different response for concepts like carrying and
    multiplication, not just simpler wording of the same explanation --
    found live after a 4-year-old got the same carrying explanation written
    for at least a 2nd grader).
    """
    if topic == "addition" and _is_multidigit_addition_question(message):
        return _build_multidigit_addition_answer(message, tier)

    if len(numbers) != 2:
        return None

    a, b = numbers

    if topic == "addition":
        if tier == AgeTier.PRESCHOOL:
            return _preschool_addition(a, b)
        return (
            f"Let's picture it: {a} pennies in one hand, {b} more in the other. "
            f"Push them all into one pile and count everything -- that's {a} plus {b}, "
            f"which is {a + b}."
        )
    if topic == "subtraction":
        if tier == AgeTier.PRESCHOOL:
            return _preschool_subtraction(a, b)
        return (
            f"Picture {a} apples on a table. Take {b} of them away and count what's "
            f"left on the table -- that's {a} minus {b}, which is {a - b}."
        )
    if topic == "multiplication":
        if tier == AgeTier.PRESCHOOL:
            return _preschool_redirect("multiplication")
        return (
            f"Think of it as {a} groups with {b} things in each group. Instead of "
            f"counting them one at a time, {a} times {b} gives you the total right "
            f"away: {a * b}."
        )
    if topic == "fractions":
        if tier == AgeTier.PRESCHOOL:
            return _preschool_redirect("fractions")
        num, denom = a, b  # first number is the numerator, matching the illustration's a/b caption
        if denom == 0:
            return None
        return (
            f"Imagine cutting something into {denom} equal pieces. If you're talking "
            f"about {num} of those pieces, that's the fraction {num}/{denom}."
        )
    return None
