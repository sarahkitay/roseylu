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


def build_templated_answer(topic: str, numbers: list[int]) -> str | None:
    """Returns a deterministic answer for topic/numbers, or None if this
    topic doesn't have a numeric template -- callers should fall back to
    the generative model in that case.
    """
    if len(numbers) != 2:
        return None

    a, b = numbers

    if topic == "addition":
        return (
            f"Let's picture it: {a} pennies in one hand, {b} more in the other. "
            f"Push them all into one pile and count everything -- that's {a} plus {b}, "
            f"which is {a + b}."
        )
    if topic == "subtraction":
        return (
            f"Picture {a} apples on a table. Take {b} of them away and count what's "
            f"left on the table -- that's {a} minus {b}, which is {a - b}."
        )
    if topic == "multiplication":
        return (
            f"Think of it as {a} groups with {b} things in each group. Instead of "
            f"counting them one at a time, {a} times {b} gives you the total right "
            f"away: {a * b}."
        )
    if topic == "fractions":
        num, denom = a, b  # first number is the numerator, matching the illustration's a/b caption
        if denom == 0:
            return None
        return (
            f"Imagine cutting something into {denom} equal pieces. If you're talking "
            f"about {num} of those pieces, that's the fraction {num}/{denom}."
        )
    return None
