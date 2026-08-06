from app.config import AgeTier
from app.knowledge.curated_qa import ALL_ENTRIES, find_answer


def test_columbus_question_gets_curated_answer():
    answer = find_answer("i need to understand how christopher columbus came and why")
    assert answer is not None
    assert "1492" in answer


def test_columbus_is_tiered_by_age():
    msg = "how did christopher columbus get to america"
    preschool = find_answer(msg, tier=AgeTier.PRESCHOOL)
    teen = find_answer(msg, tier=AgeTier.TEEN)
    assert preschool != teen
    assert len(preschool) < len(teen)  # preschool version is meaningfully simpler/shorter
    assert "1492" in teen


def test_columbus_omitted_tier_falls_back_to_default_answer():
    # callers that don't know about tiers (training/scripts/ tooling) must
    # still get a sensible answer, not a crash or None
    answer = find_answer("how did christopher columbus get to america")
    assert answer is not None
    assert "1492" in answer


def test_columbus_does_not_end_with_a_flat_hedge_conclusion():
    # regression coverage for the specific generic-AI phrasing flagged live:
    # "both parts of the story are true" as a flat, opinion-free conclusion.
    # The rewrite should end by asking the child something, not summarizing
    # for them.
    for tier in AgeTier:
        answer = find_answer("how did christopher columbus get to america", tier=tier)
        assert "both parts of the story are true" not in answer.lower()


# Regression coverage for a real gap found live: "why does mommy yell" isn't
# dangerous (no guardrail category fits) and isn't curriculum, so it fell
# through to the generative model, which returned completely unrelated
# square-root/synonym text -- not just low quality, but zero connection to
# an emotionally real question from a 4-year-old.
def test_why_parents_yell_gets_a_relevant_curated_answer():
    answer = find_answer("why does mommy yell")
    assert answer is not None
    assert "stressed" in answer.lower() or "big feelings" in answer.lower()


def test_why_parents_yell_is_tiered_by_age():
    msg = "why does daddy yell"
    preschool = find_answer(msg, tier=AgeTier.PRESCHOOL)
    teen = find_answer(msg, tier=AgeTier.TEEN)
    assert preschool != teen
    assert "stomp your feet" in preschool.lower()
    assert "school counselor" in teen.lower()


def test_why_parents_yell_points_toward_a_trusted_adult_at_every_tier():
    for tier in AgeTier:
        answer = find_answer("why do my parents yell", tier=tier)
        assert answer is not None
        assert "trust" in answer.lower()


def test_history_english_math_life_all_represented():
    subjects = {e.subject for e in ALL_ENTRIES}
    assert subjects == {"history", "english", "math", "life"}


def test_unmatched_message_returns_none():
    assert find_answer("i got in a fight with my best friend") is None


def test_every_entry_has_a_working_keyword():
    # each entry's own first keyword should route back to itself -- catches
    # copy-paste keyword collisions between entries (an earlier entry
    # accidentally shadowing a later one)
    for entry in ALL_ENTRIES:
        matched = find_answer(f"tell me about {entry.keywords[0]}")
        assert matched == entry.answer, f"keyword {entry.keywords[0]!r} did not route to {entry.topic_id}"


def test_no_duplicate_topic_ids():
    ids = [e.topic_id for e in ALL_ENTRIES]
    assert len(ids) == len(set(ids))


# Regression coverage for a real gap found live: "how to do calcuslus" (a
# typo, and different phrasing than any exact keyword) matched nothing,
# even though "calculus" is a curated topic -- keyword substring matching
# alone can't handle arbitrary typos. Fixed with a fuzzy fallback in
# find_answer(); these tests pin both the fix and the false-positive it has
# to avoid.
def test_typo_still_finds_the_curated_answer():
    answer = find_answer("how to do calcuslus")
    assert answer is not None
    assert "derivatives" in answer


def test_fuzzy_fallback_does_not_confuse_similar_but_different_words():
    # "friction" is NOT a curated topic and must never fuzzy-match "fraction"
    # (0.875 similarity -- close enough to be a real risk, deliberately
    # tuned to fall just under the 0.88 threshold). A false-positive fuzzy
    # match here would confidently hand back a wrong-subject answer.
    assert find_answer("what is friction") is None


# Regression coverage for a real gap found live: "which word is a synonym
# for gift: store, cat, present, ocean and what is a synonym" matched the
# generic ENGLISH "what is a synonym" keyword entry (a real substring of the
# message) and completely ignored the specific, answerable question asked
# first. A specific question should never lose to a generic definition just
# because both happen to share a substring.
def test_synonym_multiple_choice_answers_the_specific_question():
    answer = find_answer("which word is a synonym for gift: store, cat, present, ocean and what is a synonym")
    assert answer is not None
    assert "'present'" in answer
    assert "store" in answer and "cat" in answer and "ocean" in answer
    # the tacked-on question fragment must not be treated as an answer choice
    other_options = answer.split("(")[1].split(")")[0]
    assert "synonym" not in other_options


def test_synonym_question_without_options_gives_an_example():
    answer = find_answer("what is a synonym for happy")
    assert answer is not None
    assert "happy" in answer


def test_synonym_question_for_uncurated_word_falls_through():
    # "ubiquitous" isn't in the small curated synonym vocabulary -- must
    # fall through to the generic definition rather than returning nothing
    # or a wrong answer.
    answer = find_answer("what is a synonym for ubiquitous")
    assert answer is not None
    assert "almost the same thing" in answer  # the generic ENGLISH definition
