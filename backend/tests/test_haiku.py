from app.knowledge.haiku import build_haiku_answer, detect_haiku_topic


def test_detect_haiku_topic_from_common_phrasings():
    assert detect_haiku_topic("write a haiku about the sun") == "sun"
    assert detect_haiku_topic("make a haiku about the moon") == "moon"
    assert detect_haiku_topic("haiku about rain") == "rain"
    assert detect_haiku_topic("write me a haiku for the ocean") == "ocean"


def test_detect_haiku_topic_returns_none_for_non_haiku_requests():
    assert detect_haiku_topic("what is a haiku") is None
    assert detect_haiku_topic("how do i tie my shoe") is None


def test_build_haiku_answer_for_covered_topic_is_5_7_5():
    answer = build_haiku_answer("write a haiku about the sun")
    assert answer is not None
    lines = answer.split("\n\n")[1].split("\n")
    assert len(lines) == 3


def test_build_haiku_answer_returns_none_for_uncovered_topic():
    # "pizza" isn't in the small hand-authored set -- must fall through to
    # the normal curated/generative flow rather than fabricate something.
    assert build_haiku_answer("write a haiku about pizza") is None


def test_build_haiku_answer_returns_none_for_unresolvable_pronoun():
    # The "write a poem about it" case -- no topic to resolve at all.
    assert build_haiku_answer("write a haiku about it") is None
