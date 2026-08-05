from app.knowledge.curated_qa import ALL_ENTRIES, find_answer


def test_columbus_question_gets_curated_answer():
    answer = find_answer("i need to understand how christopher columbus came and why")
    assert answer is not None
    assert "1492" in answer


def test_history_english_math_all_represented():
    subjects = {e.subject for e in ALL_ENTRIES}
    assert subjects == {"history", "english", "math"}


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
