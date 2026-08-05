import pytest

from app.knowledge import quiz


@pytest.fixture(autouse=True)
def clean_sessions():
    quiz.SESSIONS.clear()
    yield
    quiz.SESSIONS.clear()


def test_detect_quiz_request_requires_both_trigger_word_and_topic():
    assert quiz.detect_quiz_request("quiz me about columbus") == "columbus"
    assert quiz.detect_quiz_request("make a game about columbus") == "columbus"
    assert quiz.detect_quiz_request("what's your favorite game") is None  # no topic mentioned
    assert quiz.detect_quiz_request("tell me about columbus") is None  # no quiz/game word


def test_start_quiz_creates_a_session_and_returns_first_question():
    reply = quiz.start_quiz("child-1", "columbus")
    assert quiz.has_active_session("child-1")
    assert "Question 1 of 4" in reply


def test_full_quiz_playthrough_with_all_correct_answers():
    quiz.start_quiz("child-1", "columbus")
    session = quiz.SESSIONS["child-1"]
    correct_letters = [q.correct for q in session.questions]

    replies = []
    for letter in correct_letters:
        replies.append(quiz.handle_answer("child-1", letter))

    assert not quiz.has_active_session("child-1")  # session cleared after last question
    assert "4 out of 4" in replies[-1]
    assert "Perfect" in replies[-1]
    for reply in replies[:-1]:
        assert "Yes, that's it!" in reply


def test_wrong_answer_gives_correction_and_explanation():
    quiz.start_quiz("child-1", "columbus")
    session = quiz.SESSIONS["child-1"]
    wrong_letter = next(l for l in session.questions[0].options if l != session.questions[0].correct)

    reply = quiz.handle_answer("child-1", wrong_letter)
    assert "Not quite" in reply
    assert session.questions[0].explanation in reply
    assert quiz.has_active_session("child-1")  # quiz continues after a wrong answer


def test_answer_can_be_matched_by_option_text_not_just_letter():
    quiz.start_quiz("child-1", "columbus")
    session = quiz.SESSIONS["child-1"]
    correct_text = session.questions[0].options[session.questions[0].correct]

    reply = quiz.handle_answer("child-1", f"I think it's {correct_text}")
    assert "Yes, that's it!" in reply


def test_stopping_mid_quiz_clears_the_session():
    quiz.start_quiz("child-1", "columbus")
    reply = quiz.handle_answer("child-1", "stop")
    assert not quiz.has_active_session("child-1")
    assert "stop" in reply.lower() or "no problem" in reply.lower()


def test_handle_answer_with_no_active_session_is_graceful():
    reply = quiz.handle_answer("nobody-home", "A")
    assert "not in a quiz" in reply.lower()


def test_unmatchable_answer_asks_for_clarification_without_advancing():
    quiz.start_quiz("child-1", "columbus")
    reply = quiz.handle_answer("child-1", "purple elephants")
    assert quiz.SESSIONS["child-1"].index == 0  # did not advance
    assert quiz.has_active_session("child-1")
