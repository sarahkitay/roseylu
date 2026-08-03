from app.illustration.topic_responses import build_templated_answer


def test_addition_answer_is_numerically_correct():
    answer = build_templated_answer("addition", [3, 5])
    assert "3" in answer and "5" in answer and "8" in answer


def test_subtraction_answer_is_numerically_correct():
    answer = build_templated_answer("subtraction", [5, 2])
    assert "3" in answer


def test_multiplication_answer_is_numerically_correct():
    answer = build_templated_answer("multiplication", [3, 4])
    assert "12" in answer


def test_fractions_answer_preserves_numerator_denominator_order():
    answer = build_templated_answer("fractions", [1, 4])
    assert "1/4" in answer


def test_unknown_topic_returns_none():
    assert build_templated_answer("science", [3, 5]) is None
    assert build_templated_answer(None, []) is None


def test_wrong_number_count_returns_none():
    assert build_templated_answer("addition", [3]) is None
    assert build_templated_answer("addition", []) is None
