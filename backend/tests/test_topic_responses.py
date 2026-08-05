from app.illustration.topic_responses import (
    build_templated_answer,
    needs_illustration_suppressed,
)


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


# Regression coverage for a real gap found live: "how do i do addition with
# 2 digit numbers" fell back to the single-digit penny template with
# default numbers (3, 2) -- content-wise a fine answer to "what is
# addition," but it doesn't address carrying/regrouping at all, which is
# what was actually asked. Multi-digit column addition is a distinct skill,
# not just a bigger version of single-digit counting.
def test_multidigit_addition_question_gets_carrying_explanation():
    answer = build_templated_answer("addition", [3, 2], "how do i do addition with 2 digit numbers")
    assert answer is not None
    assert "carry" in answer.lower()
    assert "24" in answer and "38" in answer  # the chosen default pair


def test_multidigit_addition_uses_real_numbers_from_the_message_when_present():
    answer = build_templated_answer("addition", [], "how do i add 45 and 27")
    assert answer is not None
    assert "45" in answer and "27" in answer
    assert "72" in answer  # 45 + 27


def test_multidigit_addition_without_carrying_still_uses_column_explanation():
    # 21 + 34: ones digits (1+4=5) don't require carrying -- the "carry"
    # word shouldn't be forced into an explanation where it doesn't apply
    answer = build_templated_answer("addition", [], "how do i add 21 and 34 with 2 digit numbers")
    assert answer is not None
    assert "carry" not in answer.lower()
    assert "55" in answer


def test_illustration_suppressed_for_multidigit_addition():
    assert needs_illustration_suppressed("addition", "how do i do addition with 2 digit numbers") is True
    assert needs_illustration_suppressed("addition", "what is 3 plus 5") is False
    assert needs_illustration_suppressed("subtraction", "how do i do addition with 2 digit numbers") is False
