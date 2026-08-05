from app.illustration.topic_classifier import classify, extract_numbers


def test_addition_detected():
    assert classify("how do i do addition for class") == "addition"
    assert classify("what is 3 plus 5") == "addition"


def test_subtraction_detected():
    assert classify("how do i do subtraction") == "subtraction"
    assert classify("if i have 5 apples and give away 2 how many are left") == "subtraction"


def test_fractions_detected():
    assert classify("can you explain fractions") == "fractions"


def test_science_detected():
    assert classify("tell me about the solar system") == "science"


def test_no_match_returns_none():
    assert classify("i got in a fight with my best friend") is None


# Regression coverage for a real bug found during live testing: "square"
# used to be a bare shapes keyword, which meant "what is a square root" and
# "what does squared mean" (both math, both answered by curated_qa.py, not
# shapes) misclassified as the "shapes" topic and rendered an irrelevant
# circle/square/triangle illustration next to the correct math answer.
def test_square_root_does_not_misclassify_as_shapes():
    assert classify("what is a square root") != "shapes"


def test_squared_does_not_misclassify_as_shapes():
    assert classify("what does squared mean") != "shapes"


def test_real_square_shape_question_still_classifies_as_shapes():
    assert classify("how many sides does a square have") == "shapes"
    assert classify("is a square the same as a rectangle") == "shapes"


def test_extract_numbers_from_message():
    assert extract_numbers("what is 3 plus 5", "addition") == [3, 5]


def test_extract_numbers_falls_back_to_topic_default():
    assert extract_numbers("how do i do addition for class", "addition") == [3, 2]


def test_extract_numbers_ignores_out_of_range_values():
    # 200 isn't renderable as individual countable objects -- falls back
    numbers = extract_numbers("what is 200 plus 5", "addition")
    assert numbers == [3, 2]
