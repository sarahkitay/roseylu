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


def test_extract_numbers_from_message():
    assert extract_numbers("what is 3 plus 5", "addition") == [3, 5]


def test_extract_numbers_falls_back_to_topic_default():
    assert extract_numbers("how do i do addition for class", "addition") == [3, 2]


def test_extract_numbers_ignores_out_of_range_values():
    # 200 isn't renderable as individual countable objects -- falls back
    numbers = extract_numbers("what is 200 plus 5", "addition")
    assert numbers == [3, 2]
