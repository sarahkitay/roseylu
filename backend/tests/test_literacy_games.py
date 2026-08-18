from app.knowledge import literacy_games


def setup_function():
    literacy_games.SESSIONS.clear()


def test_detect_letters_game_request():
    assert literacy_games.detect_literacy_game_request("let's play a letter game") == "letters"
    assert literacy_games.detect_literacy_game_request("can we play the abc game") == "letters"


def test_detect_rhymes_game_request():
    assert literacy_games.detect_literacy_game_request("let's play a rhyming game") == "rhymes"
    assert literacy_games.detect_literacy_game_request("play the rhyme game") == "rhymes"


def test_detect_returns_none_for_unrelated_message():
    assert literacy_games.detect_literacy_game_request("how do i tie my shoe") is None
    assert literacy_games.detect_literacy_game_request("what is the sun") is None


def test_start_game_shows_first_round_with_lettered_options():
    reply = literacy_games.start_game("child-1", "letters")
    assert "Round 1 of 5" in reply
    assert "A." in reply and "B." in reply and "C." in reply
    assert literacy_games.has_active_session("child-1")
    assert literacy_games.active_game("child-1") == "letters"


def test_full_game_by_letter_answers_tracks_score_and_ends_session():
    literacy_games.start_game("child-2", "letters")
    correct_letters = [chr(65 + r.correct_index) for r in literacy_games.GAMES["letters"]]
    reply = ""
    for letter in correct_letters:
        reply = literacy_games.handle_answer("child-2", letter)
    assert "That's the game!" in reply
    assert "5 out of 5" in reply
    assert "Perfect" in reply
    assert not literacy_games.has_active_session("child-2")


def test_answering_by_full_text_also_works():
    literacy_games.start_game("child-3", "rhymes")
    first_round = literacy_games.GAMES["rhymes"][0]
    correct_text = first_round.options[first_round.correct_index]
    reply = literacy_games.handle_answer("child-3", correct_text)
    assert "That's right!" in reply


def test_wrong_answer_reveals_correct_option_and_continues():
    literacy_games.start_game("child-4", "letters")
    first_round = literacy_games.GAMES["letters"][0]
    wrong_letter = "A" if first_round.correct_index != 0 else "B"
    reply = literacy_games.handle_answer("child-4", wrong_letter)
    assert "Not quite" in reply
    assert first_round.options[first_round.correct_index] in reply
    assert "Round 2" in reply
    assert literacy_games.has_active_session("child-4")
