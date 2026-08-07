from app.config import AgeTier
from app.knowledge.curated_qa import ALL_ENTRIES, find_answer, subject_for


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


def test_history_english_math_life_science_all_represented():
    subjects = {e.subject for e in ALL_ENTRIES}
    assert subjects == {"history", "english", "math", "life", "science"}


# Regression coverage for a real gap found live: "what is the sun" produced
# completely unrelated synonym/antonym text -- "sun" wasn't even a science
# keyword in topic_classifier.py, so it got no illustration either.
def test_what_is_the_sun_gets_a_relevant_curated_answer():
    answer = find_answer("what is the sun")
    assert answer is not None
    assert "star" in answer.lower()


def test_unmatched_message_returns_none():
    assert find_answer("i got in a fight with my best friend") is None


# Grief gets full 4-tier treatment (like why_do_parents_yell) because the
# right wording differs most sharply at the youngest tier -- child-grief
# guidance flags euphemisms like "went to sleep" as actively confusing/scary
# for young children, so PRESCHOOL is deliberately explicit that the pet
# isn't coming back, without being graphic about it.
def test_grief_is_tiered_and_preschool_avoids_sleep_euphemism():
    preschool = find_answer("my dog died", tier=AgeTier.PRESCHOOL)
    teen = find_answer("my dog died", tier=AgeTier.TEEN)
    assert preschool != teen
    assert "asleep" not in preschool.lower() and "went to sleep" not in preschool.lower()
    assert "stops working" in preschool.lower() or "stopped working" in preschool.lower()


def test_why_is_the_sky_blue_gets_a_relevant_curated_answer():
    # regression coverage: this exact question produced garbled text about
    # sunlight/atmosphere/scattering from the generative model before this
    # SCIENCE entry existed -- now it should get the real explanation.
    answer = find_answer("why is the sky blue")
    assert answer is not None
    assert "scatter" in answer.lower()


def test_what_is_the_moon_and_gravity_get_relevant_curated_answers():
    moon = find_answer("what is the moon")
    gravity = find_answer("what is gravity")
    assert moon is not None and "orbit" in moon.lower()
    assert gravity is not None and "pull" in gravity.lower()


# Regression coverage for a real bug caught by test_unmatched_message_returns_none
# while adding the LIFE category: the fuzzy fallback's "longest word in a
# keyword phrase is the one worth typo-tolerating" heuristic picked "friend"
# as the anchor for the friend_wont_play_with_me entry, which meant ANY
# message merely containing the word "friend" -- not just a typo of some
# distinctive word -- got hijacked into that canned answer via an exact
# (non-typo) match. Unrelated benign messages must not be swept into an
# emotionally-specific canned reply just because they share a common word.
def test_unrelated_message_containing_a_common_life_keyword_is_not_hijacked():
    assert find_answer("i got in a fight with my best friend") is None
    assert find_answer("i'm scared of spiders") is None
    assert find_answer("i made a mistake on my homework") is None


# Regression coverage for a real gap found live: "what is water made up of"
# produced garbled, unrelated text -- distinct from water_cycle (rain
# formation), so it needed its own entry.
def test_what_is_water_made_of_gets_a_relevant_curated_answer():
    answer = find_answer("what is water made up of")
    assert answer is not None
    assert "h2o" in answer.lower() or "hydrogen" in answer.lower()


# Regression coverage: reported live that "what is water made up of" gave
# age 11 and age 5 the exact same molecule/H2O explanation. All SCIENCE
# entries now have a PRESCHOOL variant.
def test_what_is_a_nucleus_gets_a_relevant_curated_answer():
    answer = find_answer("what is a nucleus")
    assert answer is not None
    assert "cell" in answer.lower()


def test_what_about_mitochondria_gets_a_relevant_curated_answer():
    answer = find_answer("what about mitochondria")
    assert answer is not None
    assert "energy" in answer.lower()


def test_remaining_cell_parts_get_relevant_curated_answers():
    membrane = find_answer("what is a cell membrane")
    wall = find_answer("what is a cell wall")
    cytoplasm = find_answer("what is cytoplasm")
    dna = find_answer("what is dna")
    assert membrane is not None and "barrier" in membrane.lower()
    assert wall is not None and "plant" in wall.lower()
    assert cytoplasm is not None and "jelly" in cytoplasm.lower()
    assert dna is not None and "chromosome" in dna.lower()


# Regression coverage for a real gap found live: "what are all the parts of
# a cell" got the narrow nucleus-only answer instead of a comprehensive
# overview, via a fuzzy "cells"~"cell" coincidence (0.89 similarity, not an
# actual typo) that a dedicated overview entry now pre-empts.
def test_cell_parts_overview_covers_all_parts_not_just_the_nucleus():
    answer = find_answer("what are all the parts of a cell")
    assert answer is not None
    for part in ["membrane", "cytoplasm", "nucleus", "mitochondria"]:
        assert part in answer.lower(), f"overview answer missing {part!r}"


def test_cell_wall_fuzzy_anchor_does_not_hijack_unrelated_animal_questions():
    # "do animal cells have a cell wall"'s longest word is "animal" -- a
    # common word (topic_classifier.py has a whole "animals" illustration
    # topic) that must not become a fuzzy-match anchor, or any unrelated
    # message mentioning an animal would get hijacked into this answer.
    assert find_answer("what is my favorite animal") is None


# Regression coverage for a real gap found live: "what makes someone pretty
# or not pretty" -- a general (and body-image-adjacent) question with no
# curated match of its own -- got hijacked into rhyme_poetry's answer via
# "makes" (the longest word in "what makes a poem a poem," an exact,
# non-typo match). Prompted a full audit of every fuzzy-eligible keyword's
# anchor in curated_qa.py; this and the following test pin a sample of what
# that audit found and fixed.
def test_makes_is_not_a_fuzzy_anchor_and_has_its_own_curated_answer():
    # Body image is one of this app's named priority topics -- a general
    # (non-self-negative) question in this space should get a thoughtful
    # curated answer, not fall to the generative model.
    answer = find_answer("what makes someone pretty or not pretty")
    assert answer is not None
    assert "opinion" in answer.lower()
    assert "rhyme" not in answer.lower()  # the original mis-hijacked answer


def test_common_words_found_in_anchor_audit_are_not_fuzzy_anchors():
    # Each of these would otherwise fuzzy-hijack via an unrelated entry's
    # keyword: "person" (moon_landing), "before" (spelling_rules), "world"
    # (world_war_2), "structure" (cell_parts_overview), "number"
    # (even_odd_numbers). Note: "what is the structure of the government"
    # is NOT one of these -- it legitimately fuzzy-matches
    # branches_of_government via the word "government" itself (that
    # entry's own anchor, not "structure," and a correct match).
    assert find_answer("who is your favorite person") is None
    assert find_answer("what happened before recess") is None
    assert find_answer("what is the biggest ocean in the world") is None
    assert find_answer("what is the structure of a sentence") is None
    assert find_answer("what number is my house") is None


def test_periodic_table_and_elements_get_relevant_curated_answers():
    atom = find_answer("what is an atom")
    element = find_answer("what is an element")
    table = find_answer("periodic table")
    assert atom is not None and "nucleus" in atom.lower()
    assert element is not None and "118" in element
    assert table is not None and "atomic number" in table.lower()


def test_science_entries_are_tiered_for_preschool():
    for message in [
        "what is water made up of", "what is the sun", "what is the moon",
        "what is gravity", "why is the sky blue", "how do plants grow",
        "water cycle", "what are the five senses", "what is a nucleus",
        "what about mitochondria", "what is a cell membrane", "what is a cell wall",
        "what is cytoplasm", "what is dna", "what is an atom", "what is an element",
        "periodic table", "what are all the parts of a cell",
    ]:
        preschool = find_answer(message, tier=AgeTier.PRESCHOOL)
        default = find_answer(message, tier=AgeTier.MIDDLE)
        assert preschool is not None and default is not None
        assert preschool != default, f"{message!r} not tiered for PRESCHOOL"


def test_subject_for_matches_the_entry_that_would_actually_answer():
    assert subject_for("how did christopher columbus get to america") == "history"
    assert subject_for("why does mommy yell") == "life"
    assert subject_for("what is the sun") == "science"
    assert subject_for("what is a decimal") == "math"


def test_subject_for_reports_english_for_the_synonym_special_case():
    # _answer_synonym_question is checked before the keyword table in
    # find_answer() -- subject_for() must agree it's "english", not fall
    # through to None just because it's handled by a different code path.
    assert subject_for("which word is a synonym for gift: store, cat, present, ocean") == "english"


def test_subject_for_returns_none_for_unmatched_message():
    assert subject_for("i got in a fight with my best friend") is None


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
