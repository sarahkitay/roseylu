from app.config import AgeTier
from app.models.schemas import ChildProfile
from app.persona.persona_engine import build_system_prompt


def _child(age: int) -> ChildProfile:
    return ChildProfile(child_id="t", age=age, persona_name="Rosey")


def test_preschool_prompt_names_the_breadth_of_real_preschooler_topics():
    # Guards against the generative fallback treating a 4-year-old's real
    # questions (feelings, family, play, literal how-to) as too simple to
    # engage with seriously -- see the PRESCHOOL entry's own comment in
    # persona_engine.py for why this was called out explicitly.
    prompt = build_system_prompt(_child(4))
    assert "feelings" in prompt.lower() or "family" in prompt.lower()
    assert "never" in prompt.lower()  # "never as too simple or silly"


def test_child_psychology_principle_present_at_every_age_tier():
    for age in (4, 8, 11, 14):
        prompt = build_system_prompt(_child(age))
        assert "trusted adult" in prompt.lower()
        assert "not a therapist" in prompt.lower()


def test_tone_still_varies_by_age_tier():
    preschool = build_system_prompt(_child(4))
    teen = build_system_prompt(_child(14))
    assert preschool != teen
    assert "capable reasoner" in teen.lower()
    assert "capable reasoner" not in preschool.lower()
