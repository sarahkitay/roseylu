import pytest

from app.config import AgeTier
from app.models.schemas import RiskCategory
from app.response.redirect_engine import build_redirect


@pytest.mark.parametrize("category", [RiskCategory.SELF_HARM, RiskCategory.BODY_IMAGE, RiskCategory.GROOMING])
@pytest.mark.parametrize("tier", list(AgeTier))
def test_tier1_templates_exist_and_are_nonempty(category, tier):
    text = build_redirect(category, tier)
    assert isinstance(text, str)
    assert len(text) > 20


def test_self_harm_response_points_to_a_trusted_adult_and_crisis_line():
    for tier in AgeTier:
        text = build_redirect(RiskCategory.SELF_HARM, tier).lower()
        assert "988" in text
        assert "trust" in text  # "trusted adult" / "someone you trust" phrasing


def test_body_image_never_evaluates_appearance():
    # the response must redirect, not actually answer the appearance question
    for tier in AgeTier:
        text = build_redirect(RiskCategory.BODY_IMAGE, tier).lower()
        assert "you are" not in text  # should never assert a judgment about the child
        assert "beautiful" in text or "beauty" in text


def test_tier2_category_gets_generic_but_nonempty_redirect():
    text = build_redirect(RiskCategory.SUBSTANCE, AgeTier.MIDDLE)
    assert len(text) > 10
