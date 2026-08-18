"""Tests for training/scripts/cost_tracker.py -- the shared $2 (by default)
spend cap for the offline dev-tooling scripts that call real external APIs.
Lives under backend/tests/ with the training/scripts path added manually,
same pattern as test_teacher_review_loop.py.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "training" / "scripts"))
from cost_tracker import BudgetExceeded, CostTracker, DEFAULT_BUDGET_USD  # noqa: E402


def test_default_budget_is_two_dollars():
    tracker = CostTracker()
    assert tracker.budget_usd == DEFAULT_BUDGET_USD == 2.00


def test_explicit_budget_overrides_default():
    tracker = CostTracker(budget_usd=0.50)
    assert tracker.budget_usd == 0.50


def test_env_var_overrides_default_when_no_explicit_budget(monkeypatch):
    monkeypatch.setenv("DEV_TOOLING_API_BUDGET_USD", "5.00")
    tracker = CostTracker()
    assert tracker.budget_usd == 5.00


def test_explicit_budget_wins_over_env_var(monkeypatch):
    monkeypatch.setenv("DEV_TOOLING_API_BUDGET_USD", "5.00")
    tracker = CostTracker(budget_usd=1.00)
    assert tracker.budget_usd == 1.00


def test_record_usage_computes_cost_from_pricing_table():
    tracker = CostTracker(budget_usd=1000.0)
    # claude-opus-5: $5/$25 per 1M tokens -> 100K in + 10K out
    cost = tracker.record_usage("claude-opus-5", 100_000, 10_000)
    expected = 100_000 * (5.00 / 1_000_000) + 10_000 * (25.00 / 1_000_000)
    assert cost == pytest.approx(expected)
    assert tracker.spent_usd == pytest.approx(expected)


def test_record_usage_accumulates_across_calls():
    tracker = CostTracker(budget_usd=1000.0)
    tracker.record_usage("claude-haiku-4-5", 1_000_000, 0)  # $1.00
    tracker.record_usage("claude-haiku-4-5", 0, 1_000_000)  # $5.00
    assert tracker.spent_usd == pytest.approx(6.00)


def test_matches_dated_or_suffixed_model_strings():
    # Real model IDs are sometimes dated/suffixed (e.g. a specific snapshot)
    # -- pricing lookup must still find the base model by substring, not
    # require an exact match.
    tracker = CostTracker(budget_usd=1000.0)
    cost = tracker.record_usage("claude-haiku-4-5-20251001", 1_000_000, 0)
    assert cost == pytest.approx(1.00)


def test_unknown_model_raises_instead_of_silently_undercounting():
    # A model with no pricing entry must fail loudly, not be tracked as
    # free -- silently undercounting spend defeats the entire point of a
    # budget cap.
    tracker = CostTracker(budget_usd=1000.0)
    with pytest.raises(ValueError):
        tracker.record_usage("some-future-model-nobody-added-yet", 100, 100)


def test_check_budget_or_raise_passes_when_under_budget():
    tracker = CostTracker(budget_usd=2.00)
    tracker.record_usage("claude-haiku-4-5", 100_000, 0)  # $0.10
    tracker.check_budget_or_raise()  # should not raise


def test_check_budget_or_raise_stops_once_cap_is_reached():
    tracker = CostTracker(budget_usd=1.00)
    tracker.record_usage("claude-haiku-4-5", 1_000_000, 0)  # exactly $1.00
    with pytest.raises(BudgetExceeded):
        tracker.check_budget_or_raise()


def test_a_request_already_in_flight_is_allowed_to_complete():
    # Enforcement is pre-request, not mid-request: check_budget_or_raise()
    # only blocks the NEXT call from starting. A call that pushes spend past
    # the cap is allowed to record its actual usage afterward -- the same
    # "the request that crosses the cap completes" trade-off Anthropic's own
    # Managed Agents session budgets document.
    tracker = CostTracker(budget_usd=1.00)
    tracker.check_budget_or_raise()  # fine, nothing spent yet
    tracker.record_usage("claude-opus-5", 1_000_000, 1_000_000)  # pushes well past $1.00
    assert tracker.spent_usd > tracker.budget_usd
    with pytest.raises(BudgetExceeded):
        tracker.check_budget_or_raise()  # the NEXT call is refused


def test_remaining_usd_never_goes_negative():
    tracker = CostTracker(budget_usd=1.00)
    tracker.record_usage("claude-opus-5", 1_000_000, 1_000_000)  # way over budget
    assert tracker.remaining_usd() == 0.0
