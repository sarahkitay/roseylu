"""Shared spend cap for the offline dev-tooling scripts that call real
external APIs -- training/scripts/simulate_student_eval.py and
training/scripts/teacher_review_loop.py. Never imported by backend/app/,
which never calls a third-party API at all (see docs/ARCHITECTURE.md) --
this exists precisely because those two scripts are the only places in the
repo where a bug or a bad flag could actually run up a real bill.

Pricing is approximate and hand-maintained. Anthropic prices below are
current published per-million-token rates as of this writing. OpenAI prices
are last-known public gpt-4o-mini/gpt-4o rates -- this skill has no live
OpenAI pricing source, so treat them as a starting point and verify against
OpenAI's own pricing page before trusting them for anything beyond "stop an
obvious runaway." Update _PRICING_PER_MILLION_TOKENS if either drifts.
"""
from __future__ import annotations

import os

# USD per 1,000,000 tokens, as (input, output).
_PRICING_PER_MILLION_TOKENS: dict[str, tuple[float, float]] = {
    # Anthropic (see claude-api skill's cached pricing table). Includes the
    # prior generation (4.x), not just the newest models -- found necessary
    # live: this repo's real .env configures TEACHER_MODEL=claude-sonnet-4-6,
    # which isn't the newest model, and an unpriced model fails loudly by
    # design (see record_usage) rather than tracking it as free -- so it
    # needs a real entry, not just a "use the latest model" assumption.
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-4-7": (5.00, 25.00),
    "claude-opus-4-6": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-sonnet-4-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    # OpenAI -- approximate, last known public rates; verify before relying on this
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
}

DEFAULT_BUDGET_USD = 2.00


class BudgetExceeded(RuntimeError):
    """Raised when the cumulative spend cap has been reached -- callers should
    catch this, report what ran before stopping, and exit cleanly rather than
    crash."""


class CostTracker:
    """Tracks cumulative estimated USD spend across a script run and refuses
    to start another API call once a budget cap is reached.

    Enforcement is a pre-request gate, not a mid-request kill switch: a call
    already in flight when the cap is reached is allowed to finish (there's
    no way to abort an in-flight completion anyway), so total spend can
    exceed the cap by at most one request's worth. This is the same
    trade-off Anthropic's own Managed Agents session budgets make -- "the
    request that crosses the cap completes" -- not a corner cut here.
    """

    def __init__(self, budget_usd: float | None = None):
        self.budget_usd = (
            budget_usd if budget_usd is not None
            else float(os.environ.get("DEV_TOOLING_API_BUDGET_USD", DEFAULT_BUDGET_USD))
        )
        self.spent_usd = 0.0

    def _price_per_token(self, model: str) -> tuple[float, float]:
        for key, (in_per_m, out_per_m) in _PRICING_PER_MILLION_TOKENS.items():
            if key in model:  # tolerates dated/suffixed model strings
                return in_per_m / 1_000_000, out_per_m / 1_000_000
        raise ValueError(
            f"No pricing entry for model {model!r} -- add it to "
            "_PRICING_PER_MILLION_TOKENS in cost_tracker.py before using it, "
            "so spend on it is never silently untracked."
        )

    def check_budget_or_raise(self) -> None:
        """Call before every API request."""
        if self.spent_usd >= self.budget_usd:
            raise BudgetExceeded(
                f"Stopping: ${self.spent_usd:.4f} spent of ${self.budget_usd:.2f} budget. "
                "Pass --budget (or set $DEV_TOOLING_API_BUDGET_USD) to raise the cap."
            )

    def record_usage(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Call after every API response with its actual token usage.
        Returns this call's cost in USD."""
        in_price, out_price = self._price_per_token(model)
        cost = input_tokens * in_price + output_tokens * out_price
        self.spent_usd += cost
        return cost

    def remaining_usd(self) -> float:
        return max(0.0, self.budget_usd - self.spent_usd)
