#!/usr/bin/env python3
"""Scores the guardrail PIPELINE (not a fine-tuned model -- none exists yet)
against training/data/seed_dataset.jsonl. This is the real calibration loop
for keyword lists and thresholds today; see training/README.md.

Usage:
    python3 training/scripts/evaluate.py [--data path/to/seed_dataset.jsonl]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "backend"))

from app.guardrails.pipeline import DEFAULT_PIPELINE  # noqa: E402
from app.models.schemas import Action  # noqa: E402

_DEFAULT_DATA = Path(__file__).parent.parent / "data" / "seed_dataset.jsonl"


def load_examples(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=_DEFAULT_DATA)
    args = parser.parse_args()

    examples = load_examples(args.data)
    total = len(examples)
    correct_action = 0
    tier1_missed: list[dict] = []
    mismatches: list[dict] = []

    per_category_total: dict[str, int] = {}
    per_category_correct: dict[str, int] = {}

    for ex in examples:
        result = DEFAULT_PIPELINE.evaluate(ex["message"])
        expected_action = Action(ex["expected_action"])
        expected_category = ex.get("expected_category")

        action_ok = result.action == expected_action
        # Treat REDIRECT/ESCALATE as interchangeable for category correctness
        # -- escalation is a severity flag on top of a redirect, not a
        # different classification (see docs/SAFETY_MODEL.md).
        category_ok = (
            (result.top_category is None and expected_category is None)
            or (result.top_category is not None and result.top_category.value == expected_category)
        )

        key = expected_category or "ALLOW"
        per_category_total[key] = per_category_total.get(key, 0) + 1
        if action_ok and category_ok:
            correct_action += 1
            per_category_correct[key] = per_category_correct.get(key, 0) + 1
        else:
            mismatches.append(
                {
                    "message": ex["message"],
                    "expected": (expected_action.value, expected_category),
                    "got": (result.action.value, result.top_category.value if result.top_category else None),
                }
            )
            if expected_category in {"SELF_HARM", "BODY_IMAGE", "GROOMING"} and result.action == Action.ALLOW:
                tier1_missed.append(ex)

    print(f"Overall: {correct_action}/{total} correct ({correct_action / total:.0%})\n")

    print("Per category:")
    for cat in sorted(per_category_total):
        c = per_category_correct.get(cat, 0)
        t = per_category_total[cat]
        flag = "  <-- REVIEW" if c < t else ""
        print(f"  {cat:20s} {c}/{t}{flag}")

    if mismatches:
        print("\nMismatches:")
        for m in mismatches:
            print(f"  \"{m['message']}\"")
            print(f"    expected={m['expected']} got={m['got']}")

    if tier1_missed:
        print(f"\n!! {len(tier1_missed)} TIER-1 CATEGORY FULLY MISSED (action=ALLOW) !!")
        print("   This is the failure mode that matters most -- fix the lexicon/thresholds")
        print("   before anything else. See docs/SAFETY_MODEL.md.")
        for ex in tier1_missed:
            print(f"   - \"{ex['message']}\"")
        sys.exit(1)


if __name__ == "__main__":
    main()
