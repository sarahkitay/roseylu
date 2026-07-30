#!/usr/bin/env python3
"""Converts training/data/seed_dataset.jsonl into chat-format SFT data at
training/data/prepared/sft.jsonl.

For REDIRECT/ESCALATE examples, the target response is generated from
`redirect_engine` (age tier defaults to MIDDLE for the seed set -- a real
dataset should include the age tier per example and generate per-tier
targets). For ALLOW examples, there's no scripted target -- see the note in
main() about why those rows are written with a null response rather than a
fabricated one.

Runs today with no external dependencies beyond what's in
backend/requirements.txt. The dataset it *produces* inherits every caveat
already on seed_dataset.jsonl -- see training/README.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "backend"))

from app.config import AgeTier  # noqa: E402
from app.models.schemas import Action, RiskCategory  # noqa: E402
from app.persona.persona_engine import build_system_prompt  # noqa: E402
from app.models.schemas import ChildProfile  # noqa: E402
from app.response.redirect_engine import build_redirect  # noqa: E402

_DEFAULT_IN = Path(__file__).parent.parent / "data" / "seed_dataset.jsonl"
_DEFAULT_OUT = Path(__file__).parent.parent / "data" / "prepared" / "sft.jsonl"

_REFERENCE_CHILD = ChildProfile(child_id="dataset-prep", age=11, persona_name="Rosey")


def build_row(example: dict) -> dict | None:
    action = Action(example["expected_action"])
    system_prompt = build_system_prompt(_REFERENCE_CHILD)

    if action == Action.ALLOW:
        # No scripted "correct" answer exists for open-ended ALLOW prompts
        # (homework help, creative writing, etc.) -- fabricating one would
        # just be training the model to imitate an engineer's guess at a
        # good answer, which isn't the point of this dataset. These rows are
        # for *classification* balance in the pipeline evaluator, not for
        # generation fine-tuning. Skip them here; a real SFT set needs
        # separately-sourced, reviewed golden responses for the ALLOW class.
        return None

    category = RiskCategory[example["expected_category"]]
    response = build_redirect(category, AgeTier.MIDDLE)

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example["message"]},
            {"role": "assistant", "content": response},
        ],
        "meta": {
            "category": category.value,
            "action": action.value,
            "status": example.get("status", "UNKNOWN"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-path", type=Path, default=_DEFAULT_IN)
    parser.add_argument("--out-path", type=Path, default=_DEFAULT_OUT)
    args = parser.parse_args()

    with open(args.in_path, "r", encoding="utf-8") as f:
        examples = [json.loads(line) for line in f if line.strip()]

    rows = [build_row(ex) for ex in examples]
    rows = [r for r in rows if r is not None]

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    skipped = len(examples) - len(rows)
    print(f"Wrote {len(rows)} SFT rows to {args.out_path}")
    print(f"Skipped {skipped} ALLOW-class rows (no scripted target -- see docstring)")


if __name__ == "__main__":
    main()
