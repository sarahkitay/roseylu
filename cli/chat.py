#!/usr/bin/env python3
"""Interactive terminal chat against the guardrail pipeline -- no server, no
network required. This is the fastest way to actually talk to the system
while it's being built.

Usage:
    python3 cli/chat.py --age 8 --name Rosey
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.generation.base_model import DEFAULT_BACKEND  # noqa: E402
from app.guardrails.pipeline import DEFAULT_PIPELINE  # noqa: E402
from app.models.schemas import Action, ChildProfile  # noqa: E402
from app.persona.persona_engine import build_system_prompt  # noqa: E402
from app.response.redirect_engine import build_generation_safety_fallback, build_redirect  # noqa: E402
from app.review_queue import DEFAULT_QUEUE  # noqa: E402

_ACTION_LABEL = {
    Action.ALLOW: "\033[92mALLOW\033[0m",
    Action.REDIRECT: "\033[93mREDIRECT\033[0m",
    Action.ESCALATE: "\033[91mESCALATE\033[0m",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--age", type=int, default=9, help="child's age, 7-15")
    parser.add_argument("--name", type=str, default="Rosey", help="persona name")
    parser.add_argument("--child-id", type=str, default="cli-dev-child")
    parser.add_argument("--verbose", action="store_true", help="print pipeline scores")
    args = parser.parse_args()

    child = ChildProfile(child_id=args.child_id, age=args.age, persona_name=args.name)
    print(f"-- talking to {args.name}, age tier {child.age_tier.value} (age {args.age}) --")
    print("-- type 'quit' to exit --\n")

    while True:
        try:
            message = input("child> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not message:
            continue
        if message.lower() in {"quit", "exit"}:
            break

        result = DEFAULT_PIPELINE.evaluate(message)

        if args.verbose:
            print(f"  [{_ACTION_LABEL[result.action]}] top_category={result.top_category}")
            for s in result.scores:
                print(f"    {s.category.value}: {s.score:.2f} via {s.matched_layers}")

        if result.action == Action.ALLOW:
            system_prompt = build_system_prompt(child)
            reply = DEFAULT_BACKEND.generate(system_prompt, message)

            # output-side check -- see main.py's chat() for why this exists
            output_check = DEFAULT_PIPELINE.evaluate(reply)
            if output_check.action != Action.ALLOW:
                if args.verbose:
                    print(f"  [output flagged: {output_check.top_category}] replacing with fallback")
                reply = build_generation_safety_fallback(child.age_tier)

            online_trainer = getattr(DEFAULT_BACKEND, "online_trainer", None)
            if online_trainer is not None:
                online_trainer.log_interaction(message, reply)
        else:
            reply = build_redirect(result.top_category, child.age_tier)
            if result.action == Action.ESCALATE:
                top_score = result.score_for(result.top_category)
                matched_layers = next(
                    (s.matched_layers for s in result.scores if s.category == result.top_category),
                    [],
                )
                DEFAULT_QUEUE.append(
                    child_id=child.child_id,
                    category=result.top_category,
                    score=top_score,
                    matched_layers=matched_layers,
                )
                print("  [logged to review queue]")

        print(f"{args.name}> {reply}\n")


if __name__ == "__main__":
    main()
