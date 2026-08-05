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

from app.guardrails.pipeline import DEFAULT_PIPELINE  # noqa: E402
from app.models.schemas import Action, ChildProfile  # noqa: E402
from app.orchestrator import handle_chat_turn  # noqa: E402

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

        if args.verbose:
            # informational only -- handle_chat_turn re-evaluates internally;
            # cheap and side-effect-free, so computing it twice here just to
            # print scores isn't worth threading through the shared function.
            preview = DEFAULT_PIPELINE.evaluate(message)
            print(f"  [{_ACTION_LABEL[preview.action]}] top_category={preview.top_category}")
            for s in preview.scores:
                print(f"    {s.category.value}: {s.score:.2f} via {s.matched_layers}")

        response = handle_chat_turn(child, message)

        if args.verbose and response.action != Action.ALLOW:
            print(f"  [{response.action.value}] top_category={response.top_category}")
        if response.action == Action.ESCALATE:
            print("  [logged to review queue]")

        print(f"{args.name}> {response.reply}\n")


if __name__ == "__main__":
    main()
