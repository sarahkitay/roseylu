#!/usr/bin/env python3
"""Continuously reviews live conversation with a "teacher" model (Anthropic
or OpenAI) and writes corrected/confirmed examples for the local model to
learn from -- the mechanism behind "Rosey teaches herself as she goes."

WHERE THIS SITS, EXACTLY: this process reviews conversation AFTER the child
already received a reply from the local model. It never sits in the
request path, never delays a reply, and never replaces what the child sees
in real time -- the child always talks to the locally-trained model, not to
this script or the teacher model behind it. What this loop produces
(confirmed-good and corrected examples in `teacher_reviewed.jsonl`) feeds
the EXISTING online-learning mechanism
(backend/app/generation/online_trainer.py) and the next full training run
(training/scripts/build_corpus.py) -- so live conversation continuously
improves what the local model learns, without ever making the live app
depend on a third-party API. See docs/ARCHITECTURE.md's "where the line
actually is" section.

Why a teacher model at all: the whole point of training a dedicated local
model (see docs/PRODUCT_VISION.md) is that it can speak fluently about hard,
"sticky" situations -- the way a thoughtful child development professional
would, not the way a generic AI assistant deflects. Getting there requires
examples written at that level. The teacher model here is prompted
specifically to write in that voice, correcting anything generic-sounding
or off, the same way a training example gets refined during editing -- not
because the app is calling out to it live.

Setup:
    pip install -r training/requirements-eval.txt
    # .env must have TEACHER_BACKEND (anthropic|openai), TEACHER_MODEL,
    # and the matching API key -- see .env.example

Usage:
    python3 training/scripts/teacher_review_loop.py --once
    python3 training/scripts/teacher_review_loop.py --watch --interval 30
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT / "backend"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(_ROOT / ".env")

_LIVE_LOG_PATH = _ROOT / "training" / "data" / "corpus" / "live_interactions.jsonl"
_REVIEWED_PATH = _ROOT / "training" / "data" / "corpus" / "teacher_reviewed.jsonl"
_CHECKPOINT_PATH = _ROOT / "training" / "runs" / "teacher_review_checkpoint.json"

_TEACHER_SYSTEM_PROMPT = """\
You are a senior child development specialist and clinical child psychologist \
reviewing a reply from Rosey, an AI companion for children ages 7-15, built \
specifically so it never sounds like a generic AI assistant and can speak \
fluently and warmly about hard or "sticky" situations the way a skilled child \
psychologist would -- validating feelings, explaining reasoning in \
age-appropriate terms, and gently guiding the child forward, never with \
generic-AI phrasing like "I'm sorry, but I can't help with that" or "As an AI...".

Given the child's message and Rosey's actual reply, evaluate it and, if it \
falls short, rewrite it in the voice described above. Respond with ONLY \
compact JSON in this exact shape:
{"appropriate": true/false, "quality": 1-5, "sounds_generic_ai": true/false, \
"corrected_reply": "<a better version in Rosey's voice, or null if the \
original is already good>", "reasoning": "<one short sentence>"}"""


_BACKEND_ALIASES = {
    "claude": "anthropic", "anthropic": "anthropic",
    "gpt": "openai", "chatgpt": "openai", "openai": "openai",
}


def normalize_backend(backend: str) -> str:
    """Accepts common natural names ("claude", "gpt") in addition to the
    canonical "anthropic"/"openai" -- found necessary immediately: the
    first real .env config used TEACHER_BACKEND=claude, which is a
    completely reasonable thing to write and previously failed every
    single review with "unknown teacher backend."
    """
    normalized = _BACKEND_ALIASES.get(backend.strip().lower())
    if normalized is None:
        raise SystemExit(
            f"Unrecognized teacher backend {backend!r}. Use one of: "
            f"{sorted(set(_BACKEND_ALIASES.values()))} (or the aliases "
            f"{sorted(_BACKEND_ALIASES)})."
        )
    return normalized


_TEACHER_MAX_TOKENS = 800  # found via a real truncated response at 500 -- the JSON
# wrapper plus a full corrected_reply sometimes needs more room than a short answer alone


def call_teacher(system: str, user: str, backend: str, model: str) -> str:
    if backend == "anthropic":
        import anthropic

        client = anthropic.Anthropic()
        response = client.messages.create(
            model=model, max_tokens=_TEACHER_MAX_TOKENS, system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in response.content if getattr(b, "type", None) == "text").strip()
    if backend == "openai":
        import openai

        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=model, max_tokens=_TEACHER_MAX_TOKENS,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return (response.choices[0].message.content or "").strip()
    raise ValueError(f"unknown teacher backend: {backend}")


def _strip_markdown_fence(raw: str) -> str:
    """Models frequently wrap JSON in ```json ... ``` even when told to
    respond with ONLY JSON -- found via 3 real responses that failed to
    parse for exactly this reason. Strips a leading/trailing fence if
    present; a no-op on already-bare JSON.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


def review_one(child_message: str, rosey_reply: str, backend: str, model: str) -> dict:
    user = f"Child's message: {child_message}\n\nRosey's reply: {rosey_reply}"
    raw = call_teacher(_TEACHER_SYSTEM_PROMPT, user, backend, model)
    try:
        return json.loads(_strip_markdown_fence(raw))
    except json.JSONDecodeError:
        return {"appropriate": None, "quality": None, "sounds_generic_ai": None,
                 "corrected_reply": None, "reasoning": f"unparseable teacher output: {raw[:150]!r}"}


def _load_checkpoint() -> int:
    if _CHECKPOINT_PATH.exists():
        return json.loads(_CHECKPOINT_PATH.read_text()).get("lines_processed", 0)
    return 0


def _save_checkpoint(lines_processed: int) -> None:
    _CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CHECKPOINT_PATH.write_text(json.dumps({"lines_processed": lines_processed}))


def process_new_interactions(backend: str, model: str) -> int:
    if not _LIVE_LOG_PATH.exists():
        return 0

    lines = _LIVE_LOG_PATH.read_text(encoding="utf-8").splitlines()
    checkpoint = _load_checkpoint()
    if checkpoint >= len(lines):
        return 0

    _REVIEWED_PATH.parent.mkdir(parents=True, exist_ok=True)
    reviewed_count = 0
    with open(_REVIEWED_PATH, "a", encoding="utf-8") as out:
        i = checkpoint
        while i < len(lines):
            line = lines[i]
            if not line.strip():
                i += 1
                continue

            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                # malformed log line can never succeed on retry -- skip it
                # permanently rather than blocking every future run on it.
                print(f"  skipping unparseable log line {i}")
                i += 1
                continue

            try:
                verdict = review_one(entry["child"], entry["rosey"], backend, model)
            except Exception as e:  # noqa: BLE001 -- likely transient (network/API) -- retry next run
                print(f"  review failed on line {i}, stopping here to retry next run: {e}")
                break

            out.write(json.dumps({
                "child": entry["child"],
                "original_reply": entry["rosey"],
                "reviewed_reply": verdict.get("corrected_reply") or entry["rosey"],
                "verdict": verdict,
                "ts": time.time(),
            }) + "\n")
            reviewed_count += 1
            i += 1

            tag = "OK" if verdict.get("appropriate") and not verdict.get("corrected_reply") else "CORRECTED"
            print(f"  [{tag}] quality={verdict.get('quality')} \"{entry['child'][:60]}\"")

    _save_checkpoint(i)
    return reviewed_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", default=None,
                         help="anthropic/openai (or aliases like claude/gpt) -- defaults to $TEACHER_BACKEND")
    parser.add_argument("--model", default=None, help="defaults to $TEACHER_MODEL")
    parser.add_argument("--watch", action="store_true", help="keep polling for new interactions instead of exiting")
    parser.add_argument("--interval", type=int, default=30, help="seconds between polls in --watch mode")
    parser.add_argument("--once", action="store_true", help="process everything new once, then exit (default)")
    args = parser.parse_args()

    backend = args.backend or os.environ.get("TEACHER_BACKEND")
    model = args.model or os.environ.get("TEACHER_MODEL")
    if not backend or not model:
        raise SystemExit(
            "No teacher configured. Set TEACHER_BACKEND (anthropic|openai) and TEACHER_MODEL "
            "in .env, or pass --backend/--model explicitly."
        )
    backend = normalize_backend(backend)  # fail fast on a bad config value,
    # BEFORE touching any log lines -- a config error is not a per-line
    # failure and shouldn't burn through the checkpoint one line at a time.

    print(f"teacher: {backend}/{model}")
    print(f"watching: {_LIVE_LOG_PATH}")
    print(f"writing reviews to: {_REVIEWED_PATH}\n")

    if args.watch:
        print(f"polling every {args.interval}s -- Ctrl+C to stop")
        try:
            while True:
                n = process_new_interactions(backend, model)
                if n:
                    print(f"reviewed {n} new interaction(s)")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nstopped")
    else:
        n = process_new_interactions(backend, model)
        print(f"\nreviewed {n} new interaction(s)")


if __name__ == "__main__":
    main()
