#!/usr/bin/env python3
"""Assembles training/data/corpus/combined.txt from four sources:

1. Public-domain children's literature (training/data/corpus/public_domain/)
   -- teaches general English fluency and narrative structure. Boilerplate
   already stripped (see git history / re-run the strip step if re-fetching).
2. Rendered redirect_engine templates, one per (category, tier), paired with
   a real example message from training/data/seed_dataset.jsonl -- teaches
   the model its own safety voice in "Child: / Rosey:" dialogue format.
3. Hand-authored synthetic dialogues (training/data/synthetic_dialogues.py)
   -- teaches general conversational Q&A in the same dialogue format.
4. Teacher-reviewed live conversation (training/data/corpus/teacher_reviewed.jsonl,
   written by training/scripts/teacher_review_loop.py) -- confirmed-good or
   corrected real exchanges, reviewed by an external "teacher" model
   specifically for whether Rosey sounds like a thoughtful child
   professional rather than a generic AI. Folding these into a FULL
   retraining run (not just online-learning bursts, which already sample
   this file live -- see online_trainer.py) means what's learned online
   doesn't evaporate the next time someone runs a fresh training pass.
   Empty/skipped gracefully if the file doesn't exist yet.

(2), (3), and (4) are duplicated several times relative to (1): the
literature corpus is far larger in raw characters, and without upweighting,
a char-level model trained on the combined corpus would mostly learn
Victorian prose and barely learn the "Child: / Rosey:" turn-taking
structure at all, which defeats the point of training persona/dialogue
behavior. This is a blunt fix (duplication, not a proper weighted sampler)
-- fine for a prototype-scale corpus, worth replacing with weighted
sampling in the training loop itself once the dialogue-format dataset is
bigger.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT / "backend"))
sys.path.insert(0, str(_ROOT / "training" / "data"))

from app.config import AgeTier  # noqa: E402
from app.models.schemas import RiskCategory  # noqa: E402
from app.response.redirect_engine import build_redirect  # noqa: E402
from synthetic_dialogues import DIALOGUES  # noqa: E402

_PUBLIC_DOMAIN_DIR = _ROOT / "training" / "data" / "corpus" / "public_domain"
_SEED_PATH = _ROOT / "training" / "data" / "seed_dataset.jsonl"
_TEACHER_REVIEWED_PATH = _ROOT / "training" / "data" / "corpus" / "teacher_reviewed.jsonl"
_OUT_PATH = _ROOT / "training" / "data" / "corpus" / "combined.txt"

_DIALOGUE_REPEATS = 8  # see module docstring


def load_literature() -> str:
    parts = []
    for f in sorted(_PUBLIC_DOMAIN_DIR.glob("*.txt")):
        parts.append(f.read_text(encoding="utf-8", errors="ignore"))
    return "\n\n".join(parts)


def example_message_per_category() -> dict[str, str]:
    with open(_SEED_PATH, "r", encoding="utf-8") as f:
        examples = [json.loads(line) for line in f if line.strip()]
    by_category: dict[str, str] = {}
    for ex in examples:
        cat = ex.get("expected_category")
        if cat and cat not in by_category:
            by_category[cat] = ex["message"]
    return by_category


def build_redirect_dialogue_block() -> str:
    cue_by_category = example_message_per_category()
    lines = []
    for category in RiskCategory:
        cue = cue_by_category.get(category.value)
        if cue is None:
            continue
        for tier in AgeTier:
            response = build_redirect(category, tier)
            lines.append(f"Child: {cue}\nRosey: {response}\n")
    return "\n".join(lines)


def build_synthetic_dialogue_block() -> str:
    return "\n".join(f"Child: {q}\nRosey: {a}\n" for q, a in DIALOGUES)


def build_teacher_reviewed_block() -> str:
    if not _TEACHER_REVIEWED_PATH.exists():
        return ""
    lines = []
    for line in _TEACHER_REVIEWED_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        reply = entry.get("reviewed_reply") or entry.get("original_reply")
        if entry.get("child") and reply:
            lines.append(f"Child: {entry['child']}\nRosey: {reply}\n")
    return "\n".join(lines)


def main() -> None:
    literature = load_literature()
    dialogue_block = build_synthetic_dialogue_block()
    redirect_block = build_redirect_dialogue_block()
    teacher_block = build_teacher_reviewed_block()

    dialogue_and_redirect = (
        dialogue_block + "\n" + redirect_block + "\n" + teacher_block + "\n"
    ) * _DIALOGUE_REPEATS

    combined = literature + "\n\n" + dialogue_and_redirect

    _OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _OUT_PATH.write_text(combined, encoding="utf-8")

    print(f"literature: {len(literature):,} chars")
    print(f"teacher-reviewed block: {len(teacher_block):,} chars"
          + (" (none yet -- run teacher_review_loop.py first)" if not teacher_block else ""))
    print(f"dialogue+redirect+teacher block (x{_DIALOGUE_REPEATS}): {len(dialogue_and_redirect):,} chars")
    print(f"combined corpus: {len(combined):,} chars -> {_OUT_PATH}")


if __name__ == "__main__":
    main()
