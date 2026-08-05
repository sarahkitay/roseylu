#!/usr/bin/env python3
"""Simulates students across grade levels asking Rosey questions, using a
real external model (Anthropic and/or OpenAI) to play the student and,
optionally, to judge the replies -- then reports where Rosey's answers are
weak, so gaps like "the model has no idea who Christopher Columbus was" get
found systematically instead of one at a time by a human clicking around
the chat UI.

THIS SCRIPT ITSELF NEVER RUNS AS PART OF THE APP. It's an offline dev tool,
squarely in the same category as Claude hand-authoring
training/data/synthetic_dialogues.py or backend/app/knowledge/curated_qa.py
-- using a capable model to help build/evaluate this project, never a
runtime dependency of it. See docs/ARCHITECTURE.md's "no third-party API in
the live path" rule, which this script does not violate: it calls
orchestrator.handle_chat_turn() the same way the CLI and the API route do,
in-process, with no network call anywhere in that path -- the network calls
here are entirely on the *simulated student* and *judge* sides, wrapping
Rosey, not living inside it.

Setup:
    pip install -r training/requirements-eval.txt
    cp .env.example .env   # then fill in ANTHROPIC_API_KEY and/or OPENAI_API_KEY

Usage:
    python3 training/scripts/simulate_student_eval.py --provider anthropic --grades 2-9 --questions-per-grade 3
    python3 training/scripts/simulate_student_eval.py --provider openai --grades 4,7,10 --no-judge
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT / "backend"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(_ROOT / ".env")

from app.models.schemas import ChildProfile  # noqa: E402
from app.orchestrator import handle_chat_turn  # noqa: E402

# Rough US grade -> age mapping, clamped to this product's supported 7-15
# range (backend/app/models/schemas.py::ChildProfile). Approximate on
# purpose -- real classrooms have a spread of ages per grade; this just
# needs to be reasonable enough for the AI tier/tone the question gets
# evaluated against to make sense.
GRADE_TO_AGE = {1: 7, 2: 7, 3: 8, 4: 9, 5: 10, 6: 11, 7: 12, 8: 13, 9: 14, 10: 15}

_DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-5",
    "openai": "gpt-4o-mini",
}

_STUDENT_SYSTEM_PROMPT = """\
You are role-playing as a real {grade_ordinal}-grade student (around age {age}) \
using a homework-helper chatbot. Ask ONE short, natural question you might \
genuinely type -- casual phrasing, sometimes a little messy or run-on, not a \
polished textbook question. Pick any subject at random: math, reading/English, \
history, science, or something personal/social that's on your mind. Output ONLY \
the question itself, nothing else -- no quotes, no preamble."""

_JUDGE_SYSTEM_PROMPT = """\
You are evaluating an AI chatbot's reply to a {age}-year-old child's question, \
for quality only -- safety is handled separately and is not your concern here. \
Score the REPLY on three axes, each 1-5 (5 = best):
- coherence: is it grammatically sound, readable English (not garbled fragments)?
- correctness: is the factual/academic content accurate and relevant to the question?
- age_fit: is the tone and complexity reasonable for this age?
Respond with ONLY compact JSON: {{"coherence": N, "correctness": N, "age_fit": N, "note": "one short phrase"}}"""


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def call_anthropic(system: str, user: str, model: str, max_tokens: int = 300) -> str:
    import anthropic

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(block.text for block in response.content if getattr(block, "type", None) == "text").strip()


def call_openai(system: str, user: str, model: str, max_tokens: int = 300) -> str:
    import openai

    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return (response.choices[0].message.content or "").strip()


def call_llm(provider: str, system: str, user: str, model: str) -> str:
    if provider == "anthropic":
        return call_anthropic(system, user, model)
    if provider == "openai":
        return call_openai(system, user, model)
    raise ValueError(f"unknown provider: {provider}")


def generate_student_question(provider: str, model: str, grade: int, age: int) -> str:
    system = _STUDENT_SYSTEM_PROMPT.format(grade_ordinal=_ordinal(grade), age=age)
    return call_llm(provider, system, "Ask your question now.", model)


def judge_reply(provider: str, model: str, age: int, question: str, reply: str) -> dict:
    system = _JUDGE_SYSTEM_PROMPT.format(age=age)
    user = f"Child's question: {question}\n\nChatbot's reply: {reply}"
    raw = call_llm(provider, system, user, model)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"coherence": None, "correctness": None, "age_fit": None, "note": f"unparseable judge output: {raw[:100]!r}"}


def parse_grades(spec: str) -> list[int]:
    grades: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            grades.update(range(int(lo), int(hi) + 1))
        elif part:
            grades.add(int(part))
    return sorted(g for g in grades if g in GRADE_TO_AGE)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--provider", choices=["anthropic", "openai"], default=None,
                         help="defaults to whichever API key is set, preferring anthropic")
    parser.add_argument("--judge-provider", choices=["anthropic", "openai"], default=None,
                         help="defaults to the same provider as --provider")
    parser.add_argument("--model", default=None, help="override the student-simulator model")
    parser.add_argument("--judge-model", default=None, help="override the judge model")
    parser.add_argument("--grades", default="2-9", help='e.g. "2-9" or "3,5,8"')
    parser.add_argument("--questions-per-grade", type=int, default=3)
    parser.add_argument("--no-judge", action="store_true", help="skip LLM-judge scoring")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    provider = args.provider or ("anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "openai" if os.environ.get("OPENAI_API_KEY") else None)
    if provider is None:
        raise SystemExit(
            "No API key found. Copy .env.example to .env and set ANTHROPIC_API_KEY "
            "and/or OPENAI_API_KEY, or pass --provider explicitly."
        )
    model = args.model or _DEFAULT_MODELS[provider]
    judge_provider = args.judge_provider or provider
    judge_model = args.judge_model or _DEFAULT_MODELS[judge_provider]

    grades = parse_grades(args.grades)
    if not grades:
        raise SystemExit(f"no valid grades parsed from {args.grades!r} (supported: 1-10)")

    print(f"student simulator: {provider}/{model}")
    print(f"judge: {'disabled' if args.no_judge else f'{judge_provider}/{judge_model}'}")
    print(f"grades: {grades}, {args.questions_per_grade} question(s) each\n")

    results = []
    for grade in grades:
        age = GRADE_TO_AGE[grade]
        for i in range(args.questions_per_grade):
            try:
                question = generate_student_question(provider, model, grade, age)
            except Exception as e:  # noqa: BLE001 -- report and continue, don't kill the whole run
                print(f"[grade {grade}] question generation failed: {e}")
                continue

            child = ChildProfile(child_id=f"eval-grade{grade}-{i}", age=age, persona_name="Rosey")
            response = handle_chat_turn(child, question)

            entry = {
                "grade": grade,
                "age": age,
                "question": question,
                "action": response.action.value,
                "topic": response.topic,
                "reply": response.reply,
            }

            if not args.no_judge and response.action == "ALLOW":
                try:
                    entry["judge"] = judge_reply(judge_provider, judge_model, age, question, response.reply)
                except Exception as e:  # noqa: BLE001
                    entry["judge"] = {"error": str(e)}

            results.append(entry)
            judge_summary = ""
            if "judge" in entry and "coherence" in entry["judge"]:
                j = entry["judge"]
                judge_summary = f" | judge: coherence={j['coherence']} correctness={j['correctness']} age_fit={j['age_fit']}"
            print(f"[grade {grade}] {entry['action']:9s} topic={str(entry['topic']):14s} \"{question[:70]}\"{judge_summary}")

    out_path = args.out or (_ROOT / "training" / "runs" / "eval" / f"simulated_student_eval_{int(time.time())}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\n{len(results)} exchanges evaluated -> {out_path}")

    scored = [r for r in results if "judge" in r and isinstance(r["judge"].get("coherence"), (int, float))]
    if scored:
        avg_coherence = sum(r["judge"]["coherence"] for r in scored) / len(scored)
        avg_correctness = sum(r["judge"]["correctness"] for r in scored) / len(scored)
        avg_age_fit = sum(r["judge"]["age_fit"] for r in scored) / len(scored)
        print(f"averages over {len(scored)} judged replies: coherence={avg_coherence:.1f} "
              f"correctness={avg_correctness:.1f} age_fit={avg_age_fit:.1f}")

        low_scores = [r for r in scored if r["judge"]["coherence"] <= 2 or r["judge"]["correctness"] <= 2]
        if low_scores:
            print(f"\n{len(low_scores)} replies flagged as low-quality -- candidates for a new "
                  f"curated_qa.py entry or synthetic_dialogues.py example:")
            for r in low_scores:
                print(f"  grade {r['grade']}: \"{r['question']}\" -> {r['judge']}")


if __name__ == "__main__":
    main()
