"""Simple, stateful mini-games for early literacy -- letter sounds and
rhyming -- for the youngest kids this app supports (see AgeTier.PRESCHOOL).

Distinct from quiz.py (fixed-content trivia quizzes like Columbus, testing
recall of something the child was just taught) and from curated_qa.py
(factual Q&A): this is about building a pre-reading skill through repeated,
game-shaped practice, not testing knowledge of a specific topic. Same
multiple-choice UX shape as quiz.py's proven pattern (letter-or-text answer
matching) -- reliable to grade over text chat, unlike an open-ended phonetic
answer ("say the B sound out loud"), which this text-only interface has no
way to check. This doesn't remove docs/BLOCKERS.md's existing, honest caveat
that a text chat interface has real limits for children who can't read or
type yet -- it's useful for kids at the edge of that limit (just starting to
recognize letters and sounds, usually with a grown-up alongside), not a
replacement for audio-based phonics instruction.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class LiteracyRound:
    prompt: str
    options: list[str]
    correct_index: int  # index into options


@dataclass
class LiteracySession:
    game_id: str
    rounds: list[LiteracyRound]
    index: int = 0
    correct_count: int = 0


SESSIONS: dict[str, LiteracySession] = {}

GAMES: dict[str, list[LiteracyRound]] = {
    "letters": [
        LiteracyRound("Which word starts with the same sound as B?", ["Ball", "Cat", "Sun"], 0),
        LiteracyRound("Which word starts with the same sound as M?", ["Dog", "Moon", "Fish"], 1),
        LiteracyRound("Which word starts with the same sound as S?", ["Sun", "Ball", "Cat"], 0),
        LiteracyRound("Which word starts with the same sound as T?", ["Bird", "Tiger", "Moon"], 1),
        LiteracyRound("Which word starts with the same sound as D?", ["Dog", "Cat", "Sun"], 0),
    ],
    "rhymes": [
        LiteracyRound("Which word rhymes with CAT?", ["Hat", "Dog", "Sun"], 0),
        LiteracyRound("Which word rhymes with SUN?", ["Moon", "Fun", "Cat"], 1),
        LiteracyRound("Which word rhymes with DOG?", ["Frog", "Sun", "Bird"], 0),
        LiteracyRound("Which word rhymes with BEE?", ["Cat", "Tree", "Moon"], 1),
        LiteracyRound("Which word rhymes with STAR?", ["Fish", "Car", "Dog"], 1),
    ],
}

_GAME_LABELS = {"letters": "letter sounds", "rhymes": "rhyming"}

_REQUEST_RE = re.compile(
    r"\b(letter|abc|alphabet)s?\b.*\bgame\b"
    r"|\bgame\b.*\b(letter|abc|alphabet)s?\b"
    r"|\brhym\w*\b.*\bgame\b"
    r"|\bgame\b.*\brhym\w*\b"
    r"|\bplay (a |the )?(letter|rhym\w*|abc|alphabet)",
    re.IGNORECASE,
)


def detect_literacy_game_request(message: str) -> str | None:
    """Returns "letters" or "rhymes" if the message is asking to play one of
    these games, else None. Deliberately narrow phrasing detection, same
    spirit as quiz.detect_quiz_request()."""
    text = message.lower()
    if not _REQUEST_RE.search(text):
        return None
    return "rhymes" if "rhym" in text else "letters"


def has_active_session(child_id: str) -> bool:
    return child_id in SESSIONS


def active_game(child_id: str) -> str | None:
    session = SESSIONS.get(child_id)
    return session.game_id if session else None


def _format_round(round_: LiteracyRound, round_number: int, total: int) -> str:
    labeled = "\n".join(f"{chr(65 + i)}. {opt}" for i, opt in enumerate(round_.options))
    return f"Round {round_number} of {total}: {round_.prompt}\n{labeled}"


def start_game(child_id: str, game_id: str) -> str:
    rounds = list(GAMES[game_id])
    SESSIONS[child_id] = LiteracySession(game_id=game_id, rounds=rounds)
    label = _GAME_LABELS[game_id]
    intro = f"Let's play a {label} game! I'll ask you a few, and you pick the answer.\n\n"
    return intro + _format_round(rounds[0], 1, len(rounds))


def _match_option(message: str, round_: LiteracyRound) -> bool:
    text = message.strip().lower()
    letter = chr(65 + round_.correct_index).lower()
    if text == letter or text.startswith(letter + ".") or text.startswith(letter + ")"):
        return True
    return round_.options[round_.correct_index].lower() in text


def _celebration(correct: int, total: int) -> str:
    if correct == total:
        return "Perfect! You got every single one right! 🌟"
    if correct >= total / 2:
        return "Great job! Want to play again sometime?"
    return "Nice try! These get easier the more you practice -- want to try again?"


def handle_answer(child_id: str, message: str) -> str:
    session = SESSIONS[child_id]
    current = session.rounds[session.index]
    got_it = _match_option(message, current)
    if got_it:
        session.correct_count += 1
    feedback = "That's right! 🎉" if got_it else f"Not quite -- it was {current.options[current.correct_index]}."

    session.index += 1
    if session.index >= len(session.rounds):
        total = len(session.rounds)
        correct = session.correct_count
        del SESSIONS[child_id]
        return f"{feedback}\n\nThat's the game! You got {correct} out of {total}. {_celebration(correct, total)}"

    next_round = session.rounds[session.index]
    return f"{feedback}\n\n{_format_round(next_round, session.index + 1, len(session.rounds))}"
