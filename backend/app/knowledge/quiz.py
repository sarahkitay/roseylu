"""A real, stateful multiple-choice quiz/game generator for curated topics.

Same philosophy as everything else in app/knowledge/ and
app/illustration/topic_responses.py: curated, deterministic content instead
of asking the small local model to invent quiz questions on the fly, which
would produce the same unreliable, occasionally-wrong output documented
throughout training/README.md. A wrong quiz question is arguably worse than
a wrong homework answer -- it actively teaches something false and asks the
child to treat it as correct.

State is a plain in-memory dict keyed by child_id, one active quiz per
child at a time -- the same "single-developer dev instance" scope already
documented for backend/app/generation/online_trainer.py, not a real
multi-user session store. See docs/BLOCKERS.md if this needs to become
durable/per-real-child later.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class QuizQuestion:
    question: str
    options: dict[str, str]  # letter -> option text, e.g. {"A": "1492", "B": "1592"}
    correct: str  # the correct letter
    explanation: str  # shown after answering, right or wrong -- this is where the actual teaching happens


@dataclass
class QuizSession:
    topic: str
    questions: list[QuizQuestion]
    index: int = 0
    correct_count: int = 0


SESSIONS: dict[str, QuizSession] = {}

QUIZZES: dict[str, list[QuizQuestion]] = {
    "columbus": [
        QuizQuestion(
            "What year did Columbus first land in the Caribbean?",
            {"A": "1592", "B": "1492", "C": "1776", "D": "1400"},
            "B",
            "1492 -- there's an old rhyme a lot of people learn: 'in fourteen hundred "
            "ninety-two, Columbus sailed the ocean blue.'",
        ),
        QuizQuestion(
            "Which of these was NOT one of Columbus's three ships?",
            {"A": "Nina", "B": "Pinta", "C": "Mayflower", "D": "Santa Maria"},
            "C",
            "The Mayflower is a totally different ship -- it carried the Pilgrims to "
            "America almost 130 years later, in 1620!",
        ),
        QuizQuestion(
            "Why did Columbus call the people he met 'Indians'?",
            {"A": "He thought he'd reached India", "B": "That's what they called themselves",
             "C": "The King of Spain told him to", "D": "He made up the word for fun"},
            "A",
            "He was completely convinced he'd sailed all the way to the Indies, near "
            "India -- even though he'd actually found a whole continent Europeans didn't "
            "know existed.",
        ),
        QuizQuestion(
            "What did Columbus name the island where he first landed?",
            {"A": "America", "B": "New Spain", "C": "San Salvador", "D": "Atlantis"},
            "C",
            "San Salvador means 'Holy Savior' in Spanish -- Columbus named it that and "
            "claimed it for Spain.",
        ),
    ],
}

_QUIZ_TOPIC_ALIASES: dict[str, list[str]] = {
    "columbus": ["columbus"],
}

_QUIZ_TRIGGER_WORDS = ["quiz", "game", "test me", "gameify", "make this fun", "quiz me"]

_STOP_WORDS = ["stop", "quit", "never mind", "nevermind", "cancel", "no thanks"]


def detect_quiz_request(message: str) -> str | None:
    """Returns a topic id if this message is asking to start a quiz/game
    about a topic that has one, else None. Deliberately looser matching
    than topic_classifier.py's illustration triggers -- "quiz me on
    columbus" doesn't need to match the same precise phrasing tuned for
    illustration relevance, just mention the topic by name alongside a
    quiz/game word.
    """
    text = message.lower()
    if not any(w in text for w in _QUIZ_TRIGGER_WORDS):
        return None
    for topic, aliases in _QUIZ_TOPIC_ALIASES.items():
        if topic in QUIZZES and any(a in text for a in aliases):
            return topic
    return None


def has_active_session(child_id: str) -> bool:
    return child_id in SESSIONS


def active_topic(child_id: str) -> str | None:
    session = SESSIONS.get(child_id)
    return session.topic if session else None


def _format_question(q: QuizQuestion, num: int, total: int) -> str:
    options_text = "\n".join(f"{letter}) {text}" for letter, text in q.options.items())
    return f"Question {num} of {total}: {q.question}\n{options_text}"


def _match_option(text: str, question: QuizQuestion) -> str | None:
    text = text.lower()
    for letter in question.options:
        if re.search(rf"\b{letter.lower()}\b", text):
            return letter
    for letter, option_text in question.options.items():
        if option_text.lower() in text:
            return letter
    return None


def _celebration(score: int, total: int) -> str:
    ratio = score / total if total else 0
    if ratio == 1:
        return "Perfect score! You really know this one. 🌟"
    if ratio >= 0.5:
        return "Nice work! Want to try another quiz sometime?"
    return "Good try! Want to go over any of these again, or ask me more about it?"


def start_quiz(child_id: str, topic: str) -> str:
    questions = QUIZZES[topic]
    SESSIONS[child_id] = QuizSession(topic=topic, questions=questions)
    intro = f"Let's play a quiz about {topic.title()}! \n\n"
    return intro + _format_question(questions[0], 1, len(questions))


def handle_answer(child_id: str, message: str) -> str:
    session = SESSIONS.get(child_id)
    if session is None:
        return "We're not in a quiz right now -- want to start one? Just ask for a quiz about a topic."

    text = message.strip().lower()
    if any(w in text for w in _STOP_WORDS):
        del SESSIONS[child_id]
        return "No problem, we can stop here! Want to talk about something else?"

    question = session.questions[session.index]
    chosen = _match_option(text, question)
    if chosen is None:
        example_letter = next(iter(question.options))
        return (
            f"Hmm, I'm not sure which answer that is -- try saying the letter, like "
            f"\"{example_letter}\", or the answer itself."
        )

    is_correct = chosen == question.correct
    if is_correct:
        session.correct_count += 1
    feedback = "Yes, that's it! " if is_correct else f"Not quite -- it's {question.correct}) {question.options[question.correct]}. "
    feedback += question.explanation

    session.index += 1
    if session.index >= len(session.questions):
        total = len(session.questions)
        score = session.correct_count
        del SESSIONS[child_id]
        return f"{feedback}\n\nThat's the quiz! You got {score} out of {total}. {_celebration(score, total)}"

    next_q = session.questions[session.index]
    return f"{feedback}\n\n{_format_question(next_q, session.index + 1, len(session.questions))}"
