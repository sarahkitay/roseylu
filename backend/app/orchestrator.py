"""Shared chat-turn orchestration.

Used by the FastAPI route (main.py), the CLI (cli/chat.py), and offline eval
tooling (training/scripts/simulate_student_eval.py) so the actual logic --
guardrail check, templated/curated/generative reply selection, output-side
check, online-learning scheduling -- exists in exactly one place. Extracted
after it started getting copy-pasted a third time; three call sites needing
the same ~40 lines is the point where duplication stops being simpler than
a shared function.
"""
from __future__ import annotations

from typing import Callable

from app.generation.base_model import DEFAULT_BACKEND
from app.guardrails.pipeline import DEFAULT_PIPELINE
from app.illustration import topic_classifier, topic_responses
from app.knowledge import curated_qa, quiz
from app.models.schemas import Action, ChatResponse, ChildProfile
from app.persona.persona_engine import build_system_prompt
from app.response.redirect_engine import build_generation_safety_fallback, build_redirect
from app.review_queue import DEFAULT_QUEUE


def _resolve_quiz(child: ChildProfile, message: str) -> tuple[str | None, str | None, list[int]]:
    """Returns (reply, topic, topic_numbers) if this turn is quiz-related
    (either an answer to an active quiz, or a request to start one), or
    (None, None, []) if it isn't -- callers fall through to the normal
    topic/templated/curated/generative decision in that case.
    """
    if quiz.has_active_session(child.child_id):
        return quiz.handle_answer(child.child_id, message), quiz.active_topic(child.child_id), []

    quiz_topic = quiz.detect_quiz_request(message)
    if quiz_topic is not None:
        return quiz.start_quiz(child.child_id, quiz_topic), quiz_topic, []

    return None, None, []


def handle_chat_turn(
    child: ChildProfile,
    message: str,
    schedule_online_learning: Callable[..., None] | None = None,
) -> ChatResponse:
    """Runs one full chat turn end to end.

    `schedule_online_learning`, if given, is called as
    `schedule_online_learning(fn, *args)` instead of calling `fn(*args)`
    directly -- FastAPI passes `background_tasks.add_task` so a training
    burst never adds latency to the HTTP response. Callers with no such
    mechanism (CLI, eval scripts) can leave this as None to log
    synchronously.
    """
    result = DEFAULT_PIPELINE.evaluate(message)

    if result.action == Action.ALLOW:
        # Quiz/game state takes priority over everything else on the ALLOW
        # path: if a quiz is already active for this child, their message
        # IS the answer to the current question, not a new question of its
        # own -- classifying it as a topic or feeding it to the model would
        # be wrong. Starting a new quiz is checked the same way, before the
        # normal topic/templated/curated/generative decision, since "quiz me
        # about columbus" should start a quiz, not trigger the Columbus
        # curated answer as if it were a factual question.
        reply, topic, topic_numbers = _resolve_quiz(child, message)

        if reply is None:
            # Topic is classified from the CHILD's question, before
            # generation, for two reasons: the illustration needs it
            # regardless, and for the handful of numeric math topics it lets
            # us skip the generative model entirely in favor of a
            # deterministic, correct answer -- see
            # app/illustration/topic_responses.py for why that's the right
            # tradeoff specifically for arithmetic (not a general fix for
            # model quality). "What is the child asking about" is also a far
            # more robust signal than classifying the model's own reply,
            # which is often unreliable at this model's scale (see
            # training/README.md).
            topic = topic_classifier.classify(message)
            topic_numbers = topic_classifier.extract_numbers(message, topic) if topic else []

            templated = (
                topic_responses.build_templated_answer(topic, topic_numbers, message, tier=child.age_tier)
                if topic else None
            )
            curated = curated_qa.find_answer(message, tier=child.age_tier) if templated is None else None
            if templated is not None:
                reply = templated
                if topic_responses.needs_illustration_suppressed(topic, message, tier=child.age_tier):
                    # e.g. multi-digit addition -- the penny-counting scene
                    # would try to render one coin per unit (62 coins for
                    # "24 + 38"), which is worse than no illustration.
                    topic = None
                    topic_numbers = []
            elif curated is not None:
                # Curated History/English/Math answers (app/knowledge/curated_qa.py)
                # -- same reasoning as the math templates above, extended
                # past pure arithmetic: the local model has no reliable
                # general knowledge at this training scale (see
                # training/README.md), so well-known curriculum topics get a
                # hand-written, correct answer instead of a generated guess.
                reply = curated
            else:
                system_prompt = build_system_prompt(child)
                reply = DEFAULT_BACKEND.generate(system_prompt, message)

        # Output-side check: the guardrail pipeline above only evaluated the
        # CHILD's message. A small, not-instruction-tuned model can produce
        # an inappropriate-sounding fragment even on a completely benign
        # input (observed directly during dev testing -- see
        # training/README.md) -- there's no upstream signal to catch that,
        # so the generated reply gets checked too before it's shown. Runs
        # on templated/curated answers too, for uniformity -- cheap, and
        # there's no good reason to special-case a bypass of a safety check.
        output_check = DEFAULT_PIPELINE.evaluate(reply)
        response_action = result.action
        response_category = None
        if output_check.action != Action.ALLOW:
            reply = build_generation_safety_fallback(child.age_tier)
            # Report this honestly as a REDIRECT, not ALLOW -- the child's
            # message was fine, but what almost got shown to them wasn't,
            # and a caller (the dev UI's action badge, a future audit log)
            # should be able to tell the difference from a clean ALLOW turn.
            response_action = Action.REDIRECT
            response_category = output_check.top_category
            # No cartoon scene accompanies a safety fallback.
            topic = None
            topic_numbers = []

        # Online learning: only ALLOW-path exchanges feed the live model --
        # REDIRECT/ESCALATE replies (from either check above) come from
        # fixed templates, not the model, so there's nothing generative to
        # learn from there. Logging the fallback text (not the flagged raw
        # generation) when the output check fired, so a training burst never
        # reinforces the output that got caught.
        online_trainer = getattr(DEFAULT_BACKEND, "online_trainer", None)
        if online_trainer is not None:
            if schedule_online_learning is not None:
                schedule_online_learning(online_trainer.log_interaction, message, reply)
            else:
                online_trainer.log_interaction(message, reply)

        return ChatResponse(
            reply=reply,
            action=response_action,
            top_category=response_category,
            topic=topic,
            topic_numbers=topic_numbers,
        )

    # REDIRECT and ESCALATE both produce the same kind of reply to the child;
    # ESCALATE additionally logs to the review queue. See docs/SAFETY_MODEL.md.
    reply = build_redirect(result.top_category, child.age_tier)

    if result.action == Action.ESCALATE:
        top_score = result.score_for(result.top_category)
        matched_layers = next(
            (s.matched_layers for s in result.scores if s.category == result.top_category), []
        )
        DEFAULT_QUEUE.append(
            child_id=child.child_id,
            category=result.top_category,
            score=top_score,
            matched_layers=matched_layers,
        )

    return ChatResponse(reply=reply, action=result.action, top_category=result.top_category)
