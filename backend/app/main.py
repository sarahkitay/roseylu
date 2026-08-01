"""FastAPI wrapper around the pipeline. Thin on purpose -- all the actual
logic lives in guardrails/, response/, persona/, and generation/. This file
just wires request -> pipeline -> (model | redirect) -> response, plus the
dev chat UI and the online-learning status endpoint.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import BackgroundTasks, FastAPI
from fastapi.responses import HTMLResponse

from app.generation.base_model import DEFAULT_BACKEND
from app.guardrails.pipeline import DEFAULT_PIPELINE
from app.illustration import topic_classifier
from app.models.schemas import Action, ChatRequest, ChatResponse
from app.persona.persona_engine import build_system_prompt
from app.response.redirect_engine import build_generation_safety_fallback, build_redirect
from app.review_queue import DEFAULT_QUEUE

app = FastAPI(
    title="SafeAI for Kids -- prototype",
    description=(
        "Prototype API. NOT reviewed for production use with real children -- "
        "see docs/BLOCKERS.md before treating this as anything but a dev harness."
    ),
)

_STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (_STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/training-status")
def training_status() -> dict:
    online_trainer = getattr(DEFAULT_BACKEND, "online_trainer", None)
    if online_trainer is None:
        return {"online_learning_active": False}
    return {"online_learning_active": True, **online_trainer.status()}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, background_tasks: BackgroundTasks) -> ChatResponse:
    result = DEFAULT_PIPELINE.evaluate(req.message)

    if result.action == Action.ALLOW:
        system_prompt = build_system_prompt(req.child)
        reply = DEFAULT_BACKEND.generate(system_prompt, req.message)

        # Output-side check: the guardrail pipeline above only evaluated the
        # CHILD's message. A small, not-instruction-tuned model can produce
        # an inappropriate-sounding fragment even on a completely benign
        # input (observed directly during dev testing -- see
        # training/README.md) -- there's no upstream signal to catch that,
        # so the generated reply gets checked too before it's shown.
        output_check = DEFAULT_PIPELINE.evaluate(reply)
        response_action = result.action
        response_category = None
        topic = None
        topic_numbers: list[int] = []
        if output_check.action != Action.ALLOW:
            reply = build_generation_safety_fallback(req.child.age_tier)
            # Report this honestly as a REDIRECT, not ALLOW -- the child's
            # message was fine, but what almost got shown to them wasn't,
            # and a caller (the dev UI's action badge, a future audit log)
            # should be able to tell the difference from a clean ALLOW turn.
            response_action = Action.REDIRECT
            response_category = output_check.top_category
        else:
            # Illustration topic is classified from the CHILD's question,
            # not the model's reply -- the reply is often too unreliable at
            # this model scale to classify against (see training/README.md),
            # but "what is the child asking about" is a much easier, more
            # robust signal. Only computed on a clean reply -- no cartoon
            # scene accompanies a safety fallback.
            topic = topic_classifier.classify(req.message)
            if topic:
                topic_numbers = topic_classifier.extract_numbers(req.message, topic)

        # Online learning: only ALLOW-path exchanges feed the live model --
        # REDIRECT/ESCALATE replies (from either check above) come from
        # fixed templates, not the model, so there's nothing generative to
        # learn from there. Logging the fallback text (not the flagged raw
        # generation) when the output check fired, so a training burst never
        # reinforces the output that got caught. Runs after the response is
        # sent (BackgroundTasks), so a training burst never adds latency to
        # the child's reply.
        online_trainer = getattr(DEFAULT_BACKEND, "online_trainer", None)
        if online_trainer is not None:
            background_tasks.add_task(online_trainer.log_interaction, req.message, reply)

        return ChatResponse(
            reply=reply,
            action=response_action,
            top_category=response_category,
            topic=topic,
            topic_numbers=topic_numbers,
        )

    # REDIRECT and ESCALATE both produce the same kind of reply to the child;
    # ESCALATE additionally logs to the review queue. See docs/SAFETY_MODEL.md.
    reply = build_redirect(result.top_category, req.child.age_tier)

    if result.action == Action.ESCALATE:
        top_score = result.score_for(result.top_category)
        matched_layers = next(
            (s.matched_layers for s in result.scores if s.category == result.top_category), []
        )
        DEFAULT_QUEUE.append(
            child_id=req.child.child_id,
            category=result.top_category,
            score=top_score,
            matched_layers=matched_layers,
        )

    return ChatResponse(reply=reply, action=result.action, top_category=result.top_category)
