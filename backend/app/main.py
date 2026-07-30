"""FastAPI wrapper around the pipeline. Thin on purpose -- all the actual
logic lives in guardrails/, response/, persona/, and generation/. This file
just wires request -> pipeline -> (model | redirect) -> response.
"""
from __future__ import annotations

from fastapi import FastAPI

from app.generation.base_model import DEFAULT_BACKEND
from app.guardrails.pipeline import DEFAULT_PIPELINE
from app.models.schemas import Action, ChatRequest, ChatResponse
from app.persona.persona_engine import build_system_prompt
from app.response.redirect_engine import build_redirect
from app.review_queue import DEFAULT_QUEUE

app = FastAPI(
    title="SafeAI for Kids -- prototype",
    description=(
        "Prototype API. NOT reviewed for production use with real children -- "
        "see docs/BLOCKERS.md before treating this as anything but a dev harness."
    ),
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    result = DEFAULT_PIPELINE.evaluate(req.message)

    if result.action == Action.ALLOW:
        system_prompt = build_system_prompt(req.child)
        reply = DEFAULT_BACKEND.generate(system_prompt, req.message)
        return ChatResponse(reply=reply, action=result.action, top_category=None)

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
