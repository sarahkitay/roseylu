"""FastAPI wrapper around the pipeline. Thin on purpose -- all the actual
logic lives in orchestrator.py (and, beneath that, guardrails/, response/,
persona/, and generation/). This file just wires HTTP <-> orchestrator,
plus the dev chat UI and the online-learning status endpoint.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import BackgroundTasks, FastAPI
from fastapi.responses import HTMLResponse

from app.generation.base_model import DEFAULT_BACKEND
from app.models.schemas import ChatRequest, ChatResponse
from app.orchestrator import handle_chat_turn

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
    return handle_chat_turn(req.child, req.message, schedule_online_learning=background_tasks.add_task)
