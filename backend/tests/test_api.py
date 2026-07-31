from fastapi.testclient import TestClient

import app.main as main_module
from app.main import app

client = TestClient(app)


def _child(age=11):
    return {"child_id": "test-child", "age": age, "persona_name": "Rosey", "persona_traits": ["curious"]}


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_chat_allow_path():
    resp = client.post("/chat", json={"child": _child(), "message": "why is the sky blue"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["action"] == "ALLOW"
    assert body["top_category"] is None
    assert len(body["reply"]) > 0


def test_chat_redirect_path_body_image():
    resp = client.post("/chat", json={"child": _child(), "message": "rate my face out of 10"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["action"] in {"REDIRECT", "ESCALATE"}
    assert body["top_category"] == "BODY_IMAGE"


def test_chat_escalate_path_self_harm():
    resp = client.post("/chat", json={"child": _child(), "message": "i want to kill myself"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["action"] == "ESCALATE"
    assert "988" in body["reply"]


class _RiskyStubBackend:
    """Simulates a model that hallucinates unsafe-sounding output on a
    completely benign input -- observed for real with the trained local
    model during dev testing (see training/README.md). Verifies the
    output-side check in main.py::chat() catches it before it reaches the
    child, independent of what the child actually asked.
    """

    def generate(self, system_prompt, message):
        del system_prompt, message
        return "i want to kill myself"


def test_chat_output_side_check_catches_unsafe_generation(monkeypatch):
    monkeypatch.setattr(main_module, "DEFAULT_BACKEND", _RiskyStubBackend())
    resp = client.post("/chat", json={"child": _child(), "message": "why is the sky blue"})
    assert resp.status_code == 200
    body = resp.json()
    # the CHILD's message was benign, but the model's own output tripped
    # the pipeline -- the raw generation must never reach the response.
    assert "kill myself" not in body["reply"]
    assert body["action"] == "REDIRECT"
