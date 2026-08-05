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
    # Uses a curated-knowledge question (deterministic answer), not a
    # question that would fall through to the raw generative model -- the
    # local model is stochastic and, per training/README.md, occasionally
    # hallucinates a fragment that trips the output-side check even on
    # completely benign prompts. That's real, known, and separately covered
    # by test_chat_output_side_check_catches_unsafe_generation below
    # (deterministically, via a mocked backend); it shouldn't make an
    # unrelated "does the ALLOW response have the right shape" test flaky.
    resp = client.post("/chat", json={"child": _child(), "message": "what is a synonym"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["action"] == "ALLOW"
    assert body["top_category"] is None
    assert len(body["reply"]) > 0


def test_chat_math_topic_uses_templated_answer_not_the_model(monkeypatch):
    # The generative model should never even be called for a numeric math
    # topic -- if it were, this stub's obviously-wrong text would leak
    # through instead of the correct templated answer.
    class _ShouldNotBeCalledBackend:
        def generate(self, system_prompt, message):
            raise AssertionError("generative model should be bypassed for numeric math topics")

    monkeypatch.setattr(main_module, "DEFAULT_BACKEND", _ShouldNotBeCalledBackend())
    resp = client.post("/chat", json={"child": _child(), "message": "what is 3 plus 5"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["action"] == "ALLOW"
    assert body["topic"] == "addition"
    assert body["topic_numbers"] == [3, 5]
    assert "8" in body["reply"]  # matches the illustration's own total


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
