"""Tests for training/scripts/teacher_review_loop.py.

Lives under backend/tests/ (with the training/scripts path added manually
below) rather than getting no coverage at all like the other training/
scripts/ tools -- justified here because manual testing already caught two
real bugs in this specific script (an unrecognized backend alias, and a
checkpoint that advanced past failed reviews so they'd never be retried),
and both are exactly the kind of subtle logic worth pinning down.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "training" / "scripts"))
import teacher_review_loop as trl  # noqa: E402
from cost_tracker import BudgetExceeded, CostTracker  # noqa: E402


def _generous_tracker() -> CostTracker:
    # A budget no test here should plausibly hit -- these tests exercise
    # checkpoint/retry logic, not the spend cap itself (see
    # test_cost_tracker.py for that).
    return CostTracker(budget_usd=1000.0)


def test_normalize_backend_accepts_aliases():
    # "claude" was the actual value used in the first real .env config --
    # every review failed with "unknown teacher backend: claude" until this
    # was added.
    assert trl.normalize_backend("claude") == "anthropic"
    assert trl.normalize_backend("Claude") == "anthropic"
    assert trl.normalize_backend("anthropic") == "anthropic"
    assert trl.normalize_backend("gpt") == "openai"
    assert trl.normalize_backend("chatgpt") == "openai"
    assert trl.normalize_backend("openai") == "openai"


def test_normalize_backend_rejects_unknown_values():
    with pytest.raises(SystemExit):
        trl.normalize_backend("gemini")


def test_review_one_handles_markdown_fenced_json(monkeypatch):
    # 3 real teacher responses in the first live batch wrapped valid JSON in
    # ```json ... ``` even though the prompt says "respond with ONLY JSON" --
    # a common model quirk, not a one-off. review_one() must not treat this
    # as unparseable.
    fenced = '```json\n{"appropriate": true, "quality": 4, "sounds_generic_ai": false, "corrected_reply": null, "reasoning": "fine"}\n```'
    monkeypatch.setattr(trl, "call_teacher", lambda *a, **k: fenced)
    verdict = trl.review_one("q", "a", "anthropic", "fake-model", _generous_tracker())
    assert verdict["quality"] == 4
    assert verdict["reasoning"] == "fine"


@pytest.fixture()
def isolated_paths(tmp_path, monkeypatch):
    live_log = tmp_path / "live_interactions.jsonl"
    reviewed = tmp_path / "teacher_reviewed.jsonl"
    checkpoint = tmp_path / "checkpoint.json"
    monkeypatch.setattr(trl, "_LIVE_LOG_PATH", live_log)
    monkeypatch.setattr(trl, "_REVIEWED_PATH", reviewed)
    monkeypatch.setattr(trl, "_CHECKPOINT_PATH", checkpoint)
    return live_log, reviewed, checkpoint


def _write_log(path: Path, entries: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")


def test_checkpoint_does_not_advance_past_a_failing_review(isolated_paths, monkeypatch):
    live_log, reviewed, checkpoint = isolated_paths
    _write_log(live_log, [
        {"child": "q1", "rosey": "a1"},
        {"child": "q2", "rosey": "a2"},  # this one will fail
        {"child": "q3", "rosey": "a3"},
    ])

    def flaky_review(child, reply, backend, model, tracker):
        if child == "q2":
            raise RuntimeError("simulated transient API failure")
        return {"appropriate": True, "quality": 5, "corrected_reply": None}

    monkeypatch.setattr(trl, "review_one", flaky_review)

    n = trl.process_new_interactions("anthropic", "fake-model", _generous_tracker())

    assert n == 1  # only q1 succeeded
    assert trl._load_checkpoint() == 1  # stopped AT q2 (index 1), not past it -- this is the actual bug that shipped
    reviewed_entries = [json.loads(line) for line in reviewed.read_text().splitlines()]
    assert [e["child"] for e in reviewed_entries] == ["q1"]

    # a second run (e.g. the failure was transient and is now fixed) must
    # retry q2, not skip straight to q3
    monkeypatch.setattr(trl, "review_one", lambda child, reply, backend, model, tracker: {"appropriate": True, "quality": 5, "corrected_reply": None})
    n2 = trl.process_new_interactions("anthropic", "fake-model", _generous_tracker())
    assert n2 == 2  # q2 and q3
    assert trl._load_checkpoint() == 3


def test_malformed_log_line_is_skipped_permanently(isolated_paths, monkeypatch):
    live_log, reviewed, checkpoint = isolated_paths
    live_log.write_text('{"child": "q1", "rosey": "a1"}\nnot valid json\n{"child": "q2", "rosey": "a2"}\n', encoding="utf-8")

    monkeypatch.setattr(trl, "review_one", lambda child, reply, backend, model, tracker: {"appropriate": True, "quality": 5, "corrected_reply": None})

    n = trl.process_new_interactions("anthropic", "fake-model", _generous_tracker())

    assert n == 2  # q1 and q2 both succeeded; the bad line was skipped, not retried forever
    assert trl._load_checkpoint() == 3


def test_budget_exceeded_stops_the_run_and_checkpoints_at_the_stopping_point(isolated_paths, monkeypatch):
    # The whole point of the spend cap: it must actually stop the run (not
    # get silently swallowed as a generic transient failure -- review_one's
    # broad except would otherwise treat BudgetExceeded exactly like a
    # flaky network error and just skip ahead), and it must checkpoint at
    # the line it stopped on so a later run with a raised budget resumes
    # there instead of re-paying for or skipping already-reviewed lines.
    live_log, reviewed, checkpoint = isolated_paths
    _write_log(live_log, [
        {"child": "q1", "rosey": "a1"},
        {"child": "q2", "rosey": "a2"},
        {"child": "q3", "rosey": "a3"},
    ])

    calls = []

    def review_then_exceed(child, reply, backend, model, tracker):
        calls.append(child)
        if child == "q2":
            raise BudgetExceeded("simulated: budget reached")
        return {"appropriate": True, "quality": 5, "corrected_reply": None}

    monkeypatch.setattr(trl, "review_one", review_then_exceed)

    with pytest.raises(BudgetExceeded):
        trl.process_new_interactions("anthropic", "fake-model", _generous_tracker())

    assert calls == ["q1", "q2"]  # stopped at q2, never reached q3
    assert trl._load_checkpoint() == 1  # checkpoint at q2 (index 1), so a retry starts there
    reviewed_entries = [json.loads(line) for line in reviewed.read_text().splitlines()]
    assert [e["child"] for e in reviewed_entries] == ["q1"]
