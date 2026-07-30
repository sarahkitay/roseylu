import tempfile
from pathlib import Path

from app.models.schemas import RiskCategory
from app.review_queue import ReviewQueue


def test_append_and_read():
    with tempfile.TemporaryDirectory() as tmp:
        queue = ReviewQueue(path=Path(tmp) / "queue.jsonl")
        queue.append("child-1", RiskCategory.SELF_HARM, 0.9, ["keyword"])
        entries = queue.read_all()
        assert len(entries) == 1
        assert entries[0]["child_id"] == "child-1"
        assert entries[0]["category"] == "SELF_HARM"


def test_no_raw_message_text_is_ever_persisted():
    with tempfile.TemporaryDirectory() as tmp:
        queue = ReviewQueue(path=Path(tmp) / "queue.jsonl")
        queue.append("child-1", RiskCategory.GROOMING, 0.8, ["keyword", "judge"])
        entries = queue.read_all()
        assert set(entries[0].keys()) == {"child_id", "category", "score", "matched_layers", "timestamp"}


def test_purge_child_removes_only_that_child():
    with tempfile.TemporaryDirectory() as tmp:
        queue = ReviewQueue(path=Path(tmp) / "queue.jsonl")
        queue.append("child-1", RiskCategory.SELF_HARM, 0.9, ["keyword"])
        queue.append("child-2", RiskCategory.BODY_IMAGE, 0.5, ["keyword"])
        queue.purge_child("child-1")
        remaining = queue.read_all()
        assert len(remaining) == 1
        assert remaining[0]["child_id"] == "child-2"
