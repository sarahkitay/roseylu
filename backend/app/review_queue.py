"""Append-only human-review escalation log.

Per docs/DATA_RETENTION_POLICY.md's current (unreviewed-by-counsel) leaning:
category + tier + timestamp + opaque child ID only, no raw message text and
no keyword/phrase evidence -- those are snippets of the child's actual words
and the product principle is that a child's curiosity deserves privacy even
from this queue. A specialist reviewing an entry sees *that* something in a
category needs a look, not a transcript. If that tradeoff changes, it needs a
deliberate decision, not a quiet addition of an `evidence` field here.

In production this is read by child development specialists, not engineers
(docs/BLOCKERS.md item 4 -- nobody is staffed to read it yet).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from app.models.schemas import RiskCategory

_DEFAULT_PATH = Path(__file__).parent.parent / "data" / "review_queue.jsonl"


@dataclass
class EscalationEntry:
    child_id: str
    category: str
    score: float
    matched_layers: list[str]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ReviewQueue:
    def __init__(self, path: Path = _DEFAULT_PATH) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def append(
        self, child_id: str, category: RiskCategory, score: float, matched_layers: list[str]
    ) -> EscalationEntry:
        entry = EscalationEntry(
            child_id=child_id,
            category=category.value,
            score=score,
            matched_layers=matched_layers,
        )
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(entry)) + "\n")
        return entry

    def read_all(self) -> list[dict]:
        if not self._path.exists():
            return []
        with open(self._path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def purge_child(self, child_id: str) -> int:
        """COPPA deletion-on-request path (docs/COMPLIANCE_COPPA.md)."""
        entries = [e for e in self.read_all() if e["child_id"] != child_id]
        removed = len(self.read_all()) - len(entries)
        with open(self._path, "w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")
        return removed


DEFAULT_QUEUE = ReviewQueue()
