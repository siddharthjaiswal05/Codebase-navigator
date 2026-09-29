"""SQLite response cache, keyed by a hash of the exact request.

Two properties matter for evaluation, and both come from the same mechanism.

Determinism: an evaluation run that hits the cache replays byte-identical model
output, so a change in a retrieval score is attributable to the retrieval
change rather than to sampling noise.

Cost: re-running the suite after a retrieval change costs nothing, because only
genuinely new prompts reach a provider.

The key covers the model, the full message list, and the sampling parameters, so
changing any of them is a miss rather than a stale hit.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS responses (
    prompt_hash TEXT PRIMARY KEY,
    model       TEXT NOT NULL,
    request     TEXT NOT NULL,
    response    TEXT NOT NULL,
    created_at  REAL NOT NULL,
    hits        INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_responses_model ON responses(model);
"""


def prompt_hash(model: str, messages: list[dict], **params: Any) -> str:
    """Stable across processes: sorted keys, no whitespace variation."""
    payload = json.dumps(
        {"model": model, "messages": messages, "params": params},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ResponseCache:
    def __init__(self, path: Path | str = ".navigator_cache/responses.db"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.executescript(SCHEMA)
        self.conn.commit()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> dict | None:
        row = self.conn.execute(
            "SELECT response FROM responses WHERE prompt_hash = ?", (key,)
        ).fetchone()
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        self.conn.execute(
            "UPDATE responses SET hits = hits + 1 WHERE prompt_hash = ?", (key,)
        )
        self.conn.commit()
        return json.loads(row[0])

    def put(self, key: str, model: str, request: dict, response: dict) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO responses "
            "(prompt_hash, model, request, response, created_at, hits) "
            "VALUES (?, ?, ?, ?, ?, COALESCE("
            "  (SELECT hits FROM responses WHERE prompt_hash = ?), 0))",
            (
                key,
                model,
                json.dumps(request, sort_keys=True),
                json.dumps(response),
                time.time(),
                key,
            ),
        )
        self.conn.commit()

    def stats(self) -> dict:
        total = self.conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0]
        lookups = self.hits + self.misses
        return {
            "cached_responses": total,
            "session_hits": self.hits,
            "session_misses": self.misses,
            "hit_rate": round(self.hits / lookups, 4) if lookups else 0.0,
            "path": str(self.path),
        }

    def clear(self) -> None:
        self.conn.execute("DELETE FROM responses")
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()
