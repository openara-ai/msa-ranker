"""Fold the JSONL event ledger into the training SoR (spec 07, ADR-012).

Idempotent: resumes from a per-file `ingest_state` byte watermark and upserts
ON CONFLICT DO NOTHING anchored on `ev_id`. A torn trailing line (no newline yet)
is left for the next run. Out-of-order events are tolerated (rows key on ids).
"""

from __future__ import annotations

import gzip
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

_SEARCH_SQL = (
    "INSERT INTO search "
    "(ev_id, search_id, user_id, query_text, query_ctx_json, "
    "flag_on, model_version, k, created_ts) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT DO NOTHING"
)
_SHOWN_SQL = (
    "INSERT INTO result_shown "
    "(ev_id, search_id, media_id, position, score, heuristic_score, features_json) "
    "VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT DO NOTHING"
)
_OPEN_SQL = (
    "INSERT INTO interaction "
    "(ev_id, search_id, media_id, user_id, action, dwell_ms, created_ts) "
    "VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT DO NOTHING"
)


def ingest(conn: sqlite3.Connection, ledger_dir: str | Path) -> dict[str, int]:
    """Ingest all ledger files into the SoR. Returns counts."""
    ledger_dir = Path(ledger_dir)
    stats = {"files": 0, "lines": 0, "inserted": 0, "skipped": 0}
    files = sorted(
        {*ledger_dir.glob("events-*.jsonl"), *ledger_dir.glob("events-*.jsonl.gz")},
        key=lambda p: p.name,
    )
    for path in files:
        stats["files"] += 1
        if path.name.endswith(".gz"):
            _ingest_gz(conn, path, stats)
        else:
            _ingest_plain(conn, path, stats)
        conn.commit()
    return stats


def _ingest_plain(conn: sqlite3.Connection, path: Path, stats: dict[str, int]) -> None:
    start = _offset(conn, path.name)
    # BACKLOG (review item H): this reads the whole post-watermark remainder into memory.
    # Fine at single-user scale (a ledger file caps at 64 MB and rotates), but a future
    # change should stream in bounded chunks while tracking the last-complete-line offset.
    with open(path, "rb") as fh:
        fh.seek(start)
        data = fh.read()
    nl = data.rfind(b"\n")
    if nl == -1:  # no complete line past the watermark yet
        return
    for line in data[: nl + 1].decode("utf-8").splitlines():
        _process(conn, line, stats)
    _set_offset(conn, path.name, start + nl + 1)


def _ingest_gz(conn: sqlite3.Connection, path: Path, stats: dict[str, int]) -> None:
    # Rotated .gz files are immutable; consume once, watermark = compressed size.
    size = path.stat().st_size
    if _offset(conn, path.name) >= size > 0:
        return
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            _process(conn, line, stats)
    _set_offset(conn, path.name, size)


def _process(conn: sqlite3.Connection, line: str, stats: dict[str, int]) -> None:
    line = line.strip()
    if not line:
        return
    stats["lines"] += 1
    try:
        e: Any = json.loads(line)
    except ValueError:
        stats["skipped"] += 1
        return
    if not isinstance(e, dict):
        stats["skipped"] += 1
        return
    ev = e.get("ev")
    # A line can be valid JSON yet missing a required field — skip it and keep going;
    # one malformed event must never abort the whole ingest run.
    try:
        if ev == "search":
            sql, params = _SEARCH_SQL, (
                e["ev_id"],
                e["search_id"],
                e.get("user_id", "default"),
                e["query"],
                json.dumps(e.get("ctx")),
                int(bool(e.get("flag_on"))),
                e.get("model_version"),
                e["k"],
                e["ts"],
            )
        elif ev == "shown":
            sql, params = _SHOWN_SQL, (
                e["ev_id"],
                e["search_id"],
                e["media_id"],
                e["position"],
                e["score"],
                e["heuristic_score"],
                json.dumps(e.get("features", {})),
            )
        elif ev == "open":
            sql, params = _OPEN_SQL, (
                e["ev_id"],
                e["search_id"],
                e["media_id"],
                e.get("user_id", "default"),
                e.get("action", "open"),
                e.get("dwell_ms"),
                e["ts"],
            )
        else:
            stats["skipped"] += 1
            return
    except (KeyError, TypeError):
        stats["skipped"] += 1
        return
    cur = conn.execute(sql, params)
    stats["inserted"] += cur.rowcount if cur.rowcount > 0 else 0


def _offset(conn: sqlite3.Connection, name: str) -> int:
    row = conn.execute(
        "SELECT byte_offset FROM ingest_state WHERE ledger_file = ?", (name,)
    ).fetchone()
    return int(row[0]) if row else 0


def _set_offset(conn: sqlite3.Connection, name: str, offset: int) -> None:
    conn.execute(
        "INSERT INTO ingest_state (ledger_file, byte_offset, updated_ts) VALUES (?, ?, ?) "
        "ON CONFLICT(ledger_file) DO UPDATE SET "
        "byte_offset = excluded.byte_offset, updated_ts = excluded.updated_ts",
        (name, offset, time.time()),
    )
