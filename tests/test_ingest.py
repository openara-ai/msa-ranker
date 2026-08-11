"""S-1.3 — ledger → SoR ingest (spec 07): golden, idempotent, watermark, torn line."""

from __future__ import annotations

import json

from msa_ranker.ingest import ingest

FIX = [
    {
        "ev": "search",
        "ev_id": "e1",
        "ts": 1.0,
        "search_id": "s1",
        "user_id": "default",
        "query": "q",
        "ctx": {"people": []},
        "flag_on": False,
        "model_version": None,
        "k": 2,
    },
    {
        "ev": "shown",
        "ev_id": "e2",
        "ts": 1.0,
        "search_id": "s1",
        "media_id": "a",
        "position": 0,
        "score": 0.8,
        "heuristic_score": 0.8,
        "features": {},
    },
    {
        "ev": "shown",
        "ev_id": "e3",
        "ts": 1.0,
        "search_id": "s1",
        "media_id": "b",
        "position": 1,
        "score": 0.5,
        "heuristic_score": 0.5,
        "features": {},
    },
    {
        "ev": "open",
        "ev_id": "e4",
        "ts": 2.0,
        "search_id": "s1",
        "media_id": "a",
        "user_id": "default",
        "action": "open",
        "dwell_ms": None,
    },
]


def _write(ledger_dir, events, name="events-20260101.jsonl"):
    p = ledger_dir / name
    p.write_text("".join(json.dumps(e) + "\n" for e in events))
    return p


def _counts(conn):
    return {
        t: conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
        for t in ("search", "result_shown", "interaction")
    }


def test_ingest_golden(db, ledger_dir):
    _write(ledger_dir, FIX)
    stats = ingest(db, ledger_dir)
    assert _counts(db) == {"search": 1, "result_shown": 2, "interaction": 1}
    assert stats["inserted"] == 4
    row = db.execute("SELECT query_text, k FROM search WHERE search_id='s1'").fetchone()
    assert row["query_text"] == "q" and row["k"] == 2
    # mapping: ctx → query_ctx_json, heuristic_score persisted
    hs = db.execute("SELECT heuristic_score FROM result_shown WHERE media_id='a'").fetchone()[0]
    assert hs == 0.8


def test_ingest_idempotent(db, ledger_dir):
    _write(ledger_dir, FIX)
    ingest(db, ledger_dir)
    ingest(db, ledger_dir)  # AC-07.1: re-ingest is a no-op
    assert _counts(db) == {"search": 1, "result_shown": 2, "interaction": 1}


def test_watermark_resume(db, ledger_dir):
    p = _write(ledger_dir, FIX)
    ingest(db, ledger_dir)
    new = {
        "ev": "search",
        "ev_id": "e9",
        "ts": 3.0,
        "search_id": "s2",
        "user_id": "default",
        "query": "q2",
        "ctx": {},
        "flag_on": True,
        "model_version": None,
        "k": 1,
    }
    with open(p, "a") as fh:
        fh.write(json.dumps(new) + "\n")
    ingest(db, ledger_dir)  # AC-07.2: only the new row is added
    assert _counts(db)["search"] == 2


def test_torn_trailing_line(db, ledger_dir):
    # AC-07.3 / spec 01 AC-01.6: a partial last line is skipped, picked up once complete.
    p = ledger_dir / "events-20260101.jsonl"
    good = "".join(json.dumps(e) + "\n" for e in FIX[:3])
    p.write_text(good + json.dumps(FIX[3])[:-5])  # torn 4th line, no newline
    ingest(db, ledger_dir)
    assert _counts(db) == {"search": 1, "result_shown": 2, "interaction": 0}
    p.write_text(good + json.dumps(FIX[3]) + "\n")  # complete it
    ingest(db, ledger_dir)
    assert _counts(db)["interaction"] == 1


def test_corrupt_line_skipped(db, ledger_dir):
    p = ledger_dir / "events-20260101.jsonl"
    p.write_text(json.dumps(FIX[0]) + "\n" + "{not json}\n" + json.dumps(FIX[1]) + "\n")
    stats = ingest(db, ledger_dir)
    assert _counts(db) == {"search": 1, "result_shown": 1, "interaction": 0}
    assert stats["skipped"] == 1


def test_malformed_event_skipped_not_aborted(db, ledger_dir):
    # Review B: a valid-JSON line missing a required field is skipped, not fatal.
    bad = {"ev": "search", "ev_id": "x", "ts": 1.0, "search_id": "s9"}  # no query/k
    p = ledger_dir / "events-20260101.jsonl"
    p.write_text(json.dumps(FIX[0]) + "\n" + json.dumps(bad) + "\n" + json.dumps(FIX[1]) + "\n")
    stats = ingest(db, ledger_dir)
    assert _counts(db) == {"search": 1, "result_shown": 1, "interaction": 0}
    assert stats["skipped"] == 1


def test_ingest_gz(db, ledger_dir):
    # Review F: the .gz path works (AC-07.1 variant).
    import gzip

    p = ledger_dir / "events-20251231.jsonl.gz"
    with gzip.open(p, "wt", encoding="utf-8") as fh:
        for e in FIX:
            fh.write(json.dumps(e) + "\n")
    ingest(db, ledger_dir)
    assert _counts(db) == {"search": 1, "result_shown": 2, "interaction": 1}
    ingest(db, ledger_dir)  # immutable .gz consumed once
    assert _counts(db) == {"search": 1, "result_shown": 2, "interaction": 1}
