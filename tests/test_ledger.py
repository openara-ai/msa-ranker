"""S-1.4 — ledger appender (spec 01): events, off-switch, best-effort, rotation."""

from __future__ import annotations

import json

from msa_ranker.ledger import LedgerWriter


def _read_all(ledger_dir):
    events = []
    for p in sorted(ledger_dir.glob("events-*.jsonl")):
        events += [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    return events


def test_append_writes_events(ledger_dir):
    w = LedgerWriter(ledger_dir)
    w.append_search(
        search_id="s1",
        user_id="default",
        query="q",
        ctx={"people": []},
        flag_on=False,
        model_version=None,
        k=2,
    )
    w.append_shown(
        search_id="s1",
        rows=[
            {"media_id": "a", "position": 0, "score": 0.8, "heuristic_score": 0.8, "features": {}},
            {"media_id": "b", "position": 1, "score": 0.5, "heuristic_score": 0.5, "features": {}},
        ],
    )
    w.append_open(search_id="s1", media_id="a", user_id="default")
    events = _read_all(ledger_dir)
    assert [e["ev"] for e in events] == ["search", "shown", "shown", "open"]
    assert len({e["ev_id"] for e in events}) == 4  # AC-01.2 unique
    assert all("ts" in e for e in events)
    # AC-01.4: position present + heuristic_score (NN1) on shown events
    shown = [e for e in events if e["ev"] == "shown"]
    assert [e["position"] for e in shown] == [0, 1]
    assert all("heuristic_score" in e for e in shown)


def test_event_logging_off_writes_nothing(ledger_dir):
    # AC-01.8 (ADR-014 off-switch)
    w = LedgerWriter(ledger_dir, event_logging=False)
    w.append_search(
        search_id="s1",
        user_id="default",
        query="q",
        ctx={},
        flag_on=True,
        model_version="m",
        k=1,
    )
    w.append_open(search_id="s1", media_id="a", user_id="default")
    assert _read_all(ledger_dir) == []


def test_append_is_best_effort(tmp_path):
    # AC-01.5 / INV-9: an unwritable target must NOT raise. Parent is a file.
    blocker = tmp_path / "afile"
    blocker.write_text("x")
    w = LedgerWriter(blocker / "sub")  # mkdir under a file → OSError, swallowed
    w.append_open(search_id="s1", media_id="a", user_id="default")  # must not raise


def test_rotation_creates_multiple_files(ledger_dir):
    w = LedgerWriter(ledger_dir, rotate_bytes=50)
    for i in range(30):
        w.append_open(search_id=f"s{i}", media_id="a", user_id="default")
    files = list(ledger_dir.glob("events-*.jsonl"))
    assert len(files) > 1
    # every line across every file is still well-formed JSON
    for p in files:
        for line in p.read_text().splitlines():
            json.loads(line)


def test_concurrent_appends_well_formed(ledger_dir):
    # AC-01.9: single-writer serialization — concurrent appends never tear.
    import threading

    w = LedgerWriter(ledger_dir, rotate_bytes=200)

    def worker(n):
        for i in range(25):
            w.append_open(search_id=f"t{n}-{i}", media_id="a", user_id="default")

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    events = _read_all(ledger_dir)
    assert len(events) == 8 * 25  # none lost
    assert len({e["ev_id"] for e in events}) == 8 * 25  # all distinct, no torn lines


def test_append_shown_ignores_stray_keys(ledger_dir):
    # Review D: a stray ev/search_id/ev_id key in a row must not override the event.
    w = LedgerWriter(ledger_dir)
    w.append_shown(
        search_id="s1",
        rows=[
            {
                "media_id": "a",
                "position": 0,
                "score": 0.1,
                "heuristic_score": 0.1,
                "features": {},
                "search_id": "HACK",
                "ev": "open",
                "ev_id": "HACK",
            }
        ],
    )
    e = _read_all(ledger_dir)[0]
    assert e["ev"] == "shown"
    assert e["search_id"] == "s1"
    assert e["ev_id"] != "HACK"


def test_shared_lock_across_instances(ledger_dir):
    # Review C: two writers on the same dir share one lock (process-wide serialization).
    a = LedgerWriter(ledger_dir)
    b = LedgerWriter(ledger_dir)
    assert a._lock is b._lock
