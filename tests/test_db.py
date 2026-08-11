"""S-1.2 — open_db + additive migration runner (FR-18, INV-7)."""

from __future__ import annotations

from msa_ranker.db import open_db, run_migrations

_TABLES = {
    "search",
    "result_shown",
    "interaction",
    "ingest_state",
    "dataset",
    "model",
    "eval",
    "_migrations",
}


def test_open_db_creates_schema(tmp_path):
    conn = open_db(tmp_path / "s.sqlite")
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert _TABLES <= tables
    assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    conn.close()


def test_interaction_pk_is_ev_id(tmp_path):
    # T7: interaction is keyed on ev_id (not (search_id, media_id, created_ts)).
    conn = open_db(tmp_path / "s.sqlite")
    pk = [r["name"] for r in conn.execute("PRAGMA table_info(interaction)") if r["pk"]]
    assert pk == ["ev_id"]
    conn.close()


def test_migrations_idempotent(tmp_path):
    # INV-7: re-opening applies nothing new; the runner only applies un-recorded files.
    path = tmp_path / "s.sqlite"
    c1 = open_db(path)
    n1 = c1.execute("SELECT count(*) FROM _migrations").fetchone()[0]
    again = run_migrations(c1)  # second pass on the same connection
    c1.close()
    c2 = open_db(path)  # fresh open
    n2 = c2.execute("SELECT count(*) FROM _migrations").fetchone()[0]
    c2.close()
    assert n1 >= 1
    assert again == []  # nothing re-applied
    assert n1 == n2


def test_open_db_is_sole_connection_path():
    # AC-07.6 / review G: no raw sqlite3.connect anywhere except db.py.
    import pathlib

    import msa_ranker

    pkg = pathlib.Path(msa_ranker.__file__).parent
    offenders = [
        p.name
        for p in pkg.rglob("*.py")
        if p.name != "db.py" and "sqlite3.connect" in p.read_text(encoding="utf-8")
    ]
    assert offenders == []
