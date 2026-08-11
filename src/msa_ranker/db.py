"""SoR connection + additive migration runner (FR-18, INV-7).

`open_db()` is the sole connection path: WAL, busy-timeout, and apply any
un-recorded `migrations/*.sql` in filename order, tracked in `_migrations`.
Shipped migrations are never edited (INV-7); changes add a new numbered file.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

_MIGRATIONS_DIR = Path(__file__).parent / "migrations"


def open_db(path: str | Path) -> sqlite3.Connection:
    """Open the SoR at `path`, applying WAL + pending migrations. Creates parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=15.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=15000")
    conn.execute("PRAGMA foreign_keys=ON")
    run_migrations(conn)
    return conn


def run_migrations(conn: sqlite3.Connection) -> list[str]:
    """Apply un-recorded `*.sql` migrations in filename order. Returns names applied."""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS _migrations ("
        "  name TEXT PRIMARY KEY,"
        "  applied_at REAL NOT NULL"
        ")"
    )
    conn.commit()
    applied = {r[0] for r in conn.execute("SELECT name FROM _migrations")}
    files = sorted(_MIGRATIONS_DIR.glob("*.sql"))
    if not files:
        raise FileNotFoundError(f"no migrations found under {_MIGRATIONS_DIR}")
    newly: list[str] = []
    for path in files:
        if path.name in applied:
            continue
        # Atomic: the leading BEGIN starts a transaction that executescript does NOT
        # auto-close, so the migration body and its `_migrations` marker commit together
        # (or roll back together on any failure) — INV-7.
        try:
            conn.executescript("BEGIN;\n" + path.read_text("utf-8"))
            conn.execute(
                "INSERT INTO _migrations (name, applied_at) VALUES (?, ?)",
                (path.name, time.time()),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        newly.append(path.name)
    return newly
