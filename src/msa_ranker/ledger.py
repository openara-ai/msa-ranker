"""Append-only JSONL event ledger writer — MSA-owned (spec 01, ADR-012).

Best-effort and off MSA's correctness path (INV-9): appends never raise. Appends and
rotation are serialized through a single process-wide writer lock (spec 01 / NN3). The
master privacy switch `event_logging` (ADR-014) gates all writes.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from .config import DEFAULT_ROTATE_BYTES
from .ids import ulid

log = logging.getLogger("msa_ranker.ledger")


class LedgerWriter:
    """Serialized, best-effort JSONL appender for the event ledger."""

    # Locks are shared per ledger directory so the writer is truly process-wide:
    # multiple LedgerWriter instances pointing at the same dir serialize together
    # (spec 01 / NN3). Keyed by the resolved absolute path.
    _locks: dict[str, threading.Lock] = {}
    _locks_guard = threading.Lock()

    def __init__(
        self,
        ledger_dir: str | Path,
        *,
        event_logging: bool = True,
        rotate_bytes: int = DEFAULT_ROTATE_BYTES,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.dir = Path(ledger_dir)
        self.event_logging = event_logging
        self.rotate_bytes = rotate_bytes
        self.clock = clock
        key = str(self.dir.resolve())
        with LedgerWriter._locks_guard:
            self._lock = LedgerWriter._locks.setdefault(key, threading.Lock())

    # ---- public API (called by the MSA shim) ----

    def append_search(
        self,
        *,
        search_id: str,
        user_id: str,
        query: str,
        ctx: Mapping[str, Any],
        flag_on: bool,
        model_version: str | None,
        k: int,
    ) -> None:
        self._append(
            {
                "ev": "search",
                "search_id": search_id,
                "user_id": user_id,
                "query": query,
                "ctx": dict(ctx),
                "flag_on": bool(flag_on),
                "model_version": model_version,
                "k": k,
            }
        )

    def append_shown(self, *, search_id: str, rows: Iterable[Mapping[str, Any]]) -> None:
        """`rows`: dicts with media_id, position, score, heuristic_score, features."""
        # Pick controlled fields explicitly — a stray `ev`/`search_id`/`ev_id` key in a
        # row must not override the event's identity.
        for r in rows:
            self._append(
                {
                    "ev": "shown",
                    "search_id": search_id,
                    "media_id": r["media_id"],
                    "position": r["position"],
                    "score": r["score"],
                    "heuristic_score": r["heuristic_score"],
                    "features": dict(r.get("features", {})),
                }
            )

    def append_open(
        self,
        *,
        search_id: str,
        media_id: str,
        user_id: str,
        dwell_ms: int | None = None,
    ) -> None:
        self._append(
            {
                "ev": "open",
                "search_id": search_id,
                "media_id": media_id,
                "user_id": user_id,
                "action": "open",
                "dwell_ms": dwell_ms,
            }
        )

    # ---- internals ----

    def _append(self, event: dict[str, Any]) -> None:
        if not self.event_logging:
            return
        record = {"ev_id": ulid(), "ts": self.clock(), **event}
        try:
            line = json.dumps(record, separators=(",", ":"), ensure_ascii=False)
            with self._lock:  # serialize appends AND rotation (spec 01 / NN3)
                self.dir.mkdir(parents=True, exist_ok=True)
                path = self._current_file()
                with open(path, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
        except Exception:  # best-effort: drop + log, never raise (INV-9)
            log.warning("ledger append dropped (best-effort)", exc_info=True)

    def _current_file(self) -> Path:
        day = time.strftime("%Y%m%d", time.gmtime(self.clock()))
        prefix = f"events-{day}"
        # All segments for the day, compressed or not, so a new index never collides
        # with an already-rotated/compressed segment (review E).
        segments = [*self.dir.glob(f"{prefix}*.jsonl"), *self.dir.glob(f"{prefix}*.jsonl.gz")]
        # Only an uncompressed .jsonl is appendable; the newest is the current file.
        writable = sorted(
            (p for p in segments if p.suffixes[-1:] == [".jsonl"]),
            key=lambda p: (p.stat().st_mtime, p.name),
        )
        if writable and writable[-1].stat().st_size < self.rotate_bytes:
            return writable[-1]
        # Need a new segment: index = max existing index + 1 (base file has no index).
        next_idx = 0
        for p in segments:
            tail = p.name[len(prefix) :]  # "" / ".jsonl" / "-NN.jsonl[.gz]"
            if tail.startswith("-"):
                num = tail[1:].split(".", 1)[0]
                if num.isdigit():
                    next_idx = max(next_idx, int(num) + 1)
        if not segments:
            return self.dir / f"{prefix}.jsonl"
        return self.dir / f"{prefix}-{next_idx:02d}.jsonl"
