"""Label construction — Click > Skip-Above (spec 03, FR-5/FR-7, ADR-007, INV-4).

The single place position-bias debiasing lives: MSA emits raw open/shown events
(ADR-012); this turns them into training labels. Get it wrong and the model
re-learns position instead of relevance.

For each search (scoped to its user): an opened result is positive; a *non-opened*
result **above** the deepest open is a negative (examined-but-passed); a non-opened
result **below** the deepest open is **dropped** (likely unexamined) — never a 0. A
search with no opens contributes nothing. Splits are query-**grouped** so two
instances of the same intent can't straddle train/eval (INV-4).
"""

from __future__ import annotations

import json
import logging
import random
import sqlite3
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

from .features import FEATURE_NAMES, QueryContext

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ShownRow:
    """One shown candidate: its media id, position, and the logged feature dict."""

    media_id: str
    position: int
    features: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchBundle:
    """A search + everything it produced — its shown rows and the media it opened."""

    search_id: str
    user_id: str
    ctx: QueryContext
    shown: list[ShownRow]
    opened: set[str]  # opened media_ids


@dataclass(frozen=True)
class LabelRow:
    """One training row. `group` is the leak-free split key (query signature).

    `features` is a tuple so the immutability is real — `frozen=True` blocks
    reassignment, and a tuple blocks in-place mutation a downstream step might do.
    """

    search_id: str
    user_id: str
    media_id: str
    features: tuple[float, ...]
    label: int
    group: str


def _norm_tokens(tokens: Iterable[str]) -> list[str]:
    """The same normalization features use: lowercase, de-dup, sort (deterministic)."""
    return sorted({t.lower() for t in tokens if t})


def _canon_date(date_intent: Any) -> Any:
    """Canonical, JSON-stable value for the date-intent component of the group key.

    Only an *explicit* `None` means "no date intent" — a falsy-but-present value
    (`0`, `""`) is a real, distinct intent and must not collapse into the None case.
    """
    if date_intent is None:
        return None
    if isinstance(date_intent, dict):
        return json.dumps(date_intent, sort_keys=True, separators=(",", ":"))
    return date_intent


def group_key(user_id: str, ctx: QueryContext) -> str:
    """A canonical, user-scoped, entity-based query signature (spec 03).

    Two *separate* searches with the same intent → the same group → the same split
    side; two users' "my dad" → different groups. Built from resolved person_ids +
    normalized visual tokens + date intent — never raw query text (ADR-008), and
    never `search_id` (which is unique and would silently leak duplicate intents
    across splits).

    Encoded as a JSON array (not a delimiter-joined string) so a `user_id` or
    `person_id` containing a separator char can't collapse two distinct intents into
    one group — a silent INV-4 violation.
    """
    return json.dumps(
        [
            user_id,
            sorted(set(ctx.people or [])),
            _norm_tokens(ctx.visual_tokens or []),
            _canon_date(ctx.date_intent),
        ],
        separators=(",", ":"),
    )


def _vector(features: dict[str, float]) -> tuple[float, ...]:
    """The feature dict as the frozen-order vector (FEATURE_NAMES is the contract)."""
    return tuple(float(features.get(name, 0.0) or 0.0) for name in FEATURE_NAMES)


def build_labels(searches: Iterable[SearchBundle]) -> list[LabelRow]:
    """Click > Skip-Above (ADR-007). One SearchBundle = a search + its shown + opens.

    Returns rows in a deterministic order (AC-03.5). A search with no usable open
    contributes zero rows (AC-03.3); a result below the deepest open is dropped, not
    labeled 0 (AC-03.4); an open with no matching shown row is an anomaly — dropped +
    logged, never fabricated (AC-03.7).
    """
    rows: list[LabelRow] = []
    for b in searches:
        opened = set(b.opened)
        if not opened:
            continue  # no preference signal (AC-03.3)
        shown_by_media = {s.media_id: s for s in b.shown}
        for media_id in opened - shown_by_media.keys():
            logger.warning(
                "orphan open dropped: search_id=%s media_id=%s has no shown row",
                b.search_id,
                media_id,
            )
        opened_shown = [shown_by_media[m] for m in opened if m in shown_by_media]
        if not opened_shown:
            continue  # every open was orphaned — no anchorable positive
        deepest = max(s.position for s in opened_shown)
        group = group_key(b.user_id, b.ctx)
        for s in b.shown:
            if s.media_id in opened:
                label = 1
            elif s.position < deepest:
                label = 0  # examined-but-passed (above the deepest open)
            else:
                continue  # below the deepest open → likely unexamined → drop
            rows.append(
                LabelRow(b.search_id, b.user_id, s.media_id, _vector(s.features), label, group)
            )
    rows.sort(key=lambda r: (r.group, r.search_id, r.media_id))
    return rows


def split(
    rows: list[LabelRow], *, seed: int, eval_frac: float
) -> tuple[list[LabelRow], list[LabelRow]]:
    """Query-GROUPED split — a `group` lands wholly in train or eval (INV-4).

    Deterministic given `seed`: groups are sorted, shuffled with a seeded RNG, and a
    round-half-up `eval_frac` fraction go to eval. Splitting on `group` (not
    `search_id`) keeps two instances of the same intent on the same side (INV-4).

    **Both sides are guaranteed non-empty**: `n_eval` is clamped to `[1, n-1]`, so a
    silently-empty eval set (which the harness would score as a valid 0 and let the
    gate run on no queries) cannot happen. Raises if a leak-free non-empty split is
    impossible: fewer than 2 groups, or `eval_frac` outside `(0, 1)`.
    """
    if not 0.0 < eval_frac < 1.0:
        raise ValueError(f"eval_frac must be in (0, 1), got {eval_frac}")
    groups = sorted({r.group for r in rows})
    n = len(groups)
    if n < 2:
        raise ValueError(f"need >= 2 query groups for a leak-free split, got {n}")
    random.Random(seed).shuffle(groups)
    # Round half up (not Python's banker's round, where round(0.5)==0), then clamp so
    # neither side is empty.
    n_eval = max(1, min(int(n * eval_frac + 0.5), n - 1))
    eval_groups = set(groups[:n_eval])
    train = [r for r in rows if r.group not in eval_groups]
    held = [r for r in rows if r.group in eval_groups]
    return train, held


def _ctx_from_json(raw: str | None) -> QueryContext:
    """Rebuild a QueryContext from a `search.query_ctx_json` blob (may be null/partial)."""
    data = json.loads(raw) if raw else None
    if not isinstance(data, dict):
        return QueryContext()
    return QueryContext(
        people=list(data.get("people") or []),
        visual_tokens=list(data.get("visual_tokens") or []),
        date_intent=data.get("date_intent"),
        place_intent=data.get("place_intent"),
    )


def iter_bundles(conn: sqlite3.Connection) -> Iterator[SearchBundle]:
    """Adapt the ingested SoR into SearchBundles for `build_labels` (spec 05 pipeline).

    Reads columns positionally (tuple-unpack / `row[i]`, both in SELECT order) so it
    doesn't depend on `conn.row_factory`. The `result_shown` read is explicitly
    `ORDER BY position, media_id` — SQLite gives no row order without it, and this is
    correctness-critical code that reasons about positions.
    """
    searches = conn.execute("SELECT search_id, user_id, query_ctx_json FROM search").fetchall()
    for sid, user_id, ctx_json in searches:
        shown = [
            ShownRow(row[0], int(row[1]), json.loads(row[2] or "{}"))
            for row in conn.execute(
                "SELECT media_id, position, features_json FROM result_shown "
                "WHERE search_id = ? ORDER BY position, media_id",
                (sid,),
            )
        ]
        opened = {
            row[0]
            for row in conn.execute(
                "SELECT media_id FROM interaction WHERE search_id = ? AND action = 'open'",
                (sid,),
            )
        }
        yield SearchBundle(sid, user_id, _ctx_from_json(ctx_json), shown, opened)
