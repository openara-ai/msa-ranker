"""Dataset freeze / replay (spec 05, FR-6, INV-4).

A `dataset_id` must be **immutable and replayable**: the SoR keeps ingesting, and a late
`open` would shift the deepest-open boundary → different labels/splits. So freeze
**materializes** the rows it composes (migration 0002):

  - `dataset_row`  — the frozen TRAIN-split label rows (post Click>Skip-Above).
  - `dataset_eval` — the frozen FULL shown candidate set per EVAL-split search, with the
    logged `heuristic_score` (baseline ranking) and the opened flag (relevance), so eval
    ranks the whole set (spec 04) independent of the training-label drop.

Replaying a `dataset_id` reads those frozen rows back — byte-identical regardless of
later ingestion (AC-05.1). The split seed + `eval_frac` live in the manifest for lineage.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from .config import DEFAULT_EVAL_FRAC, DEFAULT_K, DEFAULT_MIN_LABELED_SEARCHES
from .eval import Query
from .features import FEATURE_SET_VERSION
from .ids import ulid
from .labels import LabelRow, build_labels, iter_bundles, split
from .model import LogRegModel


class InsufficientDataError(RuntimeError):
    """Raised when too few labelled searches exist to train (the min-data gate, AC-05.4)."""


@dataclass(frozen=True)
class EvalCand:
    """One frozen eval candidate: a shown media + its baseline score + relevance."""

    search_id: str
    media_id: str
    heuristic_score: float
    features: dict[str, float]
    is_positive: bool


@dataclass(frozen=True)
class FrozenDataset:
    dataset_id: str
    seed: int
    eval_frac: float
    k: int
    train: list[LabelRow]
    eval_cands: list[EvalCand]


def freeze_dataset(
    conn: sqlite3.Connection,
    *,
    seed: int,
    eval_frac: float = DEFAULT_EVAL_FRAC,
    k: int = DEFAULT_K,
    min_searches: int = DEFAULT_MIN_LABELED_SEARCHES,
    note: str | None = None,
) -> str:
    """Build labels from the SoR, split, and materialize an immutable dataset → its id.

    Refuses below `min_searches` labelled searches (AC-05.4) — nothing is written.
    """
    bundles = list(iter_bundles(conn))
    rows = build_labels(bundles)
    labeled = {r.search_id for r in rows}
    if len(labeled) < min_searches:
        raise InsufficientDataError(
            f"only {len(labeled)} labelled searches (< {min_searches}); refusing to freeze"
        )
    # Too few distinct query groups can't yield a leak-free non-empty split — that's an
    # insufficient-data condition, not a crash. Translate it (before any writes) so the
    # CLI reports a clean refusal rather than a stack trace.
    n_groups = len({r.group for r in rows})
    if n_groups < 2:
        raise InsufficientDataError(
            f"only {n_groups} distinct query group(s); need >= 2 for a leak-free split"
        )
    train, held = split(rows, seed=seed, eval_frac=eval_frac)
    eval_search_ids = {r.search_id for r in held}

    dataset_id = ulid()
    created = time.time()
    manifest: dict[str, Any] = {
        "dataset_id": dataset_id,
        "seed": seed,
        "eval_frac": eval_frac,
        "k": k,
        "feature_set_version": FEATURE_SET_VERSION,
        "n_labeled_searches": len(labeled),
        "n_train_rows": len(train),
        "n_eval_searches": len(eval_search_ids),
        "created_ts": created,
    }

    with conn:  # one transaction — commits on success, rolls back on any error
        conn.execute(
            "INSERT INTO dataset (dataset_id, created_ts, manifest_json, note) VALUES (?,?,?,?)",
            (dataset_id, created, json.dumps(manifest), note),
        )
        conn.executemany(
            "INSERT INTO dataset_row "
            "(dataset_id, search_id, media_id, user_id, features_json, label, grp) "
            "VALUES (?,?,?,?,?,?,?)",
            [
                (
                    dataset_id,
                    r.search_id,
                    r.media_id,
                    r.user_id,
                    json.dumps(list(r.features)),
                    r.label,
                    r.group,
                )
                for r in train
            ],
        )
        conn.executemany(
            "INSERT INTO dataset_eval "
            "(dataset_id, search_id, media_id, heuristic_score, features_json, is_positive) "
            "VALUES (?,?,?,?,?,?)",
            list(_eval_candidate_params(conn, dataset_id, bundles, eval_search_ids)),
        )
    return dataset_id


def _eval_candidate_params(conn, dataset_id, bundles, eval_search_ids):
    """Freeze the full shown set + heuristic_score + opened flag for each eval search."""
    for b in bundles:
        if b.search_id not in eval_search_ids:
            continue
        hscore = {
            row[0]: float(row[1])
            for row in conn.execute(
                "SELECT media_id, heuristic_score FROM result_shown WHERE search_id = ?",
                (b.search_id,),
            )
        }
        for s in b.shown:
            yield (
                dataset_id,
                b.search_id,
                s.media_id,
                hscore.get(s.media_id, 0.0),
                json.dumps(s.features),
                int(s.media_id in b.opened),
            )


def load_dataset(conn: sqlite3.Connection, dataset_id: str) -> FrozenDataset:
    """Replay a frozen dataset by id — identical rows regardless of later ingestion."""
    drow = conn.execute(
        "SELECT manifest_json FROM dataset WHERE dataset_id = ?", (dataset_id,)
    ).fetchone()
    if drow is None:
        raise KeyError(f"no dataset {dataset_id!r}")
    manifest = json.loads(drow[0])
    train = [
        LabelRow(sid, uid, mid, tuple(json.loads(fj)), int(label), grp)
        for sid, mid, uid, fj, label, grp in conn.execute(
            "SELECT search_id, media_id, user_id, features_json, label, grp "
            "FROM dataset_row WHERE dataset_id = ? ORDER BY search_id, media_id",
            (dataset_id,),
        )
    ]
    eval_cands = [
        EvalCand(sid, mid, float(hs), json.loads(fj), bool(pos))
        for sid, mid, hs, fj, pos in conn.execute(
            "SELECT search_id, media_id, heuristic_score, features_json, is_positive "
            "FROM dataset_eval WHERE dataset_id = ? ORDER BY search_id, media_id",
            (dataset_id,),
        )
    ]
    return FrozenDataset(
        dataset_id,
        int(manifest["seed"]),
        float(manifest["eval_frac"]),
        int(manifest["k"]),
        train,
        eval_cands,
    )


def _grouped(cands: list[EvalCand]) -> dict[str, list[EvalCand]]:
    by_search: dict[str, list[EvalCand]] = defaultdict(list)
    for c in cands:
        by_search[c.search_id].append(c)
    return by_search


def baseline_eval_queries(cands: list[EvalCand]) -> list[Query]:
    """Eval queries ranked by the logged `heuristic_score` (the baseline order)."""
    out = []
    for sid, items in _grouped(cands).items():
        order = [c.media_id for c in sorted(items, key=lambda c: (-c.heuristic_score, c.media_id))]
        positives = {c.media_id for c in items if c.is_positive}
        out.append(Query(sid, order, positives))
    return out


def model_eval_queries(cands: list[EvalCand], model: LogRegModel) -> list[Query]:
    """Eval queries ranked by the model's score over the frozen feature dicts."""
    out = []
    for sid, items in _grouped(cands).items():
        order = [
            c.media_id for c in sorted(items, key=lambda c: (-model.score(c.features), c.media_id))
        ]
        positives = {c.media_id for c in items if c.is_positive}
        out.append(Query(sid, order, positives))
    return out
