"""Graded eval harness: NDCG@k (primary) + MRR (secondary), and the measured
baseline (FR-1/FR-2, ADR-004, INV-1).

The metric functions are pure and deterministic — golden unit-tested (CI). Only the
graded *model-vs-baseline runs* over a dataset are opt-in/local (spec 04). Eval ranks
the **full shown candidate set**; relevance = the opens (binary v1), independent of the
training-label drop (spec 03). The baseline ranks by the logged `heuristic_score` (NN1),
so it is computable on any traffic — including learned-served searches.

NOTE: persisting the baseline to an `eval` row (with a `dataset_id`, `is_baseline=1`)
lands in S-4, once frozen datasets exist (the `eval` table FKs to `dataset`). S-2
delivers the harness + the baseline computation.
"""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass


@dataclass
class Query:
    """One held-out query under eval: a ranked order + the opened (relevant) media.

    Plain (not frozen): it holds mutable `list`/`set` fields, so `frozen=True` would be
    misleading (it blocks reassignment, not mutation) and make it unhashable.
    """

    search_id: str
    order: list[str]  # media_ids in the order produced by the scorer under eval
    positives: set[str]  # opened media_ids (relevance = 1)


@dataclass(frozen=True)
class EvalResult:
    ndcg: float
    mrr: float
    k: int
    n_queries: int


def _dcg(order: list[str], positives: set[str], k: int) -> float:
    # 0-based rank i → discount 1/log2(i + 2); binary gain.
    return sum(1.0 / math.log2(i + 2) for i, m in enumerate(order[:k]) if m in positives)


def ndcg_at_k(order: list[str], positives: set[str], k: int) -> float:
    """Normalized DCG@k with binary relevance. 0.0 if no relevant item is present."""
    n_present = sum(1 for m in order if m in positives)
    n_ideal = min(n_present, k)
    if n_ideal == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_ideal))
    return _dcg(order, positives, k) / idcg


def mrr(order: list[str], positives: set[str], k: int) -> float:
    """Reciprocal rank of the first relevant item in the top-k; 0.0 if none."""
    for i, m in enumerate(order[:k]):
        if m in positives:
            return 1.0 / (i + 1)
    return 0.0


def evaluate(queries: list[Query], k: int) -> EvalResult:
    """Macro-average NDCG@k and MRR over queries (the mean of per-query scores)."""
    n = len(queries)
    if n == 0:
        return EvalResult(0.0, 0.0, k, 0)
    ndcg = sum(ndcg_at_k(q.order, q.positives, k) for q in queries) / n
    mean_rr = sum(mrr(q.order, q.positives, k) for q in queries) / n
    return EvalResult(ndcg, mean_rr, k, n)


def baseline_queries(conn: sqlite3.Connection, search_ids: list[str] | None = None) -> list[Query]:
    """Build queries ranked by the logged `heuristic_score` (the baseline ordering).

    Ranks the **full** shown candidate set per search; relevance = opens. Ties broken
    deterministically by `(-heuristic_score, media_id)`.
    """
    if search_ids is None:
        search_ids = [r[0] for r in conn.execute("SELECT search_id FROM search")]
    queries: list[Query] = []
    for sid in search_ids:
        # Index access (row[0]=media_id, row[1]=heuristic_score) — no dependence on
        # conn.row_factory being sqlite3.Row.
        shown = conn.execute(
            "SELECT media_id, heuristic_score FROM result_shown WHERE search_id = ?",
            (sid,),
        ).fetchall()
        if not shown:
            continue
        order = [row[0] for row in sorted(shown, key=lambda row: (-row[1], row[0]))]
        positives = {
            r[0]
            for r in conn.execute(
                "SELECT media_id FROM interaction WHERE search_id = ? AND action = 'open'",
                (sid,),
            )
        }
        queries.append(Query(sid, order, positives))
    return queries


def measure_baseline(
    conn: sqlite3.Connection, k: int, search_ids: list[str] | None = None
) -> EvalResult:
    """Measure MSA's current heuristic ordering — the baseline to beat (INV-1)."""
    return evaluate(baseline_queries(conn, search_ids), k)
