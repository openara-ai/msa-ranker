"""Training pipeline (spec 05, FR-8/10, ADR-011) — manual, reproducible.

freeze → **baseline (INV-1, before any model)** → train → eval → register. Registers
every model with its `beats_baseline` flag; deployment is the separate gated step.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from .config import DEFAULT_EVAL_FRAC, DEFAULT_K, DEFAULT_MIN_LABELED_SEARCHES
from .datasets import (
    baseline_eval_queries,
    freeze_dataset,
    load_dataset,
    model_eval_queries,
)
from .eval import evaluate
from .ids import ulid
from .model import train_logreg
from .registry import record_eval, register


def train(conn: sqlite3.Connection, dataset_id: str, *, algo: str = "logreg", seed: int) -> Any:
    """Train a model over a frozen dataset's TRAIN split → a fitted model (spec 05)."""
    if algo != "logreg":
        raise ValueError(f"unsupported algo {algo!r} (v1: logreg only — ADR-002)")
    ds = load_dataset(conn, dataset_id)
    return train_logreg([r.features for r in ds.train], [r.label for r in ds.train], seed=seed)


def run_training(
    conn: sqlite3.Connection,
    *,
    out_dir: str | Path,
    algo: str = "logreg",
    k: int = DEFAULT_K,
    seed: int = 0,
    eval_frac: float = DEFAULT_EVAL_FRAC,
    min_searches: int = DEFAULT_MIN_LABELED_SEARCHES,
    note: str | None = None,
) -> dict[str, Any]:
    """The full manual pipeline. Raises `InsufficientDataError` below the min-data gate.

    Order matters: the **baseline is measured and recorded before the model exists**
    (INV-1), so the bar is set first and can't be retrofitted to the model.
    """
    # Validate the cutoff BEFORE freezing/persisting anything — k<1 yields meaningless
    # all-zero metrics (k=0) or a div-by-zero in ndcg_at_k (k<0), and we must not leave a
    # committed dataset behind for a bad input.
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    out_dir = Path(out_dir)
    dataset_id = freeze_dataset(
        conn, seed=seed, eval_frac=eval_frac, k=k, min_searches=min_searches, note=note
    )
    ds = load_dataset(conn, dataset_id)

    # INV-1 — baseline first, on the held-out eval candidates.
    baseline = evaluate(baseline_eval_queries(ds.eval_cands), k)
    with conn:
        record_eval(conn, dataset_id=dataset_id, result=baseline, model_id=None, is_baseline=True)

    model = train(conn, dataset_id, algo=algo, seed=seed)
    model_id = ulid()
    artifact_path = out_dir / f"{model_id}.model.json"
    model.save(artifact_path)

    model_eval = evaluate(model_eval_queries(ds.eval_cands, model), k)
    manifest = register(
        conn,
        model_id=model_id,
        artifact_path=artifact_path,
        dataset_id=dataset_id,
        model_eval=model_eval,
        baseline=baseline,
        algo=algo,
        params=model.params,
        feature_set_version=model.feature_set_version,
    )
    return {
        "dataset_id": dataset_id,
        "model_id": model_id,
        "artifact": str(artifact_path),
        "k": k,
        "baseline_ndcg": baseline.ndcg,
        "model_ndcg": model_eval.ndcg,
        "beats_baseline": manifest["beats_baseline"],
        "n_eval_queries": model_eval.n_queries,
    }
