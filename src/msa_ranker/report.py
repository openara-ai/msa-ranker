"""Experiment report (FR-17, C11) — the model registry + eval lineage.

Lists **every** registered model (pass *and* fail — FR-9 records all, FR-17 shows failed
experiments too) with its NDCG@k vs the recorded baseline on the same dataset, the delta,
and the `beats_baseline` gate flag. Pure read of the SoR; quality is reported, never
asserted (INV-2).
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any


def _model_metrics(conn: sqlite3.Connection, model_id: str) -> dict[str, tuple]:
    """{metric: (value, k)} for a model's own eval rows."""
    return {
        metric: (value, k)
        for metric, value, k in conn.execute(
            "SELECT metric, value, k FROM eval WHERE model_id = ? AND is_baseline = 0", (model_id,)
        )
    }


def _baseline_metrics(conn: sqlite3.Connection, dataset_id: str) -> dict[str, tuple]:
    """{metric: (value, k)} for the dataset's recorded baseline (is_baseline=1, model NULL)."""
    return {
        metric: (value, k)
        for metric, value, k in conn.execute(
            "SELECT metric, value, k FROM eval WHERE dataset_id = ? AND is_baseline = 1",
            (dataset_id,),
        )
    }


def build_report(conn: sqlite3.Connection) -> dict[str, Any]:
    """Structured report (devdash/JSON-friendly): one entry per registered model."""
    models = conn.execute(
        "SELECT model_id, created_ts, algo, feature_set_version, beats_baseline, dataset_id "
        "FROM model ORDER BY created_ts DESC"
    ).fetchall()
    entries = []
    for model_id, created_ts, algo, fsv, beats, dataset_id in models:
        m = _model_metrics(conn, model_id)
        b = _baseline_metrics(conn, dataset_id)
        m_ndcg, k = m.get("ndcg", (None, None))
        b_ndcg, _ = b.get("ndcg", (None, None))
        entries.append(
            {
                "model_id": model_id,
                "algo": algo,
                "dataset_id": dataset_id,
                "feature_set_version": fsv,
                "k": k,
                "ndcg": m_ndcg,
                "baseline_ndcg": b_ndcg,
                "delta_ndcg": (
                    (m_ndcg - b_ndcg) if (m_ndcg is not None and b_ndcg is not None) else None
                ),
                "mrr": m.get("mrr", (None, None))[0],
                "beats_baseline": bool(beats),
                "created_ts": created_ts,
            }
        )
    n_pass = sum(1 for e in entries if e["beats_baseline"])
    return {"n_models": len(entries), "n_beats_baseline": n_pass, "models": entries}


def _fmt(x: float | None) -> str:
    return "—" if x is None else f"{x:.4f}"


def render_report(conn: sqlite3.Connection) -> str:
    """Human-readable table of the registry (the `msa-ranker report` output)."""
    rep = build_report(conn)
    if not rep["models"]:
        return "No models registered yet. Run `msa-ranker train` first."
    lines = [
        f"Models: {rep['n_models']} registered · {rep['n_beats_baseline']} beat baseline",
        "",
        f"{'model_id':26} {'algo':8} {'k':>3} {'ndcg':>8} {'baseline':>9} "
        f"{'Δndcg':>8} {'mrr':>8}  gate",
        "-" * 90,
    ]
    for e in rep["models"]:
        gate = "PASS" if e["beats_baseline"] else "fail"
        lines.append(
            f"{e['model_id']:26} {e['algo']:8} {str(e['k'] or '—'):>3} "
            f"{_fmt(e['ndcg']):>8} {_fmt(e['baseline_ndcg']):>9} {_fmt(e['delta_ndcg']):>8} "
            f"{_fmt(e['mrr']):>8}  {gate}"
        )
    return "\n".join(lines)


def report_json(conn: sqlite3.Connection) -> str:
    return json.dumps(build_report(conn), indent=2, sort_keys=True)
