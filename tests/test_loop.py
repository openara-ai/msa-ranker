"""S-5.3 — loop-1 end-to-end validation (M-1 exit evidence) + deploy gate + report.

Exercises the whole chain on a hermetic fixture:
    ledger → ingest → labels → freeze → baseline → train → eval → register → DEPLOY
           → gate_ok → Ranker.load → score
plus the gated-deploy refusal and the registry report. Model *quality* is never asserted
(INV-2); the fixture is trivially separable by construction so the gated path is reachable.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from msa_ranker.deploy import NotDeployableError, deploy
from msa_ranker.eval import EvalResult
from msa_ranker.features import QueryContext
from msa_ranker.ingest import ingest
from msa_ranker.ledger import LedgerWriter
from msa_ranker.registry import register
from msa_ranker.report import build_report, render_report
from msa_ranker.serving import Ranker, gate_ok
from msa_ranker.train import run_training


def _log_search(w, sid, token, *, opened_pos, n=4):
    """Write one search's events: heuristic_score ranks by position; the opened item gets
    a high `sim` so a model that learns `sim` beats the position-based heuristic."""
    w.append_search(
        search_id=sid,
        user_id="u1",
        query=token,
        ctx={"visual_tokens": [token]},
        flag_on=False,
        model_version=None,
        k=n,
    )
    w.append_shown(
        search_id=sid,
        rows=[
            {
                "media_id": f"{sid}-m{p}",
                "position": p,
                "score": float(n - p),
                "heuristic_score": float(n - p),
                "features": {"sim": 1.0 if p == opened_pos else 0.1},
            }
            for p in range(n)
        ],
    )
    w.append_open(search_id=sid, media_id=f"{sid}-m{opened_pos}", user_id="u1")


def _seed_ledger(ledger_dir, count=4):
    w = LedgerWriter(ledger_dir)
    for i in range(count):
        _log_search(w, f"s{i}", f"tok{i}", opened_pos=(i % 3) + 1)


def test_loop_ledger_to_serving(db, tmp_path):
    # The full M-1 loop, end to end, on a hermetic fixture.
    ledger = tmp_path / "ledger"
    ledger.mkdir()
    _seed_ledger(ledger, 4)

    stats = ingest(db, ledger)  # ledger → SoR
    assert stats["inserted"] > 0

    res = run_training(db, out_dir=tmp_path / "models", k=4, seed=0, eval_frac=0.5, min_searches=2)
    # Pipeline connectivity: ingest → … → register actually produced a registered model.
    assert db.execute("SELECT COUNT(*) FROM model").fetchone()[0] >= 1
    assert res["model_id"] in render_report(db)  # and it shows in the report

    # Deploy → gate → load → score, ALWAYS exercised: register a deterministically
    # gate-valid model over the REAL trained artifact (model_eval > baseline by
    # construction, so this asserts the serving PATH, not the trained model's quality —
    # INV-2). The deploy gate's refusal path is covered by the next test.
    register(
        db,
        model_id="GATEVALID",
        artifact_path=res["artifact"],
        dataset_id=res["dataset_id"],
        model_eval=EvalResult(0.90, 0.9, 4, 1),
        baseline=EvalResult(0.50, 0.5, 4, 1),
        algo="logreg",
        params={},
        feature_set_version="v1",
    )
    dest = tmp_path / "ltr_model_dir"
    deploy(db, "GATEVALID", dest)  # no force — beats_baseline True by construction
    assert gate_ok(dest) is True
    ranker = Ranker.load(dest)  # the real trained artifact loads through the gate
    scores = ranker.score(
        [{"features": {"sim": 1.0}}, {"features": {"sim": 0.0}}], QueryContext(), now=0.0
    )
    assert len(scores) == 2  # reorder-only contract holds at serve time


def test_deploy_gate_refuses_non_beating(db, tmp_path):
    # A model that did NOT beat baseline is refused (the deploy gate) unless forced.
    _seed_ledger(tmp_path / "ledger", 4)
    ingest(db, tmp_path / "ledger")
    res = run_training(db, out_dir=tmp_path / "models", k=4, seed=0, eval_frac=0.5, min_searches=2)
    # Register a deliberately-losing model over the same dataset + artifact.
    artifact = res["artifact"]
    register(
        db,
        model_id="LOSER",
        artifact_path=artifact,
        dataset_id=res["dataset_id"],
        model_eval=EvalResult(0.10, 0.1, 4, 1),
        baseline=EvalResult(0.90, 0.9, 4, 1),
        algo="logreg",
        params={},
        feature_set_version="v1",
    )
    with pytest.raises(NotDeployableError, match="beat baseline"):
        deploy(db, "LOSER", tmp_path / "serve")
    deploy(db, "LOSER", tmp_path / "serve", force=True)  # force overrides the gate
    # gate_ok still closes on the deployed loser (manifest beats_baseline is false) → heuristic.
    assert gate_ok(tmp_path / "serve") is False


def test_deploy_missing_model_refuses(db, tmp_path):
    with pytest.raises(NotDeployableError, match="no model"):
        deploy(db, "NOPE", tmp_path / "serve")


def test_deploy_refuses_corrupt_manifest(db, tmp_path):
    # A non-object sidecar (valid JSON but not a dict) → clean refusal, not a traceback.
    _seed_ledger(tmp_path / "ledger", 4)
    ingest(db, tmp_path / "ledger")
    res = run_training(db, out_dir=tmp_path / "models", k=4, seed=0, eval_frac=0.5, min_searches=2)
    Path(res["artifact"]).with_name(f"{res['model_id']}.manifest.json").write_text("[]")
    with pytest.raises(NotDeployableError, match="not a JSON object"):
        deploy(db, res["model_id"], tmp_path / "serve", force=True)


def test_report_shape(db, tmp_path):
    _seed_ledger(tmp_path / "ledger", 4)
    ingest(db, tmp_path / "ledger")
    res = run_training(db, out_dir=tmp_path / "models", k=4, seed=0, eval_frac=0.5, min_searches=2)
    rep = build_report(db)
    assert rep["n_models"] == 1
    entry = rep["models"][0]
    assert entry["model_id"] == res["model_id"]
    assert entry["dataset_id"] == res["dataset_id"]
    assert isinstance(entry["beats_baseline"], bool)
    assert entry["ndcg"] is not None and entry["baseline_ndcg"] is not None
