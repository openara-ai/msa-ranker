"""S-4.2/4.3 — freeze / train / registry (spec 05): immutable dataset replay, the
min-data gate, end-to-end pipeline + lineage, beats_baseline, and reproducibility.

Model *quality* is never asserted (INV-2) — these check the plumbing is deterministic,
replayable, and records correct lineage + the gate flag.
"""

from __future__ import annotations

import json

import pytest

from msa_ranker.datasets import (
    InsufficientDataError,
    freeze_dataset,
    load_dataset,
)
from msa_ranker.eval import EvalResult
from msa_ranker.features import FEATURE_SET_VERSION
from msa_ranker.model import LogRegModel, train_logreg
from msa_ranker.registry import register
from msa_ranker.train import run_training, train


def _seed(conn, sid, token, *, opened_pos, n=4, user="u1"):
    """Insert one search with n shown candidates and an open at `opened_pos`.

    The opened candidate gets a high `sim`; the baseline `heuristic_score` ranks by
    position (pos 0 first), so a model that learns `sim` can differ from the baseline.
    """
    conn.execute(
        "INSERT INTO search (ev_id, search_id, user_id, query_text, query_ctx_json, "
        "flag_on, model_version, k, created_ts) VALUES (?,?,?,?,?,?,?,?,?)",
        (f"ev-{sid}", sid, user, token, json.dumps({"visual_tokens": [token]}), 0, None, n, 1.0),
    )
    for pos in range(n):
        mid = f"{sid}-m{pos}"
        feat = {"sim": 1.0 if pos == opened_pos else 0.1}
        conn.execute(
            "INSERT INTO result_shown (ev_id, search_id, media_id, position, score, "
            "heuristic_score, features_json) VALUES (?,?,?,?,?,?,?)",
            (f"ev-{sid}-{pos}", sid, mid, pos, 0.5, float(n - pos), json.dumps(feat)),
        )
    conn.execute(
        "INSERT INTO interaction (ev_id, search_id, media_id, user_id, action, dwell_ms, "
        "created_ts) VALUES (?,?,?,?,?,?,?)",
        (f"ev-open-{sid}", sid, f"{sid}-m{opened_pos}", user, "open", None, 2.0),
    )
    conn.commit()


def _seed_many(conn, count=4):
    for i in range(count):
        _seed(conn, f"s{i}", f"tok{i}", opened_pos=(i % 3) + 1)


# --------------------------------------------------------------------------- model
def test_train_logreg_is_deterministic():
    X = [[1.0] + [0.0] * 15, [0.1] + [0.0] * 15] * 4
    y = [1, 0] * 4
    a = train_logreg(X, y, seed=0)
    b = train_logreg(X, y, seed=0)
    assert a.weights == b.weights and a.bias == b.bias


def test_model_json_roundtrip(tmp_path):
    # save → load reproduces the model exactly (no quality assertion — INV-2).
    X = [[1.0] + [0.0] * 15, [0.1] + [0.0] * 15] * 6
    y = [1, 0] * 6
    model = train_logreg(X, y, seed=0)
    loaded = LogRegModel.load(model.save(tmp_path / "m.json"))
    assert loaded.weights == model.weights and loaded.bias == model.bias
    assert loaded.std == model.std and loaded.mean == model.mean


def test_load_rejects_incompatible_or_corrupt_artifact(tmp_path):
    # from_dict refuses a mismatched feature layout/version or a zero-std (corrupt) file —
    # otherwise score() would silently corrupt on the S-5 serve path.
    good = train_logreg([[1.0] + [0.0] * 15, [0.0] * 16], [1, 0], seed=0).to_dict()
    with pytest.raises(ValueError, match="feature_set_version"):
        LogRegModel.from_dict({**good, "feature_set_version": "v999"})
    with pytest.raises(ValueError, match="feature_names"):
        LogRegModel.from_dict({**good, "feature_names": ["x"]})
    zeroed_std = [0.0] + good["standardize"]["std"][1:]
    bad_std = {**good, "standardize": {**good["standardize"], "std": zeroed_std}}
    with pytest.raises(ValueError, match="zero std"):
        LogRegModel.from_dict(bad_std)
    # A non-finite param (inf/NaN) must be refused — score() would propagate it without
    # raising, defeating the shim's exception fallback on the serve path.
    inf_weights = [float("inf")] + good["weights"][1:]
    with pytest.raises(ValueError, match="non-finite"):
        LogRegModel.from_dict({**good, "weights": inf_weights})
    with pytest.raises(ValueError, match="non-finite"):
        LogRegModel.from_dict({**good, "bias": float("nan")})
    # A wrong vector length must be rejected at load (else score()'s zip(strict=True) would
    # raise on every request, repeatedly disabling learned scoring).
    with pytest.raises(ValueError, match="vector lengths"):
        LogRegModel.from_dict({**good, "weights": good["weights"][:-1]})
    short = good["standardize"]["std"][:-1]
    short_std = {**good, "standardize": {**good["standardize"], "std": short}}
    with pytest.raises(ValueError, match="vector lengths"):
        LogRegModel.from_dict(short_std)


# --------------------------------------------------------------------------- datasets
def test_freeze_replay_is_immutable(db):
    # AC-05.1 — a frozen dataset replays to an identical row set, even after MORE events
    # land in the SoR afterward.
    _seed_many(db, 4)
    dsid = freeze_dataset(db, seed=1, eval_frac=0.5, k=4, min_searches=2)
    first = load_dataset(db, dsid)
    # A late open arrives after the freeze — must NOT change the frozen dataset.
    db.execute(
        "INSERT INTO interaction (ev_id, search_id, media_id, user_id, action, dwell_ms, "
        "created_ts) VALUES (?,?,?,?,?,?,?)",
        ("late-open", "s0", "s0-m3", "u1", "open", None, 9.0),
    )
    db.commit()
    second = load_dataset(db, dsid)
    assert first.train == second.train
    assert first.eval_cands == second.eval_cands


def test_min_data_gate_refuses_and_writes_nothing(db):
    # AC-05.4 — below the threshold, freeze refuses and no dataset row is written.
    _seed(db, "only", "tok", opened_pos=1)
    with pytest.raises(InsufficientDataError):
        freeze_dataset(db, seed=1, eval_frac=0.5, min_searches=5)
    assert db.execute("SELECT COUNT(*) FROM dataset").fetchone()[0] == 0
    assert db.execute("SELECT COUNT(*) FROM dataset_row").fetchone()[0] == 0


def test_single_group_refused_cleanly(db):
    # Enough labelled searches but only ONE query group (same intent) → can't split
    # leak-free → InsufficientDataError (not a raw ValueError stack trace).
    _seed(db, "a", "same", opened_pos=1)
    _seed(db, "b", "same", opened_pos=2)  # same token → same group
    with pytest.raises(InsufficientDataError, match="group"):
        freeze_dataset(db, seed=1, eval_frac=0.5, min_searches=2)


def test_manifest_records_lineage(db):
    # AC-05.5 — the dataset manifest records seed / eval_frac / k / feature_set_version.
    _seed_many(db, 4)
    dsid = freeze_dataset(db, seed=7, eval_frac=0.5, k=4, min_searches=2)
    manifest = json.loads(
        db.execute("SELECT manifest_json FROM dataset WHERE dataset_id=?", (dsid,)).fetchone()[0]
    )
    assert manifest["seed"] == 7 and manifest["eval_frac"] == 0.5 and manifest["k"] == 4
    assert manifest["feature_set_version"] == FEATURE_SET_VERSION


# --------------------------------------------------------------------------- registry
def test_register_computes_beats_baseline_both_ways(db, tmp_path):
    # AC-05.3 — beats_baseline = model.ndcg > baseline.ndcg, recorded pass OR fail.
    _seed_many(db, 4)
    dsid = freeze_dataset(db, seed=1, eval_frac=0.5, k=4, min_searches=2)
    artifact = tmp_path / "m.model.json"
    train(db, dsid, seed=0).save(artifact)
    win = register(
        db,
        model_id="MWIN",
        artifact_path=artifact,
        dataset_id=dsid,
        model_eval=EvalResult(0.80, 0.7, 4, 2),
        baseline=EvalResult(0.50, 0.4, 4, 2),
        algo="logreg",
        params={},
        feature_set_version=FEATURE_SET_VERSION,
    )
    lose = register(
        db,
        model_id="MLOSE",
        artifact_path=artifact,
        dataset_id=dsid,
        model_eval=EvalResult(0.40, 0.3, 4, 2),
        baseline=EvalResult(0.50, 0.4, 4, 2),
        algo="logreg",
        params={},
        feature_set_version=FEATURE_SET_VERSION,
    )
    assert win["beats_baseline"] is True and lose["beats_baseline"] is False
    flags = dict(db.execute("SELECT model_id, beats_baseline FROM model").fetchall())
    assert flags["MWIN"] == 1 and flags["MLOSE"] == 0  # both registered (NN3)


# --------------------------------------------------------------------------- pipeline
def test_end_to_end_pipeline(db, tmp_path):
    # AC-05.2 / AC-05.5 — pipeline → loadable artifact + valid manifest + correct lineage.
    _seed_many(db, 4)
    res = run_training(db, out_dir=tmp_path, k=4, seed=0, eval_frac=0.5, min_searches=2)

    model = LogRegModel.load(res["artifact"])  # artifact loads
    assert model.feature_set_version == FEATURE_SET_VERSION

    manifest = json.loads((tmp_path / f"{res['model_id']}.manifest.json").read_text())
    assert manifest["dataset_id"] == res["dataset_id"]  # lineage
    assert manifest["model_id"] == res["model_id"]
    assert isinstance(manifest["beats_baseline"], bool)

    # eval rows: a baseline row (is_baseline=1, model_id NULL) + model rows (is_baseline=0).
    rows = db.execute(
        "SELECT is_baseline, model_id, metric FROM eval WHERE dataset_id=?", (res["dataset_id"],)
    ).fetchall()
    assert any(r[0] == 1 and r[1] is None for r in rows)  # INV-1 baseline persisted
    assert any(r[0] == 0 and r[1] == res["model_id"] for r in rows)
    model_row = db.execute(
        "SELECT dataset_id, feature_set_version FROM model WHERE model_id=?", (res["model_id"],)
    ).fetchone()
    assert model_row[0] == res["dataset_id"] and model_row[1] == FEATURE_SET_VERSION


def test_run_training_rejects_bad_k_without_side_effects(db, tmp_path):
    # k < 1 is rejected before any freeze/persist — no dataset/model left behind.
    _seed_many(db, 4)
    for bad_k in (0, -1):
        with pytest.raises(ValueError, match="k must be"):
            run_training(db, out_dir=tmp_path, k=bad_k, eval_frac=0.5, min_searches=2)
    assert db.execute("SELECT COUNT(*) FROM dataset").fetchone()[0] == 0
    assert db.execute("SELECT COUNT(*) FROM model").fetchone()[0] == 0


def test_retrain_from_dataset_is_reproducible(db):
    # AC-05.6 — same dataset_id + seed → identical model and eval.
    _seed_many(db, 4)
    dsid = freeze_dataset(db, seed=2, eval_frac=0.5, k=4, min_searches=2)
    m1 = train(db, dsid, seed=0)
    m2 = train(db, dsid, seed=0)
    assert m1.weights == m2.weights and m1.bias == m2.bias
