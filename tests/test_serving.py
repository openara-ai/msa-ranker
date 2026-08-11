"""S-5.1 — serving library (spec 06): the cold-start gate truth table, load-raises
behaviour, and deterministic reorder-only scoring (INV-2/INV-6).

Quality is never asserted (INV-2) — these check the gate decisions and the scoring
plumbing (length, input order, determinism, feature reuse), not which item ranks higher.
The flag-off byte-identical equality + HTTP fallback + no-write tests live MSA-side (S-5.2).
"""

from __future__ import annotations

import hashlib
import json

import pytest

from msa_ranker.features import FEATURE_SET_VERSION, QueryContext, feature_dict
from msa_ranker.model import LogRegModel, train_logreg
from msa_ranker.serving import Ranker, gate_ok

CTX = QueryContext(visual_tokens=["dog"])


def _deploy(model_dir, *, beats=True, fsv=FEATURE_SET_VERSION, sha_ok=True, artifact="model.json"):
    """Write a deployed model dir: an artifact + a canonical manifest.json."""
    model_dir.mkdir(parents=True, exist_ok=True)
    model = train_logreg([[1.0] + [0.0] * 15, [0.0] * 16], [1, 0], seed=0)
    path = model.save(model_dir / artifact)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "model_id": "m1",
        "artifact": artifact,
        "feature_set_version": fsv,
        "artifact_sha": sha if sha_ok else "0" * 64,
        "beats_baseline": beats,
    }
    (model_dir / "manifest.json").write_text(json.dumps(manifest))
    return model_dir


# ------------------------------------------------------------------- gate truth table
def test_gate_serves_only_on_all_conditions(tmp_path):
    # AC-06.2 / 06.7 / 06.9 — the one row that serves: beats + version match + sha match.
    assert gate_ok(_deploy(tmp_path / "ok")) is True


def test_gate_closed_cases(tmp_path):
    assert gate_ok(tmp_path / "missing") is False  # no manifest at all
    assert gate_ok(_deploy(tmp_path / "lose", beats=False)) is False  # AC-06.2
    assert gate_ok(_deploy(tmp_path / "skew", fsv="v999")) is False  # AC-06.7 version skew
    assert gate_ok(_deploy(tmp_path / "tamper", sha_ok=False)) is False  # AC-06.9 sha mismatch
    # beats_baseline must be the literal JSON true — a truthy string "false" must NOT serve.
    strd = _deploy(tmp_path / "strbeat")
    m = json.loads((strd / "manifest.json").read_text())
    m["beats_baseline"] = "false"
    (strd / "manifest.json").write_text(json.dumps(m))
    assert gate_ok(strd) is False


def test_gate_closed_on_missing_artifact(tmp_path):
    d = _deploy(tmp_path / "noart")
    (d / "model.json").unlink()  # manifest references an artifact that isn't there
    assert gate_ok(d) is False


def test_gate_closed_on_non_object_or_pathy_manifest(tmp_path):
    # A syntactically-valid but non-object manifest must close the gate, not crash.
    bad = tmp_path / "nonobj"
    bad.mkdir()
    (bad / "manifest.json").write_text("[]")
    assert gate_ok(bad) is False
    (bad / "manifest.json").write_text("null")
    assert gate_ok(bad) is False
    # An artifact that isn't a plain basename (traversal / absolute) is rejected.
    d = _deploy(tmp_path / "trav")
    m = json.loads((d / "manifest.json").read_text())
    m["artifact"] = "../model.json"
    (d / "manifest.json").write_text(json.dumps(m))
    assert gate_ok(d) is False


# ------------------------------------------------------------------- load
def test_load_round_trips_a_deployed_model(tmp_path):
    ranker = Ranker.load(_deploy(tmp_path / "ok"))
    assert ranker.feature_set_version == FEATURE_SET_VERSION
    assert isinstance(ranker.model, LogRegModel)


def test_load_raises_on_missing_manifest(tmp_path):
    # AC-06.8 — load is free to raise; the shim guards startup and fails open.
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError):
        Ranker.load(tmp_path / "empty")


def test_load_revalidates_gate_no_toctou(tmp_path):
    # load re-runs the full gate on the snapshot it reads — a model that fails the gate
    # (here: beats_baseline false, as if a manifest were swapped after gate_ok) is refused.
    d = _deploy(tmp_path / "ungated", beats=False)
    assert gate_ok(d) is False
    with pytest.raises(ValueError):
        Ranker.load(d)


def test_load_raises_on_incompatible_artifact(tmp_path):
    # An artifact trained for another feature version, with a manifest sha that MATCHES it
    # so the gate passes — load then reaches LogRegModel.from_dict, which rejects the
    # version skew. (Re-pointing the sha is what forces the from_dict path rather than the
    # earlier sha-mismatch gate path.)
    d = _deploy(tmp_path / "bad")
    artifact = d / "model.json"
    blob = json.loads(artifact.read_text())
    blob["feature_set_version"] = "v999"
    artifact.write_text(json.dumps(blob))
    m = json.loads((d / "manifest.json").read_text())
    m["artifact_sha"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    (d / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(ValueError, match="feature_set_version"):
        Ranker.load(d)


# ------------------------------------------------------------------- scoring
def test_score_is_length_preserving_ordered_and_deterministic(tmp_path):
    # AC-06.4 (reorder-only) — one score per candidate, in input order, no add/drop.
    ranker = Ranker.load(_deploy(tmp_path / "ok"))
    cands = [{"features": {"sim": 1.0}}, {"features": {"sim": 0.0}}, {"features": {"sim": 0.5}}]
    scores = ranker.score(cands, CTX, now=0.0)
    assert len(scores) == len(cands)  # no add/drop
    assert scores == [ranker.model.score(c["features"]) for c in cands]  # positional, not quality
    assert ranker.score(cands, CTX, now=0.0) == scores  # deterministic (INV-2)


def test_score_uses_precomputed_features_then_falls_back(tmp_path):
    ranker = Ranker.load(_deploy(tmp_path / "ok"))
    # precomputed `features` is used verbatim (no train/serve skew) — even if raw fields differ
    cand = {"raw_similarity_score": 0.9, "features": {"sim": 0.1}}
    assert ranker.score([cand], CTX, now=0.0)[0] == ranker.model.score({"sim": 0.1})
    # absent `features` → computed via the shared extractor at the seam
    raw = {"raw_similarity_score": 0.9}
    assert ranker.score([raw], CTX, now=0.0)[0] == ranker.model.score(feature_dict(raw, CTX, 0.0))


def test_score_treats_empty_features_dict_as_absent(tmp_path):
    # An empty `features` dict must fall back to the extractor (not score an all-zero vector).
    ranker = Ranker.load(_deploy(tmp_path / "ok"))
    raw = {"raw_similarity_score": 0.9}
    cand_empty = {"raw_similarity_score": 0.9, "features": {}}
    assert ranker.score([cand_empty], CTX, now=0.0) == ranker.score([raw], CTX, now=0.0)


def test_score_raises_on_non_finite_so_shim_falls_back(tmp_path):
    # A NaN/inf feature → non-finite score → raise, so the shim serves the heuristic (FR-14)
    # instead of emitting an inf/NaN that destabilizes the sort or breaks response JSON.
    ranker = Ranker.load(_deploy(tmp_path / "ok"))
    with pytest.raises(ValueError, match="non-finite"):
        ranker.score([{"features": {"sim": float("inf")}}], CTX, now=0.0)
