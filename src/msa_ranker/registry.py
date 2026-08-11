"""Model registry (spec 05, FR-9) + eval persistence.

Registers **every** evaluated model — pass or fail (`beats_baseline` recorded both
ways, NN3) — and writes the sidecar **manifest** the serving gate reads (architecture
§10c). Deployment (copying artifact+manifest to MSA's `ltr_model_dir`) is a separate,
gated step (spec 06): only `beats_baseline` models serve.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

from .eval import EvalResult
from .ids import ulid


def record_eval(
    conn: sqlite3.Connection,
    *,
    dataset_id: str,
    result: EvalResult,
    model_id: str | None = None,
    is_baseline: bool = False,
) -> None:
    """Persist NDCG@k + MRR rows into `eval` (model_id is NULL for the baseline)."""
    created = time.time()
    conn.executemany(
        "INSERT INTO eval (eval_id, model_id, dataset_id, metric, k, value, is_baseline, "
        "created_ts) VALUES (?,?,?,?,?,?,?,?)",
        [
            (
                ulid(),
                model_id,
                dataset_id,
                "ndcg",
                result.k,
                result.ndcg,
                int(is_baseline),
                created,
            ),
            (ulid(), model_id, dataset_id, "mrr", result.k, result.mrr, int(is_baseline), created),
        ],
    )


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
    os.replace(tmp, path)  # atomic — MSA never reads a half-written manifest (§11)


def register(
    conn: sqlite3.Connection,
    *,
    model_id: str,
    artifact_path: str | Path,
    dataset_id: str,
    model_eval: EvalResult,
    baseline: EvalResult,
    algo: str,
    params: dict[str, Any],
    feature_set_version: str,
) -> dict[str, Any]:
    """Write the `model` + model `eval` rows and the sidecar manifest. Returns the manifest.

    `beats_baseline` is `model.ndcg@k > baseline.ndcg@k` on the same held-out split —
    exactly what the serving gate (spec 06) reads.
    """
    # Store an ABSOLUTE artifact path so `deploy` (possibly run from a different cwd)
    # resolves it correctly — a relative `--out` would otherwise be re-resolved against
    # the deploy process's working directory.
    artifact_path = Path(artifact_path).resolve()
    beats = model_eval.ndcg > baseline.ndcg
    created = time.time()
    sha = _sha256(artifact_path)

    manifest = {
        "model_id": model_id,
        "algo": algo,
        "params": params,
        "dataset_id": dataset_id,
        "feature_set_version": feature_set_version,
        "trained_ts": created,
        "artifact": artifact_path.name,  # filename the serving loader resolves (spec 06)
        "artifact_sha": sha,
        "eval": {f"ndcg@{model_eval.k}": model_eval.ndcg, "mrr": model_eval.mrr},
        "baseline": {f"ndcg@{baseline.k}": baseline.ndcg},
        "beats_baseline": beats,
    }
    # Write the manifest BEFORE committing the DB rows, so the DB is the single commit
    # point: a failed manifest write leaves no `model` row (vs. a registered model whose
    # serving manifest is missing). A stray manifest with no DB row is harmless (not yet
    # deployed). The artifact itself was already written atomically by the caller.
    _atomic_write_json(artifact_path.with_name(f"{model_id}.manifest.json"), manifest)
    with conn:
        conn.execute(
            "INSERT INTO model (model_id, created_ts, artifact_path, artifact_sha, algo, "
            "params_json, feature_set_version, beats_baseline, dataset_id) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (
                model_id,
                created,
                str(artifact_path),
                sha,
                algo,
                json.dumps(params),
                feature_set_version,
                int(beats),
                dataset_id,
            ),
        )
        record_eval(conn, dataset_id=dataset_id, result=model_eval, model_id=model_id)
    return manifest
