"""Deploy a trained model to MSA's serving dir (spec 05 §gated-deploy, spec 06 §layout,
architecture §11 handoff).

Registration records *every* model; **deployment is the separate, gated step** — only a
`beats_baseline` model is copied into `ltr_model_dir`, written temp-then-rename (atomic)
so MSA never reads a half-written model. The deployed layout the serving gate expects:

    <ltr_model_dir>/
      manifest.json   ← the model's sidecar, renamed to the canonical name
      <artifact>      ← the artifact, copied under its registered basename
                        (whatever `model.artifact_path.name` is; named in manifest["artifact"])

Writing the artifact first and the manifest last makes the manifest the commit point: a
partial copy leaves no readable manifest, so the gate stays closed (heuristic). All file
errors are surfaced as NotDeployableError so the CLI reports a clean refusal.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
from pathlib import Path

from .serving import MANIFEST_NAME


class NotDeployableError(RuntimeError):
    """Raised when a model is missing or did not beat baseline (the deploy gate)."""


def _atomic_copy(src: Path, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dest)


def _atomic_write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
    os.replace(tmp, path)


def deploy(
    conn: sqlite3.Connection,
    model_id: str,
    dest_dir: str | Path,
    *,
    force: bool = False,
) -> Path:
    """Copy `model_id`'s artifact + manifest into `dest_dir` (the serving `ltr_model_dir`).

    Refuses a model that did not beat baseline unless `force=True` (the gate, spec 05).
    Returns the destination dir. Idempotent: re-deploying overwrites atomically.
    """
    row = conn.execute(
        "SELECT artifact_path, beats_baseline FROM model WHERE model_id = ?", (model_id,)
    ).fetchone()
    if row is None:
        raise NotDeployableError(f"no model {model_id!r} in the registry")
    artifact_path = Path(row[0])
    beats_baseline = bool(row[1])
    if not beats_baseline and not force:
        raise NotDeployableError(
            f"{model_id} did not beat baseline; refusing to deploy (pass force=True to override)"
        )
    if not artifact_path.is_file():
        raise NotDeployableError(f"artifact for {model_id!r} is missing at {artifact_path}")

    manifest_src = artifact_path.with_name(f"{model_id}.manifest.json")
    try:
        manifest = json.loads(manifest_src.read_text())
    except (OSError, ValueError) as exc:  # missing / unreadable / invalid JSON
        raise NotDeployableError(
            f"cannot read manifest for {model_id!r} at {manifest_src}: {exc}"
        ) from exc
    if not isinstance(manifest, dict):  # valid JSON but not an object (e.g. [] / null)
        raise NotDeployableError(
            f"manifest for {model_id!r} at {manifest_src} is not a JSON object"
        )
    # The deployed manifest must name the artifact by the basename it will sit beside.
    manifest["artifact"] = artifact_path.name

    dest = Path(dest_dir)
    try:
        dest.mkdir(parents=True, exist_ok=True)
        _atomic_copy(artifact_path, dest / artifact_path.name)  # artifact first …
        _atomic_write_json(dest / MANIFEST_NAME, manifest)  # … manifest last = commit point
    except OSError as exc:  # surface as a clean refusal, not a raw traceback
        raise NotDeployableError(f"failed to write deploy to {dest}: {exc}") from exc
    return dest
