"""In-process learned scorer at MSA's rerank seam (spec 06, ADR-001, FR-13/14).

On MSA's **correctness path** → fully deterministic and golden-tested (INV-2). Two pieces:

  - ``gate_ok(model_dir)`` — the cold-start gate (ADR-011): serve the learned model only
    if a manifest + artifact exist, the model **beats baseline**, its
    ``feature_set_version`` matches the running extractor, and the artifact's sha matches
    the manifest (integrity — reject a corrupt/tampered/half-written artifact).
  - ``Ranker`` — loads the model and ``score()``s a candidate batch as a pure function of
    the loaded model: one float per candidate **in input order** (INV-6 reorder-only — the
    seam sorts; the Ranker never adds/drops candidates and opens no write handle to
    ``media.sqlite``).

The flag (``enable_learning_to_rank``) and the fail-open fallback live in the MSA shim
(spec 06); this library is gate + scorer only.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .features import FEATURE_SET_VERSION, QueryContext, feature_dict
from .model import LogRegModel

MANIFEST_NAME = "manifest.json"  # the canonical deployed manifest in MSA's ltr_model_dir

__all__ = ["FEATURE_SET_VERSION", "Ranker", "gate_ok"]


def _read_manifest(model_dir: str | Path) -> dict[str, Any] | None:
    """Parse ``<model_dir>/manifest.json``; None if missing, unreadable, or not an object.

    A syntactically-valid but non-object manifest (``[]``, ``null``, a number) must close
    the gate, not crash a downstream ``.get()`` — so anything but a mapping returns None.
    """
    path = Path(model_dir) / MANIFEST_NAME
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _validated_artifact_bytes(
    model_dir: str | Path, manifest: dict[str, Any] | None
) -> bytes | None:
    """The artifact **bytes** if the (model_dir, manifest) pair passes the full gate, else None.

    This is the **single** source of gate truth, used by both ``gate_ok`` (→ bool) and
    ``Ranker.load`` (→ raises). It reads the artifact **once** and returns those exact
    bytes, so load parses the same bytes that were hashed — no TOCTOU window where the
    manifest *or* the artifact file is swapped between gate and load and an ungated /
    unhashed model is then served.
    """
    if not isinstance(manifest, dict):
        return None
    if manifest.get("beats_baseline") is not True:  # require literal JSON true, not "false"/1
        return None
    if manifest.get("feature_set_version") != FEATURE_SET_VERSION:
        return None
    name = manifest.get("artifact")
    # Require a plain basename — reject missing/None, absolute paths, and traversal.
    if not isinstance(name, str) or not name or name != Path(name).name:
        return None
    try:
        data = (Path(model_dir) / name).read_bytes()
    except OSError:  # missing / unreadable / is-a-directory
        return None
    if hashlib.sha256(data).hexdigest() != manifest.get("artifact_sha"):
        return None
    return data


def gate_ok(model_dir: str | Path) -> bool:
    """Return True only if the deployed model is safe to serve (the gate truth table).

    Flag-independent: the MSA shim ANDs this with the ``enable_learning_to_rank`` flag.
    Any failure (missing/corrupt manifest, beats_baseline not literally true, version
    skew, bad/missing artifact, sha mismatch) closes the gate → the heuristic serves (INV-3).
    """
    return _validated_artifact_bytes(model_dir, _read_manifest(model_dir)) is not None


@dataclass(frozen=True)
class Ranker:
    """A loaded model ready to score at the seam. Pure given the model (INV-2)."""

    model: LogRegModel
    feature_set_version: str

    @classmethod
    def load(cls, model_dir: str | Path) -> Ranker:
        """Load the deployed model, **re-running the full gate** on the loaded snapshot.

        ``load`` repeats every ``gate_ok`` check against the manifest it reads and parses
        the **exact artifact bytes that were hashed** — closing the TOCTOU window where the
        manifest or the artifact file is swapped between ``gate_ok`` and ``load``. **Raises**
        on any failure; startup is guarded by the shim, which fails open (AC-06.8).
        """
        data = _validated_artifact_bytes(model_dir, _read_manifest(model_dir))
        if data is None:
            raise ValueError(
                f"model in {model_dir} failed gate validation (missing/corrupt/ungated/tampered)"
            )
        model = LogRegModel.from_dict(json.loads(data))  # raises on version/layout/zero-std
        return cls(model=model, feature_set_version=model.feature_set_version)

    def score(
        self,
        candidates: Sequence[Mapping[str, Any]],
        ctx: QueryContext,
        *,
        now: float,
    ) -> list[float]:
        """One score per candidate, in **input order** (len == len(candidates), INV-6).

        Uses each candidate's precomputed ``features`` dict when present — the same vector
        computed and logged at the seam, so there is no train/serve skew — and otherwise
        computes it via the shared extractor. Deterministic given the model + inputs.

        **Raises** on a non-finite score (e.g. a NaN/inf feature emitted upstream) so the
        shim's exception fallback runs and the whole request serves the heuristic (FR-14)
        — a silent inf/NaN would otherwise destabilize the sort or break response JSON.
        """
        scores: list[float] = []
        for c in candidates:
            feats = c.get("features")
            # An *empty* dict is treated as absent (not an all-zero vector) — fall back to
            # the extractor so a `"features": {}` upstream can't silently mis-score.
            if not (isinstance(feats, Mapping) and feats):
                feats = feature_dict(c, ctx, now)
            s = self.model.score(feats)
            if not math.isfinite(s):
                raise ValueError("non-finite score (NaN/inf feature) — fall back to heuristic")
            scores.append(s)
        return scores
