"""Pure-Python logistic regression — the v1 model (ADR-002, spec 05).

Dependency-free on purpose: the artifact is a small portable JSON (weights + the
train-time standardization), and serving (spec 06) scores with a dot product — no ML
runtime on MSA's path. Deterministic given the data (full-batch gradient descent from a
zero init), so re-training the same frozen dataset reproduces the model (AC-05.6).

The model is *out of the correctness path* (INV-2) — quality is graded by eval, never
asserted. This module just has to be deterministic and loadable.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .features import FEATURE_NAMES, FEATURE_SET_VERSION


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)  # avoid overflow for very negative z
    return e / (1.0 + e)


def _as_vector(features: Mapping[str, float] | Sequence[float]) -> list[float]:
    """A feature dict (by name) or an already-ordered sequence → the FEATURE_NAMES vector."""
    if isinstance(features, Mapping):
        return [float(features.get(name, 0.0) or 0.0) for name in FEATURE_NAMES]
    vec = [float(v) for v in features]
    if len(vec) != len(FEATURE_NAMES):
        raise ValueError(f"feature vector length {len(vec)} != {len(FEATURE_NAMES)}")
    return vec


@dataclass(frozen=True)
class LogRegModel:
    """Logistic-regression weights + the train-time standardization (mean/std)."""

    weights: tuple[float, ...]
    bias: float
    mean: tuple[float, ...]
    std: tuple[float, ...]
    feature_set_version: str
    algo: str
    params: dict[str, Any]

    def score(self, features: Mapping[str, float] | Sequence[float]) -> float:
        """Return the logit. Monotonic with probability, so it ranks identically."""
        x = _as_vector(features)
        z = self.bias
        for w, xi, m, s in zip(self.weights, x, self.mean, self.std, strict=True):
            z += w * ((xi - m) / s)
        return z

    def proba(self, features: Mapping[str, float] | Sequence[float]) -> float:
        return _sigmoid(self.score(features))

    def to_dict(self) -> dict[str, Any]:
        return {
            "algo": self.algo,
            "feature_set_version": self.feature_set_version,
            "feature_names": list(FEATURE_NAMES),
            "weights": list(self.weights),
            "bias": self.bias,
            "standardize": {"mean": list(self.mean), "std": list(self.std)},
            "params": self.params,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> LogRegModel:
        # Validate the artifact matches the running feature contract — `score()`
        # vectorizes with the CURRENT FEATURE_NAMES, so loading a model trained for a
        # different layout/version would silently corrupt scoring (correctness-critical
        # on the S-5 serve path). Refuse instead; the serving gate (spec 06) also checks
        # the manifest version up front and falls back to the heuristic.
        fsv = d.get("feature_set_version")
        if fsv != FEATURE_SET_VERSION:
            raise ValueError(
                f"model feature_set_version {fsv!r} != extractor {FEATURE_SET_VERSION!r}"
            )
        names = d.get("feature_names")
        if names is not None and list(names) != list(FEATURE_NAMES):
            raise ValueError("model feature_names do not match the current FEATURE_NAMES layout")
        std = d["standardize"]
        weights = tuple(float(w) for w in d["weights"])
        bias = float(d["bias"])
        mean = tuple(float(m) for m in std["mean"])
        std_vals = tuple(float(s) for s in std["std"])
        # weights/mean/std must each align with FEATURE_NAMES — a hash-matching artifact
        # with the wrong vector length would load "ready" but then raise from score()'s
        # zip(strict=True) on EVERY request (disabling learned scoring per-request). Reject
        # once at load instead, so the gate closes → heuristic.
        n = len(FEATURE_NAMES)
        if not (len(weights) == len(mean) == len(std_vals) == n):
            raise ValueError(
                f"model artifact vector lengths "
                f"(weights={len(weights)}, mean={len(mean)}, std={len(std_vals)}) "
                f"!= {n} (FEATURE_NAMES) — corrupted/incompatible"
            )
        # All params must be finite — a hash-matching artifact can still carry inf/NaN
        # (JSON `1e309`/`NaN`), which `score()` would propagate WITHOUT raising, silently
        # corrupting ordering / breaking response serialization instead of tripping the
        # shim's exception fallback. Refuse at load so the gate fails closed → heuristic.
        if not all(math.isfinite(v) for v in (*weights, bias, *mean, *std_vals)):
            raise ValueError("model artifact has a non-finite parameter (inf/NaN) — corrupted")
        if any(s == 0.0 for s in std_vals):
            raise ValueError("model artifact has a zero std — invalid or corrupted file")
        return cls(
            weights=weights,
            bias=bias,
            mean=mean,
            std=std_vals,
            feature_set_version=str(fsv),
            algo=d["algo"],
            params=dict(d.get("params", {})),
        )

    def save(self, path: str | Path) -> Path:
        """Atomic write (temp-then-rename) so a reader never sees a half-written model."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        os.replace(tmp, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> LogRegModel:
        return cls.from_dict(json.loads(Path(path).read_text()))


def train_logreg(
    feature_rows: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    seed: int = 0,
    l2: float = 1.0,
    lr: float = 0.5,
    epochs: int = 400,
) -> LogRegModel:
    """Fit a regularized logistic regression by deterministic full-batch GD.

    `feature_rows` are vectors in FEATURE_NAMES order. Features are standardized
    (z-score) using the train mean/std, which are baked into the artifact so serving
    applies the identical transform.
    """
    n = len(feature_rows)
    if n == 0 or n != len(labels):
        raise ValueError(f"need matching non-empty rows/labels, got {n}/{len(labels)}")
    d = len(FEATURE_NAMES)
    rows = [_as_vector(r) for r in feature_rows]
    y = [float(v) for v in labels]

    mean = [sum(r[j] for r in rows) / n for j in range(d)]
    var = [sum((r[j] - mean[j]) ** 2 for r in rows) / n for j in range(d)]
    std = [math.sqrt(v) or 1.0 for v in var]  # constant feature → std 1 (standardizes to 0)
    xs = [[(r[j] - mean[j]) / std[j] for j in range(d)] for r in rows]

    w = [0.0] * d
    b = 0.0
    for _ in range(epochs):
        preds = [_sigmoid(b + sum(w[j] * row[j] for j in range(d))) for row in xs]
        err = [preds[i] - y[i] for i in range(n)]
        grad_b = sum(err) / n
        grad_w = [
            (sum(err[i] * xs[i][j] for i in range(n)) / n) + (l2 * w[j] / n) for j in range(d)
        ]
        b -= lr * grad_b
        for j in range(d):
            w[j] -= lr * grad_w[j]

    return LogRegModel(
        weights=tuple(w),
        bias=b,
        mean=tuple(mean),
        std=tuple(std),
        feature_set_version=FEATURE_SET_VERSION,
        algo="logreg",
        params={"l2": l2, "lr": lr, "epochs": epochs, "seed": seed},
    )
