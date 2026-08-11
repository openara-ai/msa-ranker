# Spec 06 — Serving: flag, fallback, gate

> Realizes **FR-13 / FR-14**, enforces **INV-2/3/6/9**, implements **ADR-001** (seam),
> **ADR-011** (cold-start gate). The in-process scorer at MSA's rerank seam — on MSA's
> correctness path, so fully deterministic + golden-tested. Status: draft.

## Purpose

Define exactly how the learned scorer attaches at the seam, when it serves vs falls
back, and the guarantees that keep MSA's search byte-identical to today when it should be.

## Interface

```python
# msa_ranker/serving.py  (library, loaded in MSA's process)
FEATURE_SET_VERSION: str                      # must match spec 02

class Ranker:
    feature_set_version: str
    @classmethod
    def load(cls, model_dir: Path) -> "Ranker": ...        # raises on missing/corrupt
    def score(self, candidates: list[dict], ctx: QueryContext, *, now: float) -> list[float]:
        """Pure given the loaded model. len(out) == len(candidates), same order. INV-2.
        `now` is the caller-supplied unix timestamp for recency features — passed in
        (not `time.time()` inside) so scoring is deterministic. The shim passes the same
        `now` used to compute the logged features, so there is no train/serve skew."""

def gate_ok(model_dir: Path) -> bool:
    """ADR-011: artifact+manifest exist AND manifest.beats_baseline AND
       manifest.feature_set_version == FEATURE_SET_VERSION AND
       sha256(artifact bytes) == manifest.artifact_sha (integrity — reject a corrupt
       or tampered/half-written artifact before serving)."""
```

```python
# MSA shim (in the MSA repo) — startup + the seam at engine.py:542-551
# startup (lifespan) — load is GUARDED: gate_ok() can pass yet load() raise
# (corrupt artifact / race with the atomic rename); must fail open, never crash MSA:
try:
    state.ranker = (
        serving.Ranker.load(dir)
        if cfg.enable_learning_to_rank and serving.gate_ok(dir) else None
    )
except Exception:
    log.warning("ranker load failed at startup — using heuristic")   # ADR-013 / INV-3
    state.ranker = None
# seam (`now` = the request timestamp also used to compute the logged features — pass it
# in so score() is deterministic and train/serve-skew-free; it is a required kw-only arg):
if state.ranker is not None:
    try: scores = state.ranker.score(candidates, ctx, now=now)
    except Exception: scores = heuristic_scores(candidates, q, ctx)   # FR-14 fail-safe
else:
    scores = heuristic_scores(candidates, q, ctx)                     # INV-3
```

## Usage

The shim resolves the model **once at startup** (gate applied there, not per request),
stashing a ready `Ranker` or `None` on app state. At the seam, a non-`None` ranker
scores the candidate batch; everything else (errors, flag off, gate closed) takes the
existing `score_breakdown()` path. Scoring writes `m["score"]` + the breakdown keys, so
the existing sort/top-k/format is untouched. See the seam + fallback edges in
[architecture §1](../architecture.md#1-the-sketch-topology) and §6 — not duplicated here.

**Even when the learned model serves, the cheap `score_breakdown()` heuristic score is
still computed and logged as `heuristic_score` on each `shown` event** (NN1/spec 01) — so
baseline eval (spec 04) is reconstructable on learned-served traffic. The heuristic is
microseconds, so this adds nothing meaningful to the latency budget (NFR-3).

## Guarantees (the invariants this spec owns)

- **INV-3 — flag off ≡ today.** Flag off / not ready ⇒ ordering **byte-identical** to
  the current heuristic.
- **INV-2 — deterministic.** Fixed model + inputs ⇒ deterministic `score()`; quality is
  eval, never asserted here.
- **INV-6 — reorder-only.** Output is a **permutation of the input candidates** (no
  adds/drops); **no write handle** to `media.sqlite`.
- **INV-9 — isolation.** An unwritable ledger / absent package never degrades the path.

## Gate truth table

| flag | artifact+manifest | beats_baseline | feat-version match | → serves |
|---|---|---|---|---|
| off | — | — | — | heuristic |
| on | missing | — | — | heuristic |
| on | present | false | — | heuristic |
| on | present | true | mismatch | heuristic |
| on | present | true | match | **learned** |

`gate_ok()` additionally verifies **`sha256(artifact) == manifest.artifact_sha`**; a
mismatch (corrupt / tampered / half-written) closes the gate → heuristic.

## Deployment layout (`ltr_model_dir`)

`gate_ok()` / `Ranker.load()` read a **canonical `manifest.json`** in `ltr_model_dir`,
and resolve the artifact from `manifest["artifact"]` (a **plain basename** — absolute
paths / traversal are rejected). The training registry (spec 05) emits per-model files
named `{model_id}.manifest.json` + `{model_id}.model.json`; the separate **deployment**
step (S-5.3, gated on `beats_baseline`) copies the chosen pair into `ltr_model_dir`,
renaming the sidecar to `manifest.json`:

```text
<ltr_model_dir>/
  manifest.json          ← the chosen {model_id}.manifest.json, renamed
  {model_id}.model.json  ← the artifact (filename carried in manifest["artifact"])
```

`Ranker.load()` re-runs the **full gate** on the manifest snapshot it reads (not just
"manifest exists"), so an atomic `manifest.json` swap between `gate_ok()` and `load()`
can't serve an ungated/tampered model — load fails and the shim falls open.

## Ownership

- **Implements:** agent — `msa_ranker.serving` (C2) + the MSA-side shim (MSA repo).
- **Reviews / owns:** **human — correctness-critical** (this is MSA's serving path).
  **Not self-merged** (agent-instructions §3);
  human owns the INV-3 golden truth.

## Acceptance criteria

- **AC-06.1** Flag-off ordering == heuristic ordering, byte-identical, on fixtures (INV-3).
- **AC-06.2** Each gate-truth-table row produces the stated path.
- **AC-06.3** A scoring error mid-request ⇒ heuristic fallback, response still 200 (FR-14).
- **AC-06.4** Output set is a permutation of the input candidate set — no add/drop (INV-6).
- **AC-06.5** No write handle to `media.sqlite` is opened by the serving path (INV-6).
- **AC-06.6** Unwritable-ledger fault ⇒ `/search` returns the full set, order unchanged;
  `/track/open` ⇒ 204 (INV-9).
- **AC-06.7** `feature_set_version` mismatch ⇒ gate closed → heuristic (no silent
  mis-scoring).
- **AC-06.8** **Gate passes but `Ranker.load()` raises** (corrupt artifact) ⇒ startup
  sets `ranker=None` (logged) and MSA **still starts** on the heuristic — the exception
  never reaches the lifespan (ADR-013/INV-3).
- **AC-06.9** **`artifact_sha` mismatch** (artifact bytes ≠ `manifest.artifact_sha`) ⇒
  `gate_ok()` returns False → heuristic (no serving of a corrupt/tampered model).

## Tests

| Test | Type | Case → asserts | AC | Owner |
|---|---|---|---|---|
| flag-off equality | golden/contract | flag off, fixture query → order byte-identical to heuristic | 06.1 | human |
| gate matrix | contract | each truth-table row → correct path | 06.2/06.7 | agent |
| scorer raises | integration | `score()` throws mid-request → heuristic, HTTP 200 | 06.3 | agent |
| load raises | integration | gate ok but `load()` throws at startup → `ranker=None`, MSA starts | 06.8 | human |
| sha mismatch | contract | artifact bytes ≠ `manifest.artifact_sha` → `gate_ok()` False → heuristic | 06.9 | human |
| permutation | contract | learned path → output multiset == input | 06.4 | human |
| no-write | contract | serving path opens no write handle to `media.sqlite` | 06.5 | human |
| ledger fault | integration | unwritable ledger dir → `/search` full set + same order; `/track/open` 204 | 06.6 | agent |
| version skew | contract | manifest `feature_set_version != v1` → gate closed | 06.7 | agent |

**Fixtures:** a seeded candidate set with a known heuristic order; a tiny trained model
with its manifest (from spec 05); a manifest with `beats_baseline=false` and one with a
skewed `feature_set_version`; a temp-dir ledger made read-only for the fault case.
Correctness-critical tests (flag-off equality, permutation, no-write) are human-reviewed.
