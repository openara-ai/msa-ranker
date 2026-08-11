# Spec 05 — Training, registry & manifest

> Realizes **FR-6 / FR-8 / FR-9 / FR-10**, **ADR-002** (model progression),
> **ADR-011** (manual trigger), **ADR-003** (offline). The offline pipeline that turns
> a dataset into a served-ready artifact. Status: draft.

## Purpose

Define the **manual, reproducible** freeze → train → eval → register flow that emits the
**model + manifest** MSA consumes (architecture §10).

## Interface

```python
# msa_ranker/datasets.py / train.py / registry.py
def freeze_dataset(sor, *, filters: dict, seed: int, eval_frac: float) -> str  # → dataset_id;
#   freezes watermark + exact ev_id set + seed + eval_frac (fully immutable/replayable) — FR-6
def train(sor, dataset_id: str, *, algo: str = "logreg", seed: int) -> Path   # → artifact
def register(sor, artifact: Path, dataset_id: str,
             model_eval: EvalResult, baseline: EvalResult) -> str       # → model_id; writes
                                                                        # model+eval rows + manifest.json
# CLI (manual, ADR-011): msa-ranker train [--algo logreg] [--k 10]
#   pipeline: ingest (spec 07) → build_labels (03) → freeze →
#             measure_baseline (04, INV-1: BEFORE any model) → train →
#             evaluate (04) → register EVERY model (with beats_baseline) — deploy only passers
```

## Contract

1. **Dataset freeze (FR-6) — must be *immutable*:** a manifest of `search_id`s + seed is
   **not** replayable, because the SoR keeps ingesting (a late `open` shifts the
   deepest-open boundary → different labels/splits). So freeze names **exactly which
   events**: an **ingest watermark + the exact set of source `ev_id`s** (`search`/`shown`/
   `open`) that compose the dataset (or, equivalently, **materialize the frozen label
   rows**). Replaying a `dataset_id` then yields a byte-identical row set regardless of
   later ingestion (FR-6/INV-4). The `dataset` row records the watermark + `ev_id` set +
   **split seed *and* `eval_frac`** (or the exact group→split assignment) — otherwise
   changing `eval_frac` would re-split the same `dataset_id` and change the artifact/metrics
   (NN2).
2. **Measure baseline (spec 04 / INV-1) — BEFORE any model:** score the MSA heuristic
   order on the held-out split → the baseline `eval` row (`is_baseline=1`). This step
   runs *before* training so the bar exists first.
3. **Train (FR-8/10):** v1 = regularized **logistic/linear** over spec-02 features
   (ADR-002); deterministic given `seed`; host or VM (ADR-003). Records
   `feature_set_version`.
4. **Eval the model (spec 04):** NDCG@k / MRR on the same held-out split → model `eval`
   rows, compared against the step-2 baseline.
5. **Register (FR-9) — *every* evaluated model:** write the `model` + `eval` rows + the
   **manifest JSON** (carrying `beats_baseline`) for **every** trained model, pass *or*
   fail — FR-9 records all, and the FR-17 report shows failed experiments too.
   **Deployment** (copying the artifact+manifest to `ltr_model_dir`) is the separate,
   gated step — only `beats_baseline=true` models serve (spec 06). This also makes the
   `beats_baseline=false` gate path reachable through the documented pipeline.

- **Trigger:** manual CLI (ADR-011); a **minimum labelled-search threshold** gates the
  first run (refuse + message below it).
- `beats_baseline = model.ndcg@k > baseline.ndcg@k` (same split) — the serving gate
  (spec 06) reads exactly this.

## Usage

Run by the developer on the host/VM after enough interactions accrue; produces the
artifact+manifest that gets copied to MSA's `ltr_model_dir` (architecture §11 handoff).
Re-running from a `dataset_id`+seed reproduces the artifact within tolerance.

## Ownership

- **Implements:** agent — `msa_ranker.{datasets,train,registry}` (C7/C8/C10).
- **Reviews / owns:** human owns the **acceptance** (`beats_baseline` truth, the
  min-data threshold) and reviews. Model *quality* is eval, never asserted (INV-2).

## Acceptance criteria

- **AC-05.1** A frozen `dataset_id` replays to an identical row set (FR-6).
- **AC-05.2** Train→eval→register on a tiny fixture → loadable artifact + schema-valid
  manifest with correct lineage (`model.dataset_id`).
- **AC-05.3** `beats_baseline` computed correctly vs the recorded baseline (golden).
- **AC-05.4** Below the min-data threshold, training **refuses** clearly (no garbage
  model written).
- **AC-05.5** `feature_set_version` recorded in the manifest, matches spec 02.
- **AC-05.6** Re-train from the same `dataset_id`+seed reproduces eval within tolerance.

## Tests

| Test | Type | Case → asserts | AC | Owner |
|---|---|---|---|---|
| dataset replay | unit | freeze → replay by id → identical rows | 05.1 | agent |
| tiny pipeline | integration | end-to-end on a fixture → artifact loads, manifest valid, lineage set | 05.2/05.5 | agent |
| beats_baseline | unit/golden | known model vs known baseline → correct flag | 05.3 | human |
| min-data gate | unit | below threshold → refuses, no artifact | 05.4 | agent |
| reproducibility | integration | same dataset_id+seed → eval within tolerance | 05.6 | agent |

**Fixtures:** a tiny frozen dataset with known labels; a known baseline `eval`; a
below-threshold ledger for the refusal case. (Quality is graded, not asserted — INV-2.)
