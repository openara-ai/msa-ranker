# Runbook — train, evaluate, deploy, serve (loop-1)

> The **manual** operator loop (ADR-011): training is human-triggered offline on CPU; the
> model is handed to MSA via a path-based copy (architecture §11). Nothing here runs
> automatically. Commands are the `msa-ranker` CLI; `<SoR>` is the training store
> (default `~/.msa-ranker/msa-ranker.sqlite`), `<ltr_dir>` is MSA's `ranker.ltr_model_dir`.

## 0. Prerequisites

- MSA is logging interactions: `ranker.event_logging: true` (default) → it appends a JSONL
  **event ledger** under MSA's data area (`ranker.ledger_dir`, default `<data_dir>/ranker-ledger`,
  alongside `index/` — **not** `logs/`, which is disposable; spec 01).
- Enough **labelled searches** have accrued (searches with at least one `open`). Below the
  floor, `train` refuses (no garbage model). Wait for soak if so.
- `msa-ranker` installed. Operators: `pip install` the released wheel from the GitHub
  release assets — it ships the CLI and the serving library, no clone needed.
  Developers: an editable install from a clone (`pip install -e ".[dev]"`); see
  [Getting Started](getting-started.md).

## 1. Train (ingest → labels → baseline → train → eval → register)

```bash
msa-ranker train --db <SoR> --ledger-dir <MSA ledger dir> --out <models_dir> --k 10
```

One command runs the whole offline pipeline: ingest the ledger into the SoR → build
**Click > Skip-Above** labels → freeze an immutable dataset → measure the **baseline first**
(INV-1) → train the logreg → evaluate → **register every model** (pass or fail) with its
`beats_baseline` flag and a sidecar `manifest.json`. It prints the model id, NDCG@k
(model vs baseline), and `beats_baseline`. Re-running with the same data + `--seed` is
reproducible. (`--ledger-dir` is optional — omit it if the SoR is already ingested.)

## 2. Review the registry

```bash
msa-ranker report --db <SoR>          # table; add --json for machine output
```

Lists every trained model with NDCG@k vs baseline, Δ, and the gate (`PASS`/`fail`) —
including failed experiments (FR-17). Pick a `PASS` model to deploy.

## 3. Deploy (the gated handoff)

```bash
msa-ranker deploy --db <SoR> --model-id <id> --dest <ltr_dir>
```

Copies the artifact + a canonical `manifest.json` into `<ltr_dir>` (atomic temp-rename).
**Refuses a model that did not beat baseline** (use `--force` only to stage a known-failing
model deliberately). The serving gate re-checks `beats_baseline`, the feature-set version,
and the artifact sha at load, so a partial or stale copy fails closed → heuristic.

## 4. Enable serving in MSA

In MSA's `config.yaml`:

```yaml
ranker:
  enable_learning_to_rank: true     # master flag — default OFF (search byte-identical, INV-3)
  ltr_model_dir: <ltr_dir>          # where step 3 deployed
```

Restart MSA. At startup it gates + loads the model once; if the gate is closed or load
fails, it logs and serves the **heuristic** (MSA still starts — AC-06.8). When active, the
learned model reorders the existing candidate set only (INV-6); the heuristic score is
still logged on every `shown` event (NN1), so the baseline stays reconstructable.

## 5. Roll back / disable

Set `enable_learning_to_rank: false` (or remove `ltr_model_dir`) and restart → instant
return to the heuristic. To swap models, deploy a different id into `<ltr_dir>` and restart.

## Privacy / off-switch

- `ranker.event_logging: false` stops all ledger writes (ADR-014) — no new labels collected.
- The ledger + SoR are **local and private** (INV-5/10); raw query/media/person refs never
  leave the private ledger. Logging failure never degrades MSA search or open (INV-9).
