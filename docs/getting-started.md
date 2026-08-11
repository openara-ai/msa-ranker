# Getting Started

A first-run walkthrough for working on `msa-ranker` and running the loop once end-to-end.
This is the dev/operator on-ramp; the full operator reference is the
[runbook](runbook.md), and the design starts at [architecture.md](architecture.md).

`msa-ranker` is the learned reranker for
[media-search-agent](https://github.com/openara-ai/media-search-agent) (MSA). It is
trained offline on CPU and served in-process inside MSA, and you run the whole
training loop through its command line (the `msa-ranker` CLI). You do **not** need a
GPU. Setup (the clone and `pip install`) needs the network once; after that, training
and serving run fully offline.

## 1. Prerequisites

- **macOS, Linux, or Windows.** The package is pure Python and cross-platform;
  training and serving are CPU-only.
- **Python 3.11+** and `git`.
- For the end-to-end loop, an MSA install that has been **logging interactions**. MSA
  writes the append-only event ledger this project trains on (see
  [How MSA consumes it](../README.md#how-msa-consumes-it)). You can explore the CLI and
  tests without one.

## 2. Set up a dev environment

On macOS / Linux (or WSL):

```bash
git clone https://github.com/openara-ai/msa-ranker.git
cd msa-ranker
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
```

On Windows (PowerShell), only the activation step differs:

```powershell
git clone https://github.com/openara-ai/msa-ranker.git
cd msa-ranker
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

This installs the library, the `msa-ranker` CLI, and the dev tools (editable install).

## 3. Verify the install

Run the same gates CI runs (lint, format check, tests):

```bash
ruff check . && black --check . && pytest
```

A green run means your environment is good. `msa-ranker --help` lists the subcommands:
`ingest`, `train`, `report`, and `deploy`, plus an `eval` stub whose help text points
back into `train` (evaluation runs there; it is not a standalone command).

## 4. Run the loop once

With an MSA event ledger available, one command runs the whole offline pipeline
(ingest, label, freeze, measure baseline, train, eval, register):

```bash
msa-ranker train --db <SoR> --ledger-dir <MSA ledger dir> --out <models_dir> --k 10
```

It prints the model id, NDCG@k (model vs. the pre-training baseline), and whether the model
`beats_baseline`. Review every trained model, including failed experiments, with:

```bash
msa-ranker report --db <SoR>
```

Then deploy a gate-passing model into MSA's serving directory:

```bash
msa-ranker deploy --db <SoR> --model-id <id> --dest <ltr_model_dir>
```

`deploy` refuses a model that did not beat baseline. The full step-by-step (prerequisites,
the data floor, rollback) is in the [runbook](runbook.md).

## 5. Turn it on in MSA

Point MSA at the deployed model and flip the flag in MSA's `config.yaml`:

```yaml
ranker:
  enable_learning_to_rank: true   # master flag, default OFF
  ltr_model_dir: <dir from step 4>
```

Restart MSA. If the flag is off, the model is missing, or the load-time gate fails, MSA
logs it and serves the existing heuristic unchanged. Search always starts.

## Where to go next

- [Architecture](architecture.md): topology, the two data flows, the SoR schema.
- [Runbook](runbook.md): the full manual operator loop and rollback.
- [Feature specs](specs/): per-stage behaviour + acceptance criteria.
- [Agentic Development](agentic-development.md): how this project was built.
