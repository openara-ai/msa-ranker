# Getting Started

This page is the starting point for `msa-ranker`: it describes the two workflows for
using the project and owns the first-run setup for each. The design starts at
[architecture.md](architecture.md); the recurring operator loop lives in the
[runbook](runbook.md).

`msa-ranker` is the learned reranker for
[media-search-agent](https://github.com/openara-ai/media-search-agent) (MSA). It is
trained offline on CPU and served in-process inside MSA, and you run the whole
training loop through its command line (the `msa-ranker` CLI). You do **not** need a
GPU. Setup needs the network once; after that, training and serving run fully
offline.

## The two workflows

Most users follow **workflow A**: install the released wheel and operate the training
loop against your own MSA install. Follow **workflow B** only if you want to work on
the package itself and modify the MLOps loop (features, labels, training, evaluation,
serving).

### A. Operate: use a released ranker with your MSA install

No clone needed: the released wheel ships both the serving library and the
`msa-ranker` CLI. Install it from the GitHub release assets:

```bash
pip install https://github.com/openara-ai/msa-ranker/releases/download/v0.1.1/msa_ranker-0.1.1-py3-none-any.whl
```

Or, from an already-downloaded file: `pip install ./msa_ranker-0.1.1-py3-none-any.whl`.

With MSA logging interactions, follow the [runbook](runbook.md) for the recurring
loop: train → review → deploy → enable → rollback.

### B. Develop: work on the package itself

Clone the repo, install editable, run the gates, and run the loop once end-to-end.
The walkthrough below covers this workflow.

## Workflow B walkthrough: developing the package

### 1. Prerequisites

- **macOS, Linux, or Windows.** The package is pure Python and cross-platform;
  training and serving are CPU-only.
- **Python 3.11+** and `git`.
- For the end-to-end loop, an MSA install that has been **logging interactions**. MSA
  writes the append-only event ledger this project trains on (see
  [How MSA consumes it](../README.md#how-msa-consumes-it)). You can explore the CLI and
  tests without one.

### 2. Set up a dev environment

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

### 3. Verify the install

Run the same gates CI runs (lint, format check, tests):

```bash
ruff check . && black --check . && pytest
```

A green run means your environment is good. `msa-ranker --help` lists the subcommands:
`ingest`, `train`, `report`, and `deploy`, plus an `eval` stub whose help text points
back into `train` (evaluation runs there; it is not a standalone command).

### 4. Run the loop once

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

`deploy` refuses a model that did not beat baseline.

### 5. Turn it on in MSA

Point MSA's `ranker.ltr_model_dir` at the deployed model and flip
`ranker.enable_learning_to_rank` in its `config.yaml`, then restart. The
[runbook](runbook.md) owns the exact steps, the safety behaviour, the data floor, and
the rollback.

## Where to go next

- [Runbook](runbook.md): the recurring operator loop and rollback.
- [Architecture](architecture.md): topology, the two data flows, the SoR schema.
- [Feature specs](specs/): per-stage behaviour + acceptance criteria.
- [Agentic Development](agentic-development.md): how this project was built.
