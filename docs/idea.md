# Idea — msa-ranker (learned ranking for media-search-agent)

> Stage 1 artifact (development-workflow.md). The seed and the *why now*. One-pager;
> a living doc. This kicks off the spine: research → requirements → architecture →
> specs/testing → roadmap → impl loop → release.

> **Session kickoff (read first — no `CLAUDE.md` exists yet).**
> This is a fresh repo; this doc is the only project artifact so far.
> 1. Adopt the `standards` repo as the source of truth.
> 2. Read this Idea doc in full.
> 3. Before producing artifacts, **ask me the open questions in the "Open questions"
>    section** — several block requirements/architecture and must not be guessed
>    (MSA stack + result fields, how the ranker attaches to MSA, the product-data
>    storage decision, GPU specifics).
> 4. Then proceed through the workflow spine, producing each stage's standard
>    artifact as a *living* doc: research notes → `docs/requirements.md` →
>    `docs/architecture.md` (+ decision-ledger `docs/invariants.md`/`docs/adrs.md`)
>    → `docs/specs/` + `docs/testing.md` → `docs/roadmap.md`. The
>    decision-ledger is a backplane that accretes — open ADRs as decisions are made,
>    don't front-load a fixed set.
> 5. Set up the agent contract per `agent-instructions.md`: create `CLAUDE.md`
>    (vendor the Base conventions block verbatim, then fill the project layer from
>    what you learn) and the thin `AGENTS.md` pointer, from that standard's
>    templates.
> 6. **Do not write code** until requirements + architecture exist and I approve.
>    Follow the standards' git/never-do rules throughout (stage named files only;
>    no push/PR/merge unless I ask).

## The problem worth solving (one sentence)
MSA returns media search results in a fixed/heuristic order; relevance to the user
is left on the table because the ordering doesn't learn from which results the user
actually opens.

## What this adds
A non-LLM supervised **learning-to-rank** model that reorders MSA's existing result
set using features of each result + the query, trained on the user's own logged
interactions, served inside MSA behind a flag with a deterministic fallback to the
current ordering.

## Why now
- MSA phase 1 (search) exists and produces result sets to reorder — the prerequisite
  is in place.
- It introduces a non-LLM AI model into the project's agentic-engineering practice.
- It exercises a full, owned ML lifecycle (data → train → eval → deploy → monitor →
  retrain) end to end; its engineering process is observed by devdash (pull-based —
  via normal sessions, commits, PRs, and internal/metrics/, nothing pushed).

## Goals (priority order)
1. Measurably improve MSA result ordering over the current baseline (graded eval:
   NDCG@k / MRR on held-out interactions).
2. Run the full lifecycle cleanly — reproducible data, training, graded eval,
   flagged deployment, monitoring that feeds retraining.
3. Integrate the model cleanly behind a safe toggle without compromising MSA's
   existing search path.

## Hard design principles (carry into requirements/architecture; record as invariants)
- **Baseline + metric before any model.** Implement the graded eval harness
  (NDCG@k / MRR) and measure MSA's *current* ordering as the baseline to beat
  before training anything.
- **Model out of the correctness path** (testing.md). Feature extraction, the
  flag/fallback, the serving path, and metrics recording are deterministic and
  golden-tested; ranking quality is **eval** (graded), not pass/fail tests.
- **Smallest complete loop first.** A trivial ranker that beats the baseline,
  served behind the flag, ships before model sophistication.
- **Owned, reproducible, leak-free data.** Labels come from logged interactions;
  data is versioned/replayable; no train/eval leakage.
- **Safe degradation.** Flag off → MSA behaves exactly as today.

## Open questions to resolve in research (stage 2) — confirm with developer
- MSA's stack, and what a search result looks like (fields available per result →
  drives feature design).
- LTR approach: pointwise vs pairwise vs listwise for a small-data start, with an
  upgrade path (e.g. LightGBM LambdaMART); the ranking metric best fitting MSA's
  result style.
- How the ranker attaches to MSA (in-process vs sidecar) and how the flag/fallback
  is wired.
- Data versioning: is DVC warranted yet or premature at this data scale?
- **Product data storage (ADR-worthy).** Where/how to store (a) ML lifecycle data —
  interaction logs as labels, dataset versions, eval scores per model version — and
  (b) production serving telemetry — append-heavy, irreplaceable, private, doubling as
  monitoring substrate + future training labels. NOT `internal/metrics/` (that's the
  process-metrics standard's). Study devdash's `docs/data-model.md` as precedent (it
  uses a dedicated versioned-migration SQLite system-of-record outside the repo for
  exactly this class of data). Decide schema, location, retention, and the
  public/private (`internal/`-and-gitignored) story.
- Register the repo root under devdash's git_roots so its engineering process is
  picked up (devdash is pull-based — no event schema to invent here).
- GPU specifics (sets model-size choices).

## Scope of the first complete loop (keep small)
1. Graded eval harness (NDCG@k / MRR) + measured baseline on current ordering.
2. Interaction logging (query, results shown w/ fields, opened result = label);
   versioned/replayable.
3. Deliberately simple baseline ranker; trained on logged interactions; eval-scored.
4. Served behind a flag in MSA with fallback.
5. Serving behaviour recorded to msa-ranker's product-telemetry store (design TBD —
   see architecture-stage open question) → also the next round of training labels.
   This is ML telemetry, not engineering-process metrics and not a devdash input.

## Non-goals (initial)
- No model sophistication / tuning before the loop is complete.
- No LLM fine-tuning or RAG (separate concerns; not this project).
- No building of or pushing to devdash (it observes pull-based; this repo only keeps
  the normal session/git/PR/internal-metrics trail it reads).
