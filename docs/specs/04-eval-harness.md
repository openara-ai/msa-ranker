# Spec 04 — Eval harness + baseline

> Realizes **FR-1 / FR-2**, **ADR-004** (NDCG@k primary, MRR secondary), **INV-1**
> (baseline before any model). The graded scorer — the ruler everything is measured
> against. Status: draft.

## Purpose

Given an ordering of a query's candidate set and its labels, produce **NDCG@k** and
**MRR**, and measure MSA's current heuristic as the **baseline** — *before* any model.

## Interface

```python
# msa_ranker/eval.py  (C9) — as implemented in S-2. Relevance is BINARY in v1, so it is
# a set of opened media_ids (`positives`), not a graded dict; graded rel is a later upgrade.
def ndcg_at_k(order: list[str], positives: set[str], k: int) -> float
def mrr(order: list[str], positives: set[str], k: int) -> float

@dataclass
class Query:        # one held-out query: search_id, order (ranked media_ids), positives
    search_id: str; order: list[str]; positives: set[str]

@dataclass(frozen=True)
class EvalResult: ndcg: float; mrr: float; k: int; n_queries: int

def evaluate(queries: list[Query], k: int) -> EvalResult            # macro-avg over queries
def baseline_queries(conn, search_ids=None) -> list[Query]          # rank by heuristic_score
def measure_baseline(conn, k: int, search_ids=None) -> EvalResult   # MSA heuristic; INV-1
```

## Contract

- **Input:** per query (a held-out `search_id`): the **full shown candidate set** (every
  `result_shown` row) + **relevance** = the opens (binary; graded later). **Eval ranks
  the full candidate set, not the training-label subset** — spec 03's skip-above *drop*
  is a *training-label* construction; for eval, unopened ≠ excluded and unopened ≠
  forced-negative. Relevance is "was it opened"; everything shown participates in the
  ranking, unopened simply contributes gain 0. (This avoids both truncating the candidate
  set and silently converting deliberately-unlabeled rows into negatives.)
- **Output:** per-query NDCG@k + MRR, **macro-averaged** across queries (and per-user
  macro-average for multi-user later).
- **Formulas** (0-based `i`): `DCG@k = Σ_{i<k} rel_i / log2(i+2)`; `NDCG = DCG/IDCG`;
  `MRR = 1/(rank of first positive in top-k)`, else 0. `k` configurable.
- **Baseline:** rank each query by the **logged `heuristic_score`** (recorded on every
  `shown` event, even when the learned model served — NN1/spec 01), score it through the
  same harness → `eval` row `is_baseline=1`. Because `heuristic_score` is always logged,
  the baseline is computable on **any** traffic, including learned-served searches.
- **CI vs opt-in:** the **deterministic metric functions** (`ndcg_at_k`/`mrr`/macro-avg)
  are **golden unit-tested in CI**; only the **graded model-vs-baseline evaluation runs**
  over a dataset are **opt-in / local-only, never CI** (INV-2).
- Ties broken deterministically (`(−score, media_id)`); every model eval reported as **Δ
  vs baseline** on the same split, never absolute (ADR-004).

## Usage

Called by the training run (spec 05 step) — `measure_baseline` first (INV-1), then
`evaluate(model)` — both writing `eval` rows the registry/report read. See the
metrics intuition + worked example in [research §2](../research.md).

## Ownership

- **Implements:** agent — `msa_ranker.eval` (C9).
- **Reviews / owns:** human owns the **golden NDCG/MRR** (hand-calculated) and the
  metric definition.

## Acceptance criteria

- **AC-04.1** Golden NDCG/MRR reproduce the [research §2](../research.md) example:
  Ranker A → NDCG@5 ≈ 0.92, MRR 1.00; Ranker B → 0.50, 0.25.
- **AC-04.2** `k` truncation correct at bounds (`k=1`, `k ≥ |candidates|`).
- **AC-04.3** No-positive query handled identically for model and baseline.
- **AC-04.4** Tie-breaking deterministic (golden).
- **AC-04.5** A baseline `eval` row exists and predates the first model eval (INV-1).
- **AC-04.6** Macro-average equals the mean of per-query scores (golden).

## Tests

| Test | Type | Case → asserts | AC | Owner |
|---|---|---|---|---|
| worked example | unit/golden | research §2 two rankers → 0.92/1.00 and 0.50/0.25 | 04.1 | human |
| k bounds | unit | `k=1`, `k≥N` → correct truncation | 04.2 | agent |
| no positives | unit | query with no opens → NDCG 0 / MRR 0, same both sides | 04.3 | agent |
| tie-break | unit/golden | equal scores → deterministic order | 04.4 | agent |
| baseline-first | contract | baseline `eval` row predates first model eval (INV-1) | 04.5 | human |
| macro-average | unit/golden | mean of per-query == reported | 04.6 | agent |

**Fixtures:** the research §2 two-ranker example as the headline golden; small
labeled-query sets with hand-computed NDCG/MRR; a no-positive query. Hermetic,
fixed-seed. The **metric-function golden tests run in CI**; only the **graded
model-vs-baseline runs** over a dataset are opt-in/local-only.
