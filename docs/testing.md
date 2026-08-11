# Testing — msa-ranker

> **Stage 5 artifact** (test strategy + harness), governed by the
> testing standard. What we test, how, and the
> fail-closed gate. A *living* doc. Status: **draft** (harness wired in Stage 7 with
> the toolchain). Specs' acceptance criteria (`AC-nn.x`) are the unit of coverage —
> see [`specs/`](./specs/).

## Guiding principle

**Tests anchor to independent truth, never the code's own output** — golden values are
hand-computed or reference-calculated. And the **model is kept out of the correctness
path** (INV-2): feature extraction, the flag/fallback/gate, the serving wiring, the
ledger append, and ingest are **deterministic and golden/contract-tested**; the model's
**ranking quality is eval (graded), never pass/fail** ([research §8 test-vs-eval](./research.md#ranking--evaluation)).

## Categories used

| Category | Here | Where |
|---|---|---|
| **Static gates** | format + lint (warnings = errors) | CI (Stage 7) |
| **Unit (golden)** — the bulk | feature vectors (02), labels (03), NDCG/MRR (04), ingest state (07) vs hand-computed truth | CI |
| **Contract / invariant** | one test per contract-testable invariant (below) | CI |
| **Integration** | seam branch + `/track/open` + ledger append over a child MSA process | CI |
| **Property-based** | ledger↔ingest round-trip; ingest idempotency; split leak-freeness | CI |
| **Security scans** | deps/advisories, secret scan | CI |
| **Eval (graded)** | NDCG@k/MRR model-vs-baseline | **opt-in, never CI** |
| **Real-data regression** | the developer's real ledger | **local only** |
| **Manual runbook** | below | human |

## Contract tests (one per contract-testable invariant)

| Invariant | Test (fails when violated) | Spec AC |
|---|---|---|
| **INV-2** | feature extraction + scorer routing are deterministic (golden) | AC-02.1, AC-06.2 |
| **INV-3** | flag-off ordering byte-identical to the heuristic | AC-06.1 |
| **INV-4** | query-grouped split shares no group across train/eval | AC-03.6 |
| **INV-6** | reranked set is a permutation of input; no `media.sqlite` write handle | AC-06.4/.5 |
| **INV-7** | migration runner applies only un-recorded files; shipped files unchanged | AC-07.5 |
| **INV-9** | unwritable ledger ⇒ `/search` full set + identical order; `/track/open` 204 | AC-06.6 |
| **INV-10** | no report/export/git-tracked file contains raw query / media_id / person_id | AC-01.7 |

INV-1 (baseline-first) and INV-8 (loop-before-sophistication) are **process gates** at
milestone exit, not unit tests ([invariants.md](./invariants.md)).

## Harness

In-process + hermetic by default: a **seeded synthetic generator** emits a fixture
**event ledger**; tests ingest it into a **temp SoR** (fresh per test, parallel-safe)
and assert against **golden** features/labels/scores. Only integration spawns a child
MSA process over loopback; only opt-in **eval** touches a model. No real data, network,
or secrets in CI.

## Test-data strategy

Default = **synthetic, seeded, with pre-computed golden answers** (repeatable). The
highest-value check is a **real-data regression** on the developer's own ledger, run
**locally only** (never committed — internal/).

## CI gate (fail-closed)

Mirrors the local dev loop: format → lint (warnings = errors) → build (locked deps) →
security scans → unit · contract · integration · property → (post-build) smoke. Any red
stage blocks merge. **Eval and real-data regression are not in CI.** *Toolchain/commands
are TBD until Stage 6/7 — expected `ruff`/`black`/`pytest`, mirroring MSA; the
project's agent contract (Commands) is the source once set.* The agentic code-review
workflow runs beside this gate per code-review-automation.

## Manual runbook (human-only — gates milestone exit)

1. **Real-data validation** — labels/eval are sane on the developer's real ledger.
2. **Flag-off degradation (INV-3)** — with the flag off, MSA search is unchanged.
3. **Ledger-fault degradation (INV-9)** — make the ledger dir unwritable; confirm
   search + open still work, events drop with a logged warning.
4. **Recovery** — rebuild the SoR from the ledger via ingest (idempotent); verify row
   counts before reporting success.
5. **Publishability & redaction (INV-5 / INV-10)** — no ledger/SoR path or raw
   telemetry is tracked by git; no generated/`internal/` report contains raw queries,
   `media_id`s, or `person_id`s; nothing private leaks if MSA goes public.
6. **Baseline-first (INV-1)** — a recorded baseline predates the first served model.
7. **Logging opt-out (ADR-014)** — set `ranker.event_logging=false`; confirm **no** new
   ledger events are written while search + open still work.

## Correctness-critical

The **serving path** (feature extraction, flag/fallback/gate — spec 06) is on MSA's
correctness path: human-reviewed, **not self-merged** (agent-instructions §3).
