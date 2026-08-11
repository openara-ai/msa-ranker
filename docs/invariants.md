# Decision ledger — invariants (msa-ranker)

> The binding, **non-negotiable** rules of this project — the other half of the
> **decision-ledger backplane** (development workflow).
> Invariants are derived from the idea's hard design principles and the ADRs
> ([`adrs.md`](./adrs.md)). Per the testing standard:
> **each invariant gets a contract test that fails when it's violated** — an
> invariant without such a test is a comment, not a guardrail. Tests are written
> when the code they guard exists (Stage 7); the **Contract test** column names the
> intended check now.
>
> A task that seems to require violating an invariant is a **design change** — stop
> and ask; amend via a new ADR, never a silent in-line edit.

| Id | Invariant | Source | Contract test (intended) |
|---|---|---|---|
| **INV-1** | **Baseline + metric before any model.** The graded eval harness (NDCG@k / MRR) exists and MSA's *current* ordering is measured as the baseline-to-beat **before** any model is trained or served. | idea; [ADR-004](./adrs.md#adr-004) | Process gate at milestone exit: a recorded baseline score exists and predates the first model's eval row. |
| **INV-2** | **Model out of the correctness path.** Feature extraction, the flag/fallback routing, the serving path, and the telemetry write are **deterministic** and **golden-tested**. Ranking *quality* is measured by **eval (graded)**, never by a pass/fail test. | idea; testing.md; [ADR-001](./adrs.md#adr-001), [ADR-004](./adrs.md#adr-004) | Golden tests on feature extraction + scorer plumbing (fixed inputs → fixed outputs); no assertion anywhere ties a *quality* number to a pass/fail threshold. |
| **INV-3** | **Safe degradation.** Flag **off** → MSA behaves **exactly as today** (the deterministic heuristic). The learned path is purely additive. | idea; [ADR-001](./adrs.md#adr-001) | Golden test: flag-off output for a fixture query is byte-identical to the heuristic `score_breakdown()` ordering. |
| **INV-4** | **Owned, reproducible, leak-free data.** Labels come only from logged interactions; datasets are versioned/replayable (by manifest id); **no train/eval leakage** (query-grouped splits — the same query never straddles splits). | idea; [ADR-005](./adrs.md#adr-005), [ADR-006](./adrs.md#adr-006) | Splitter test: no query id appears in both train and eval; a dataset manifest replays to an identical row set. |
| **INV-5** | **Telemetry SoR is private and outside the repo.** The system-of-record lives outside the repo and is gitignored; **raw telemetry is never committed**. Only redacted/aggregate artifacts + human overlays (under `internal/`) are committed. | idea; [ADR-005](./adrs.md#adr-005) | Publishability test: no SoR path or raw-telemetry file is tracked by git; repo-public simulation leaks nothing private. |
| **INV-6** | **MSA search-path integrity.** The ranker **only reorders** MSA's existing candidate set — it never changes retrieval/recall (the output is a permutation of the input candidates, same multiset) and **never writes** to MSA's index DB. | idea (clean integration); [ADR-001](./adrs.md#adr-001) | Contract test: reranked set ≡ input set as a multiset (no adds/drops); the ranker holds no write handle to `index/media.sqlite`. |
| **INV-7** | **Additive migrations only.** Schema changes are forward `migrations/000N_*.sql` files tracked in `_migrations`; a **shipped migration is never edited or dropped**. | [ADR-005](./adrs.md#adr-005) | CI check: shipped migration files are unchanged vs the default branch; the runner only applies un-recorded files. |
| **INV-8** | **Smallest complete loop first.** No model sophistication (LambdaMART tuning, neural reranker, IPW) lands before the loop — eval → log → train → serve-behind-flag → telemetry — is closed end-to-end and beats baseline. | idea; [ADR-002](./adrs.md#adr-002) | Process gate at milestone exit: the five loop stages are demonstrably complete before any "upgrade" sprint opens. |
| **INV-9** | **Telemetry/ledger isolation.** A ledger or model-store failure (absent, unwritable, corrupt) — or an absent `msa_ranker` package — **never degrades MSA's search or open path**; events drop-and-log, never propagate. | [ADR-013](./adrs.md#adr-013); INV-2/3 | Fault-injection test: an unwritable ledger ⇒ `/search` returns the full result set with **byte-identical ordering**; `/track/open` ⇒ 204. |
| **INV-10** | **Sensitive-content isolation.** **Real/production** query text and media/person references exist **only** in the private ledger/SoR (outside the repo) — **no generated/exported artifact, `internal/` report, or devdash input contains them**; generated reports are aggregate/redacted. *(Hand-authored design docs may use **synthetic placeholders** — `<query_text>`, `<person_id>` — which are exempt.)* | [ADR-014](./adrs.md#adr-014); INV-5 | Redaction check: scan **generated/exported** reports + the live ledger-derived outputs for real query strings / `media_id` / `person_id` ⇒ none; publishability check (no ledger content git-tracked). |

## Notes

- **INV-1 and INV-8 are process gates**, enforced at milestone exit (the human runs
  them), not unit-testable assertions — but they are still binding and recorded
  here so the gate is explicit.
- **INV-2, INV-3, INV-4, INV-6, INV-7, INV-9, INV-10** are contract-testable and
  **must** get a failing-on-violation test in the same diff as the code they guard.
- **INV-5** is checked by the publishability item of the manual runbook
  (testing standard).
