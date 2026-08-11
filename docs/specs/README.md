# Feature specs — msa-ranker

> **Stage 5 artifacts** of the development workflow
> (Feature specs + test strategy). One spec per loop-1 feature: detailed behaviour,
> the I/O contract, and **checkable acceptance criteria**. Driver: agent drafts,
> human refines. Status: **draft**.
>
> Specs **reference** the design rather than restate it: [`requirements.md`](../requirements.md)
> (FR/NFR), [`architecture.md`](../architecture.md) (topology, the two stores §10),
> the decision ledger [`adrs.md`](../adrs.md) / [`invariants.md`](../invariants.md).
> Test strategy spanning all of them lives in [`../testing.md`](../testing.md).

## The specs

| # | Spec | Realizes | Crux? |
|---|---|---|---|
| 01 | [Event ledger format](./01-event-ledger.md) | FR-3/16, ADR-012 | |
| 02 | [Feature extraction](./02-feature-extraction.md) | FR-4, INV-2 | **yes** |
| 03 | [Label construction](./03-label-construction.md) | FR-5/7, ADR-007, INV-4 | **yes** |
| 04 | [Eval harness + baseline](./04-eval-harness.md) | FR-1/2, ADR-004, INV-1 | |
| 05 | [Training, registry & manifest](./05-training-registry.md) | FR-6/8/9/10, ADR-011 | |
| 06 | [Serving: flag, fallback, gate](./06-serving-flag-fallback.md) | FR-13/14, INV-2/3/6, ADR-011 | |
| 07 | [Ledger ingest → SoR](./07-ledger-ingest.md) | FR-18, ADR-012, INV-7 | |

The two **crux** specs (feature extraction, label construction) carry the subtle
correctness; the rest are mostly contracts over what architecture already pins.

## Spec conventions

Each spec follows the same shape:

- **Purpose** — one paragraph: what it's for.
- **Interface** — concrete signatures (functions/classes/endpoints + types) to
  implement against. Signatures, not implementations (we're still pre-code).
- **Contract / Behaviour** — the detailed rules, edge cases, and (for the stores) the
  data shapes.
- **Usage** — how it's called in context. Drawings are **link-first** — reference the
  [architecture](../architecture.md) diagram; add a *local* drawing only where it adds
  detail (e.g. the skip-above labeling in 03, the ingest fold in 07).
- **Ownership** — which component/side **implements** it (MSA shim C1–C3 vs the
  `msa_ranker` library C4–C11) and the **human/agent** split; **correctness-critical**
  specs (the serving path; golden-truth for the crux specs) are **human-reviewed /
  not self-merged** (agent-instructions §3, testing standard).
- **Acceptance criteria** — numbered, checkable `AC-nn.x` (stable ids).
- **Tests** — a test-case table (**type · case→asserts · AC · owner**) + the
  **fixtures** needed; each row maps to [`../testing.md`](../testing.md).

Rules: behaviour is deterministic unless explicitly an **eval** (graded) concern
(INV-2); a spec found wrong/ambiguous in build is **edited here** (feedback edge), not
worked around in code.
