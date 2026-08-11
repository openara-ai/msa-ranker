# Agentic Development: How This Project Was Built

`msa-ranker` was built with **agentic engineering**: AI coding agents act as the code
authors, working a defined workflow under human direction and review. The human owns the
decisions, the priorities, and every merge; the agent does the reading, drafting,
building, and testing, and it proposes, never declares, that work is done. This doc is
the playbook: the workflow, the artifacts, the review mesh, and the guardrails that keep
an agent productive without letting it quietly break things.

It is deliberately a *curated, public* account. The raw process records (session
telemetry, estimate-vs-actual metrics, the learnings and bug ledgers) live in a private
`internal/` area and are not part of this writeup.

## The workflow spine

Work followed a single spine from idea to release, each stage producing a standard,
living artifact in [`docs/`](.):

1. **Idea** → the seed and the *why now*.
2. **Research** → [research.md](research.md): a time-boxed investigation of the approach
   (non-LLM learning-to-rank, the metric, how to attach to MSA); the spike that fed the
   ADRs.
3. **Requirements** → [requirements.md](requirements.md): FR-1…18 / NFR-1…10, scope, non-goals.
4. **Architecture** → [architecture.md](architecture.md): topology, the two data flows, the
   system-of-record schema, sketched early.
5. **Specs + test strategy** → [docs/specs/](specs/) (one per feature) + [testing.md](testing.md).
6. **Decision ledger** → [adrs.md](adrs.md) and [invariants.md](invariants.md), captured
   continuously as decisions were made.
7. **Roadmap & execution** → releases / milestones / sprints with estimates.
8. **Implementation loop** → build → lint → format → test → review → fix, one small slice
   at a time, with the docs updated when reality diverged.

Artifacts are the source of truth, not the chat, and they are living. When the build
revealed a doc was wrong, the doc was changed, not just the code.

## The decision ledger: ADRs + invariants

Two binding, non-negotiable backplanes governed every change:

- **[Architecture Decision Records](adrs.md)**: ADR-001…014, each an accepted, dated
  decision with its context and consequences. The agent follows them; it does not
  re-derive or silently override them.
- **[Invariants](invariants.md)**: INV-1…10, the properties that must always hold (e.g.
  flag-off ordering is byte-identical to MSA today; the ranker only reorders and never
  writes MSA's index; ledger/telemetry failure never degrades search). If a task seemed
  to require violating one, the rule was **stop and ask**: that's a design change, not
  an implementation detail.

## Per-agent instruction files

The agent's contract lives in two files at the repo root, both shipped on the public
mirror:

- **`CLAUDE.md`**: the public operating contract, covering session-start orientation,
  the dev loop, git workflow, the review-mesh triage protocol, the milestone-exit
  gate, and the consolidated never-do list. The base conventions are inherited from a
  shared, cross-project standard and not restated per repo; the project layer adds
  the stack-specific guardrails. Project-specific state (current milestone and sprint
  state, private planning links) lives in a private companion file under `internal/`
  that the contract loads next in the development repo; the public copy stands on
  its own.
- **`AGENTS.md`**: a thin pointer so any agent toolchain lands on the same contract.

These start each session: the agent reads the contract and the design docs it lists, in
order, before touching code.

## The review mesh

Every change went through two layers of review before it could land:

- **Automated agentic review.** A GitHub Actions workflow runs an AI reviewer on each pull
  request (configured default-branch-only, so review wiring is never iterated on from a
  feature branch). It runs alongside the CI gate, not in place of it.
- **Human-led triage.** Every review comment is triaged with a priority and the triage is
  shared *before* fixing:
  - **P0**: critical: live breakage, security-exploitable now, data loss in flight.
  - **P1**: broken / unsafe; regression of existing behaviour.
  - **P2**: actively surprises the user or breaks their stated intent.
  - **P3**: polish / defensive hardening / cosmetic.

  Fixes are made, approved, committed, and then answered with one **per-finding** reply,
  token-first (`Pn — Fixed in <sha>. <one line>`), bound to the finding's file and line,
  so the outcome of each comment is traceable.

## Guardrails

The agent operated inside hard guardrails, not on trust:

- **Correctness-critical paths get human review.** The ranker sits on MSA's serving path,
  so the flag, the deterministic fallback, and feature extraction are treated as
  correctness-critical: golden-tested against independent truth and human-reviewed. The
  *model* itself is not on that path (quality is measured by offline eval, not pass/fail
  tests).
- **Never propose red code**: nothing that fails to compile or fails its tests.
- **Git discipline**: stage named files only (never `git add -A`); small, focused commits
  that each build and pass; no force-push, no `--no-verify`, no `reset --hard` without
  instruction. Pushes, PRs, and merges to `main` are human actions, requested explicitly.
- **Public vs private split**: `docs/` is public-facing; `internal/` holds metrics,
  planning records, and process notes and stays private. A doc-separation check keeps
  public docs from linking into private paths, and secrets never enter git at all.

## The CI gate

CI is fail-closed and mirrors the local dev loop exactly:

```text
ruff check .  →  black --check .  →  pytest
```

A red gate blocks the merge. The agentic reviewer runs beside it.

## Where humans stayed in the loop

- **Decisions**: requirements, architecture, ADRs, and priorities are human-owned; the
  agent drafts, the human decides.
- **Milestone exit**: the agent proposes exit *with evidence* (ticked criteria, passing
  test names, smoke output) and then stops; the human runs the manual checks and signs off.
- **Merges**: every merge to `main` is a human action.
- **Correctness-critical code**: never self-merged by the agent.

## What this project deliberately does NOT do

- **No LLM in the ranker.** It is a small, supervised, non-LLM model (auditable,
  CPU-trainable, and cheap to serve).
- **No model on the correctness path.** Deterministic plumbing is golden-tested; the model
  only reorders an existing candidate set behind a flag, with a deterministic fallback.
- **No changes to retrieval** and **no writes to MSA's index**: the ranker reorders only.
- **No automatic training or deployment**: training is manually triggered offline; the
  model is handed to MSA via an explicit, gated copy.

## Tools used

- **[Claude Code](https://www.anthropic.com/claude-code)**: the coding agent.
- **Python 3.11+**, a pure-Python (dependency-free) logistic-regression model today, with
  a LightGBM LambdaMART tier on the roadmap.
- **ruff** (lint), **black** (format), **pytest** (tests), wired into a fail-closed
  **GitHub Actions** CI gate.

## Further reading

- [README](../README.md): what the project is and how MSA consumes it.
- [Architecture](architecture.md) · [ADRs](adrs.md) · [Invariants](invariants.md) ·
  [Research](research.md) · [Testing](testing.md) · [Runbook](runbook.md).
