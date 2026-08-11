# Contributing to msa-ranker

`msa-ranker` is the learned reranker for
[media-search-agent](https://github.com/openara-ai/media-search-agent) (MSA). It is in
active single-maintainer development under a specific, documented workflow, so the
highest-leverage ways to help are not always "send a PR."

## The highest-leverage contribution right now is opening an issue

Bug reports, "this doesn't fit my data" reports, and questions about the design are
very welcome. Particularly useful:

- **Reranking results that look wrong**: what you searched, what you expected, what
  you got. The reranker only reorders MSA's existing candidate set, so include enough
  to tell retrieval from ranking.
- **Integration friction with MSA**: install/version-pin issues, the optional-import
  guard, the serving flag/fallback.
- **Documentation gaps**: anything in the docs that was wrong, missing, or confusing.

## Design is the source of truth

This project is design-first. Substantive changes must respect (or explicitly amend)
the decision ledger:

- **Invariants**: [docs/invariants.md](docs/invariants.md). Non-negotiable; a change
  that needs to violate one is a design change, not an implementation detail.
- **ADRs**: [docs/adrs.md](docs/adrs.md). Architecture decisions and their rationale.
- **Specs + testing**: [docs/specs/](docs/specs/) and [docs/testing.md](docs/testing.md).
  Behaviour is anchored to per-feature `AC-nn.x` acceptance criteria and golden tests.

**Please open an issue to discuss before sending a non-trivial PR.** PRs that conflict
with an ADR or an invariant tend to need substantial rework.

## A note on AI-assisted contributions

This project is itself built with AI coding agents under a specific workflow. AI
assistance is not a problem, but agents working *without* the project's context (the
ADRs, the invariants, the existing patterns) tend to produce PRs that don't fit. If you
send an AI-assisted PR, read the relevant ADR(s) and spec first, and mention in the PR
what you used and what you reviewed.

## Dev environment

Pure-Python, zero runtime dependencies. Supported dev environments are macOS, Linux,
and Windows (PowerShell or WSL2).

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
```

The dev tools (`ruff`, `black`, `pytest`) are installed **into the virtualenv** by
`pip install -e ".[dev]"`, not globally. Activate the venv first (`. .venv/bin/activate`,
or `.venv\Scripts\Activate.ps1` on Windows) or the commands below will not be found.

The dev loop (mirrors CI; warnings are errors):

```bash
ruff check .
black --check .
pytest
```

## Workflow expectations

- **Branch from `main`** with a descriptive name (`feature/<topic>` or `fix/<topic>`).
- **Commit messages** follow [Conventional Commits](https://www.conventionalcommits.org/):
  `feat:`, `fix:`, `docs:`, `chore:`, and so on.
- **CI must pass before review**: the same `ruff` / `black` / `pytest` gate runs locally.
- **One thing per PR**: small, reviewable diffs land faster.
- **Correctness-critical paths** (the flag/fallback and feature extraction on MSA's
  serving path) get human review and are not self-merged.

## Privacy

Never commit interaction data. The event ledger and the training system-of-record live
**outside the repo** and are gitignored (INV-5 / INV-10). Tests use synthetic fixtures
only.

## Code of conduct

Be respectful and constructive. Disagreements about technical choices are expected and
welcome; personal attacks are not.
