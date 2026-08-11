# Changelog

All notable, user-visible changes to `msa-ranker` are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The public API surface that SemVer governs is the serving/telemetry contract —
`msa_ranker.serving`, `msa_ranker.ledger`, and `msa_ranker.features` (see
[ADR-010](docs/adrs.md#adr-010)). `feature_set_version` is the separate runtime
model-compatibility check.

## [Unreleased]

<!--
  Public, user-visible changelog: list only changes that a consumer of the
  package (chiefly media-search-agent) or an operator running the CLI would
  notice. Internal process work does not belong here. Curate the real entries
  here before cutting a release, then move them under a dated version heading.
-->

## [0.1.1] - 2026-08-10

First public release. (`v0.1.0` was tagged on the private development repo as a
release-path rehearsal and was never published; its planned content ships here.)

### Added

- First public release of the learned (non-LLM) learning-to-rank reranker for
  [media-search-agent](https://github.com/openara-ai/media-search-agent). Reorders
  MSA's existing candidate set behind a config flag, learning from logged
  interactions, with the existing heuristic as a deterministic fallback.
- Importable serving library (`msa_ranker.serving`, `.ledger`, `.features`):
  pure-Python, zero runtime dependencies, cross-platform (macOS/Linux/Windows).
- `msa-ranker` CLI: `ingest`, `train`, `report`, `deploy` (evaluation runs inside
  `train`).
- Public documentation set: visitor-oriented README (the MLOps loop and a
  before/after reranking example), a getting-started guide, an
  agentic-development guide, and the public agent contract.

### Privacy

- **Local interaction logging is ON by default when the package is installed and
  enabled by the host.** When MSA is configured to use the reranker, it appends an
  **append-only local event ledger** (searches shown + opens) in MSA's own data
  area. The app never transmits the ledger anywhere and it is never committed to
  any repository; copying it to a separate training machine is a manual operator
  step. This is what makes the reranker learn from your usage.
- **Off-switch:** set `ranker.event_logging: false` to disable all ledger writes
  ([ADR-014](docs/adrs.md#adr-014)). Logging failure never degrades MSA search or
  opens (fail-open).
- Raw query text and media/person references stay in the private local ledger/SoR
  and are never exported; generated reports are aggregate/redacted (INV-5 / INV-10).
