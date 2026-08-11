# Spec 01 — Event ledger format

> Realizes **FR-3 / FR-16**, **ADR-012** (decouple), **ADR-009** (`/track/open`),
> **ADR-014** (privacy). The append-only JSONL log MSA emits — the single
> label/telemetry source. Status: draft.

## Purpose

Define the on-disk contract of MSA's **event ledger** so the training-side ingester
(spec 07) can fold it deterministically, and so the append path is dumb, robust, and
off MSA's correctness path (INV-9).

## Interface

```python
# msa_ranker/ledger.py  (MSA side, C3) — best-effort, NEVER raises (INV-9)
def append_search(search_id: str, user_id: str, query: str, ctx: dict,
                  flag_on: bool, model_version: str | None, k: int, ts: float) -> None
def append_shown(search_id: str, rows: list[ShownRow], ts: float) -> None   # k rows, 1 call
def append_open(search_id: str, media_id: str, user_id: str, ts: float,
                dwell_ms: int | None = None) -> None
# Each wraps its write in try/except → log + health-counter + drop. The /search
# appends run in a FastAPI BackgroundTask (post-response) so they add no latency.
```

## Contract

- **Location & ownership:** the ledger is **MSA's output**, so it lives in **MSA's data
  area** — a path **MSA owns**, configured in MSA's `config.yaml` (e.g. `ranker.ledger_dir`,
  default under MSA's data dir alongside `index/` / `logs/`), **gitignored** in the MSA
  repo (INV-5). It is **not** under `~/.msa-ranker/` — that namespace is the
  **training-owned** SoR. Training reads a copy/snapshot via the handoff (architecture §11).
- **Format:** UTF-8, **one JSON object per line** (JSONL), **append-only** — no
  rewrites/deletes.
- **Common fields:** `ev` (`"search"|"shown"|"open"`), `ev_id` (ULID, **globally
  unique** — the idempotency anchor: **stored in every SoR ingest table** and ingest
  upserts `ON CONFLICT(ev_id)`, spec 07 / architecture §10), `ts` (UNIX float).
- **`search`:** `search_id`, `user_id` (default `"default"` — ADR-008), `query` (raw),
  `ctx` (`{people:[person_id], date_intent, visual_tokens}`), `flag_on`,
  `model_version` (or `null`), `k`.
- **`shown`:** `search_id`, `media_id`, `position` (**0-based** — the *served* rank),
  `score` (the served score), **`heuristic_score`** (the deterministic `score_breakdown()`
  score for this result — **always recorded, even when the learned model served**),
  `features` (name→value, spec 02). `heuristic_score` lets baseline eval (spec 04) rank by
  the heuristic on *any* traffic — without it, learned-served searches can't reconstruct
  their baseline and the retraining loop (FR-16) / `beats_baseline` gate break (NN1).
- **`open`:** `search_id`, `media_id`, `user_id`, `action` (`"open"`), `dwell_ms?`.
- **Privacy off-switch (ADR-014):** all appends are gated by MSA config
  **`ranker.event_logging` (default `true`)**. When `false`, **no** events are written
  (`search`/`shown`/`open`); serving is unaffected and `/track/open` still returns 204.
  Orthogonal to the serving flag (`ranker.enable_learning_to_rank`).

Example in [architecture §10](../architecture.md#10-the-two-stores--the-model-artifact).

## Rotation, naming & retention

- **Naming:** `events-YYYYMMDD.jsonl` (one per day); if the size cap is hit within a day,
  roll to `events-YYYYMMDD-NN.jsonl` (`NN` = `00`, `01`, …). **Lexical order =
  chronological**, so ingest (spec 07) globs `events-*.jsonl*` and processes in filename
  order.
- **Size cap:** rotate at a configurable cap (**default 64 MB**) or on the day boundary,
  whichever first. (Sizing: a `shown` line ≈ 0.3–0.6 KB; a k=50 search ≈ ~25 KB; even
  heavy daily use is a few MB/day — the cap is a safety backstop, not a routine event.)
- **Compression:** rotated (non-current) files may be gzipped → `events-*.jsonl.gz`;
  ingest reads both `.jsonl` and `.jsonl.gz`.
- **Retention:** the ledger is the **irreplaceable raw archive** (ADR-012) — **kept
  indefinitely, never auto-deleted**. Compaction/archival of fully-ingested old files is
  a `[later]` option (the SoR is the derived view; the ledger is the source of truth).
- **Concurrency — single-writer:** MSA serves concurrently (multiple threads/workers), so
  appends **and** rotation are serialized through one **process-wide writer** (an append
  lock or a single-writer queue). Each event line is written atomically (one `write` +
  newline); rotation (size/day rollover, opening the next file) happens **under the same
  lock** so lines never interleave or tear across a roll. The append stays best-effort —
  lock contention/failure drops-and-logs, never blocks `/search` (INV-9).

## Usage

The `/search` handler mints `search_id`, returns it with the results, and — **after the
response** (BackgroundTask) — calls `append_search` + `append_shown` (positions 0…k-1).
`/track/open` calls `append_open`. The round-trip is drawn in
[architecture §1 / §4](../architecture.md#1-the-sketch-topology); not duplicated here.

## Ownership

- **Implements:** agent — `msa_ranker.ledger` (C3) + the MSA-side `/search`/`/track/open`
  glue (MSA repo).
- **Reviews / owns:** human — on MSA's request path; the **INV-9 best-effort/no-raise**
  behaviour is correctness-relevant (review, don't self-merge the append wiring).

## Privacy & sensitive data (ADR-014 / INV-10)

| Field | Sensitivity | Note |
|---|---|---|
| `query`, `ctx.visual_tokens` | **High** | raw user intent in free text |
| `media_id`, `ctx.people`/`person_id` | **Medium** | re-identifiable via MSA's DB |
| `open`, `dwell_ms`, `user_id` | **Medium** | behavioural; which household member |
| `features` | **Low** | derived numerics only — no raw tags/place/GPS/captions (spec 02) |

Ledger is **local & private** (outside repo, gitignored — INV-5), **not a devdash
input**. Raw `query` is kept **verbatim** (re-derivation, ADR-012) but **never leaves
the private store**; reports are aggregate/redacted (INV-10). No secrets in the ledger.

## Acceptance criteria

- **AC-01.1** Every line parses + validates against its `ev` schema (golden fixtures).
- **AC-01.2** `ev_id` unique; `search_id` correlates a search to its `shown`/`open`.
- **AC-01.3** One `/search` of size k → exactly 1 `search` + k `shown`, positions 0…k-1.
- **AC-01.4** `position` present on every `shown` (required by spec 03).
- **AC-01.5** Unwritable ledger ⇒ append dropped, `/search` still returns, `/track/open`
  ⇒ 204 (INV-9; tested in spec 06).
- **AC-01.6** A torn trailing line doesn't corrupt earlier lines.
- **AC-01.7** Redaction (INV-10): no **generated/exported** report or devdash input
  contains **real** `query` text / `media_id` / `person_id`; hand-authored design docs
  may use synthetic placeholders (`<query_text>`, `<person_id>`) — exempt.
- **AC-01.9** Concurrency: under N concurrent appends (incl. a rollover), every line is
  well-formed JSON (no interleaving/torn lines) and no event is lost — single-writer
  serialization holds.
- **AC-01.8** Off-switch (ADR-014): `ranker.event_logging=false` ⇒ **zero** events
  written (search/shown/open); `/search` still returns; `/track/open` ⇒ 204. Default
  (`true`) logs normally.

## Tests

| Test | Type | Case → asserts | AC | Owner |
|---|---|---|---|---|
| event schema | unit/golden | each event type validates; bad shape rejected | 01.1 | agent |
| correlation | unit | k-result search → 1 search + k shown, positions 0…k-1, unique ev_id | 01.2/01.3/01.4 | agent |
| round-trip | property | generator → ledger → expected event counts | 01.1–01.3 | agent |
| torn line | unit | truncated last line → earlier lines intact | 01.6 | agent |
| redaction | contract | scan reports/git-tracked files → no raw query/media_id/person_id | 01.7 | human |
| append fault | integration | unwritable dir → drop, no raise (cross-ref spec 06 AC-06.6) | 01.5 | agent |
| off-switch | contract | `event_logging=false` → zero events; `/search` ok; `/track/open` 204 | 01.8 | human |
| concurrency | property | N concurrent appends + a rollover → all lines well-formed, none lost | 01.9 | human |

**Fixtures:** the seeded synthetic event generator (shared with spec 07); golden
ledgers; a per-event JSON Schema; a read-only temp ledger dir for the fault path.
