# Spec 02 — Feature extraction  *(crux)*

> Realizes **FR-4**, enforces **INV-2** (deterministic, golden-tested). The pure
> transform `(candidate, query_context) → feature vector`, used identically at serve
> time (spec 06) and to populate `shown.features` (spec 01). Status: draft.

## Purpose

Pin the **exact v1 feature set, order, types, and missing-value handling** so the
vector is deterministic and reproducible — same inputs, same vector, on the serving
host and the training box alike.

## Interface

```python
# msa_ranker/features.py  (C1)
FEATURE_SET_VERSION = "v1"
FEATURE_NAMES: tuple[str, ...]                 # the 16 names below, frozen order (immutable)

def extract(candidate: dict, ctx: QueryContext, now: float,
            people: PersonLookup) -> list[float]:
    """Pure: returns a vector of len(FEATURE_NAMES). No clock/RNG; `now` is passed in.
       `people` is a read-only MSA-DB lookup (INV-6). Determinism = INV-2."""

class PersonLookup(Protocol):
    def person_ids_for_media(self, media_id: str) -> set[str]: ...   # read-only
```

## Contract — v1 feature list (frozen as `feature_set_version = "v1"`)

| # | Name | Source | Type | Missing → |
|---|---|---|---|---|
| 0 | `sim` | `raw_similarity_score` | float | 0.0 |
| 1–4 | `src_img`,`src_vid`,`src_cap`,`src_asr` | `source_scores[*]` | float | 0.0 |
| 5 | `num_sources` | count of contributing sources | int | 0 |
| 6 | `is_person_expand` | `source`/`source_scores` | 0/1 | 0 |
| 7 | `person_hits` | \|resolved query `person_id`s ∩ result `person_id`s\| | int | 0 |
| 8 | `has_person_intent` | query has ≥1 resolved person | 0/1 | 0 |
| 9 | `tag_overlap` | \|`visual_tokens` ∩ result `tags`\| | int | 0 |
| 10 | `is_video` | `type == "video"` | 0/1 | 0 |
| 11 | `recency_days` | `(now − date)` in days | float | 0.0 (+ flag 12) |
| 12 | `has_date` | `date` present | 0/1 | 0 |
| 13 | `has_gps` | `gps_lat`/`gps_lon` present | 0/1 | 0 |
| 14 | `place_match` | query place intent matches result `place` | 0/1 | 0 |
| 15 | `query_len` | count of `visual_tokens` | int | 0 |

- All numeric; booleans 0/1; **no scaling** (tree); the linear baseline standardizes
  internally (spec 05) — the *logged* vector is unscaled.
- Missing → filled per table, never NaN; each soft field has a `has_*` companion.
- **Resolve-then-rank (ADR-008):** person features key on `person_id`s, never names.
- **`position` is NOT a feature** (ADR-007) — debias signal only.

## Usage

Called per candidate at the rerank seam (spec 06, via `Ranker.score`) and to fill each
`shown.features` (spec 01). `now` is supplied by the caller so the function stays pure.
Adding/removing/reordering a feature **bumps `FEATURE_SET_VERSION`**; spec 06's gate
refuses a model whose manifest version ≠ the running extractor's.

## Implementation notes (Stage 7 / S-3 — as built, binding)

Decisions made while wiring the MSA integration; the design docs follow the code here.

- **Computed once at the engine seam, not in the API.** MSA's `QueryEngine.search_for_serving`
  computes the vector at the rerank seam — where `source_scores`, `faces`, and the query
  context all exist — and attaches it to each result. The shim logs that vector verbatim and
  serving (S-5) will reuse the *same* call, so logged features == served features (no
  train/serve skew). The API never recomputes from a reduced result view.
- **`person_id` carried with the name.** MSA's `_enrich_people` resolves each face to
  `(name, person_id)` in one query, and query people are resolved name→`person_id` via a
  name→id map. So `person_hits` keys on resolved ids (AC-02.4 / ADR-008) with no extra lookup.
- **UI filters merged into the context.** Filter-panel selections are merged with text-inferred
  intent when building `QueryContext`, so a search reflects filter-expressed intent, not only
  typed text. In the *current* Search UI that means **place** (→ the `place_match` feature) and
  **date** (→ logged `date_intent`; note no v1 feature consumes date *intent* yet — `recency_days`
  uses the item's own date). The Search tab has **no person filter** (people live in Browse, which
  isn't a logged search), so person intent comes from typed names; the **people-merge is
  future-proof** — inert until/unless a person filter is added to Search.
- **v1 semantics settled in review:** `sim` comes **only** from `raw_similarity_score` (no
  fallback to `score` — that is ranking output and would leak into training); `recency_days`
  is **unclamped** (`(now-date)/days`, negative for future/clock-skewed dates); `num_sources`
  counts only the four named `_SOURCES`; `FEATURE_NAMES` is an immutable `tuple`; naive ISO
  dates are pinned to **UTC** (AC-02.1 determinism).
- **Never breaks search (INV-9).** Both the `msa_ranker` import and the per-search feature
  computation are guarded — an absent package or a runtime extractor failure disables features
  for that search (logged) and the shim then skips logging rather than writing empty-feature
  rows; search itself never errors.
- **Orphaned `open` events (S-4 handoff).** When logging is skipped, `search_id` is still
  returned to the client; a later `POST /track/open` for it produces an **orphaned `open`**
  (no matching `search`/`shown`). Not a bug — S-4 label construction joins on `search_id` and
  drops orphans naturally — but S-4 must expect them.

## Ownership

- **Implements:** agent — `msa_ranker.features` (C1).
- **Reviews / owns:** human owns the **golden vectors** (independent truth) and the
  frozen feature list; the layout contract is correctness-relevant (train/serve skew).

## Acceptance criteria

- **AC-02.1** Determinism: identical `(candidate, ctx, now)` → byte-identical vector (INV-2).
- **AC-02.2** Fixed layout: length + `FEATURE_NAMES` match `v1`, in order.
- **AC-02.3** Missing-field handling: absent `source_scores`/`date`/`gps`/`place` →
  table fills + correct `has_*` flags.
- **AC-02.4** `person_hits`/`has_person_intent` from resolved `person_id`s, not names.
- **AC-02.5** No `position` (or any rank signal) in `FEATURE_NAMES`.
- **AC-02.6** Pure: no write to `media.sqlite` (INV-6); no internal wall-clock read.

## Tests

| Test | Type | Case → asserts | AC | Owner |
|---|---|---|---|---|
| golden vector | unit/golden | reference candidate → hand-computed vector | 02.1/02.2 | human |
| determinism | unit | repeat call → identical bytes | 02.1 | agent |
| missing fields | unit/golden | each absent field → fill + `has_*` (one case per path) | 02.3 | human |
| person resolve | unit | name→id fixture → correct `person_hits`/intent | 02.4 | agent |
| layout guard | contract | `FEATURE_NAMES`+version stable (catches accidental drift) | 02.2/02.5 | agent |
| purity | contract | extraction opens no write handle; reads no clock | 02.6 | human |

**Fixtures:** reference candidate dicts (full + each-field-missing) with **hand-computed
golden vectors** (independent truth); a `media_id → person_id` map; a fixed `now`.
