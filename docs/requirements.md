# Requirements — msa-ranker

> **Stage 3 artifact** of the development workflow
> (Requirements). What it must do (FR), the constraints it must hold (NFR), what's
> **in scope** for the first release, and what's an explicit **non-goal**. Driver:
> human specs, agent drafts — **refine freely**. Status: **draft**.
>
> Binding decisions live in the ledger — [`adrs.md`](./adrs.md) (ADR-001…014),
> [`invariants.md`](./invariants.md) (INV-1…10); the *why/landscape* is in
> [`research.md`](./research.md). Requirements **reference** those, not restate
> them. A requirement found infeasible in architecture is edited *here* (a feedback
> edge), not silently worked around.
>
> **ID scheme:** `FR-n` functional, `NFR-n` non-functional. Each is tagged
> **[loop-1]** (in the first complete loop) or **[later]** (pinned upgrade). IDs are
> stable and never reused — specs and contract tests cite them.
>
> `loop-1` names the **scope** — the first complete lifecycle loop (data → train →
> eval → serve → telemetry). The roadmap's **Milestone M-1** (planning vocabulary,
> Stage 6) is what **delivers** it: *completing loop-1 = exiting M-1* (gate in §6).

## 1. Context & users

- **One product, one primary user (may expand to a household of a few members).**
  msa-ranker reorders results for a personal media library (MSA) — centered on one
  primary user, but MSA may run on a home server used by a few family members. Labels
  come only from those users' logged opens — no crowd, no editorial relevance set. v1
  trains a **single pooled model**, but logs **per-user identity** so per-user
  personalization stays a future upgrade ([ADR-008](./adrs.md#adr-008)). Either way
  data is small, **cold-start is the dominant risk**, and "relevance" means *this*
  user's (or household's) intent.
- **It is an add-on to a working system.** MSA phase-1 search exists and returns an
  ordered candidate set; msa-ranker's job is strictly to **reorder** that set
  better, behind a flag, with MSA's current behaviour as the untouched fallback.
- **It owns a full ML lifecycle.** Data → train → eval → deploy → monitor →
  retrain, end to end, reproducibly — that lifecycle *is* the product as much as
  the ranking lift.

## 2. Functional requirements

### Eval & baseline (the metric comes first — INV-1)

- **FR-1 [loop-1] — Graded eval harness.** Compute **NDCG@k** (primary) and **MRR**
  (secondary) over held-out logged interactions, given a model's ordering of a
  query's candidate set vs the observed opens. Deterministic; `k` configurable.
- **FR-2 [loop-1] — Measured baseline.** Score **MSA's current heuristic ordering**
  with FR-1 and record it as the baseline-to-beat, **before** any model is trained.
  Every later model's quality is reported as Δ vs this baseline.

### Data, labels & reproducibility (owned, leak-free — INV-4)

- **FR-3 [loop-1] — Interaction logging.** Capture, per search: the **user/profile
  id** of who searched ([ADR-008](./adrs.md#adr-008)), the query text + decomposed
  query context, the **ordered result set shown** with each result's feature vector
  and **display position**, and which result(s) the user **opened** (the label).
  Append to the **event ledger** (MSA's output — ADR-012; ingested to the SoR, FR-18).
  Gated by a default-on privacy switch **`ranker.event_logging`** (ADR-014) — off ⇒ no
  logging, serving unaffected.
- **FR-4 [loop-1] — Deterministic feature extraction.** A pure transform from an
  MSA candidate + query context → a fixed feature vector. Same inputs → same
  vector (golden-testable). The exact engineered feature set is pinned in specs
  (Stage 5) from the inputs catalogued in [research §3](./research.md).
- **FR-5 [loop-1] — Label construction.** A defined, testable transform from raw
  logs → training labels, **position-bias-aware** via **Click > Skip-Above**
  ([ADR-007](./adrs.md#adr-007)): an opened result is positive; un-opened results
  ranked *above* the deepest open are (likely) negatives; results *below* the deepest
  open are unlabeled (dropped).
- **FR-6 [loop-1] — Versioned, *immutable*, replayable datasets.** Freeze a training set
  as a **manifest** that names exactly which events compose it — an **ingest watermark +
  the exact source `ev_id` set** (not just `search_id`s/filters, which keep mutating as
  the SoR ingests) — addressable by id; replaying an id yields a **byte-identical** row
  set regardless of later ingestion. No DVC ([ADR-006](./adrs.md#adr-006)).
- **FR-7 [loop-1] — Leak-free splitting.** Train/eval splits are **query-grouped** —
  the same query never straddles splits.

### Model lifecycle (smallest loop first — INV-8)

- **FR-8 [loop-1] — Baseline ranker.** A deliberately simple **pointwise** model
  (regularized logistic/linear over the feature vector) trained on a versioned
  dataset, whose bar is to **beat the FR-2 baseline** on FR-1.
- **FR-9 [loop-1] — Model registry/versioning.** Record each model: artifact
  ref/hash, hyperparameters, **training dataset id** (FR-6), and its eval scores
  (FR-1) — so any served model is traceable to the exact data it learned from.
- **FR-10 [loop-1] — Offline training.** Training runs out-of-band and manual
  (**CPU, any platform** per [ADR-003](./adrs.md#adr-003)), reading the SoR, emitting a
  small versioned **portable** artifact the MSA host loads. Reproducible from a dataset id.
- **FR-11-model [later] — LambdaMART upgrade.** LightGBM `lambdarank` over the same
  feature contract, swappable without re-plumbing ([ADR-002](./adrs.md#adr-002)).
- **FR-12-model [later] — Neural reranker.** A GPU cross-encoder over query +
  CLIP/caption/tag features. Deferred.

### Serving & integration (safe, additive — INV-2, INV-3, INV-6)

- **FR-13 [loop-1] — In-process serving behind a flag.** Load the model in MSA's
  process; a `config.yaml` flag routes the existing rerank seam
  (`engine.py` → `score_breakdown()`) to the learned scorer
  ([ADR-001](./adrs.md#adr-001)).
- **FR-14 [loop-1] — Deterministic fallback.** Flag **off** ⇒ MSA's current
  heuristic ordering runs **byte-for-byte unchanged** (INV-3). Any model
  load/scoring error fails **safe** to the heuristic, never to an error.
- **FR-15 [loop-1] — Reorder-only.** The served set is a **permutation of MSA's
  input candidates** — no adds, drops, or changes to retrieval/recall; no write to
  MSA's index DB (INV-6).

### Serving telemetry & monitoring (the substrate that retrains — INV-5)

- **FR-16 [loop-1] — Serving telemetry.** Record every production rerank to the SoR:
  user/profile id, flag state, model version, shown set + features + scores +
  positions, and subsequent opens. This is **the same event stream** as FR-3 — one
  source of truth, doubling as the next round of training labels.
- **FR-17 [loop-1] — Monitoring view.** A way to observe, over time, the served
  metrics and the model-vs-baseline trend (a CLI/report over the SoR is sufficient
  for v1) — the signal that decides when to retrain.

### Storage (durable, private — INV-5, INV-7)

- **FR-18 [loop-1] — System-of-record.** A dedicated **SQLite** SoR **outside the
  repo** (`~/.msa-ranker/`), WAL, opened via a single `open_db()` helper, with
  **additive versioned SQL migrations** ([ADR-005](./adrs.md#adr-005)). Schema
  sketch in [research §4](./research.md).

## 3. Non-functional requirements

- **NFR-1 — Safe degradation.** With the flag off, MSA is indistinguishable from
  today in behaviour and performance (INV-3). The learned path is purely additive.
- **NFR-2 — Deterministic correctness path.** Feature extraction, flag/fallback
  routing, serving wiring, and the telemetry write are deterministic and
  **golden-tested**; ranking *quality* is measured by **eval (graded)**, never a
  pass/fail test (INV-2).
- **NFR-3 — Latency.** Reranking MSA's top-K (~50–200 candidates) adds **negligible
  serving latency** (target: sub-millisecond to low-ms; dominated by the existing
  embedding + ANN search). A hard budget is set in architecture.
- **NFR-4 — Privacy & publishability.** The SoR and all raw telemetry live **outside
  the repo / gitignored**; nothing private leaks if MSA is open-sourced; no secrets
  in git, ever (INV-5). Only redacted/aggregate artifacts + `internal/` overlays are
  committed. Sensitive content stays in the private ledger, never exported (INV-10), and
  the user has a default-on **collection opt-out** (`ranker.event_logging`, ADR-014).
- **NFR-5 — Reproducibility.** Any training run is replayable from a dataset id
  (FR-6); feature extraction is deterministic (FR-4); model→data lineage is recorded
  (FR-9).
- **NFR-6 — Data durability & integrity.** Telemetry is append-only and
  irreplaceable; migrations are additive-only (a shipped migration is never edited —
  INV-7); the SoR survives MSA restarts/upgrades.
- **NFR-7 — MSA path integrity.** msa-ranker never degrades MSA's retrieval/recall
  and never holds a write handle to MSA's index DB (INV-6).
- **NFR-8 — Portability.** Cross-platform, pure-Python package + CLI (mac/linux/
  windows). Serving runs on the MSA host; training runs **anywhere on CPU** (GPU only
  for the deferred neural tier); the model artifact is small and **platform-portable**
  (train on one OS, serve on another). Mirrors MSA's toolchain.
- **NFR-9 — Testability.** Every contract-testable invariant (INV-2/3/4/6/7) has a
  failing-on-violation test alongside the code; the eval harness is hermetic and
  seed-repeatable.
- **NFR-10 — Maintainability.** Conventions mirror MSA (ruff/black/pytest, FastAPI
  integration style) so the two repos read alike.

## 4. Scope — first release

**In scope (the smallest complete loop, [loop-1] above):** the FR-1…FR-10, FR-13…
FR-18 set — i.e. eval harness + measured baseline → interaction logging with
positions → deterministic features → position-bias-aware labels → versioned dataset
→ simple pointwise ranker that beats baseline → served in-process behind a flag with
safe fallback → serving telemetry + a monitoring view → the external SQLite SoR.

The release ships when that loop is closed **and the model beats the baseline on
NDCG@k** (see §6).

## 5. Non-goals (explicit — initial release)

- **Model sophistication before the loop closes** — no LambdaMART tuning, neural
  reranker, or inverse-propensity weighting until [loop-1] is done end to end
  (INV-8). These are **[later]** upgrades with pinned paths, not gaps.
- **LLM fine-tuning or RAG** — a separate concern; not this project.
- **DVC / heavyweight data versioning** — manifests suffice at this scale
  ([ADR-006](./adrs.md#adr-006)).
- **A sidecar / separate serving service** — in-process only for now
  ([ADR-001](./adrs.md#adr-001)); revisit if a resident-GPU reranker lands.
- **Per-user / personalized modeling** — v1 pools all household users into **one**
  model; per-user models and user-as-feature personalization are **[later]** upgrades
  ([ADR-008](./adrs.md#adr-008)). Data capture *is* user-aware now (FR-3/FR-16), so
  this isn't foreclosed. Cross-device label sync remains out of scope.
- **Building or pushing to devdash** — it observes pull-based; this repo only keeps
  the normal session/git/PR/`internal/metrics/` trail it reads.
- **Changing MSA's retrieval, recall, or index** — reorder-only (INV-6).

## 6. Acceptance — first-milestone definition of done

The first milestone **M-1** (which delivers `loop-1`) exits (proposed by the agent
with evidence, signed off by the human —
agent-instructions §6) when:

1. FR-1 harness exists and is hermetic/seed-repeatable; FR-2 baseline is **recorded**.
2. The [loop-1] pipeline runs end to end: log → versioned dataset → train → eval →
   serve-behind-flag → telemetry.
3. The trained ranker **beats the recorded baseline on NDCG@k** on a leak-free
   held-out split (FR-7), Δ reported.
4. Flag-off is proven **byte-identical** to current MSA ordering (INV-3 contract
   test green); error paths fall back safe (FR-14).
5. Contract tests for INV-2/3/4/6/7 are green; the publishability check (INV-5)
   passes; nothing private is tracked by git.

## 7. Open questions → resolve in architecture (Stage 4)

> **Mostly resolved** in [`architecture.md`](./architecture.md) §11 (ADR-009…011,
> Proposed). Retained here with pointers for traceability; the feature *list* remains
> a specs-stage (Stage 5) item.

- **Feature set & encoding.** The exact engineered features (numeric/categorical
  encodings, query↔result interaction features) from the [research §3](./research.md)
  inputs — and how `source_scores`/tags are encoded. **Forward-looking
  (resolve-then-rank):** key person features and the **label/split key** on
  **resolved `person_id`s**, not raw possessive query strings, and **user-scope**
  them — so relationship-relative queries ("my dad") disambiguate by searcher and
  never conflate or leak across users ([ADR-008](./adrs.md#adr-008), INV-4).
- **Opens capture mechanism.** *How* an "open" is observed (a new MSA API
  endpoint? client signal? log tail?) — the precise integration point for FR-3/FR-16
  is undecided; MSA has **no interaction logging today**.
- **User identity capture.** MSA appears to have **no user/auth concept** today, yet
  v1 logs must be user-aware ([ADR-008](./adrs.md#adr-008)). *How* "who searched" is
  captured — a UI profile selector, a per-device/session signal, or a profiles
  concept in MSA — is undecided and needs confirming against MSA's actual model.
- **Cold-start behaviour.** What the served ranker does before enough labels exist
  (e.g. stay on heuristic until a data threshold) — a serving-policy decision.
- **Latency budget number** for NFR-3, and where feature extraction sits relative to
  the existing candidate loop.
- **Retrain trigger.** What signal/threshold in FR-17 prompts a retrain (manual for
  v1? data-volume? metric drift?).
- **SoR concrete schema** — tables/columns/indexes/migrations formalized from the
  [research §4](./research.md) sketch.
