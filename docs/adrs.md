# Decision ledger — ADRs (msa-ranker)

> Architecture Decision Records. Part of the **decision-ledger backplane** of the
> development workflow — established
> here, it **accretes**: new decisions add new ADRs; a change to a decision is a
> *new* ADR that supersedes, never an in-place rewrite of history. Binding rules
> that fall out of these decisions live in [`invariants.md`](./invariants.md).
>
> Each ADR: **Status · Context · Decision · Consequences**, plus the invariants it
> establishes. Ids are stable and never reused. ADR-001…007 were opened at Stage 2
> (research) closeout; expect more through architecture and build.
>
> **Dates are relative build-days** (`Dn` from kickoff), per the
> planning vocabulary Window rule — public
> docs plan in relative time, absolute dates stay internal. `D1` = project kickoff
> (this ledger's first day).

---

## ADR-001 — Learned reranking attaches in-process at the existing seam, behind a flag {#adr-001}

- **Status:** Accepted (D1)
- **Context:** MSA already reranks at `engine.py:542-551`, calling
  `score_breakdown()` (`rerankers.py:38-70`) per candidate, then sorts + top-K. MSA
  is a single embedded-Qdrant process holding a file lock. A learned scorer has the
  same shape (features → score). Reranking ~50–200 candidates with a tree/linear
  model is microseconds–low-ms, dwarfed by the embedding + ANN search already done.
- **Decision:** Serve the ranker **in-process**, loaded once at MSA startup. A
  `config.yaml` flag (e.g. `ranker.enable_learning_to_rank`, default **off**)
  routes the rerank seam to the learned scorer; **off → the existing
  `score_breakdown()` heuristic runs unchanged** as the deterministic fallback.
  A sidecar is rejected for now (RPC latency per search, a second process to
  run/version, no current need).
- **Consequences:** Lowest latency, no new process, matches MSA's single-box grain.
  The flag indirection localizes any future swap (e.g. a resident-GPU neural
  reranker → revisit a sidecar then). Establishes **INV-2** (deterministic
  plumbing) and **INV-3** (flag-off ≡ today).

## ADR-002 — Model progression: trivial baseline → LightGBM LambdaMART → neural (deferred) {#adr-002}

- **Status:** Accepted (D1)
- **Context:** A single-user library starts data-starved with implicit,
  position-biased labels (see ADR-007). LambdaMART (LightGBM `lambdarank`) is the
  standard tabular LTR model and the right *eventual* choice, but overfits
  tens–hundreds of labelled searches and presumes a proven label pipeline. The
  "smallest complete loop first" principle governs.
- **Decision:** The first complete loop ships a **deliberately simple pointwise
  baseline** (regularized logistic/linear over a handful of features) whose only
  bar is to **beat the measured heuristic baseline**. **LightGBM LambdaMART** is
  the pinned next model once data accrues. A **neural cross-encoder** over query +
  CLIP/caption/tag features is the GPU-enabled stretch upgrade — **deferred**.
- **Consequences:** Loop and label pipeline (the real risk) are validated before
  model sophistication. The feature contract is shared across the progression, so
  upgrades are model swaps, not rewrites. Establishes **INV-8** (loop before
  sophistication).

## ADR-003 — Train offline (CPU, anywhere); serve a small artifact in-process {#adr-003}

- **Status:** Accepted (D1); compute framing **refined (D2)** — see below.
- **Context:** The **planned model path is CPU-only**: the v1 linear/logistic baseline
  and the **LightGBM LambdaMART** upgrade (ADR-002) train trivially on CPU on *any*
  platform (mac/linux/windows). A **GPU is needed only for the deferred neural
  cross-encoder tier**, and even there it's a speed/practicality accelerator (CUDA on
  linux/windows, Apple-Silicon MPS on mac, or slow CPU), not a hard requirement.
  Serving (ADR-001) is in-process on the MSA host.
- **Decision:** **Training is out-of-band and manual** (ADR-011), reading a ledger copy
  → its SoR (ADR-012), emitting a **versioned, small, portable artifact** the MSA host
  loads. **Training runs wherever is convenient** — the dev Mac, the linux server, or
  (only for the neural tier) a GPU box — *independent of where serving runs*; only the
  **model+manifest** crosses to the server. In-process *serving* ≠ in-process
  *training*. The artifact is **platform-portable** (mac-trained → linux-served).
- **Consequences:** No GPU is required for the entire v1→LambdaMART path; the GPU
  ceiling stays open for the neural upgrade without entangling MSA's runtime. Clean
  train/serve separation; artifacts versioned + tied to their training dataset (ADR-006).

## ADR-004 — Ranking metric: NDCG@k primary, MRR secondary; baseline before any model {#adr-004}

- **Status:** Accepted (D1)
- **Context:** Users open one/few items per search. NDCG@k is graded and
  position-discounted and is exactly what LambdaMART optimizes — keeping eval and
  training objective aligned. MRR captures "how high was the first opened result."
- **Decision:** **NDCG@k** is the primary eval metric, **MRR** secondary. The
  graded eval harness and a **measured baseline on MSA's current ordering** are
  built and recorded **before any model is trained**. Ranking quality is **eval
  (graded)**, never a pass/fail test.
- **Consequences:** A model's value is always stated as Δ vs a real baseline.
  Establishes **INV-1** (baseline + metric first) and reinforces **INV-2**
  (quality is eval, not test).

## ADR-005 — Product data: dedicated external SQLite system-of-record (mirrors devdash) {#adr-005}

- **Status:** Accepted (D1); storage **shape amended by [ADR-012](#adr-012)** (D2) —
  the single shared SoR is split into an MSA-owned JSONL ledger + a training-owned
  SoR; the **principles below (additive migrations, private/outside-repo,
  irreplaceable, devdash precedent) are retained**.
- **Context:** The ranker owns two intertwined data classes: **ML lifecycle**
  (interaction-log labels, dataset versions, eval scores per model version) and
  **serving telemetry** (append-heavy, irreplaceable, private; doubles as
  monitoring substrate + future labels). This is **not** `internal/metrics/` (the
  process-metrics standard), **not** MSA's `index/media.sqlite` (couples to MSA's
  index + lock), and **not** devdash (a separate observer). devdash's
  `docs/data-model.md` is the proven precedent for exactly this class of data.
- **Decision:** A **dedicated SQLite system-of-record outside the repo** (e.g.
  `~/.msa-ranker/msa-ranker.sqlite`), **WAL mode**, opened through a single
  `open_db()` helper, with **plain versioned SQL migrations** (`migrations/0001_*.sql`,
  tracked in a `_migrations` table, **additive-only**). The SoR is **gitignored /
  outside the repo** so append-heavy private telemetry can't leak when MSA goes
  public. Only **redacted/aggregate** artifacts (eval scorecards, dataset
  *manifests*) and human-edited overlays are committed (under `internal/`). The
  label log and the serving-telemetry log are the **same events** viewed at two
  times — one source of truth.
  - First-cut tables: `search` (incl. **user/profile id** — ADR-008), `result_shown`
    (incl. **position** + feature vector), `interaction` (the label), `dataset`
    (versioned manifest), `model`
    (artifact ref + params + training dataset id), `eval` (model_id, metric,
    value). Refined in architecture.
- **Consequences:** Clean public/private split; irreplaceable data lives outside
  the repo; schema-versioning via migrations is the reproducibility spine.
  Establishes **INV-4** (owned/reproducible/leak-free), **INV-5** (SoR private,
  outside repo), **INV-7** (additive migrations only).

## ADR-006 — Reproducibility via dataset-manifest snapshots, not DVC {#adr-006}

- **Status:** Accepted (D1)
- **Context:** Reproducible, replayable training sets are required (INV-4). DVC is
  one option but adds tooling overhead unjustified at single-user data scale.
- **Decision:** A `dataset` row in the SoR (ADR-005) freezes a **manifest** that names
  **exactly which events** compose the set — an **ingest watermark + the exact source
  `ev_id` set** (or the materialized label rows), **not** mutable `search_id`s/filters
  (which keep changing as the SoR ingests). This makes each run **replayable to a
  byte-identical row set** by id, immune to later ingestion. Models reference their
  training dataset id. **DVC is not adopted**; revisit only if datasets grow large.
- **Consequences:** Versioned, **immutable**, replayable datasets without DVC ceremony
  (FR-6/INV-4). Each model is traceable to the exact data it learned from. *(Corrects an
  earlier framing where a `search_id`+seed manifest looked replayable but wasn't —
  surfaced in review.)*

## ADR-007 — Position-bias-aware implicit feedback; store result position {#adr-007}

- **Status:** Accepted (D1)
- **Context:** Labels are implicit (opens only). Higher-ranked results are opened
  more *regardless of relevance*; a naive model would re-derive the current order.
  A non-open is not a clean negative.
- **Decision:** **Persist each result's display `position`** on `result_shown`.
  Implicit feedback is interpreted **position-bias-aware** via **Click > Skip-Above**
  (Joachims): an opened result is a **positive**; an **un-opened result ranked ABOVE
  (more prominent than) the deepest opened result** was examined-but-passed → a
  **negative**; results **below the deepest open** are likely *unexamined* →
  **unlabeled** (dropped, not negative). Inverse-propensity weighting is a deferred
  upgrade. *(Rationale: users examine top-down, so non-opens below the last open carry
  no preference signal — labeling them negative would re-introduce the very position
  bias this ADR exists to remove.)*
- **Consequences:** The schema must capture position at serve time (ties into
  ADR-005). Training-label construction is a defined, testable transform over the
  logs, not an afterthought. Reduces the risk of the model learning position
  instead of relevance.

## ADR-008 — Multi-user: pooled model in v1, user-aware data capture; per-user deferred {#adr-008}

- **Status:** Accepted (D2)
- **Context:** MSA may run on a home server used by **2–5 family members**. A single
  **pooled** model learns the *average* of their tastes and can be dominated by the
  heaviest user; but small data means **cold-start dominates**, so fragmenting into
  per-user models is premature. Critically, **user-blind logs would permanently
  foreclose** any later per-user modelling — historical labels can't be un-mixed.
  MSA today appears to have **no user/auth concept**, so *how* identity is captured is
  itself unresolved.
- **Decision:** v1 trains **one pooled model** over all household interactions
  ([ADR-002](#adr-002) unchanged). **Data capture is user-aware from day one:** a
  **nullable `user/profile id`** on `search` / `result_shown` / `interaction`
  (defaulting to a single `default` profile when no identity is available).
  **Per-user models** and **user-as-feature** personalization are **deferred**
  upgrades; once multi-user data exists, eval **macro-averages across users** so a
  heavy user doesn't dominate. *How* "who searched" is captured (UI profile selector,
  per-device/session signal, or an MSA profiles concept) is deferred to architecture.
- **Consequences:** Keeps the loop small (INV-8) while preserving the per-user
  upgrade path — the one non-negotiable is that the SoR schema carries a user/profile
  column **now** ([ADR-005](#adr-005)). Telemetry privacy (INV-5) gains a
  **household-internal** dimension (several people's behaviour in one store).
  Introduces an **architecture dependency**: an identity source must be settled
  before user-aware logging is real. Supersedes the strict "single-user only" framing
  in requirements.
- **Note — relationship-relative queries (forward-looking).** MSA will eventually
  resolve possessive/relationship terms ("my dad's photos") against the *searcher's*
  people graph, so the same surface query is **inherently single-user-relative** —
  it means different things per user. This makes user identity a **correctness**
  dependency for query *interpretation*, not only a personalization one. Implication
  for the ranker: **resolve-then-rank** — features and labels key on the **resolved
  absolute entities** (`person_id`s), never the raw possessive string, and the
  label/split key is **user-scoped** so two users' "my dad" never conflate (INV-4).
  Resolution is MSA's job; once resolved to absolute features, the **pooled** model
  stays correct across users (personalization-by-relationship lives in query
  understanding, upstream of the ranker).

## ADR-009 — Open-signal capture: `search_id` correlation + `/track/open` {#adr-009}

- **Status:** Accepted (D2) — ratified after review (incl. the MSA API + frontend change).
- **Context:** MSA's frontend opens a result by fetching `/images/{id}` or
  `/videos/{id}`, but `/search` returns **no correlation id** — so an open can't be
  tied to the search and **position** it came from, and position is required for
  debiasing ([ADR-007](#adr-007)). Passively intercepting the file endpoints lacks
  search context, fires on thumbnails / re-opens / browser cache, and can't recover
  position.
- **Decision:** `/search` returns a **`search_id`** (ULID). At serve time the
  telemetry writer logs `search` + `result_shown` (with **position** + features) keyed
  by `search_id`. A new **`POST /track/open {search_id, media_id}`** records the
  `interaction` (the label). The frontend holds the `search_id` from the response and
  posts it when a result is opened. **Passive file-endpoint interception is rejected**
  as the primary signal (a later dwell augmentation via drawer-close is optional).
- **Consequences:** Accurate `(search, position, media)` label tuples; enables
  position-bias-aware labels. Requires a **small MSA API + frontend change** (the
  shim, [ADR-010](#adr-010)). `search_id` also correlates future signals (dwell,
  multiple opens).

## ADR-010 — `msa_ranker` is an importable library; thin in-process shim {#adr-010}

- **Status:** Accepted (D2).
- **Context:** [ADR-001](#adr-001) sets in-process serving. The ranker logic should be
  reusable, testable, and versioned independently of MSA, yet execute inside MSA's
  process.
- **Decision:** `msa_ranker` is an **installable Python package** exposing serving +
  telemetry + lifecycle APIs; MSA depends on it and calls it from a **thin shim** at
  the seam and the new endpoints. The library reads MSA's `media.sqlite`
  **read-only** (`connect_readonly`) for feature enrichment and **writes only its own
  SoR** — never a write handle to MSA's DB (INV-6).
- **Packaging & distribution (D2):** `msa_ranker` is a **pure-Python, cross-platform**
  package (mac/linux/windows) + CLI; the **public API (`serving`/`ledger`/`features`) +
  semver** is the contract (`feature_set_version` is the separate runtime model-compat
  check). **Development = editable install on the Mac** (`pip install -e`, both repos
  local — fast iteration, full loop testable on CPU). **Release = a pinned
  `msa_ranker==X.Y.Z`** installed on the linux server when MSA is updated (serving
  role), mirroring MSA's `requirements-api.txt` runtime split; pin **tightly** (it's our
  moving contract on MSA's serving path). The **exact server-install mechanism is TBD**
  (default: a versioned wheel bundled with MSA's deploy + `pip install`; alt: private
  index / git-tag). Training (CLI) runs wherever (ADR-003) — server install is for
  *serving*; only the model+manifest must reach the server.
- **Consequences:** Clean repo boundary (ranker testable in isolation; MSA glue
  minimal); independent versioning; reinforces INV-6. The shim API is a contract that
  must stay stable across versions. Editable-over-shared-folder finickiness is a
  non-issue (dev is single-machine on the Mac).

## ADR-011 — Serving cold-start gate + manual retrain (v1) {#adr-011}

- **Status:** Accepted (D2).
- **Context:** Early models are data-starved and may not beat the baseline; serving a
  regressing model would harm search. Retrain cadence must keep the human in the loop
  (INV-8).
- **Decision:** **(a) Cold-start gate** — with the flag on, the learned model serves
  only if an artifact exists **and** its registry eval **beats the recorded baseline**
  ([ADR-004](#adr-004)); otherwise the heuristic. **(b) Manual retrain** — training is
  human-initiated via the CLI for v1; data-volume / metric-drift auto-triggers are
  deferred (monitoring, FR-17). A minimum labelled-search threshold gates the first
  training run.
- **Consequences:** The model can never *silently* degrade search (gate); "flag on"
  and "model good enough" are decoupled; the human owns when a new model ships. Adds a
  registry lookup at startup/serve.

## ADR-012 — Decouple the planes: JSONL event ledger (MSA) + training-owned SoR + model handoff {#adr-012}

- **Status:** Accepted (D2) — supersedes the single-SoR **shape** of [ADR-005](#adr-005);
  its principles are retained.
- **Context:** ADR-005 had both planes read **and** write one SQLite SoR — coupling two
  otherwise-independent workflows, complicating the single-writer/resilience story, and
  making offline training reach into MSA's runtime store. The planes in fact share only
  two things: labeled data and the trained model.
- **Decision:** Model them as **independent producers exchanging two file artifacts**:
  1. **MSA emits an append-only JSONL event ledger** (`search`/`shown`/`open` events,
     `search_id`-correlated, `ev_id`-unique) — its single output, single writer, no
     migrations. MSA emits **events, not labels**; label construction
     ([ADR-007](#adr-007)) is training-side.
  2. **Training ingests** a copy of the ledger into **its own SQLite SoR** (additive
     migrations + the dataset/model/eval registry), derives labels, trains, evals, and
     emits a **model artifact + eval manifest** — its single output.
  3. MSA consumes model+manifest; the manifest drives the cold-start gate
     ([ADR-011](#adr-011)) with **no access to the training SoR**.
  ADR-005's storage principles carry to the training SoR **and** the ledger: additive
  migrations (INV-7), private/outside-repo (INV-5), irreplaceable. Maps to devdash's
  raw-archive (L1) → derived (L2) precedent.
- **Consequences:** single writer per store; full plane independence (training runs off
  a ledger copy, never touches MSA's runtime); trivial portability (two files over a
  shared folder — architecture §11); the "telemetry = labels" property is preserved (the
  ledger is both); ingest is idempotent (`ev_id` watermark). `/track/open`
  ([ADR-009](#adr-009)) now appends an `open` event to the ledger.

## ADR-013 — Ledger/telemetry is best-effort and fail-open {#adr-013}

- **Status:** Accepted (D2).
- **Context:** MSA's serving must never break, slow, or error because of the ranker or
  its ledger (INV-2/3). Failure cases: package absent, ledger path missing/unwritable,
  model/manifest absent, scoring error.
- **Decision:** The ledger append is **best-effort, off the correctness path**.
  `/search` appends events in a **post-response background task**, guarded — a failure is
  logged + counted and **dropped**, never raised. `/track/open` is fire-and-forget
  (**204 even on a dropped append**; the media open is a separate `GET`). A **missing
  package** ⇒ guarded import ⇒ MSA byte-identical to today; **missing/stale model** ⇒
  heuristic (FR-14); **unwritable ledger** ⇒ drop-and-log + health counter.
  **Drop-and-log suffices** — the append-only **local ledger *is* the durable buffer**
  (there is no shared/remote write to fail mid-flight, unlike a DB), so **no
  spool-and-replay layer is needed**. The only residual fault is the local ledger dir
  being unwritable (disk full / RO / perms), which is drop-and-log by necessity — a
  spool would live on the same disk.
- **Consequences:** serving and the user's open never depend on the ledger/model; worst
  case is a *lost label*, loudly surfaced. No deferred durability upgrade — the ledger
  is already the buffer. Establishes **INV-9** (telemetry isolation).

## ADR-014 — Ledger privacy: sensitive content stays private, never exported {#adr-014}

- **Status:** Accepted (D2).
- **Context:** The event ledger persists user behaviour. Sensitivity audit (spec 01):
  **raw `query` text + `visual_tokens`** (high — intimate free text); **`media_id` /
  `person_id`** (re-identifiable to private photos/people via MSA's DB); `user_id` +
  `open`/`dwell` (behavioural); **`features`** are derived numerics (low — by design,
  spec 02, no raw tags/place/GPS/captions). The store is already local/private (INV-5)
  and not a devdash input; what's unresolved is the raw-query handling and what may ever
  *leave* the private store.
- **Decision:** **Keep the raw query verbatim in the private ledger** — it's local-only,
  MSA already logs it, and raw events enable re-derivation ([ADR-012](#adr-012)). But
  establish **INV-10**: **real/production** query text and media/person references exist
  **only** in the private ledger/SoR (outside the repo); **no generated/exported
  artifact, `internal/` report, or devdash input contains them.** Any
  observability/showcase report is **aggregate/redacted** (counts, scores,
  model-vs-baseline — never queries or media_ids). Hand-authored design docs may use
  **synthetic placeholders** (`<query_text>`, `<person_id>`) — exempt, so the redaction
  test targets real-data outputs, not illustrations.
  Multi-user co-mingling is a known **household-internal** property ([ADR-008](#adr-008)):
  SoR access = access to every household member's behaviour. No secrets in the ledger.
  **Collection opt-out:** an MSA config switch **`ranker.event_logging` (default `on`** —
  labels are needed to learn) disables **all** ledger appends (`search`/`shown`/`open`)
  when off; serving is unaffected and `/track/open` still returns 204. It is the master
  privacy off-switch and is **orthogonal to the serving flag**
  (`ranker.enable_learning_to_rank`) — you may log without serving (collect before a
  model exists) or serve without logging (use a model, collect nothing new).
- **Consequences:** maximal local utility + re-derivability with **zero external
  leakage**; the observability report (architecture §4) must redact; a
  redaction/publishability check enforces INV-10. Keeping `features` derived-numeric
  (spec 02) is now load-bearing — adding raw tags/place/captions/GPS would re-open this.
  `ranker.event_logging` gives the user a one-switch privacy opt-out without disabling
  the ranker's serving.

| ADR | Title | Status |
|---|---|---|
| [001](#adr-001) | In-process reranking behind a flag | Accepted |
| [002](#adr-002) | Model progression (trivial → LambdaMART → neural) | Accepted |
| [003](#adr-003) | Offline CPU training (anywhere), in-process serving | Accepted |
| [004](#adr-004) | NDCG@k primary / MRR; baseline-first | Accepted |
| [005](#adr-005) | External SQLite system-of-record | Accepted |
| [006](#adr-006) | Dataset manifests, not DVC | Accepted |
| [007](#adr-007) | Position-bias-aware implicit labels | Accepted |
| [008](#adr-008) | Multi-user: pooled v1, user-aware capture | Accepted |
| [009](#adr-009) | Open-signal capture (search_id + /track/open) | Accepted |
| [010](#adr-010) | msa_ranker library + thin in-process shim | Accepted |
| [011](#adr-011) | Serving cold-start gate + manual retrain (v1) | Accepted |
| [012](#adr-012) | Decouple: JSONL ledger + training SoR + model handoff | Accepted |
| [013](#adr-013) | Ledger/telemetry best-effort / fail-open | Accepted |
| [014](#adr-014) | Ledger privacy: sensitive content private, never exported | Accepted |
