# CLAUDE.md

> Read this at the start of every session. It captures the conventions,
> guardrails, and workflow; the authoritative design docs live in `docs/`. When
> this file and a design doc disagree, the design doc wins — flag the conflict.
> Project-specific state (current milestone/sprint state, private planning docs)
> lives in a private companion file — see the bottom of this doc.

## Base conventions

1. **Session start.** Read `CLAUDE.md` fully, then the design docs it lists, in
   order. This file captures conventions; the design docs are authoritative.
   Don't re-derive decisions already recorded in the ADRs — follow them.
2. **Respect the decision ledger.** Invariants + ADRs are **non-negotiable**. If a
   task seems to require violating one, **stop and ask** — that's a design change,
   not an implementation detail.
3. **The dev loop (every change).** Smallest coherent slice → **build** → **lint**
   (warnings treated as errors) → **format** → write **tests alongside** the code,
   anchored to **independent truth** (golden answers, not the code's own output) →
   green → propose a **focused diff with rationale**. **Never propose red code**
   (non-compiling or failing). **Correctness-critical paths** (money, crypto, auth,
   dispatch/isolation) get human review — don't self-merge them. *(The project
   layer supplies the exact build/lint/format/test commands.)*
4. **Git workflow.**
   - `main` is the default branch; confirm the current branch at session start;
     for substantial work, **branch first** (`feature/<desc>` or `fix/<desc>`).
   - Commits small + focused, each **building and passing tests** (keeps
     `git bisect` useful). **Stage named files only — never `git add -A` / `git
     add .`** (so stray or generated files are never committed by accident). Run
     the dev-loop gates before committing.
   - Commit message: `<scope>: <short summary>` + a body (what changed and why;
     reference the milestone + ADR) + a trailer
     `Co-Authored-By: <agent model> <noreply@anthropic.com>`.
   - **Push or open a PR only when the developer asks.** Use `gh` for GitHub.
   - **Never** force-push, `--no-verify`, `reset --hard` without instruction, or
     **merge to `main`** (always a developer action).
5. **Review mesh / PR loop.** For hands-off review, use a **ready (non-draft) PR**
   and hold merge with branch protection or a "do not merge" label — *not* a draft
   (drafts suppress most auto-reviewers). Triage **every** review comment with a
   priority, share the triage, and **wait for confirmation before fixing**. Then:
   make the code fixes → get approval → commit → post **per-finding** replies.
   - **P0** — critical: live breakage, security-exploitable now, data loss in flight.
   - **P1** — broken / unsafe, regression of existing behaviour.
   - **P2** — actively surprises the user or breaks their stated intent.
   - **P3** — polish / defensive hardening / cosmetic.
   - Keep a reviewer's own priority label if it set one — assign only when missing.
   - **Reply format — the metrics layer parses the PR thread, not the chat.** Reply
     **inline on the finding, token-first**: `Pn — Fixed in <sha>. <one line>` — lead
     with a bare `P0`–`P3` token, then the lifecycle verb (`Fixed in <sha>` / `deferred`
     / `refuted`) + the fix SHA. One priority + one verb + one SHA per comment, bound to
     its `(file, line)`. A priority described only in words ("this was high-priority")
     is **not** parsed.
     - **One reply per finding — never a consolidated list** (a single "addressed all
       feedback" comment records only its first item and drops the rest). A roll-up may
       be added *in addition*, not *instead*.
     - **A no-fix reply still leads with the priority + verb:** `P3 — deferred; <why>.`
       or `P2 — refuted: <why-not-real>.`
     - **Label other reviewers' unlabelled findings too** — reply inline leading with
       the `Pn` you assigned.
6. **Milestone exit is not yours to declare.** When the exit criteria look met,
   **propose exit with evidence** (ticked criteria + proof: passing test names,
   smoke output), confirm the suite is green, list caveats — then **stop**. The
   developer runs the manual checks and signs off.
7. **Artifacts are the source of truth, not the chat — and they're living.** When
   build reveals a doc is wrong or incomplete, **change the doc, not just the
   code** (silent divergence is the failure mode). After a confirmed merge, update
   the project's current state and any affected design docs.
8. **Never-do list (consolidated).**
   - Never push or open a PR unless asked.
   - Never `git add -A` / `git add .` — stage named files only.
   - Never force-push, `--no-verify`, or `reset --hard` without instruction.
   - Never merge to `main` — a developer action.
   - Never propose red code (non-compiling or failing tests).
   - Never violate an invariant silently — stop and ask.
   - Never hand-roll crypto; never bake secrets into code / tests / fixtures /
     logs / git.
   - Never self-merge correctness-critical paths.

## Project layer

### Stack

A **non-LLM supervised learning-to-rank reranker** for
[media-search-agent](https://github.com/openara-ai/media-search-agent) (MSA). MSA is
Python 3.11+, FastAPI + Pydantic, SQLite + embedded Qdrant, CLIP/facenet/RT-DETR on
PyTorch. The ranker attaches **in-process** at MSA's existing rerank seam, behind a
`config.yaml` flag with the current heuristic as deterministic fallback
([ADR-001](docs/adrs.md#adr-001)). It reorders MSA's *existing* candidate set only —
never changes retrieval, never writes MSA's index DB ([INV-6](docs/invariants.md)).
Models progress **trivial pointwise baseline → LightGBM LambdaMART → neural
cross-encoder (deferred)** ([ADR-002](docs/adrs.md#adr-002)); trained offline + manual
on **CPU, any platform** (GPU only for the deferred neural tier), served as a small
portable artifact ([ADR-003](docs/adrs.md#adr-003)). Cross-platform package + CLI;
dev = editable install, release = a pinned `msa_ranker` wheel in MSA
([ADR-010](docs/adrs.md#adr-010)). Product data is **decoupled into two file
artifacts** ([ADR-012](docs/adrs.md#adr-012)): MSA appends an **append-only JSONL
event ledger** (its only output); training **ingests** it into its **own SQLite SoR**
(additive migrations, WAL, private/outside-repo) and ships back a **model +
manifest**. Single writer per store; the ledger append is best-effort / fail-open
([ADR-013](docs/adrs.md#adr-013)).

### Orientation (read order)

1. [`docs/research.md`](docs/research.md) — Stage 2 findings, the chosen approach.
2. [`docs/adrs.md`](docs/adrs.md) + [`docs/invariants.md`](docs/invariants.md) — the
   decision ledger (binding). **Follow these; don't re-derive.**
3. [`docs/requirements.md`](docs/requirements.md) — FR-1…18 / NFR-1…10, scope,
   non-goals, first-milestone acceptance.
4. [`docs/architecture.md`](docs/architecture.md) — topology, components, the two
   data flows, the seam, the concrete SoR schema.
5. [`docs/specs/`](docs/specs/) (01–07) + [`docs/testing.md`](docs/testing.md) —
   feature specs (per-feature behaviour + `AC-nn.x` acceptance) + test strategy.
6. [`docs/idea.md`](docs/idea.md) — the seed and the *why now*.

*(The roadmap and metrics ledgers are part of the private planning layer — see the
private companion file.)*

### Invariants (quick reference → docs/invariants.md)

- **INV-1** baseline + metric before any model · **INV-2** model out of the
  correctness path (deterministic plumbing golden-tested; quality is eval) ·
  **INV-3** flag off ≡ MSA today · **INV-4** owned/reproducible/leak-free data ·
  **INV-5** ledger + SoR private + outside the repo · **INV-6** ranker only
  reorders, never writes MSA's index · **INV-7** additive migrations only ·
  **INV-8** smallest complete loop before sophistication · **INV-9** ledger/telemetry
  failure never degrades MSA search or open · **INV-10** raw query/media/person refs
  stay in the private ledger, never exported.

### Commands

Python `src/` package; dev in a venv. Setup: `python -m venv .venv && . .venv/bin/activate
&& pip install -e ".[dev]"`.

- **Lint** (warnings = errors): `ruff check .`
- **Format**: `black .` (check: `black --check .`)
- **Test**: `pytest`
- **Run (CLI)**: `msa-ranker` with subcommands `ingest`, `train`, `report`, `deploy`
  (plus an `eval` stub whose help points into `train`).

**CI gate** (`.github/workflows/ci.yml`, fail-closed, mirrors the local loop):
`ruff check .` → `black --check .` → `pytest`. The agentic Claude review runs beside it
(`.github/workflows/claude-pr-review.yml`, default-branch-only).

### Running locally & secrets

MSA appends a JSONL **event ledger** in **MSA's own data area** (config-driven,
gitignored — MSA owns its output); the **training SoR** (`~/.msa-ranker/msa-ranker.sqlite`)
is built by ingesting it — both outside the repo, gitignored. Logging has a default-on
privacy off-switch **`ranker.event_logging`** (off ⇒ no ledger writes; ADR-014). No
secrets in git — ever ([INV-5](docs/invariants.md)). Training is
**manually triggered**; serving loads a model+manifest from a directory handoff
([ADR-012](docs/adrs.md#adr-012)).

### Project guardrails

- The ranker is on **MSA's serving path** — treat the flag/fallback and feature
  extraction as **correctness-critical** (golden-tested; human review). The *model*
  is not, but the plumbing around it is.
- **Never write to MSA's `index/media.sqlite`** — the ranker reads features read-only
  and writes only its own ledger (MSA side) / SoR (training side).
- The code-review workflow is **default-branch-only** — never iterate on
  `.github/workflows/*review*.yml` from a feature branch.
- **Public/private split.** A curated subset of this repo mirrors to a public
  repository; `docs/`, `src/`, `tests/`, and the top-level public files ship, and
  everything else (including `internal/`) is private by default. Public docs must
  never link to private paths — the doc-separation check enforces this. New docs
  default to `internal/docs/`; put a doc in `docs/` only when it is deliberately
  public.

## For maintainers of this repo

If a file exists at `internal/docs/CLAUDE-private.md`, **read it next before
responding**. It holds everything repo-private: the provenance of the base
conventions, the extended orientation (roadmap, metrics ledgers, exit proposals),
the publish/mirror layer, and the project's current state. This public `CLAUDE.md`
carries only what is safe and useful on the public mirror.

If you're using this `CLAUDE.md` as a starter for your own project, delete the
section above and customize from here.
