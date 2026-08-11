-- 0001_initial.sql — msa-ranker training SoR baseline schema (architecture §10).
-- Additive-only (INV-7): never edit a shipped migration; add 000N_*.sql for changes.
-- `_migrations` is created/managed by the runner (db.open_db); not declared here.
-- `ev_id` (the per-event ULID) is the SoR idempotency anchor; ingest upserts
-- ON CONFLICT DO NOTHING (spec 07).

CREATE TABLE IF NOT EXISTS ingest_state (
  ledger_file TEXT PRIMARY KEY,
  byte_offset INTEGER NOT NULL,
  updated_ts  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS search (
  ev_id          TEXT NOT NULL UNIQUE,
  search_id      TEXT PRIMARY KEY,
  user_id        TEXT NOT NULL DEFAULT 'default',
  query_text     TEXT NOT NULL,
  query_ctx_json TEXT,
  flag_on        INTEGER NOT NULL,
  model_version  TEXT,
  k              INTEGER NOT NULL,
  created_ts     REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS result_shown (
  ev_id           TEXT NOT NULL UNIQUE,
  search_id       TEXT NOT NULL,
  media_id        TEXT NOT NULL,
  position        INTEGER NOT NULL,
  score           REAL NOT NULL,
  heuristic_score REAL NOT NULL,   -- always logged (NN1) → baseline replay (spec 04)
  features_json   TEXT NOT NULL,
  PRIMARY KEY (search_id, media_id)
);
CREATE INDEX IF NOT EXISTS idx_shown_search_pos ON result_shown (search_id, position);

CREATE TABLE IF NOT EXISTS interaction (
  ev_id      TEXT PRIMARY KEY,     -- ULID; avoids the (…,created_ts) collision (T7)
  search_id  TEXT NOT NULL,
  media_id   TEXT NOT NULL,
  user_id    TEXT NOT NULL DEFAULT 'default',
  action     TEXT NOT NULL DEFAULT 'open',
  dwell_ms   INTEGER,
  created_ts REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_inter_search ON interaction (search_id, media_id);

CREATE TABLE IF NOT EXISTS dataset (
  dataset_id    TEXT PRIMARY KEY,
  created_ts    REAL NOT NULL,
  manifest_json TEXT NOT NULL,     -- watermark + ev_id set + seed + eval_frac (T1/NN2)
  note          TEXT
);

CREATE TABLE IF NOT EXISTS model (
  model_id            TEXT PRIMARY KEY,
  created_ts          REAL NOT NULL,
  artifact_path       TEXT NOT NULL,
  artifact_sha        TEXT NOT NULL,
  algo                TEXT NOT NULL,
  params_json         TEXT,
  feature_set_version TEXT,
  beats_baseline      INTEGER,     -- every model registered, pass or fail (NN3)
  dataset_id          TEXT NOT NULL REFERENCES dataset (dataset_id)
);

CREATE TABLE IF NOT EXISTS eval (
  eval_id     TEXT PRIMARY KEY,
  model_id    TEXT REFERENCES model (model_id),   -- NULL for the heuristic baseline row
  dataset_id  TEXT NOT NULL REFERENCES dataset (dataset_id),
  metric      TEXT NOT NULL,
  k           INTEGER,
  value       REAL NOT NULL,
  is_baseline INTEGER NOT NULL DEFAULT 0,
  created_ts  REAL NOT NULL
);
