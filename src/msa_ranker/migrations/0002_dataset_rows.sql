-- 0002_dataset_rows.sql — materialize a frozen dataset (spec 05, FR-6, INV-4).
-- Additive-only (INV-7). A `dataset` is made replayable by MATERIALIZING the rows it
-- composes, so a later `open`/`shown` ingestion can never shift its labels/splits.
--   dataset_row  — the frozen TRAIN-split label rows (post Click>Skip-Above drop).
--   dataset_eval — the frozen FULL shown candidate set per EVAL-split search, with the
--                  heuristic_score (baseline ranking) + opened flag (relevance), so eval
--                  ranks the whole set (spec 04) independent of the training-label drop.

CREATE TABLE IF NOT EXISTS dataset_row (
  dataset_id    TEXT NOT NULL REFERENCES dataset (dataset_id),
  search_id     TEXT NOT NULL,
  media_id      TEXT NOT NULL,
  user_id       TEXT NOT NULL DEFAULT 'default',
  features_json TEXT NOT NULL,   -- the frozen feature vector (FEATURE_NAMES order)
  label         INTEGER NOT NULL,
  grp           TEXT NOT NULL,   -- the leak-free split key (spec 03)
  PRIMARY KEY (dataset_id, search_id, media_id)
);

CREATE TABLE IF NOT EXISTS dataset_eval (
  dataset_id      TEXT NOT NULL REFERENCES dataset (dataset_id),
  search_id       TEXT NOT NULL,
  media_id        TEXT NOT NULL,
  heuristic_score REAL NOT NULL,
  features_json   TEXT NOT NULL,
  is_positive     INTEGER NOT NULL,   -- opened at freeze time (relevance = 1)
  PRIMARY KEY (dataset_id, search_id, media_id)
);
-- No secondary index: the PK (dataset_id, search_id, media_id) B-tree already serves the
-- `WHERE dataset_id = ?` / `... AND search_id = ?` reads via its leading prefix.
