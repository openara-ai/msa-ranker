"""Configuration constants/defaults for msa-ranker.

The training SoR lives outside the repo (ADR-005). The event ledger is MSA-owned and
lives in MSA's data area (spec 01); its path is supplied by the caller / MSA config.
"""

from __future__ import annotations

from pathlib import Path

# Training-owned SoR (outside the repo, gitignored — ADR-005/INV-5).
DEFAULT_DB_PATH = Path.home() / ".msa-ranker" / "msa-ranker.sqlite"

# Ledger rotation size cap (spec 01).
DEFAULT_ROTATE_BYTES = 64 * 1024 * 1024  # 64 MB

# Training defaults (spec 05). The min-data gate refuses the first train below a floor
# of labelled searches (ADR-011) so no garbage model is written.
DEFAULT_K = 10
DEFAULT_EVAL_FRAC = 0.3
DEFAULT_MIN_LABELED_SEARCHES = 10
