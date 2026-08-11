from __future__ import annotations

import pytest

from msa_ranker.db import open_db


@pytest.fixture
def db(tmp_path):
    conn = open_db(tmp_path / "sor.sqlite")
    yield conn
    conn.close()


@pytest.fixture
def ledger_dir(tmp_path):
    d = tmp_path / "ledger"
    d.mkdir()
    return d
