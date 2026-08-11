"""S-2 — eval harness (spec 04). Golden anchored to research §2; baseline from SoR."""

from __future__ import annotations

import time

import pytest

from msa_ranker.eval import Query, evaluate, measure_baseline, mrr, ndcg_at_k

# research §2 worked example: 5 candidates, opened R1 + R3.
POSITIVES = {"R1", "R3"}
ORDER_A = ["R1", "R2", "R3", "R4", "R5"]  # opens up top
ORDER_B = ["R2", "R4", "R5", "R1", "R3"]  # opens buried


def test_ndcg_mrr_golden_research_example():
    # AC-04.1 — the headline golden, hand-computed in research §2.
    assert ndcg_at_k(ORDER_A, POSITIVES, 5) == pytest.approx(0.92, abs=0.01)
    assert mrr(ORDER_A, POSITIVES, 5) == pytest.approx(1.00)
    assert ndcg_at_k(ORDER_B, POSITIVES, 5) == pytest.approx(0.50, abs=0.01)
    assert mrr(ORDER_B, POSITIVES, 5) == pytest.approx(0.25)


def test_k_bounds():
    # AC-04.2 — k=1 sees only the top; k >= N is the full list.
    assert mrr(ORDER_A, POSITIVES, 1) == pytest.approx(1.0)
    assert mrr(ORDER_B, POSITIVES, 1) == 0.0  # first positive at rank 4, outside k=1
    assert ndcg_at_k(ORDER_A, POSITIVES, 1) == pytest.approx(1.0)  # R1 alone is ideal@1
    assert ndcg_at_k(ORDER_B, POSITIVES, 99) == pytest.approx(ndcg_at_k(ORDER_B, POSITIVES, 5))


def test_no_positive_query():
    # AC-04.3 — a query with no relevant item scores 0/0.
    assert ndcg_at_k(["a", "b", "c"], set(), 3) == 0.0
    assert mrr(["a", "b", "c"], set(), 3) == 0.0


def test_macro_average_is_mean_of_per_query():
    # AC-04.6
    qs = [Query("s1", ORDER_A, POSITIVES), Query("s2", ORDER_B, POSITIVES)]
    res = evaluate(qs, 5)
    assert res.n_queries == 2
    expected_ndcg = (ndcg_at_k(ORDER_A, POSITIVES, 5) + ndcg_at_k(ORDER_B, POSITIVES, 5)) / 2
    assert res.ndcg == pytest.approx(expected_ndcg)
    assert res.mrr == pytest.approx((1.0 + 0.25) / 2)


def test_evaluate_empty():
    res = evaluate([], 5)
    assert res.n_queries == 0
    assert res.ndcg == 0.0
    assert res.mrr == 0.0
    assert res.k == 5


# ---- baseline from the SoR (AC-04.4 tie-break + measure_baseline wiring) ----


def _seed_search(conn, search_id, shown, opens):
    """shown: list of (media_id, heuristic_score); opens: set of media_id."""
    conn.execute(
        "INSERT INTO search "
        "(ev_id, search_id, user_id, query_text, flag_on, k, created_ts) "
        "VALUES (?, ?, 'default', 'q', 0, ?, ?)",
        (f"s-{search_id}", search_id, len(shown), time.time()),
    )
    for pos, (media_id, hs) in enumerate(shown):
        conn.execute(
            "INSERT INTO result_shown "
            "(ev_id, search_id, media_id, position, score, heuristic_score, features_json) "
            "VALUES (?, ?, ?, ?, ?, ?, '{}')",
            (f"sh-{search_id}-{media_id}", search_id, media_id, pos, hs, hs),
        )
    for media_id in opens:
        conn.execute(
            "INSERT INTO interaction "
            "(ev_id, search_id, media_id, user_id, action, created_ts) "
            "VALUES (?, ?, ?, 'default', 'open', ?)",
            (f"op-{search_id}-{media_id}", search_id, media_id, time.time()),
        )
    conn.commit()


def test_measure_baseline_from_sor(db):
    # heuristic order = R1..R5 by score; opens R1,R3 → matches research §2 Ranker A.
    shown = [("R1", 0.9), ("R2", 0.8), ("R3", 0.7), ("R4", 0.6), ("R5", 0.5)]
    _seed_search(db, "q1", shown, {"R1", "R3"})
    res = measure_baseline(db, k=5)
    assert res.n_queries == 1
    assert res.ndcg == pytest.approx(0.92, abs=0.01)
    assert res.mrr == pytest.approx(1.0)


def test_baseline_tie_break_deterministic(db):
    # AC-04.4 — equal heuristic_scores break by media_id asc, deterministically.
    shown = [("b", 0.5), ("a", 0.5), ("c", 0.5)]  # all tied
    _seed_search(db, "q2", shown, {"c"})
    res = measure_baseline(db, k=3)
    # order is a,b,c → c at rank 3 → MRR 1/3
    assert res.mrr == pytest.approx(1 / 3)
