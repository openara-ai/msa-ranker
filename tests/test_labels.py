"""S-4.1 — label construction (spec 03, the ADR-007 crux): Click > Skip-Above golden
sets, determinism, leak-free grouped split (INV-4), orphan handling, group canonical.

Golden answers are the spec's hand-computed labels (independent truth), not the code's
own output.
"""

from __future__ import annotations

import json
import logging

import pytest

from msa_ranker.features import FEATURE_NAMES, QueryContext
from msa_ranker.labels import (
    LabelRow,
    SearchBundle,
    ShownRow,
    build_labels,
    group_key,
    iter_bundles,
    split,
)


def _bundle(search_id, opened_positions, *, n=5, user_id="u1", ctx=None):
    """A search with n shown rows at positions 0..n-1; `opened_positions` are opened."""
    ctx = ctx if ctx is not None else QueryContext(visual_tokens=["dog"])
    shown = [ShownRow(f"{search_id}-m{p}", p, {}) for p in range(n)]
    opened = {f"{search_id}-m{p}" for p in opened_positions}
    return SearchBundle(search_id, user_id, ctx, shown, opened)


def _labels_by_pos(rows, search_id):
    """{position: label} for a search_id (position parsed from the m<pos> media id)."""
    return {int(r.media_id.rsplit("m", 1)[1]): r.label for r in rows if r.search_id == search_id}


def test_skip_above_golden():
    # AC-03.1 — open@2 of 5 → positives {2}, negatives {0,1}, dropped {3,4}.
    rows = build_labels([_bundle("s1", {2}, n=5)])
    assert _labels_by_pos(rows, "s1") == {0: 0, 1: 0, 2: 1}


def test_multiple_opens_deepest_golden():
    # AC-03.2 — opens @{1,3} → deepest=3; non-opens above (0,2) negative; below (4) dropped.
    rows = build_labels([_bundle("s2", {1, 3}, n=5)])
    assert _labels_by_pos(rows, "s2") == {0: 0, 1: 1, 2: 0, 3: 1}


def test_no_open_search_contributes_nothing():
    # AC-03.3 — a search with no opens → zero rows.
    assert build_labels([_bundle("s3", set(), n=5)]) == []


def test_below_deepest_is_dropped_not_zero():
    # AC-03.4 — below-deepest-open non-opens are dropped, never labeled 0.
    rows = build_labels([_bundle("s4", {1}, n=5)])
    positions = _labels_by_pos(rows, "s4")
    assert set(positions) == {0, 1}  # 2,3,4 dropped entirely
    assert 2 not in positions and 0 in positions  # nothing below deepest leaked in as 0


def test_determinism():
    # AC-03.5 — same input → identical rows (incl. order).
    bundles = [_bundle("s5", {2}), _bundle("s6", {0, 1})]
    assert build_labels(bundles) == build_labels(bundles)


def test_feature_vector_is_frozen_order():
    rows = build_labels([_bundle("s7", {2})])
    assert all(len(r.features) == len(FEATURE_NAMES) for r in rows)


def test_orphan_open_dropped_and_logged(caplog):
    # AC-03.7 — an open with no shown row is dropped + logged, not a crash.
    shown = [ShownRow("m0", 0, {}), ShownRow("m1", 1, {})]
    bundle = SearchBundle("s8", "u1", QueryContext(), shown, {"m1", "ghost"})
    with caplog.at_level(logging.WARNING):
        rows = build_labels([bundle])
    assert {(r.media_id, r.label) for r in rows} == {("m0", 0), ("m1", 1)}
    assert "orphan open" in caplog.text and "ghost" in caplog.text


def test_all_opens_orphaned_contributes_nothing():
    shown = [ShownRow("m0", 0, {})]
    bundle = SearchBundle("s9", "u1", QueryContext(), shown, {"ghost"})
    assert build_labels([bundle]) == []


def test_group_canonical_same_intent_same_group():
    # AC-03.8 — same user + resolved people + normalized tokens → same group, regardless
    # of order/case; a different user → a different group.
    a = QueryContext(people=["p2", "p1"], visual_tokens=["Dog", "sky"])
    b = QueryContext(people=["p1", "p2"], visual_tokens=["sky", "dog"])
    assert group_key("u1", a) == group_key("u1", b)
    assert group_key("u1", a) != group_key("u2", a)  # user-scoped


def test_grouped_split_is_leak_free():
    # AC-03.6 — no `group` spans train and eval (INV-4); same seed → same partition.
    same_intent = QueryContext(people=["p1"], visual_tokens=["dog"])
    bundles = [
        _bundle("a1", {1}, ctx=same_intent),  # these two share a group...
        _bundle("a2", {0, 1}, ctx=same_intent),  # ...so must land on the same side
        _bundle("b1", {1}, ctx=QueryContext(visual_tokens=["cat"])),
        _bundle("c1", {1}, ctx=QueryContext(visual_tokens=["sky"])),
        _bundle("d1", {1}, ctx=QueryContext(visual_tokens=["sea"])),
    ]
    rows = build_labels(bundles)
    train, held = split(rows, seed=7, eval_frac=0.5)
    train_groups = {r.group for r in train}
    held_groups = {r.group for r in held}
    assert train_groups.isdisjoint(held_groups)  # INV-4: no group straddles
    assert train_groups | held_groups == {r.group for r in rows}  # nothing lost
    assert split(rows, seed=7, eval_frac=0.5) == (train, held)  # deterministic


def test_labelrow_is_immutable():
    # Frozen (reassignment raises) AND features is a tuple (no in-place mutation).
    r = LabelRow("s", "u", "m", (0.0,), 1, "g")
    assert isinstance(r.features, tuple)
    try:
        r.label = 0  # type: ignore[misc]
    except Exception as exc:  # FrozenInstanceError
        assert "cannot assign" in str(exc) or "FrozenInstance" in type(exc).__name__
    else:
        raise AssertionError("LabelRow should be frozen")


def test_split_rejects_unsplittable_inputs():
    # A leak-free non-empty split is impossible with < 2 groups or eval_frac outside (0,1).
    one_group = build_labels([_bundle("g1", {1}), _bundle("g2", {1})])  # same default ctx
    assert len({r.group for r in one_group}) == 1
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            split(one_group, seed=1, eval_frac=bad)
    with pytest.raises(ValueError, match="2 query groups"):
        split(one_group, seed=1, eval_frac=0.5)


def test_split_never_empties_a_side():
    # Codex P1: a tiny eval_frac (or banker's-rounding) must not yield an empty eval set.
    rows = build_labels(
        [_bundle(f"s{i}", {1}, ctx=QueryContext(visual_tokens=[f"tok{i}"])) for i in range(3)]
    )
    assert len({r.group for r in rows}) == 3
    for frac in (0.01, 0.5, 0.99):
        train, held = split(rows, seed=3, eval_frac=frac)
        assert {r.group for r in train} and {r.group for r in held}  # both non-empty


def test_iter_bundles_reads_sor(db):
    # The SoR loader feeds build_labels in the spec-05 pipeline. Insert a search with
    # 3 shown + 1 open, then label via iter_bundles → open@1 → pos{1}=1, {0}=0, {2} drop.
    db.execute(
        "INSERT INTO search (ev_id, search_id, user_id, query_text, query_ctx_json, "
        "flag_on, model_version, k, created_ts) VALUES (?,?,?,?,?,?,?,?,?)",
        ("e1", "q1", "u1", "dog", json.dumps({"visual_tokens": ["dog"]}), 0, None, 3, 1.0),
    )
    for pos, mid in enumerate(("m0", "m1", "m2")):
        db.execute(
            "INSERT INTO result_shown (ev_id, search_id, media_id, position, score, "
            "heuristic_score, features_json) VALUES (?,?,?,?,?,?,?)",
            (f"s-{mid}", "q1", mid, pos, 0.5, 0.5, json.dumps({"sim": 0.9})),
        )
    db.execute(
        "INSERT INTO interaction (ev_id, search_id, media_id, user_id, action, dwell_ms, "
        "created_ts) VALUES (?,?,?,?,?,?,?)",
        ("o1", "q1", "m1", "u1", "open", None, 2.0),
    )
    db.commit()

    rows = build_labels(iter_bundles(db))
    assert {(r.media_id, r.label) for r in rows} == {("m0", 0), ("m1", 1)}
    sim_idx = FEATURE_NAMES.index("sim")
    assert next(r for r in rows if r.media_id == "m1").features[sim_idx] == 0.9
    # Canonical JSON signature from ctx: [user, people, norm_tokens, date].
    assert all(r.group == group_key("u1", QueryContext(visual_tokens=["dog"])) for r in rows)
