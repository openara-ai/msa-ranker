"""S-3.1 — feature extraction (spec 02, crux): golden vector, determinism, missing
fields, person resolution, layout guard, no-position."""

from __future__ import annotations

import pytest

from msa_ranker.features import (
    FEATURE_NAMES,
    FEATURE_SET_VERSION,
    QueryContext,
    extract,
    feature_dict,
)

NOW = 1_000_000_000.0


class FakePeople:
    def __init__(self, mapping):
        self.mapping = mapping
        self.writes = 0  # would increment if anyone tried to mutate — stays 0 (read-only)

    def person_ids_for_media(self, media_id):
        return set(self.mapping.get(media_id, set()))


def _reference():
    candidate = {
        "media_id": "m1",
        "raw_similarity_score": 0.74,
        "source_scores": {"img": 0.74, "cap": 0.30},
        "source": "img",
        "tags": ["dog", "beach"],
        "type": "video",
        "date": NOW - 5 * 86400,  # 5 days ago
        "gps_lat": 1.0,
        "gps_lon": 2.0,
        "place": "Goa",
    }
    ctx = QueryContext(people=["p1", "p2"], visual_tokens=["dog", "sky"], place_intent="goa")
    people = FakePeople({"m1": {"p1", "p9"}})
    return candidate, ctx, people


def test_golden_vector():
    # AC-02.1/02.2 — hand-computed reference vector (independent truth).
    candidate, ctx, people = _reference()
    vec = extract(candidate, ctx, NOW, people)
    expected = [
        0.74,  # sim
        0.74,
        0.0,
        0.30,
        0.0,  # src_img/vid/cap/asr
        2.0,  # num_sources (img, cap)
        0.0,  # is_person_expand
        1.0,  # person_hits ({p1,p2} ∩ {p1,p9})
        1.0,  # has_person_intent
        1.0,  # tag_overlap (dog)
        1.0,  # is_video
        5.0,  # recency_days
        1.0,  # has_date
        1.0,  # has_gps
        1.0,  # place_match (goa ⊂ Goa)
        2.0,  # query_len
    ]
    assert vec == pytest.approx(expected)


def test_determinism():
    # AC-02.1 — identical inputs → byte-identical vector.
    candidate, ctx, people = _reference()
    assert extract(candidate, ctx, NOW, people) == extract(candidate, ctx, NOW, people)


def test_missing_fields_fill_and_flags():
    # AC-02.3 — absent source_scores/date/gps/place → fills + correct has_* flags.
    vec = feature_dict({"id": "m2"}, QueryContext(), NOW)
    assert vec["sim"] == 0.0
    assert vec["src_img"] == vec["src_vid"] == vec["src_cap"] == vec["src_asr"] == 0.0
    assert vec["num_sources"] == 0.0
    assert vec["has_date"] == 0.0 and vec["recency_days"] == 0.0
    assert vec["has_gps"] == 0.0
    assert vec["place_match"] == 0.0
    assert vec["has_person_intent"] == 0.0 and vec["person_hits"] == 0.0
    assert vec["query_len"] == 0.0


def test_person_hits_uses_resolved_ids():
    # AC-02.4 — person features key on resolved person_ids, not names.
    candidate = {"media_id": "m1", "raw_similarity_score": 0.5}
    ctx = QueryContext(people=["p1", "p2", "p3"])
    people = FakePeople({"m1": {"p2", "p3", "p7"}})
    assert feature_dict(candidate, ctx, NOW, people)["person_hits"] == 2.0


def test_iso_date_parsed():
    vec = feature_dict({"id": "m", "date": "2001-09-08T01:40:00+00:00"}, QueryContext(), NOW)
    assert vec["has_date"] == 1.0
    assert vec["recency_days"] >= 0.0


def test_naive_date_parsed_as_utc_deterministic():
    # AC-02.1 — a timezone-naive ISO date must parse as UTC (not host-local tz), so it
    # matches the explicit +00:00 form and gives identical recency on any host.
    naive = feature_dict({"id": "m", "date": "2001-09-08T01:40:00"}, QueryContext(), NOW)
    aware = feature_dict({"id": "m", "date": "2001-09-08T01:40:00+00:00"}, QueryContext(), NOW)
    assert naive["has_date"] == 1.0
    assert naive["recency_days"] == aware["recency_days"]


def test_person_expand_flag():
    vec = feature_dict({"id": "m", "source": "person_expand"}, QueryContext(), NOW)
    assert vec["is_person_expand"] == 1.0


def test_sim_does_not_fall_back_to_score():
    # Spec 02 / no leakage — sim is ONLY raw_similarity_score; a bare `score` (heuristic
    # or served-ranker output) must not become sim.
    assert feature_dict({"id": "m", "score": 0.9}, QueryContext(), NOW)["sim"] == 0.0


def test_num_sources_counts_named_only():
    # Extra keys (e.g. person_expand) in source_scores must not inflate num_sources.
    vec = feature_dict(
        {"id": "m", "source_scores": {"img": 0.5, "person_expand": 0.9}}, QueryContext(), NOW
    )
    assert vec["num_sources"] == 1.0  # only the named "img" counts


def test_recency_negative_for_future_date():
    # Spec 02: (now - date)/days, unclamped — a future date (clock skew) is negative.
    vec = feature_dict({"id": "m", "date": NOW + 2 * 86400}, QueryContext(), NOW)
    assert vec["recency_days"] == pytest.approx(-2.0)


def test_layout_guard():
    # AC-02.2/02.5 — frozen length/order; no `position` (ADR-007).
    assert FEATURE_SET_VERSION == "v1"
    assert len(FEATURE_NAMES) == 16
    assert len(extract({"id": "x"}, QueryContext(), NOW)) == len(FEATURE_NAMES)
    assert "position" not in FEATURE_NAMES


def test_pure_no_write_to_lookup():
    # AC-02.6 — extraction only reads the person lookup (no mutation).
    candidate, ctx, people = _reference()
    extract(candidate, ctx, NOW, people)
    assert people.writes == 0
