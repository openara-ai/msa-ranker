"""Deterministic feature extraction (spec 02, FR-4, INV-2).

Pure: `(candidate, query_context, now) → fixed-length float vector`, used identically
at serve time (spec 06) and to populate `shown.features` (spec 01). Same inputs → same
vector. The only outside read is a **read-only** person lookup (INV-6). `position` is
NOT a feature (ADR-007). Person features key on resolved `person_id`s (ADR-008).

Bump `FEATURE_SET_VERSION` whenever the list/order changes — the serving gate refuses a
model whose manifest version != the running extractor (spec 06).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

FEATURE_SET_VERSION = "v1"

# Frozen order — index positions are the contract (spec 02). Immutable tuple so an
# importer can't silently corrupt the layout (rename/append). Do not reorder.
FEATURE_NAMES: tuple[str, ...] = (
    "sim",
    "src_img",
    "src_vid",
    "src_cap",
    "src_asr",
    "num_sources",
    "is_person_expand",
    "person_hits",
    "has_person_intent",
    "tag_overlap",
    "is_video",
    "recency_days",
    "has_date",
    "has_gps",
    "place_match",
    "query_len",
)

_SOURCES = ("img", "vid", "cap", "asr")


@dataclass
class QueryContext:
    """Decomposed query (resolved entities — ADR-008), as logged in `search.ctx`."""

    people: list[str] = field(default_factory=list)  # resolved person_ids
    visual_tokens: list[str] = field(default_factory=list)
    date_intent: Any = None
    place_intent: str | None = None


class PersonLookup(Protocol):
    """Read-only resolution of a media id → its person_ids (INV-6)."""

    def person_ids_for_media(self, media_id: str) -> set[str]: ...


class _NoPeople:
    def person_ids_for_media(self, media_id: str) -> set[str]:
        return set()


def _ts(value: Any) -> float | None:
    """Coerce a date value (epoch number or ISO-8601 string) to a UNIX timestamp."""
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str) and value:
        try:
            dt = datetime.fromisoformat(value)
        except ValueError:
            return None
        # A naive datetime would otherwise be read in the host's local tz, so the same
        # input yields different recency on train vs serve hosts — pin naive to UTC (AC-02.1).
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.timestamp()
    return None


def extract(
    candidate: Mapping[str, Any],
    ctx: QueryContext,
    now: float,
    people: PersonLookup | None = None,
) -> list[float]:
    """Return the v1 feature vector (len == len(FEATURE_NAMES)) for one candidate."""
    people = people or _NoPeople()
    src_scores: Mapping[str, Any] = candidate.get("source_scores") or {}
    media_id = candidate.get("media_id") or candidate.get("id") or ""

    # sim is defined ONLY from raw_similarity_score (spec 02). No fallback to `score` —
    # that is a heuristic/served-ranker output and would leak ranking signal into training.
    sim = float(candidate.get("raw_similarity_score") or 0.0)
    per_source = [float(src_scores.get(s, 0.0) or 0.0) for s in _SOURCES]
    # Count only the named contributing sources, so num_sources stays aligned with
    # per_source (ignores extra keys like "person_expand").
    num_sources = float(sum(1 for s in _SOURCES if src_scores.get(s)))
    is_person_expand = float(
        candidate.get("source") == "person_expand" or "person_expand" in src_scores
    )

    query_people = set(ctx.people or [])
    result_people = people.person_ids_for_media(media_id) if media_id else set()
    person_hits = float(len(query_people & result_people))
    has_person_intent = float(bool(query_people))

    tokens = {t.lower() for t in (ctx.visual_tokens or [])}
    tags = {str(t).lower() for t in (candidate.get("tags") or [])}
    tag_overlap = float(len(tokens & tags))

    is_video = float(candidate.get("type") == "video")

    ts = _ts(candidate.get("date"))
    has_date = float(ts is not None)
    # Spec 02: (now - date) / days. Not clamped — a future date (clock skew) yields a
    # negative value, which is real signal and distinct from a now-dated item.
    recency_days = (now - ts) / 86400.0 if ts is not None else 0.0

    has_gps = float(candidate.get("gps_lat") is not None and candidate.get("gps_lon") is not None)

    place = candidate.get("place")
    place_match = float(
        bool(ctx.place_intent) and bool(place) and ctx.place_intent.lower() in str(place).lower()
    )

    query_len = float(len(ctx.visual_tokens or []))

    vector = [
        sim,
        *per_source,
        num_sources,
        is_person_expand,
        person_hits,
        has_person_intent,
        tag_overlap,
        is_video,
        recency_days,
        has_date,
        has_gps,
        place_match,
        query_len,
    ]
    if len(vector) != len(
        FEATURE_NAMES
    ):  # layout invariant (correctness-critical; not -O strippable)
        raise RuntimeError(f"feature vector length {len(vector)} != {len(FEATURE_NAMES)}")
    return vector


def feature_dict(
    candidate: Mapping[str, Any],
    ctx: QueryContext,
    now: float,
    people: PersonLookup | None = None,
) -> dict[str, float]:
    """`extract()` zipped with `FEATURE_NAMES` — the form logged in `shown.features`."""
    return dict(zip(FEATURE_NAMES, extract(candidate, ctx, now, people), strict=True))
