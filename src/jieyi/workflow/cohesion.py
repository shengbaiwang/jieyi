"""Group cohesion pass: repair cross-segment wording after drafts commit.

Drafts translate each segment in isolation. A second pass revisits committed
draft groups (5-10 segments by default) with sources and drafts visible
together, then revises only cross-segment cohesion: pronouns, connectives,
tense/aspect, repeated wording. Single-segment meaning, structure, and
placeholders are never renegotiated here.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from jieyi.domain.models import Segment, SegmentKind

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class CohesionGroup:
    segments: tuple[Segment, ...]

    @property
    def ordinals(self) -> tuple[int, ...]:
        return tuple(segment.ordinal for segment in self.segments)

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(segment.id for segment in self.segments)


def plan_cohesion_groups(
    segments: list[Segment],
    *,
    min_segments: int = 2,
    max_segments: int = 8,
    max_group_chars: int = 6_000,
) -> list[CohesionGroup]:
    """Split freshly translated segments into cohesion groups.

    Boundaries: heading segments start a new group, heading_path changes cut,
    and oversized/long groups cut. Order is document order; callers filter to
    segments that actually need the pass beforehand.
    """
    ordered = sorted(segments, key=lambda item: item.ordinal)
    groups: list[CohesionGroup] = []
    current: list[Segment] = []
    current_chars = 0

    def flush() -> None:
        nonlocal current, current_chars
        if len(current) >= max(2, min_segments):
            groups.append(CohesionGroup(segments=tuple(current)))
        current = []
        current_chars = 0

    previous: Segment | None = None
    for segment in ordered:
        size = len(segment.source_text or "")
        hard_cut = (
            bool(current)
            and (
                segment.kind is SegmentKind.HEADING
                or (previous is not None and segment.heading_path != previous.heading_path)
                or len(current) >= max(2, max_segments)
                or current_chars + size > max(500, max_group_chars)
            )
        )
        if hard_cut:
            flush()
        current.append(segment)
        current_chars += size
        previous = segment
    flush()
    return groups


def build_cohesion_messages(
    *,
    context: str,
    sources: dict[str, str],
    drafts: dict[str, str],
    ordered_ids: list[str],
) -> list[dict[str, str]]:
    """Build a strict JSON revision prompt for one cohesion group."""
    system = (
        "Revise committed draft translations for cross-segment cohesion only. "
        "Read every source together with its draft, then return one value per "
        "segment ID.\n"
        "You may only adjust pronouns and zero-anaphora, connectives between "
        "segments, tense and aspect consistency, repeated-word choice, and "
        "politeness or register consistency across the group.\n"
        "Do not change single-segment meaning, add or omit claims, merge or "
        "split segments, reorder content, or touch placeholder tokens shaped "
        "like [[JY_PH_0000]]. Every placeholder must appear exactly once, "
        "unchanged, in each revised value.\n"
        "Return one valid JSON object only. Its keys must be exactly the "
        "supplied segment IDs and each value must be only that segment's "
        "revised translation. Do not add markdown, commentary, or extra keys."
    )
    if context.strip():
        system += f"\n\n{context.strip()}"
    items = [
        {
            "id": segment_id,
            "source": sources[segment_id],
            "draft": drafts[segment_id],
        }
        for segment_id in ordered_ids
    ]
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": json.dumps({"segments": items}, ensure_ascii=False, separators=(",", ":")),
        },
    ]


def parse_cohesion_revisions(text: str, expected_ids: set[str]) -> dict[str, str]:
    """Parse and strictly validate a cohesion response keyed by segment ID."""
    cleaned = _FENCE.sub("", (text or "").strip())
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError("cohesion response is not valid JSON") from exc
    if isinstance(value, dict) and isinstance(value.get("translations"), dict):
        value = value["translations"]
    if isinstance(value, list):
        value = {
            str(item.get("id", "")): item.get("text", "")
            for item in value
            if isinstance(item, dict)
        }
    if not isinstance(value, dict):
        raise TypeError("cohesion response has an unsupported JSON shape")
    revisions = {
        str(key): str(item).strip()
        for key, item in value.items()
        if str(key) in expected_ids and isinstance(item, str) and str(item).strip()
    }
    missing = expected_ids - revisions.keys()
    extra = set(value) - expected_ids
    if missing or extra:
        raise ValueError(
            "cohesion response IDs mismatch; "
            f"missing={sorted(missing)}, extra={sorted(str(item) for item in extra)}"
        )
    return revisions
