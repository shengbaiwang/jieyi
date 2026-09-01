from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass

from jieyi.domain.models import Segment, SegmentKind

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_FOOTNOTE = re.compile(r"^\[\^[^]]+\]:\s*")


@dataclass(frozen=True, slots=True)
class ParsedBlock:
    kind: SegmentKind
    text: str
    heading_path: str
    source_refs: tuple[str, ...] = ()
    segmentation_confidence: float = 1.0
    segmentation_reason: str = ""
    segmenter_version: str = ""


def _normalise_for_identity(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return " ".join(text.split()).strip().casefold()


def parse_text(text: str, source_format: str = "markdown") -> list[ParsedBlock]:
    """Parse TXT/Markdown without losing human-readable block boundaries."""
    normalised = text.replace("\r\n", "\n").replace("\r", "\n").strip("\n")
    if not normalised:
        return []

    raw_blocks = re.split(r"\n[ \t]*\n+", normalised)
    headings: list[str] = []
    result: list[ParsedBlock] = []

    for raw in raw_blocks:
        block = raw.strip()
        if not block:
            continue

        heading_match = _HEADING.match(block) if "\n" not in block else None
        if source_format == "markdown" and heading_match:
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()
            headings = headings[: level - 1]
            headings.append(title)
            result.append(ParsedBlock(SegmentKind.HEADING, title, " / ".join(headings)))
            continue

        if source_format == "markdown" and _FOOTNOTE.match(block):
            kind = SegmentKind.FOOTNOTE
        elif source_format == "markdown" and all(
            line.lstrip().startswith(">") for line in block.splitlines()
        ):
            kind = SegmentKind.BLOCKQUOTE
            block = "\n".join(line.lstrip()[1:].lstrip() for line in block.splitlines())
        else:
            kind = SegmentKind.PARAGRAPH

        result.append(ParsedBlock(kind, block, " / ".join(headings)))

    return result


def segments_from_text(
    document_id: str, text: str, source_format: str = "markdown"
) -> list[Segment]:
    """Build content-addressed segment keys and deterministic IDs.

    The key is independent of paragraph order, so inserting an unrelated paragraph does
    not invalidate existing identities. Duplicate blocks receive a deterministic suffix.
    """
    return segments_from_blocks(document_id, parse_text(text, source_format))


def segments_from_blocks(document_id: str, blocks: list[ParsedBlock]) -> list[Segment]:
    """Build stable segments from blocks produced by any ingestion adapter."""
    occurrences: defaultdict[str, int] = defaultdict(int)
    segments: list[Segment] = []

    for ordinal, block in enumerate(blocks):
        identity = "\x1f".join(
            [block.kind.value, block.heading_path, _normalise_for_identity(block.text)]
        )
        base = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        duplicate_index = occurrences[base]
        occurrences[base] += 1
        stable_key = f"{base}:{duplicate_index}"
        segment_digest = hashlib.sha256(
            f"{document_id}\x1f{stable_key}".encode()
        ).hexdigest()[:24]
        segments.append(
            Segment(
                id=f"seg_{segment_digest}",
                document_id=document_id,
                stable_key=stable_key,
                ordinal=ordinal,
                kind=block.kind,
                source_text=block.text,
                heading_path=block.heading_path,
                source_refs=block.source_refs,
                segmentation_confidence=block.segmentation_confidence,
                segmentation_reason=block.segmentation_reason,
                segmenter_version=block.segmenter_version,
            )
        )
    return segments
