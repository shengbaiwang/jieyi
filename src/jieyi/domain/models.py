from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from .reasoning import normalize_compute_mode


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


class SegmentKind(StrEnum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    BLOCKQUOTE = "blockquote"
    FOOTNOTE = "footnote"
    LIST_ITEM = "list_item"
    TABLE_CELL = "table_cell"
    CAPTION = "caption"
    VERSE = "verse"


class SegmentStatus(StrEnum):
    SOURCE = "source"
    MACHINE_TRANSLATED = "machine_translated"
    HUMAN_CONFIRMED = "human_confirmed"


class JobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TermStatus(StrEnum):
    PROPOSED = "proposed"
    APPROVED = "approved"
    DEPRECATED = "deprecated"
    FORBIDDEN = "forbidden"


class CandidateStage(StrEnum):
    DRAFT = "draft"
    REPAIR = "repair"


class IssueSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class Project:
    id: str
    name: str
    source_lang: str
    target_lang: str
    domain: str = "humanities_and_social_sciences"
    style_guide: str = ""
    quote_policy: str = "preserve_source_citations"
    created_at: str = field(default_factory=utc_now)


@dataclass(frozen=True, slots=True)
class Document:
    id: str
    project_id: str
    title: str
    source_format: str
    source_hash: str
    created_at: str = field(default_factory=utc_now)


@dataclass(frozen=True, slots=True)
class Segment:
    id: str
    document_id: str
    stable_key: str
    ordinal: int
    kind: SegmentKind
    source_text: str
    heading_path: str = ""
    machine_translation: str | None = None
    edited_translation: str | None = None
    reviewed_translation: str | None = None
    accepted_translation: str | None = None
    status: SegmentStatus = SegmentStatus.SOURCE
    source_refs: tuple[str, ...] = ()
    segmentation_confidence: float = 1.0
    segmentation_reason: str = ""
    segmenter_version: str = ""


@dataclass(frozen=True, slots=True)
class TermEntry:
    id: str
    project_id: str
    source: str
    target: str
    status: TermStatus = TermStatus.PROPOSED
    scope: str = "project"
    domain: str = ""
    rationale: str = ""
    forbidden_targets: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    context_keywords: tuple[str, ...] = ()
    sense: str = ""
    disambiguation: str = ""
    enforcement: str = "auto"  # auto | global | contextual
    created_at: str = field(default_factory=utc_now)


@dataclass(frozen=True, slots=True)
class ModelSpec:
    provider: str
    model: str
    temperature: float = 0.1


@dataclass(frozen=True, slots=True)
class TranslationRecipe:
    draft: ModelSpec
    neighbor_radius: int = 1
    max_context_chars: int = 12_000
    tm_enabled: bool = True
    tm_threshold: float = 0.78
    tm_max_results: int = 3
    batch_size: int = 10
    concurrency: int = 3
    max_concurrency: int = 5
    max_batch_chars: int = 4_000
    draft_thinking: bool = False
    draft_compute_mode: str = "economy"
    draft_reasoning_effort: str = "none"
    max_output_tokens: int = 6_000
    token_budget: int = 2_000_000
    segment_ranges: tuple[tuple[int, int], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TranslationRecipe:
        concurrency = int(value.get("concurrency", 3))
        legacy_max_concurrency = 5 if concurrency == 3 else concurrency
        return cls(
            draft=ModelSpec(**value["draft"]),
            neighbor_radius=int(value.get("neighbor_radius", 1)),
            max_context_chars=int(value.get("max_context_chars", 12_000)),
            tm_enabled=bool(value.get("tm_enabled", True)),
            tm_threshold=float(value.get("tm_threshold", 0.78)),
            tm_max_results=int(value.get("tm_max_results", 3)),
            batch_size=int(value.get("batch_size", 10)),
            concurrency=concurrency,
            max_concurrency=int(
                value.get("max_concurrency", legacy_max_concurrency)
            ),
            max_batch_chars=int(value.get("max_batch_chars", 4_000)),
            draft_thinking=bool(value.get("draft_thinking", False)),
            draft_compute_mode=normalize_compute_mode(
                value.get("draft_compute_mode") or value.get("draft_reasoning_effort"),
                "economy",
            ),
            draft_reasoning_effort=str(
                value.get(
                    "draft_reasoning_effort",
                    "high" if value.get("draft_thinking", False) else "none",
                )
            ),
            max_output_tokens=int(value.get("max_output_tokens", 6_000)),
            token_budget=int(value.get("token_budget", 2_000_000)),
            segment_ranges=tuple(
                (int(item[0]), int(item[1]))
                for item in value.get("segment_ranges", ())
            ),
        )


@dataclass(frozen=True, slots=True)
class TranslationRequest:
    project: Project
    document: Document
    segment: Segment
    context: str
    segment_context: str = ""
    task: CandidateStage = CandidateStage.DRAFT
    existing_translation: str | None = None
    issue_summary: str = ""
    atom_boundaries: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class TranslationResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    reasoning_tokens: int = 0
    prompt_cache_hit_tokens: int = 0
    prompt_cache_miss_tokens: int = 0
    raw_response: str | None = None


@dataclass(frozen=True, slots=True)
class QualityIssue:
    code: str
    message: str
    severity: IssueSeverity = IssueSeverity.WARNING
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TranslationMemoryMatch:
    id: str
    project_id: str
    source_text: str
    target_text: str
    similarity: float
    source_segment_id: str | None = None
    updated_at: str = field(default_factory=utc_now)


@dataclass(frozen=True, slots=True)
class Job:
    id: str
    document_id: str
    recipe: TranslationRecipe
    status: JobStatus = JobStatus.PENDING
    next_ordinal: int = 0
    total_cost_usd: float = 0.0
    last_error: str | None = None
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
