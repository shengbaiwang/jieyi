from __future__ import annotations

import hashlib

from jieyi.domain.models import (
    Document,
    Job,
    ModelSpec,
    Project,
    TranslationRecipe,
    new_id,
)
from jieyi.domain.reasoning import (
    COMPUTE_MODES,
    LEGACY_EFFORTS,
    legacy_effort_for_mode,
    normalize_compute_mode,
)
from jieyi.ingestion import extract_epub, segments_from_blocks, segments_from_text
from jieyi.ingestion.epub_roundtrip import parse_epub_archive


def create_project(
    store,
    *,
    name: str,
    source_lang: str,
    target_lang: str,
    domain: str = "humanities_and_social_sciences",
    style_guide: str = "",
    quote_policy: str = "preserve_source_citations",
) -> Project:
    project = Project(
        id=new_id("proj"),
        name=name,
        source_lang=source_lang,
        target_lang=target_lang,
        domain=domain,
        style_guide=style_guide,
        quote_policy=quote_policy,
    )
    return store.create_project(project)


def create_document(
    store,
    *,
    project_id: str,
    title: str,
    text: str,
    source_format: str = "markdown",
) -> Document:
    store.get_project(project_id)
    if source_format not in {"txt", "markdown"}:
        raise ValueError("The MVP accepts only 'txt' and 'markdown'")
    source_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    document = Document(
        id=new_id("doc"),
        project_id=project_id,
        title=title,
        source_format=source_format,
        source_hash=source_hash,
    )
    segments = segments_from_text(document.id, text, source_format)
    if not segments:
        raise ValueError("Document contains no translatable blocks")
    return store.create_document(document, segments)


def create_epub_document(
    store,
    *,
    project_id: str,
    file_data: bytes,
    title: str | None = None,
) -> Document:
    """Import the canonical text model and retain the complete EPUB package."""
    store.get_project(project_id)
    source_hash = hashlib.sha256(file_data).hexdigest()
    book = extract_epub(file_data)
    archive = parse_epub_archive(file_data, book.source_atoms)
    existing = store.find_document_by_source_hash(project_id, source_hash)
    if existing is not None:
        store.attach_epub_archive(existing.id, archive)
        return existing

    document = Document(
        id=new_id("doc"),
        project_id=project_id,
        title=title or book.title,
        source_format="epub",
        source_hash=source_hash,
    )
    segments = segments_from_blocks(document.id, list(book.blocks))
    if not segments:
        raise ValueError("EPUB contains no translatable blocks")
    created = store.create_document(document, segments)
    store.attach_epub_archive(created.id, archive)
    return created


def create_job(
    store,
    *,
    document_id: str,
    draft_provider: str,
    draft_model: str,
    tm_enabled: bool = True,
    tm_threshold: float = 0.78,
    tm_max_results: int = 3,
    batch_size: int = 10,
    concurrency: int = 3,
    max_concurrency: int | None = None,
    max_batch_chars: int = 4_000,
    draft_thinking: bool = False,
    draft_compute_mode: str | None = None,
    draft_reasoning_effort: str | None = None,
    max_output_tokens: int = 6_000,
    token_budget: int = 2_000_000,
    segment_ranges: list[tuple[int, int]] | tuple[tuple[int, int], ...] = (),
) -> Job:
    store.get_document(document_id)
    segment_count = len(store.list_segments(document_id))
    if not 0 <= tm_threshold <= 1:
        raise ValueError("tm_threshold must be between 0 and 1")
    if tm_max_results < 0:
        raise ValueError("tm_max_results cannot be negative")
    if not 1 <= batch_size <= 20:
        raise ValueError("batch_size must be between 1 and 20")
    if not 1 <= concurrency <= 12:
        raise ValueError("concurrency must be between 1 and 12")
    resolved_max_concurrency = concurrency if max_concurrency is None else max_concurrency
    if not concurrency <= resolved_max_concurrency <= 12:
        raise ValueError("max_concurrency must be between concurrency and 12")
    if max_batch_chars < 500:
        raise ValueError("max_batch_chars must be at least 500")
    if max_output_tokens < 256 or token_budget < 1:
        raise ValueError("token limits must be positive")
    draft_request = draft_compute_mode or draft_reasoning_effort
    allowed_controls = set(COMPUTE_MODES) | set(LEGACY_EFFORTS)
    if draft_request and draft_request not in allowed_controls:
        raise ValueError("draft_compute_mode is not supported")
    resolved_draft_mode = normalize_compute_mode(
        draft_request or ("high" if draft_thinking else "none"), "economy"
    )
    resolved_draft_effort = legacy_effort_for_mode(resolved_draft_mode)
    normalized_ranges = tuple(sorted((int(start), int(end)) for start, end in segment_ranges))
    for index, (start, end) in enumerate(normalized_ranges):
        if start < 0 or end < start or end >= segment_count:
            raise ValueError("segment_ranges must contain valid inclusive document ordinals")
        if index and start <= normalized_ranges[index - 1][1]:
            raise ValueError("segment_ranges must be ordered and non-overlapping")
    recipe = TranslationRecipe(
        draft=ModelSpec(provider=draft_provider, model=draft_model),
        tm_enabled=tm_enabled,
        tm_threshold=tm_threshold,
        tm_max_results=tm_max_results,
        batch_size=batch_size,
        concurrency=concurrency,
        max_concurrency=resolved_max_concurrency,
        max_batch_chars=max_batch_chars,
        draft_thinking=resolved_draft_mode != "economy",
        draft_compute_mode=resolved_draft_mode,
        draft_reasoning_effort=resolved_draft_effort,
        max_output_tokens=max_output_tokens,
        token_budget=token_budget,
        segment_ranges=normalized_ranges,
    )
    return store.create_job(Job(id=new_id("job"), document_id=document_id, recipe=recipe))
