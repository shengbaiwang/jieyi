from __future__ import annotations

from dataclasses import dataclass, replace

from jieyi.context.compiler import compile_neighbor_context
from jieyi.domain.models import Segment, TermEntry, TranslationRequest
from jieyi.protection import ProtectedText
from jieyi.terminology import matching_terms, render_terminology_constraints


@dataclass(frozen=True, slots=True)
class PreparedTranslation:
    request: TranslationRequest
    protected: ProtectedText
    terms: list[TermEntry]
    structured: bool


def project_context(project, max_chars: int) -> str:
    lines = [
        "# PROJECT",
        f"Source language: {project.source_lang}",
        f"Target language: {project.target_lang}",
        f"Domain: {project.domain}",
        f"Quote policy: {project.quote_policy}",
    ]
    if project.style_guide.strip():
        lines.extend(["", "# STYLE GUIDE", project.style_guide.strip()])
    value = "\n".join(lines)
    if len(value) > max_chars:
        return value[:max(0, max_chars - 40)] + "\n[context truncated by budget]"
    return value


def prepare_translation(
    store, codec, project, document, segment: Segment, recipe,
    approved_terms: list[TermEntry], segments_by_ordinal: dict[int, Segment],
    *, shared_context: str | None = None,
) -> PreparedTranslation:
    """One lossless preparation path for optimized execution and its preview.

    Do not shorten the source, terms, or neighboring passages to save tokens.
    A sole outer EPUB atom has no alignment decision for a model to make;
    its exact envelope is restored locally after inner placeholder validation.
    """
    structured_source = store.epub_translation_source(segment.id)
    protected = codec.encode(structured_source or segment.source_text)
    if structured_source:
        protected = protected.compact_single_atom()
    relevant = matching_terms(segment.source_text, approved_terms)
    terms = render_terminology_constraints(segment.source_text, relevant)
    shared = shared_context if shared_context is not None else project_context(
        project, recipe.max_context_chars,
    )
    radius = max(0, recipe.neighbor_radius)
    neighbors = [
        segments_by_ordinal[ordinal]
        for ordinal in range(max(0, segment.ordinal - radius), segment.ordinal + radius + 1)
        if ordinal != segment.ordinal and ordinal in segments_by_ordinal
    ]
    neighbors_text = compile_neighbor_context(
        segment, neighbors,
        max_chars=max(0, recipe.max_context_chars - len(shared) - len(terms) - 4),
        include_translations=False,
    )
    request = TranslationRequest(
        project=project, document=document,
        segment=replace(segment, source_text=protected.masked),
        atom_boundaries=protected.atom_boundaries if structured_source else (),
        context=shared,
        segment_context="\n\n".join(filter(None, [terms, neighbors_text])),
    )
    return PreparedTranslation(request, protected, relevant, bool(structured_source))


def initial_output_budget(request: TranslationRequest, compute_mode: str, cap: int) -> int:
    """Reserve room for reasoning without forcing extra output or changing effort.

    max_tokens is a ceiling, not a target. Previously very short sources gave
    thinking models just 512 tokens and often paid for a second full request.
    """
    source_chars = len(request.segment.source_text) + len(request.existing_translation or "")
    reasoning_room = {"economy": 0, "balanced": 2048, "performance": 4096}[compute_mode]
    return min(cap, max(512, source_chars * 2) + reasoning_room)
