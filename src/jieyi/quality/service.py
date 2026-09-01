from __future__ import annotations

from jieyi.domain.models import Segment, TermEntry, TermStatus
from jieyi.terminology import matching_terms

from .checks import DETECTOR_VERSION, run_deterministic_checks

_QUALITY_META_KEY = "quality_detector_version"


def visible_translation(segment: Segment) -> str:
    return (
        segment.accepted_translation
        or segment.reviewed_translation
        or segment.edited_translation
        or segment.machine_translation
        or ""
    )


def relevant_terms(source: str, terms: list[TermEntry]) -> list[TermEntry]:
    return matching_terms(
        source,
        [term for term in terms if term.status is TermStatus.APPROVED],
    )


def refresh_segment_quality(store, segment_id: str, *, job_id: str | None = None):
    """Recompute the current findings after any target-text mutation."""
    segment = store.get_segment(segment_id)
    target = visible_translation(segment)
    jobs = store.list_jobs(segment.document_id)
    owning_job_id = job_id or (jobs[0].id if jobs else None)
    if owning_job_id is None:
        return []
    project = store.get_project_for_document(segment.document_id)
    terms = relevant_terms(segment.source_text, store.list_terms(project.id))
    issues = (
        run_deterministic_checks(
            segment.source_text,
            target,
            terms,
            segment_kind=segment.kind,
        )
        if target
        else []
    )
    store.replace_issues(
        owning_job_id,
        segment.id,
        issues,
        target_text=target,
        detector_version=DETECTOR_VERSION,
    )
    return issues


def reindex_document_quality(store, document_id: str) -> int:
    """Rebuild current findings for every translated segment in a document."""
    jobs = store.list_jobs(document_id)
    if not jobs:
        return 0
    count = 0
    for segment in store.list_segments(document_id):
        if not visible_translation(segment):
            continue
        refresh_segment_quality(store, segment.id, job_id=jobs[0].id)
        count += 1
    return count


def reindex_project_quality(store, project_id: str) -> int:
    return sum(
        reindex_document_quality(store, document.id)
        for document in store.list_documents(project_id)
    )


def reindex_all_quality(store, *, force: bool = False) -> int:
    """Run once per detector version; interrupted runs safely restart next launch."""
    if not force and store.get_meta(_QUALITY_META_KEY) == DETECTOR_VERSION:
        return 0
    count = sum(
        reindex_project_quality(store, project.id) for project in store.list_projects()
    )
    store.set_meta(_QUALITY_META_KEY, DETECTOR_VERSION)
    return count
