from __future__ import annotations

from jieyi.domain.models import Segment, TermEntry, TermStatus
from jieyi.terminology import matching_terms

from .checks import DETECTOR_VERSION, run_deterministic_checks

_QUALITY_META_KEY = "quality_detector_version"
# Only this detector's findings are superseded by deterministic reindexing.
_DETERMINISTIC_CODES = {
    "empty_translation", "footnote_mismatch", "approved_term_missing", "forbidden_term_used",
    "ambiguous_term_unresolved", "terminology_pending", "number_mismatch", "numeric_mismatch",
    "parenthesis_mismatch", "bracket_mismatch", "unbalanced_parentheses",
}


def requires_human_review(issue: dict) -> bool:
    return issue.get("details", {}).get("requires_human", issue["severity"] != "info")


def document_issues(store, document_id: str) -> list[dict]:
    from .terminology_review import TerminologyReviewRepository, contextual_issues
    findings = [issue for issue in store.list_issues(document_id)
                if issue["code"] != "terminology_pending"]
    findings.extend(contextual_issues(TerminologyReviewRepository(store), document_id))
    return sorted(findings, key=lambda issue: (issue["ordinal"], issue["id"]))


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
        replace_codes=_DETERMINISTIC_CODES,
    )
    return issues


def reindex_document_quality(store, document_id: str) -> int:
    """Rebuild deterministic snapshots, retaining findings from other sources.

    Fetch shared context once, and do not open a write transaction for clean
    segments with no previous detector findings. This keeps version migrations
    bounded even for book-sized projects.
    """
    jobs = store.list_jobs(document_id)
    if not jobs:
        return 0
    project = store.get_project_for_document(document_id)
    terms = store.list_terms(project.id)
    with store._connect() as connection:
        previous_ids = {
            row["segment_id"] for row in connection.execute(
                "SELECT i.segment_id, i.code FROM issues i "
                "JOIN segments s ON s.id=i.segment_id "
                "WHERE s.document_id=? AND i.resolved=0", (document_id,),
            ) if row["code"] in _DETERMINISTIC_CODES
        }
    count = 0
    for segment in store.list_segments(document_id):
        target = visible_translation(segment)
        if not target:
            continue
        findings = run_deterministic_checks(
            segment.source_text, target, terms, segment_kind=segment.kind,
        )
        if findings or segment.id in previous_ids:
            store.replace_issues(
                jobs[0].id, segment.id, findings, target_text=target,
                detector_version=DETECTOR_VERSION, replace_codes=_DETERMINISTIC_CODES,
            )
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
