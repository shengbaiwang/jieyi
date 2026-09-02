from __future__ import annotations

import re
import unicodedata
from collections import Counter

from jieyi.domain.models import IssueSeverity, QualityIssue, SegmentKind, TermEntry
from jieyi.terminology import resolve_terminology, term_appears

DETECTOR_VERSION = "6"

_FOOTNOTE_REF = re.compile(r"\[\^[^]]+\]")


def _normalise_text(text: str) -> str:
    value = unicodedata.normalize("NFKC", text)
    return value.translate(str.maketrans({"−": "-", "–": "-", "—": "-", "﹣": "-"}))


def _counter(pattern: re.Pattern[str], text: str) -> Counter[str]:
    normalised = _normalise_text(text)
    return Counter(match.group(0) for match in pattern.finditer(normalised))


def _issue(
    code: str,
    message: str,
    severity: IssueSeverity,
    details: dict | None = None,
    *,
    confidence: str = "high",
) -> QualityIssue:
    return QualityIssue(
        code,
        message,
        severity,
        {**(details or {}), "detector_version": DETECTOR_VERSION, "confidence": confidence},
    )


def run_deterministic_checks(
    source: str,
    target: str,
    terms: list[TermEntry] | tuple[TermEntry, ...] = (),
    *,
    segment_kind: SegmentKind | str | None = None,
) -> list[QualityIssue]:
    """Check high-confidence invariants without guessing about punctuation or numbers."""
    issues: list[QualityIssue] = []

    if source.strip() and not target.strip():
        return [_issue("empty_translation", "译文为空。", IssueSeverity.ERROR)]

    source_footnotes = _counter(_FOOTNOTE_REF, source)
    target_footnotes = _counter(_FOOTNOTE_REF, target)
    if source_footnotes != target_footnotes:
        issues.append(
            _issue(
                "footnote_mismatch",
                "脚注标记在原文和译文中不一致。",
                IssueSeverity.ERROR,
                {"source": dict(source_footnotes), "target": dict(target_footnotes)},
            )
        )

    terminology = resolve_terminology(source, terms)
    for evidence in terminology.enforced:
        term = evidence.term
        if not term_appears(target, term.target):
            issues.append(
                _issue(
                    "approved_term_missing",
                    f"已批准术语“{'／'.join(evidence.matched_forms)}”应统一译为“{term.target}”。",
                    IssueSeverity.ERROR,
                    {
                        "term_id": term.id,
                        "source": term.source,
                        "matched_forms": list(evidence.matched_forms),
                        "target": term.target,
                        "sense": term.sense,
                    },
                )
            )
        forbidden_hits = [
            item for item in term.forbidden_targets if item and term_appears(target, item)
        ]
        if forbidden_hits:
            issues.append(
                _issue(
                    "forbidden_term_used",
                    f"译文使用了术语“{term.source}”的禁用译法。",
                    IssueSeverity.ERROR,
                    {"term_id": term.id, "hits": forbidden_hits},
                )
            )

    for ambiguity in terminology.ambiguous:
        issues.append(
            _issue(
                "terminology_pending",
                f"术语“{ambiguity.source_form}”待逐处核验义项与译法，尚未判定为错误。",
                IssueSeverity.INFO,
                {
                    "source_form": ambiguity.source_form,
                    "requires_human": False,
                    "source": "terminology_rule",
                    "candidates": [
                        {
                            "term_id": item.term.id,
                            "source": item.term.source,
                            "target": item.term.target,
                            "sense": item.term.sense,
                            "context_keywords": list(item.term.context_keywords),
                            "disambiguation": item.term.disambiguation,
                        }
                        for item in ambiguity.candidates
                    ],
                },
                confidence="unverified",
            )
        )

    return issues
