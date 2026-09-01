from __future__ import annotations

import re
import unicodedata
from collections import Counter

from jieyi.domain.models import IssueSeverity, QualityIssue, SegmentKind, TermEntry
from jieyi.terminology import resolve_terminology, term_appears

from .numeric import compare_numeric_facts

DETECTOR_VERSION = "4"

_PAREN_CITATION = re.compile(
    r"\([^()\n]{0,80}\b(?:1[5-9]|20)\d{2}[a-z]?[^()\n]{0,40}\)", re.IGNORECASE
)
_BRACKET_CITATION = re.compile(r"\[(?:\d{1,4}(?:\s*[-,;]\s*\d{1,4})*)\]")
_FOOTNOTE_REF = re.compile(r"\[\^[^]]+\]")


def _normalise_text(text: str) -> str:
    value = unicodedata.normalize("NFKC", text)
    return value.translate(str.maketrans({"−": "-", "–": "-", "—": "-", "﹣": "-"}))


def _counter(pattern: re.Pattern[str], text: str) -> Counter[str]:
    normalised = _normalise_text(text)
    return Counter(match.group(0) for match in pattern.finditer(normalised))


def _parentheses_balanced(text: str) -> bool:
    stack: list[str] = []
    line_start = 0
    for index, char in enumerate(text):
        if char == "\n":
            line_start = index + 1
            continue
        if char in "(（":
            stack.append(char)
            continue
        if char not in ")）":
            continue
        if stack:
            stack.pop()
            continue
        # A leading "1)" or "12）" is a list marker, not an unmatched pair.
        if re.fullmatch(r"\s*\d{1,3}", text[line_start:index]):
            continue
        return False
    return not stack


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
    """Check high-confidence invariants and flag localisable numbers as review risks."""
    issues: list[QualityIssue] = []

    if source.strip() and not target.strip():
        return [_issue("empty_translation", "译文为空。", IssueSeverity.ERROR)]

    # References are protected by the translation protocol and should round-trip
    # exactly after Unicode normalisation. A mismatch is therefore a hard error.
    for code, label, pattern in (
        ("citation_mismatch", "括号引文", _PAREN_CITATION),
        ("bracket_reference_mismatch", "方括号引用", _BRACKET_CITATION),
        ("footnote_mismatch", "脚注标记", _FOOTNOTE_REF),
    ):
        source_values = _counter(pattern, source)
        target_values = _counter(pattern, target)
        if source_values != target_values:
            issues.append(
                _issue(
                    code,
                    f"{label}在原文和译文中不一致。",
                    IssueSeverity.ERROR,
                    {"source": dict(source_values), "target": dict(target_values)},
                )
            )

    is_footnote = segment_kind is SegmentKind.FOOTNOTE or segment_kind == SegmentKind.FOOTNOTE.value
    if not is_footnote:
        comparison = compare_numeric_facts(source, target)
        if comparison.has_missing:
            issues.append(
                _issue(
                    "number_mismatch",
                    "原文中的数字事实未在译文中找到可靠的等价表达，请核对。",
                    IssueSeverity.WARNING,
                    {
                        "missing_typed": dict(comparison.missing_typed),
                        "missing_values": dict(comparison.missing_raw),
                        "source_facts": dict(comparison.source.typed),
                        "target_facts": dict(comparison.target.typed),
                    },
                    confidence="high",
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
                "ambiguous_term_unresolved",
                f"术语“{ambiguity.source_form}”存在多个义项，当前语境不足以自动确定译法。",
                IssueSeverity.WARNING,
                {
                    "source_form": ambiguity.source_form,
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
                confidence="medium",
            )
        )

    if not _parentheses_balanced(target):
        issues.append(
            _issue(
                "unbalanced_parentheses",
                "译文括号可能未配对。",
                IssueSeverity.WARNING,
                confidence="medium",
            )
        )

    return issues
