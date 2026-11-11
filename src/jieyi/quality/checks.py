from __future__ import annotations

import re
import unicodedata
from collections import Counter
from html import unescape

from jieyi.domain.models import IssueSeverity, QualityIssue, SegmentKind, TermEntry
from jieyi.protection import ProtectedTextCodec
from jieyi.quality.output import output_problems
from jieyi.quality.references import is_retained_reference
from jieyi.terminology import resolve_terminology, term_appears

DETECTOR_VERSION = "8"

_FOOTNOTE_REF = re.compile(r"\[\^[^]]+\]")
_PLACEHOLDER = re.compile(r"\[\[JY_PH_\d{4}\]\]")
_WORD = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*")
_EAST_ASIAN = re.compile(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]")
_TARGET_SCRIPTS = {
    "zh": re.compile(r"[\u3400-\u9fff]"),
    "ja": re.compile(r"[\u3400-\u9fff\u3040-\u30ff]"),
    "ko": re.compile(r"[\uac00-\ud7af]"),
}


def _untranslated_text(
    source: str, target: str, source_lang: str | None, target_lang: str | None,
) -> bool:
    """Find copied prose, leaving short names and protected technical text alone."""
    if not source_lang or not target_lang:
        return False
    source_language = source_lang.strip().lower().replace("_", "-").split("-")[0]
    target_language = target_lang.strip().lower().replace("_", "-").split("-")[0]
    if not source_language or not target_language or source_language == target_language:
        return False

    def prose(text: str) -> str:
        # Match the translation contract: citations, URLs, code, markup and
        # footnote markers are allowed to survive unchanged.
        masked = ProtectedTextCodec().encode(text).masked
        return unicodedata.normalize("NFKC", unescape(_PLACEHOLDER.sub(" ", masked)))

    original, translated = prose(source), prose(target)
    source_letters = "".join(char for char in original if char.isalpha())
    if not source_letters:
        return False
    target_script = _TARGET_SCRIPTS.get(target_language)
    if target_script and len(target_script.findall(source_letters)) >= len(source_letters) / 2:
        # A passage already written in the target script may need no translation.
        return False
    if len(_EAST_ASIAN.findall(source_letters)) >= len(source_letters) / 2:
        is_prose = len(source_letters) >= 8
    else:
        words = _WORD.findall(original)
        # Several lowercase words distinguish ordinary prose from most personal
        # names, acronyms and compact labels; include short punctuated sentences.
        sentence_end = original.rstrip().endswith((".", "!", "?"))
        is_prose = (
            len(source_letters) >= 10
            and sum(word[0].islower() for word in words) >= 2
            and (len(words) >= 4 or (len(words) >= 3 and sentence_end))
        ) or (
            source_letters.isupper() and len(words) >= 6
            and len(source_letters) >= 20 and sentence_end
        )
    if not is_prose:
        return False

    def comparable(text: str) -> str:
        return "".join(char for char in text.casefold() if char.isalnum())

    return comparable(original) == comparable(translated) and not is_retained_reference(source)


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
    source_lang: str | None = None,
    target_lang: str | None = None,
) -> list[QualityIssue]:
    """Check high-confidence invariants without guessing about punctuation or numbers."""
    issues: list[QualityIssue] = []

    if source.strip() and not target.strip():
        return [_issue("empty_translation", "译文为空。", IssueSeverity.ERROR)]

    if _untranslated_text(source, target, source_lang, target_lang):
        issues.append(_issue(
            "untranslated_text",
            "译文与原文相同，疑似未成功翻译，请重新翻译或人工核实。",
            IssueSeverity.ERROR,
            {"source_lang": source_lang, "target_lang": target_lang},
        ))

    for code in output_problems(source, target):
        issues.append(_issue(
            code, "译文疑似混入原文复述、翻译标签或解释，请重新翻译本段。", IssueSeverity.ERROR,
        ))

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
