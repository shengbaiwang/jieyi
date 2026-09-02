from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from jieyi.domain.models import TermEntry, TermStatus


def normalize_term_text(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold().strip()


def term_spans(text: str, term: str) -> tuple[tuple[int, int], ...]:
    """Return original offsets, including when casefold/NFKC changes string length."""
    needle = normalize_term_text(term)
    if not needle:
        return ()
    pieces = [unicodedata.normalize("NFKC", char).casefold() for char in text]
    haystack = "".join(pieces)
    offsets = [index for index, piece in enumerate(pieces) for _ in piece]
    def western(char):
        name = unicodedata.name(char, "")
        return char.isdigit() or any(script in name for script in ("LATIN", "GREEK", "CYRILLIC"))
    spans = []
    start = 0
    while (index := haystack.find(needle, start)) >= 0:
        end = index + len(needle)
        before = haystack[index - 1] if index else ""
        after = haystack[end] if end < len(haystack) else ""
        if not (western(needle[0]) and before and (western(before) or before == "_")) and not (
            western(needle[-1]) and after and (western(after) or after == "_")
        ):
            span = (offsets[index], offsets[end - 1] + 1)
            if normalize_term_text(text[span[0] : span[1]]) == needle and span not in spans:
                spans.append(span)
        start = end
    return tuple(spans)


def term_appears(text: str, term: str) -> bool:
    return bool(term_spans(text, term))


def source_forms(term: TermEntry) -> tuple[str, ...]:
    seen: set[str] = set()
    forms: list[str] = []
    for value in (term.source, *term.aliases):
        normalized = normalize_term_text(value)
        if normalized and normalized not in seen:
            seen.add(normalized)
            forms.append(value.strip())
    return tuple(forms)


def requires_context(term: TermEntry) -> bool:
    if term.enforcement != "auto":
        return term.enforcement != "global"
    return bool(term.sense.strip() or term.disambiguation.strip() or term.context_keywords)


@dataclass(frozen=True, slots=True)
class TermOccurrence:
    start: int
    end: int
    text: str
    sentence: str


@dataclass(frozen=True, slots=True)
class TermEvidence:
    term: TermEntry
    matched_forms: tuple[str, ...]
    context_hits: tuple[str, ...]
    occurrences: tuple[TermOccurrence, ...] = ()


@dataclass(frozen=True, slots=True)
class AmbiguousTerm:
    source_form: str
    candidates: tuple[TermEvidence, ...]


@dataclass(frozen=True, slots=True)
class TerminologyResolution:
    enforced: tuple[TermEvidence, ...] = ()
    ambiguous: tuple[AmbiguousTerm, ...] = ()
    candidates: tuple[TermEvidence, ...] = ()

    @property
    def matched_terms(self) -> tuple[TermEntry, ...]:
        return tuple(item.term for item in self.candidates)


def _sentence(source: str, start: int, end: int) -> str:
    left = list(re.finditer(r"[.!?。！？\n]", source[:start]))
    right = re.search(r"[.!?。！？\n]", source[end:])
    return source[
        left[-1].end() if left else 0 : end + right.end() if right else len(source)
    ].strip()


def resolve_terminology(
    source: str,
    terms: list[TermEntry] | tuple[TermEntry, ...],
) -> TerminologyResolution:
    """Approval fixes a translation for a sense, never proves that sense applies.

    Keywords remain hints for a semantic check. A longer approved phrase owns its
    span; independent short-word occurrences elsewhere in the paragraph remain.
    """
    matches = [
        (term, form, start, end)
        for term in terms
        if term.status is TermStatus.APPROVED
        for form in source_forms(term)
        for start, end in term_spans(source, form)
    ]
    matches = [
        match
        for match in matches
        if not any(
            other[2] <= match[2]
            and other[3] >= match[3]
            and other[3] - other[2] > match[3] - match[2]
            for other in matches
        )
    ]
    evidence = []
    for term in terms:
        hits = [match for match in matches if match[0].id == term.id]
        if not hits:
            continue
        evidence.append(
            TermEvidence(
                term=term,
                matched_forms=tuple(dict.fromkeys(hit[1] for hit in hits)),
                context_hits=tuple(
                    key for key in term.context_keywords if term_appears(source, key)
                ),
                occurrences=tuple(
                    TermOccurrence(start, end, source[start:end], _sentence(source, start, end))
                    for start, end in sorted({(hit[2], hit[3]) for hit in hits})
                ),
            )
        )
    enforced, conditional = [], []
    for item in evidence:
        competing = any(
            other.term.id != item.term.id
            and any(
                a.start == b.start and a.end == b.end
                for a in item.occurrences
                for b in other.occurrences
            )
            for other in evidence
        )
        if requires_context(item.term) or competing:
            conditional.append(item)
        else:
            enforced.append(item)
    by_form: dict[str, list[TermEvidence]] = {}
    for item in conditional:
        by_form.setdefault(normalize_term_text(item.term.source), []).append(item)
    return TerminologyResolution(
        enforced=tuple(enforced),
        ambiguous=tuple(
            AmbiguousTerm(items[0].term.source, tuple(items)) for items in by_form.values()
        ),
        candidates=tuple(evidence),
    )


def matching_terms(source: str, terms: list[TermEntry] | tuple[TermEntry, ...]) -> list[TermEntry]:
    return list(resolve_terminology(source, terms).matched_terms)


def render_terminology_constraints(
    source: str,
    terms: list[TermEntry] | tuple[TermEntry, ...],
) -> str:
    resolution = resolve_terminology(source, terms)
    sections: list[str] = []
    if resolution.enforced:
        sections += [
            "# APPROVED TERMINOLOGY — MANDATORY CONSTRAINTS",
            "Use these globally approved translations at their matched occurrences.",
        ]
        for evidence in resolution.enforced:
            term = evidence.term
            line = f"- {' | '.join(source_forms(term))} -> {term.target} [MANDATORY]"
            if term.rationale:
                line += f"; rationale: {term.rationale}"
            if term.forbidden_targets:
                line += f"; never use: {', '.join(term.forbidden_targets)}"
            sections.append(line)
    if resolution.ambiguous:
        sections += [
            "# CONDITIONALLY APPROVED TERMINOLOGY — CHECK EACH OCCURRENCE",
            (
                "For each occurrence, determine whether the approved sense applies before using its "
                "target. Other senses, grammar uses, compounds and retained citations may be NOT "
                "APPLICABLE even when there is only one approved entry. Keywords are hints, not proof. "
                "Do not insert a target elsewhere to satisfy a string check. Use the natural translation "
                "when no approved sense applies. Ask for human judgment only if uncertainty remains."
            ),
        ]
        for ambiguity in resolution.ambiguous:
            for evidence in ambiguity.candidates:
                term = evidence.term
                sections.append(
                    f"- {' | '.join(source_forms(term))} -> {term.target} [CONDITIONAL]; "
                    f"sense: {term.sense or 'context-dependent'}; "
                    f"guidance: {term.disambiguation or term.rationale or 'check local meaning'}; "
                    f"keyword hints: {', '.join(term.context_keywords) or 'none'}; "
                    f"forbidden in this sense: {', '.join(term.forbidden_targets) or 'none'}"
                )
                for occurrence in evidence.occurrences:
                    sections.append(
                        f"  - source[{occurrence.start}:{occurrence.end}]: "
                        f"{occurrence.text}; sentence: {occurrence.sentence}"
                    )
    return "\n".join(sections)
