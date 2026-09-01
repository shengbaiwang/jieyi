from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from jieyi.domain.models import TermEntry, TermStatus

_WESTERN_TERM_CHAR = re.compile(r"[A-Za-z0-9_]")


def normalize_term_text(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold().strip()


def term_appears(text: str, term: str) -> bool:
    """Match Western terms by token boundaries and CJK terms by substring."""
    haystack = normalize_term_text(text)
    needle = normalize_term_text(term)
    if not needle:
        return False
    western_start = bool(_WESTERN_TERM_CHAR.match(needle[0]))
    western_end = bool(_WESTERN_TERM_CHAR.match(needle[-1]))
    start = 0
    while True:
        index = haystack.find(needle, start)
        if index < 0:
            return False
        before = haystack[index - 1] if index else ""
        end = index + len(needle)
        after = haystack[end] if end < len(haystack) else ""
        if not (western_start and before and _WESTERN_TERM_CHAR.match(before)) and not (
            western_end and after and _WESTERN_TERM_CHAR.match(after)
        ):
            return True
        start = index + 1


def source_forms(term: TermEntry) -> tuple[str, ...]:
    seen: set[str] = set()
    forms: list[str] = []
    for value in (term.source, *term.aliases):
        normalized = normalize_term_text(value)
        if normalized and normalized not in seen:
            seen.add(normalized)
            forms.append(value.strip())
    return tuple(forms)


@dataclass(frozen=True, slots=True)
class TermEvidence:
    term: TermEntry
    matched_forms: tuple[str, ...]
    context_hits: tuple[str, ...]


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


def resolve_terminology(
    source: str,
    terms: list[TermEntry] | tuple[TermEntry, ...],
) -> TerminologyResolution:
    """Resolve aliases and explicit context clues without forcing ambiguous senses."""
    evidence: list[TermEvidence] = []
    by_form: dict[str, list[TermEvidence]] = {}
    display_form: dict[str, str] = {}
    for term in terms:
        if term.status is not TermStatus.APPROVED:
            continue
        matched = tuple(form for form in source_forms(term) if term_appears(source, form))
        if not matched:
            continue
        context_hits = tuple(
            keyword for keyword in term.context_keywords if term_appears(source, keyword)
        )
        item = TermEvidence(term=term, matched_forms=matched, context_hits=context_hits)
        evidence.append(item)
        for form in matched:
            key = normalize_term_text(form)
            display_form.setdefault(key, form)
            by_form.setdefault(key, []).append(item)

    enforced_ids: set[str] = set()
    ambiguous: list[AmbiguousTerm] = []
    for key, raw_candidates in by_form.items():
        candidates = list({item.term.id: item for item in raw_candidates}.values())
        contextual = [item for item in candidates if item.context_hits]
        selected: TermEvidence | None = None
        if contextual:
            best_score = max(len(item.context_hits) for item in contextual)
            best = [item for item in contextual if len(item.context_hits) == best_score]
            if len(best) == 1:
                selected = best[0]
            else:
                candidates = best
        else:
            defaults = [item for item in candidates if not item.term.context_keywords]
            if len(defaults) == 1:
                selected = defaults[0]
            elif len(candidates) == 1 and not candidates[0].term.context_keywords:
                selected = candidates[0]

        if selected is not None:
            enforced_ids.add(selected.term.id)
        else:
            ambiguous.append(
                AmbiguousTerm(
                    source_form=display_form[key],
                    candidates=tuple(candidates),
                )
            )

    enforced = tuple(item for item in evidence if item.term.id in enforced_ids)
    return TerminologyResolution(
        enforced=enforced,
        ambiguous=tuple(ambiguous),
        candidates=tuple(evidence),
    )


def matching_terms(
    source: str,
    terms: list[TermEntry] | tuple[TermEntry, ...],
) -> list[TermEntry]:
    return list(resolve_terminology(source, terms).matched_terms)


def render_terminology_constraints(
    source: str,
    terms: list[TermEntry] | tuple[TermEntry, ...],
) -> str:
    resolution = resolve_terminology(source, terms)
    sections: list[str] = []
    if resolution.enforced:
        sections.append("# APPROVED TERMINOLOGY — MANDATORY CONSTRAINTS")
        sections.append(
            "Use every selected target exactly and consistently. These are constraints, not references."
        )
        for evidence in resolution.enforced:
            term = evidence.term
            forms = " | ".join(source_forms(term))
            line = f"- {forms} -> {term.target} [MANDATORY]"
            if term.sense:
                line += f"; sense: {term.sense}"
            if evidence.context_hits:
                line += f"; context matched: {', '.join(evidence.context_hits)}"
            if term.rationale:
                line += f"; rationale: {term.rationale}"
            if term.forbidden_targets:
                line += f"; never use: {', '.join(term.forbidden_targets)}"
            sections.append(line)
    if resolution.ambiguous:
        if sections:
            sections.append("")
        sections.append("# TERMINOLOGY REQUIRING CONTEXTUAL DISAMBIGUATION")
        sections.append(
            "Choose only the approved sense supported by this passage. Do not blend senses or guess silently."
        )
        for ambiguity in resolution.ambiguous:
            sections.append(f'- Ambiguous source form: "{ambiguity.source_form}"')
            for evidence in ambiguity.candidates:
                term = evidence.term
                label = term.sense or term.source
                guidance = term.disambiguation or term.rationale or "human confirmation required"
                keywords = ", ".join(term.context_keywords) or "none supplied"
                sections.append(
                    f"  - {label} -> {term.target}; context keywords: {keywords}; guidance: {guidance}"
                )
    return "\n".join(sections)
