from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from typing import Any

from jieyi.domain.models import ModelSpec, Segment, TranslationResult, new_id, utc_now

_WORD_RE = re.compile(r"[^\W\d_]+(?:[-’'][^\W\d_]+)*", re.UNICODE)
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_ACRONYM_RE = re.compile(r"^(?:[A-Z]\.?){2,}$")
_DEFINITION_CUE_RE = re.compile(
    r"\b(?:called|defined as|known as|means|refers? to|termed|so-called)\b"
    r"|(?:称为|定义为|所谓|意指|是指)",
    re.IGNORECASE,
)
_QUOTED_RE = re.compile(r"[“\"‘']([^“”\"‘’'\n]{2,80})[”\"’']")
_SPACE_RE = re.compile(r"\s+")

# Deliberately small: this is a recall gate, never an automatic approver.
_EN_STOP = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "being",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "he",
        "her",
        "hers",
        "him",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "may",
        "might",
        "more",
        "most",
        "must",
        "no",
        "not",
        "of",
        "on",
        "one",
        "or",
        "our",
        "ours",
        "she",
        "should",
        "so",
        "than",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "to",
        "too",
        "under",
        "up",
        "us",
        "very",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
        "yours",
        "chapter",
        "figure",
        "table",
        "section",
        "part",
    ]
)
_CJK_STOP = frozenset(
    [
        "的",
        "了",
        "和",
        "与",
        "及",
        "或",
        "是",
        "在",
        "为",
        "有",
        "被",
        "把",
        "将",
        "对",
        "于",
        "从",
        "中",
        "上",
        "下",
        "这",
        "那",
        "其",
        "之",
        "而",
        "也",
        "都",
        "很",
        "更",
        "最",
        "一个",
        "一种",
        "本文",
        "本章",
        "作者",
        "章节",
        "部分",
    ]
)


@dataclass(frozen=True, slots=True)
class DiscoveryConfig:
    max_candidates: int = 40
    max_evidence_per_candidate: int = 6
    max_model_candidates: int = 40
    model_batch_size: int = 8
    min_score: float = 0.34


@dataclass(frozen=True, slots=True)
class Evidence:
    id: str
    segment_id: str
    ordinal: int
    source_form: str
    quote: str
    start_offset: int
    end_offset: int
    heading_path: str
    reason: str


@dataclass(slots=True)
class _Accumulator:
    key: str
    forms: Counter[str] = field(default_factory=Counter)
    frequency: int = 0
    segment_frequency: int = 0
    last_segment_id: str = ""
    rng_state: int = 0
    occurrences: list[tuple[Segment, int, int, str, str]] = field(default_factory=list)
    token_tuple: tuple[str, ...] = ()
    marker_reasons: set[str] = field(default_factory=set)
    left_contexts: Counter[str] = field(default_factory=Counter)
    right_contexts: Counter[str] = field(default_factory=Counter)
    translated_segments: set[str] = field(default_factory=set)


def _normalise_form(value: str) -> str:
    return _SPACE_RE.sub(" ", unicodedata.normalize("NFKC", value)).casefold().strip()


def _is_cjk(value: str) -> bool:
    return bool(_CJK_RE.search(value))


def _words_with_spans(text: str) -> list[tuple[str, int, int]]:
    return [(match.group(), match.start(), match.end()) for match in _WORD_RE.finditer(text)]


def _is_stop_phrase(tokens: tuple[str, ...]) -> bool:
    normalized = tuple(_normalise_form(token) for token in tokens)
    stop = _CJK_STOP if any(_is_cjk(token) for token in normalized) else _EN_STOP
    if not normalized or all(token in stop or len(token) == 1 for token in normalized):
        return True
    if len(normalized) > 1 and (normalized[0] in stop or normalized[-1] in stop):
        return True
    return len(normalized) > 2 and sum(token in stop for token in normalized[1:-1]) > 1


def _candidate_reason(
    segment: Segment,
    text: str,
    start: int,
    end: int,
    tokens: tuple[str, ...],
) -> set[str]:
    reasons: set[str] = set()
    value = text[start:end]
    if segment.kind.value == "heading":
        reasons.add("heading")
    if _ACRONYM_RE.match(value.replace(" ", "")):
        reasons.add("acronym")
    if "-" in value or "–" in value:
        reasons.add("hyphenated")
    prefix = text[max(0, start - 60) : start]
    suffix = text[end : min(len(text), end + 60)]
    previous = prefix.rstrip()[-1:] if prefix.rstrip() else ""
    if (
        previous
        and previous not in ".!?。！？"
        and value[:1].isupper()
        and any(character.islower() for character in value)
    ):
        reasons.add("capitalized")
    definition_after = re.match(
        r"\s*(?:(?:is|are)\s+)?(?:defined as|means?|refers? to)\b|\s*(?:是指|意指|定义为)",
        suffix,
        re.IGNORECASE,
    )
    definition_before = re.search(
        r"(?:called|termed|known as|so-called)\s*$|(?:称为|所谓)\s*$",
        prefix,
        re.IGNORECASE,
    )
    if definition_after or definition_before:
        reasons.add("definition_cue")
    if len(tokens) > 1:
        reasons.add("multiword")
    return reasons


def _record(
    accumulators: dict[str, _Accumulator],
    segment: Segment,
    start: int,
    end: int,
    tokens: tuple[str, ...],
    reason: str = "",
    seen_spans: set[tuple[str, int, int]] | None = None,
) -> None:
    value = segment.source_text[start:end].strip()
    if not value or len(value) > 100 or _is_stop_phrase(tokens):
        return
    key = _normalise_form(value)
    if not key or key.isdigit():
        return
    accumulator = accumulators.setdefault(key, _Accumulator(key=key))
    reasons = _candidate_reason(segment, segment.source_text, start, end, tokens)
    if reason:
        reasons.add(reason)
    span_key = (key, start, end)
    if seen_spans is not None and span_key in seen_spans:
        accumulator.marker_reasons.update(reasons)
        return
    if seen_spans is not None:
        seen_spans.add(span_key)
    accumulator.forms[value] += 1
    accumulator.frequency += 1
    if accumulator.last_segment_id != segment.id:
        accumulator.segment_frequency += 1
        accumulator.last_segment_id = segment.id
    accumulator.token_tuple = tuple(_normalise_form(token) for token in tokens)
    accumulator.marker_reasons.update(reasons)
    occurrence = (segment, start, end, value, ",".join(sorted(reasons)))
    reservoir_size = 32
    if len(accumulator.occurrences) < reservoir_size:
        accumulator.occurrences.append(occurrence)
    else:
        accumulator.rng_state = (1664525 * accumulator.rng_state + 1013904223) & 0xFFFFFFFF
        slot = accumulator.rng_state % accumulator.frequency
        if slot < reservoir_size:
            accumulator.occurrences[slot] = occurrence
    before = _WORD_RE.findall(segment.source_text[max(0, start - 80) : start])
    after = _WORD_RE.search(segment.source_text[end : min(len(segment.source_text), end + 80)])
    if before:
        context = _normalise_form(before[-1])
        if context in accumulator.left_contexts or len(accumulator.left_contexts) < 32:
            accumulator.left_contexts[context] += 1
    if after:
        context = _normalise_form(after.group())
        if context in accumulator.right_contexts or len(accumulator.right_contexts) < 32:
            accumulator.right_contexts[context] += 1
    if (
        segment.accepted_translation
        or segment.reviewed_translation
        or segment.edited_translation
        or segment.machine_translation
    ) and len(accumulator.translated_segments) < 3:
        accumulator.translated_segments.add(segment.id)


def _collect_candidates(segments: Iterable[Segment]) -> dict[str, _Accumulator]:
    accumulators: dict[str, _Accumulator] = {}
    for segment in segments:
        seen_spans: set[tuple[str, int, int]] = set()
        text = segment.source_text
        words = _words_with_spans(text)
        western = [item for item in words if not _is_cjk(item[0])]
        for index in range(len(western)):
            for size in range(1, min(5, len(western) - index) + 1):
                window = western[index : index + size]
                start, end = window[0][1], window[-1][2]
                between = text[start:end]
                if "\n" in between or re.search(r"[.!?;:。！？；：]", between):
                    break
                tokens = tuple(item[0] for item in window)
                _record(accumulators, segment, start, end, tokens, seen_spans=seen_spans)

        for match in re.finditer(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]{2,}", text):
            run = match.group()
            for size in range(2, min(8, len(run)) + 1):
                for offset in range(len(run) - size + 1):
                    start = match.start() + offset
                    end = start + size
                    _record(
                        accumulators,
                        segment,
                        start,
                        end,
                        (text[start:end],),
                        "cjk_ngram",
                        seen_spans,
                    )

        for match in _QUOTED_RE.finditer(text):
            value = match.group(1).strip()
            inner_start = text.find(value, match.start(), match.end())
            tokens = tuple(item[0] for item in _words_with_spans(value)) or (value,)
            _record(
                accumulators,
                segment,
                inner_start,
                inner_start + len(value),
                tokens,
                "quoted",
                seen_spans,
            )
    return accumulators


def _nested_c_values(accumulators: dict[str, _Accumulator]) -> dict[str, float]:
    parents: dict[str, list[int]] = defaultdict(list)
    by_tokens = {
        accumulator.token_tuple: accumulator
        for accumulator in accumulators.values()
        if accumulator.token_tuple
    }
    for tokens, accumulator in by_tokens.items():
        if len(tokens) <= 1:
            continue
        frequency = accumulator.frequency
        for size in range(1, len(tokens)):
            for start in range(len(tokens) - size + 1):
                child = tokens[start : start + size]
                if child in by_tokens:
                    parents[by_tokens[child].key].append(frequency)
    scores: dict[str, float] = {}
    for key, accumulator in accumulators.items():
        frequency = accumulator.frequency
        nested = parents.get(key, ())
        adjusted = frequency - (sum(nested) / len(nested) if nested else 0)
        length = max(1, len(accumulator.token_tuple))
        scores[key] = max(0.0, math.log2(length + 1) * adjusted)
    return scores


def _representative_evidence(accumulator: _Accumulator, maximum: int) -> list[Evidence]:
    occurrences = accumulator.occurrences
    if len(occurrences) <= maximum:
        selected = occurrences
    else:
        indexes = {
            min(len(occurrences) - 1, int((index + 0.5) * len(occurrences) / maximum))
            for index in range(maximum)
        }
        selected = [occurrences[index] for index in sorted(indexes)]
    result: list[Evidence] = []
    for segment, start, end, form, reason in selected:
        quote_start = max(0, start - 100)
        quote_end = min(len(segment.source_text), end + 100)
        result.append(
            Evidence(
                id=new_id("evidence"),
                segment_id=segment.id,
                ordinal=segment.ordinal,
                source_form=form,
                quote=segment.source_text[quote_start:quote_end],
                start_offset=start,
                end_offset=end,
                heading_path=segment.heading_path,
                reason=reason,
            )
        )
    return result


def mine_term_candidates(
    segments: list[Segment],
    config: DiscoveryConfig | None = None,
    *,
    source_lang: str = "",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Mine evidence-bound candidates with language-aware lexical boundaries."""
    from jieyi.term_mining_v2 import mine_candidates_v2

    return mine_candidates_v2(segments, config or DiscoveryConfig(), source_lang=source_lang)


def _extract_json(value: str) -> dict[str, Any]:
    text = value.strip()
    fence = chr(96) * 3
    if text.startswith(fence):
        text = re.sub(r"^.{3}(?:json)?\s*", "", text)
        text = re.sub(r"\s*.{3}$", "", text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return {}
        try:
            payload = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return {}
    return payload if isinstance(payload, dict) else {}


def _analysis_messages(
    candidates: list[dict[str, Any]], source_lang: str, target_lang: str
) -> list[dict[str, str]]:
    cards = []
    for candidate in candidates:
        cards.append(
            {
                "candidate_id": candidate["id"],
                "canonical_form": candidate["canonical_form"],
                "observed_forms": candidate["forms"],
                "frequency": candidate["frequency"],
                "risk_score": candidate["risk_score"],
                "candidate_type": candidate.get("candidate_type", "unclassified"),
                "boundary_confidence": candidate.get("boundary_confidence", 0.0),
                "evidence": [
                    {
                        "evidence_id": evidence["id"],
                        "quote": evidence["quote"],
                        "heading": evidence["heading_path"],
                    }
                    for evidence in candidate["evidence"]
                ],
            }
        )
    system = (
        "You are reviewing automatically mined terminology for a book translation. "
        "Use only the supplied verbatim evidence. Never introduce a source term, alias, "
        "definition, sense, or claim unsupported by that evidence. Distinguish lexical "
        "forms from concepts and split genuinely different senses. Prefer concepts whose "
        "mistranslation or inconsistent translation would materially affect the book. "
        "Every supplied candidate ID must receive at least one keep or omit decision, even "
        "when the answer is omit. A proposal is advisory and will require human approval. "
        "Return strict JSON only."
    )
    user = {
        "source_language": source_lang,
        "target_language": target_lang,
        "task": (
            "Return a decision for every supplied candidate. Set keep=false for ordinary words, "
            "sentence fragments, incidental names, metadata, or items that do not need a stable "
            "book-wide translation. Suggest a target only for keep=true. Return multiple rows for "
            "one candidate only when the evidence clearly supports distinct senses."
        ),
        "schema": {
            "proposals": [
                {
                    "candidate_id": "exact supplied id",
                    "keep": True,
                    "sense_key": "short stable label",
                    "sense": "concise source-language sense distinction",
                    "concept_definition": "evidence-bound definition",
                    "target": "suggested target-language term; empty when keep=false",
                    "rationale": "why consistency matters, or why this should be omitted",
                    "disambiguation": "how to select this sense",
                    "confidence": 0.0,
                    "evidence_ids": ["exact supplied evidence id"],
                }
            ]
        },
        "candidates": cards,
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


async def enrich_candidates(
    candidates: list[dict[str, Any]],
    *,
    provider: Any,
    model: ModelSpec,
    source_lang: str,
    target_lang: str,
    config: DiscoveryConfig,
    compute_mode: str = "balanced",
) -> tuple[list[dict[str, Any]], dict[str, float | int]]:
    """Assess every bounded card, retry omissions once, and retain an auditable decision."""
    by_id = {candidate["id"]: candidate for candidate in candidates}
    usage: dict[str, float | int] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "cost_usd": 0.0,
        "model_calls": 0,
        "invalid_proposals": 0,
        "model_decisions": 0,
        "missing_decisions": 0,
        "model_kept": 0,
        "model_omitted": 0,
    }
    ranked = candidates[: config.max_model_candidates]
    for start in range(0, len(ranked), config.model_batch_size):
        batch = ranked[start : start + config.model_batch_size]
        unresolved = list(batch)
        decided_ids: set[str] = set()
        for _attempt in range(2):
            if not unresolved:
                break
            result: TranslationResult = await provider.complete(
                _analysis_messages(unresolved, source_lang, target_lang),
                model,
                compute_mode=compute_mode,
                max_tokens=4_000,
            )
            usage["prompt_tokens"] += result.prompt_tokens
            usage["completion_tokens"] += result.completion_tokens
            usage["reasoning_tokens"] += result.reasoning_tokens
            usage["cost_usd"] += result.cost_usd
            usage["model_calls"] += 1
            payload = _extract_json(result.text)
            proposals = payload.get("proposals", [])
            if not isinstance(proposals, list):
                usage["invalid_proposals"] += 1
                continue
            unresolved_by_id = {candidate["id"]: candidate for candidate in unresolved}
            grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for proposal in proposals:
                if not isinstance(proposal, dict):
                    usage["invalid_proposals"] += 1
                    continue
                candidate = unresolved_by_id.get(str(proposal.get("candidate_id", "")))
                if candidate is None:
                    usage["invalid_proposals"] += 1
                    continue
                valid_evidence = {item["id"] for item in candidate["evidence"]}
                proposed_evidence = proposal.get("evidence_ids", [])
                if not isinstance(proposed_evidence, list):
                    proposed_evidence = []
                evidence_ids = [
                    value
                    for value in proposed_evidence
                    if isinstance(value, str) and value in valid_evidence
                ]
                if not evidence_ids:
                    usage["invalid_proposals"] += 1
                    continue
                keep = bool(proposal.get("keep", False))
                try:
                    confidence = float(proposal.get("confidence") or 0.0)
                except (TypeError, ValueError):
                    confidence = 0.0
                grouped[candidate["id"]].append(
                    {
                        "id": new_id("sense"),
                        "sense_key": str(proposal.get("sense_key") or "default").strip()[:120],
                        "sense": str(proposal.get("sense", "")).strip()[:500],
                        "concept_definition": str(proposal.get("concept_definition", "")).strip()[
                            :1_000
                        ],
                        "proposed_target": (
                            str(proposal.get("target", "")).strip()[:300] if keep else ""
                        ),
                        "rationale": str(proposal.get("rationale", "")).strip()[:1_000],
                        "disambiguation": str(proposal.get("disambiguation", "")).strip()[:1_000],
                        "confidence": max(0.0, min(1.0, confidence)),
                        "ai_recommended": keep,
                        "evidence_ids": evidence_ids,
                        "proposer": f"{model.provider}:{model.model}",
                        "status": "pending",
                    }
                )
            for candidate_id, senses in grouped.items():
                by_id[candidate_id]["senses"] = senses
                decided_ids.add(candidate_id)
            unresolved = [candidate for candidate in batch if candidate["id"] not in decided_ids]
        usage["model_decisions"] += len(decided_ids)
        usage["missing_decisions"] += len(unresolved)

    for candidate in ranked:
        recommendations = [sense.get("ai_recommended") for sense in candidate["senses"]]
        if any(value is True for value in recommendations):
            usage["model_kept"] += 1
        elif recommendations and all(value is False for value in recommendations):
            usage["model_omitted"] += 1
    return candidates, usage


def discovery_fingerprint(document_hash: str, config: DiscoveryConfig) -> str:
    payload = json.dumps(
        {"document_hash": document_hash, "config": asdict(config)},
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def new_discovery_run(
    *,
    document_id: str,
    fingerprint: str,
    config: DiscoveryConfig,
    provider: str = "",
    model: str = "",
) -> dict[str, Any]:
    return {
        "id": new_id("terrun"),
        "document_id": document_id,
        "status": "running",
        "fingerprint": fingerprint,
        "config": asdict(config),
        "coverage": {},
        "provider": provider,
        "model": model,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "cost_usd": 0.0,
        "error": "",
        "created_at": utc_now(),
        "completed_at": "",
    }
