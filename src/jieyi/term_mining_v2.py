from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from jieyi.domain.models import Segment, SegmentKind, new_id

_WORD_RE = re.compile(r"[^\W\d_]+(?:[-’'][^\W\d_]+)*", re.UNICODE)
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_SPACE_RE = re.compile(r"\s+")
_CLEAN_JOIN_RE = re.compile(r"^\s+$")
_ACRONYM_RE = re.compile(r"^(?:[A-ZÀ-ÖØ-Þ]\.?){2,8}$")
_ROMAN_RE = re.compile(r"^[IVXLCDM]+$", re.IGNORECASE)
_QUOTED_PATTERNS = (
    re.compile(r"«([^«»\n]{2,100})»"),
    re.compile(r"“([^“”\n]{2,100})”"),
    re.compile(r"\"([^\"\n]{2,100})\""),
    re.compile(r"‘([^‘’\n]{2,100})’"),
)
_FRENCH_CLITIC_RE = re.compile(r"^(?:[cdjlmnst]|qu|jusqu|lorsqu|puisqu)[’'](.+)$", re.IGNORECASE)
_METADATA_RE = re.compile(
    r"\b(?:isbn|copyright|all rights reserved|printed in|imprim[ée] en|d[ée]p[oô]t l[ée]gal|"
    r"format epub|r[ée]alisation|impression|table of contents|table des mati[èe]res|contents|du m[êe]me auteur|note de l[’']?[ée]diteur)\b|©",
    re.IGNORECASE,
)

_STOPWORDS: dict[str, frozenset[str]] = {
    "en": frozenset(
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
    ),
    "fr": frozenset(
        [
            "a",
            "afin",
            "ai",
            "ainsi",
            "alors",
            "après",
            "au",
            "aucun",
            "aussi",
            "autre",
            "aux",
            "avec",
            "avoir",
            "bon",
            "ce",
            "ceci",
            "cela",
            "ces",
            "cet",
            "cette",
            "ceux",
            "chaque",
            "chez",
            "comme",
            "comment",
            "dans",
            "de",
            "dedans",
            "dehors",
            "depuis",
            "des",
            "deux",
            "devant",
            "doit",
            "donc",
            "dont",
            "du",
            "elle",
            "elles",
            "en",
            "encore",
            "entre",
            "est",
            "est-ce",
            "et",
            "eu",
            "eux",
            "fait",
            "faire",
            "fois",
            "font",
            "hors",
            "ici",
            "il",
            "ils",
            "je",
            "juste",
            "la",
            "le",
            "les",
            "leur",
            "leurs",
            "lui",
            "ma",
            "mais",
            "me",
            "même",
            "mes",
            "moi",
            "mon",
            "ne",
            "ni",
            "nos",
            "notre",
            "nous",
            "on",
            "ont",
            "ou",
            "où",
            "par",
            "parce",
            "pas",
            "pendant",
            "peu",
            "peut",
            "plus",
            "pour",
            "pourquoi",
            "quand",
            "que",
            "quel",
            "quelle",
            "quelles",
            "quels",
            "qui",
            "sa",
            "sans",
            "se",
            "sera",
            "ses",
            "si",
            "soi",
            "soit",
            "son",
            "sont",
            "sous",
            "sur",
            "ta",
            "tandis",
            "te",
            "tel",
            "telle",
            "tes",
            "toi",
            "ton",
            "tous",
            "tout",
            "toute",
            "toutes",
            "très",
            "tu",
            "un",
            "une",
            "vos",
            "votre",
            "vous",
            "y",
            "à",
            "ça",
            "était",
            "été",
            "être",
            "passe",
            "passent",
            "passé",
            "passer",
            "va",
            "vont",
            "vais",
            "allait",
            "allant",
            "suis",
            "sommes",
            "met",
            "mettent",
            "prendre",
            "prend",
            "pleut",
            "allument",
            "sonner",
            "d",
            "l",
            "j",
            "m",
            "n",
            "s",
            "t",
            "qu",
            "c",
            "chapitre",
            "figure",
            "table",
            "section",
            "partie",
        ]
    ),
    "de": frozenset(
        [
            "aber",
            "als",
            "am",
            "an",
            "auch",
            "auf",
            "aus",
            "bei",
            "bin",
            "bis",
            "bist",
            "da",
            "dadurch",
            "daher",
            "darum",
            "das",
            "dass",
            "dein",
            "deine",
            "dem",
            "den",
            "der",
            "des",
            "die",
            "dies",
            "diese",
            "doch",
            "dort",
            "du",
            "durch",
            "ein",
            "eine",
            "einem",
            "einen",
            "einer",
            "eines",
            "er",
            "es",
            "für",
            "gegen",
            "gewesen",
            "hat",
            "haben",
            "hier",
            "hin",
            "hinter",
            "ich",
            "im",
            "in",
            "ist",
            "ja",
            "jede",
            "jeder",
            "jedes",
            "kann",
            "kein",
            "mit",
            "muss",
            "nach",
            "nicht",
            "nichts",
            "noch",
            "nun",
            "nur",
            "ob",
            "oder",
            "ohne",
            "sehr",
            "sein",
            "seine",
            "selbst",
            "sich",
            "sie",
            "sind",
            "so",
            "über",
            "um",
            "und",
            "uns",
            "unter",
            "vom",
            "von",
            "vor",
            "war",
            "waren",
            "was",
            "weg",
            "weil",
            "weiter",
            "welche",
            "wenn",
            "werde",
            "werden",
            "wie",
            "wieder",
            "wir",
            "wird",
            "wo",
            "zu",
            "zum",
            "zur",
        ]
    ),
    "es": frozenset(
        [
            "a",
            "al",
            "algo",
            "algunas",
            "algunos",
            "ante",
            "antes",
            "como",
            "con",
            "contra",
            "cual",
            "cuando",
            "de",
            "del",
            "desde",
            "donde",
            "dos",
            "el",
            "ella",
            "ellas",
            "ellos",
            "en",
            "entre",
            "era",
            "erais",
            "eran",
            "eras",
            "eres",
            "es",
            "esa",
            "esas",
            "ese",
            "eso",
            "esos",
            "esta",
            "estaba",
            "estado",
            "estas",
            "este",
            "esto",
            "estos",
            "fue",
            "ha",
            "hacia",
            "hasta",
            "hay",
            "la",
            "las",
            "le",
            "les",
            "lo",
            "los",
            "más",
            "me",
            "mi",
            "mis",
            "mucha",
            "muchos",
            "muy",
            "nada",
            "ni",
            "no",
            "nos",
            "o",
            "otra",
            "para",
            "pero",
            "poco",
            "por",
            "porque",
            "que",
            "quien",
            "se",
            "ser",
            "si",
            "sin",
            "sobre",
            "son",
            "su",
            "sus",
            "te",
            "tiene",
            "todo",
            "tu",
            "tus",
            "un",
            "una",
            "uno",
            "y",
            "ya",
        ]
    ),
    "it": frozenset(
        [
            "a",
            "ad",
            "al",
            "alla",
            "allo",
            "anche",
            "che",
            "chi",
            "con",
            "contro",
            "cui",
            "da",
            "dal",
            "dalla",
            "delle",
            "di",
            "dove",
            "due",
            "e",
            "ed",
            "era",
            "essere",
            "fra",
            "gli",
            "ha",
            "hai",
            "hanno",
            "il",
            "in",
            "io",
            "la",
            "le",
            "lo",
            "loro",
            "ma",
            "mi",
            "molto",
            "nei",
            "nel",
            "nella",
            "no",
            "noi",
            "non",
            "o",
            "ogni",
            "per",
            "però",
            "più",
            "quale",
            "quando",
            "questa",
            "questo",
            "se",
            "senza",
            "si",
            "sono",
            "su",
            "sul",
            "tra",
            "tu",
            "tua",
            "un",
            "una",
            "uno",
            "vi",
            "voi",
        ]
    ),
    "pt": frozenset(
        [
            "a",
            "ao",
            "aos",
            "as",
            "com",
            "como",
            "da",
            "das",
            "de",
            "do",
            "dos",
            "e",
            "ela",
            "ele",
            "eles",
            "em",
            "entre",
            "era",
            "essa",
            "esse",
            "esta",
            "este",
            "eu",
            "foi",
            "há",
            "isso",
            "isto",
            "já",
            "mais",
            "mas",
            "me",
            "mesmo",
            "meu",
            "minha",
            "muito",
            "na",
            "nas",
            "nem",
            "no",
            "nos",
            "nós",
            "o",
            "os",
            "ou",
            "para",
            "pela",
            "pelo",
            "por",
            "porque",
            "qual",
            "quando",
            "que",
            "quem",
            "se",
            "sem",
            "ser",
            "seu",
            "sua",
            "são",
            "também",
            "tem",
            "um",
            "uma",
            "você",
        ]
    ),
}

_CONNECTORS: dict[str, frozenset[str]] = {
    "en": frozenset({"of", "the", "and", "for"}),
    "fr": frozenset({"de", "du", "des", "la", "le", "les", "et"}),
    "de": frozenset({"von", "der", "des", "und", "zu"}),
    "es": frozenset({"de", "del", "la", "las", "los", "y"}),
    "it": frozenset({"di", "del", "della", "dei", "e"}),
    "pt": frozenset({"de", "da", "do", "das", "dos", "e"}),
}

_DEFINITION_RE: dict[str, re.Pattern[str]] = {
    "en": re.compile(
        r"\b(?:is|are)\s+(?:defined as|called|known as)|\b(?:means?|refers? to)\b", re.IGNORECASE
    ),
    "fr": re.compile(
        r"\b(?:est|sont)\s+(?:défini(?:e|s|es)? comme|appelé(?:e|s|es)?)|\b(?:désigne|signifie|on appelle)\b",
        re.IGNORECASE,
    ),
}


@dataclass(frozen=True, slots=True)
class _Token:
    text: str
    normalized: str
    start: int
    end: int


@dataclass(slots=True)
class _Accumulator:
    key: str
    token_tuple: tuple[str, ...]
    forms: Counter[str] = field(default_factory=Counter)
    frequency: int = 0
    segment_ids: set[str] = field(default_factory=set)
    methods: set[str] = field(default_factory=set)
    occurrences: list[tuple[Segment, int, int, str, str]] = field(default_factory=list)
    left_contexts: Counter[str] = field(default_factory=Counter)
    right_contexts: Counter[str] = field(default_factory=Counter)
    translated_segments: set[str] = field(default_factory=set)
    rng_state: int = 0


def _normalize(value: str) -> str:
    return _SPACE_RE.sub(" ", unicodedata.normalize("NFKC", value)).casefold().strip()


def _language_code(source_lang: str, sample: str) -> str:
    requested = source_lang.casefold().replace("_", "-").split("-", 1)[0]
    words = [_normalize(match.group()) for match in _WORD_RE.finditer(sample[:20_000])]
    scores = {
        language: sum(word in stopwords for word in words)
        for language, stopwords in _STOPWORDS.items()
    }
    if not words or not scores:
        return requested if requested in _STOPWORDS else "en"
    detected = max(scores, key=scores.get)
    requested_score = scores.get(requested, 0)
    detected_score = scores[detected]
    if requested in _STOPWORDS and (
        detected == requested
        or detected_score < 20
        or detected_score < max(1, requested_score) * 2.5
    ):
        return requested
    return detected


def _tokenize(text: str, language: str) -> list[_Token]:
    result: list[_Token] = []
    for match in _WORD_RE.finditer(text):
        value = match.group()
        start = match.start()
        if language == "fr":
            clitic = _FRENCH_CLITIC_RE.match(value)
            if clitic:
                suffix = clitic.group(1)
                start = match.end() - len(suffix)
                value = suffix
        result.append(_Token(value, _normalize(value), start, match.end()))
    return result


def _is_capitalized(token: _Token) -> bool:
    first = token.text[:1]
    return bool(first and first.isupper()) or bool(_ACRONYM_RE.fullmatch(token.text))


def _is_content(token: _Token, stopwords: frozenset[str]) -> bool:
    return (
        len(token.normalized) >= 2
        and token.normalized not in stopwords
        and not token.normalized.isdigit()
        and not _ROMAN_RE.fullmatch(token.normalized)
    )


def _chunks(text: str, tokens: list[_Token]) -> list[list[_Token]]:
    chunks: list[list[_Token]] = []
    current: list[_Token] = []
    for token in tokens:
        if current and not _CLEAN_JOIN_RE.fullmatch(text[current[-1].end : token.start]):
            chunks.append(current)
            current = []
        current.append(token)
    if current:
        chunks.append(current)
    return chunks


def _is_metadata(segment: Segment) -> bool:
    text = segment.source_text
    context = f"{segment.heading_path}\n{text}"
    hits = len(_METADATA_RE.findall(context))
    return hits >= 2 or (hits == 1 and len(text) < 240)


def _visible_translation(segment: Segment) -> str:
    return (
        segment.accepted_translation
        or segment.reviewed_translation
        or segment.edited_translation
        or segment.machine_translation
        or ""
    )


def _record(
    accumulators: dict[str, _Accumulator],
    segment: Segment,
    start: int,
    end: int,
    tokens: tuple[str, ...],
    method: str,
    seen: set[tuple[str, int, int]],
) -> None:
    value = segment.source_text[start:end].strip()
    key = _normalize(value)
    if not key or len(value) > 120:
        return
    occurrence_key = (key, start, end)
    accumulator = accumulators.setdefault(key, _Accumulator(key=key, token_tuple=tokens))
    accumulator.methods.add(method)
    if occurrence_key in seen:
        return
    seen.add(occurrence_key)
    accumulator.forms[value] += 1
    accumulator.frequency += 1
    accumulator.segment_ids.add(segment.id)
    reason = ",".join(sorted(accumulator.methods | {method}))
    occurrence = (segment, start, end, value, reason)
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
        context = _normalize(before[-1])
        if context in accumulator.left_contexts or len(accumulator.left_contexts) < 32:
            accumulator.left_contexts[context] += 1
    if after:
        context = _normalize(after.group())
        if context in accumulator.right_contexts or len(accumulator.right_contexts) < 32:
            accumulator.right_contexts[context] += 1
    if _visible_translation(segment) and len(accumulator.translated_segments) < 3:
        accumulator.translated_segments.add(segment.id)


def _collect_latin(segments: list[Segment], language: str) -> tuple[dict[str, _Accumulator], int]:
    accumulators: dict[str, _Accumulator] = {}
    stopwords = _STOPWORDS.get(language, _STOPWORDS["en"])
    connectors = _CONNECTORS.get(language, frozenset())
    definition_re = _DEFINITION_RE.get(language, _DEFINITION_RE["en"])
    metadata_segments = 0
    for segment in segments:
        if _is_metadata(segment):
            metadata_segments += 1
            continue
        text = segment.source_text
        tokens = _tokenize(text, language)
        seen: set[tuple[str, int, int]] = set()
        chunks = _chunks(text, tokens)
        for chunk in chunks:
            for token in chunk:
                if _is_content(token, stopwords):
                    method = "acronym" if _ACRONYM_RE.fullmatch(token.text) else "unigram"
                    if "-" in token.text or "–" in token.text:
                        method = "hyphenated"
                    _record(
                        accumulators,
                        segment,
                        token.start,
                        token.end,
                        (token.normalized,),
                        method,
                        seen,
                    )
            for index in range(len(chunk)):
                for size in (2, 3):
                    window = chunk[index : index + size]
                    if len(window) != size:
                        continue
                    if not _is_content(window[0], stopwords) or not _is_content(
                        window[-1], stopwords
                    ):
                        continue
                    if sum(token.normalized in stopwords for token in window[1:-1]) > 1:
                        continue
                    _record(
                        accumulators,
                        segment,
                        window[0].start,
                        window[-1].end,
                        tuple(token.normalized for token in window),
                        "phrase",
                        seen,
                    )

            index = 0
            while index < len(chunk):
                if not _is_capitalized(chunk[index]):
                    index += 1
                    continue
                end = index + 1
                last_capital = end
                while end < len(chunk):
                    token = chunk[end]
                    if _is_capitalized(token):
                        last_capital = end + 1
                        end += 1
                        continue
                    if (
                        token.normalized in connectors
                        and end + 1 < len(chunk)
                        and _is_capitalized(chunk[end + 1])
                    ):
                        end += 1
                        continue
                    break
                window = chunk[index:last_capital]
                if len(window) > 1:
                    _record(
                        accumulators,
                        segment,
                        window[0].start,
                        window[-1].end,
                        tuple(token.normalized for token in window),
                        "proper_name",
                        seen,
                    )
                index = max(index + 1, last_capital)

        for pattern in _QUOTED_PATTERNS:
            for match in pattern.finditer(text):
                value = match.group(1).strip()
                inner_start = text.find(value, match.start(), match.end())
                quoted_tokens = _tokenize(value, language)
                content = [token for token in quoted_tokens if _is_content(token, stopwords)]
                if 1 <= len(content) <= 6:
                    _record(
                        accumulators,
                        segment,
                        inner_start,
                        inner_start + len(value),
                        tuple(token.normalized for token in quoted_tokens),
                        "quoted",
                        seen,
                    )

        if segment.kind is SegmentKind.HEADING:
            heading_tokens = _tokenize(text, language)
            content = [token for token in heading_tokens if _is_content(token, stopwords)]
            if 1 <= len(content) <= 10:
                _record(
                    accumulators,
                    segment,
                    heading_tokens[0].start,
                    heading_tokens[-1].end,
                    tuple(token.normalized for token in heading_tokens),
                    "heading",
                    seen,
                )

        for cue in definition_re.finditer(text):
            before = [token for token in tokens if token.end <= cue.start()]
            if not before:
                continue
            tail = before[-3:]
            while tail and tail[0].normalized in stopwords:
                tail.pop(0)
            while tail and tail[-1].normalized in stopwords:
                tail.pop()
            if tail:
                _record(
                    accumulators,
                    segment,
                    tail[0].start,
                    tail[-1].end,
                    tuple(token.normalized for token in tail),
                    "definition_cue",
                    seen,
                )
    for accumulator in accumulators.values():
        if "acronym" in accumulator.methods and (
            _ROMAN_RE.fullmatch(accumulator.key)
            or any(form != form.upper() for form in accumulator.forms)
        ):
            accumulator.methods.discard("acronym")
    return accumulators, metadata_segments


def _collect_cjk(segments: list[Segment]) -> tuple[dict[str, _Accumulator], int]:
    accumulators: dict[str, _Accumulator] = {}
    metadata_segments = 0
    for segment in segments:
        if _is_metadata(segment):
            metadata_segments += 1
            continue
        text = segment.source_text
        seen: set[tuple[str, int, int]] = set()
        for match in re.finditer(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]{2,}", text):
            run = match.group()
            for size in range(2, min(6, len(run)) + 1):
                for offset in range(len(run) - size + 1):
                    start = match.start() + offset
                    end = start + size
                    value = text[start:end]
                    _record(
                        accumulators,
                        segment,
                        start,
                        end,
                        (value,),
                        "cjk_ngram",
                        seen,
                    )
        for pattern in _QUOTED_PATTERNS:
            for match in pattern.finditer(text):
                value = match.group(1).strip()
                if 2 <= len(value) <= 20:
                    start = text.find(value, match.start(), match.end())
                    _record(
                        accumulators,
                        segment,
                        start,
                        start + len(value),
                        (value,),
                        "quoted",
                        seen,
                    )
    return accumulators, metadata_segments


def _nested_statistics(
    accumulators: dict[str, _Accumulator],
) -> tuple[dict[str, float], dict[str, list[tuple[str, int]]]]:
    parents: dict[str, list[tuple[str, int]]] = defaultdict(list)
    by_tokens = {
        accumulator.token_tuple: accumulator
        for accumulator in accumulators.values()
        if accumulator.token_tuple
    }
    for tokens, accumulator in by_tokens.items():
        if len(tokens) <= 1:
            continue
        for size in range(1, len(tokens)):
            for start in range(len(tokens) - size + 1):
                child = tokens[start : start + size]
                if child in by_tokens:
                    parents[by_tokens[child].key].append((accumulator.key, accumulator.frequency))
    scores: dict[str, float] = {}
    for key, accumulator in accumulators.items():
        nested = parents.get(key, ())
        adjusted = accumulator.frequency
        if nested:
            adjusted -= sum(item[1] for item in nested) / len(nested)
        scores[key] = max(
            0.0,
            math.log2(max(2, len(accumulator.token_tuple) + 1)) * adjusted,
        )
    return scores, parents


def _candidate_type(accumulator: _Accumulator) -> str:
    if "proper_name" in accumulator.methods and len(accumulator.token_tuple) > 1:
        return "named_entity"
    if {"hyphenated", "acronym"} & accumulator.methods:
        return "lexical_risk"
    return "concept"


def _association(accumulator: _Accumulator, accumulators: dict[str, _Accumulator]) -> float:
    if len(accumulator.token_tuple) <= 1:
        return 1.0
    component_frequencies = [
        item.frequency
        for token in accumulator.token_tuple
        if (item := accumulators.get(token)) is not None
    ]
    if not component_frequencies:
        return 0.0
    geometric_mean = math.exp(
        sum(math.log(max(1, frequency)) for frequency in component_frequencies)
        / len(component_frequencies)
    )
    return min(1.0, accumulator.frequency / max(1.0, geometric_mean))


def _eligible(accumulator: _Accumulator, candidate_type: str, association: float) -> bool:
    methods = accumulator.methods
    strong_explicit = bool({"definition_cue", "quoted", "heading"} & methods)
    token_count = len(accumulator.token_tuple)
    if candidate_type == "named_entity":
        return accumulator.frequency >= 2 or strong_explicit
    if strong_explicit:
        return True
    if token_count > 1:
        return accumulator.frequency >= 2 and association >= 0.12
    if {"hyphenated", "acronym"} & methods:
        return accumulator.frequency >= 2
    return accumulator.frequency >= 3


def _boundary_confidence(accumulator: _Accumulator) -> float:
    methods = accumulator.methods
    if {"quoted", "heading", "definition_cue"} & methods:
        return 1.0
    if "proper_name" in methods:
        return 0.95
    if "phrase" in methods and accumulator.frequency >= 2:
        return 0.85
    if {"hyphenated", "acronym"} & methods:
        return 0.8
    return 0.65


def _evidence(accumulator: _Accumulator, maximum: int) -> list[dict[str, Any]]:
    occurrences = accumulator.occurrences
    if len(occurrences) <= maximum:
        selected = occurrences
    else:
        indexes = {
            min(len(occurrences) - 1, int((index + 0.5) * len(occurrences) / maximum))
            for index in range(maximum)
        }
        selected = [occurrences[index] for index in sorted(indexes)]
    result = []
    for segment, start, end, form, reason in selected:
        result.append(
            {
                "id": new_id("evidence"),
                "segment_id": segment.id,
                "ordinal": segment.ordinal,
                "source_form": form,
                "quote": segment.source_text[
                    max(0, start - 100) : min(len(segment.source_text), end + 100)
                ],
                "start_offset": start,
                "end_offset": end,
                "heading_path": segment.heading_path,
                "reason": reason,
            }
        )
    return result


def mine_candidates_v2(
    segments: list[Segment],
    config: Any,
    *,
    source_lang: str = "",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sample = "\n".join(segment.source_text for segment in segments[:200])
    requested = source_lang.casefold().replace("_", "-").split("-", 1)[0]
    cjk_characters = len(_CJK_RE.findall(sample))
    visible_characters = sum(not character.isspace() for character in sample)
    use_cjk = requested in {"zh", "ja", "ko"} or (
        cjk_characters >= 20 and cjk_characters / max(1, visible_characters) >= 0.15
    )
    if use_cjk:
        accumulators, metadata_segments = _collect_cjk(segments)
        language = "cjk"
    else:
        language = _language_code(source_lang, sample)
        accumulators, metadata_segments = _collect_latin(segments, language)
    c_values, parents = _nested_statistics(accumulators)
    max_c = max(c_values.values(), default=1.0) or 1.0
    max_frequency = max((item.frequency for item in accumulators.values()), default=1)
    last_ordinal = max((segment.ordinal for segment in segments), default=0)
    scored: list[tuple[float, _Accumulator, str, float, dict[str, float]]] = []
    rejected_by_gate = 0
    for accumulator in accumulators.values():
        candidate_type = _candidate_type(accumulator)
        association = _association(accumulator, accumulators)
        if not _eligible(accumulator, candidate_type, association):
            rejected_by_gate += 1
            continue
        explicit = max(
            1.0 if "definition_cue" in accumulator.methods else 0.0,
            0.8 if "heading" in accumulator.methods else 0.0,
            0.65 if "quoted" in accumulator.methods else 0.0,
        )
        bins = {
            min(7, int(item[0].ordinal * 8 / max(1, last_ordinal + 1)))
            for item in accumulator.occurrences
        }
        dispersion = len(bins) / min(8, max(1, len(accumulator.segment_ids)))
        c_value = c_values[accumulator.key] / max_c
        frequency = math.log2(accumulator.frequency + 1) / math.log2(max_frequency + 1)
        boundary = _boundary_confidence(accumulator)
        multiword = min(1.0, max(0, len(accumulator.token_tuple) - 1) / 2)
        translation = min(1.0, len(accumulator.translated_segments) / 3)
        score = (
            0.24 * c_value
            + 0.17 * frequency
            + 0.13 * dispersion
            + 0.16 * explicit
            + 0.10 * boundary
            + 0.12 * association
            + 0.04 * multiword
            + 0.04 * translation
        )
        if candidate_type == "named_entity":
            score -= 0.04
        components = {
            "c_value": round(c_value, 6),
            "frequency": round(frequency, 6),
            "dispersion": round(dispersion, 6),
            "explicit": round(explicit, 6),
            "boundary": round(boundary, 6),
            "association": round(association, 6),
            "multiword": round(multiword, 6),
            "translation_risk": round(translation, 6),
        }
        scored.append(
            (max(0.0, min(1.0, score)), accumulator, candidate_type, boundary, components)
        )

    scored.sort(key=lambda item: (-item[0], -item[1].frequency, item[1].key))
    corpus_budget = max(12, min(40, round(math.sqrt(max(1, len(segments))) * 1.2)))
    queue_ceiling = min(config.max_candidates, corpus_budget)
    plain_budget = min(5, max(3, round(math.log2(max(2, len(segments))) / 2)))
    named_budget = min(10, max(4, round(math.sqrt(max(1, len(segments))) / 3)))
    plain_used = 0
    named_used = 0
    selected: list[tuple[float, _Accumulator, str, float, dict[str, float]]] = []
    for item in scored:
        score, accumulator, candidate_type, _, _ = item
        if score < config.min_score:
            continue
        is_plain = accumulator.methods == {"unigram"}
        if is_plain:
            if plain_used >= plain_budget:
                continue
            plain_used += 1
        if candidate_type == "named_entity":
            if named_used >= named_budget:
                continue
            named_used += 1
        nested = parents.get(accumulator.key, ())
        if nested and any(
            parent_frequency == accumulator.frequency
            and len(accumulators[parent_key].token_tuple) <= 3
            and not ({"definition_cue", "quoted", "heading"} & accumulator.methods)
            for parent_key, parent_frequency in nested
        ):
            continue
        selected.append(item)
        if len(selected) >= queue_ceiling:
            break

    candidates: list[dict[str, Any]] = []
    type_counts: Counter[str] = Counter()
    for rank, (score, accumulator, candidate_type, boundary, components) in enumerate(selected, 1):
        evidence = _evidence(accumulator, config.max_evidence_per_candidate)
        type_counts[candidate_type] += 1
        candidates.append(
            {
                "id": new_id("lexeme"),
                "lexeme_key": accumulator.key,
                "canonical_form": accumulator.forms.most_common(1)[0][0],
                "forms": [item[0] for item in accumulator.forms.most_common()],
                "frequency": accumulator.frequency,
                "segment_frequency": len(accumulator.segment_ids),
                "risk_score": round(score, 6),
                "rank": rank,
                "candidate_type": candidate_type,
                "boundary_confidence": round(boundary, 6),
                "score_components": components,
                "extraction_methods": sorted(accumulator.methods | {"c_value_v2"}),
                "evidence": evidence,
                "senses": [
                    {
                        "id": new_id("sense"),
                        "sense_key": "unclassified",
                        "sense": "",
                        "concept_definition": "",
                        "proposed_target": "",
                        "rationale": "由语言边界与全文统计召回；须依据所列原文证据人工判断。",
                        "disambiguation": "",
                        "confidence": 0.0,
                        "ai_recommended": None,
                        "evidence_ids": [item["id"] for item in evidence],
                        "proposer": "deterministic-v2",
                        "status": "pending",
                    }
                ],
            }
        )
    coverage = {
        "segments_total": len(segments),
        "segments_scanned": len(segments),
        "characters_scanned": sum(len(item.source_text) for item in segments),
        "metadata_segments_excluded": metadata_segments,
        "raw_lexemes": len(accumulators),
        "linguistic_gate_rejections": rejected_by_gate,
        "ranked_candidates": len(scored),
        "retained_candidates": len(candidates),
        "candidate_types": dict(type_counts),
        "review_queue_ceiling": queue_ceiling,
        "plain_unigram_budget": plain_budget,
        "named_entity_budget": named_budget,
        "truncated": len(candidates) >= queue_ceiling and len(scored) > len(candidates),
        "language_profile": language,
        "algorithm": "linguistic-boundary+c-value+risk-ensemble-v2",
    }
    return candidates, coverage
