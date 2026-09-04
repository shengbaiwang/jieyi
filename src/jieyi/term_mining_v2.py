from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import pairwise
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
            "same",
            "great",
            "first",
            "second",
            "third",
            "earliest",
            "latest",
            "new",
            "old",
            "other",
            "another",
            "many",
            "much",
            "less",
            "few",
            "several",
            "however",
            "also",
            "just",
            "even",
            "still",
            "said",
            "says",
            "made",
            "make",
            "makes",
            "got",
            "get",
            "gets",
            "et",
            "al",
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
            "avait",
            "avais",
            "avaient",
            "avions",
            "aviez",
            "avons",
            "avez",
            "étais",
            "étaient",
            "étions",
            "étiez",
            "serai",
            "serait",
            "seront",
            "bien",
            "autres",
            "quelque",
            "quelques",
            "nouveau",
            "nouveaux",
            "nouvelle",
            "nouvelles",
            "premier",
            "premiers",
            "première",
            "premières",
            "rien",
            "jamais",
            "toujours",
            "déjà",
            "beaucoup",
            "moins",
            "pouvoir",
            "peux",
            "peuvent",
            "pouvait",
            "pouvaient",
            "pourtant",
            "devoir",
            "doivent",
            "devait",
            "vouloir",
            "veux",
            "veut",
            "voulait",
            "dit",
            "disait",
            "disent",
            "grand",
            "grande",
            "grands",
            "grandes",
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
            "diesem",
            "diesen",
            "dieser",
            "dieses",
            "andere",
            "anderem",
            "anderen",
            "anderer",
            "anderes",
            "erste",
            "erstem",
            "ersten",
            "erster",
            "erstes",
            "deren",
            "dessen",
            "sondern",
            "wurde",
            "wurden",
            "worden",
            "neue",
            "neuen",
            "neuer",
            "neues",
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
            "ihr",
            "ihre",
            "ihrem",
            "ihren",
            "ihrer",
            "ihres",
            "man",
            "mein",
            "meine",
            "meinem",
            "meinen",
            "meiner",
            "meines",
            "unser",
            "unsere",
            "unserem",
            "unseren",
            "unserer",
            "unseres",
            "euer",
            "eure",
            "eurem",
            "euren",
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
            "seit",
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

# Ordinary discourse nouns are weak glossary candidates unless the book explicitly
# defines them. This is deliberately narrower than the language stopword lists.
_GENERIC_GLOSSARY_NOUNS: dict[str, frozenset[str]] = {
    "en": frozenset(
        {
            "answer",
            "case",
            "chapter",
            "example",
            "fact",
            "form",
            "kind",
            "matter",
            "part",
            "person",
            "place",
            "point",
            "possibility",
            "question",
            "reason",
            "result",
            "section",
            "situation",
            "thing",
            "time",
            "way",
            "word",
            "year",
        }
    ),
    "de": frozenset(
        {
            "angelegenheit",
            "anteil",
            "anspruch",
            "art",
            "aufgabe",
            "aufgaben",
            "auge",
            "augenblick",
            "bedeutung",
            "begriff",
            "beispiel",
            "bedingung",
            "betrachtung",
            "darstellung",
            "ding",
            "entwicklung",
            "fall",
            "form",
            "formulierung",
            "frage",
            "fragestellung",
            "funktion",
            "gegenstand",
            "gelegenheit",
            "geschichte",
            "gestalt",
            "gewicht",
            "grund",
            "hand",
            "haltung",
            "interesse",
            "jahr",
            "jahrhundert",
            "leistung",
            "ma\u00dfstab",
            "mensch",
            "mittel",
            "m\u00f6glichkeit",
            "person",
            "produktion",
            "sachverhalt",
            "seite",
            "sinn",
            "situation",
            "stelle",
            "umstand",
            "umst\u00e4nde",
            "untersuchung",
            "ver\u00e4nderung",
            "vergr\u00f6\u00dferung",
            "verfahren",
            "verh\u00e4ltnis",
            "versuch",
            "verschiebung",
            "vorgang",
            "vorstellung",
            "wert",
            "wissenschaft",
            "wort",
            "zeit",
            "zeitung",
            "zusammenhang",
        }
    ),
    "fr": frozenset(
        {
            "ann\u00e9e",
            "cas",
            "chose",
            "exemple",
            "fait",
            "forme",
            "partie",
            "possibilit\u00e9",
            "question",
            "raison",
            "situation",
            "temps",
        }
    ),
    "es": frozenset({"a\u00f1o", "caso", "cosa", "ejemplo", "forma", "parte", "pregunta", "raz\u00f3n", "tiempo"}),
    "it": frozenset({"anno", "caso", "cosa", "esempio", "forma", "parte", "ragione", "tempo"}),
    "pt": frozenset({"ano", "caso", "coisa", "exemplo", "forma", "parte", "raz\u00e3o", "tempo"}),
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
    "de": re.compile(
        r"\b(?:ist|sind|wird|werden)\s+(?:als\s+)?(?:bezeichnet|definiert|genannt)"
        r"|\b(?:bedeutet|bezeichnet|nennt\s+man|versteht\s+man\s+unter)\b",
        re.IGNORECASE,
    ),
    "es": re.compile(
        r"\b(?:es|son)\s+(?:definid[oa]s?\s+como|llamad[oa]s?)"
        r"|\b(?:significa|designa|se\s+denomina)\b",
        re.IGNORECASE,
    ),
    "it": re.compile(
        r"\b(?:è|sono)\s+(?:definit[oaie]+\s+come|chiamat[oaie]+)"
        r"|\b(?:significa|designa|si\s+chiama)\b",
        re.IGNORECASE,
    ),
    "pt": re.compile(
        r"\b(?:é|são)\s+(?:definid[oa]s?\s+como|chamad[oa]s?)"
        r"|\b(?:significa|designa|denomina-se)\b",
        re.IGNORECASE,
    ),
}

_COORDINATORS: dict[str, frozenset[str]] = {
    "en": frozenset({"and", "or", "versus"}),
    "fr": frozenset({"et", "ou"}),
    "de": frozenset({"und", "oder"}),
    "es": frozenset({"y", "o"}),
    "it": frozenset({"e", "o"}),
    "pt": frozenset({"e", "ou"}),
}
_DETERMINERS: dict[str, frozenset[str]] = {
    "en": frozenset(
        {
            "a",
            "an",
            "the",
            "this",
            "that",
            "these",
            "those",
            "some",
            "any",
            "each",
            "every",
            "his",
            "her",
            "its",
            "our",
            "their",
        }
    ),
    "fr": frozenset(
        {
            "le",
            "la",
            "les",
            "un",
            "une",
            "des",
            "du",
            "ce",
            "cet",
            "cette",
            "ces",
            "mon",
            "ma",
            "mes",
            "son",
            "sa",
            "ses",
            "notre",
            "votre",
            "leur",
        }
    ),
    "de": frozenset(
        {
            "der",
            "die",
            "das",
            "den",
            "dem",
            "des",
            "ein",
            "eine",
            "einen",
            "einem",
            "einer",
            "eines",
            "dies",
            "diese",
            "dieser",
            "dieses",
        }
    ),
    "es": frozenset(
        {
            "el",
            "la",
            "los",
            "las",
            "un",
            "una",
            "unos",
            "unas",
            "este",
            "esta",
            "estos",
            "estas",
            "ese",
            "esa",
            "esos",
            "esas",
        }
    ),
    "it": frozenset(
        {
            "il",
            "lo",
            "la",
            "i",
            "gli",
            "le",
            "un",
            "uno",
            "una",
            "questo",
            "questa",
            "questi",
            "queste",
        }
    ),
    "pt": frozenset(
        {
            "o",
            "a",
            "os",
            "as",
            "um",
            "uma",
            "uns",
            "umas",
            "este",
            "esta",
            "estes",
            "estas",
            "esse",
            "essa",
        }
    ),
}
_NOMINAL_SUFFIXES: dict[str, tuple[str, ...]] = {
    "en": (
        "ability",
        "ance",
        "archy",
        "cracy",
        "culture",
        "dom",
        "ence",
        "erty",
        "ery",
        "graphy",
        "hood",
        "ics",
        "ism",
        "ity",
        "logy",
        "ment",
        "ness",
        "ology",
        "ship",
        "sion",
        "tion",
        "ure",
    ),
    "fr": (
        "ance",
        "ence",
        "eur",
        "euse",
        "té",
        "isme",
        "ité",
        "logie",
        "ment",
        "sion",
        "tion",
    ),
    "de": (
        "anz",
        "barkeit",
        "enz",
        "heit",
        "ik",
        "ismus",
        "keit",
        "ment",
        "schaft",
        "tät",
        "tion",
        "ung",
        "ur",
        "wert",
    ),
    "es": (
        "ancia",
        "encia",
        "ismo",
        "ista",
        "idad",
        "logía",
        "miento",
        "sión",
        "ción",
    ),
    "it": (
        "anza",
        "enza",
        "ismo",
        "ista",
        "ità",
        "logia",
        "mento",
        "sione",
        "zione",
    ),
    "pt": (
        "ância",
        "ência",
        "ismo",
        "ista",
        "dade",
        "logia",
        "mento",
        "são",
        "ção",
    ),
}
_TERMINAL_BOUNDARY_RE = re.compile(r"^\s*(?:[.!?…]|$)")


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
    # lower() keeps German ß distinct from ss (for example Maße versus Masse).
    # Case-insensitive matching is a later concern; lexeme identity must not collapse them.
    return _SPACE_RE.sub(" ", unicodedata.normalize("NFKC", value)).lower().strip()


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


def _morphological_stem(value: str, language: str) -> str:
    """Return a conservative inflection key, never an invented display form."""
    word = value.lower()
    if language == "en" and "-" in word and word.endswith("s") and len(word) >= 6:
        return word[:-1]
    if len(word) < 5 or "-" in word or "’" in word or "'" in word:
        return word
    if language == "de":
        for suffix in ("ern", "em", "en", "er", "es", "e", "s", "n"):
            if word.endswith(suffix) and len(word) - len(suffix) >= 4:
                return word[: -len(suffix)]
    elif language == "en":
        if word.endswith("ies") and len(word) >= 6:
            return word[:-3] + "y"
        for suffix in ("ing", "ed", "es", "s"):
            if word.endswith(suffix) and len(word) - len(suffix) >= 4:
                return word[: -len(suffix)]
    elif language == "fr":
        if word.endswith("eaux") and len(word) >= 7:
            return word[:-1]
        if word.endswith("aux") and len(word) >= 6:
            return word[:-3] + "al"
        if word.endswith(("s", "x")) and len(word) - 1 >= 4:
            return word[:-1]
    elif language == "es":
        for suffix in ("es", "s"):
            if word.endswith(suffix) and len(word) - len(suffix) >= 4:
                return word[: -len(suffix)]
    elif language == "it":
        for suffix in ("i", "e"):
            if word.endswith(suffix) and len(word) - 1 >= 4:
                return word[:-1]
    elif language == "pt":
        for suffix, replacement in (("ões", "ão"), ("ães", "ão")):
            if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                return word[: -len(suffix)] + replacement
        for suffix in ("es", "s"):
            if word.endswith(suffix) and len(word) - len(suffix) >= 4:
                return word[: -len(suffix)]
    return word


def _merge_inflectional_variants(
    accumulators: dict[str, _Accumulator], language: str
) -> dict[str, _Accumulator]:
    """Merge conservative surface families and retain every exact form as evidence."""
    grouped: dict[tuple[str, ...], list[_Accumulator]] = defaultdict(list)
    for accumulator in accumulators.values():
        family = tuple(_morphological_stem(token, language) for token in accumulator.token_tuple)
        grouped[family].append(accumulator)
    merged: dict[str, _Accumulator] = {}
    for family, items in grouped.items():
        if len(items) == 1:
            merged[items[0].key] = items[0]
            continue
        representative = min(
            items,
            key=lambda item: (
                item.token_tuple != family,
                sum(len(token) for token in item.token_tuple),
                -item.frequency,
                item.key,
            ),
        )
        combined = _Accumulator(key=representative.key, token_tuple=family)
        combined.methods.add("morphological_family")
        for item in items:
            combined.forms.update(item.forms)
            combined.frequency += item.frequency
            combined.segment_ids.update(item.segment_ids)
            combined.methods.update(item.methods)
            combined.occurrences.extend(item.occurrences)
            combined.left_contexts.update(item.left_contexts)
            combined.right_contexts.update(item.right_contexts)
            combined.translated_segments.update(item.translated_segments)
        combined.occurrences.sort(key=lambda occurrence: (occurrence[0].ordinal, occurrence[1]))
        combined.occurrences = combined.occurrences[:32]
        if len(combined.translated_segments) > 3:
            combined.translated_segments = set(sorted(combined.translated_segments)[:3])
        merged[combined.key] = combined
    return merged


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
            coordinators = _COORDINATORS.get(language, frozenset())
            for index in range(len(chunk) - 2):
                window = chunk[index : index + 3]
                if (
                    window[1].normalized in coordinators
                    and _is_capitalized(window[0])
                    and _is_capitalized(window[2])
                ):
                    _record(
                        accumulators,
                        segment,
                        window[0].start,
                        window[-1].end,
                        tuple(token.normalized for token in window),
                        "coordination_phrase",
                        seen,
                    )
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
                    prefix = text[: token.start].rstrip()
                    at_sentence_boundary = not prefix or prefix[-1] in ".!?…"
                    if language != "de" and _is_capitalized(token) and not at_sentence_boundary:
                        _record(
                            accumulators,
                            segment,
                            token.start,
                            token.end,
                            (token.normalized,),
                            "capitalized_token",
                            seen,
                        )
            for index in range(len(chunk)):
                for size in range(2, 6):
                    window = chunk[index : index + size]
                    if len(window) != size:
                        continue
                    if not _is_content(window[0], stopwords) or not _is_content(
                        window[-1], stopwords
                    ):
                        continue
                    if sum(token.normalized in stopwords for token in window[1:-1]) > max(
                        1, (size - 1) // 2
                    ):
                        continue
                    methods = {"phrase"}
                    if any(token.normalized in connectors for token in window[1:-1]):
                        methods.add("connector_phrase")
                    if language == "de" and _is_capitalized(window[-1]):
                        methods.add("noun_phrase")
                    candidate_tokens = tuple(token.normalized for token in window)
                    _record(
                        accumulators,
                        segment,
                        window[0].start,
                        window[-1].end,
                        candidate_tokens,
                        "phrase",
                        seen,
                    )
                    for method in methods - {"phrase"}:
                        _record(
                            accumulators,
                            segment,
                            window[0].start,
                            window[-1].end,
                            candidate_tokens,
                            method,
                            seen,
                        )

            index = 0
            while index < len(chunk):
                if not _is_capitalized(chunk[index]) or not _is_content(chunk[index], stopwords):
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
                    spans = [window]
                    capital_indexes = [
                        position for position, token in enumerate(window) if _is_capitalized(token)
                    ]
                    for left, right in pairwise(capital_indexes):
                        subwindow = window[left : right + 1]
                        if len(subwindow) > 1 and all(
                            token.normalized in connectors for token in subwindow[1:-1]
                        ):
                            spans.append(subwindow)
                    for candidate_window in spans:
                        candidate_tokens = tuple(token.normalized for token in candidate_window)
                        _record(
                            accumulators,
                            segment,
                            candidate_window[0].start,
                            candidate_window[-1].end,
                            candidate_tokens,
                            "proper_name",
                            seen,
                        )
                        if any(token.normalized in connectors for token in candidate_window[1:-1]):
                            _record(
                                accumulators,
                                segment,
                                candidate_window[0].start,
                                candidate_window[-1].end,
                                candidate_tokens,
                                "connector_phrase",
                                seen,
                            )
                        if _TERMINAL_BOUNDARY_RE.match(text[candidate_window[-1].end :]):
                            _record(
                                accumulators,
                                segment,
                                candidate_window[0].start,
                                candidate_window[-1].end,
                                candidate_tokens,
                                "rhetorical_boundary",
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


def _candidate_type(accumulator: _Accumulator, language: str) -> str:
    if language == "de" and "proper_name" in accumulator.methods:
        surface_tokens = _WORD_RE.findall(accumulator.forms.most_common(1)[0][0])
        if (
            len(surface_tokens) > 1
            and all(token[:1].isupper() for token in surface_tokens)
        ):
            return "named_entity"
    if language != "de" and (
        {"proper_name", "capitalized_token"} & accumulator.methods
        and _capitalized_surface_ratio(accumulator) >= 0.65
    ):
        return "named_entity"
    if {"hyphenated", "acronym"} & accumulator.methods:
        return "lexical_risk"
    return "concept"


def _association(
    accumulator: _Accumulator,
    singletons: dict[str, _Accumulator],
) -> float:
    if len(accumulator.token_tuple) <= 1:
        return 0.35
    component_frequencies = [
        item.frequency
        for token in accumulator.token_tuple
        if (item := singletons.get(token)) is not None
    ]
    if not component_frequencies:
        return 0.0
    geometric_mean = math.exp(
        sum(math.log(max(1, frequency)) for frequency in component_frequencies)
        / len(component_frequencies)
    )
    return min(1.0, accumulator.frequency / max(1.0, geometric_mean))


def _lexical_specificity(accumulator: _Accumulator) -> float:
    lengths = [len(token.replace("-", "")) for token in accumulator.token_tuple]
    if not lengths:
        return 0.0
    mean_length = sum(lengths) / len(lengths)
    return min(1.0, max(0.0, (mean_length - 3.0) / 9.0))


def _context_diversity(accumulator: _Accumulator) -> float:
    counts = accumulator.left_contexts + accumulator.right_contexts
    total = sum(counts.values())
    if total <= 1:
        return 0.0
    unique = len(counts)
    opportunities = min(16, max(2, accumulator.frequency * 2))
    return min(1.0, unique / opportunities)


def _capitalized_surface_ratio(accumulator: _Accumulator) -> float:
    total = sum(accumulator.forms.values())
    if not total:
        return 0.0
    capitalized = sum(
        count
        for form, count in accumulator.forms.items()
        if any(token[:1].isupper() for token in _WORD_RE.findall(form))
    )
    return capitalized / total


def _orthographic_termhood(accumulator: _Accumulator, language: str) -> float:
    return _capitalized_surface_ratio(accumulator) if language == "de" else 0.0


def _derivational_termhood(accumulator: _Accumulator, language: str) -> float:
    if not accumulator.token_tuple or language == "cjk":
        return 0.0
    suffixes = _NOMINAL_SUFFIXES.get(language, ())
    content_tokens = [
        token.replace("-", "")
        for token in accumulator.token_tuple
        if token not in _CONNECTORS.get(language, frozenset())
    ]
    return (
        1.0
        if any(
            len(token) >= len(suffix) + 3 and token.endswith(suffix)
            for token in content_tokens
            for suffix in suffixes
        )
        else 0.0
    )


def _nominal_termhood(accumulator: _Accumulator, language: str) -> float:
    structural_signal = 0.85 if "noun_phrase" in accumulator.methods else 0.0
    determiner_hits = sum(
        count
        for context, count in accumulator.left_contexts.items()
        if context in _DETERMINERS.get(language, frozenset())
    )
    determiner_signal = min(1.0, 1.5 * determiner_hits / max(1, accumulator.frequency))
    return max(
        _derivational_termhood(accumulator, language),
        structural_signal,
        _orthographic_termhood(accumulator, language),
        determiner_signal,
    )


def _compactness(accumulator: _Accumulator) -> float:
    length = len(accumulator.token_tuple)
    if length <= 3:
        return 1.0
    if length == 4:
        return 0.75
    return 0.5


def _phrase_structure(accumulator: _Accumulator) -> float:
    methods = accumulator.methods
    if "coordination_phrase" in methods:
        structural = 1.0
    elif "connector_phrase" in methods:
        structural = 0.95
    elif "noun_phrase" in methods:
        structural = 0.85
    elif "proper_name" in methods:
        structural = 0.8
    elif "phrase" in methods and accumulator.frequency >= 2:
        structural = 0.6
    else:
        structural = 0.25
    return min(1.0, 0.8 * structural + 0.2 * _compactness(accumulator))


def _canonical_form(accumulator: _Accumulator) -> str:
    if "morphological_family" not in accumulator.methods:
        return accumulator.forms.most_common(1)[0][0]
    return min(
        accumulator.forms,
        key=lambda form: (
            len(_WORD_RE.findall(form)),
            len(_normalize(form)),
            -accumulator.forms[form],
            _normalize(form),
        ),
    )


def _ordered_forms(accumulator: _Accumulator, canonical: str) -> list[str]:
    return [canonical] + [form for form, _ in accumulator.forms.most_common() if form != canonical]


def _eligible(
    accumulator: _Accumulator,
    candidate_type: str,
    association: float,
    phrase_structure: float,
    language: str,
) -> bool:
    methods = accumulator.methods
    strong_explicit = bool({"definition_cue", "quoted", "heading"} & methods)
    token_count = len(accumulator.token_tuple)
    if candidate_type == "named_entity":
        return accumulator.frequency >= 2 or strong_explicit
    if strong_explicit or "coordination_phrase" in methods:
        return True
    if language == "de" and "proper_name" in methods:
        return True
    if "cjk_ngram" in methods:
        return accumulator.frequency >= 2
    if token_count > 1:
        return accumulator.frequency >= 2 and association >= 0.04 and phrase_structure >= 0.4
    if {"hyphenated", "acronym"} & methods:
        return accumulator.frequency >= 2
    if language == "de" and (
        _capitalized_surface_ratio(accumulator) < 0.6
        and _derivational_termhood(accumulator, language) == 0
    ):
        return False
    return accumulator.frequency >= 2 and len(accumulator.key) >= 4


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


def mine_candidates_v3(
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
    surface_lexemes = len(accumulators)
    if language != "cjk":
        accumulators = _merge_inflectional_variants(accumulators, language)
    c_values, _ = _nested_statistics(accumulators)
    max_c = max(c_values.values(), default=1.0) or 1.0
    max_frequency = max((item.frequency for item in accumulators.values()), default=1)
    singletons = {
        item.token_tuple[0]: item for item in accumulators.values() if len(item.token_tuple) == 1
    }
    last_ordinal = max((segment.ordinal for segment in segments), default=0)
    scored: list[tuple[float, _Accumulator, str, float, dict[str, float]]] = []
    rejected_by_gate = 0
    for accumulator in accumulators.values():
        candidate_type = _candidate_type(accumulator, language)
        association = _association(accumulator, singletons)
        phrase_structure = _phrase_structure(accumulator)
        if not _eligible(accumulator, candidate_type, association, phrase_structure, language):
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
        bin_coverage = len(bins) / 8
        segment_denominator = math.log2(min(32, max(1, len(segments))) + 1)
        segment_spread = math.log2(len(accumulator.segment_ids) + 1) / segment_denominator
        dispersion = min(1.0, 0.65 * bin_coverage + 0.35 * segment_spread)
        c_value = c_values[accumulator.key] / max_c
        frequency = math.log2(accumulator.frequency + 1) / math.log2(max_frequency + 1)
        c_value_floor = 0.75 if len(accumulator.token_tuple) == 1 else 0.35
        c_value = max(c_value, frequency * c_value_floor)
        boundary = _boundary_confidence(accumulator)
        if "cjk_ngram" in accumulator.methods:
            multiword = min(1.0, max(0, len(accumulator.key) - 2) / 3)
        else:
            multiword = min(1.0, max(0, len(accumulator.token_tuple) - 1) / 3)
        translation = min(1.0, len(accumulator.translated_segments) / 3)
        specificity = _lexical_specificity(accumulator)
        context_diversity = _context_diversity(accumulator)
        orthographic = _orthographic_termhood(accumulator, language)
        nominal = _nominal_termhood(accumulator, language)
        derivational = _derivational_termhood(accumulator, language)
        compactness = _compactness(accumulator)
        score = (
            0.17 * c_value
            + 0.14 * frequency
            + 0.10 * dispersion
            + 0.10 * explicit
            + 0.06 * boundary
            + 0.08 * association
            + 0.05 * multiword
            + 0.07 * specificity
            + 0.04 * context_diversity
            + 0.06 * orthographic
            + 0.07 * nominal
            + 0.04 * phrase_structure
            + 0.02 * translation
        )
        if candidate_type == "named_entity":
            score -= 0.03
        components = {
            "c_value": round(c_value, 6),
            "frequency": round(frequency, 6),
            "dispersion": round(dispersion, 6),
            "explicit": round(explicit, 6),
            "boundary": round(boundary, 6),
            "association": round(association, 6),
            "multiword": round(multiword, 6),
            "specificity": round(specificity, 6),
            "context_diversity": round(context_diversity, 6),
            "orthographic_termhood": round(orthographic, 6),
            "nominal_termhood": round(nominal, 6),
            "derivational_termhood": round(derivational, 6),
            "phrase_structure": round(phrase_structure, 6),
            "compactness": round(compactness, 6),
            "translation_risk": round(translation, 6),
        }
        scored.append(
            (max(0.0, min(1.0, score)), accumulator, candidate_type, boundary, components)
        )

    scored.sort(key=lambda item: (-item[0], -item[1].frequency, item[1].key))
    if language == "cjk":
        corpus_units = (
            sum(
                len(_CJK_RE.findall(segment.source_text))
                for segment in segments
                if not _is_metadata(segment)
            )
            / 2
        )
    else:
        corpus_units = sum(
            len(_WORD_RE.findall(segment.source_text))
            for segment in segments
            if not _is_metadata(segment)
        )
    corpus_budget = max(25, min(70, round(20 + 0.45 * math.sqrt(max(1, corpus_units)))))
    queue_ceiling = min(config.max_candidates, corpus_budget)

    def selection_lane(item) -> str:
        _, accumulator, candidate_type, _, components = item
        if candidate_type == "named_entity":
            return "named_entity"
        if candidate_type == "lexical_risk":
            return "lexical_risk"
        methods = accumulator.methods
        if "coordination_phrase" in methods:
            return "coordination_phrase"
        salient_methods = {
            "definition_cue",
            "quoted",
            "heading",
            "rhetorical_boundary",
        }
        if len(accumulator.token_tuple) > 1 and (
            salient_methods & methods
            or ("connector_phrase" in methods and components["derivational_termhood"] > 0)
        ):
            return "salient_phrase"
        if "cjk_ngram" in methods:
            return "phrase" if len(accumulator.key) >= 3 else "core_unigram"
        if len(accumulator.token_tuple) > 1:
            return "phrase"
        if components["derivational_termhood"] > 0:
            return "derived_unigram"
        return "core_unigram"

    score_qualified = [item for item in scored if item[0] >= config.min_score]

    def passes_local_precision_gate(item) -> bool:
        """Require corroborating evidence; a lane quota must never manufacture candidates."""
        _, accumulator, candidate_type, _, components = item
        methods = accumulator.methods
        frequency = accumulator.frequency
        segment_frequency = len(accumulator.segment_ids)
        token_count = len(accumulator.token_tuple)
        definition = "definition_cue" in methods
        generic_nouns = _GENERIC_GLOSSARY_NOUNS.get(language, frozenset())
        content_tokens = [
            token
            for token in accumulator.token_tuple
            if token not in _CONNECTORS.get(language, frozenset())
        ]
        if (
            not definition
            and token_count > 1
            and content_tokens
            and content_tokens[-1] in generic_nouns
        ):
            return False
        quoted_term = "quoted" in methods and token_count <= 3 and (
            frequency >= 2
            or (
                components["specificity"] >= 0.3
                and "proper_name" not in methods
            )
        )

        if definition or quoted_term:
            return True
        if candidate_type == "named_entity":
            return frequency >= 8 and segment_frequency >= 3
        if candidate_type == "lexical_risk":
            return "acronym" in methods or (
                frequency >= 3
                and segment_frequency >= 2
                and components["context_diversity"] >= 0.45
            )
        if language == "cjk":
            return (
                len(accumulator.key) >= 3
                and frequency >= 3
                and segment_frequency >= 2
            )
        if token_count > 1:
            if token_count > 3:
                return False
            if "coordination_phrase" in methods:
                return (
                    frequency >= 2
                    and segment_frequency >= 2
                    and components["association"] >= 0.08
                )
            if "connector_phrase" in methods:
                if frequency >= 2:
                    return (
                        segment_frequency >= 2
                        and components["association"] >= 0.08
                        and (
                            components["derivational_termhood"] > 0
                            or components["specificity"] >= 0.35
                        )
                    )
                # Preserve only a narrow class of one-off, explicitly rhetorical
                # concept phrases whose recurring component anchors them in the book.
                return (
                    "rhetorical_boundary" in methods
                    and components["derivational_termhood"] > 0
                    and 0.08 <= components["association"] <= 0.25
                    and components["specificity"] >= 0.4
                )
            return (
                frequency >= 3
                and segment_frequency >= 2
                and "noun_phrase" in methods
                and components["association"] >= 0.08
                and components["phrase_structure"] >= 0.6
            )
        if accumulator.key in generic_nouns:
            return False
        canonical = _canonical_form(accumulator)
        if language == "de" and not canonical[:1].isupper():
            return False
        if language == "de" and _capitalized_surface_ratio(accumulator) < 0.75:
            return False
        if components["derivational_termhood"] > 0:
            return (
                frequency >= 3
                and segment_frequency >= 2
                and components["specificity"] >= 0.35
                and components["context_diversity"] >= 0.45
            )
        return (
            frequency >= 5
            and segment_frequency >= 3
            and components["context_diversity"] >= 0.5
            and (frequency >= 12 or components["specificity"] >= 0.3)
        )

    qualified = [item for item in score_qualified if passes_local_precision_gate(item)]
    local_precision_rejections = len(score_qualified) - len(qualified)

    if language == "de":
        # German needs independent capacity for compounds, short nouns, and concept phrases.
        lane_targets = {
            "core_unigram": round(queue_ceiling * 0.40),
            "derived_unigram": round(queue_ceiling * 0.30),
            "phrase": round(queue_ceiling * 0.05),
            "salient_phrase": round(queue_ceiling * 0.15),
            "coordination_phrase": round(queue_ceiling * 0.05),
            "lexical_risk": round(queue_ceiling * 0.05),
            "named_entity": 0,
        }
    else:
        lane_targets = {
            "core_unigram": round(queue_ceiling * 0.25),
            "derived_unigram": round(queue_ceiling * 0.30),
            "phrase": round(queue_ceiling * 0.17),
            "salient_phrase": round(queue_ceiling * 0.15),
            "coordination_phrase": round(queue_ceiling * 0.03),
            "lexical_risk": round(queue_ceiling * 0.05),
            "named_entity": round(queue_ceiling * 0.05),
        }
    allocated = sum(lane_targets.values())
    if allocated < queue_ceiling:
        lane_targets["core_unigram"] += queue_ceiling - allocated
    elif allocated > queue_ceiling:
        lane_targets["phrase"] = max(0, lane_targets["phrase"] - (allocated - queue_ceiling))

    def lane_score(item) -> float:
        score, accumulator, _, _, components = item
        lane = selection_lane(item)
        if lane == "core_unigram":
            return (
                0.28 * components["frequency"]
                + 0.16 * components["dispersion"]
                + 0.22 * components["nominal_termhood"]
                + 0.14 * components["orthographic_termhood"]
                + 0.12 * components["context_diversity"]
                + 0.08 * components["specificity"]
            )
        if lane == "derived_unigram":
            return (
                0.35 * components["derivational_termhood"]
                + 0.22 * components["frequency"]
                + 0.15 * components["dispersion"]
                + 0.13 * components["specificity"]
                + 0.10 * components["context_diversity"]
                + 0.05 * components["orthographic_termhood"]
            )
        if lane == "phrase":
            return (
                0.25 * components["phrase_structure"]
                + 0.20 * components["association"]
                + 0.18 * components["c_value"]
                + 0.12 * components["frequency"]
                + 0.10 * components["specificity"]
                + 0.08 * components["compactness"]
                + 0.07 * components["dispersion"]
            )
        if lane == "lexical_risk":
            return (
                0.40 * components["frequency"]
                + 0.20 * components["dispersion"]
                + 0.15 * components["specificity"]
                + 0.15 * components["boundary"]
                + 0.10 * components["context_diversity"]
            )
        if lane == "coordination_phrase":
            repeated = 1.0 if accumulator.frequency >= 2 else 0.0
            return (
                0.35 * repeated
                + 0.25 * components["frequency"]
                + 0.15 * components["association"]
                + 0.15 * components["phrase_structure"]
                + 0.10 * components["compactness"]
            )
        if lane == "salient_phrase":
            methods = accumulator.methods
            marker = max(
                1.0 if "definition_cue" in methods else 0.0,
                0.9 if "quoted" in methods else 0.0,
                0.85 if "heading" in methods else 0.0,
                0.82 if "coordination_phrase" in methods else 0.0,
                0.78 if {"rhetorical_boundary", "connector_phrase"} <= methods else 0.0,
                0.7 if "rhetorical_boundary" in methods else 0.0,
                0.65
                if ("connector_phrase" in methods and components["derivational_termhood"] > 0)
                else 0.0,
            )
            return (
                0.32 * marker
                + 0.15 * components["phrase_structure"]
                + 0.08 * components["derivational_termhood"]
                + 0.10 * components["nominal_termhood"]
                + 0.08 * components["specificity"]
                + 0.05 * components["compactness"]
                + 0.12 * components["frequency"]
                + 0.10 * components["dispersion"]
            )
        return score

    lane_items = {
        lane: sorted(
            (item for item in qualified if selection_lane(item) == lane),
            key=lambda item: (
                -lane_score(item),
                -item[0],
                -item[1].frequency,
                item[1].key,
            ),
        )
        for lane in lane_targets
    }
    selected: list[tuple[float, _Accumulator, str, float, dict[str, float]]] = []
    selected_keys: set[str] = set()
    lane_counts: Counter[str] = Counter()
    lane_positions: dict[str, int] = {}

    def add_candidate(item) -> None:
        accumulator = item[1]
        if accumulator.key in selected_keys:
            return
        lane = selection_lane(item)
        selected.append(item)
        selected_keys.add(accumulator.key)
        lane_counts[lane] += 1
        lane_positions[accumulator.key] = lane_counts[lane]

    for lane, target in lane_targets.items():
        for item in lane_items[lane][:target]:
            add_candidate(item)

    hard_caps = {
        "named_entity": max(lane_targets["named_entity"], math.ceil(queue_ceiling * 0.20)),
        "salient_phrase": max(lane_targets["salient_phrase"], math.ceil(queue_ceiling * 0.20)),
        "coordination_phrase": max(
            lane_targets["coordination_phrase"], math.ceil(queue_ceiling * 0.08)
        ),
        "lexical_risk": max(lane_targets["lexical_risk"], math.ceil(queue_ceiling * 0.08)),
    }
    for item in qualified:
        if len(selected) >= queue_ceiling:
            break
        lane = selection_lane(item)
        if lane in hard_caps and lane_counts[lane] >= hard_caps[lane]:
            continue
        add_candidate(item)

    lane_order = {
        "coordination_phrase": 0,
        "lexical_risk": 1,
        "salient_phrase": 2,
        "derived_unigram": 3,
        "core_unigram": 4,
        "phrase": 5,
        "named_entity": 6,
    }
    selected.sort(
        key=lambda item: (
            (lane_positions[item[1].key] - 0.5) / max(1, lane_counts[selection_lane(item)]),
            lane_order[selection_lane(item)],
            -item[0],
            item[1].key,
        )
    )

    candidates: list[dict[str, Any]] = []
    type_counts: Counter[str] = Counter()
    selected_keys_with_frequency = {
        (item[1].token_tuple, item[1].frequency)
        for item in selected
    }
    complete_selected = []
    for item in selected:
        tokens = item[1].token_tuple
        frequency = item[1].frequency
        nested_in_complete_form = any(
            len(parent_tokens) > len(tokens)
            and parent_frequency == frequency
            and any(
                parent_tokens[start : start + len(tokens)] == tokens
                for start in range(len(parent_tokens) - len(tokens) + 1)
            )
            for parent_tokens, parent_frequency in selected_keys_with_frequency
        )
        if not nested_in_complete_form:
            complete_selected.append(item)
    local_redundancy_rejections = len(selected) - len(complete_selected)
    selected = complete_selected
    lane_counts = Counter(selection_lane(item) for item in selected)

    for rank, (score, accumulator, candidate_type, boundary, components) in enumerate(selected, 1):
        evidence = _evidence(accumulator, config.max_evidence_per_candidate)
        type_counts[candidate_type] += 1
        canonical = _canonical_form(accumulator)
        candidates.append(
            {
                "id": new_id("lexeme"),
                "lexeme_key": accumulator.key,
                "canonical_form": canonical,
                "forms": _ordered_forms(accumulator, canonical),
                "frequency": accumulator.frequency,
                "segment_frequency": len(accumulator.segment_ids),
                "risk_score": round(score, 6),
                "rank": rank,
                "candidate_type": candidate_type,
                "boundary_confidence": round(boundary, 6),
                "score_components": components,
                "extraction_methods": sorted(accumulator.methods | {"c_value_v3"}),
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
                        "proposer": "deterministic-v3.2",
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
        "raw_lexemes": surface_lexemes,
        "lexeme_families": len(accumulators),
        "linguistic_gate_rejections": rejected_by_gate,
        "ranked_candidates": len(scored),
        "score_qualified_candidates": len(score_qualified),
        "local_precision_rejections": local_precision_rejections,
        "qualified_candidates": len(qualified),
        "local_redundancy_rejections": local_redundancy_rejections,
        "retained_candidates": len(candidates),
        "candidate_types": dict(type_counts),
        "review_queue_ceiling": queue_ceiling,
        "plain_unigram_budget": (lane_targets["core_unigram"] + lane_targets["derived_unigram"]),
        "named_entity_budget": hard_caps["named_entity"],
        "selection_lane_targets": lane_targets,
        "selection_lanes": dict(lane_counts),
        "corpus_units": round(corpus_units),
        "truncated": len(qualified) > len(candidates),
        "language_profile": language,
        "selection_policy": "strict_local_admission",
        "candidate_status": "unverified_until_human_approval",
        "model_keep_confidence_floor": 0.75,
        "algorithm": "multilingual-family+c-value+strict-local-review-v3.2",
    }
    return candidates, coverage
