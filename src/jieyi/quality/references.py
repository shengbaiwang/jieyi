"""Recognize bibliographic entries that may legitimately retain their source text."""
from __future__ import annotations

import re
import unicodedata

_YEAR = r"(?:1[5-9]|20)\d{2}[a-z]?"
_NAME_TOKEN = re.compile(r"[^\W\d_]+(?:[’'-][^\W\d_]+)*\.?", re.UNICODE)
_PARTICLES = {"de", "del", "della", "di", "du", "la", "le", "van", "von", "al", "bin", "and", "et", "&"}
_LOCATOR = r"\d+(?:n\d+)?(?:\s*[-–—]\s*\d+(?:n\d+)?)?"
_INDEX = re.compile(rf"^(.+?),\s*({_LOCATOR}(?:\s*[,;]\s*{_LOCATOR})*)\.?$")
_AUTHOR_YEAR = re.compile(rf"^(.+?)[.,]\s*{_YEAR}(?:[.,:]\s*|\s+)(\S.+)$")
_SHORT_REFERENCE = re.compile(rf"^(.+?)\s+{_YEAR}(?:\s*[:,]\s*{_LOCATOR})?$")
_QUOTED_TITLE = re.compile(r'''“[^”]{3,}”|"[^"]{3,}"|(?<!\w)‘[^’]{3,}’|(?<!\w)'[^']{3,}' ''')


def _author(text: str, *, full_name: bool = False) -> bool:
    text = re.sub(r"\((?:eds?|editors?)\)", "", text, flags=re.IGNORECASE).strip(" ,.;")
    text = re.sub(r"\bet al\.?$", "", text).strip()
    tokens = text.replace(",", " ").split()
    if not tokens or len(tokens) > 12:
        return False
    capitalized = 0
    for token in tokens:
        if token in _PARTICLES:
            continue
        if not _NAME_TOKEN.fullmatch(token) or not token[0].isupper():
            return False
        capitalized += 1
    return capitalized >= (2 if full_name else 1)


def is_retained_reference(text: str) -> bool:
    """Require citation structure or an index locator list, never just a year."""
    value = " ".join(unicodedata.normalize("NFKC", text).split())
    if not value or len(value) > 2_000:
        return False
    numbered = re.match(r"^\d{1,4}[.)]\s+", value)
    if numbered:
        value = value[numbered.end():]
    parenthetical = value.startswith("(") and value.endswith(")")
    if parenthetical:
        value = value[1:-1].strip()

    index_value = re.split(r"\.\s+See also\b", value, flags=re.IGNORECASE)[0]
    index = _INDEX.fullmatch(index_value)
    if index and (
        parenthetical
        or len(re.findall(_LOCATOR, index[2])) >= 2
        or ("," in index[1] and _author(index[1].split(",", 1)[0]))
    ):
        return True

    # Publication details also identify entries split across PDF paragraphs.
    # Require an author prefix or an editor credit in addition to the imprint.
    author_prefix = _author(value.split(",", 1)[0]) if "," in value else False
    imprint = re.search(r"\([^()]{0,100}:\s*[^()]{1,180}\)", value)
    if imprint and (
        re.search(_YEAR, imprint[0]) or re.search(r"\bPress\b", imprint[0])
    ) and (author_prefix or re.search(r"\beds?\.", value)):
        return True

    # An author's repeated-entry dash and an immediately following year are
    # distinctive bibliography conventions; prose containing a date is not.
    if re.match(rf"^[—–-]{{1,3}}\.?\s*{_YEAR}[.,]?\s+\S", value):
        return True
    author_year = _AUTHOR_YEAR.match(value)
    if author_year and "," in author_year[1] and _author(author_year[1], full_name=True):
        return True

    references = value.rstrip(".").split(";")
    if references and all(
        (match := _SHORT_REFERENCE.fullmatch(part.strip())) and _author(match[1])
        for part in references
    ):
        return True

    # Quoted titles occur after an author, rather than a reporting clause such
    # as "Darwin wrote". Require a locator, an inverted name or a numbered note.
    title = _QUOTED_TITLE.search(value)
    if title:
        prefix = value[:title.start()].strip(" ,.;")
        suffix = value[title.end():]
        if _author(prefix) and (
            numbered or "," in prefix or re.search(r"\d", suffix)
        ):
            return True
        if (numbered and not prefix and re.search(_YEAR, suffix)
                and re.search(rf"[, :]\s*{_LOCATOR}\.?$", suffix)):
            return True

    # Some short notes contain only a full name and an unquoted book title.
    # Limit this to an enclosing citation or numbered note, with no extra prose.
    if numbered or parenthetical:
        author, separator, title = value.partition(",")
        title = title.strip()
        if (separator and _author(author, full_name=True) and title
                and title[0].isupper() and not re.search(r"[.!?]\s+\S", title)):
            return True
    return False
