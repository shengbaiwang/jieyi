from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

_ARABIC_NUMBER = re.compile(
    r"(?<!\d)[+\-−]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?(?!\d)"
)
_VERSION_TOKEN = re.compile(r"\bv\d+(?:\.\d+)+(?:[_-]r?\d+)?\b", re.IGNORECASE)
_MONTHS = {
    name: number
    for number, name in enumerate(
        ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"],
        1,
    )
}
_MONTH_PATTERN = re.compile(r"\b(" + "|".join(_MONTHS) + r")\b", re.IGNORECASE)
_ORDINALS = {
    name: number
    for number, name in enumerate(
        ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth", "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth", "seventeenth", "eighteenth", "nineteenth", "twentieth", "twenty-first"],
        1,
    )
}
_WORD_CENTURY = re.compile(
    r"\b(" + "|".join(re.escape(item) for item in _ORDINALS) + r")\s+centur(?:y|ies)\b",
    re.IGNORECASE,
)
_DIGIT_CENTURY = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)\s+centur(?:y|ies)\b", re.IGNORECASE)
_ENGLISH_DECADE = re.compile(r"(?<!\d)(\d{2,3}0)s\b", re.IGNORECASE)
_ENGLISH_SHORT_DECADE = re.compile(r"(?<!\d)(?:['’])?(\d{2})s\b", re.IGNORECASE)
_CHINESE_COORDINATED_DECADE = re.compile(
    r"(?<!\d)(\d{1,2})\s*世纪\s*(\d{1,2})\s*年代(?:末|初)?\s*"
    r"(?:和|与|、|至|到)\s*(\d{1,2})\s*年代"
)
_CHINESE_DECADE = re.compile(r"(?<!\d)(\d{1,2})\s*世纪\s*(\d{1,2})\s*年代")
_CHINESE_ABSOLUTE_DECADE = re.compile(r"(?<!\d)(\d{2,3}0)\s*年代")
_CHINESE_SHORT_DECADE = re.compile(r"(?<!\d)(\d{1,2})\s*年代")
_CHINESE_WORD_DECADE = re.compile(
    r"([零〇一二两三四五六七八九十]+)\s*世纪\s*"
    r"([零〇一二两三四五六七八九十]+)\s*年代"
)
_CHINESE_WORD_SHORT_DECADE = re.compile(r"([零〇一二两三四五六七八九十]+)\s*年代")
_CHINESE_WORD_CENTURY = re.compile(r"([零〇一二两三四五六七八九十]+)\s*世纪")
_CHINESE_CONTRACTED_CENTURY = re.compile(
    r"([一二两三四五六七八九])([一二三四五六七八九])\s*世纪"
)
_ENGLISH_SCALE = re.compile(
    r"(?<![\d.])([+\-]?\d[\d,]*(?:\.\d+)?)\s*[-‐‑–—]?\s*"
    r"(hundred|thousand|million|billion|trillion)\b",
    re.IGNORECASE,
)
_CHINESE_SCALE = re.compile(r"(?<![\d.])([+\-]?\d[\d,]*(?:\.\d+)?)\s*([百千万亿])")
_PERCENT = re.compile(
    r"(?<![\d.])([+\-]?\d[\d,]*(?:\.\d+)?)\s*(%|percent\b|per\s+cent\b)",
    re.IGNORECASE,
)
_TIME = re.compile(r"(?<!\d)([01]?\d|2[0-3]):([0-5]\d)(?!\d)")
_CHINESE_TIME = re.compile(r"(?<!\d)([01]?\d|2[0-3])\s*点(?:钟)?")
_LOCATOR = re.compile(r"(?<!\d)(\d{1,3})[.:](\d{1,4})(?!\d)")
_CHINESE_MONTH = re.compile(r"(?<!\d)(1[0-2]|0?[1-9])\s*月")
_CHINESE_WORD_MONTH = re.compile(r"([一二两三四五六七八九十]+)\s*月")
_CHINESE_CENTURY = re.compile(r"(?<!\d)(\d{1,2})\s*世纪")
_CHINESE_NUMBER = re.compile(r"[零〇一二两三四五六七八九十百千万亿点]+")
_CHINESE_CONTRACTED_RANGE = re.compile(r"([一二两三四五六七八九])([一二三四五六七八九])([十百千万])")
_CHINESE_FRACTION = re.compile(
    r"([零〇一二两三四五六七八九十百]+)分之([零〇一二两三四五六七八九十百]+)"
)
_CHINESE_HALF = re.compile(r"(?:一半|半数)")
_CHINESE_TENTHS = re.compile(r"([一二两三四五六七八九十])成")
_LEADING_MARKER = re.compile(r"^\s*(\d{1,3})(?=(?:[)）、]|[.．]\s))")
_INLINE_NOTE_AFTER_SENTENCE = re.compile(
    r"(?<!\d)[.!?。！？:：…][\"'”’）)\]}]{0,2}\d{1,3}(?=\s|$)"
)
_INLINE_NOTE_AFTER_WORD = re.compile(r"(?<=[A-Za-z])\d{1,3}(?=\s|[.,;:!?]|$)")
_YEAR_FOOTNOTE = re.compile(r"(?<=\d{4}\.)\d{1,3}(?=\s|$)")
_PROTECTED = re.compile(
    r"https?://[^\s<>\]\)]+"
    r"|\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b"
    r"|\[(?:\d{1,4}(?:\s*[-,;]\s*\d{1,4})*)\]"
    r"|\[\^[^]]+\]",
    re.IGNORECASE,
)
_SCALE = {
    "hundred": Decimal(100),
    "thousand": Decimal(1_000),
    "million": Decimal(1_000_000),
    "billion": Decimal(1_000_000_000),
    "trillion": Decimal(1_000_000_000_000),
    "百": Decimal(100),
    "千": Decimal(1_000),
    "万": Decimal(10_000),
    "亿": Decimal(100_000_000),
}
_CHINESE_DIGIT = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
                  "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_CHINESE_UNIT = {"十": 10, "百": 100, "千": 1_000, "万": 10_000, "亿": 100_000_000}


@dataclass(frozen=True, slots=True)
class NumericFacts:
    typed: Counter[str]
    raw: Counter[str]
    chinese: Counter[str]


@dataclass(frozen=True, slots=True)
class NumericComparison:
    missing_typed: Counter[str]
    missing_raw: Counter[str]
    source: NumericFacts
    target: NumericFacts

    @property
    def has_missing(self) -> bool:
        return bool(self.missing_typed or self.missing_raw)


def _normalise_text(text: str) -> str:
    value = unicodedata.normalize("NFKC", text)
    return value.translate(str.maketrans({"−": "-", "–": "-", "—": "-", "﹣": "-"}))


def _decimal(value: str) -> Decimal:
    return Decimal(value.replace(",", "").rstrip("%").lstrip("+−-"))


def _decimal_key(value: Decimal) -> str:
    if value == value.to_integral():
        return str(value.quantize(Decimal(1)))
    return format(value.normalize(), "f")


def _mask(chars: list[str], start: int, end: int) -> None:
    chars[start:end] = " " * (end - start)


def _take(
    chars: list[str],
    pattern: re.Pattern[str],
    typed: Counter[str],
    convert,
) -> None:
    text = "".join(chars)
    for match in pattern.finditer(text):
        fact = convert(match)
        if fact is not None:
            typed[fact] += 1
            _mask(chars, match.start(), match.end())


def _english_month_fact(match: re.Match[str]) -> str | None:
    token = match.group(0)
    name = token.casefold()
    if not token[0].isupper():
        return None
    if name in {"may", "march"}:
        before = match.string[max(0, match.start() - 12):match.start()].casefold()
        after = match.string[match.end():match.end() + 12]
        if not re.search(r"(?:\b(?:in|on|by|during|since|until|from|this|last|next)\s*)$", before) and not re.match(r"\s+\d", after):
            return None
    return f"month:{_MONTHS[name]}"


def _chinese_integer(value: str) -> int | None:
    if not value:
        return None
    if all(char in _CHINESE_DIGIT for char in value):
        return int("".join(str(_CHINESE_DIGIT[char]) for char in value))
    total = section = number = 0
    for char in value:
        if char in _CHINESE_DIGIT:
            number = _CHINESE_DIGIT[char]
            continue
        unit = _CHINESE_UNIT.get(char)
        if unit is None:
            return None
        if unit < 10_000:
            section += (number or 1) * unit
        else:
            section += number
            total += (section or 1) * unit
            section = 0
        number = 0
    return total + section + number


def _chinese_decimal(value: str) -> Decimal | None:
    scale = Decimal(1)
    if value.endswith(("万", "亿")) and "点" in value:
        scale = _SCALE[value[-1]]
        value = value[:-1]
    if "点" not in value:
        integer = _chinese_integer(value)
        return Decimal(integer) if integer is not None else None
    whole, fraction = value.split("点", 1)
    whole_value = _chinese_integer(whole)
    if whole_value is None or not fraction or not all(char in _CHINESE_DIGIT for char in fraction):
        return None
    decimal = Decimal(f"{whole_value}." + "".join(str(_CHINESE_DIGIT[c]) for c in fraction))
    return decimal * scale


def _extract_chinese_numbers(text: str) -> Counter[str]:
    values: Counter[str] = Counter()
    for match in _CHINESE_NUMBER.finditer(text):
        token = match.group(0)
        before = text[match.start() - 1] if match.start() else ""
        after = text[match.end()] if match.end() < len(text) else ""
        following = text[match.end():match.end() + 5]
        informative = (
            any(char in token for char in "十百千万亿点")
            or before == "第"
            or after in "年月日世纪代个名次倍岁页章节卷号点时分秒元块吨米"
            or re.match(r"(?:英镑|比索|法郎|美元|古尔登|便士|先令)", following)
        )
        if not informative:
            continue
        value = _chinese_decimal(token)
        if value is not None:
            values[_decimal_key(value)] += 1
    for match in _CHINESE_CONTRACTED_RANGE.finditer(text):
        first = _CHINESE_DIGIT[match[1]] * _CHINESE_UNIT[match[3]]
        second = _CHINESE_DIGIT[match[2]] * _CHINESE_UNIT[match[3]]
        values[str(first)] += 1
        values[str(second)] += 1
    return values


def extract_numeric_facts(
    text: str, *, ignore_note_markers: bool = False
) -> NumericFacts:
    # NFKC maps superscript/circled digits to their ordinary forms. Keeping them
    # comparable avoids one-sided masking when a translation changes note style.
    normalised = _normalise_text(text)
    chars = list(normalised)
    typed: Counter[str] = Counter()

    _take(chars, _VERSION_TOKEN, typed, lambda m: f"version:{m.group(0).casefold()}")

    patterns = [_PROTECTED]
    if ignore_note_markers:
        patterns.extend(
            (
                _LEADING_MARKER,
                _YEAR_FOOTNOTE,
                _INLINE_NOTE_AFTER_SENTENCE,
                _INLINE_NOTE_AFTER_WORD,
            )
        )
    for pattern in patterns:
        snapshot = "".join(chars)
        for match in pattern.finditer(snapshot):
            _mask(chars, match.start(), match.end())

    snapshot = "".join(chars)
    for match in _CHINESE_COORDINATED_DECADE.finditer(snapshot):
        century = int(match[1])
        typed[f"decade:{(century - 1) * 100 + int(match[2])}"] += 1
        typed[f"decade:{(century - 1) * 100 + int(match[3])}"] += 1
        _mask(chars, match.start(), match.end())

    snapshot = "".join(chars)
    for match in _CHINESE_CONTRACTED_CENTURY.finditer(snapshot):
        typed[f"century:{_CHINESE_DIGIT[match[1]]}"] += 1
        typed[f"century:{_CHINESE_DIGIT[match[2]]}"] += 1
        _mask(chars, match.start(), match.end())

    def word_decade(match: re.Match[str]) -> str | None:
        century = _chinese_integer(match[1])
        decade = _chinese_integer(match[2])
        if century is None or decade is None:
            return None
        return f"decade:{(century - 1) * 100 + decade}"

    _take(chars, _CHINESE_DECADE, typed, lambda m: f"decade:{(int(m[1]) - 1) * 100 + int(m[2])}")
    _take(chars, _CHINESE_WORD_DECADE, typed, word_decade)
    _take(chars, _ENGLISH_DECADE, typed, lambda m: f"decade:{int(m[1])}")
    _take(chars, _ENGLISH_SHORT_DECADE, typed, lambda m: f"decade-short:{int(m[1])}")
    _take(chars, _CHINESE_ABSOLUTE_DECADE, typed, lambda m: f"decade:{int(m[1])}")
    _take(chars, _CHINESE_SHORT_DECADE, typed, lambda m: f"decade-short:{int(m[1])}")
    _take(
        chars,
        _CHINESE_WORD_SHORT_DECADE,
        typed,
        lambda m: f"decade-short:{_chinese_integer(m[1])}",
    )
    _take(chars, _WORD_CENTURY, typed, lambda m: f"century:{_ORDINALS[m[1].casefold()]}")
    _take(chars, _DIGIT_CENTURY, typed, lambda m: f"century:{int(m[1])}")
    _take(chars, _CHINESE_CENTURY, typed, lambda m: f"century:{int(m[1])}")
    _take(
        chars,
        _CHINESE_WORD_CENTURY,
        typed,
        lambda m: f"century:{_chinese_integer(m[1])}",
    )
    _take(chars, _MONTH_PATTERN, typed, _english_month_fact)
    _take(chars, _CHINESE_MONTH, typed, lambda m: f"month:{int(m[1])}")
    _take(
        chars,
        _CHINESE_WORD_MONTH,
        typed,
        lambda m: (
            f"month:{_chinese_integer(m[1])}"
            if _chinese_integer(m[1]) is not None
            and 1 <= _chinese_integer(m[1]) <= 12
            else None
        ),
    )

    def scaled(match: re.Match[str]) -> str | None:
        try:
            return f"quantity:{_decimal_key(_decimal(match[1]) * _SCALE[match[2].casefold()])}"
        except (InvalidOperation, KeyError):
            return None

    def fraction_percent(match: re.Match[str]) -> str | None:
        denominator = _chinese_integer(match[1])
        numerator = _chinese_integer(match[2])
        if not denominator or numerator is None:
            return None
        value = Decimal(numerator) * Decimal(100) / Decimal(denominator)
        return f"percent-approx:{_decimal_key(value)}"

    _take(chars, _PERCENT, typed, lambda m: f"percent:{_decimal_key(_decimal(m[1]))}")
    _take(chars, _CHINESE_FRACTION, typed, fraction_percent)
    _take(chars, _CHINESE_HALF, typed, lambda _: "percent:50")
    _take(
        chars,
        _CHINESE_TENTHS,
        typed,
        lambda m: f"percent:{(_chinese_integer(m[1]) or 0) * 10}",
    )
    _take(chars, _ENGLISH_SCALE, typed, scaled)
    _take(chars, _CHINESE_SCALE, typed, scaled)
    _take(chars, _TIME, typed, lambda m: f"time:{int(m[1])}:{int(m[2])}")
    _take(chars, _CHINESE_TIME, typed, lambda m: f"time:{int(m[1])}:0")
    _take(chars, _LOCATOR, typed, lambda m: f"locator:{int(m[1])}:{int(m[2])}")

    remainder = "".join(chars)
    raw: Counter[str] = Counter()
    for match in _ARABIC_NUMBER.finditer(remainder):
        try:
            raw[_decimal_key(_decimal(match.group(0)))] += 1
        except InvalidOperation:
            continue
    return NumericFacts(typed=typed, raw=raw, chinese=_extract_chinese_numbers(remainder))


def _typed_scalar_pool(facts: Counter[str]) -> Counter[str]:
    pool: Counter[str] = Counter()
    for fact, count in facts.items():
        kind, _, value = fact.partition(":")
        if kind in {"quantity", "percent", "percent-approx", "month", "century"} and value:
            pool[value] += count
    return pool


def _typed_alternatives(fact: str) -> tuple[str, ...]:
    kind, _, value = fact.partition(":")
    try:
        number = int(value)
    except ValueError:
        return ()
    if kind == "decade-short":
        return tuple(f"decade:{century + number}" for century in range(1500, 2100, 100))
    if kind == "decade" and number % 100 == 0:
        return (f"century:{number // 100 + 1}",)
    return ()


def _missing_typed(source: Counter[str], target: Counter[str]) -> Counter[str]:
    # Translation may mention the same fact fewer times without losing it. Presence,
    # rather than raw repetition count, is the high-precision invariant.
    missing: Counter[str] = Counter()
    for fact in source:
        if target[fact] > 0:
            continue
        if any(target[item] > 0 for item in _typed_alternatives(fact)):
            continue
        if fact.startswith("percent:"):
            value = Decimal(fact.partition(":")[2])
            if any(
                item_count > 0
                and item.startswith("percent-approx:")
                and abs(Decimal(item.partition(":")[2]) - value) <= Decimal(4)
                for item, item_count in target.items()
            ):
                continue
        missing[fact] = 1
    return missing


def compare_numeric_facts(source: str, target: str) -> NumericComparison:
    # Source-side note markers are not translation facts. Target-side markers stay
    # visible so changes such as ``10`` -> ``¹⁰`` cannot create a false omission.
    source_facts = extract_numeric_facts(source, ignore_note_markers=True)
    target_facts = extract_numeric_facts(target)
    missing_typed = _missing_typed(source_facts.typed, target_facts.typed)
    target_pool = target_facts.raw + target_facts.chinese + _typed_scalar_pool(target_facts.typed)
    missing_raw = Counter(
        {value: 1 for value in source_facts.raw if target_pool[value] == 0}
    )
    return NumericComparison(
        missing_typed=missing_typed,
        missing_raw=missing_raw,
        source=source_facts,
        target=target_facts,
    )
