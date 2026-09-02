from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from html import escape

_TOKEN = re.compile(r"\[\[JY_PH_(\d{4})\]\]")
_PROTECTED = re.compile(
    r"(?P<placeholder_literal>\[\[JY_PH_\d{4}\]\])"
    r"|(?P<code_block>```[\s\S]*?```)"
    r"|(?P<inline_code>`[^`\n]+`)"
    r"|(?P<html_tag><[^>\n]+>)"
    r"|(?P<footnote>\[\^[^]\n]+\])"
    r"|(?P<url>https?://[^\s<>\]\)]+)"
    r"|(?P<doi>\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b)"
    r"|(?P<citation>\([^()\n]{0,100}\b(?:1[5-9]|20)\d{2}[a-z]?[^()\n]{0,60}\))"
    r"|(?P<bracket_reference>\[(?:\d{1,4}(?:\s*[-,;]\s*\d{1,4})*)\])",
    re.IGNORECASE,
)


def extract_placeholder_tokens(text: str) -> tuple[str, ...]:
    """Return protocol placeholders in their textual order."""
    return tuple(match.group(0) for match in _TOKEN.finditer(text))


class PlaceholderIntegrityError(ValueError):
    def __init__(self, message: str, *, missing: tuple[str, ...] = (), extra: tuple[str, ...] = ()):
        super().__init__(message)
        self.missing = missing
        self.extra = extra


@dataclass(frozen=True, slots=True)
class ProtectedSpan:
    token: str
    text: str
    kind: str


@dataclass(frozen=True, slots=True)
class ProtectedText:
    masked: str
    spans: tuple[ProtectedSpan, ...]

    @property
    def tokens(self) -> tuple[str, ...]:
        return tuple(span.token for span in self.spans)

    @property
    def atom_boundaries(self) -> tuple[tuple[str, str], ...]:
        """Expose typed EPUB boundaries without exposing source DOM addresses."""
        pairs: list[tuple[str, str]] = []
        opening: str | None = None
        for span in self.spans:
            if re.match(r"<jy-atom\s", span.text):
                if opening is not None:
                    raise PlaceholderIntegrityError("Nested SourceAtom boundaries in source")
                opening = span.token
            elif span.text == "</jy-atom>":
                if opening is None:
                    raise PlaceholderIntegrityError("Unmatched SourceAtom boundary in source")
                pairs.append((opening, span.token))
                opening = None
        if opening is not None:
            raise PlaceholderIntegrityError("Unclosed SourceAtom boundary in source")
        return tuple(pairs)

    def assemble_atom_repair(self, response: str) -> str:
        """Rebuild wrappers locally from a complete, strictly keyed fragment response."""
        pairs = self.atom_boundaries
        if not pairs:
            return response
        cleaned = response.strip()
        if cleaned.startswith("```json\n") and cleaned.endswith("```"):
            cleaned = cleaned[8:-3].strip()
        elif cleaned.startswith("```\n") and cleaned.endswith("```"):
            cleaned = cleaned[4:-3].strip()

        def unique_object(items):
            result = {}
            for key, value in items:
                if key in result:
                    raise ValueError(f"Duplicate SourceAtom repair key: {key}")
                result[key] = value
            return result

        try:
            values = json.loads(cleaned, object_pairs_hook=unique_object)
        except ValueError as exc:
            raise PlaceholderIntegrityError(f"SourceAtom repair requires valid JSON: {exc}") from exc
        expected = {opening for opening, _ in pairs}
        if not isinstance(values, dict) or set(values) != expected:
            raise PlaceholderIntegrityError("SourceAtom repair must contain exactly the requested keys")
        pieces: list[str] = []
        for opening, closing in pairs:
            value = values[opening]
            if not isinstance(value, str) or not value.strip():
                raise PlaceholderIntegrityError(f"SourceAtom repair is empty: {opening}")
            source_fragment = self.masked.split(opening, 1)[1].split(closing, 1)[0]
            if extract_placeholder_tokens(value) != extract_placeholder_tokens(source_fragment):
                raise PlaceholderIntegrityError(
                    f"SourceAtom repair changed inline placeholders in {opening}"
                )
            # JSON values are text with protected inline tokens, never model-supplied XML.
            pieces.append(opening + escape(value.strip(), quote=False) + closing)
        assembled = "".join(pieces)
        self.validate(assembled)
        return assembled

    def mask_translation(self, translated_text: str) -> str:
        """Mask restored protected values before sending a draft to a reviewer."""
        masked = translated_text
        for span in self.spans:
            if span.text not in masked:
                raise PlaceholderIntegrityError(
                    f"Protected value is missing before review: {span.text}",
                    missing=(span.token,),
                )
            masked = masked.replace(span.text, span.token, 1)
        return masked

    def restore(self, candidate: str) -> str:
        self.validate(candidate)
        restored = candidate
        for span in self.spans:
            restored = restored.replace(span.token, span.text, 1)
        return restored

    def validate(self, candidate: str) -> None:
        expected = self.tokens
        found = extract_placeholder_tokens(candidate)
        if not expected:
            if found:
                raise PlaceholderIntegrityError(
                    f"Unexpected placeholder tokens in unprotected text: {found}", extra=found
                )
            return

        expected_counts = Counter(expected)
        found_counts = Counter(found)
        missing = tuple(token for token in expected if found_counts[token] < expected_counts[token])
        extra = tuple(token for token in found if expected_counts[token] < found_counts[token])

        if missing or extra:
            raise PlaceholderIntegrityError(
                f"Placeholder count mismatch; missing={missing or 'none'}, extra={extra or 'none'}",
                missing=missing,
                extra=extra,
            )
        if found != expected:
            raise PlaceholderIntegrityError(
                f"Placeholder order changed; expected={expected}, found={found}"
            )

    def repair_surplus_placeholders(self, candidate: str) -> str | None:
        """Remove only provably surplus duplicates while preserving all translated text."""
        expected = self.tokens
        matches = list(_TOKEN.finditer(candidate))
        if not expected or len(matches) <= len(expected):
            return None

        expected_set = set(expected)
        if any(match.group(0) not in expected_set for match in matches):
            return None

        kept_indexes: set[int] = set()
        expected_index = 0
        for match_index, match in enumerate(matches):
            if expected_index < len(expected) and match.group(0) == expected[expected_index]:
                kept_indexes.add(match_index)
                expected_index += 1
        if expected_index != len(expected):
            return None

        pieces: list[str] = []
        cursor = 0
        for match_index, match in enumerate(matches):
            pieces.append(candidate[cursor : match.start()])
            if match_index in kept_indexes:
                pieces.append(match.group(0))
            cursor = match.end()
        pieces.append(candidate[cursor:])
        repaired = "".join(pieces)
        self.validate(repaired)
        return repaired


class ProtectedTextCodec:
    """Protect citations and technical spans with a strict round-trip protocol."""

    def encode(self, text: str) -> ProtectedText:
        spans: list[ProtectedSpan] = []
        pieces: list[str] = []
        cursor = 0
        # A source may legitimately quote the internal protocol syntax. Reserve
        # every literal token found in the source so generated masks can never
        # collide with it; the literal itself is protected like any other span.
        reserved_tokens = set(extract_placeholder_tokens(text))
        next_token_index = 0

        def allocate_token() -> str:
            nonlocal next_token_index
            while next_token_index <= 9_999:
                token = f"[[JY_PH_{next_token_index:04d}]]"
                next_token_index += 1
                if token not in reserved_tokens:
                    reserved_tokens.add(token)
                    return token
            raise PlaceholderIntegrityError("Source contains too many placeholder-like tokens")

        for match in _PROTECTED.finditer(text):
            pieces.append(text[cursor : match.start()])
            token = allocate_token()
            kind = match.lastgroup or "protected"
            spans.append(ProtectedSpan(token=token, text=match.group(0), kind=kind))
            pieces.append(token)
            cursor = match.end()

        pieces.append(text[cursor:])
        return ProtectedText(masked="".join(pieces), spans=tuple(spans))
