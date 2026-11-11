"""Chinese font fallbacks for translated XHTML, resolved from the source tree."""

from __future__ import annotations

import posixpath
import re
from urllib.parse import unquote, urlsplit

from jieyi.ingestion.epub_structure import (
    _CSS_COMMENT,
    _CSS_RULE,
    _local_name,
    _selector_matches,
    _selector_specificity,
)
from jieyi.ingestion.typography import CJK_FAMILIES, font_family_role

_INHERITED = ("font-family", "font-style", "font-weight", "font-variant")
_DEFAULTS = {"font-family": "serif", "font-style": "normal", "font-weight": "normal"}
_FONT_SHORTHAND = re.compile(
    r"^(.*?)\b(?:\d*\.?\d+(?:px|pt|pc|em|rem|ex|ch|vw|vh|%)|"
    r"xx-small|x-small|small|medium|large|x-large|xx-large)"
    r"(?:\s*/\s*[^\s]+)?\s+(.+)$",
    re.IGNORECASE,
)
_IMPORT = re.compile(
    r"@import\s+(?:url\(\s*['\"]?([^'\")]+)['\"]?\s*\)|['\"]([^'\"]+)['\"])[^;]*;", re.IGNORECASE
)


def _declarations(style: dict[str, str]) -> dict[str, str]:
    result = {}
    font = style.get("font", "")
    if font in {"inherit", "unset", "initial"}:
        result.update(dict.fromkeys(_INHERITED, font))
    shorthand = _FONT_SHORTHAND.match(font)
    if shorthand:
        prefix, family = shorthand.groups()
        result.update({"font-family": family, "font-style": "normal", "font-weight": "normal"})
        if re.search(r"\b(?:italic|oblique)\b", prefix):
            result["font-style"] = "italic"
        weight = re.search(r"\b(?:bold|bolder|[1-9]00)\b", prefix)
        if weight:
            result["font-weight"] = weight.group(0)
    result.update({key: value for key, value in style.items() if key in _INHERITED})
    return result


class _TypographyResolver:
    """Resolve font declarations without changing ingestion/segmentation rules."""

    def __init__(self, sources):
        self.rules = []
        for source in sources:
            for match in _CSS_RULE.finditer(_CSS_COMMENT.sub("", source)):
                for selector in match.group(1).split(","):
                    selector = selector.strip()
                    if selector and not selector.startswith("@"):
                        self.rules.append((selector, match.group(2)))

    def computed(self, element, ancestors):
        winners = {}

        def add(declarations, priority):
            for index, match in enumerate(re.finditer(r"([\w-]+)\s*:\s*([^;]+)", declarations)):
                key, value = match.groups()
                important = bool(re.search(r"!\s*important\s*$", value, re.IGNORECASE))
                value = (
                    re.sub(r"!\s*important\s*$", "", value, flags=re.IGNORECASE).strip().casefold()
                )
                for property_name, expanded in _declarations({key.casefold(): value}).items():
                    rank = (important, *priority, index)
                    if property_name not in winners or rank >= winners[property_name][0]:
                        winners[property_name] = (rank, expanded)

        for order, (selector, declarations) in enumerate(self.rules):
            if _selector_matches(element, ancestors, selector):
                add(declarations, (0, *_selector_specificity(selector), order))
        add(element.attrib.get("style", ""), (1, 0, 0, 0, 0))
        return {key: value for key, (_, value) in winners.items()}


def _computed(resolver, element, ancestors, inherited):
    result = dict(inherited)
    tag = _local_name(element.tag)
    if tag in {"em", "i", "cite", "dfn", "var"}:
        result["font-style"] = "italic"
    if tag in {"strong", "b", "th", "h1", "h2", "h3", "h4", "h5", "h6"}:
        result["font-weight"] = "bold"
    if tag in {"code", "pre", "kbd", "samp"}:
        result["font-family"] = "monospace"
    for key, value in resolver.computed(element, ancestors).items():
        if value in {"inherit", "unset"}:
            result[key] = inherited.get(key, _DEFAULTS.get(key, "normal"))
        else:
            result[key] = _DEFAULTS.get(key, "normal") if value == "initial" else value
    return result


class TranslationTypography:
    def __init__(self, root, css_sources=()):
        self.resolver = _TypographyResolver(css_sources)
        self.styles = {}
        self.ancestors = {}

        def walk(element, ancestors, inherited):
            style = _computed(self.resolver, element, ancestors, inherited)
            self.styles[element] = style
            self.ancestors[element] = ancestors
            for child in element:
                walk(child, ancestors + (element,), style)

        walk(root, (), _DEFAULTS)

    def apply(self, target, source):
        """Decorate only translated elements; originals retain their own typography."""

        def walk(element, ancestors, style):
            if _local_name(element.tag) in {
                "svg",
                "math",
                "img",
                "br",
                "hr",
                "audio",
                "video",
                "source",
            }:
                return
            family = style.get("font-family", "serif")
            role = font_family_role(family)
            italic = style.get("font-style", "normal").startswith(("italic", "oblique"))
            if italic and role != "mono":
                role = "kai"
            # Retain source Latin faces before the CJK fallback, but put generic
            # families last: a generic serif earlier would consume all CJK glyphs.
            names = [name.strip() for name in family.split(",") if name.strip()]
            generic = {"serif", "sans-serif", "monospace", "cursive", "fantasy", "system-ui"}
            latin = [name for name in names if name.strip("\"'").casefold() not in generic]
            stack = ", ".join(latin + [CJK_FAMILIES[role]])
            weight = style.get("font-weight", "normal")
            if not re.fullmatch(r"normal|bold|bolder|lighter|\d{1,4}", weight):
                weight = "normal"
            overrides = f"font-family: {stack} !important; font-weight: {weight} !important;"
            overrides += f" font-style: {'italic' if italic else 'normal'} !important;"
            element.attrib["style"] = (
                element.attrib.get("style", "").rstrip("; ") + "; " + overrides
            ).lstrip("; ")
            element.attrib["data-jy-typeface"] = role
            context = source if element is target else element
            for child in element:
                child_style = self.styles.get(child)
                if child_style is None:
                    child_style = _computed(self.resolver, child, ancestors + (context,), style)
                walk(child, ancestors + (context,), child_style)

        walk(target, self.ancestors[source], self.styles[source])


def book_typography(store, document_id, root, base_path):
    """Load only this spine's local styles, including imports, in source order."""
    sources = []
    visiting = set()

    def load(url, base):
        parsed = urlsplit(url.strip())
        if parsed.scheme or parsed.netloc or not parsed.path:
            return
        candidate = posixpath.normpath(
            posixpath.join(posixpath.dirname(base), unquote(parsed.path))
        )
        if candidate.startswith(("../", "/")) or candidate in visiting:
            return
        visiting.add(candidate)
        try:
            resource = store.get_epub_resource(document_id, candidate)
        except KeyError:
            return
        if resource["media_type"] != "text/css":
            return
        data = bytes(resource["data"])
        try:
            css = data.decode("utf-8-sig")
        except UnicodeDecodeError:
            css = data.decode("latin-1")
        for match in _IMPORT.finditer(css):
            load(match.group(1) or match.group(2), candidate)
        sources.append(_IMPORT.sub("", css))
        visiting.remove(candidate)

    for element in root.iter():
        if (
            _local_name(element.tag) == "link"
            and "stylesheet" in element.attrib.get("rel", "").split()
        ):
            load(element.attrib.get("href", ""), base_path)
        elif _local_name(element.tag) == "style" and element.text:
            for match in _IMPORT.finditer(element.text):
                load(match.group(1) or match.group(2), base_path)
            sources.append(_IMPORT.sub("", element.text))
    return TranslationTypography(root, sources)
