from __future__ import annotations

import re
from dataclasses import dataclass, replace
from itertools import pairwise
from xml.etree import ElementTree as ET

from jieyi.domain.models import SegmentKind
from jieyi.ingestion.plaintext import ParsedBlock

SEGMENTER_VERSION = "epub-structure-v1"

_SPACE = re.compile(r"\s+")
_CSS_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_CSS_RULE = re.compile(r"([^{}]+)\{([^{}]*)\}")
_DECLARATION = re.compile(r"([\w-]+)\s*:\s*([^;!]+)(?:\s*!important)?\s*(?:;|$)")
_TERMINAL = re.compile(r"[.!?。！？…][\"'’”»\)\]\}]*$")
_LOWERCASE_START = re.compile(r"^[\"'‘“«\(\[]*[a-zà-öø-ÿ]")
_UPPERCASE_START = re.compile(r"^[\"'‘“«\(\[]*[A-ZÀ-ÖØ-Þ]")
_LIST_MARKER = re.compile(r"^(?:[•◦▪‣–—-]|\(?\d{1,3}[.)]|[a-zA-Z][.)])\s+")

_BLOCK_DISPLAYS = {
    "block",
    "flex",
    "grid",
    "list-item",
    "table",
    "table-row",
    "table-cell",
}
_BLOCK_TAGS = {
    "address",
    "article",
    "aside",
    "blockquote",
    "body",
    "caption",
    "dd",
    "div",
    "dl",
    "dt",
    "figcaption",
    "figure",
    "footer",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "hr",
    "li",
    "main",
    "nav",
    "ol",
    "p",
    "pre",
    "section",
    "table",
    "tbody",
    "td",
    "tfoot",
    "th",
    "thead",
    "tr",
    "ul",
}
_SEMANTIC_TAGS = {
    "address",
    "blockquote",
    "caption",
    "dd",
    "dt",
    "figcaption",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "li",
    "p",
    "pre",
    "td",
    "th",
    "tr",
}
_NOTE_MARKERS = {"footnote", "endnote", "doc-footnote", "doc-endnote"}
_VERSE_MARKERS = {"poem", "poetry", "stanza", "verse", "verseline", "linegroup"}
_PAGEBREAK_MARKERS = {"pagebreak", "doc-pagebreak"}
_IGNORED_CONTENT_TAGS = {"script", "style", "noscript", "template"}


def _local_name(value: str) -> str:
    return value.rsplit("}", 1)[-1].rsplit(":", 1)[-1].lower()


def _tokens(value: str) -> set[str]:
    return {part.casefold() for part in re.split(r"[^\w-]+", value) if part}


def _epub_types(element: ET.Element) -> set[str]:
    values: set[str] = set()
    for key, value in element.attrib.items():
        if _local_name(key) == "type":
            values.update(_tokens(value))
    return values


def _normalise_text(value: str, *, preserve: bool = False) -> str:
    if preserve:
        return value.replace("\r\n", "\n").replace("\r", "\n").strip("\n")
    return _SPACE.sub(" ", value).strip()


def _parse_declarations(value: str) -> dict[str, str]:
    return {
        match.group(1).casefold(): match.group(2).strip().casefold()
        for match in _DECLARATION.finditer(value)
    }


@dataclass(frozen=True, slots=True)
class _CssRule:
    selector: str
    declarations: tuple[tuple[str, str], ...]
    specificity: tuple[int, int, int]
    order: int


def _selector_specificity(selector: str) -> tuple[int, int, int]:
    return (
        selector.count("#"),
        selector.count(".") + selector.count("["),
        sum(
            1
            for part in re.split(r"[\s>+~]+", selector)
            if re.match(r"^[a-zA-Z][\w-]*", part)
        ),
    )


def _simple_selector_matches(element: ET.Element, selector: str) -> bool:
    selector = re.sub(r"::?[\w-]+(?:\([^)]*\))?", "", selector.strip())
    if not selector or selector == "*":
        return True
    if any(character in selector for character in "[+~"):
        return False
    tag_match = re.match(r"^([a-zA-Z][\w-]*|\*)", selector)
    if (
        tag_match
        and tag_match.group(1) != "*"
        and _local_name(element.tag) != tag_match.group(1).casefold()
    ):
        return False
    element_id = element.attrib.get("id", "")
    ids = re.findall(r"#([\w-]+)", selector)
    if ids and element_id not in ids:
        return False
    classes = _tokens(element.attrib.get("class", ""))
    return all(name.casefold() in classes for name in re.findall(r"\.([\w-]+)", selector))


def _selector_matches(
    element: ET.Element, ancestors: tuple[ET.Element, ...], selector: str
) -> bool:
    # EPUB stylesheets overwhelmingly use tag/class/id and descendant selectors.
    # Unsupported sibling and attribute selectors are ignored rather than guessed.
    normalised_selector = re.sub(r"\s*>\s*", " ", selector.strip())
    parts = [part for part in re.split(r"\s+", normalised_selector) if part]
    if not parts or not _simple_selector_matches(element, parts[-1]):
        return False
    ancestor_index = len(ancestors) - 1
    for part in reversed(parts[:-1]):
        while ancestor_index >= 0 and not _simple_selector_matches(
            ancestors[ancestor_index], part
        ):
            ancestor_index -= 1
        if ancestor_index < 0:
            return False
        ancestor_index -= 1
    return True


class StyleResolver:
    def __init__(self, css_sources: tuple[str, ...] = ()):
        rules: list[_CssRule] = []
        order = 0
        for source in css_sources:
            cleaned = _CSS_COMMENT.sub("", source)
            for match in _CSS_RULE.finditer(cleaned):
                declarations = tuple(_parse_declarations(match.group(2)).items())
                if not declarations:
                    continue
                for selector in match.group(1).split(","):
                    selector = selector.strip()
                    if not selector or selector.startswith("@"):
                        continue
                    rules.append(
                        _CssRule(
                            selector=selector,
                            declarations=declarations,
                            specificity=_selector_specificity(selector),
                            order=order,
                        )
                    )
                    order += 1
        self.rules = tuple(rules)

    def computed(
        self, element: ET.Element, ancestors: tuple[ET.Element, ...]
    ) -> dict[str, str]:
        tag = _local_name(element.tag)
        result = {"display": "block" if tag in _BLOCK_TAGS else "inline"}
        matched = [
            rule for rule in self.rules if _selector_matches(element, ancestors, rule.selector)
        ]
        for rule in sorted(matched, key=lambda item: (item.specificity, item.order)):
            result.update(rule.declarations)
        result.update(_parse_declarations(element.attrib.get("style", "")))
        return result


@dataclass(frozen=True, slots=True)
class SourceAtom:
    id: str
    spine_path: str
    dom_path: str
    semantic_path: str
    parent_path: str
    ordinal: int
    tag: str
    text: str
    kind: SegmentKind
    role: str
    heading_level: int = 0
    class_tokens: tuple[str, ...] = ()
    style_signature: str = ""
    pagebreak_before: bool = False
    continuation_hint: bool = False


@dataclass(frozen=True, slots=True)
class BoundaryDecision:
    left_id: str
    right_id: str
    action: str
    confidence: float
    score: float
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Context:
    kind: SegmentKind = SegmentKind.PARAGRAPH
    role: str = "paragraph"
    semantic_path: str = ""
    heading_level: int = 0
    preserve_whitespace: bool = False


def _element_role(element: ET.Element, inherited: _Context) -> _Context:
    tag = _local_name(element.tag)
    markers = _epub_types(element) | _tokens(element.attrib.get("class", ""))
    if markers & _NOTE_MARKERS:
        return replace(inherited, kind=SegmentKind.FOOTNOTE, role="footnote")
    if tag == "blockquote":
        return replace(inherited, kind=SegmentKind.BLOCKQUOTE, role="blockquote")
    if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
        return replace(
            inherited,
            kind=SegmentKind.HEADING,
            role="heading",
            heading_level=int(tag[1]),
        )
    if tag in {"li", "dt", "dd"}:
        return replace(inherited, kind=SegmentKind.LIST_ITEM, role="list-item")
    if tag in {"td", "th"}:
        return replace(inherited, kind=SegmentKind.TABLE_CELL, role="table-cell")
    if tag == "tr":
        return replace(inherited, role="table-row")
    if tag in {"figcaption", "caption"}:
        return replace(inherited, kind=SegmentKind.CAPTION, role="caption")
    if tag == "address":
        return replace(inherited, role="address")
    if tag == "pre" or markers & _VERSE_MARKERS:
        return replace(
            inherited,
            kind=SegmentKind.VERSE,
            role="verse",
            preserve_whitespace=True,
        )
    return inherited


def extract_source_atoms(
    root: ET.Element,
    *,
    spine_path: str,
    css_sources: tuple[str, ...] = (),
    start_ordinal: int = 0,
) -> list[SourceAtom]:
    """Extract loss-aware text atoms while retaining structural provenance."""
    resolver = StyleResolver(css_sources)
    atoms: list[SourceAtom] = []
    pending_pagebreak = False

    def path_for(parent_path: str, element: ET.Element, index: int) -> str:
        return f"{parent_path}/{_local_name(element.tag)}[{index}]"

    def inline_text(
        element: ET.Element,
        ancestors: tuple[ET.Element, ...],
        path: str,
        context: _Context,
    ) -> str:
        nonlocal pending_pagebreak
        if _local_name(element.tag) in _IGNORED_CONTENT_TAGS:
            return ""
        markers = _epub_types(element) | _tokens(element.attrib.get("class", ""))
        style = resolver.computed(element, ancestors)
        if style.get("display") == "none" or style.get("visibility") == "hidden":
            return ""
        if markers & _PAGEBREAK_MARKERS:
            pending_pagebreak = True
            return ""
        if _local_name(element.tag) == "br":
            return "\n"
        pieces = [element.text or ""]
        counts: dict[str, int] = {}
        for child in element:
            child_tag = _local_name(child.tag)
            counts[child_tag] = counts.get(child_tag, 0) + 1
            child_path = path_for(path, child, counts[child_tag])
            child_style = resolver.computed(child, ancestors + (element,))
            child_markers = _epub_types(child) | _tokens(child.attrib.get("class", ""))
            if child_markers & _PAGEBREAK_MARKERS:
                pending_pagebreak = True
            elif child_style.get("display") != "none":
                child_value = inline_text(
                    child, ancestors + (element,), child_path, context
                )
                if child_style.get("display") in _BLOCK_DISPLAYS and child_value.strip():
                    if pieces and pieces[-1] and not pieces[-1].endswith((" ", "\n")):
                        pieces.append("\n")
                    pieces.append(child_value)
                    pieces.append("\n")
                else:
                    pieces.append(child_value)
            pieces.append(child.tail or "")
        return "".join(pieces)

    def emit(
        value: str,
        *,
        element: ET.Element,
        path: str,
        semantic_path: str,
        parent_path: str,
        context: _Context,
        style: dict[str, str],
        fragment_index: int,
    ) -> None:
        nonlocal pending_pagebreak
        text = _normalise_text(value, preserve=context.preserve_whitespace)
        if not text:
            return
        class_tokens = tuple(sorted(_tokens(element.attrib.get("class", ""))))
        signature_fields = (
            style.get("display", ""),
            style.get("text-align", ""),
            style.get("text-indent", ""),
            style.get("margin", ""),
            style.get("margin-top", ""),
            style.get("margin-bottom", ""),
            style.get("font-style", ""),
            style.get("font-weight", ""),
        )
        atom_id = f"{spine_path}#{path}::text[{fragment_index}]"
        atoms.append(
            SourceAtom(
                id=atom_id,
                spine_path=spine_path,
                dom_path=path,
                semantic_path=semantic_path or path,
                parent_path=parent_path,
                ordinal=start_ordinal + len(atoms),
                tag=_local_name(element.tag),
                text=text,
                kind=context.kind,
                role=context.role,
                heading_level=context.heading_level,
                class_tokens=class_tokens,
                style_signature="\x1f".join(signature_fields),
                pagebreak_before=pending_pagebreak,
                continuation_hint=bool(
                    set(class_tokens)
                    & {"continuation", "continued", "continue", "cont", "softbreak"}
                ),
            )
        )
        pending_pagebreak = False

    def walk(
        element: ET.Element,
        *,
        ancestors: tuple[ET.Element, ...],
        path: str,
        parent_path: str,
        inherited: _Context,
    ) -> None:
        nonlocal pending_pagebreak
        if _local_name(element.tag) in _IGNORED_CONTENT_TAGS:
            return
        style = resolver.computed(element, ancestors)
        markers = _epub_types(element) | _tokens(element.attrib.get("class", ""))
        if style.get("display") == "none" or style.get("visibility") == "hidden":
            return
        if markers & _PAGEBREAK_MARKERS:
            pending_pagebreak = True
            return

        context = _element_role(element, inherited)
        if _local_name(element.tag) in _SEMANTIC_TAGS:
            context = replace(context, semantic_path=path)
        semantic_path = context.semantic_path or path

        # Split mixed block content into fragments without dropping parent text or
        # child tails. Inline descendants stay inside the current fragment.
        buffer: list[str] = [element.text or ""]
        fragment_index = 0
        child_counts: dict[str, int] = {}

        def flush() -> None:
            nonlocal fragment_index
            emit(
                "".join(buffer),
                element=element,
                path=path,
                semantic_path=semantic_path,
                parent_path=parent_path,
                context=context,
                style=style,
                fragment_index=fragment_index,
            )
            fragment_index += 1
            buffer.clear()

        for child in element:
            child_tag = _local_name(child.tag)
            child_counts[child_tag] = child_counts.get(child_tag, 0) + 1
            child_path = path_for(path, child, child_counts[child_tag])
            child_style = resolver.computed(child, ancestors + (element,))
            child_markers = _epub_types(child) | _tokens(child.attrib.get("class", ""))
            child_is_block = child_style.get("display") in _BLOCK_DISPLAYS
            if child_markers & _PAGEBREAK_MARKERS:
                flush()
                pending_pagebreak = True
            elif child_is_block:
                flush()
                walk(
                    child,
                    ancestors=ancestors + (element,),
                    path=child_path,
                    parent_path=path,
                    inherited=context,
                )
            else:
                buffer.append(
                    inline_text(child, ancestors + (element,), child_path, context)
                )
            buffer.append(child.tail or "")
        flush()

    body = next((item for item in root.iter() if _local_name(item.tag) == "body"), root)
    root_tag = _local_name(body.tag)
    walk(
        body,
        ancestors=(),
        path=f"/{root_tag}[1]",
        parent_path="",
        inherited=_Context(),
    )
    return atoms


def _unclosed_delimiter(text: str) -> bool:
    pairs = (("(", ")"), ("[", "]"), ("{", "}"), ("«", "»"), ("“", "”"))
    return any(text.count(opening) > text.count(closing) for opening, closing in pairs)


def _boundary(left: SourceAtom, right: SourceAtom) -> BoundaryDecision:
    reasons: list[str] = []
    score = 0.0

    if left.semantic_path == right.semantic_path:
        return BoundaryDecision(
            left.id,
            right.id,
            "merge",
            0.99,
            10.0,
            ("same-semantic-container",),
        )

    protected_roles = {"list-item", "table-cell", "table-row", "verse", "footnote"}
    if left.role in protected_roles or right.role in protected_roles:
        return BoundaryDecision(
            left.id,
            right.id,
            "split",
            0.99,
            -10.0,
            ("protected-structural-role",),
        )

    if left.kind != right.kind:
        return BoundaryDecision(
            left.id,
            right.id,
            "split",
            0.99,
            -10.0,
            ("different-semantic-kinds",),
        )

    if left.kind is SegmentKind.HEADING:
        combined = len(left.text) + len(right.text)
        if combined <= 180 and left.spine_path == right.spine_path:
            return BoundaryDecision(
                left.id,
                right.id,
                "merge",
                0.90,
                6.0,
                ("adjacent-composite-heading",),
            )
        return BoundaryDecision(
            left.id, right.id, "split", 0.95, -6.0, ("separate-headings",)
        )

    if left.parent_path == right.parent_path:
        score += 0.75
        reasons.append("same-parent")
    if left.class_tokens and left.class_tokens == right.class_tokens:
        score += 0.5
        reasons.append("same-class")
    if left.style_signature and left.style_signature == right.style_signature:
        score += 0.25
        reasons.append("same-style")
    if left.continuation_hint or right.continuation_hint:
        score += 2.5
        reasons.append("publisher-continuation-hint")
    if right.pagebreak_before:
        score += 2.0
        reasons.append("pagebreak-continuation")
    if left.spine_path != right.spine_path:
        score -= 1.0
        reasons.append("cross-spine")

    stripped_left = left.text.rstrip()
    stripped_right = right.text.lstrip()
    if _TERMINAL.search(stripped_left):
        score -= 4.0
        reasons.append("terminal-punctuation")
    elif stripped_left.endswith((",", ";", ":", "—", "–")):
        score += 2.5
        reasons.append("continuation-punctuation")
    else:
        score += 0.75
        reasons.append("nonterminal-ending")

    if _LOWERCASE_START.search(stripped_right):
        score += 3.0
        reasons.append("lowercase-continuation")
    elif _UPPERCASE_START.search(stripped_right):
        score -= 1.5
        reasons.append("uppercase-start")
    if _LIST_MARKER.search(stripped_right):
        score -= 4.0
        reasons.append("list-marker")
    if _unclosed_delimiter(stripped_left):
        score += 2.0
        reasons.append("unclosed-delimiter")
    if stripped_left.endswith(("-", "‐", "‑")) and _LOWERCASE_START.search(stripped_right):
        score += 2.5
        reasons.append("hyphenated-continuation")

    action = "merge" if score >= 6.5 else "split"
    distance = abs(score - 6.5)
    confidence = min(0.98, 0.55 + distance / 10.0)
    return BoundaryDecision(left.id, right.id, action, confidence, score, tuple(reasons))


def _join_text(left: str, right: str, left_atom: SourceAtom, right_atom: SourceAtom) -> str:
    if not left:
        return right
    if not right:
        return left
    if left_atom.role in {"address", "verse"}:
        return f"{left.rstrip()}\n{right.lstrip()}"
    if left_atom.kind is SegmentKind.HEADING:
        return f"{left.rstrip()}\n{right.lstrip()}"
    if left.rstrip().endswith(("-", "‐", "‑")) and _LOWERCASE_START.search(right.lstrip()):
        return f"{left.rstrip()[:-1]}{right.lstrip()}"
    return f"{left.rstrip()} {right.lstrip()}"


def reflow_atoms(
    atoms: list[SourceAtom], headings: list[str]
) -> tuple[list[ParsedBlock], list[BoundaryDecision]]:
    """Reconstruct conservative semantic units with explainable boundaries."""
    if not atoms:
        return [], []
    decisions = [_boundary(left, right) for left, right in pairwise(atoms)]
    groups: list[list[SourceAtom]] = [[atoms[0]]]
    group_decisions: list[list[BoundaryDecision]] = [[]]
    for atom, decision in zip(atoms[1:], decisions):
        if decision.action == "merge":
            groups[-1].append(atom)
            group_decisions[-1].append(decision)
        else:
            groups.append([atom])
            group_decisions.append([])

    blocks: list[ParsedBlock] = []
    for group, joins in zip(groups, group_decisions):
        first = group[0]
        text = first.text
        for previous, current in pairwise(group):
            text = _join_text(text, current.text, previous, current)
        if first.kind is SegmentKind.HEADING:
            level = first.heading_level or 1
            del headings[level - 1 :]
            headings.append(text.replace("\n", " "))
        reasons = sorted({reason for item in joins for reason in item.reasons})
        confidence = min((item.confidence for item in joins), default=1.0)
        blocks.append(
            ParsedBlock(
                kind=first.kind,
                text=text,
                heading_path=" / ".join(headings),
                source_refs=tuple(item.id for item in group),
                segmentation_confidence=confidence,
                segmentation_reason=",".join(reasons) or "single-structural-block",
                segmenter_version=SEGMENTER_VERSION,
            )
        )
    return blocks, decisions
