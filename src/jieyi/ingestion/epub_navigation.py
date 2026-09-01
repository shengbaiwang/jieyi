from __future__ import annotations

import posixpath
import re
from collections.abc import Iterable
from dataclasses import dataclass, replace
from html.entities import html5
from typing import Any
from urllib.parse import unquote, urlsplit
from xml.etree import ElementTree as ET

from jieyi.ingestion.plaintext import ParsedBlock

_SPACE = re.compile(r"\s+")
_NAMED_ENTITY = re.compile(rb"&([A-Za-z][A-Za-z0-9]+);")


@dataclass(frozen=True, slots=True)
class EpubNavigationEntry:
    label: str
    path: str
    fragment: str
    level: int


@dataclass(frozen=True, slots=True)
class PositionedNavigationEntry:
    entry: EpubNavigationEntry
    atom: Any


def _local_name(value: str) -> str:
    return value.rsplit("}", 1)[-1].rsplit(":", 1)[-1].casefold()


def _normal(value: str) -> str:
    return _SPACE.sub(" ", value).strip()


def _strip_doctype(data: bytes) -> bytes:
    """Remove a declaration without ever evaluating its external or internal subset."""
    upper = data.upper()
    start = upper.find(b"<!DOCTYPE")
    if start < 0:
        return data
    quote: int | None = None
    subset_depth = 0
    for index in range(start + len(b"<!DOCTYPE"), len(data)):
        value = data[index]
        if quote is not None:
            if value == quote:
                quote = None
            continue
        if value in (ord("'"), ord('"')):
            quote = value
        elif value == ord("["):
            subset_depth += 1
        elif value == ord("]") and subset_depth:
            subset_depth -= 1
        elif value == ord(">") and subset_depth == 0:
            return data[:start] + data[index + 1 :]
    raise ValueError("Unterminated XML doctype declaration")


def parse_xml_resource(data: bytes, label: str) -> ET.Element:
    """Parse saved EPUB XML while permitting, but never loading, a standard NCX DTD."""
    if b"<!ENTITY" in data.upper():
        raise ValueError(f"Entity declarations are not allowed in {label}")
    data = _strip_doctype(data)

    def replace_html_entity(match: re.Match[bytes]) -> bytes:
        name = match.group(1).decode("ascii")
        if name in {"amp", "apos", "gt", "lt", "quot"}:
            return match.group(0)
        value = html5.get(f"{name};")
        return value.encode("utf-8") if value is not None else match.group(0)

    try:
        return ET.fromstring(_NAMED_ENTITY.sub(replace_html_entity, data))
    except ET.ParseError as exc:
        raise ValueError(f"Invalid XML in {label}: {exc}") from exc


def _resolved_target(nav_path: str, href: str) -> tuple[str, str]:
    parsed = urlsplit(href.strip())
    path = unquote(parsed.path)
    if path:
        path = posixpath.normpath(posixpath.join(posixpath.dirname(nav_path), path))
    else:
        path = nav_path
    if path.startswith(("/", "../")) or path == "..":
        return "", ""
    return path, unquote(parsed.fragment)


def _attribute_tokens(element: ET.Element, name: str) -> set[str]:
    return {
        token.casefold()
        for key, value in element.attrib.items()
        if _local_name(key) == name
        for token in re.split(r"\s+", value.strip())
        if token
    }


def _epub3_entries(root: ET.Element, nav_path: str) -> list[EpubNavigationEntry]:
    navs = [item for item in root.iter() if _local_name(item.tag) == "nav"]
    toc = next(
        (
            item
            for item in navs
            if "toc" in _attribute_tokens(item, "type")
            or "doc-toc" in _attribute_tokens(item, "role")
        ),
        navs[0] if navs else None,
    )
    if toc is None:
        return []

    entries: list[EpubNavigationEntry] = []

    def direct_children(element: ET.Element, tag: str) -> list[ET.Element]:
        return [child for child in element if _local_name(child.tag) == tag]

    def item_link(item: ET.Element) -> ET.Element | None:
        queue = list(item)
        while queue:
            child = queue.pop(0)
            if _local_name(child.tag) in {"ol", "ul"}:
                continue
            if _local_name(child.tag) in {"a", "span"}:
                return child
            queue[0:0] = list(child)
        return None

    def walk_list(list_element: ET.Element, level: int) -> None:
        for item in direct_children(list_element, "li"):
            link = item_link(item)
            href = link.attrib.get("href", "") if link is not None else ""
            label = _normal("".join(link.itertext())) if link is not None else ""
            if label and href:
                path, fragment = _resolved_target(nav_path, href)
                if path:
                    entries.append(EpubNavigationEntry(label, path, fragment, level))
            for child_list in direct_children(item, "ol") + direct_children(item, "ul"):
                walk_list(child_list, level + 1)

    for top_list in direct_children(toc, "ol") + direct_children(toc, "ul"):
        walk_list(top_list, 0)
    return entries


def _ncx_entries(root: ET.Element, nav_path: str) -> list[EpubNavigationEntry]:
    nav_map = next((item for item in root.iter() if _local_name(item.tag) == "navmap"), None)
    if nav_map is None:
        return []
    entries: list[EpubNavigationEntry] = []

    def walk(point: ET.Element, level: int) -> None:
        label = ""
        href = ""
        for child in point:
            tag = _local_name(child.tag)
            if tag == "navlabel":
                label = _normal("".join(child.itertext()))
            elif tag == "content":
                href = child.attrib.get("src", "")
        if label and href:
            path, fragment = _resolved_target(nav_path, href)
            if path:
                entries.append(EpubNavigationEntry(label, path, fragment, level))
        for child in point:
            if _local_name(child.tag) == "navpoint":
                walk(child, level + 1)

    for point in nav_map:
        if _local_name(point.tag) == "navpoint":
            walk(point, 0)
    return entries


def parse_epub_navigation(data: bytes, nav_path: str) -> tuple[EpubNavigationEntry, ...]:
    root = parse_xml_resource(data, nav_path)
    if _local_name(root.tag) == "ncx":
        return tuple(_ncx_entries(root, nav_path))
    return tuple(_epub3_entries(root, nav_path))


def _value(item: Any, name: str) -> Any:
    return item.get(name) if isinstance(item, dict) else getattr(item, name)


def _document_index(root: ET.Element) -> tuple[dict[str, int], dict[str, str]]:
    body = next((item for item in root.iter() if _local_name(item.tag) == "body"), root)
    order: dict[str, int] = {}
    fragments: dict[str, str] = {}

    def walk(element: ET.Element, path: str) -> None:
        order[path] = len(order)
        element_id = element.attrib.get("id")
        if element_id:
            fragments[element_id] = path
        counts: dict[str, int] = {}
        for child in element:
            tag = _local_name(child.tag)
            counts[tag] = counts.get(tag, 0) + 1
            walk(child, f"{path}/{tag}[{counts[tag]}]")

    walk(body, f"/{_local_name(body.tag)}[1]")
    return order, fragments


def position_navigation(
    entries: Iterable[EpubNavigationEntry],
    roots: dict[str, ET.Element],
    atoms: Iterable[Any],
) -> tuple[PositionedNavigationEntry, ...]:
    atoms_by_path: dict[str, list[Any]] = {}
    for atom in atoms:
        atoms_by_path.setdefault(str(_value(atom, "spine_path")), []).append(atom)
    for values in atoms_by_path.values():
        values.sort(key=lambda item: int(_value(item, "ordinal")))

    indexes = {path: _document_index(root) for path, root in roots.items()}
    positioned: list[PositionedNavigationEntry] = []
    for entry in entries:
        path_atoms = atoms_by_path.get(entry.path, [])
        if not path_atoms:
            continue
        if not entry.fragment:
            positioned.append(PositionedNavigationEntry(entry, path_atoms[0]))
            continue
        order, fragments = indexes.get(entry.path, ({}, {}))
        anchor_path = fragments.get(entry.fragment)
        if anchor_path is None:
            continue

        atom = next(
            (
                item
                for item in path_atoms
                if anchor_path == _value(item, "dom_path")
                or anchor_path == _value(item, "semantic_path")
            ),
            None,
        )
        ancestor_path = anchor_path
        while atom is None and "/" in ancestor_path[1:]:
            ancestor_path = ancestor_path.rsplit("/", 1)[0]
            atom = next(
                (item for item in path_atoms if _value(item, "dom_path") == ancestor_path),
                None,
            )
        if atom is None:
            atom = next(
                (
                    item
                    for item in path_atoms
                    if str(_value(item, "dom_path")).startswith(anchor_path + "/")
                ),
                None,
            )
        if atom is None and anchor_path in order:
            anchor_order = order[anchor_path]
            atom = next(
                (
                    item
                    for item in path_atoms
                    if order.get(str(_value(item, "dom_path")), -1) >= anchor_order
                ),
                None,
            )
        if atom is not None:
            positioned.append(PositionedNavigationEntry(entry, atom))
    return tuple(positioned)


def apply_navigation_headings(
    blocks: list[ParsedBlock],
    atoms: list[Any],
    positioned: Iterable[PositionedNavigationEntry],
) -> list[ParsedBlock]:
    atom_ordinals = {str(_value(atom, "id")): int(_value(atom, "ordinal")) for atom in atoms}
    boundaries = sorted(
        (
            int(_value(item.atom, "ordinal")),
            order,
            item.entry,
        )
        for order, item in enumerate(positioned)
    )
    if not boundaries:
        return blocks

    result: list[ParsedBlock] = []
    stack: list[str] = []
    boundary_index = 0
    for block in blocks:
        block_ordinal = min(
            (atom_ordinals[ref] for ref in block.source_refs if ref in atom_ordinals),
            default=-1,
        )
        while boundary_index < len(boundaries) and boundaries[boundary_index][0] <= block_ordinal:
            _, _, entry = boundaries[boundary_index]
            stack = stack[: entry.level]
            stack.append(entry.label)
            boundary_index += 1
        if stack:
            result.append(replace(block, heading_path=" / ".join(stack)))
        else:
            result.append(block)
    return result
