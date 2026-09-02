from __future__ import annotations

import hashlib
import io
import mimetypes
import posixpath
import re
import zipfile
from dataclasses import dataclass
from html import escape
from xml.etree import ElementTree as ET

from jieyi.ingestion.epub import (
    EpubIngestionError,
    _local_name,
    _parse_xml,
    _resolve_member,
    _safe_member_name,
    _validate_archive,
)
from jieyi.ingestion.epub_structure import SourceAtom

_SPACE = re.compile(r"\s+")
_PATH_PART = re.compile(r"/([^/\[]+)\[(\d+)\]")
_XML_DECLARATION = re.compile(r"^\s*<\?xml[^>]*\?>", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class EpubResource:
    path: str
    media_type: str
    properties: str
    data: bytes
    sha256: str


@dataclass(frozen=True, slots=True)
class EpubSpineItem:
    index: int
    idref: str
    path: str
    media_type: str
    properties: str
    linear: bool
    fixed_layout: bool


@dataclass(frozen=True, slots=True)
class EpubAtomRecord:
    atom_id: str
    spine_index: int
    spine_path: str
    dom_path: str
    semantic_path: str
    ordinal: int
    source_text: str
    source_markup: str
    node_refs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EpubTextNodeRecord:
    node_id: str
    atom_id: str
    spine_path: str
    dom_path: str
    slot: str
    ordinal: int
    source_text: str


@dataclass(frozen=True, slots=True)
class EpubArchive:
    original_data: bytes
    source_hash: str
    package_path: str
    package_version: str
    nav_path: str | None
    cover_path: str | None
    rendition_layout: str
    page_progression_direction: str
    metadata: tuple[tuple[str, str], ...]
    resources: tuple[EpubResource, ...]
    spine: tuple[EpubSpineItem, ...]
    atoms: tuple[EpubAtomRecord, ...]
    text_nodes: tuple[EpubTextNodeRecord, ...]


def _metadata_values(package: ET.Element) -> dict[str, list[str]]:
    values: dict[str, list[str]] = {}
    for element in package.iter():
        tag = _local_name(element.tag)
        if tag == "meta":
            key = element.attrib.get("property") or element.attrib.get("name") or ""
            value = (element.text or element.attrib.get("content") or "").strip()
        elif tag in {"title", "creator", "language", "identifier", "publisher", "rights"}:
            key = tag
            value = "".join(element.itertext()).strip()
        else:
            continue
        if key and value:
            values.setdefault(key.casefold(), []).append(_SPACE.sub(" ", value))
    return values


def _path_map(root: ET.Element) -> dict[str, ET.Element]:
    body = next((item for item in root.iter() if _local_name(item.tag) == "body"), root)
    result: dict[str, ET.Element] = {}

    def walk(element: ET.Element, path: str) -> None:
        result[path] = element
        counts: dict[str, int] = {}
        for child in element:
            tag = _local_name(child.tag)
            counts[tag] = counts.get(tag, 0) + 1
            walk(child, f"{path}/{tag}[{counts[tag]}]")

    walk(body, f"/{_local_name(body.tag)}[1]")
    return result


def _inner_xml(element: ET.Element) -> str:
    pieces = [escape(element.text or "", quote=False)]
    for child in element:
        raw = ET.tostring(child, encoding="unicode", short_empty_elements=True)
        if child.tail and raw.endswith(escape(child.tail, quote=False)):
            raw = raw[: -len(escape(child.tail, quote=False))]
        pieces.append(raw)
        pieces.append(escape(child.tail or "", quote=False))
    return "".join(pieces).strip()


def _text_slots(
    element: ET.Element,
    *,
    spine_path: str,
    element_path: str,
) -> list[tuple[str, str, str, str]]:
    slots: list[tuple[str, str, str, str]] = []

    def walk(item: ET.Element, path: str) -> None:
        if item.text and item.text.strip():
            slots.append((f"{spine_path}#{path}::text", path, "text", item.text))
        counts: dict[str, int] = {}
        for child in item:
            tag = _local_name(child.tag)
            counts[tag] = counts.get(tag, 0) + 1
            child_path = f"{path}/{tag}[{counts[tag]}]"
            walk(child, child_path)
            if child.tail and child.tail.strip():
                slots.append((f"{spine_path}#{child_path}::tail", child_path, "tail", child.tail))

    walk(element, element_path)
    return slots


def _normal(value: str) -> str:
    return _SPACE.sub(" ", value).strip()


def _map_atoms(
    source_atoms: tuple[SourceAtom, ...],
    spine_paths: list[str],
    xhtml_roots: dict[str, ET.Element],
) -> tuple[tuple[EpubAtomRecord, ...], tuple[EpubTextNodeRecord, ...]]:
    path_indexes = {path: _path_map(root) for path, root in xhtml_roots.items()}
    spine_indexes = {path: index for index, path in enumerate(spine_paths)}
    path_counts: dict[tuple[str, str], int] = {}
    for atom in source_atoms:
        key = (atom.spine_path, atom.dom_path)
        path_counts[key] = path_counts.get(key, 0) + 1

    claimed: set[str] = set()
    atom_records: list[EpubAtomRecord] = []
    node_records: list[EpubTextNodeRecord] = []
    node_ordinal = 0
    for atom in source_atoms:
        element = path_indexes.get(atom.spine_path, {}).get(atom.dom_path)
        candidates = (
            _text_slots(element, spine_path=atom.spine_path, element_path=atom.dom_path)
            if element is not None
            else []
        )
        wanted = _normal(atom.text)
        selected: list[tuple[str, str, str, str]] = []
        combined = ""
        for candidate in candidates:
            node_id, _, _, source_text = candidate
            if node_id in claimed:
                continue
            value = _normal(source_text)
            if not value:
                continue
            trial = _normal(f"{combined} {value}")
            if value in wanted or wanted in value or wanted.startswith(trial):
                selected.append(candidate)
                combined = trial
                if combined == wanted or wanted in combined:
                    break
        if not selected and candidates:
            selected = [candidate for candidate in candidates if candidate[0] not in claimed]
        if not selected:
            selected = [(
                f"{atom.id}::fragment",
                atom.dom_path,
                "fragment",
                atom.text,
            )]

        refs: list[str] = []
        for node_id, dom_path, slot, source_text in selected:
            claimed.add(node_id)
            refs.append(node_id)
            node_records.append(
                EpubTextNodeRecord(
                    node_id=node_id,
                    atom_id=atom.id,
                    spine_path=atom.spine_path,
                    dom_path=dom_path,
                    slot=slot,
                    ordinal=node_ordinal,
                    source_text=source_text,
                )
            )
            node_ordinal += 1

        if element is not None and path_counts[(atom.spine_path, atom.dom_path)] == 1:
            markup = _inner_xml(element)
        else:
            markup = escape(atom.text, quote=False)
        atom_records.append(
            EpubAtomRecord(
                atom_id=atom.id,
                spine_index=spine_indexes.get(atom.spine_path, 0),
                spine_path=atom.spine_path,
                dom_path=atom.dom_path,
                semantic_path=atom.semantic_path,
                ordinal=atom.ordinal,
                source_text=atom.text,
                source_markup=markup or escape(atom.text, quote=False),
                node_refs=tuple(refs),
            )
        )
    return tuple(atom_records), tuple(node_records)


def parse_epub_archive(
    data: bytes,
    source_atoms: tuple[SourceAtom, ...] = (),
) -> EpubArchive:
    """Preserve an EPUB byte-for-byte and index every package resource and text address."""
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except (zipfile.BadZipFile, OSError) as exc:
        raise EpubIngestionError("File is not a valid EPUB/ZIP container") from exc

    with archive:
        members = _validate_archive(archive)
        container_path = "META-INF/container.xml"
        if container_path not in members:
            raise EpubIngestionError("EPUB is missing META-INF/container.xml")
        container = _parse_xml(archive.read(members[container_path]), container_path)
        package_path = next(
            (
                item.attrib.get("full-path", "")
                for item in container.iter()
                if _local_name(item.tag) == "rootfile" and item.attrib.get("full-path")
            ),
            "",
        )
        package_path = _safe_member_name(package_path)
        if package_path not in members:
            raise EpubIngestionError(f"EPUB package document is missing: {package_path}")
        package_data = archive.read(members[package_path])
        package = _parse_xml(package_data, package_path)
        package_dir = posixpath.dirname(package_path)
        package_version = package.attrib.get("version", "")
        metadata_values = _metadata_values(package)

        manifest: dict[str, dict[str, str]] = {}
        path_manifest: dict[str, dict[str, str]] = {}
        for item in package.iter():
            if _local_name(item.tag) != "item":
                continue
            item_id = item.attrib.get("id")
            href = item.attrib.get("href")
            if not item_id or not href:
                continue
            path = _resolve_member(package_dir, href)
            entry = {
                "id": item_id,
                "path": path,
                "media_type": item.attrib.get("media-type", ""),
                "properties": item.attrib.get("properties", ""),
            }
            manifest[item_id] = entry
            path_manifest[path] = entry

        nav_path = next(
            (
                item["path"]
                for item in manifest.values()
                if "nav" in item["properties"].split()
            ),
            None,
        )
        spine_element = next(
            (item for item in package.iter() if _local_name(item.tag) == "spine"),
            None,
        )
        if nav_path is None and spine_element is not None:
            toc_id = spine_element.attrib.get("toc", "")
            if toc_id in manifest:
                nav_path = manifest[toc_id]["path"]

        cover_path = next(
            (
                item["path"]
                for item in manifest.values()
                if "cover-image" in item["properties"].split()
            ),
            None,
        )
        if cover_path is None:
            cover_ids = metadata_values.get("cover", [])
            if cover_ids and cover_ids[0] in manifest:
                cover_path = manifest[cover_ids[0]]["path"]
        if cover_path is None:
            for reference in package.iter():
                if (
                    _local_name(reference.tag) == "reference"
                    and "cover" in reference.attrib.get("type", "").casefold().split()
                    and reference.attrib.get("href")
                ):
                    cover_path = _resolve_member(package_dir, reference.attrib["href"])
                    break
        cover_item = path_manifest.get(cover_path or "", {})
        if (
            cover_path
            and cover_path in members
            and (
                cover_item.get("media_type") in {"application/xhtml+xml", "text/html"}
                or cover_path.casefold().endswith((".xhtml", ".html", ".htm"))
            )
        ):
            cover_document = _parse_xml(archive.read(members[cover_path]), cover_path)
            image_href = next(
                (
                    value
                    for element in cover_document.iter()
                    if _local_name(element.tag) in {"img", "image"}
                    for key, value in element.attrib.items()
                    if _local_name(key) in {"src", "href"} and value
                ),
                None,
            )
            if image_href:
                cover_path = _resolve_member(posixpath.dirname(cover_path), image_href)

        rendition_layout = (
            metadata_values.get("rendition:layout", [""])[0]
            or ("pre-paginated" if metadata_values.get("fixed-layout", [""])[0].casefold() in {"true", "yes"} else "reflowable")
        )
        page_progression = (
            spine_element.attrib.get("page-progression-direction", "")
            if spine_element is not None
            else ""
        )

        spine_items: list[EpubSpineItem] = []
        spine_paths: list[str] = []
        if spine_element is not None:
            for itemref in spine_element:
                if _local_name(itemref.tag) != "itemref":
                    continue
                idref = itemref.attrib.get("idref", "")
                item = manifest.get(idref)
                if item is None:
                    continue
                properties = " ".join(
                    filter(None, [item["properties"], itemref.attrib.get("properties", "")])
                )
                fixed = (
                    rendition_layout == "pre-paginated"
                    or "rendition:layout-pre-paginated" in properties.split()
                )
                spine_items.append(
                    EpubSpineItem(
                        index=len(spine_items),
                        idref=idref,
                        path=item["path"],
                        media_type=item["media_type"],
                        properties=properties,
                        linear=itemref.attrib.get("linear", "yes").casefold() != "no",
                        fixed_layout=fixed,
                    )
                )
                spine_paths.append(item["path"])

        resources: list[EpubResource] = []
        xhtml_roots: dict[str, ET.Element] = {}
        for path, info in members.items():
            raw = archive.read(info)
            manifest_item = path_manifest.get(path, {})
            media_type = manifest_item.get("media_type") or mimetypes.guess_type(path)[0] or "application/octet-stream"
            properties = manifest_item.get("properties", "")
            resources.append(
                EpubResource(
                    path=path,
                    media_type=media_type,
                    properties=properties,
                    data=raw,
                    sha256=hashlib.sha256(raw).hexdigest(),
                )
            )
            if path in spine_paths and (
                media_type in {"application/xhtml+xml", "text/html"}
                or path.casefold().endswith((".xhtml", ".html", ".htm"))
            ):
                xhtml_roots[path] = _parse_xml(raw, path)

        atoms, text_nodes = _map_atoms(source_atoms, spine_paths, xhtml_roots)
        flat_metadata = tuple(
            (key, value)
            for key, values in sorted(metadata_values.items())
            for value in values
        )
        return EpubArchive(
            original_data=data,
            source_hash=hashlib.sha256(data).hexdigest(),
            package_path=package_path,
            package_version=package_version,
            nav_path=nav_path,
            cover_path=cover_path,
            rendition_layout=rendition_layout or "reflowable",
            page_progression_direction=page_progression,
            metadata=flat_metadata,
            resources=tuple(resources),
            spine=tuple(spine_items),
            atoms=atoms,
            text_nodes=text_nodes,
        )


def build_structured_source(atom_rows: list[dict]) -> str:
    """Wrap original DOM fragments so a model must return every atom independently."""
    return "".join(
        f'<jy-atom data-jy-id="{escape(str(row["atom_id"]), quote=True)}">'
        f'{row["source_markup"]}</jy-atom>'
        for row in atom_rows
    )


def parse_structured_translation(
    value: str,
    expected_atom_ids: tuple[str, ...],
) -> tuple[str, dict[str, tuple[str, str]]]:
    """Validate atom identity/order and return plain and inline-markup translations."""
    cleaned = _XML_DECLARATION.sub("", value).strip()
    try:
        root = ET.fromstring(f"<jy-root>{cleaned}</jy-root>")
    except ET.ParseError as exc:
        raise ValueError(f"Structured EPUB translation is not valid inline XML: {exc}") from exc
    if (root.text or "").strip() or any((item.tail or "").strip() for item in root):
        raise ValueError(
            "Structured EPUB translation contains text outside SourceAtom boundaries"
        )
    if any(item.tag != "jy-atom" for item in root):
        raise ValueError("Structured EPUB translation contains an unexpected outer element")
    if any(child.tag == "jy-atom" for item in root for child in item.iter() if child is not item):
        raise ValueError("Structured EPUB translation contains nested SourceAtom boundaries")
    items = list(root)
    found = tuple(item.attrib.get("data-jy-id", "") for item in items)
    if found != expected_atom_ids:
        raise ValueError(
            f"SourceAtom response mismatch; expected={expected_atom_ids}, found={found}"
        )
    translations: dict[str, tuple[str, str]] = {}
    plain_parts: list[str] = []
    for atom_id, item in zip(found, items, strict=True):
        markup = _inner_xml(item)
        plain = _normal("".join(item.itertext()))
        if not plain:
            raise ValueError(f"SourceAtom translation is empty: {atom_id}")
        translations[atom_id] = (plain, markup)
        plain_parts.append(plain)
    return " ".join(plain_parts), translations

