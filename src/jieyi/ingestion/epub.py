from __future__ import annotations

import io
import posixpath
import re
import zipfile
from dataclasses import dataclass
from html.entities import html5
from pathlib import PurePosixPath
from urllib.parse import unquote, urlsplit
from xml.etree import ElementTree as ET

from jieyi.ingestion.epub_navigation import (
    EpubNavigationEntry,
    apply_navigation_headings,
    parse_epub_navigation,
    position_navigation,
)
from jieyi.ingestion.epub_structure import (
    SEGMENTER_VERSION,
    BoundaryDecision,
    SourceAtom,
    extract_source_atoms,
    reflow_atoms,
)
from jieyi.ingestion.plaintext import ParsedBlock

_MAX_ENTRIES = 10_000
_MAX_TOTAL_UNCOMPRESSED = 512 * 1024 * 1024
_MAX_SINGLE_ENTRY = 64 * 1024 * 1024
_MAX_COMPRESSION_RATIO = 1_000
_SPACE = re.compile(r"\s+")
_NAMED_ENTITY = re.compile(rb"&([A-Za-z][A-Za-z0-9]+);")


class EpubIngestionError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class EpubBook:
    title: str
    blocks: tuple[ParsedBlock, ...]
    spine_items: tuple[str, ...]
    source_atoms: tuple[SourceAtom, ...] = ()
    boundaries: tuple[BoundaryDecision, ...] = ()
    navigation: tuple[EpubNavigationEntry, ...] = ()
    segmenter_version: str = SEGMENTER_VERSION


def _local_name(value: str) -> str:
    return value.rsplit("}", 1)[-1].rsplit(":", 1)[-1]


def _safe_member_name(name: str) -> str:
    path = PurePosixPath(name.replace("\\", "/"))
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise EpubIngestionError(f"Unsafe EPUB member path: {name}")
    return path.as_posix()


def _validate_archive(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    infos = archive.infolist()
    if len(infos) > _MAX_ENTRIES:
        raise EpubIngestionError("EPUB contains too many archive entries")

    total = 0
    safe: dict[str, zipfile.ZipInfo] = {}
    for info in infos:
        name = _safe_member_name(info.filename)
        if info.is_dir():
            continue
        if info.file_size > _MAX_SINGLE_ENTRY:
            raise EpubIngestionError(f"EPUB entry is too large: {name}")
        total += info.file_size
        if total > _MAX_TOTAL_UNCOMPRESSED:
            raise EpubIngestionError("EPUB uncompressed size exceeds the safety limit")
        if info.compress_size == 0 and info.file_size > 0:
            raise EpubIngestionError(f"Invalid compression metadata for EPUB entry: {name}")
        if info.compress_size and info.file_size > 1_000_000:
            ratio = info.file_size / info.compress_size
            if ratio > _MAX_COMPRESSION_RATIO:
                raise EpubIngestionError(f"Suspicious compression ratio for EPUB entry: {name}")
        if name in safe:
            raise EpubIngestionError(f"Duplicate EPUB member path: {name}")
        safe[name] = info
    return safe


def _parse_xml(data: bytes, label: str) -> ET.Element:
    upper = data.upper()
    if b"<!DOCTYPE" in upper or b"<!ENTITY" in upper:
        raise EpubIngestionError(f"DTD or entity declarations are not allowed in {label}")

    def replace_html_entity(match: re.Match[bytes]) -> bytes:
        name = match.group(1).decode("ascii")
        if name in {"amp", "apos", "gt", "lt", "quot"}:
            return match.group(0)
        value = html5.get(f"{name};")
        return value.encode("utf-8") if value is not None else match.group(0)

    data = _NAMED_ENTITY.sub(replace_html_entity, data)
    try:
        return ET.fromstring(data)
    except ET.ParseError as exc:
        raise EpubIngestionError(f"Invalid XML in {label}: {exc}") from exc


def _resolve_member(base_dir: str, href: str) -> str:
    path = unquote(urlsplit(href).path)
    combined = posixpath.normpath(posixpath.join(base_dir, path))
    return _safe_member_name(combined)


def _metadata_title(package: ET.Element) -> str:
    for element in package.iter():
        if _local_name(element.tag) == "title" and element.text and element.text.strip():
            return _SPACE.sub(" ", element.text).strip()
    return "Untitled EPUB"


def extract_epub(data: bytes) -> EpubBook:
    """Read an EPUB container in spine order without extracting files to disk."""
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
        rootfile_path = None
        for element in container.iter():
            if _local_name(element.tag) == "rootfile":
                rootfile_path = element.attrib.get("full-path")
                if rootfile_path:
                    break
        if not rootfile_path:
            raise EpubIngestionError("EPUB container does not declare a package document")
        rootfile_path = _safe_member_name(rootfile_path)
        if rootfile_path not in members:
            raise EpubIngestionError(f"EPUB package document is missing: {rootfile_path}")

        package = _parse_xml(archive.read(members[rootfile_path]), rootfile_path)
        package_dir = posixpath.dirname(rootfile_path)
        manifest: dict[str, dict[str, str]] = {}
        spine_ids: list[str] = []
        for element in package.iter():
            tag = _local_name(element.tag)
            if tag == "item" and element.attrib.get("id") and element.attrib.get("href"):
                manifest[element.attrib["id"]] = {
                    "path": _resolve_member(package_dir, element.attrib["href"]),
                    "media_type": element.attrib.get("media-type", ""),
                    "properties": element.attrib.get("properties", ""),
                }
            elif (
                tag == "itemref"
                and element.attrib.get("idref")
                and element.attrib.get("linear", "yes").casefold() != "no"
            ):
                spine_ids.append(element.attrib["idref"])

        if not spine_ids:
            raise EpubIngestionError("EPUB package has no readable spine")

        nav_path = next(
            (
                item["path"]
                for item in manifest.values()
                if "nav" in item["properties"].split()
            ),
            None,
        )
        if nav_path is None:
            spine_element = next(
                (element for element in package.iter() if _local_name(element.tag) == "spine"),
                None,
            )
            toc_id = spine_element.attrib.get("toc", "") if spine_element is not None else ""
            if toc_id in manifest:
                nav_path = manifest[toc_id]["path"]
        navigation: tuple[EpubNavigationEntry, ...] = ()
        if nav_path and nav_path in members:
            try:
                navigation = parse_epub_navigation(archive.read(members[nav_path]), nav_path)
            except ValueError:
                # A broken optional navigation document must not make readable spine
                # content impossible to import.
                navigation = ()

        headings: list[str] = []
        source_atoms: list[SourceAtom] = []
        spine_paths: list[str] = []
        xhtml_roots: dict[str, ET.Element] = {}
        css_sources: list[str] = []
        for item in manifest.values():
            if item["media_type"] != "text/css" or item["path"] not in members:
                continue
            raw_css = archive.read(members[item["path"]])
            try:
                css_sources.append(raw_css.decode("utf-8-sig"))
            except UnicodeDecodeError:
                css_sources.append(raw_css.decode("latin-1"))
        for item_id in spine_ids:
            item = manifest.get(item_id)
            if not item:
                raise EpubIngestionError(f"Spine references unknown manifest item: {item_id}")
            if "nav" in item["properties"].split():
                continue
            path = item["path"]
            media_type = item["media_type"]
            if media_type not in {"application/xhtml+xml", "text/html"} and not path.lower().endswith(
                (".xhtml", ".html", ".htm")
            ):
                continue
            if path not in members:
                raise EpubIngestionError(f"Spine document is missing: {path}")
            root = _parse_xml(archive.read(members[path]), path)
            xhtml_roots[path] = root
            inline_css = tuple(
                "".join(element.itertext())
                for element in root.iter()
                if _local_name(element.tag) == "style"
            )
            source_atoms.extend(
                extract_source_atoms(
                    root,
                    spine_path=path,
                    css_sources=tuple(css_sources) + inline_css,
                    start_ordinal=len(source_atoms),
                )
            )
            spine_paths.append(path)

        blocks, boundaries = reflow_atoms(source_atoms, headings)
        positioned_navigation = position_navigation(navigation, xhtml_roots, source_atoms)
        blocks = apply_navigation_headings(blocks, source_atoms, positioned_navigation)
        if not blocks:
            raise EpubIngestionError("EPUB spine contains no readable text blocks")
        return EpubBook(
            title=_metadata_title(package),
            blocks=tuple(blocks),
            spine_items=tuple(spine_paths),
            source_atoms=tuple(source_atoms),
            boundaries=tuple(boundaries),
            navigation=navigation,
        )
