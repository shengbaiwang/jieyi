from __future__ import annotations

import base64
import hashlib
import io
import posixpath
import re
import zipfile
from collections import Counter
from urllib.parse import quote, unquote, urlsplit
from xml.etree import ElementTree as ET

from jieyi.ingestion.epub import EpubIngestionError, _local_name, _parse_xml, _safe_member_name

_DANGEROUS_ELEMENTS = {
    "script",
    "base",
    "iframe",
    "object",
    "embed",
    "applet",
    "form",
    "input",
    "button",
    "textarea",
    "select",
    "option",
    "foreignobject",
    "animate",
    "animatetransform",
    "set",
}
_URL_ATTRIBUTES = {"href", "src", "poster", "data", "action", "formaction"}
_CSS_IMPORT = re.compile(
    r"@import\s+(?:url\(\s*(['\"]?)(.*?)\1\s*\)|(['\"])(.*?)\3)"
    r"\s*([^;]*);?",
    re.IGNORECASE,
)
_CSS_URL = re.compile(r"url\(\s*(['\"]?)(.*?)\1\s*\)", re.IGNORECASE)
_CSS_DANGER = re.compile(
    r"(?:expression\s*\(|javascript\s*:|vbscript\s*:|behavior\s*:|-moz-binding\s*:)",
    re.IGNORECASE,
)
_SAFE_RESOURCE_TYPES = {
    "text/css",
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
    "image/avif",
    "image/svg+xml",
    "font/ttf",
    "font/otf",
    "font/woff",
    "font/woff2",
    "application/font-woff",
    "application/vnd.ms-opentype",
    "audio/mpeg",
    "audio/mp4",
    "audio/ogg",
}
_READER_STYLE = """
.jy-bilingual-pair {
  display: grid !important;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1fr) !important;
  align-items: start !important;
  column-gap: clamp(1.5rem, 5vw, 4rem) !important;
}
.jy-original,
.jy-translation {
  display: block !important;
  min-width: 0 !important;
  overflow-wrap: anywhere !important;
}
.jy-translation {
  margin: 0 !important;
  padding-inline-start: 1rem !important;
  border-inline-start: 1px solid rgba(40,100,190,.28) !important;
  color: inherit !important;
}
.jy-translation.jy-source-italic,
[data-jy-atoms].jy-source-italic,
.jy-translation :is(em, i),
[data-jy-atoms] :is(em, i) {
  font-family: "Kaiti SC", STKaiti, KaiTi, "楷体", serif !important;
  font-style: normal !important;
  font-synthesis: none !important;
}
.jy-bilingual-pair > :is(.jy-original, .jy-translation) > :first-child { margin-block-start: 0 !important; }
.jy-bilingual-pair > :is(.jy-original, .jy-translation) > :last-child { margin-block-end: 0 !important; }
.jy-translation.jy-missing-translation {
  color: #6f7782 !important;
  font-style: italic !important;
}
.jy-missing-translation { opacity: .55 !important; }
"""
_COMFORT_STYLE = """
html, body {
  width: auto !important; height: auto !important; overflow: visible !important;
  writing-mode: horizontal-tb !important;
}
body {
  max-width: 48rem !important; margin: 0 auto !important;
  padding: clamp(1.25rem, 3vw, 2rem) clamp(1.5rem, 5vw, 3.5rem) !important;
  color: #1d2025 !important; background: #fff !important;
  font-size: 18px !important; line-height: 1.75 !important;
  text-rendering: optimizeLegibility; -webkit-font-smoothing: antialiased;
}
body :is(p, li, blockquote, dd, dt, figcaption, td, th) { color: inherit !important; line-height: inherit !important; }
body :is(h1, h2, h3, h4, h5, h6) { color: #121417 !important; line-height: 1.3 !important; }
a { color: #185f9f !important; }
[style*="position:absolute"], [style*="position: absolute"],
[style*="position:fixed"], [style*="position: fixed"] {
  position: static !important; inset: auto !important;
  width: auto !important; height: auto !important;
  transform: none !important; overflow: visible !important;
}
img, svg, table { max-width: 100% !important; height: auto !important; }
.jy-translation { overflow: visible !important; }
"""
_FAITHFUL_FIXED_STYLE = """
.jy-translation, [data-jy-atoms] {
  max-height: 38vh !important;
  overflow: auto !important;
  overflow-wrap: anywhere !important;
  background: rgba(255,255,255,.88) !important;
}
"""
_RESIZE_SCRIPT = (
    """(()=>{let pending=false;const root=document.documentElement;"""
    """const documentId=root.getAttribute("data-jy-document-id");"""
    """const spineIndex=Number(root.getAttribute("data-jy-spine-index"));"""
    """const send=()=>{pending=false;const body=document.body;const height=body?Math.max(body.offsetHeight,Math.ceil(body.getBoundingClientRect().height)):root.offsetHeight;parent.postMessage({type:"jy-epub-resize",documentId,spineIndex,height:Math.ceil(height)},"*");};"""
    """const queue=()=>{if(pending)return;pending=true;requestAnimationFrame(send);};"""
    """addEventListener("message",event=>{const data=event.data||{};const ordinal=Number(data.segmentOrdinal);if(data.type!=="jy-epub-locate"||data.documentId!==documentId||Number(data.spineIndex)!==spineIndex||!Number.isInteger(ordinal))return;const target=document.querySelector('[data-jy-segment-ordinals~="'+ordinal+'"]');if(!target)return;const top=target.getBoundingClientRect().top+(window.scrollY||0);parent.postMessage({type:"jy-epub-location",documentId,spineIndex,segmentOrdinal:ordinal,top:Math.max(0,Math.round(top))},"*");});"""
    """addEventListener("load",queue);if(document.fonts){document.fonts.ready.then(queue);}if("ResizeObserver" in window){new ResizeObserver(queue).observe(document.body||document.documentElement);}queue();})();"""
)
_RESIZE_SCRIPT_HASH = base64.b64encode(
    hashlib.sha256(_RESIZE_SCRIPT.encode("utf-8")).digest()
).decode("ascii")


def _resource_path(base_path: str, raw_url: str) -> str | None:
    parsed = urlsplit(raw_url.strip())
    if parsed.scheme or parsed.netloc:
        return None
    if parsed.path == "":
        return None
    candidate = posixpath.normpath(
        posixpath.join(posixpath.dirname(base_path), unquote(parsed.path))
    )
    try:
        return _safe_member_name(candidate)
    except EpubIngestionError:
        return None


def _resource_url(document_id: str, path: str, fragment: str = "") -> str:
    route = (
        "content"
        if path.casefold().endswith((".xhtml", ".html", ".htm"))
        else "resources"
    )
    value = (
        f"/documents/{quote(document_id, safe='')}/epub/{route}/"
        f"{quote(path, safe='/')}"
    )
    return f"{value}#{quote(fragment, safe='')}" if fragment else value


def sanitize_css(source: str, *, document_id: str, base_path: str) -> str:
    source = _CSS_DANGER.sub("", source)

    def replace_import(match: re.Match[str]) -> str:
        raw = (match.group(2) or match.group(4) or "").strip()
        path = _resource_path(base_path, raw)
        if path is None or not path.casefold().endswith(".css"):
            return ""
        media = (match.group(5) or "").strip()
        suffix = f" {media}" if media else ""
        return f'@import url("{_resource_url(document_id, path)}"){suffix};'

    source = _CSS_IMPORT.sub(replace_import, source)

    def replace_url(match: re.Match[str]) -> str:
        raw = match.group(2).strip()
        internal_prefix = f"/documents/{quote(document_id, safe='')}/epub/"
        if raw.startswith(internal_prefix):
            return f'url("{raw}")'
        parsed = urlsplit(raw)
        if parsed.scheme == "data":
            if raw.casefold().startswith(
                ("data:image/", "data:font/", "data:application/font", "data:application/vnd.ms-opentype")
            ):
                return f'url("{raw}")'
            return "url()"
        path = _resource_path(base_path, raw)
        if path is None:
            return "url()"
        return f'url("{_resource_url(document_id, path, parsed.fragment)}")'

    return _CSS_URL.sub(replace_url, source)


def _path_map(root: ET.Element) -> dict[str, ET.Element]:
    body = next((item for item in root.iter() if _local_name(item.tag) == "body"), root)
    result: dict[str, ET.Element] = {}

    def walk(element: ET.Element, path: str) -> None:
        result[path] = element
        counts: Counter[str] = Counter()
        for child in element:
            tag = _local_name(child.tag)
            counts[tag] += 1
            walk(child, f"{path}/{tag}[{counts[tag]}]")

    walk(body, f"/{_local_name(body.tag)}[1]")
    return result


def _remove_element(parent: ET.Element, child: ET.Element) -> None:
    tail = child.tail
    index = list(parent).index(child)
    parent.remove(child)
    if tail:
        if index and len(parent):
            previous = list(parent)[index - 1]
            previous.tail = (previous.tail or "") + tail
        else:
            parent.text = (parent.text or "") + tail


def _sanitize_tree(root: ET.Element, *, document_id: str, base_path: str) -> ET.Element:
    parents = {child: parent for parent in root.iter() for child in parent}
    for element in list(root.iter()):
        tag = _local_name(element.tag)
        parent = parents.get(element)
        if tag in _DANGEROUS_ELEMENTS and parent is not None:
            _remove_element(parent, element)
            continue
        for key in list(element.attrib):
            local = _local_name(key)
            value = element.attrib[key]
            if local.startswith("on") or local in {
                "srcdoc",
                "srcset",
                "nonce",
                "integrity",
                "base",
            }:
                del element.attrib[key]
                continue
            if local == "style":
                cleaned = sanitize_css(value, document_id=document_id, base_path=base_path)
                if cleaned:
                    element.attrib[key] = cleaned
                else:
                    del element.attrib[key]
                continue
            if local not in _URL_ATTRIBUTES and local != "href":
                continue
            parsed = urlsplit(value.strip())
            internal_prefixes = tuple(
                f"/documents/{quote(document_id, safe='')}/epub/{route}/"
                for route in ("resources", "content")
            )
            if value.startswith(internal_prefixes):
                continue
            if value.startswith("#"):
                element.attrib[key] = "#" + quote(value[1:], safe="")
                continue
            if parsed.scheme == "data" and local in {"src", "poster"}:
                if value.casefold().startswith(("data:image/", "data:font/")):
                    continue
                del element.attrib[key]
                continue
            path = _resource_path(base_path, value)
            if path is None:
                del element.attrib[key]
            else:
                element.attrib[key] = _resource_url(document_id, path, parsed.fragment)
        if tag == "style" and element.text:
            element.text = sanitize_css(
                element.text,
                document_id=document_id,
                base_path=base_path,
            )
        if (
            tag == "meta"
            and element.attrib.get("http-equiv", "").casefold() == "refresh"
            and parent is not None
        ):
            _remove_element(parent, element)
    return root


def _element_namespace(element: ET.Element) -> str:
    return element.tag.split("}", 1)[0] + "}" if "}" in element.tag else ""


def _adopt_namespace(element: ET.Element, namespace: str) -> None:
    if namespace and "}" not in element.tag:
        element.tag = namespace + element.tag
    for child in element:
        _adopt_namespace(child, namespace)


def _fragment(markup: str, *, namespace: str = "") -> tuple[str, list[ET.Element]]:
    try:
        root = ET.fromstring(f"<jy-root>{markup}</jy-root>")
    except ET.ParseError:
        root = ET.Element("jy-root")
        root.text = markup
    children = list(root)
    for child in children:
        _adopt_namespace(child, namespace)
    return root.text or "", children


def _replace_inner(element: ET.Element, markup: str) -> None:
    for child in list(element):
        element.remove(child)
    text, children = _fragment(markup, namespace=_element_namespace(element))
    element.text = text
    for child in children:
        element.append(child)


def _replace_text_nodes(
    paths: dict[str, ET.Element],
    group: list[dict],
    node_rows: dict[str, dict],
) -> None:
    for atom in group:
        refs = [node_rows[ref] for ref in atom["node_refs"] if ref in node_rows]
        if not refs:
            continue
        for index, node in enumerate(refs):
            element = paths.get(node["dom_path"])
            if element is None:
                continue
            value = atom["translation_text"] if index == 0 else ""
            if node["slot"] == "tail":
                element.tail = value
            elif node["slot"] == "text" or index == 0:
                element.text = value


def _pair_translation(
    element: ET.Element,
    markup: str,
    atom_ids: list[str],
    *,
    missing: bool = False,
    source_italic: bool = False,
) -> None:
    namespace = _element_namespace(element)
    original = ET.Element(namespace + "span", {"class": "jy-original"})
    original.text = element.text
    element.text = None
    for child in list(element):
        element.remove(child)
        original.append(child)

    translated = ET.Element(
        namespace + "span",
        {
            "class": "jy-translation"
            + (" jy-source-italic" if source_italic else "")
            + (" jy-missing-translation" if missing else ""),
            "lang": "zh",
            "data-jy-for": " ".join(atom_ids),
        },
    )
    text, children = _fragment(markup, namespace=namespace)
    translated.text = text
    for child in children:
        translated.append(child)
    element.attrib["class"] = (
        element.attrib.get("class", "") + " jy-bilingual-pair"
    ).strip()
    element.append(original)
    element.append(translated)


def _source_uses_italic(atoms: list[dict]) -> bool:
    """Return whether the source styling marks any atom as italic or oblique."""
    for atom in atoms:
        fields = str(atom.get("style_signature") or "").split("\x1f")
        font_style = fields[6].strip().casefold() if len(fields) > 6 else ""
        if font_style.startswith(("italic", "oblique")):
            return True
    return False


def _inject_style(root: ET.Element, css: str) -> None:
    namespace = _element_namespace(root)
    head = next((item for item in root.iter() if _local_name(item.tag) == "head"), None)
    if head is None:
        head = ET.Element(namespace + "head")
        root.insert(0, head)
    style = ET.Element(
        _element_namespace(head) + "style",
        {"type": "text/css", "data-jy-reader": "true"},
    )
    style.text = css
    head.append(style)


def _inject_resize_reporter(
    root: ET.Element, *, document_id: str, spine_index: int
) -> None:
    root.attrib["data-jy-document-id"] = document_id
    root.attrib["data-jy-spine-index"] = str(spine_index)
    namespace = _element_namespace(root)
    head = next((item for item in root.iter() if _local_name(item.tag) == "head"), None)
    if head is None:
        head = ET.Element(namespace + "head")
        root.insert(0, head)
    namespace = _element_namespace(head)
    script = ET.Element(
        f"{namespace}script",
        {"type": "application/javascript", "data-jy-reader-resize": "true"},
    )
    script.text = _RESIZE_SCRIPT
    head.append(script)


def render_spine(
    store,
    document_id: str,
    spine_index: int,
    *,
    mode: str,
    layout: str,
) -> tuple[bytes, str]:
    if mode not in {"original", "translated", "bilingual"}:
        raise ValueError("mode must be original, translated, or bilingual")
    if layout not in {"faithful", "comfort"}:
        raise ValueError("layout must be faithful or comfort")
    spine = store.list_epub_spine(document_id)
    item = next((entry for entry in spine if entry["spine_index"] == spine_index), None)
    if item is None:
        raise KeyError(f"EPUB spine item not found: {spine_index}")
    resource = store.get_epub_resource(document_id, item["path"])
    root = _parse_xml(bytes(resource["data"]), item["path"])
    paths = _path_map(root)
    root = _sanitize_tree(root, document_id=document_id, base_path=item["path"])

    atoms = (
        store.list_epub_atoms_for_spine(document_id, spine_index)
        if mode != "original"
        else store.list_epub_locations_for_spine(document_id, spine_index)
    )
    grouped: dict[str, list[dict]] = {}
    for atom in atoms:
        grouped.setdefault(atom["dom_path"], []).append(atom)
    for dom_path, group in grouped.items():
        element = paths.get(dom_path)
        if element is None:
            continue
        ordinals = sorted({int(atom["segment_ordinal"]) for atom in group})
        element.attrib["data-jy-segment-ordinals"] = " ".join(map(str, ordinals))

    if mode != "original":
        node_rows = {
            item["node_id"]: item
            for item in store.list_epub_text_nodes_for_spine(
                document_id, spine_index
            )
        }
        for dom_path, group in grouped.items():
            element = paths.get(dom_path)
            if element is None:
                continue
            translated = [atom for atom in group if atom.get("translation_text")]
            if not translated:
                if mode == "translated":
                    element.attrib["class"] = (
                        element.attrib.get("class", "") + " jy-missing-translation"
                    ).strip()
                    missing = [dict(atom) for atom in group]
                    missing[0]["translation_text"] = "〔尚未翻译〕"
                    _replace_text_nodes(paths, missing, node_rows)
                else:
                    _pair_translation(
                        element,
                        "〔尚未翻译〕",
                        [atom["atom_id"] for atom in group],
                        missing=True,
                    )
                continue
            markup = " ".join(atom["translation_markup"] for atom in translated)
            source_italic = _source_uses_italic(translated)
            if mode == "translated":
                if len(group) == 1:
                    _replace_inner(element, markup)
                else:
                    _replace_text_nodes(paths, group, node_rows)
                element.attrib["lang"] = "zh"
                if source_italic:
                    element.attrib["class"] = (
                        element.attrib.get("class", "") + " jy-source-italic"
                    ).strip()
                element.attrib["data-jy-atoms"] = " ".join(
                    atom["atom_id"] for atom in translated
                )
            else:
                _pair_translation(
                    element,
                    markup,
                    [atom["atom_id"] for atom in translated],
                    source_italic=source_italic,
                )

    root = _sanitize_tree(root, document_id=document_id, base_path=item["path"])
    css = _READER_STYLE
    if layout == "comfort" or not item["fixed_layout"]:
        css += _COMFORT_STYLE
    elif item["fixed_layout"] and mode != "original":
        css += _FAITHFUL_FIXED_STYLE
    _inject_style(root, css)
    if not item["fixed_layout"]:
        _inject_resize_reporter(
            root, document_id=document_id, spine_index=spine_index
        )
    payload = ET.tostring(
        root,
        encoding="utf-8",
        xml_declaration=True,
        short_empty_elements=True,
    )
    return payload, item["media_type"] or "application/xhtml+xml"


def _render_export_spine(store, document_id: str, spine_index: int) -> bytes:
    """Render target text into the original XHTML without reader-only rewrites."""
    spine = store.list_epub_spine(document_id)
    item = next((entry for entry in spine if entry["spine_index"] == spine_index), None)
    if item is None:
        raise KeyError(f"EPUB spine item not found: {spine_index}")
    resource = store.get_epub_resource(document_id, item["path"])
    root = _parse_xml(bytes(resource["data"]), item["path"])
    paths = _path_map(root)

    atoms = store.list_epub_atoms_for_spine(document_id, spine_index)
    node_rows = {
        row["node_id"]: row
        for row in store.list_epub_text_nodes_for_spine(document_id, spine_index)
    }
    grouped: dict[str, list[dict]] = {}
    for atom in atoms:
        grouped.setdefault(atom["dom_path"], []).append(atom)
    for dom_path, group in grouped.items():
        element = paths.get(dom_path)
        if element is None:
            continue
        translated = [atom for atom in group if atom.get("translation_text")]
        if not translated:
            missing = [dict(atom) for atom in group]
            if missing:
                missing[0]["translation_text"] = "〔尚未翻译〕"
                _replace_text_nodes(paths, missing, node_rows)
            continue
        markup = " ".join(atom["translation_markup"] for atom in translated)
        if len(group) == 1:
            _replace_inner(element, markup)
        else:
            _replace_text_nodes(paths, group, node_rows)

    project = store.get_project_for_document(document_id)
    root.attrib["lang"] = project.target_lang
    xml_lang = "{http://www.w3.org/XML/1998/namespace}lang"
    if xml_lang in root.attrib:
        root.attrib[xml_lang] = project.target_lang
    return ET.tostring(
        root,
        encoding="utf-8",
        xml_declaration=True,
        short_empty_elements=True,
    )


def export_translated_epub(store, document_id: str) -> bytes:
    """Build an EPUB with translated spine documents and original book assets."""
    original = store.get_original_epub(document_id)
    spine_by_path = {
        item["path"]: item["spine_index"]
        for item in store.list_epub_spine(document_id)
    }
    output = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(original)) as source, zipfile.ZipFile(output, "w") as target:
        entries = source.infolist()
        entries.sort(key=lambda item: item.filename != "mimetype")
        for info in entries:
            data = source.read(info)
            if info.filename in spine_by_path:
                data = _render_export_spine(store, document_id, spine_by_path[info.filename])
            if info.filename == "mimetype":
                target.writestr(info, data, compress_type=zipfile.ZIP_STORED)
            else:
                target.writestr(info, data)
    return output.getvalue()


def render_content_path(
    store,
    document_id: str,
    path: str,
) -> tuple[bytes, str]:
    spine = store.list_epub_spine(document_id)
    item = next((entry for entry in spine if entry["path"] == path), None)
    if item is not None:
        return render_spine(
            store,
            document_id,
            item["spine_index"],
            mode="original",
            layout="faithful",
        )
    resource = store.get_epub_resource(document_id, path)
    media_type = str(resource["media_type"] or "").split(";", 1)[0]
    if media_type not in {"application/xhtml+xml", "text/html"}:
        raise PermissionError(f"EPUB content is not XHTML: {media_type}")
    root = _parse_xml(bytes(resource["data"]), path)
    root = _sanitize_tree(root, document_id=document_id, base_path=path)
    _inject_style(root, _READER_STYLE)
    payload = ET.tostring(
        root,
        encoding="utf-8",
        xml_declaration=True,
        short_empty_elements=True,
    )
    return payload, media_type


def safe_resource(store, document_id: str, path: str) -> tuple[bytes, str]:
    resource = store.get_epub_resource(document_id, path)
    media_type = str(resource["media_type"] or "application/octet-stream").split(";", 1)[0]
    if media_type not in _SAFE_RESOURCE_TYPES:
        raise PermissionError(f"EPUB resource type is not renderable: {media_type}")
    data = bytes(resource["data"])
    if media_type == "text/css":
        try:
            source = data.decode("utf-8-sig")
        except UnicodeDecodeError:
            source = data.decode("latin-1")
        data = sanitize_css(
            source,
            document_id=document_id,
            base_path=path,
        ).encode("utf-8")
    elif media_type == "image/svg+xml":
        root = _parse_xml(data, path)
        root = _sanitize_tree(root, document_id=document_id, base_path=path)
        data = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    return data, media_type

def reader_csp(resource_origin: str = "", *, allow_resize_script: bool = False) -> str:
    origin = resource_origin.rstrip("/")
    same_book = f"'self' {origin}".strip() if origin.startswith(("http://", "https://")) else "'self'"
    sandbox = "sandbox allow-scripts" if allow_resize_script else "sandbox"
    scripts = (
        f"script-src 'sha256-{_RESIZE_SCRIPT_HASH}'"
        if allow_resize_script
        else "script-src 'none'"
    )
    return (
        f"{sandbox}; default-src 'none'; {scripts}; connect-src 'none'; "
        f"img-src {same_book} data:; style-src {same_book} 'unsafe-inline'; "
        f"font-src {same_book} data:; media-src {same_book} data:; "
        "frame-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none'"
    )
