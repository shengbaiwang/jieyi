"""Translate native PDF text objects while retaining original graphics and page geometry."""

from __future__ import annotations

import io
import os
import unicodedata
from collections import defaultdict
from ctypes import c_float, c_void_p, cast
from pathlib import Path
from threading import RLock

import pypdfium2 as pdfium
from pypdf import PdfReader, PdfWriter
from pypdf.generic import NameObject, StreamObject
from pypdfium2 import raw
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas

from jieyi.ingestion.pdf import _RENDER_LOCK, extract_pdf

_EXPORT_LOCK = RLock()


def _font():
    name = "JieyiCJK"
    if name in pdfmetrics.getRegisteredFontNames():
        return name
    candidates = [
        os.getenv("JIEYI_PDF_FONT", ""),
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        "C:/Windows/Fonts/msyh.ttc",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            try:
                pdfmetrics.registerFont(TTFont(name, candidate, subfontIndex=0))
                return name
            except (ValueError, OSError):
                continue
    raise ValueError("未找到可嵌入的中文字体，请通过 JIEYI_PDF_FONT 指定中文 TTF/TTC 字体。")


def _translation(segment):
    return (
        segment.accepted_translation
        or segment.reviewed_translation
        or segment.edited_translation
        or segment.machine_translation
        or ""
    ).strip()


def _inside(inner, outer, tolerance=1.2):
    return all(
        (
            inner[0] >= outer[0] - tolerance,
            inner[1] >= outer[1] - tolerance,
            inner[2] <= outer[2] + tolerance,
            inner[3] <= outer[3] + tolerance,
        )
    )


def _overlaps(a, b):
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def _graphic_intersects(obj, rect):
    if obj.type != raw.FPDF_PAGEOBJ_PATH:
        return True
    matrix = obj.get_matrix()
    previous = None
    for index in range(raw.FPDFPath_CountSegments(obj)):
        segment = raw.FPDFPath_GetPathSegment(obj, index)
        x, y = c_float(), c_float()
        if not raw.FPDFPathSegment_GetPoint(segment, x, y):
            return True
        point = matrix.on_point(x.value, y.value)
        if previous is not None and raw.FPDFPathSegment_GetType(segment) != raw.FPDF_SEGMENT_MOVETO:
            bounds = (
                min(previous[0], point[0]) - 0.5,
                min(previous[1], point[1]) - 0.5,
                max(previous[0], point[0]) + 0.5,
                max(previous[1], point[1]) + 0.5,
            )
            if _overlaps(bounds, rect):
                return True
        previous = point
    return False


def _object_key(obj):
    return cast(obj, c_void_p).value


def _text_geometry(page):
    """Use visible glyphs, excluding PDF spacing/indentation from object bounds."""
    geometry = defaultdict(list)
    textpage = page.get_textpage()
    try:
        for index in range(textpage.count_chars()):
            code = raw.FPDFText_GetUnicode(textpage, index)
            if not code or chr(code).isspace():
                continue
            obj = raw.FPDFText_GetTextObject(textpage, index)
            if obj:
                geometry[_object_key(obj)].append(textpage.get_charbox(index))
    finally:
        textpage.close()
    return geometry


def _bounds(obj, geometry):
    boxes = geometry.get(_object_key(obj))
    if not boxes:
        return obj.get_bounds()
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def _belongs(obj, region, geometry):
    boxes = geometry.get(_object_key(obj), [obj.get_bounds()])
    return all(any(_inside(box, line) for line in region["lines"]) for box in boxes)


def _centered(region, bbox):
    return region["rect"][2] - region["rect"][0] < (bbox[2] - bbox[0]) * 0.85 and all(
        abs((line[0] + line[2] - bbox[0] - bbox[2]) / 2) < 4 for line in region["lines"]
    )


def _expanded_region(region, neighbors, page, objects, geometry):
    """Use whitespace within the original column, stopping before all other content."""
    x0, y0, x1, y1 = region["rect"]
    bbox = page.get_bbox()
    size = region["font_size"]
    centered = _centered(region, bbox)
    if centered:
        margin = (bbox[2] - bbox[0]) * 0.08
        left, right = bbox[0] + margin, bbox[2] - margin
    else:
        # A short line is not a narrow column. Recover the column from aligned
        # neighboring paragraphs, without spanning a differently aligned column.
        aligned = [
            r["rect"]
            for r in neighbors
            if abs(r["rect"][0] - x0) <= size * 3 and r["rect"][0] < x1 and r["rect"][2] > x0
        ]
        left = x0
        right = min(bbox[2] - 8, max([x1] + [r[2] for r in aligned]))
    bottom = max(bbox[1] + 8, y0 - size * 3)
    for obj in objects:
        bounds = _bounds(obj, geometry)
        if obj.type == raw.FPDF_PAGEOBJ_TEXT and _belongs(obj, region, geometry):
            continue
        candidate = (left, bottom, right, y1)
        if not _overlaps(bounds, candidate):
            continue
        if obj.type != raw.FPDF_PAGEOBJ_TEXT and not _graphic_intersects(obj, candidate):
            continue
        # Keep a gap to the next paragraph; never move text or cover an illustration.
        if bounds[3] <= y0 + 1:
            bottom = max(bottom, bounds[3] + 1)
        elif bounds[0] >= x1 - 1:
            right = min(right, (bounds[0] + x1) / 2 - 0.5)
        elif bounds[2] <= x0 + 1:
            left = max(left, (bounds[2] + x0) / 2 + 0.5)
    # Also reserve the full typographic boxes of neighboring paragraphs. Ink bounds
    # are smaller and would otherwise let two translated paragraphs touch.
    for other in neighbors:
        if other is region:
            continue
        bounds = other["rect"]
        if bounds[3] <= y0 + 0.1 and min(right, bounds[2]) > max(left, bounds[0]):
            bottom = max(bottom, bounds[3] + 1)
    if centered:
        center = (x0 + x1) / 2
        radius = max((x1 - x0) / 2, min(center - left, right - center))
        left, right = center - radius, center + radius
    return {**region, "draw_rect": [min(left, x0), min(bottom, y0), max(right, x1), y1]}


def _text_key(text):
    return "".join(unicodedata.normalize("NFKC", text).split())


def _wrap(text, font, size, width):
    lines = []
    for paragraph in text.splitlines() or [text]:
        line = ""
        for char in paragraph:
            if line and pdfmetrics.stringWidth(line + char, font, size) > width:
                # Prefer a word boundary for Latin text without dropping characters.
                space = line.rfind(" ")
                if char.isascii() and char.isalpha() and space > len(line) // 2:
                    lines.append(line[:space])
                    line = line[space + 1 :] + char
                else:
                    lines.append(line)
                    line = char
            else:
                line += char
        lines.append(line)
    return lines


def _fit(text, regions, font, *, shrink=True):
    base = min(float(region["font_size"]) for region in regions)
    minimum = min(base, max(5.5, base * 0.7)) if shrink else base
    for size in [base - step * 0.5 for step in range(max(1, int((base - minimum) * 2) + 1))]:
        remaining = text
        fitted = []
        for region in regions:
            x0, y0, x1, y1 = region.get("draw_rect", region["rect"])
            # Keep the baseline/descent inside the original text area.
            capacity = max(0, int((y1 - y0 - size) / (size * 1.22)) + 1)
            if capacity == 0 or x1 - x0 < size:
                break
            lines = _wrap(remaining, font, size, x1 - x0)
            selected = lines[:capacity]
            fitted.append((region, selected, size))
            remaining = "\n".join(lines[capacity:])
        if not remaining and len(fitted) == len(regions):
            return fitted
    return None


def ensure_pdf_layout(store, document_id):
    with _EXPORT_LOCK:
        metadata = store.get_pdf_metadata(document_id)
        if not metadata.get("layout"):
            book = extract_pdf(store.get_original_pdf(document_id))
            metadata["layout"] = book.layout
            store.update_pdf_metadata(document_id, metadata)
        return metadata


def _appendix(items, font, title):
    stream = io.BytesIO()
    canvas = Canvas(stream, pagesize=(595, 842))
    y = 0
    for page_number, ordinal, text in items:
        label = f"原 PDF 第 {page_number} 页 · 段落 {ordinal + 1}"
        for line in [label] + _wrap(text, font, 11, 499) + [""]:
            if y < 55:
                if y:
                    canvas.showPage()
                canvas.setFont(font, 13)
                canvas.drawString(48, 795, "译文续页 · " + title[:28])
                canvas.setFont(font, 9)
                canvas.drawString(48, 775, "原页的图片与排版完整保留；以下译文未覆盖原页。")
                y = 743
            canvas.setFont(font, 11)
            canvas.drawString(48, y, line)
            y -= 16
    canvas.save()
    return PdfReader(stream)


def compose_pdf(source: bytes, metadata: dict, segments, *, bilingual=False, only_page=None):
    """Return (PDF bytes, layout report). No images/paths/forms are ever deleted or masked."""
    with _EXPORT_LOCK, _RENDER_LOCK:
        font = _font()
        original = PdfReader(io.BytesIO(source))
        layouts = defaultdict(list)
        by_page = defaultdict(list)
        for index, item in enumerate(metadata["layout"]):
            layouts[_text_key(item["source_text"])].append((index, item))
            for region in item["regions"]:
                by_page[region["page"]].append(region)
        plans = []
        overflow = []
        consumed = set()
        for segment in segments:
            refs = list(
                dict.fromkeys(
                    int(ref.rsplit(":", 1)[-1])
                    for ref in segment.source_refs
                    if ref.startswith("pdf:page:")
                )
            )
            candidates = layouts.get(_text_key(segment.source_text), [])
            match = next(
                (
                    (index, item)
                    for index, item in candidates
                    if index not in consumed
                    and list(dict.fromkeys(r["page"] for r in item["regions"])) == refs
                ),
                None,
            )
            if match:
                consumed.add(match[0])
            target = _translation(segment)
            if not target or (only_page and only_page not in refs):
                continue
            if not match:
                overflow.append((refs[0] if refs else 1, segment.ordinal, target))
                continue
            # Fit the complete segment even for a one-page preview. The final page
            # selection happens after typesetting, exactly as in the exported PDF.
            plans.append((segment, target, match[1]["regions"]))
        overlays = defaultdict(list)
        with pdfium.PdfDocument(source) as document:
            pages = {}
            objects = {}
            geometry = {}
            expanded_regions = {}
            try:
                for segment, target, regions in plans:
                    for region in regions:
                        number = region["page"]
                        if number not in pages:
                            pages[number] = document[number - 1]
                            objects[number] = list(pages[number].get_objects(max_depth=1))
                            geometry[number] = _text_geometry(pages[number])
                            # Compute against the untouched page, so preview/export and
                            # translation order cannot change the space assigned to a block.
                            for original_region in by_page[number]:
                                expanded_regions[id(original_region)] = _expanded_region(
                                    original_region,
                                    by_page[number],
                                    pages[number],
                                    objects[number],
                                    geometry[number],
                                )
                    fitted = _fit(target, regions, font, shrink=False)
                    if not fitted:
                        expanded = [expanded_regions[id(r)] for r in regions]
                        fitted = _fit(target, expanded, font)
                    if not fitted:
                        overflow.append((regions[0]["page"], segment.ordinal, target))
                        continue
                    removals = []
                    safe = True
                    for region, lines, size in fitted:
                        number = region["page"]
                        page = pages[number]
                        rect = region["rect"]
                        draw_rect = region.get("draw_rect", rect)
                        selected = []
                        if page.get_rotation() or not _inside(draw_rect, page.get_bbox(), 0.1):
                            safe = False
                            break
                        for obj in objects[number]:
                            bounds = _bounds(obj, geometry[number])
                            if obj.type == raw.FPDF_PAGEOBJ_TEXT:
                                if _overlaps(bounds, rect) and _belongs(
                                    obj, region, geometry[number]
                                ):
                                    selected.append(obj)
                                elif _overlaps(bounds, draw_rect):
                                    safe = False
                                    break
                            elif _overlaps(bounds, draw_rect) and _graphic_intersects(
                                obj, draw_rect
                            ):
                                safe = False
                                break
                        if not safe or not selected:
                            safe = False
                            break
                        removals.append((number, selected))
                    if not safe:
                        overflow.append((regions[0]["page"], segment.ordinal, target))
                        continue
                    for number, selected in removals:
                        for obj in selected:
                            pages[number].remove_obj(obj)
                            objects[number].remove(obj)
                            obj.close()
                    for region, lines, size in fitted:
                        overlays[region["page"]].append((region, lines, size))
                for number in overlays:
                    pages[number].gen_content()
                edited_stream = io.BytesIO()
                if overlays:
                    document.save(edited_stream)
                    edited = PdfReader(edited_stream)
                else:
                    edited = original
            finally:
                for page in pages.values():
                    page.close()
        # Preserve the original catalog/bookmarks. Only replace content of modified pages.
        writer = PdfWriter()
        writer.clone_document_from_reader(original)
        edited_writer = PdfWriter(clone_from=edited) if overlays else None
        overlay_readers = []
        for number, draws in overlays.items():
            page = edited_writer.pages[number - 1]
            box = page.mediabox
            stream = io.BytesIO()
            canvas = Canvas(stream, pagesize=(max(float(box.right), 1), max(float(box.top), 1)))
            for region, lines, size in draws:
                canvas.setFillColorRGB(0.08, 0.08, 0.08)
                canvas.setFont(font, size)
                x0, _, _, y1 = region.get("draw_rect", region["rect"])
                centered = _centered(region, tuple(float(v) for v in box))
                for index, line in enumerate(lines):
                    y = y1 - size * 0.85 - index * size * 1.22
                    if centered:
                        canvas.drawCentredString(
                            (region["rect"][0] + region["rect"][2]) / 2, y, line
                        )
                    else:
                        canvas.drawString(x0, y, line)
            canvas.save()
            overlay_reader = PdfReader(stream)
            overlay_readers.append(overlay_reader)
            page.merge_page(overlay_reader.pages[0])
            # Replacing only page content/resources keeps links, geometry and outline targets.
            for key in ("/Contents", "/Resources"):
                value = page.raw_get(key).clone(writer)
                # PDF streams must be indirect objects; a direct stream silently blanks
                # the page in PDFium/Poppler even when text extraction still succeeds.
                if isinstance(value, StreamObject):
                    value = writer._add_object(value)
                writer.pages[number - 1][NameObject(key)] = value
        report = {
            "replaced_blocks": sum(len(items) for items in overlays.values()),
            "overflow_count": len(overflow),
            "overflow": [{"page": p, "ordinal": o, "translation": t} for p, o, t in overflow],
        }
        if only_page:
            single = PdfWriter()
            single.add_page(writer.pages[only_page - 1])
            writer = single
        elif bilingual:
            # Interleave untouched originals with translated layout pages. Rebuild page links
            # through pypdf's importer; original and translated graphics both remain native.
            paired = PdfWriter()
            for index, page in enumerate(writer.pages):
                paired.add_page(original.pages[index])
                paired.add_page(page)
            for item in metadata.get("navigation", []):
                paired.add_outline_item(item["title"], (item["page"] - 1) * 2)
            writer = paired
        if overflow and not only_page:
            appendix = _appendix(overflow, font, str((original.metadata or {}).get("/Title", "")))
            writer.add_outline_item("译文续页", len(writer.pages))
            writer.append(appendix, import_outline=False)
        output = io.BytesIO()
        writer.write(output)
        return output.getvalue(), report


def export_translated_pdf(store, document_id, *, bilingual=False):
    metadata = ensure_pdf_layout(store, document_id)
    return compose_pdf(
        store.get_original_pdf(document_id),
        metadata,
        store.list_segments(document_id),
        bilingual=bilingual,
    )[0]
