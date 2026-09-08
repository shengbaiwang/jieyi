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
_SUPERSCRIPT_DIGITS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")


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
        (
            segment.accepted_translation
            or segment.reviewed_translation
            or segment.edited_translation
            or segment.machine_translation
            or ""
        )
        .strip()
        .translate(_SUPERSCRIPT_DIGITS)
    )


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
    # PDFium and pdfplumber derive glyph bounds independently. Type1 overhangs
    # routinely differ by a little more than one point, which used to make an
    # otherwise exact paragraph fall back to the untranslated source.
    tolerance = max(2.0, float(region["font_size"]) * 0.3)
    return all(any(_inside(box, line, tolerance) for line in region["lines"]) for box in boxes)


def _native_region(region, page):
    """Map displayed pdfplumber coordinates back to native PDF object space."""
    rotation = page.get_rotation() % 360
    if not rotation:
        return region
    left, bottom, right, top = page.get_bbox()

    def point(x, y):
        if rotation == 90:
            return right + left - y, x
        if rotation == 180:
            return right + left - x, top + bottom - y
        if rotation == 270:
            return y, top + bottom - x
        raise ValueError(f"不支持的 PDF 页面旋转角度：{rotation}")

    def rectangle(rect):
        points = [
            point(rect[0], rect[1]),
            point(rect[0], rect[3]),
            point(rect[2], rect[1]),
            point(rect[2], rect[3]),
        ]
        return [
            min(p[0] for p in points),
            min(p[1] for p in points),
            max(p[0] for p in points),
            max(p[1] for p in points),
        ]

    native = {**region, "rect": rectangle(region["rect"])}
    native["lines"] = [rectangle(line) for line in region["lines"]]
    if "draw_rect" in region:
        native["draw_rect"] = rectangle(region["draw_rect"])
    return native


def _centered(region, bbox):
    return (
        not region.get("force_left")
        and region["rect"][2] - region["rect"][0] < (bbox[2] - bbox[0]) * 0.85
        and all(abs((line[0] + line[2] - bbox[0] - bbox[2]) / 2) < 4 for line in region["lines"])
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


def _fit(text, regions, font, *, shrink=True, compact=False, minimum=None):
    base = min(float(region["font_size"]) for region in regions)
    if compact:
        # Preserve every character while reclaiming purely presentational blank
        # lines emitted by some models between bilingual title lines.
        text = "\n".join(line for line in text.splitlines() if line.strip())
    floor = (
        min(base, float(minimum))
        if minimum is not None
        else min(base, max(5.5, base * 0.7))
        if shrink
        else base
    )
    leading = 1.08 if compact else 1.22
    sizes = [base - step * 0.5 for step in range(max(1, int((base - floor) * 2) + 1))]
    if sizes[-1] > floor + 0.01:
        sizes.append(floor)
    for size in sizes:
        remaining = text
        fitted = []
        for region in regions:
            x0, y0, x1, y1 = region.get("draw_rect", region["rect"])
            # Keep the baseline/descent inside the original text area.
            capacity = max(0, int((y1 - y0 - size) / (size * leading)) + 1)
            if capacity == 0 or x1 - x0 < size:
                break
            lines = _wrap(remaining, font, size, x1 - x0)
            selected = lines[:capacity]
            fitted.append(({**region, "line_spacing": leading}, selected, size))
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


def compose_pdf(source: bytes, metadata: dict, segments, *, bilingual=False, only_page=None):
    """Return (PDF bytes, layout report). No images/paths/forms are ever deleted or masked."""
    with _EXPORT_LOCK, _RENDER_LOCK:
        font = _font()
        original = PdfReader(io.BytesIO(source))
        segments = list(segments)
        layouts = defaultdict(list)
        by_page = defaultdict(list)
        page_local_regions = defaultdict(list)
        for index, item in enumerate(metadata["layout"]):
            layouts[_text_key(item["source_text"])].append((index, item))
            item_pages = list(dict.fromkeys(region["page"] for region in item["regions"]))
            for region in item["regions"]:
                by_page[region["page"]].append(region)
            if len(item_pages) == 1:
                page_local_regions[item_pages[0]].extend(item["regions"])
        plans = []
        unmatched = []
        failed = []
        consumed = set()
        replaced_segment_ids = set()
        reflowed_pages = set()
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
                unmatched.append((segment, target, refs))
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

            def load_page(number):
                if number in pages:
                    return
                pages[number] = document[number - 1]
                objects[number] = list(pages[number].get_objects(max_depth=1))
                geometry[number] = _text_geometry(pages[number])
                # Always measure expansion on the untouched page so preview and
                # export make identical layout decisions.
                for original_region in by_page[number]:
                    expanded_regions[id(original_region)] = _expanded_region(
                        original_region,
                        by_page[number],
                        pages[number],
                        objects[number],
                        geometry[number],
                    )

            try:
                for segment, target, regions in plans:
                    for region in regions:
                        number = region["page"]
                        load_page(number)
                    fitted = _fit(target, regions, font, shrink=False)
                    if not fitted:
                        expanded = [expanded_regions[id(r)] for r in regions]
                        fitted = _fit(target, expanded, font)
                    if not fitted:
                        fitted = _fit(target, expanded, font, compact=True)
                    if not fitted:
                        failed.append((segment, target, regions))
                        continue
                    removals = []
                    safe = True
                    for region, lines, size in fitted:
                        number = region["page"]
                        page = pages[number]
                        native_region = _native_region(region, page)
                        rect = native_region["rect"]
                        draw_rect = native_region.get("draw_rect", rect)
                        selected = []
                        if not _inside(draw_rect, page.get_bbox(), 0.1):
                            safe = False
                            break
                        for obj in objects[number]:
                            bounds = _bounds(obj, geometry[number])
                            if obj.type == raw.FPDF_PAGEOBJ_TEXT:
                                if _overlaps(bounds, rect) and _belongs(
                                    obj, native_region, geometry[number]
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
                        failed.append((segment, target, regions))
                        continue
                    for number, selected in removals:
                        for obj in selected:
                            pages[number].remove_obj(obj)
                            objects[number].remove(obj)
                            obj.close()
                    for region, lines, size in fitted:
                        overlays[region["page"]].append((segment.id, region, lines, size))
                    replaced_segment_ids.add(segment.id)
                # A paragraph may still fail because a translated citation no longer
                # fits the single source line. Reflow related text inside the same
                # page/column instead of retaining English or adding continuation pages.
                failed_by_page = defaultdict(list)
                cross_page_failed = []
                for item in failed:
                    item_pages = list(dict.fromkeys(r["page"] for r in item[2]))
                    if len(item_pages) == 1:
                        failed_by_page[item_pages[0]].append(item)
                    else:
                        cross_page_failed.append(item)

                def remove_group_text(number, regions):
                    load_page(number)
                    native_regions = [_native_region(region, pages[number]) for region in regions]
                    lines = [line for region in native_regions for line in region["lines"]]
                    tolerance = max(
                        2.0, max(float(region["font_size"]) for region in regions) * 0.3
                    )
                    selected = []
                    for obj in objects[number]:
                        if obj.type != raw.FPDF_PAGEOBJ_TEXT:
                            continue
                        boxes = geometry[number].get(_object_key(obj), [obj.get_bounds()])
                        if all(
                            any(_inside(box, line, tolerance) for line in lines) for box in boxes
                        ):
                            selected.append(obj)
                    if not selected:
                        raise ValueError(
                            f"PDF 第 {number} 页原文对象无法定位，已停止导出以避免漏译。"
                        )
                    for obj in selected:
                        pages[number].remove_obj(obj)
                        objects[number].remove(obj)
                        obj.close()

                def group_splits_text_object(number, regions):
                    """Detect layout regions that cover only part of one PDF text object."""
                    load_page(number)
                    native_regions = [_native_region(region, pages[number]) for region in regions]
                    lines = [line for region in native_regions for line in region["lines"]]
                    tolerance = max(
                        2.0, max(float(region["font_size"]) for region in regions) * 0.3
                    )
                    for obj in objects[number]:
                        if obj.type != raw.FPDF_PAGEOBJ_TEXT:
                            continue
                        boxes = geometry[number].get(_object_key(obj), [obj.get_bounds()])
                        contained = [
                            any(_inside(box, line, tolerance) for line in lines) for box in boxes
                        ]
                        if any(contained) and not all(contained):
                            return True
                    return False

                # Multi-page paragraphs retain their own page regions. An extremely
                # long target may shrink further, but every character stays on the
                # original physical pages.
                for segment, target, regions in cross_page_failed:
                    fitted = _fit(target, regions, font, compact=True, minimum=0.75)
                    if not fitted:
                        raise ValueError(
                            f"第 {segment.ordinal + 1} 段无法在原页完整排版，已停止导出。"
                        )
                    for number in dict.fromkeys(r["page"] for r in regions):
                        remove_group_text(number, [r for r in regions if r["page"] == number])
                    for region, lines, size in fitted:
                        overlays[region["page"]].append((segment.id, region, lines, size))
                        reflowed_pages.add(region["page"])
                    replaced_segment_ids.add(segment.id)

                full_reflow_pages = set()
                for segment, _, refs in unmatched:
                    if len(refs) != 1:
                        raise ValueError(
                            f"第 {segment.ordinal + 1} 段缺少唯一原页映射，已停止导出以避免漏译。"
                        )
                    full_reflow_pages.add(refs[0])

                pages_to_reflow = sorted(set(failed_by_page) | full_reflow_pages)
                for number in pages_to_reflow:
                    load_page(number)
                    if number in full_reflow_pages:
                        group_regions = list(page_local_regions[number])
                        group_segments = [
                            segment
                            for segment in segments
                            if [
                                int(ref.rsplit(":", 1)[-1])
                                for ref in segment.source_refs
                                if ref.startswith("pdf:page:")
                            ]
                            == [number]
                        ]
                    else:
                        bases = [
                            min(float(r["font_size"]) for r in regions)
                            for _, _, regions in failed_by_page[number]
                        ]
                        group_plans = [
                            plan
                            for plan in plans
                            if list(dict.fromkeys(r["page"] for r in plan[2])) == [number]
                            and any(
                                abs(min(float(r["font_size"]) for r in plan[2]) - base)
                                <= base * 0.12
                                for base in bases
                            )
                        ]
                        group_segments = [plan[0] for plan in group_plans]
                        group_regions = [r for plan in group_plans for r in plan[2]]
                    if not group_regions or not group_segments:
                        raise ValueError(
                            f"PDF 第 {number} 页缺少可重排文字区，已停止导出以避免漏译。"
                        )
                    # Some PDFs store a whole line as one text object even when the
                    # extractor splits it into several layout segments. Removing only
                    # one such segment is impossible, so reflow every text region on
                    # this physical page as one page-local operation.
                    if number not in full_reflow_pages and group_splits_text_object(
                        number, group_regions
                    ):
                        group_regions = list(page_local_regions[number])
                        group_segments = [
                            segment
                            for segment in segments
                            if [
                                int(ref.rsplit(":", 1)[-1])
                                for ref in segment.source_refs
                                if ref.startswith("pdf:page:")
                            ]
                            == [number]
                        ]
                    group_segments.sort(key=lambda item: item.ordinal)
                    target = "\n".join(
                        _translation(segment) or segment.source_text for segment in group_segments
                    )
                    rect = [
                        min(r["rect"][0] for r in group_regions),
                        min(r["rect"][1] for r in group_regions),
                        max(r["rect"][2] for r in group_regions),
                        max(r["rect"][3] for r in group_regions),
                    ]
                    flow_region = {
                        "page": number,
                        "font_size": min(float(r["font_size"]) for r in group_regions),
                        "rect": rect,
                        "lines": [line for r in group_regions for line in r["lines"]],
                        "force_left": True,
                    }
                    group_ids = {id(region) for region in group_regions}
                    native_flow_rect = _native_region(flow_region, pages[number])["rect"]
                    blocked = any(
                        id(other) not in group_ids and _overlaps(other["rect"], rect)
                        for other in by_page[number]
                    ) or any(
                        obj.type != raw.FPDF_PAGEOBJ_TEXT
                        and _overlaps(_bounds(obj, geometry[number]), native_flow_rect)
                        and _graphic_intersects(obj, native_flow_rect)
                        for obj in objects[number]
                    )
                    fit_regions = group_regions if blocked else [flow_region]
                    fitted = _fit(target, fit_regions, font, compact=True)
                    if not fitted:
                        fitted = _fit(target, fit_regions, font, compact=True, minimum=0.75)
                    if not fitted:
                        raise ValueError(f"PDF 第 {number} 页译文超过物理页面容量，已停止导出。")
                    remove_group_text(number, group_regions)
                    segment_ids = {segment.id for segment in group_segments}
                    overlays[number] = [
                        draw for draw in overlays[number] if draw[0] not in segment_ids
                    ]
                    for region, lines, size in fitted:
                        overlays[number].append((f"page-flow:{number}", region, lines, size))
                    replaced_segment_ids.update(
                        segment.id for segment in group_segments if _translation(segment)
                    )
                    reflowed_pages.add(number)

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
            rotation = page.rotation % 360
            if rotation == 90:
                canvas.transform(0, 1, -1, 0, float(box.right) + float(box.left), 0)
            elif rotation == 180:
                canvas.transform(
                    -1,
                    0,
                    0,
                    -1,
                    float(box.right) + float(box.left),
                    float(box.top) + float(box.bottom),
                )
            elif rotation == 270:
                canvas.transform(0, -1, 1, 0, 0, float(box.top) + float(box.bottom))
            for _, region, lines, size in draws:
                canvas.setFillColorRGB(0.08, 0.08, 0.08)
                canvas.setFont(font, size)
                x0, _, _, y1 = region.get("draw_rect", region["rect"])
                centered = not rotation and _centered(region, tuple(float(v) for v in box))
                for index, line in enumerate(lines):
                    y = y1 - size * 0.85 - index * size * region.get("line_spacing", 1.22)
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
            "replaced_blocks": len(replaced_segment_ids),
            "overflow_count": 0,
            "overflow": [],
            "reflowed_pages": sorted(reflowed_pages),
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
