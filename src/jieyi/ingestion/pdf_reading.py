"""Passive PDF text geometry and internal destinations for the reading surface."""

from __future__ import annotations

import io

import pdfplumber
from pypdf import PdfReader
from pypdf.generic import ArrayObject, Destination, Fit, NumberObject

from jieyi.ingestion.pdf import _page_lines


def _destination(reader, value):
    if isinstance(value, str):
        value = reader.named_destinations.get(str(value))
    if isinstance(value, ArrayObject) and len(value) >= 2:
        try:
            value = Destination("reader", value[0], Fit(str(value[1]), value[2:]))
        except (TypeError, ValueError, IndexError):
            return None
    if not isinstance(value, Destination):
        return None
    try:
        raw_page = value.get("/Page")
        index = int(raw_page) if isinstance(raw_page, NumberObject) else reader.get_destination_page_number(value)
        if index is None or not 0 <= index < len(reader.pages):
            return None
        page = reader.pages[index]
        height = float(page.mediabox.height)
        top = value.top
        offset = (float(page.mediabox.top) - float(top)) / height if top is not None else 0
        return {"page": index + 1, "offset": max(0, min(1, offset))}
    except (TypeError, ValueError, IndexError, KeyError, ZeroDivisionError):
        return None


def pdf_page_interactions(data: bytes, page_number: int) -> dict:
    reader = PdfReader(io.BytesIO(data))
    if reader.is_encrypted:
        reader.decrypt("")
    page = reader.pages[page_number - 1]
    width, height = float(page.mediabox.width), float(page.mediabox.height)
    left, bottom = float(page.mediabox.left), float(page.mediabox.bottom)

    def rect(value):
        x0, y0, x1, y1 = map(float, value)
        points = [(x0 - left, y0 - bottom), (x1 - left, y1 - bottom)]
        rotation = page.rotation % 360
        if rotation == 90:
            points = [(y, width - x) for x, y in points]
            screen_height, screen_width = width, height
        elif rotation == 180:
            points = [(width - x, height - y) for x, y in points]
            screen_height, screen_width = height, width
        elif rotation == 270:
            points = [(height - y, x) for x, y in points]
            screen_height, screen_width = width, height
        else:
            screen_height, screen_width = height, width
        xs, ys = zip(*points)
        return [min(xs) / screen_width, 1 - max(ys) / screen_height,
                (max(xs) - min(xs)) / screen_width, (max(ys) - min(ys)) / screen_height]

    links, annotations = [], []
    for reference in page.get("/Annots", []):
        try:
            annotation = reference.get_object()
            bounds = annotation.get("/Rect")
            if not bounds or len(bounds) != 4:
                continue
            action = annotation.get("/A", {})
            if hasattr(action, "get_object"):
                action = action.get_object()
            target = annotation.get("/Dest")
            if action.get("/S") == "/GoTo":
                target = action.get("/D")
            destination = _destination(reader, target)
            if annotation.get("/Subtype") == "/Link" and destination:
                links.append({"rect": rect(bounds), **destination})
            elif annotation.get("/Subtype") in {"/Text", "/FreeText", "/Highlight"}:
                content = str(annotation.get("/Contents", "")).strip()
                if content:
                    annotations.append({"rect": rect(bounds), "text": content[:20000]})
        except (TypeError, ValueError, KeyError, IndexError, ZeroDivisionError, AttributeError):
            continue
    return {"links": links, "annotations": annotations}


def pdf_page_text(data: bytes, page_number: int) -> dict:
    """Text offsets are tied to the rendered PDF, including translated variants."""
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        page = pdf.pages[page_number - 1]
        lines, parts, offset = [], [], 0
        for line in _page_lines(page):
            text = line["text"]
            # Browser selections and persisted ranges use UTF-16 code units.
            length = len(text.encode("utf-16-le")) // 2
            lines.append({"text": text, "start": offset, "end": offset + length,
                          "rect": [line["x0"] / page.width, line["top"] / page.height,
                                   (line["x1"] - line["x0"]) / page.width,
                                   (line["bottom"] - line["top"]) / page.height]})
            parts.append(text)
            offset += length + 1
        return {"width": page.width, "height": page.height,
                "text": "\n".join(parts), "lines": lines}


def pdf_search_pages(data: bytes) -> list[dict]:
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        result = []
        for index, page in enumerate(pdf.pages):
            text = "\n".join(line["text"] for line in _page_lines(page))
            result.append({"page": index + 1, "text": text})
            page.close()
        return result
