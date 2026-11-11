"""A successful native contents row cannot be consumed by nearby body reflow."""

import io
from dataclasses import replace

import pdfplumber
from reportlab.pdfgen.canvas import Canvas

from jieyi.domain.models import Segment, SegmentKind
from jieyi.ingestion.pdf import GEOMETRY_VERSION
from jieyi.ingestion.pdf_export import compose_pdf


def test_body_fallback_keeps_adjacent_contents_row_pinned():
    stream = io.BytesIO()
    canvas = Canvas(stream, pagesize=(400, 600))
    canvas.setFont("Times-Roman", 12)
    texts = ["A short source paragraph.",
             "A neighboring source paragraph with sufficiently wide lines.",
             "Contents section.....................................46"]
    rows = [(530,), (490, 475), (440,)]
    for text, ys in zip(texts, rows, strict=True):
        for y in ys:
            canvas.drawString(30, y, text)
    canvas.save()
    source = stream.getvalue()
    regions = []
    with pdfplumber.open(io.BytesIO(source)) as pdf:
        chars = pdf.pages[0].chars
        for ys in rows:
            owned = [char for char in chars if any(abs(char["y0"] - (y - 2.604)) < .01
                                                   for y in ys)]
            rect = [min(c["x0"] for c in owned), min(c["y0"] for c in owned),
                    max(c["x1"] for c in owned), max(c["y1"] for c in owned)]
            lines = []
            for y in sorted({c["y0"] for c in owned}, reverse=True):
                line = [c for c in owned if c["y0"] == y]
                lines.append([min(c["x0"] for c in line), y,
                              max(c["x1"] for c in line), max(c["y1"] for c in line)])
            regions.append({"page": 1, "font_size": 12, "font_name": "Times-Roman",
                            "rect": rect, "lines": lines})
    source_texts = [" ".join([text] * len(ys))
                    for text, ys in zip(texts, rows, strict=True)]
    segments = [Segment(id=str(index), document_id="pinned-row", stable_key=str(index),
                        ordinal=index, kind=SegmentKind.PARAGRAPH, source_text=text,
                        source_refs=("pdf:page:1",))
                for index, text in enumerate(source_texts)]
    inside = "译" * 180
    targets = [inside, "注", "目录条目.....................................46"]
    segments = [replace(segment, machine_translation=target)
                for segment, target in zip(segments, targets, strict=True)]
    metadata = {"geometry_version": GEOMETRY_VERSION,
                "layout": [{"source_text": text, "regions": [region]}
                           for text, region in zip(source_texts, regions, strict=True)]}
    result, report = compose_pdf(source, metadata, segments)
    assert report["reflowed_pages"] == [1]
    with pdfplumber.open(io.BytesIO(result)) as pdf:
        chars = pdf.pages[0].chars
        assert sum(c["text"] == "译" for c in chars) == len(inside)
        assert sum(c["text"] == "注" for c in chars) == 1
        pinned = [c for c in chars if c["text"] in "目录条目46"]
        assert len(pinned) == 6
        assert all(abs(c["y0"] - regions[2]["rect"][1]) < 1 for c in pinned)
        assert all(c["y0"] > regions[2]["rect"][3]
                   for c in chars if c["text"] in "译注")
