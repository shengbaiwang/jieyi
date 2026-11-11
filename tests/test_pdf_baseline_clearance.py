"""Different replacement-font descenders clear a proven lower border safely."""

import io
from dataclasses import replace

import pdfplumber
import pypdfium2 as pdfium
import pytest
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen.canvas import Canvas

from jieyi.ingestion.pdf import GEOMETRY_VERSION, extract_pdf
from jieyi.ingestion.pdf_export import _measured_lines, _text_geometry, compose_pdf
from jieyi.ingestion.plaintext import segments_from_blocks


def _heading_fixture(font="Times-Bold", size=12, *, obstruction=None):
    stream = io.BytesIO()
    canvas = Canvas(stream, pagesize=(600, 600))
    baseline = 500
    line_top = baseline - size * 0.15
    source_y0 = baseline + pdfmetrics.getDescent(font) * size / 1000
    source_top = source_y0 + size
    if obstruction == "container":
        canvas.setFillColorRGB(0.95, 0.95, 0.9)
        canvas.rect(25, source_y0 - 1, 540, size + 1.2, fill=1, stroke=0)
    canvas.setFont(font, size)
    canvas.setFillColorRGB(0.3, 0.1, 0.15)
    canvas.drawString(30, baseline, "CHAPTER SUMMARY AND READING GUIDE")
    canvas.setFillColorRGB(0.45, 0.18, 0.22)
    canvas.rect(20, line_top - 0.6, 540, 0.6, fill=1, stroke=0)
    if obstruction == "table":
        for x in (20, 559.4):
            canvas.rect(x, line_top - 40, 0.6, 80, fill=1, stroke=0)
    else:
        canvas.rect(20, line_top - 40, 0.6, 40, fill=1, stroke=0)
    if obstruction == "above":
        canvas.setFillColorRGB(0.2, 0.3, 0.4)
        canvas.rect(25, source_top + 0.2, 540, 3, fill=1, stroke=0)
    canvas.save()
    source = stream.getvalue()
    book = extract_pdf(source)
    segments = [
        replace(segment, machine_translation="章节小结")
        for segment in segments_from_blocks("baseline", list(book.blocks))
    ]
    assert len(segments) == 1
    return source, {"layout": book.layout, "geometry_version": GEOMETRY_VERSION}, segments, line_top


@pytest.mark.parametrize("font,size", [("Times-Bold", 12), ("Helvetica", 18), ("Times-Roman", 9)])
def test_different_font_descenders_raise_only_draw_region_and_preserve_native_border(font, size):
    source, metadata, segments, border_top = _heading_fixture(font, size)
    original_region = metadata["layout"][0]["regions"][0]
    initial_boxes = _measured_lines(original_region, ["章节小结"], size, (0, 0, 600, 600))
    initial_bottom = min(box[1] for box in initial_boxes)
    assert initial_bottom < border_top
    result, report = compose_pdf(source, metadata, segments)
    assert report["reflowed_pages"] == []
    assert report["replaced_blocks"] == 1
    with pdfplumber.open(io.BytesIO(source)) as pdf:
        original_rects = pdf.pages[0].rects
    with pdfplumber.open(io.BytesIO(result)) as pdf:
        page = pdf.pages[0]
        assert page.extract_text() == "章节小结"
        assert all(char["size"] == pytest.approx(size) for char in page.chars)
        assert all(
            char["non_stroking_color"] == pytest.approx((0.3, 0.1, 0.15), abs=1 / 255)
            for char in page.chars
        )
        assert len(page.rects) == len(original_rects) == 2
        for original, actual in zip(original_rects, page.rects, strict=True):
            for key in ("x0", "x1", "top", "bottom", "fill", "stroke", "non_stroking_color"):
                assert actual[key] == pytest.approx(original[key])
    with pdfium.PdfDocument(result) as document:
        boxes = [box for owned in _text_geometry(document[0]).values() for box in owned]
    moved_bottom = min(box[1] for box in boxes)
    assert moved_bottom >= border_top + size * 0.04 - 0.001
    assert 0 < moved_bottom - initial_bottom <= size * 0.3 + 0.001
    # The caller's source ownership remains unchanged by translated draw positioning.
    assert original_region["rect"] == metadata["layout"][0]["regions"][0]["rect"]
    assert "draw_rect" not in original_region


@pytest.mark.parametrize("obstruction", ["above", "table", "container"])
def test_baseline_cannot_cross_upper_graphic_table_rule_or_container(obstruction):
    source, metadata, segments, _ = _heading_fixture(obstruction=obstruction)
    with pytest.raises(ValueError, match="PDF|排版|图形|遮挡"):
        compose_pdf(source, metadata, segments)
