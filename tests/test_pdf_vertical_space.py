"""A paragraph may use bounded vertical whitespace without crossing native content."""

import io
from dataclasses import replace

import pdfplumber
import pytest
from reportlab.pdfgen.canvas import Canvas

from jieyi.ingestion.pdf import GEOMETRY_VERSION, extract_pdf
from jieyi.ingestion.pdf_export import compose_pdf
from jieyi.ingestion.plaintext import segments_from_blocks


def _footer_fixture(barrier=None, tail_indent=0, upper_blocked=False):
    stream = io.BytesIO()
    canvas = Canvas(stream, pagesize=(400, 600))
    canvas.setFont("Helvetica", 12)
    canvas.drawString(30, 450, "A body paragraph above the copyright and DOI.")
    canvas.setFont("Times-Roman", 8)
    lines = [
        "Thirteen Months in China: a historical annotated translation,",
        "Authors and editors, Oxford University Press (2017),",
        "All rights reserved.",
        "DOI: 10.0000/example.book.chapter",
    ]
    for y, line in zip((60, 50, 40, 30), lines, strict=True):
        canvas.drawString(30 + (tail_indent if y == 30 else 0), y, line)
    if barrier == "rule":
        canvas.rect(25, 27, 300, 1, fill=1, stroke=0)
    elif barrier == "folio":
        canvas.drawString(30, 20, "9")
    if upper_blocked:
        canvas.rect(25, 68, 300, 1, fill=1, stroke=0)
    canvas.save()
    source = stream.getvalue()
    book = extract_pdf(source)
    assert len(book.blocks) == 3
    assert len(book.layout[1]["regions"][0]["lines"]) == 3
    segments = [replace(s, machine_translation=(
        "译" * 110 if s.source_text.startswith("Thirteen Months")
        else s.source_text if s.source_text.startswith("DOI:") else "正文。"
    )) for s in segments_from_blocks("footer-tail", list(book.blocks))]
    return source, {"layout": book.layout, "geometry_version": GEOMETRY_VERSION}, segments


@pytest.mark.parametrize("tail_indent", [0, .4])
def test_footer_uses_available_vertical_space_at_original_size(tail_indent):
    source, metadata, segments = _footer_fixture(tail_indent=tail_indent)
    result, report = compose_pdf(source, metadata, segments)
    assert report["replaced_blocks"] == 3
    assert report["reflowed_pages"] == []
    assert report["min_font_size"] == 8
    with pdfplumber.open(io.BytesIO(result)) as pdf:
        page = pdf.pages[0]
        assert len(pdf.pages) == 1
        assert page.extract_text().count("译") == 110
        assert "DOI: 10.0000/example.book.chapter" in page.extract_text()
        assert all(c["size"] == pytest.approx(8) for c in page.chars if c["text"] == "译")
        assert min(c["y0"] for c in page.chars) >= 8


@pytest.mark.parametrize("barrier", ["rule", "folio"])
def test_vertical_space_cannot_cross_graphics_or_untranslated_native_folio(barrier):
    source, metadata, segments = _footer_fixture(barrier, upper_blocked=True)
    with pytest.raises(ValueError, match="字号|排版|图形|遮挡"):
        compose_pdf(source, metadata, segments)
