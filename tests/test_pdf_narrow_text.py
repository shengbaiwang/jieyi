"""Narrow chapter numerals must fit by glyph advance rather than one-em width."""

import io
from dataclasses import replace

import pdfplumber
import pytest
from reportlab.pdfgen.canvas import Canvas

from jieyi.ingestion.pdf import GEOMETRY_VERSION, extract_pdf
from jieyi.ingestion.pdf_export import (
    _fit,
    _font,
    _region_font,
    _StyledText,
    _text_width,
    compose_pdf,
)
from jieyi.ingestion.plaintext import segments_from_blocks
from jieyi.ingestion.typography import FontStyle


@pytest.mark.parametrize("text", ["I", "II"])
def test_native_narrow_chapter_numeral_preserves_full_size(text):
    stream = io.BytesIO()
    canvas = Canvas(stream, pagesize=(400, 600))
    canvas.setFont("Helvetica", 16)
    canvas.drawString(300, 450, text)
    canvas.setFont("Helvetica", 12)
    canvas.drawString(30, 350, "A body paragraph follows below this narrow chapter number.")
    canvas.save()
    source = stream.getvalue()
    book = extract_pdf(source)
    body = "译文正文用于验证窄标题可以保留原字号。"
    segments = [replace(s, machine_translation=text if s.source_text == text else body)
                for s in segments_from_blocks("narrow-title", list(book.blocks))]
    assert len(segments) == 2
    region = next(item for item in book.layout if item["source_text"] == text)["regions"][0]
    assert region["rect"][2] - region["rect"][0] < 16
    assert _text_width(text, _region_font(region), 16) <= region["rect"][2] - region["rect"][0]
    result, report = compose_pdf(
        source, {"layout": book.layout, "geometry_version": GEOMETRY_VERSION}, segments
    )
    assert report["replaced_blocks"] == 2
    assert report["reflowed_pages"] == []
    with pdfplumber.open(io.BytesIO(result)) as pdf:
        assert pdf.pages[0].extract_text() == text + "\n" + body
        title_chars = [c for c in pdf.pages[0].chars if c["text"] == "I"]
        assert len(title_chars) == len(text)
        assert all(c["size"] == pytest.approx(16) for c in title_chars)


def test_narrow_region_uses_styled_glyph_width_and_preserves_character_fonts():
    font = _font(FontStyle(family="sans"))
    text = _StyledText("I", [font])
    width = _text_width(text, font, 16) + .01
    region = {"page": 1, "font_size": 16, "font_name": "Times-Roman",
              "rect": [30, 100, 30 + width, 116], "lines": [[30, 100, 30 + width, 116]]}
    assert width < 16
    fitted = _fit(text, [region], _font(), shrink=False)
    assert fitted is not None
    assert fitted[0][2] == 16
    assert fitted[0][1][0].fonts == text.fonts


@pytest.mark.parametrize("text", ["W", "宽", "I\nI"])
def test_wide_or_multiline_text_cannot_borrow_capacity_from_narrow_title(text):
    region = {"page": 1, "font_size": 16, "font_name": "Helvetica",
              "rect": [30, 100, 34.448, 116], "lines": [[30, 100, 34.448, 116]]}
    assert _fit(text, [region], _font()) is None


@pytest.mark.parametrize("width", [0, -1])
def test_nonpositive_width_remains_invalid(width):
    region = {"page": 1, "font_size": 16, "rect": [30, 100, 30 + width, 116],
              "lines": [[30, 100, 30 + width, 116]]}
    assert _fit("I", [region], _font()) is None
