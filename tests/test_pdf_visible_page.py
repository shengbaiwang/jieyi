"""Visible page boxes exclude prepress text without discarding visible glyphs."""

import io
from collections import Counter
from dataclasses import replace

import pdfplumber
import pytest
from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen.canvas import Canvas

from jieyi.ingestion.pdf import GEOMETRY_VERSION, _page_lines, extract_pdf
from jieyi.ingestion.pdf_export import compose_pdf
from jieyi.ingestion.plaintext import segments_from_blocks

MARK = "0_Frontmatter.indd iv 20/06/17 1:12 PM"


def _fixture(trim_media=False, rotation=0):
    stream = io.BytesIO()
    canvas = Canvas(stream, pagesize=(400, 600))
    canvas.setFont("Helvetica", 12)
    canvas.drawString(30, 450, "A visible body paragraph within the page crop bounds.")
    canvas.setFont("Helvetica", 6)
    canvas.drawString(30, 10, MARK)
    canvas.line(190, 3, 190, 17)
    canvas.line(183, 10, 197, 10)
    canvas.save()
    source = stream.getvalue()
    writer = PdfWriter(clone_from=io.BytesIO(source))
    page = writer.pages[0]
    page.cropbox.lower_left, page.cropbox.upper_right = (20, 40), (380, 560)
    if trim_media:
        page.mediabox.lower_left, page.mediabox.upper_right = (20, 40), (380, 560)
    if rotation:
        page.rotate(rotation)
    cropped = io.BytesIO()
    writer.write(cropped)
    return source, cropped.getvalue()


@pytest.mark.parametrize("trim_media", [False, True])
@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_new_import_excludes_only_wholly_invisible_printer_text(trim_media, rotation):
    _, cropped = _fixture(trim_media, rotation)
    with pdfplumber.open(io.BytesIO(cropped)) as pdf:
        text = " ".join(line["text"] for line in _page_lines(pdf.pages[0]))
    assert "Frontmatter" not in text
    # Rotated line extraction may reorder text; visible characters must survive.
    assert Counter(text.replace(" ", "")) == Counter(
        "A visible body paragraph within the page crop bounds.".replace(" ", "")
    )


@pytest.mark.parametrize("trim_media", [False, True])
def test_legacy_export_leaves_invisible_source_marks_and_graphics_untouched(trim_media):
    source, cropped = _fixture(trim_media)
    old = extract_pdf(source)
    assert any(MARK in block.text for block in old.blocks)
    segments = [replace(s, machine_translation=(
        "不应绘制的印刷标记" if "Frontmatter" in s.source_text else "可见正文。"
    )) for s in segments_from_blocks("cropped", list(old.blocks))]
    metadata = {"layout": old.layout, "geometry_version": GEOMETRY_VERSION}
    result, report = compose_pdf(cropped, metadata, segments)
    assert report["replaced_blocks"] == 1
    assert report["reflowed_pages"] == []
    with pdfplumber.open(io.BytesIO(result)) as pdf:
        page = pdf.pages[0]
        assert "可见正文。" in page.extract_text()
        assert "不应绘制" not in page.extract_text()
        assert MARK in page.extract_text()
        assert len(page.lines) == 2
    before = PdfReader(io.BytesIO(cropped)).pages[0]
    after = PdfReader(io.BytesIO(result)).pages[0]
    assert tuple(after.cropbox) == tuple(before.cropbox)
    assert tuple(after.mediabox) == tuple(before.mediabox)


def test_partially_visible_glyph_is_retained():
    stream = io.BytesIO()
    canvas = Canvas(stream, pagesize=(400, 600))
    canvas.setFont("Helvetica", 12)
    canvas.drawString(30, 450, "A visible body paragraph within the page crop bounds.")
    canvas.drawString(30, 38, "EDGE")  # Glyphs cross the crop boundary at y=40.
    canvas.save()
    writer = PdfWriter(clone_from=io.BytesIO(stream.getvalue()))
    writer.pages[0].cropbox.lower_left = (20, 40)
    cropped = io.BytesIO()
    writer.write(cropped)
    book = extract_pdf(cropped.getvalue())
    assert any("EDGE" in block.text for block in book.blocks)
