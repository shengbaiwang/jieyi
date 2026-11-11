"""Avoid repeated native geometry work without changing export placement."""

import io
from dataclasses import replace

import pypdfium2 as pdfium
from pypdfium2 import raw
from reportlab.pdfgen.canvas import Canvas

from jieyi.ingestion import pdf_export
from jieyi.ingestion.pdf import GEOMETRY_VERSION, extract_pdf
from jieyi.ingestion.plaintext import segments_from_blocks


def source_with_separate_paragraphs():
    stream = io.BytesIO()
    canvas = Canvas(stream, pagesize=(400, 600))
    canvas.setFont("Helvetica", 12)
    canvas.drawString(35, 530, "The first paragraph has enough space for a short translation.")
    canvas.drawString(35, 430, "The second paragraph also has enough space.")
    canvas.rect(30, 50, 320, 100)
    canvas.save()
    return stream.getvalue()


def test_source_object_bounds_are_measured_once_per_page(monkeypatch):
    with pdfium.PdfDocument(source_with_separate_paragraphs()) as document:
        page = document[0]
        geometry = pdf_export._text_geometry(page)
        obj = next(obj for obj in page.get_objects()
                   if obj.type == raw.FPDF_PAGEOBJ_PATH)
        original = obj.get_bounds
        calls = []

        def measured():
            calls.append(True)
            return original()

        monkeypatch.setattr(obj, "get_bounds", measured)
        expected = original()
        for _ in range(100):
            assert pdf_export._bounds(obj, geometry) == expected
        assert len(calls) == 1
        # A separate page analysis owns its cache and measures independently.
        assert pdf_export._bounds(obj, pdf_export._text_geometry(page)) == expected
        assert len(calls) == 2
        page.close()


def test_export_only_expands_paragraphs_that_need_more_space(monkeypatch):
    source = source_with_separate_paragraphs()
    book = extract_pdf(source)
    metadata = {"layout": book.layout, "geometry_version": GEOMETRY_VERSION}
    segments = segments_from_blocks("test", list(book.blocks))
    assert len(segments) == 2
    segments = [replace(segment, machine_translation="简短译文。") for segment in segments]
    expanded = []
    original = pdf_export._expanded_region

    def measure(region, *args):
        expanded.append(region)
        return original(region, *args)

    monkeypatch.setattr(pdf_export, "_expanded_region", measure)
    _, report = pdf_export.compose_pdf(source, metadata, segments)
    assert report["replaced_blocks"] == 2
    assert expanded == []

    # A longer translation still measures whitespace on the original page,
    # before the preceding paragraph's objects have been removed.
    segments[1] = replace(segments[1], machine_translation="海湾沿岸的人们保持着长期的贸易交流。" * 3)
    _, report = pdf_export.compose_pdf(source, metadata, segments)
    assert report["replaced_blocks"] == 2
    assert len(expanded) == 1
    assert expanded[0]["rect"] == book.layout[1]["regions"][0]["rect"]


def test_translated_pages_share_the_embedded_chinese_font():
    from pypdf import PdfReader

    def translated_book(pages):
        stream = io.BytesIO()
        canvas = Canvas(stream, pagesize=(400, 600))
        for index in range(pages):
            canvas.setFont("Helvetica", 12)
            canvas.drawString(35, 530, chr(65 + index) + " paragraph appears on this page.")
            canvas.showPage()
        canvas.save()
        source = stream.getvalue()
        book = extract_pdf(source)
        segments = [replace(segment, machine_translation="每页使用同一套中文字体。")
                    for segment in segments_from_blocks("test", list(book.blocks))]
        return pdf_export.compose_pdf(
            source, {"layout": book.layout, "geometry_version": GEOMETRY_VERSION}, segments,
        )[0]

    single = translated_book(1)
    multiple = translated_book(8)
    reader = PdfReader(io.BytesIO(multiple))
    font_references = []
    for page in reader.pages:
        fonts = page["/Resources"]["/Font"]
        embedded = [reference for reference in fonts.values()
                    if reference.get_object().get("/FontDescriptor")]
        assert len(embedded) == 1
        font_references.append(embedded[0])
        assert "每页使用同一套中文字体。" in page.extract_text()
    assert len({reference.idnum for reference in font_references}) == 1
    # Page content grows normally; the font program is not copied eight times.
    assert len(multiple) < len(single) * 3
