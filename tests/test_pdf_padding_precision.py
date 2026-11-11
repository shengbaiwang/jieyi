"""Subpoint Word shading extensions are bounded and do not excuse masks."""

import io
from dataclasses import replace

import pdfplumber
import pytest
from reportlab.pdfgen.canvas import Canvas

from jieyi.ingestion.pdf import GEOMETRY_VERSION, extract_pdf
from jieyi.ingestion.pdf_export import compose_pdf
from jieyi.ingestion.plaintext import segments_from_blocks


def _padding_fixture(extension, *, foreground=False, framed=False):
    stream = io.BytesIO()
    canvas = Canvas(stream, pagesize=(400, 600))
    canvas.setFillColorRGB(250 / 255, 250 / 255, 250 / 255)
    canvas.rect(30, 350, 340, 100, fill=1, stroke=0)
    canvas.setFillColorRGB(0.2, 0.2, 0.2)
    canvas.setFont("Times-Roman", 12)
    for y, line in zip(
        (430, 408, 386, 364),
        [
            "A native shaded callout contains several lines of text.",
            "Its source layout gives the words a readable shared container.",
            "The translation must preserve every original background path.",
            "The final line retains its native color.",
        ],
        strict=True,
    ):
        canvas.drawString(40, y, line)
    canvas.setFillColorRGB(250 / 255, 250 / 255, 250 / 255)
    # Source typographic y0 is 361.396 while actual ink remains above this strip.
    # This empty padding rectangle therefore touches layout bounds, not glyphs.
    canvas.rect(30 - extension, 350, 340 + 2 * extension, 11.6, fill=1, stroke=0)
    if foreground:
        canvas.rect(30, 383, 340, 12, fill=1, stroke=0)
    if framed:
        canvas.setFillColorRGB(0.6, 0.45, 0.2)
        for rect in [
            (29.25, 350, 0.8, 100),
            (369.95, 350, 0.8, 100),
            (29.25, 349.2, 341.5, 0.8),
            (29.25, 450, 341.5, 0.8),
        ]:
            canvas.rect(*rect, fill=1, stroke=0)
    canvas.save()
    source = stream.getvalue()
    book = extract_pdf(source)
    segments = [
        replace(segment, machine_translation="译文保留清晰可读的文字与原有底色。")
        for segment in segments_from_blocks("padding-precision", list(book.blocks))
    ]
    return source, {"layout": book.layout, "geometry_version": GEOMETRY_VERSION}, segments


def test_subpoint_empty_padding_extension_preserves_all_native_paths():
    source, metadata, segments = _padding_fixture(0.2)
    result, report = compose_pdf(source, metadata, segments)
    assert report["reflowed_pages"] == []
    assert report["replaced_blocks"] == len(segments)
    with pdfplumber.open(io.BytesIO(source)) as pdf:
        original_rects = pdf.pages[0].rects
    with pdfplumber.open(io.BytesIO(result)) as pdf:
        page = pdf.pages[0]
        assert len(original_rects) == len(page.rects) == 2
        for original, actual in zip(original_rects, page.rects, strict=True):
            for key in ("x0", "x1", "top", "bottom", "fill", "stroke", "non_stroking_color"):
                assert actual[key] == pytest.approx(original[key])
        translated = [char for char in page.chars if "\u4e00" <= char["text"] <= "\u9fff"]
        assert translated
        assert all(char["size"] == pytest.approx(12) for char in translated)


def test_padding_extension_outside_bounded_tolerance_remains_an_obstacle():
    source, metadata, segments = _padding_fixture(0.4)
    with pytest.raises(ValueError, match="PDF|图形|遮挡|排版"):
        compose_pdf(source, metadata, segments)


def test_matching_color_foreground_mask_remains_an_obstacle():
    source, metadata, segments = _padding_fixture(0.2, foreground=True)
    with pytest.raises(ValueError, match="PDF|图形|遮挡|排版"):
        compose_pdf(source, metadata, segments)


def test_padding_may_reach_centres_of_matching_native_frame_edges():
    source, metadata, segments = _padding_fixture(0.4, framed=True)
    result, report = compose_pdf(source, metadata, segments)
    assert report["reflowed_pages"] == []
    assert report["replaced_blocks"] == len(segments)
    with pdfplumber.open(io.BytesIO(source)) as pdf:
        original_rects = pdf.pages[0].rects
    with pdfplumber.open(io.BytesIO(result)) as pdf:
        assert len(original_rects) == len(pdf.pages[0].rects) == 6
        for original, actual in zip(original_rects, pdf.pages[0].rects, strict=True):
            for key in ("x0", "x1", "top", "bottom", "fill", "stroke", "non_stroking_color"):
                assert actual[key] == pytest.approx(original[key])


@pytest.mark.parametrize("extension,foreground", [(0.8, False), (0.4, True)])
def test_frame_does_not_allow_padding_spill_or_foreground_mask(extension, foreground):
    source, metadata, segments = _padding_fixture(extension, framed=True, foreground=foreground)
    with pytest.raises(ValueError, match="PDF|图形|遮挡|排版"):
        compose_pdf(source, metadata, segments)
