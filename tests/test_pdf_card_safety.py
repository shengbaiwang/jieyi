"""Pastel multiline text cards preserve their native fills and separate borders."""

import io
from dataclasses import replace

import pdfplumber
import pytest
from PIL import Image, ImageChops
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen.canvas import Canvas

from jieyi.ingestion.pdf import GEOMETRY_VERSION, extract_pdf, render_pdf_page
from jieyi.ingestion.pdf_export import compose_pdf
from jieyi.ingestion.plaintext import segments_from_blocks

CARD_COLORS = [(244 / 255, 233 / 255, 233 / 255), (230 / 255, 242 / 255, 1)]


def _card_fixture(color, *, obstruction=None):
    stream = io.BytesIO()
    canvas = Canvas(stream, pagesize=(400, 600))
    if obstruction == "image":
        picture = Image.new("RGB", (60, 20), (20, 100, 180))
        canvas.drawImage(ImageReader(picture), 30, 350, 340, 100)

    def fill_card():
        canvas.setFillColorRGB(*color)
        canvas.rect(30, 350, 340, 100, fill=1, stroke=0)

    if obstruction != "foreground":
        fill_card()
    # The outline is a separate native object and must remain an actual obstacle.
    canvas.setStrokeColorRGB(0.65, 0.2, 0.2)
    canvas.setLineWidth(0.6)
    canvas.rect(30, 350, 340, 100, fill=0, stroke=1)
    canvas.setFont("Times-Roman", 12)
    canvas.setFillColorRGB(0, 0, 0)
    lines = [
        "Idiom: a phrase with a meaning established by common use.",
        "Its meaning is understood through the surrounding context.",
        "The words belong together in a stable and familiar expression.",
        "A translation should be clear and preserve the intended sense.",
    ]
    for y, line in zip((430, 412, 394, 376), lines, strict=True):
        canvas.drawString(40, y, line)
    if obstruction == "foreground":
        fill_card()
    canvas.save()
    source = stream.getvalue()
    book = extract_pdf(source)
    segments = segments_from_blocks("pastel-card", list(book.blocks))
    translated = [
        replace(segment, machine_translation="习语应结合上下文理解，译文需要清晰准确。")
        for segment in segments
    ]
    return source, {"layout": book.layout, "geometry_version": GEOMETRY_VERSION}, translated


@pytest.mark.parametrize("color", CARD_COLORS, ids=["pink", "blue"])
def test_multiline_pastel_card_preserves_font_fill_and_separate_border(color):
    source, metadata, segments = _card_fixture(color)
    result, report = compose_pdf(source, metadata, segments)
    assert report["reflowed_pages"] == []
    assert report["replaced_blocks"] == len(segments)
    with pdfplumber.open(io.BytesIO(result)) as pdf:
        page = pdf.pages[0]
        translated = [char for char in page.chars if "\u4e00" <= char["text"] <= "\u9fff"]
        assert translated
        assert all(char["size"] == pytest.approx(12) for char in translated)
        assert "Idiom:" not in page.extract_text()
        actual_rects = page.rects
    with pdfplumber.open(io.BytesIO(source)) as pdf:
        original_rects = pdf.pages[0].rects
        assert len(original_rects) == len(actual_rects) == 2
        assert [(rect["fill"], rect["stroke"]) for rect in actual_rects] == [
            (True, False),
            (False, True),
        ]
        for original, actual in zip(original_rects, actual_rects, strict=True):
            keys = ["x0", "x1", "top", "bottom", "fill", "stroke"]
            if original["fill"]:
                keys.append("non_stroking_color")
            if original["stroke"]:
                keys.extend(["linewidth", "stroking_color"])
            for key in keys:
                assert actual[key] == pytest.approx(original[key])
    # Compare the complete left outline and adjacent pastel margin, away from text.
    before = Image.open(io.BytesIO(render_pdf_page(source, 1, 400))).convert("RGB")
    after = Image.open(io.BytesIO(render_pdf_page(result, 1, 400))).convert("RGB")
    assert (
        ImageChops.difference(
            before.crop((28, 148, 39, 252)), after.crop((28, 148, 39, 252))
        ).getbbox()
        is None
    )


@pytest.mark.parametrize("color", CARD_COLORS, ids=["pink", "blue"])
@pytest.mark.parametrize("obstruction", ["image", "foreground"])
def test_pastel_card_cannot_hide_illustration_or_foreground_mask(color, obstruction):
    source, metadata, segments = _card_fixture(color, obstruction=obstruction)
    with pytest.raises(ValueError, match="PDF|排版|图形|遮挡|障碍"):
        compose_pdf(source, metadata, segments)
