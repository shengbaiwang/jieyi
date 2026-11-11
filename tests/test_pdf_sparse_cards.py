"""Narrow verse and padded definitions still own their solid text containers."""

import io
from dataclasses import replace

import pdfplumber
import pytest
from PIL import Image
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen.canvas import Canvas

from jieyi.ingestion.pdf import GEOMETRY_VERSION, extract_pdf
from jieyi.ingestion.pdf_export import compose_pdf
from jieyi.ingestion.plaintext import segments_from_blocks


def _sparse_card_fixture(style, obstruction=None):
    stream = io.BytesIO()
    canvas = Canvas(stream, pagesize=(400, 600))
    bottom, height = (60, 480) if style == "narrow-verse" else (450, 36)
    if obstruction == "image":
        picture = Image.new("RGB", (40, 40), (25, 90, 150))
        canvas.drawImage(ImageReader(picture), 30, bottom, 340, height)

    def fill_card():
        canvas.setFillColorRGB(1, 1, 1)
        canvas.rect(30, bottom, 340, height, fill=1, stroke=0)

    if obstruction != "foreground":
        fill_card()
    canvas.setFillColorRGB(0, 0, 0)
    canvas.setFont("Times-Roman", 12)
    if style == "narrow-verse":
        for index in range(24):
            canvas.drawString(40, 520 - index * 18, "A narrow poem line.")
    else:
        canvas.drawString(40, 461, "A brief definition with generous padding.")
    if obstruction == "foreground":
        fill_card()
    canvas.save()
    source = stream.getvalue()
    book = extract_pdf(source)
    segments = [replace(segment, machine_translation="译文内容")
                for segment in segments_from_blocks("sparse-card", list(book.blocks))]
    return source, {"layout": book.layout, "geometry_version": GEOMETRY_VERSION}, segments


@pytest.mark.parametrize("style", ["narrow-verse", "padded-definition"])
def test_sparse_text_container_preserves_readability_and_native_fill(style):
    source, metadata, segments = _sparse_card_fixture(style)
    result, report = compose_pdf(source, metadata, segments)
    assert report["reflowed_pages"] == []
    assert report["replaced_blocks"] == len(segments)
    with pdfplumber.open(io.BytesIO(result)) as pdf:
        chars = pdf.pages[0].chars
        assert len(chars) == 4 * len(segments)
        assert all(char["text"] in "译文内容" for char in chars)
        assert all(char["size"] == pytest.approx(12) for char in chars)
        assert len(pdf.pages[0].rects) == 1
        assert pdf.pages[0].rects[0]["non_stroking_color"] == (1, 1, 1)


@pytest.mark.parametrize("style", ["narrow-verse", "padded-definition"])
@pytest.mark.parametrize("obstruction", ["foreground", "image"])
def test_sparse_text_does_not_make_masks_or_artwork_ignorable(style, obstruction):
    source, metadata, segments = _sparse_card_fixture(style, obstruction)
    with pytest.raises(ValueError, match="PDF|排版|图形|遮挡|障碍"):
        compose_pdf(source, metadata, segments)
