"""A source space fill inherits only verified, touching, same-color row fills."""

import io
from dataclasses import replace

import pdfplumber
import pypdfium2 as pdfium
import pytest
from PIL import Image
from pypdfium2 import raw
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen.canvas import Canvas

from jieyi.ingestion.pdf import GEOMETRY_VERSION, extract_pdf
from jieyi.ingestion.pdf_export import (
    _background_objects,
    _bounds,
    _object_key,
    _text_geometry,
    compose_pdf,
)
from jieyi.ingestion.plaintext import segments_from_blocks


def _bridge_fixture(*, option=None):
    stream = io.BytesIO()
    canvas = Canvas(stream, pagesize=(400, 600))
    width = 18 if option == "wide" else 3
    offset = 0.25 if option == "gap" else 0
    bridge = [150 + offset, 548, 150 + offset + width, 563]
    if option == "image":
        canvas.drawImage(ImageReader(Image.new("RGB", (3, 15), "blue")), 150, 548, width, 15)
    canvas.setFillColorRGB(1, 1, 1)
    canvas.rect(30, 548, 120, 15, fill=1, stroke=0)
    if option != "foreground":
        canvas.rect(bridge[0], 548, width, 15, fill=1, stroke=0)
    if option != "one-side":
        canvas.setFillColorRGB(*((0.96, 0.96, 0.96) if option == "color" else (1, 1, 1)))
        canvas.rect(
            150 + width,
            547.5 if option == "height" else 548,
            180,
            15.5 if option == "height" else 15,
            fill=1,
            stroke=0,
        )
    canvas.setFont("Times-Roman", 12)
    canvas.setFillColorRGB(0, 0, 0)
    canvas.drawString(30, 550, "First shaded fragment")
    canvas.drawString(150 + width, 550, "continues with readable text.")
    if option == "foreground":
        canvas.drawString(150, 550, "i")
        canvas.setFillColorRGB(1, 1, 1)
        canvas.rect(bridge[0], 548, width, 15, fill=1, stroke=0)
    canvas.save()
    source = stream.getvalue()
    book = extract_pdf(source)
    segments = [
        replace(segment, machine_translation="浅色底纹中的完整译文。")
        for segment in segments_from_blocks("space-bridge", list(book.blocks))
    ]
    return source, {"layout": book.layout, "geometry_version": GEOMETRY_VERSION}, segments, bridge


def _bridge_is_background(source, bridge):
    with pdfium.PdfDocument(source) as document:
        page = document[0]
        objects = list(page.get_objects(max_depth=1))
        geometry = _text_geometry(page)
        backgrounds = _background_objects(page, objects, geometry)
        matching = [
            obj
            for obj in objects
            if obj.type == raw.FPDF_PAGEOBJ_PATH and _bounds(obj, geometry) == pytest.approx(bridge)
        ]
        assert len(matching) == 1
        return _object_key(matching[0]) in backgrounds


def test_verified_same_color_run_backgrounds_own_their_narrow_space_bridge():
    source, metadata, segments, bridge = _bridge_fixture()
    assert _bridge_is_background(source, bridge)
    result, report = compose_pdf(source, metadata, segments)
    assert report["reflowed_pages"] == []
    assert report["replaced_blocks"] == len(segments)
    with pdfplumber.open(io.BytesIO(source)) as pdf:
        originals = pdf.pages[0].rects
    with pdfplumber.open(io.BytesIO(result)) as pdf:
        page = pdf.pages[0]
        assert page.extract_text() == "浅色底纹中的完整译文。"
        assert all(char["size"] == pytest.approx(12) for char in page.chars)
        assert len(page.rects) == len(originals) == 3
        for original, actual in zip(originals, page.rects, strict=True):
            for key in ("x0", "x1", "top", "bottom", "fill", "stroke", "non_stroking_color"):
                assert actual[key] == pytest.approx(original[key])


@pytest.mark.parametrize(
    "option", ["one-side", "gap", "wide", "color", "height", "foreground", "image"]
)
def test_unproven_space_patch_remains_an_obstacle(option):
    source, metadata, segments, bridge = _bridge_fixture(option=option)
    assert not _bridge_is_background(source, bridge)
    with pytest.raises(ValueError, match="PDF|图形|遮挡|排版"):
        compose_pdf(source, metadata, segments)
