"""Paragraph backplates constrain expansion and fallback to their owned box."""

import io
from dataclasses import replace

import pdfplumber
import pytest
from reportlab.pdfgen.canvas import Canvas

from jieyi.domain.models import Segment, SegmentKind
from jieyi.ingestion.pdf import GEOMETRY_VERSION
from jieyi.ingestion.pdf_export import compose_pdf


def _callout_fixture(*, body_rows=(400,), split_callout=False):
    stream = io.BytesIO()
    canvas = Canvas(stream, pagesize=(400, 600))
    canvas.setFillColorRGB(.99, .90, .94)
    canvas.rect(30, 460, 200, 90, fill=1, stroke=0)
    canvas.setFillColorRGB(0, 0, 0)
    canvas.setFont("Times-Roman", 12)
    line = "A source callout with several words."
    for y in ((530, 490, 475) if split_callout else (530, 490)):
        canvas.drawString(30, y, line)
    body = "An ordinary body paragraph is wider than the source callout box."
    for y in body_rows:
        canvas.drawString(30, y, body)
    canvas.save()
    source = stream.getvalue()
    with pdfplumber.open(io.BytesIO(source)) as pdf:
        chars = pdf.pages[0].chars
        regions = []
        card_groups = (
            [[char for char in chars if char["y0"] > 520],
             [char for char in chars if 460 < char["y0"] < 510]]
            if split_callout else [[char for char in chars if char["y0"] > 480]]
        )
        for owned in card_groups + [[char for char in chars if char["y0"] < 450]]:
            rect = [min(char["x0"] for char in owned), min(char["y0"] for char in owned),
                    max(char["x1"] for char in owned), max(char["y1"] for char in owned)]
            lines = []
            for y in sorted({char["y0"] for char in owned}, reverse=True):
                row = [char for char in owned if char["y0"] == y]
                lines.append([min(char["x0"] for char in row), y,
                              max(char["x1"] for char in row), max(char["y1"] for char in row)])
            regions.append({"page": 1, "font_size": 12, "font_name": "Times-Roman",
                            "rect": rect, "lines": lines})
    texts = ([line, " ".join([line] * 2)] if split_callout
             else [" ".join([line] * 2)]) + [" ".join([body] * len(body_rows))]
    segments = [Segment(id=str(index), document_id="callout", stable_key=str(index),
                        ordinal=index, kind=SegmentKind.PARAGRAPH, source_text=text,
                        source_refs=("pdf:page:1",)) for index, text in enumerate(texts)]
    metadata = {"geometry_version": GEOMETRY_VERSION,
                "layout": [{"source_text": text, "regions": [region]}
                           for text, region in zip(texts, regions, strict=True)]}
    return source, metadata, segments


def test_borderless_callout_cannot_expand_to_wider_neighboring_body():
    source, metadata, segments = _callout_fixture()
    target = "长译文" * 30
    segments[0] = replace(segments[0], machine_translation=target)
    result, report = compose_pdf(source, metadata, segments)
    assert report["replaced_blocks"] == 1
    with pdfplumber.open(io.BytesIO(result)) as pdf:
        translated = [char for char in pdf.pages[0].chars if char["text"] in "长译文"]
        assert len(translated) == len(target)
        assert all(30 - .1 <= char["x0"] and char["x1"] <= 230 + .1
                   and 460 - .1 <= char["y0"] and char["y1"] <= 550 + .1
                   for char in translated)
        assert all(char["size"] >= 8.4 - .01 for char in translated)
        assert len(pdf.pages[0].rects) == 1


@pytest.mark.parametrize("stale_mapping", [False, True])
def test_callout_fallback_cannot_borrow_space_outside_its_container(stale_mapping):
    source, metadata, segments = _callout_fixture(body_rows=(440, 425, 410))
    segments[0] = replace(segments[0], machine_translation="长" * 400,
                          source_text="Edited callout source." if stale_mapping
                          else segments[0].source_text)
    segments[1] = replace(segments[1], machine_translation="卡片外正文")
    with pytest.raises(ValueError, match="容器|卡片|边界|可读字号"):
        compose_pdf(source, metadata, segments)


def test_callout_fallback_groups_only_paragraphs_owned_by_same_container():
    source, metadata, segments = _callout_fixture(
        body_rows=(440, 425, 410), split_callout=True,
    )
    inside = "译" * 120
    segments = [replace(segment, machine_translation=target)
                for segment, target in zip(segments, (inside, "注", "正文"), strict=True)]
    result, report = compose_pdf(source, metadata, segments)
    assert report["reflowed_pages"] == [1]
    with pdfplumber.open(io.BytesIO(result)) as pdf:
        chars = pdf.pages[0].chars
        card_text = [char for char in chars if char["text"] in "译注"]
        assert len(card_text) == len(inside) + 1
        assert all(30 - .1 <= char["x0"] and char["x1"] <= 230 + .1
                   and 460 - .1 <= char["y0"] and char["y1"] <= 550 + .1
                   for char in card_text)
        body_text = [char for char in chars if char["text"] in "正文"]
        assert len(body_text) == 2
        assert all(char["y1"] < 460 for char in body_text)
