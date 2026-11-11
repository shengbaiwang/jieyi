"""A contents row joined to a following heading keeps its native number mask."""

import io
from collections import Counter
from dataclasses import replace

import pdfplumber
import pytest
from reportlab.pdfgen.canvas import Canvas

from jieyi.domain.models import Segment, SegmentKind
from jieyi.ingestion.pdf import GEOMETRY_VERSION
from jieyi.ingestion.pdf_export import compose_pdf


def _compound_fixture():
    stream = io.BytesIO()
    canvas = Canvas(stream, pagesize=(400, 600))
    canvas.setFont("Times-Roman", 12)
    canvas.drawString(30, 550, "A section........................................")
    canvas.drawString(320, 550, "99")
    canvas.setFillColorRGB(1, 1, 1)
    canvas.rect(316, 546, 30, 16, fill=1, stroke=0)
    canvas.setFillColorRGB(0, 0, 0)
    canvas.drawString(319.5, 550, "46")
    canvas.drawString(30, 530, "Appendix: A reliable heading.")
    canvas.save()
    source = stream.getvalue()
    lines = []
    parts = []
    with pdfplumber.open(io.BytesIO(source)) as pdf:
        chars = pdf.pages[0].chars
        for top_half in (True, False):
            owned = [c for c in chars if (c["y0"] > 540) == top_half]
            lines.append([min(c["x0"] for c in owned), min(c["y0"] for c in owned),
                          max(c["x1"] for c in owned), max(c["y1"] for c in owned)])
            parts.append("".join(c["text"] for c in sorted(owned, key=lambda c: c["x0"])))
    text = " ".join(parts)
    folio = parts[0].rsplit(".", 1)[-1]
    segment = Segment(id="compound", document_id="compound", stable_key="compound", ordinal=0,
                      kind=SegmentKind.PARAGRAPH, source_text=text, source_refs=("pdf:page:1",),
                      machine_translation="目录条目........................" + folio + " 附录：可靠标题")
    region = {"page": 1, "font_size": 12, "font_name": "Times-Roman", "lines": lines,
              "rect": [min(line[0] for line in lines), min(line[1] for line in lines),
                       max(line[2] for line in lines), max(line[3] for line in lines)]}
    metadata = {"geometry_version": GEOMETRY_VERSION,
                "layout": [{"source_text": text, "regions": [region]}]}
    return source, metadata, segment, folio


def test_compound_contents_row_and_heading_keep_all_text_and_native_folio():
    source, metadata, segment, folio = _compound_fixture()
    result, report = compose_pdf(source, metadata, [segment])
    assert report["replaced_blocks"] == 1
    assert report["reflowed_pages"] == []
    with pdfplumber.open(io.BytesIO(result)) as pdf:
        chars = pdf.pages[0].chars
        text = "".join(c["text"] for c in chars)
        assert Counter(c for c in text if c.isdigit()) == Counter(folio)
        assert Counter(c for c in text if "\u4e00" <= c <= "\u9fff") == Counter("目录条目附录可靠标题")
        assert "A section" not in text and "Appendix" not in text
        assert len(pdf.pages[0].rects) == 1
        assert "".join(c["text"] for c in chars if c["y0"] > 540
                       and "\u4e00" <= c["text"] <= "\u9fff") == "目录条目"
        assert "".join(c["text"] for c in chars if c["y1"] < 545
                       and "\u4e00" <= c["text"] <= "\u9fff") == "附录可靠标题"


def test_compound_contents_rejects_changed_folio():
    source, metadata, segment, folio = _compound_fixture()
    segment = replace(segment, machine_translation=segment.machine_translation.replace(folio, "999"))
    with pytest.raises(ValueError, match="目录|页码"):
        compose_pdf(source, metadata, [segment])


def test_compound_contents_rejects_unconfirmed_native_line_ownership():
    source, metadata, segment, _ = _compound_fixture()
    changed = segment.source_text.replace("A section", "A different section")
    segment = replace(segment, source_text=changed)
    metadata["layout"][0]["source_text"] = changed
    with pytest.raises(ValueError, match="归属|目录"):
        compose_pdf(source, metadata, [segment])


def test_compound_contents_cannot_fall_back_to_unclipped_folio_region():
    source, metadata, segment, _ = _compound_fixture()
    segment = replace(segment, machine_translation=segment.machine_translation.replace("目录条目", "长标题" * 100))
    with pytest.raises(ValueError, match="目录|可读字号"):
        compose_pdf(source, metadata, [segment])


def test_compound_contents_requires_a_nonblank_translated_heading():
    source, metadata, segment, folio = _compound_fixture()
    segment = replace(segment, machine_translation="目录条目........................" + folio + "   ")
    with pytest.raises(ValueError, match="目录|映射"):
        compose_pdf(source, metadata, [segment])
