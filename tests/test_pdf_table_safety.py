"""Do not misreport lost PDF table ownership as ordinary text capacity."""

import io
from dataclasses import replace

import pdfplumber
import pytest
from reportlab.pdfgen.canvas import Canvas

from jieyi.ingestion.pdf import GEOMETRY_VERSION, extract_pdf
from jieyi.ingestion.pdf_export import compose_pdf
from jieyi.ingestion.plaintext import segments_from_blocks


def test_collapsed_table_row_reports_cell_structure_instead_of_only_shortening_text():
    stream = io.BytesIO()
    canvas = Canvas(stream, pagesize=(420, 600))
    canvas.setFont("Times-Roman", 12)
    canvas.rect(30, 530, 360, 35, fill=0, stroke=1)
    canvas.line(210, 530, 210, 565)
    canvas.drawString(40, 545, "Left cell content.")
    canvas.drawString(230, 545, "Right cell content.")
    canvas.save()
    source = stream.getvalue()
    book = extract_pdf(source)
    segments = segments_from_blocks("collapsed-table", list(book.blocks))
    # Reproduce legacy ingestion: one segment spans both genuine table cells.
    assert len(segments) == 1
    assert "Left cell content." in segments[0].source_text
    assert "Right cell content." in segments[0].source_text
    region = book.layout[0]["regions"][0]
    assert region["rect"][0] < 210 < region["rect"][2]
    with pdfplumber.open(io.BytesIO(source)) as pdf:
        assert any(line["x0"] == line["x1"] == 210 for line in pdf.pages[0].lines)
    segments[0] = replace(
        segments[0], machine_translation="跨越单元格的译文不能覆盖表格边界。" * 12
    )
    with pytest.raises(ValueError, match="表格线|段落结构") as caught:
        compose_pdf(
            source,
            {"layout": book.layout, "geometry_version": GEOMETRY_VERSION},
            segments,
        )
    assert "缩短译文" not in str(caught.value)
