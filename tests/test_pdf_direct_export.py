"""Production PDF export completes without a layout approval gate."""

import io
from dataclasses import replace

import pdfplumber
from fastapi.testclient import TestClient
from PIL import Image, ImageChops
from pypdf import PdfReader
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen.canvas import Canvas

from jieyi.api.app import create_app
from jieyi.ingestion.pdf import GEOMETRY_VERSION, extract_pdf, render_pdf_page
from jieyi.ingestion.pdf_export import compose_pdf
from jieyi.ingestion.plaintext import segments_from_blocks


def _table_fixture(*, extra_paragraph=False):
    stream = io.BytesIO()
    canvas = Canvas(stream, pagesize=(420, 600))
    canvas.setFont("Times-Roman", 12)
    canvas.rect(30, 530, 360, 35, fill=0, stroke=1)
    canvas.line(210, 530, 210, 565)
    canvas.drawString(40, 545, "Left cell content.")
    canvas.drawString(230, 545, "Right cell content.")
    if extra_paragraph:
        canvas.drawString(40, 380, "Untranslated paragraph remains.")
    canvas.setFillColorRGB(.2, .4, .8)
    canvas.rect(35, 50, 100, 100, fill=1, stroke=0)
    picture = Image.new("RGB", (30, 30), (220, 110, 30))
    canvas.drawImage(ImageReader(picture), 230, 50, 100, 100)
    canvas.save()
    source = stream.getvalue()
    book = extract_pdf(source)
    segments = segments_from_blocks("table", list(book.blocks))
    assert len(segments) == (2 if extra_paragraph else 1)
    return source, {"layout": book.layout, "geometry_version": GEOMETRY_VERSION}, segments


def _text(source):
    return "".join("".join(page.extract_text().split())
                   for page in PdfReader(io.BytesIO(source)).pages)


def test_collapsed_table_exports_full_translation_and_keeps_native_graphics():
    source, metadata, segments = _table_fixture()
    target = "跨越单元格的译文直接导出。" * 30 + "完整结束标记"
    segments[0] = replace(segments[0], machine_translation=target)
    exported, report = compose_pdf(source, metadata, segments, strict=False)
    assert report["direct_pages"] == [1]
    assert _text(exported) == target
    before, after = PdfReader(io.BytesIO(source)), PdfReader(io.BytesIO(exported))
    assert len(after.pages) == 1
    assert [image.data for image in before.pages[0].images] == [
        image.data for image in after.pages[0].images
    ]
    with pdfplumber.open(io.BytesIO(source)) as original, \
            pdfplumber.open(io.BytesIO(exported)) as translated:
        assert len(translated.pages[0].lines) == len(original.pages[0].lines)
        assert len(translated.pages[0].rects) == len(original.pages[0].rects)
    original = Image.open(io.BytesIO(render_pdf_page(source, 1, 420)))
    translated = Image.open(io.BytesIO(render_pdf_page(exported, 1, 420)))
    assert ImageChops.difference(
        original.crop((0, 300, 420, 600)), translated.crop((0, 300, 420, 600))
    ).getbbox() is None


def test_very_long_paragraph_keeps_its_complete_ending():
    source, metadata, segments = _table_fixture()
    target = "必须完整保留的正文内容。" * 350 + "这是最后一句结束标记"
    segments[0] = replace(segments[0], accepted_translation=target)
    result, report = compose_pdf(source, metadata, segments, strict=False)
    assert report["direct_pages"] == [1]
    assert _text(result) == target


def test_direct_export_api_returns_pdf_and_preview_uses_the_same_layout(tmp_path):
    source, _, _ = _table_fixture()
    with TestClient(create_app(str(tmp_path / "direct.db"))) as client:
        project = client.post("/projects", json={
            "name": "Direct export", "source_lang": "en", "target_lang": "zh-CN",
        }).json()
        imported = client.post(f"/projects/{project['id']}/documents/pdf", content=source)
        assert imported.status_code == 201, imported.text
        root = f"/documents/{imported.json()['id']}"
        segment = client.get(root + "/segments").json()[0]
        target = "跨表格直接导出全文。" * 30 + "最后标记"
        updated = client.patch(f"/segments/{segment['id']}/draft", json={"translation": target})
        assert updated.status_code == 200, updated.text
        exported = client.get(root + "/export?format=book")
        assert exported.status_code == 200, exported.text[:100]
        assert exported.headers["content-type"] == "application/pdf"
        assert _text(exported.content) == target
        preview = client.get(root + "/pdf/pages/1?mode=translated")
        assert preview.status_code == 200, preview.text[:100]
        assert preview.content == render_pdf_page(exported.content, 1, 1200)


def test_direct_page_rebuild_respects_deleted_source_segments():
    source, metadata, segments = _table_fixture()
    # The old importer merged the cells into one source object; deletion still
    # means removing that source rather than redrawing it in page fallback.
    result, report = compose_pdf(source, metadata, [], deleted_segments=segments, strict=False)
    assert report["direct_pages"] == [1]
    assert _text(result) == ""


def _cross_page_fixture():
    stream = io.BytesIO()
    canvas = Canvas(stream, pagesize=(420, 600))
    canvas.setFont("Times-Roman", 12)
    canvas.drawString(35, 530, "The communities of the Persian Gulf and inter-")
    canvas.showPage()
    canvas.setFont("Times-Roman", 12)
    canvas.drawString(35, 530, "national trade throughout its long coastal history.")
    canvas.rect(30, 430, 360, 35, fill=0, stroke=1)
    canvas.line(210, 430, 210, 465)
    canvas.drawString(40, 445, "Left cell content.")
    canvas.drawString(230, 445, "Right cell content.")
    canvas.save()
    source = stream.getvalue()
    book = extract_pdf(source)
    original = segments_from_blocks("cross-page", list(book.blocks))
    # Reproduce an existing, valid source paragraph continued across two pages.
    source_text = original[0].source_text + original[1].source_text
    segments = [
        replace(original[0], source_text=source_text,
                source_refs=("pdf:page:1", "pdf:page:2"), machine_translation="中" * 45),
        replace(original[2], ordinal=1, machine_translation="表格内容" * 100),
    ]
    metadata = {"geometry_version": GEOMETRY_VERSION, "layout": [
        {"source_text": source_text,
         "regions": book.layout[0]["regions"] + book.layout[1]["regions"]},
        book.layout[2],
    ]}
    return source, metadata, segments


def test_cross_page_fallback_has_the_same_split_in_page_preview():
    source, metadata, segments = _cross_page_fixture()
    exported, report = compose_pdf(source, metadata, segments, strict=False)
    assert report["direct_pages"] == [1, 2]
    assert _text(exported) == "中" * 45 + "表格内容" * 100
    for page in (1, 2):
        preview, _ = compose_pdf(source, metadata, segments, strict=False, only_page=page)
        assert render_pdf_page(preview, 1, 420) == render_pdf_page(exported, page, 420)


def test_mapping_free_edit_replaces_its_anchor_once_and_preserves_other_text():
    source, metadata, segments = _table_fixture(extra_paragraph=True)
    target = "映射缺失的编辑译文仍需完整导出。"
    segments[0] = replace(segments[0], source_text="Edited source.", source_refs=(),
                          edited_translation=target)
    result, report = compose_pdf(source, metadata, segments, strict=False)
    assert report["direct_pages"] == [1]
    assert _text(result) == target + "Untranslatedparagraphremains."
    preview, _ = compose_pdf(source, metadata, segments, strict=False, only_page=1)
    assert render_pdf_page(preview, 1, 420) == render_pdf_page(result, 1, 420)


def test_edited_cross_page_source_exports_translation_once_without_old_text():
    source, metadata, segments = _cross_page_fixture()
    target = "跨页编辑后的全文必须只保留一次。" * 20 + "跨页结束标记"
    segments = [replace(segments[0], source_text="Edited cross-page source.",
                        machine_translation=target),
                replace(segments[1], machine_translation=None)]
    result, report = compose_pdf(source, metadata, segments, strict=False)
    assert report["direct_pages"] == [1, 2]
    assert _text(result) == target + "Leftcellcontent.Rightcellcontent."
    for page in (1, 2):
        preview, _ = compose_pdf(source, metadata, segments, strict=False, only_page=page)
        assert render_pdf_page(preview, 1, 420) == render_pdf_page(result, page, 420)
