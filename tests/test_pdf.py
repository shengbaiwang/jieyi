import io
import time

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from jieyi.api.app import create_app
from jieyi.ingestion.pdf import _line_text, extract_pdf, render_pdf_page


def pdf_fixture(*, blank=False, password=None):
    writer = PdfWriter()
    writer.add_metadata({"/Title": "Gulf History"})
    pages = [
        [("History 1", 40, 760), ("The Persian Gulf connects communities and inter-", 40, 720)],
        [
            ("History 2", 40, 760),
            ("national trade across the sea.", 40, 720),
            ("A new paragraph begins here.", 52, 690),
        ],
        [("History 3", 40, 760), ("Another chapter follows the coast.", 40, 720)],
    ]
    for lines in pages:
        page = writer.add_blank_page(width=600, height=800)
        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})}
        )
        stream = DecodedStreamObject()
        content = "\n".join(f"BT /F1 12 Tf {x} {y} Td ({text}) Tj ET" for text, x, y in lines)
        stream.set_data(b"" if blank else content.encode())
        page[NameObject("/Contents")] = writer._add_object(stream)
    writer.add_outline_item("Trade", 0)
    writer.add_outline_item("Coast", 2)
    if password:
        writer.encrypt(password)
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


def test_pdf_layout_provenance_and_bookmarks():
    progress = []
    book = extract_pdf(pdf_fixture(), lambda done, total: progress.append((done, total)))
    assert book.title == "Gulf History"
    assert len(book.pages) == 3
    assert [entry["title"] for entry in book.navigation] == ["Trade", "Coast"]
    assert "international trade" in book.blocks[0].text
    assert book.blocks[0].source_refs == ("pdf:page:1", "pdf:page:2")
    assert all("History" not in block.text for block in book.blocks)
    assert progress[-1] == (3, 3)
    assert book.pages[1]["label"] == "2"


def test_ligatures_preserve_words_without_duplicate_glyphs():
    chars = [
        {"text": text, "x0": x, "x1": x + 5, "top": 0, "size": 10}
        for text, x in [("ﬁ", 0), ("ﬁ", 0), ("s", 5), ("h", 10), ("s", 20)]
    ]
    assert _line_text(chars) == "fish s"


@pytest.mark.parametrize(
    "data, message",
    [
        (b"not a PDF", "有效"),
        (b"%PDF-1.7 broken", "解析失败"),
        (pdf_fixture(blank=True), "OCR"),
        (pdf_fixture(password="secret"), "密码"),
    ],
)
def test_unreadable_pdf_has_actionable_error(data, message):
    with pytest.raises(ValueError, match=message):
        extract_pdf(data)


def test_original_page_renders_and_checks_range():
    assert render_pdf_page(pdf_fixture(), 1, width=400).startswith(b"\x89PNG")
    with pytest.raises(ValueError):
        render_pdf_page(pdf_fixture(), 4)


def test_pdf_api_import_reuse_translate_export_delete(tmp_path):
    with TestClient(create_app(str(tmp_path / "pdf.db"))) as client:
        data = pdf_fixture()
        inspection = client.post("/imports/pdf/inspect", content=data)
        assert inspection.status_code == 202
        identifier = inspection.json()["id"]
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            state = client.get(f"/imports/pdf/inspect/{identifier}").json()
            if state["status"] != "processing":
                break
            # While parsing, other API requests remain available.
            assert client.get("/projects").status_code == 200
            time.sleep(0.02)
        assert state["status"] == "ready", state
        assert "book" not in state
        project = client.post(
            "/projects", json={"name": "PDF", "source_lang": "en", "target_lang": "zh-CN"}
        ).json()
        path = f"/projects/{project['id']}/documents/pdf"
        response = client.post(path, content=data)
        assert response.status_code == 201, response.text
        doc = response.json()
        assert doc["source_format"] == "pdf"
        assert client.post(path, content=data).json()["id"] == doc["id"]
        root = f"/documents/{doc['id']}"
        assert client.get(root + "/pdf/original").content == data
        assert client.get(root + "/pdf/pages/2").headers["content-type"] == "image/png"
        assert client.get(root + "/pdf/pages/4").status_code == 404
        assert client.get(root + "/pdf").json()["pages"][0]["number"] == 1
        overview = client.get(root + "/overview").json()
        assert [ch["title"] for ch in overview["chapters"]] == ["Trade", "Coast"]
        segments = client.get(root + "/segments").json()
        assert segments[0]["source_refs"] == ["pdf:page:1", "pdf:page:2"]
        job = client.post(
            root + "/jobs",
            json={"draft_provider": "echo", "draft_model": "dry-run", "segment_ranges": [[0, 0]]},
        ).json()
        assert client.post(f"/jobs/{job['id']}/run").status_code == 200
        assert client.get(root + "/segments").json()[0]["machine_translation"]
        saved = client.patch(
            f"/segments/{segments[0]['id']}/draft",
            json={"translation": "波斯湾连接沿岸社区和国际贸易。"},
        )
        assert saved.status_code == 200, saved.text
        assert "layout" not in client.get(root + "/pdf").json()
        preview = client.get(root + "/pdf/pages/1?mode=translated")
        assert preview.status_code == 200
        assert preview.content.startswith(b"\x89PNG")
        notes = client.get(root + "/pdf/pages/1/translation-notes").json()
        assert notes["overflow"] == []
        assert notes["overflow_count"] == 0
        assert client.get(root + "/pdf/pages/4?mode=translated").status_code == 404
        exported = client.get(root + "/export?format=text&bilingual=true")
        assert "波斯湾连接" in exported.text
        assert "Persian Gulf" in exported.text
        pdf_export = client.get(root + "/export?format=book")
        assert pdf_export.status_code == 200, pdf_export.text[:100]
        assert pdf_export.content.startswith(b"%PDF")
        assert "translated.pdf" in pdf_export.headers["content-disposition"]
        assert preview.content == render_pdf_page(pdf_export.content, 1, 1200)
        original = client.get(root + "/segments").json()[1]
        split = client.post(
            f"/segments/{original['id']}/split",
            json={"source_text": original["source_text"], "selection_start": 0, "selection_end": 5},
        )
        assert split.status_code == 200, split.text
        split_segments = client.get(root + "/segments").json()
        assert split_segments[1]["source_refs"] == ["pdf:page:2"]
        assert split_segments[2]["source_refs"] == ["pdf:page:2"]
        assert client.delete(root).status_code == 204
        assert client.get(root + "/pdf/original").status_code == 404
        assert client.get(root + "/pdf/pages/1").status_code == 404


def test_pdf_api_failure_and_size_limit(tmp_path, monkeypatch):
    import jieyi.api.pdf_routes as routes

    monkeypatch.setattr(routes, "MAX_BYTES", 10)
    with TestClient(create_app(str(tmp_path / "errors.db"))) as client:
        assert client.post("/imports/pdf/inspect", content=b"").status_code == 422
        assert client.post("/imports/pdf/inspect", content=b"x" * 11).status_code == 413
        assert client.get("/imports/pdf/inspect/missing").status_code == 404


def _typeset_pdf(draw):
    from reportlab.pdfgen.canvas import Canvas

    output = io.BytesIO()
    canvas = Canvas(output, pagesize=(600, 800))
    draw(canvas)
    canvas.save()
    return output.getvalue()


@pytest.mark.parametrize("font", ["Helvetica-Bold", "Times-Bold", "Times-Italic", "Courier-Bold"])
@pytest.mark.parametrize("alignment", ["left", "center", "right"])
def test_multiline_heading_keeps_centered_lines_and_separates_author_and_body(font, alignment):
    def draw(canvas):
        canvas.setFont(font, 18)
        draw_title, x = {
            "left": (canvas.drawString, 40),
            "center": (canvas.drawCentredString, 300),
            "right": (canvas.drawRightString, 560),
        }[alignment]
        draw_title(x, 710, "The History of Coastal")
        draw_title(x, 684, "Communities and Trade")
        canvas.setFont("Helvetica-Oblique", 14)
        canvas.drawCentredString(300, 652, "A. Researcher")
        canvas.setFont("Helvetica", 10)
        canvas.drawString(40, 600, "The opening paragraph explains the history of the region")
        canvas.drawString(40, 588, "and the communities connected by maritime trade.")

    book = extract_pdf(_typeset_pdf(draw))
    assert [(block.kind.value, block.text) for block in book.blocks] == [
        ("heading", "The History of Coastal Communities and Trade"),
        ("paragraph", "A. Researcher"),
        (
            "paragraph",
            (
                "The opening paragraph explains the history of the region "
                "and the communities connected by maritime trade."
            ),
        ),
    ]
    assert len(book.layout[0]["regions"][0]["lines"]) == 2


@pytest.mark.parametrize(
    "initial,remainder,space,expected",
    [
        ("T", "he first paragraph continues across several lines", 0, "The first"),
        ("I", "n this chapter the first paragraph continues", 0, "In this"),
        ("I", "remember the first paragraph of this chapter", 4, "I remember"),
        ("T", "HE FIRST PARAGRAPH CONTINUES ACROSS SEVERAL LINES", 0, "THE FIRST"),
    ],
)
@pytest.mark.parametrize("raised", [True, False])
def test_large_initial_joins_first_word_and_preserves_paragraph_geometry(
    initial,
    remainder,
    space,
    expected,
    raised,
):
    def draw(canvas):
        canvas.setFont("Helvetica", 26)
        canvas.drawString(40, 700 if raised else 683, initial)
        width = canvas.stringWidth(initial, "Helvetica", 26)
        canvas.setFont("Helvetica", 10)
        canvas.drawString(40 + width + space, 700, remainder)
        canvas.drawString(40 if raised else 40 + width, 688, "and finishes here.")
        canvas.drawString(52, 670, "Another paragraph remains a separate translation unit.")

    book = extract_pdf(_typeset_pdf(draw))
    assert len(book.blocks) == 2
    assert book.blocks[0].text.startswith(expected)
    assert book.blocks[0].text.endswith("and finishes here.")
    assert book.blocks[0].kind.value == "paragraph"
    assert book.blocks[1].text.startswith("Another paragraph")
    region = book.layout[0]["regions"][0]
    assert region["rect"][0] == 40
    assert region["font_size"] == 10
    assert region["rect"][3] > 700
    assert len(region["lines"]) == 2
    assert len(book.layout) == len(book.blocks)
    assert all(item["source_text"] == block.text for item, block in zip(book.layout, book.blocks))


def test_standalone_letter_heading_and_numbered_list_are_not_drop_caps():
    def draw(canvas):
        canvas.setFont("Helvetica-Bold", 24)
        canvas.drawString(40, 730, "A")
        canvas.setFont("Helvetica", 10)
        canvas.drawString(40, 690, "An independent paragraph under the letter heading.")
        canvas.drawString(40, 650, "1. First numbered item continues")
        canvas.drawString(52, 638, "onto its hanging-indented second line.")
        canvas.drawString(40, 620, "2. The second numbered item stays separate.")

    book = extract_pdf(_typeset_pdf(draw))
    assert [block.text for block in book.blocks] == [
        "A",
        "An independent paragraph under the letter heading.",
        "1. First numbered item continues onto its hanging-indented second line.",
        "2. The second numbered item stays separate.",
    ]


def test_double_spaced_body_lines_do_not_become_individual_paragraphs():
    def draw(canvas):
        canvas.setFont("Helvetica", 10)
        for y, text in zip(
            [700, 678, 656, 634],
            [
                "This double spaced paragraph has several lines of continuous text",
                "and its ordinary line spacing should not trigger separate blocks",
                "because all these sentences belong to one paragraph and should",
                "remain together for translation.",
            ],
        ):
            canvas.drawString(40, y, text)
        canvas.drawString(52, 612, "This indented paragraph is separate.")

    book = extract_pdf(_typeset_pdf(draw))
    assert len(book.blocks) == 2
    assert book.blocks[0].text.endswith("remain together for translation.")


def test_bold_body_sized_multiline_heading_is_not_swallowed_by_prose():
    def draw(canvas):
        canvas.setFont("Helvetica", 10)
        canvas.drawString(40, 720, "The preceding paragraph ends at this point.")
        canvas.setFont("Helvetica-Bold", 10)
        canvas.drawString(40, 690, "Regional history and")
        canvas.drawString(40, 676, "the development of maritime communities")
        canvas.setFont("Helvetica", 10)
        canvas.drawString(40, 650, "The following paragraph contains the discussion")
        canvas.drawString(40, 638, "of the topic introduced by this section heading.")

    book = extract_pdf(_typeset_pdf(draw))
    assert [block.kind.value for block in book.blocks] == ["paragraph", "heading", "paragraph"]
    assert book.blocks[1].text == "Regional history and the development of maritime communities"


def test_numbered_contents_title_wraps_without_absorbing_author_or_next_entry():
    def draw(canvas):
        canvas.setFont("Helvetica", 10)
        canvas.drawString(40, 710, "1 The long history of a coastal region:")
        canvas.drawString(55, 698, "the period of maritime trade (1500-1800) 27")
        canvas.setFont("Helvetica-Oblique", 10)
        canvas.drawString(55, 686, "A. Researcher")
        canvas.setFont("Helvetica", 10)
        canvas.drawString(40, 662, "2 Another chapter in the history of the coast 41")

    book = extract_pdf(_typeset_pdf(draw))
    assert [block.text for block in book.blocks] == [
        ("1 The long history of a coastal region: the period of maritime trade (1500-1800) 27"),
        "A. Researcher",
        "2 Another chapter in the history of the coast 41",
    ]


def _old_pdf_document(tmp_path):
    from dataclasses import replace

    from jieyi.persistence.sqlite import SQLiteStore
    from jieyi.workflow.services import create_pdf_document, create_project

    store = SQLiteStore(str(tmp_path / "resegment.db"))
    store.migrate()
    project = create_project(store, name="PDF", source_lang="en", target_lang="zh-CN")
    book = extract_pdf(pdf_fixture())
    first = book.blocks[0]
    broken = replace(
        book,
        blocks=(
            replace(first, text=first.text[:1]),
            replace(first, text=first.text[1:]),
            *book.blocks[1:],
        ),
    )
    doc = create_pdf_document(store, project_id=project.id, file_data=pdf_fixture(), book=broken)
    return store, doc, book


def test_pdf_resegmentation_preserves_translation_ids_and_is_repeatable(tmp_path):
    from jieyi.ingestion.pdf import VERSION
    from jieyi.workflow.services import resegment_pdf_document

    store, doc, book = _old_pdf_document(tmp_path)
    old = store.list_segments(doc.id)
    store.set_machine_translation(old[-1].id, "另一章沿着海岸展开。")
    store.save_segment_draft(old[-1].id, "下一章继续讨论沿岸地区。")
    translated = store.get_segment(old[-1].id)
    report = resegment_pdf_document(store, doc.id, book=book)
    new = store.list_segments(doc.id)
    assert report["previous_count"] == len(old)
    assert report["segment_count"] == len(old) - 1
    assert [item.source_text for item in new] == [block.text for block in book.blocks]
    assert new[-1].id == translated.id
    assert new[-1].machine_translation == translated.machine_translation
    assert new[-1].edited_translation == translated.edited_translation
    assert store.get_original_pdf(doc.id) == pdf_fixture()
    metadata = store.get_pdf_metadata(doc.id)
    assert metadata["segmenter_version"] == VERSION
    assert len(metadata["layout"]) == len(new)
    again = resegment_pdf_document(store, doc.id, book=book)
    assert again["replaced_count"] == 0
    assert store.list_segments(doc.id) == new


@pytest.mark.parametrize("protect", ["translation", "manual", "source_edit", "job"])
def test_pdf_resegmentation_refuses_unsafe_changes_atomically(tmp_path, protect):
    from jieyi.workflow.services import create_job, resegment_pdf_document

    store, doc, book = _old_pdf_document(tmp_path)
    first = store.list_segments(doc.id)[0]
    if protect == "translation":
        store.save_segment_draft(first.id, "已有校对内容")
    elif protect == "source_edit":
        store.update_segment_source(first.id, "The manually corrected source text")
    elif protect == "manual":
        with store._connect() as connection:
            connection.execute(
                "UPDATE segments SET segmenter_version = 'manual-v2' WHERE id = ?", (first.id,)
            )
    else:
        create_job(store, document_id=doc.id, draft_provider="echo", draft_model="dry-run")
    before = store.list_segments(doc.id)
    metadata = store.get_pdf_metadata(doc.id)
    with pytest.raises(ValueError, match="已有译文|未完成"):
        resegment_pdf_document(store, doc.id, book=book)
    assert store.list_segments(doc.id) == before
    assert store.get_pdf_metadata(doc.id) == metadata
