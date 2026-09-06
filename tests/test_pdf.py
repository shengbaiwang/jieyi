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
        exported = client.get(root + "/export?format=book&bilingual=true")
        assert "波斯湾连接" in exported.text
        assert "Persian Gulf" in exported.text
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
