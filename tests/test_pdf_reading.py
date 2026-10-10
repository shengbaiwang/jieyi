import io

from fastapi.testclient import TestClient
from pypdf import PdfWriter
from pypdf.annotations import Link, Text
from pypdf.generic import Fit
from test_pdf import pdf_fixture

from jieyi.api.app import create_app
from jieyi.ingestion.pdf_reading import pdf_page_interactions, pdf_page_text


def linked_pdf():
    writer = PdfWriter(clone_from=io.BytesIO(pdf_fixture()))
    writer.add_annotation(0, Link(rect=(40, 710, 110, 730), target_page_index=2,
                                  fit=Fit.xyz(left=40, top=640, zoom=0)))
    writer.add_annotation(2, Link(rect=(40, 710, 110, 730), target_page_index=0))
    writer.add_annotation(0, Link(rect=(40, 650, 110, 670), url="javascript:alert(1)"))
    writer.add_annotation(2, Text(rect=(150, 600, 170, 620), text="An original note."))
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


def test_internal_links_and_original_notes_are_passive():
    data = linked_pdf()
    first = pdf_page_interactions(data, 1)
    assert len(first["links"]) == 1
    assert first["links"][0]["page"] == 3
    assert first["links"][0]["offset"] == 0.2
    assert first["links"][0]["rect"] == [40 / 600, 1 - 730 / 800, 70 / 600, 20 / 800]
    assert "javascript" not in str(first)
    last = pdf_page_interactions(data, 3)
    assert last["links"][0]["page"] == 1
    assert last["annotations"][0]["text"] == "An original note."


def test_original_text_offsets_and_geometry_are_selectable():
    result = pdf_page_text(pdf_fixture(), 2)
    assert result["width"] == 600
    assert result["height"] == 800
    assert "national trade" in result["text"]
    for line in result["lines"]:
        assert result["text"][line["start"]:line["end"]] == line["text"]
        x, y, width, height = line["rect"]
        assert 0 <= x < 1 and 0 <= y < 1 and width > 0 and height > 0


def test_reader_api_search_pagination_text_and_translation(tmp_path):
    with TestClient(create_app(str(tmp_path / "reading.db"))) as client:
        project = client.post("/projects", json={"name": "PDF Reading", "source_lang": "en",
                                                 "target_lang": "zh-CN"}).json()
        document = client.post(f"/projects/{project['id']}/documents/pdf", content=linked_pdf()).json()
        root = f"/documents/{document['id']}/pdf"
        text = client.get(root + "/pages/1/text")
        assert text.status_code == 200, text.text
        assert text.json()["links"][0]["page"] == 3
        assert text.json()["mode"] == "original"
        assert client.get(root + "/pages/0/text").status_code == 404
        assert client.get(root + "/pages/4/text").status_code == 404
        assert client.get(root + "/pages/1/text?mode=executable").status_code == 422
        search = client.get(root + "/search", params={"q": "History", "limit": 1}).json()
        assert search["total"] == 3
        assert len(search["results"]) == 1
        assert search["results"][0]["page"] == 1
        assert client.get(root + "/search", params={"q": "History", "offset": 1, "limit": 1}).json()["results"][0]["page"] == 2
        # No regular expression syntax or wildcard search is executed.
        assert client.get(root + "/search", params={"q": ".*"}).json()["total"] == 0
        assert client.get(root + "/search", params={"q": "   "}).status_code == 422
        assert client.get(root + "/search", params={"q": "History", "limit": 101}).status_code == 422
        hit = client.get(root + "/search", params={"q": "NATIONAL trade"}).json()["results"][0]
        page_text = client.get(root + "/pages/2/text").json()["text"]
        assert page_text[hit["start"]:hit["end"]] == hit["quote"]
        segments = client.get(f"/documents/{document['id']}/segments").json()
        translation = "波斯湾连接沿岸社区和国际贸易。"
        assert client.patch(f"/segments/{segments[0]['id']}/draft", json={"translation": translation}).status_code == 200
        translated = client.get(root + "/pages/1/text?mode=translated")
        assert translated.status_code == 200, translated.text
        assert "波斯湾" in translated.json()["text"]
        assert translated.json()["links"] == []
        assert client.get(root + "/search", params={"q": "波斯湾", "mode": "translated"}).json()["total"] == 1
        assert client.patch(f"/segments/{segments[0]['id']}/draft", json={"translation": "全新译文用于检索。"}).status_code == 200
        assert client.get(root + "/search", params={"q": "波斯湾", "mode": "translated"}).json()["total"] == 0
        assert "全新译文" in client.get(root + "/pages/1/text?mode=translated").json()["text"]
        assert client.delete(f"/documents/{document['id']}").status_code == 204
        assert client.get(root + "/search", params={"q": "History"}).status_code == 404
        assert client.get(root + "/pages/1/text").status_code == 404
