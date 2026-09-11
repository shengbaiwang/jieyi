import io

import pytest
from fastapi.testclient import TestClient
from test_epub import build_epub

from jieyi.api.app import create_app
from jieyi.persistence.sqlite import SQLiteStore


@pytest.fixture
def workspace(tmp_path):
    path = str(tmp_path / "editing.db")
    with TestClient(create_app(path)) as client:
        project = client.post("/projects", json={"name": "Editing", "source_lang": "en", "target_lang": "zh-CN"}).json()
        yield client, project, SQLiteStore(path)


def document(client, project, text, source_format="markdown"):
    doc = client.post(f"/projects/{project['id']}/documents", json={"title": "Blocks", "text": text, "source_format": source_format}).json()
    return doc, client.get(f"/documents/{doc['id']}/segments").json()


def test_delete_restore_preserves_source_translation_and_order(workspace):
    client, project, store = workspace
    doc, items = document(client, project, "First.\n\nSecond.\n\nThird.")
    selected = items[1]
    client.patch(f"/segments/{selected['id']}/confirm", json={"translation": "第二段。"})
    result = client.delete(f"/segments/{selected['id']}")
    assert result.status_code == 200, result.text
    assert result.json()["segment_count"] == 2
    assert result.json()["segment"]["source_text"] == "Third."
    assert client.get(f"/segments/{selected['id']}/source").status_code == 404
    assert [s.ordinal for s in store.list_segments(doc['id'])] == [0, 1]
    result = client.post(f"/segments/{selected['id']}/restore")
    assert result.status_code == 200, result.text
    assert result.json()["segment"]["accepted_translation"] == "第二段。"
    assert result.json()["segment"]["status"] == "human_confirmed"
    assert [s.source_text for s in store.list_segments(doc['id'])] == ["First.", "Second.", "Third."]
    assert client.post(f"/segments/{selected['id']}/restore").status_code == 404


def test_delete_last_segment_and_restore(workspace):
    client, project, _ = workspace
    doc, items = document(client, project, "Only paragraph.")
    result = client.delete(f"/segments/{items[0]['id']}")
    assert result.status_code == 200
    assert result.json()["segment"] is None
    overview = client.get(f"/documents/{doc['id']}/overview").json()
    assert overview["segment_count"] == 0 and overview["chapters"] == []
    assert client.post(f"/segments/{items[0]['id']}/restore").json()["segment"]["ordinal"] == 0


@pytest.mark.parametrize("text,index,direction,kind,source", [
    ("# First\n\n# Second\n\nBody.", 0, "next", "heading", "First Second"),
    ("# First\n\n# Second\n\nBody.", 1, "previous", "heading", "First Second"),
    ("Body.\n\n# Heading\n\nMore.", 1, "previous", "paragraph", "Body.\n\nHeading"),
    ("# Heading\n\nBody.", 0, "next", "heading", "Heading Body."),
])
def test_headings_merge_and_outline_updates(workspace, text, index, direction, kind, source):
    client, project, _ = workspace
    doc, items = document(client, project, text)
    client.patch(f"/segments/{items[index]['id']}/confirm", json={"translation": "原译文"})
    response = client.post(f"/segments/{items[index]['id']}/merge", json={"direction": direction})
    assert response.status_code == 200, response.text
    merged = response.json()["segment"]
    assert merged["kind"] == kind and merged["source_text"] == source
    assert merged["edited_translation"] == "原译文" and merged["accepted_translation"] is None
    chapters = client.get(f"/documents/{doc['id']}/overview").json()["chapters"]
    assert len(chapters) == 1
    assert chapters[0]["title"] == (source if kind == "heading" else "正文")


def test_delete_heading_removes_outline_and_restores_it(workspace):
    client, project, _ = workspace
    doc, items = document(client, project, "# First\n\nBody.\n\n# Second\n\nMore.")
    selected = next(item for item in items if item['source_text'] == 'Second')
    client.delete(f"/segments/{selected['id']}")
    assert [c['title'] for c in client.get(f"/documents/{doc['id']}/overview").json()['chapters']] == ['First']
    client.post(f"/segments/{selected['id']}/restore")
    assert [c['title'] for c in client.get(f"/documents/{doc['id']}/overview").json()['chapters']] == ['First', 'Second']


def test_epub_delete_and_restore_shared_text_node(workspace):
    from jieyi.ingestion.epub_reader import _render_export_spine, render_spine
    client, project, store = workspace
    doc = client.post(f"/projects/{project['id']}/documents/epub", content=build_epub(), headers={"Content-Type": "application/epub+zip"}).json()
    items = client.get(f"/documents/{doc['id']}/segments").json()
    original = next(item for item in items if item['source_text'] == 'A cited passage.')
    split = client.post(f"/segments/{original['id']}/split", json={"source_text": original['source_text'], "selection_start": 2, "selection_end": 7}).json()['segment']
    spine = store.list_epub_atoms_for_segment(split['id'])[0]['spine_index']
    assert client.delete(f"/segments/{split['id']}").status_code == 200
    for mode in ('original', 'bilingual'):
        content, _ = render_spine(store, doc['id'], spine, mode=mode, layout="faithful")
        assert b'cited' not in content
        assert b'passage.' in content
    exported = _render_export_spine(store, doc['id'], spine, bilingual=True)
    assert b'cited' not in exported and b'passage.' in exported
    restored = client.post(f"/segments/{split['id']}/restore")
    assert restored.status_code == 200, restored.text
    content, _ = render_spine(store, doc['id'], spine, mode='original', layout='faithful')
    assert b'cited' in content


def test_pdf_deleted_source_does_not_reappear_in_export():
    from pypdf import PdfReader
    from test_pdf_export import fixture, image_hashes

    from jieyi.ingestion.pdf_export import compose_pdf
    source, metadata, segments = fixture()
    removed = segments[0]
    output, _ = compose_pdf(source, metadata, segments[1:], deleted_segments=[removed])
    before, after = PdfReader(io.BytesIO(source)), PdfReader(io.BytesIO(output))
    assert removed.source_text not in after.pages[0].extract_text()
    assert 'The Gulf connects' not in after.pages[0].extract_text()
    assert image_hashes(before) == image_hashes(after)


def test_restore_nested_heading_preserves_descendant_levels(workspace):
    client, project, store = workspace
    doc, items = document(client, project, "# First\n\n## Second\n\n### Third\n\nBody.")
    original_paths = [item.heading_path for item in store.list_segments(doc['id'])]
    selected = next(item for item in items if item['source_text'] == 'Second')
    client.delete(f"/segments/{selected['id']}")
    trash = client.get(f"/documents/{doc['id']}/deleted-segments").json()
    assert trash[0]['id'] == selected['id']
    response = client.post(f"/segments/{selected['id']}/restore")
    assert response.status_code == 200, response.text
    assert [item.heading_path for item in store.list_segments(doc['id'])] == original_paths


def test_epub_can_delete_original_shared_atom_and_merge_survivors(workspace):
    from jieyi.ingestion.epub_reader import render_spine
    client, project, store = workspace
    doc = client.post(f"/projects/{project['id']}/documents/epub", content=build_epub(), headers={"Content-Type": "application/epub+zip"}).json()
    original = next(item for item in client.get(f"/documents/{doc['id']}/segments").json() if item['source_text'] == 'A cited passage.')
    split = client.post(f"/segments/{original['id']}/split", json={"source_text": original['source_text'], "selection_start": 2, "selection_end": 7}).json()['segment']
    spine = store.list_epub_atoms_for_segment(split['id'])[0]['spine_index']
    client.delete(f"/segments/{original['id']}")
    content, _ = render_spine(store, doc['id'], spine, mode='original', layout='faithful')
    assert b'cited passage.' in content and b'A cited' not in content
    merged = client.post(f"/segments/{split['id']}/merge", json={"direction": "next"})
    assert merged.status_code == 200, merged.text
    restored = client.post(f"/segments/{original['id']}/restore")
    assert restored.status_code == 200, restored.text
    atom_ids = [a['segment_id'] for a in store.list_epub_atoms_for_spine(doc['id'], spine)]
    assert atom_ids.index(original['id']) < atom_ids.index(split['id'])
