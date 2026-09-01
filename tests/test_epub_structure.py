import io
import tempfile
import unittest
import zipfile
from pathlib import Path

from jieyi.domain.models import SegmentKind
from jieyi.ingestion import extract_epub
from jieyi.persistence import SQLiteStore
from jieyi.workflow import create_epub_document, create_project


def build_structural_epub() -> bytes:
    container = b"""<?xml version="1.0"?>
<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container" version="1.0">
  <rootfiles><rootfile full-path="OPS/content.opf"
    media-type="application/oebps-package+xml"/></rootfiles>
</container>"""
    package = b"""<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:title>Structural EPUB</dc:title>
  </metadata>
  <manifest>
    <item id="chapter" href="chapter.xhtml" media-type="application/xhtml+xml"/>
    <item id="style" href="book.css" media-type="text/css"/>
  </manifest>
  <spine><itemref idref="chapter"/></spine>
</package>"""
    css = b"""
h1 > span, address > span { display: block; }
.continuation { text-indent: 1em; }
.hidden { display: none; }
"""
    chapter = b"""<html xmlns="http://www.w3.org/1999/xhtml"
 xmlns:epub="http://www.idpf.org/2007/ops">
<head><link rel="stylesheet" href="book.css"/></head>
<body>
  <h1><span class="stack">Materialism I:</span>
      <span class="stack">The Pursuit of Profit</span></h1>
  <div>Before <p>Nested paragraph.</p> after the nested block.</div>
  <p class="continuation">Sentence split across</p>
  <p class="continuation">two layout blocks.</p>
  <p class="continuation">Un 63</p>
  <p class="continuation">Un 70</p>
  <address><span class="stack">8 Blackstock Mews</span>
           <span class="stack">London N4 2BT</span></address>
  <ul><li>First item</li><li>Second item</li></ul>
  <table><tr><td>Alpha</td><td>Beta</td></tr></table>
  <p>Visible <span class="hidden">duplicate</span>text.</p>
  <p>Chapter body<span epub:type="pagebreak"></span> continues.</p>
</body></html>"""
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        archive.writestr("META-INF/container.xml", container)
        archive.writestr("OPS/content.opf", package)
        archive.writestr("OPS/book.css", css)
        archive.writestr("OPS/chapter.xhtml", chapter)
    return output.getvalue()


class EpubStructureTests(unittest.TestCase):
    def test_reflows_css_fragments_without_overmerging_short_observations(self):
        book = extract_epub(build_structural_epub())
        texts = [block.text for block in book.blocks]
        self.assertIn("Materialism I:\nThe Pursuit of Profit", texts)
        self.assertIn("Sentence split across two layout blocks.", texts)
        self.assertIn("8 Blackstock Mews\nLondon N4 2BT", texts)
        self.assertIn("Un 63", texts)
        self.assertIn("Un 70", texts)
        self.assertNotIn("Un 63 Un 70", texts)
        self.assertIn("Visible text.", texts)
        self.assertNotIn("duplicate", " ".join(texts))

    def test_preserves_mixed_content_lists_tables_and_source_provenance(self):
        book = extract_epub(build_structural_epub())
        by_text = {block.text: block for block in book.blocks}
        self.assertIn("Before", by_text)
        self.assertIn("Nested paragraph.", by_text)
        self.assertIn("after the nested block.", by_text)
        self.assertEqual(by_text["First item"].kind, SegmentKind.LIST_ITEM)
        self.assertEqual(by_text["Second item"].kind, SegmentKind.LIST_ITEM)
        self.assertEqual(by_text["Alpha"].kind, SegmentKind.TABLE_CELL)
        self.assertEqual(by_text["Beta"].kind, SegmentKind.TABLE_CELL)
        merged = by_text["Sentence split across two layout blocks."]
        self.assertEqual(len(merged.source_refs), 2)
        self.assertIn("lowercase-continuation", merged.segmentation_reason)
        self.assertTrue(book.source_atoms)
        self.assertTrue(book.boundaries)

    def test_database_roundtrip_keeps_segmentation_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteStore(Path(directory) / "test.db")
            store.migrate()
            project = create_project(
                store,
                name="Structural",
                source_lang="en",
                target_lang="zh-CN",
            )
            document = create_epub_document(
                store,
                project_id=project.id,
                file_data=build_structural_epub(),
            )
            segments = store.list_segments(document.id)
            merged = next(
                item
                for item in segments
                if item.source_text == "Sentence split across two layout blocks."
            )
            self.assertEqual(len(merged.source_refs), 2)
            self.assertEqual(merged.segmenter_version, "epub-structure-v1")
            self.assertGreaterEqual(merged.segmentation_confidence, 0.5)


if __name__ == "__main__":
    unittest.main()
