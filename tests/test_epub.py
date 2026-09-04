import io
import json
import tempfile
import unittest
import zipfile
from contextlib import redirect_stdout
from pathlib import Path

from jieyi.cli import main as cli_main
from jieyi.domain.models import SegmentKind
from jieyi.ingestion import EpubIngestionError, extract_epub
from jieyi.ingestion.epub_navigation import parse_epub_navigation
from jieyi.persistence import SQLiteStore
from jieyi.workflow import create_project


def build_epub(*, malicious_member: bool = False, doctype: bytes = b"") -> bytes:
    container = b"""<?xml version="1.0"?>
<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container" version="1.0">
  <rootfiles><rootfile full-path="OPS/content.opf"
    media-type="application/oebps-package+xml"/></rootfiles>
</container>"""
    package = b"""<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:title>Test Theory Book</dc:title>
  </metadata>
  <manifest>
    <item id="chapter-two" href="two.xhtml" media-type="application/xhtml+xml"/>
    <item id="chapter-one" href="one.xhtml" media-type="application/xhtml+xml"/>
    <item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>
    <item id="cover" href="cover.png" media-type="image/png" properties="cover-image"/>
  </manifest>
  <spine>
    <itemref idref="chapter-two"/>
    <itemref idref="nav"/>
    <itemref idref="chapter-one"/>
  </spine>
</package>"""
    chapter_two = doctype + b"""<html xmlns="http://www.w3.org/1999/xhtml"
 xmlns:epub="http://www.idpf.org/2007/ops"><body>
  <h1><a id="second"/>Second Chapter</h1>
  <p>Agency <em>matters</em>.&nbsp;Still.</p>
  <blockquote><p>A cited passage.</p></blockquote>
  <aside epub:type="footnote"><p>Footnote text.</p></aside>
</body></html>"""
    chapter_one = doctype + b"""<html xmlns="http://www.w3.org/1999/xhtml"><body>
  <h1><a id="first"/>First Chapter</h1><p>Structure follows.</p>
</body></html>"""
    nav = doctype + b"""<html xmlns="http://www.w3.org/1999/xhtml"
 xmlns:epub="http://www.idpf.org/2007/ops"><body>
<nav epub:type="toc"><ol>
<li><a href="two.xhtml#second">Second from TOC</a></li>
<li><a href="one.xhtml#first">First from TOC</a></li>
</ol></nav></body></html>"""

    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        archive.writestr("META-INF/container.xml", container)
        archive.writestr("OPS/content.opf", package)
        archive.writestr("OPS/two.xhtml", chapter_two)
        archive.writestr("OPS/one.xhtml", chapter_one)
        archive.writestr("OPS/cover.png", b"\x89PNG\r\n\x1a\ncover")
        archive.writestr("OPS/nav.xhtml", nav)
        if malicious_member:
            archive.writestr("../outside.txt", "unsafe")
    return output.getvalue()


class EpubTests(unittest.TestCase):
    def test_extracts_metadata_and_blocks_in_spine_order(self):
        book = extract_epub(build_epub())
        self.assertEqual(book.title, "Test Theory Book")
        self.assertEqual(book.spine_items, ("OPS/two.xhtml", "OPS/one.xhtml"))
        self.assertEqual(
            [block.text for block in book.blocks],
            [
                "Second Chapter",
                "Agency matters. Still.",
                "A cited passage.",
                "Footnote text.",
                "First Chapter",
                "Structure follows.",
            ],
        )
        self.assertEqual(book.blocks[2].kind, SegmentKind.BLOCKQUOTE)
        self.assertEqual(book.blocks[3].kind, SegmentKind.FOOTNOTE)
        self.assertNotIn("Duplicate TOC", [block.text for block in book.blocks])
        self.assertEqual([item.label for item in book.navigation], ["Second from TOC", "First from TOC"])
        self.assertEqual(book.blocks[0].heading_path, "Second from TOC")
        self.assertEqual(book.blocks[-1].heading_path, "First from TOC")

    def test_epub2_ncx_navigation_accepts_standard_external_dtd(self):
        ncx = b'''<?xml version="1.0"?>
<!DOCTYPE ncx PUBLIC "-//NISO//DTD ncx 2005-1//EN" "http://www.daisy.org/z3986/2005/ncx-2005-1.dtd">
<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/"><navMap>
<navPoint><navLabel><text>Part</text></navLabel><content src="chapter.xhtml#part"/>
<navPoint><navLabel><text>1</text></navLabel><content src="chapter.xhtml#one"/></navPoint>
</navPoint></navMap></ncx>'''
        entries = parse_epub_navigation(ncx, "OPS/toc.ncx")
        self.assertEqual([(item.label, item.level) for item in entries], [("Part", 0), ("1", 1)])
        self.assertEqual(entries[1].path, "OPS/chapter.xhtml")

    def test_epub2_accepts_standard_xhtml_11_external_dtd(self):
        doctype = b'''<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.1//EN"
"http://www.w3.org/TR/xhtml11/DTD/xhtml11.dtd">'''
        expected = extract_epub(build_epub())
        actual = extract_epub(build_epub(doctype=doctype))
        self.assertEqual(actual, expected)

    def test_rejects_archive_path_traversal(self):
        with self.assertRaisesRegex(EpubIngestionError, "Unsafe EPUB member path"):
            extract_epub(build_epub(malicious_member=True))

    def test_rejects_dtd_and_entity_declarations(self):
        with self.assertRaisesRegex(EpubIngestionError, "DTD or entity"):
            extract_epub(build_epub(
                doctype=b'<!DOCTYPE html [<!ENTITY xxe SYSTEM "file:///unused.txt">]>'
            ))

    def test_plain_html_doctype_preserves_content_and_navigation(self):
        expected = extract_epub(build_epub())
        actual = extract_epub(build_epub(doctype=b"<!DOCTYPE html>"))
        self.assertEqual(actual, expected)

    def test_cli_imports_epub_without_manual_format_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "cli.db"
            epub_path = root / "book.epub"
            epub_path.write_bytes(build_epub())
            store = SQLiteStore(database)
            store.migrate()
            project = create_project(
                store,
                name="CLI EPUB",
                source_lang="en",
                target_lang="zh-CN",
            )
            output = io.StringIO()
            with redirect_stdout(output):
                result = cli_main(
                    [
                        "--db",
                        str(database),
                        "document-import",
                        "--project",
                        project.id,
                        "--file",
                        str(epub_path),
                    ]
                )
            self.assertEqual(result, 0)
            payload = json.loads(output.getvalue())
            self.assertEqual(payload["source_format"], "epub")
            self.assertEqual(payload["title"], "Test Theory Book")


if __name__ == "__main__":
    unittest.main()
