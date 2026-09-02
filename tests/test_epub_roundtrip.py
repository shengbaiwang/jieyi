import asyncio
import io
import tempfile
import unittest
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from fastapi.testclient import TestClient

from jieyi.api.app import create_app
from jieyi.ingestion import extract_epub
from jieyi.ingestion.epub_reader import render_spine
from jieyi.ingestion.epub_roundtrip import parse_epub_archive
from jieyi.persistence import SQLiteStore
from jieyi.providers import EchoProvider, ProviderRegistry
from jieyi.workflow import (
    TranslationEngine,
    create_epub_document,
    create_job,
    create_project,
)


def build_roundtrip_epub(*, version: str = "3.0", fixed_layout: bool = True) -> bytes:
    container = b"""<?xml version="1.0"?>
<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container" version="1.0">
  <rootfiles><rootfile full-path="EPUB/package.opf"
    media-type="application/oebps-package+xml"/></rootfiles>
</container>"""
    if version == "2.0":
        cover_metadata = '<meta name="cover" content="cover-art"/>'
        cover_properties = ""
        fixed_metadata = '<meta name="fixed-layout" content="true"/>' if fixed_layout else ""
    else:
        cover_metadata = ""
        cover_properties = ' properties="cover-image"'
        fixed_metadata = '<meta property="rendition:layout">pre-paginated</meta>' if fixed_layout else ""
    package = f"""<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf" version="{version}">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:title>Round Trip Book</dc:title>
    {cover_metadata}
    {fixed_metadata}
  </metadata>
  <manifest>
    <item id="chapter" href="Text/chapter.xhtml" media-type="application/xhtml+xml"/>
    <item id="style" href="Styles/book.css" media-type="text/css"/>
    <item id="cover-art" href="Images/cover.svg" media-type="image/svg+xml"{cover_properties}/>
    <item id="font" href="Fonts/book.woff2" media-type="font/woff2"/>
    <item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>
  </manifest>
  <spine><itemref idref="chapter"/></spine>
</package>""".encode()
    chapter = b"""<html xmlns="http://www.w3.org/1999/xhtml">
<head>
  <base href="https://evil.example/"/>
  <link rel="stylesheet" href="../Styles/book.css"/>
  <script src="https://evil.example/tracker.js">alert(1)</script>
</head>
<body onload="steal()">
  <h1>Round <em>Trip</em></h1>
  <p onclick="steal()">Agency <strong>matters</strong>.<br/>
    <a href="https://evil.example/out">external</a>
    <a href="../nav.xhtml#toc">toc</a>
    <img src="../Images/cover.svg" srcset="https://evil.example/2x.png 2x" alt="cover"/>
  </p>
  <table><tr><td>Alpha</td><td>Beta</td></tr></table>
  <div class="poem"><p>First line<br/>Second line</p></div>
</body>
</html>"""
    css = b"""@font-face{font-family:Book;src:url('../Fonts/book.woff2')}
body{font-family:Book;background-image:url(https://evil.example/pixel.png)}
h1{color:#735;} @import url('nested.css'); @import url('https://evil.example/import.css');"""
    svg = b"""<svg xmlns="http://www.w3.org/2000/svg" width="120" height="160"
 onload="steal()"><script>alert(1)</script><rect width="120" height="160" fill="#735"/>
 <image href="https://evil.example/pixel.png"/></svg>"""
    nav = b"""<html xmlns="http://www.w3.org/1999/xhtml"><body>
<nav id="toc"><ol><li><a href="Text/chapter.xhtml">Chapter</a></li></ol></nav>
</body></html>"""

    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        archive.writestr("META-INF/container.xml", container)
        archive.writestr("EPUB/package.opf", package)
        archive.writestr("EPUB/Text/chapter.xhtml", chapter)
        archive.writestr("EPUB/Styles/book.css", css)
        archive.writestr("EPUB/Styles/nested.css", b"p{line-height:1.6}")
        archive.writestr("EPUB/Images/cover.svg", svg)
        archive.writestr("EPUB/Fonts/book.woff2", b"fake-font-data")
        archive.writestr("EPUB/nav.xhtml", nav)
    return output.getvalue()


class EpubRoundTripTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.store = SQLiteStore(Path(self.tempdir.name) / "roundtrip.db")
        self.store.migrate()
        self.project = create_project(
            self.store,
            name="Round trip",
            source_lang="en",
            target_lang="zh-CN",
        )

    def tearDown(self):
        self.tempdir.cleanup()

    def test_preserves_original_resources_cover_spine_and_text_node_mapping(self):
        payload = build_roundtrip_epub()
        document = create_epub_document(
            self.store,
            project_id=self.project.id,
            file_data=payload,
        )

        package = self.store.get_epub_package(document.id)
        self.assertEqual(package["cover_path"], "EPUB/Images/cover.svg")
        self.assertEqual(package["rendition_layout"], "pre-paginated")
        self.assertEqual(self.store.get_original_epub(document.id), payload)
        self.assertEqual(
            self.store.get_epub_resource(document.id, "EPUB/Fonts/book.woff2")["data"],
            b"fake-font-data",
        )
        self.assertTrue(self.store.list_epub_spine(document.id)[0]["fixed_layout"])
        mappings = self.store.list_epub_mappings(document.id)
        self.assertTrue(mappings["atoms"])
        self.assertTrue(mappings["text_nodes"])
        self.assertTrue(all(item["node_refs"] for item in mappings["atoms"]))
        self.assertTrue(
            all("::" in item["node_id"] for item in mappings["text_nodes"])
        )

    def test_reflowable_reader_exposes_segment_locations_to_parent(self):
        document = create_epub_document(
            self.store,
            project_id=self.project.id,
            file_data=build_roundtrip_epub(fixed_layout=False),
        )
        rendered, _ = render_spine(
            self.store, document.id, 0, mode="original", layout="comfort"
        )
        value = rendered.decode()
        self.assertIn("data-jy-segment-ordinals", value)
        self.assertIn("jy-epub-locate", value)
        self.assertIn("jy-epub-location", value)

    def test_epub2_meta_cover_is_resolved(self):
        payload = build_roundtrip_epub(version="2.0")
        book = extract_epub(payload)
        archive = parse_epub_archive(payload, book.source_atoms)
        self.assertEqual(archive.package_version, "2.0")
        self.assertEqual(archive.cover_path, "EPUB/Images/cover.svg")
        self.assertEqual(archive.rendition_layout, "pre-paginated")

    def test_same_hash_resource_completion_preserves_translations_and_decisions(self):
        payload = build_roundtrip_epub()
        document = create_epub_document(
            self.store,
            project_id=self.project.id,
            file_data=payload,
        )
        first = self.store.list_segments(document.id)[0]
        self.store.confirm_segment(first.id, "往返", rationale="checked")

        imported_again = create_epub_document(
            self.store,
            project_id=self.project.id,
            file_data=payload,
        )

        self.assertEqual(imported_again.id, document.id)
        reloaded = self.store.get_segment(first.id)
        self.assertEqual(reloaded.accepted_translation, "往返")
        self.assertEqual(len(self.store.list_audit_events("segment", first.id)), 1)
        self.assertEqual(self.store.get_original_epub(document.id), payload)

    def test_translation_returns_per_atom_and_preserves_inline_markup(self):
        payload = build_roundtrip_epub()
        document = create_epub_document(
            self.store,
            project_id=self.project.id,
            file_data=payload,
        )
        registry = ProviderRegistry()
        registry.register("echo", EchoProvider())
        engine = TranslationEngine(self.store, registry)
        job = create_job(
            self.store,
            document_id=document.id,
            draft_provider="echo",
            draft_model="dry-run",
            review_policy="never",
        )

        asyncio.run(engine.run(job.id, max_segments=2))

        second = self.store.list_segments(document.id)[1]
        structured = self.store.epub_structured_translation(second.id)
        self.assertIsNotNone(structured)
        self.assertIn("strong", structured)
        reattached = create_epub_document(
            self.store,
            project_id=self.project.id,
            file_data=payload,
        )
        self.assertEqual(reattached.id, document.id)
        self.assertIsNotNone(self.store.epub_structured_translation(second.id))
        atoms = self.store.list_epub_atoms_for_segment(second.id)
        self.assertGreaterEqual(len(atoms), 1)
        translated, _ = render_spine(
            self.store,
            document.id,
            0,
            mode="translated",
            layout="faithful",
        )
        self.assertIn(b"strong", translated)
        self.assertIn(b"jy-source-italic", translated)
        self.assertIn(b"cover.svg", translated)
        self.assertIn(b"table", translated)
        bilingual, _ = render_spine(
            self.store,
            document.id,
            0,
            mode="bilingual",
            layout="faithful",
        )
        bilingual_text = bilingual.decode()
        self.assertIn("jy-bilingual-pair", bilingual_text)
        self.assertIn("jy-original", bilingual_text)
        self.assertIn("jy-translation", bilingual_text)
        self.assertIn("jy-source-italic", bilingual_text)
        self.assertIn("font-family: \"Kaiti SC\"", bilingual_text)
        self.assertIn("font-style: normal !important", bilingual_text)
        self.assertIn(
            "grid-template-columns: minmax(0, 1fr) minmax(0, 1fr)",
            bilingual_text,
        )
        self.assertNotIn('content: "原文"', bilingual_text)
        self.assertNotIn('content: "译文"', bilingual_text)
        self.assertLess(
            bilingual_text.index("jy-original"),
            bilingual_text.index("jy-translation"),
        )
        root = ET.fromstring(bilingual)
        namespace = "{http://www.w3.org/1999/xhtml}"
        self.assertIsNotNone(
            root.find(f".//{namespace}style[@data-jy-reader='true']")
        )
        pair = next(
            element
            for element in root.iter()
            if "jy-bilingual-pair" in element.attrib.get("class", "").split()
        )
        self.assertEqual(
            [child.tag for child in pair],
            [namespace + "span", namespace + "span"],
        )

    def test_bilingual_reader_keeps_a_right_column_for_missing_translations(self):
        document = create_epub_document(
            self.store,
            project_id=self.project.id,
            file_data=build_roundtrip_epub(),
        )

        bilingual, _ = render_spine(
            self.store,
            document.id,
            0,
            mode="bilingual",
            layout="faithful",
        )

        value = bilingual.decode()
        self.assertIn("jy-bilingual-pair", value)
        self.assertIn("jy-translation jy-missing-translation", value)
        self.assertIn("〔尚未翻译〕", value)


class EpubReaderApiTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.client = TestClient(create_app(str(Path(self.tempdir.name) / "api.db")))
        self.project = self.client.post(
            "/projects",
            json={"name": "Reader", "source_lang": "en", "target_lang": "zh-CN"},
        ).json()
        response = self.client.post(
            f"/projects/{self.project['id']}/documents/epub",
            content=build_roundtrip_epub(),
            headers={"Content-Type": "application/epub+zip"},
        )
        self.assertEqual(response.status_code, 201)
        self.document = response.json()

    def tearDown(self):
        self.client.close()
        self.tempdir.cleanup()

    def test_reader_manifest_and_original_download(self):
        manifest = self.client.get(f"/documents/{self.document['id']}/epub")
        self.assertEqual(manifest.status_code, 200)
        self.assertTrue(manifest.json()["cover_url"].endswith("EPUB/Images/cover.svg"))
        self.assertEqual(manifest.json()["modes"], ["original", "translated", "bilingual"])
        self.assertTrue(manifest.json()["segment_locations"])
        self.assertEqual(
            set(manifest.json()["segment_locations"][0]),
            {"segment_ordinal", "spine_index"},
        )
        original = self.client.get(
            f"/documents/{self.document['id']}/epub/original"
        )
        self.assertEqual(original.content, build_roundtrip_epub())

    def test_translated_book_export_is_a_valid_epub_with_original_assets(self):
        segments = self.client.get(
            f"/documents/{self.document['id']}/segments"
        ).json()
        confirmed = self.client.patch(
            f"/segments/{segments[0]['id']}/confirm",
            json={"translation": "往返书籍", "rationale": "export test"},
        )
        self.assertEqual(confirmed.status_code, 200)

        exported = self.client.get(
            f"/documents/{self.document['id']}/export",
            params={"format": "book"},
        )
        self.assertEqual(exported.status_code, 200)
        self.assertEqual(exported.headers["content-type"], "application/epub+zip")
        with zipfile.ZipFile(io.BytesIO(exported.content)) as archive:
            self.assertEqual(archive.namelist()[0], "mimetype")
            self.assertEqual(
                archive.getinfo("mimetype").compress_type,
                zipfile.ZIP_STORED,
            )
            chapter = archive.read("EPUB/Text/chapter.xhtml").decode()
            self.assertIn("往返书籍", chapter)
            self.assertEqual(
                archive.read("EPUB/Fonts/book.woff2"),
                b"fake-font-data",
            )

    def test_rendered_xhtml_and_resources_are_sanitized(self):
        rendered = self.client.get(
            f"/documents/{self.document['id']}/epub/spine/0",
            params={"mode": "original", "layout": "faithful"},
        )
        self.assertEqual(rendered.status_code, 200)
        value = rendered.text.casefold()
        self.assertNotIn("<script", value)
        self.assertNotIn("onload=", value)
        self.assertNotIn("onclick=", value)
        self.assertNotIn("https://evil.example", value)
        self.assertIn("/epub/resources/epub/images/cover.svg", value)
        self.assertIn("/epub/content/epub/nav.xhtml#toc", value)
        self.assertIn("default-src 'none'", rendered.headers["content-security-policy"])
        self.assertIn("script-src 'sha256-", rendered.headers["content-security-policy"])
        self.assertIn("sandbox allow-scripts", rendered.headers["content-security-policy"])

        target = self.client.get(
            f"/documents/{self.document['id']}/epub/spine/0",
            params={"mode": "translated", "layout": "faithful"},
        )
        self.assertIn("尚未翻译", target.text)
        self.assertNotIn("Agency", target.text)

        linked = self.client.get(
            f"/documents/{self.document['id']}/epub/content/EPUB/nav.xhtml"
        )
        self.assertEqual(linked.status_code, 200)
        self.assertIn("script-src 'none'", linked.headers["content-security-policy"])

        css = self.client.get(
            f"/documents/{self.document['id']}/epub/resources/EPUB/Styles/book.css"
        )
        self.assertEqual(css.status_code, 200)
        self.assertNotIn("evil.example", css.text)
        self.assertIn(
            "/epub/resources/EPUB/Styles/nested.css",
            css.text,
        )
        self.assertIn("/epub/resources/EPUB/Fonts/book.woff2", css.text)
        nested = self.client.get(
            f"/documents/{self.document['id']}/epub/resources/EPUB/Styles/nested.css"
        )
        self.assertEqual(nested.status_code, 200)

        svg = self.client.get(
            f"/documents/{self.document['id']}/epub/resources/EPUB/Images/cover.svg"
        )
        self.assertEqual(svg.status_code, 200)
        self.assertNotIn("<script", svg.text.casefold())
        self.assertNotIn("onload=", svg.text.casefold())
        self.assertNotIn("evil.example", svg.text.casefold())

    def test_wrong_hash_cannot_replace_existing_book(self):
        different = build_roundtrip_epub(version="2.0")
        response = self.client.put(
            f"/documents/{self.document['id']}/epub/source",
            content=different,
            headers={"Content-Type": "application/epub+zip"},
        )
        self.assertEqual(response.status_code, 409)


if __name__ == "__main__":
    unittest.main()
