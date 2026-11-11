"""Typography must survive language fallback, translation and PDF reflow."""

import io
import zipfile
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from xml.etree import ElementTree as ET

import pdfplumber
import pytest
from reportlab import __file__ as reportlab_file
from reportlab.pdfgen.canvas import Canvas
from test_epub_roundtrip import build_roundtrip_epub

from jieyi.ingestion.epub_reader import export_translated_epub, render_spine
from jieyi.ingestion.epub_typography import TranslationTypography
from jieyi.ingestion.pdf import GEOMETRY_VERSION, enrich_layout_fonts, extract_pdf
from jieyi.ingestion.pdf_export import compose_pdf
from jieyi.ingestion.plaintext import segments_from_blocks
from jieyi.ingestion.typography import font_family_role, pdf_font_style
from jieyi.persistence import SQLiteStore
from jieyi.workflow import create_epub_document, create_project


@pytest.mark.parametrize(
    ("family", "role"),
    [
        ('"Times New Roman", serif', "serif"),
        ('"Helvetica Neue", sans-serif', "sans"),
        ('"Source Han Sans SC", serif', "sans"),
        ('"Noto Serif CJK SC", sans-serif', "serif"),
        ('"Book Typeface", monospace', "mono"),
        ('"楷体", serif', "kai"),
        ('"Calibri", serif', "sans"),
        ('"Aptos", serif', "sans"),
        ('"Futura", serif', "sans"),
        ('"Palatino", sans-serif', "serif"),
    ],
)
def test_source_font_roles_respect_fallback_order(family, role):
    assert font_family_role(family) == role


def test_inherited_css_and_inline_emphasis_keep_distinct_translated_faces():
    root = ET.fromstring(
        '<html><body><p class="note">引文 <strong>重要</strong>'
        '<em>强调 <span style="font-style:normal">直立</span></em></p></body></html>'
    )
    typography = TranslationTypography(
        root, ["body {font-family:Helvetica,sans-serif}", ".note {font-style:oblique}"]
    )
    paragraph = root.find(".//p")
    typography.apply(paragraph, paragraph)
    assert paragraph.attrib["data-jy-typeface"] == "kai"
    assert paragraph.find("strong").attrib["data-jy-typeface"] == "kai"
    assert "font-weight: bold" in paragraph.find("strong").attrib["style"]
    assert paragraph.find(".//span").attrib["data-jy-typeface"] == "sans"
    assert "font-style: italic" in paragraph.attrib["style"]


def typography_book():
    chapter = b"""<html xmlns="http://www.w3.org/1999/xhtml"><head>
<link rel="stylesheet" href="../Styles/book.css"/></head><body>
<h1>Editorial hierarchy</h1><p id="body">Body <em>emphasis</em> <strong>weight</strong>.</p>
<p class="aside">A different voice.</p><p class="quote">A quoted passage.</p>
</body></html>"""
    css = b"""@import url('nested.css');
body {font-family: Georgia,serif} h1 {font-family:Helvetica,sans-serif;font-weight:700}
.aside {font-family:Helvetica,sans-serif} .quote {font:italic 1em Georgia,serif}"""
    result = io.BytesIO()
    with (
        zipfile.ZipFile(io.BytesIO(build_roundtrip_epub(fixed_layout=False))) as source,
        zipfile.ZipFile(result, "w") as target,
    ):
        for info in source.infolist():
            data = source.read(info)
            if info.filename == "EPUB/Text/chapter.xhtml":
                data = chapter
            elif info.filename == "EPUB/Styles/book.css":
                data = css
            target.writestr(info, data)
    return result.getvalue()


@pytest.mark.parametrize("target_lang", ["zh-CN", "fr"])
def test_epub_reader_and_exports_share_typography_without_styling_originals(tmp_path, target_lang):
    store = SQLiteStore(tmp_path / "book.db")
    store.migrate()
    project = create_project(store, name="Typography", source_lang="en", target_lang=target_lang)
    document = create_epub_document(store, project_id=project.id, file_data=typography_book())
    for segment in store.list_segments(document.id):
        # Retain the protected source markup, as the actual translation workflow does.
        plain = store.capture_epub_translation(
            segment.id, store.epub_translation_source(segment.id), "machine"
        )
        store.set_machine_translation(segment.id, plain)
    namespace = {"h": "http://www.w3.org/1999/xhtml"}
    original, _ = render_spine(store, document.id, 0, mode="original", layout="comfort")
    assert b"data-jy-typeface=" not in original
    for mode in ("translated", "bilingual"):
        rendered, _ = render_spine(store, document.id, 0, mode=mode, layout="comfort")
        with zipfile.ZipFile(
            io.BytesIO(export_translated_epub(store, document.id, bilingual=mode == "bilingual"))
        ) as archive:
            exported = archive.read("EPUB/Text/chapter.xhtml")
        for payload in (rendered, exported):
            root = ET.fromstring(payload)
            if target_lang == "fr":
                assert b"data-jy-typeface=" not in payload
                continue
            heading = root.find(".//h:h1", namespace)
            body = root.find('.//h:p[@id="body"]', namespace)
            aside = root.find('.//h:p[@class="aside"]', namespace)
            quote = root.find('.//h:p[@class="quote"]', namespace)
            if mode == "bilingual":
                # Grid wrappers must not change which source font rules apply.
                pairs = root.findall('.//h:span[@class="jy-translation"]', namespace)
                heading, body, aside, quote = pairs
                for node in root.findall('.//h:span[@class="jy-original"]', namespace):
                    assert all("data-jy-typeface" not in child.attrib for child in node.iter())
            assert heading.attrib["data-jy-typeface"] == "sans"
            assert "font-weight: 700" in heading.attrib["style"]
            assert body.attrib["data-jy-typeface"] == "serif"
            assert body.find("h:em", namespace).attrib["data-jy-typeface"] == "kai"
            assert "font-weight: bold" in body.find("h:strong", namespace).attrib["style"]
            assert aside.attrib["data-jy-typeface"] == "sans"
            assert quote.attrib["data-jy-typeface"] == "kai"
            assert "Kaiti SC" in quote.attrib["style"]


def test_pdf_distinguishes_fonts_in_legacy_layout_and_after_reflow(monkeypatch):
    # Bundled TrueType faces make the assertion portable; use Latin target text
    # so glyph coverage is independent of installed Chinese fonts.
    fonts = Path(reportlab_file).parent / "fonts"
    monkeypatch.setenv("JIEYI_PDF_FONT_SERIF", str(fonts / "Vera.ttf"))
    monkeypatch.setenv("JIEYI_PDF_FONT_SANS", str(fonts / "VeraIt.ttf"))
    monkeypatch.setenv("JIEYI_PDF_FONT_SERIF_BOLD", str(fonts / "VeraBd.ttf"))
    stream = io.BytesIO()
    canvas = Canvas(stream, pagesize=(400, 600))
    for y, font, text in [
        (530, "Times-Roman", "Body text in a serif face."),
        (480, "Helvetica", "A separate sans serif passage."),
        (430, "Times-Bold", "A bold passage with distinct weight."),
    ]:
        canvas.setFont(font, 12)
        canvas.drawString(35, y, text)
    canvas.save()
    source = stream.getvalue()
    book = extract_pdf(source)
    segments = segments_from_blocks("typography", list(book.blocks))
    legacy = deepcopy(book.layout)
    for item in legacy:
        for region in item["regions"]:
            del region["font_name"]
    assert [
        r["font_name"] for item in enrich_layout_fonts(source, legacy) for r in item["regions"]
    ] == ["Times-Roman", "Helvetica", "Times-Bold"]
    assert all("font_name" not in r for item in legacy for r in item["regions"])
    for long in (False, True):
        texts = ["Body translation.", "Sans translation.", "Bold translation."]
        if long:
            texts = [text * 6 for text in texts]
        translated = [
            replace(segment, machine_translation=text) for segment, text in zip(segments, texts)
        ]
        if long:
            # A source edit exercises actual page-local fallback with readable text.
            translated[0] = replace(translated[0], source_text=translated[0].source_text + " Revised.")
        output, report = compose_pdf(
            source, {"layout": legacy, "geometry_version": GEOMETRY_VERSION}, translated
        )
        assert bool(report["reflowed_pages"]) == long
        with pdfplumber.open(io.BytesIO(output)) as pdf:
            text = "".join(char["text"] for char in pdf.pages[0].chars)
            assert all(value.replace(" ", "") in text.replace(" ", "") for value in texts)
            faces = {char["fontname"] for char in pdf.pages[0].chars if char["text"].isalpha()}
            assert len(faces) == 3
        preview, _ = compose_pdf(source, {"layout": legacy}, translated, only_page=1)
        from jieyi.ingestion.pdf import render_pdf_page

        assert render_pdf_page(output, 1, 400) == render_pdf_page(preview, 1, 400)


def test_subset_font_names_retain_weight_and_emphasis():
    assert pdf_font_style("ABCDEF+Times-BoldItalic").bold
    assert pdf_font_style("ABCDEF+Times-BoldItalic").chinese_family == "kai"
    assert pdf_font_style("Helvetica-Oblique").italic


def test_css_font_priority_shorthand_reset_and_artwork_are_preserved():
    root = ET.fromstring(
        '<html><body><p style="font-family:serif"><em>强调</em>'
        '<span style="font-weight:bold;font:normal 1em Georgia,serif">正文</span>'
        "<svg><text>illustration</text></svg></p></body></html>"
    )
    typography = TranslationTypography(
        root, ["p {font-family:Helvetica,sans-serif !important}", "p em {font-weight:700}"]
    )
    source = root.find(".//p")
    target = ET.fromstring(ET.tostring(source))
    target.tag = "span"  # Bilingual wrapper still uses the original p selector.
    typography.apply(target, source)
    assert target.attrib["data-jy-typeface"] == "sans"
    assert "font-weight: 700" in target.find("em").attrib["style"]
    assert "font-weight: normal" in target.find("span").attrib["style"]
    assert target.find("svg").attrib == {}
    assert target.find("svg/text").attrib == {}
