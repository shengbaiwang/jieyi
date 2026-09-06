import hashlib
import io
from dataclasses import replace

from PIL import Image, ImageChops
from pypdf import PdfReader
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen.canvas import Canvas

from jieyi.ingestion.pdf import extract_pdf, render_pdf_page
from jieyi.ingestion.pdf_export import compose_pdf
from jieyi.ingestion.plaintext import segments_from_blocks


def fixture():
    stream = io.BytesIO()
    canvas = Canvas(stream, pagesize=(400, 600))
    canvas.setTitle("Illustrated Gulf")
    canvas.bookmarkPage("chapter")
    canvas.addOutlineEntry("Introduction", "chapter")
    canvas.setFont("Helvetica", 12)
    canvas.drawString(35, 530, "The Gulf connects people across the ocean.")
    canvas.drawString(35, 510, "These communities share a long history of trade.")
    canvas.setStrokeColorRGB(0.1, 0.3, 0.8)
    canvas.rect(15, 15, 370, 570, fill=0)
    image = Image.new("RGB", (100, 80), (30, 130, 190))
    canvas.drawImage(ImageReader(image), 40, 220, 200, 160)
    canvas.showPage()
    canvas.setPageSize((500, 400))
    canvas.drawImage(ImageReader(image), 50, 50, 300, 240)
    canvas.save()
    source = stream.getvalue()
    book = extract_pdf(source)
    segments = segments_from_blocks("test", list(book.blocks))
    return source, {"layout": book.layout, "navigation": book.navigation}, segments


def image_hashes(reader):
    return [
        [hashlib.sha256(image.image.tobytes()).hexdigest() for image in page.images]
        for page in reader.pages
    ]


def test_native_layout_keeps_images_geometry_and_outline():
    source, metadata, segments = fixture()
    segments[0] = replace(segments[0], machine_translation="海湾连接着海洋两岸的人们。")
    result, report = compose_pdf(source, metadata, segments)
    assert report["replaced_blocks"] == 1, report
    assert report["overflow_count"] == 0
    before, after = PdfReader(io.BytesIO(source)), PdfReader(io.BytesIO(result))
    assert len(after.pages) == len(before.pages) == 2
    assert image_hashes(before) == image_hashes(after)
    assert [list(p.mediabox) for p in before.pages] == [list(p.mediabox) for p in after.pages]
    assert after.outline[0].title == "Introduction"
    assert "海湾连接" in after.pages[0].extract_text()
    assert "The Gulf connects" not in after.pages[0].extract_text()
    # Modified pages must still render their original image and border pixels.
    modified = Image.open(io.BytesIO(render_pdf_page(result, 1, 400))).convert("RGB")
    original_page = Image.open(io.BytesIO(render_pdf_page(source, 1, 400))).convert("RGB")
    assert (
        ImageChops.difference(
            modified.crop((0, 150, 400, 600)), original_page.crop((0, 150, 400, 600))
        ).getbbox()
        is None
    )
    assert modified.crop((35, 55, 320, 90)).getextrema()[0][0] < 100
    # An image-only page has exactly the same pixels, not a recreated approximation.
    original_image = Image.open(io.BytesIO(render_pdf_page(source, 2, 600))).convert("RGB")
    translated_image = Image.open(io.BytesIO(render_pdf_page(result, 2, 600))).convert("RGB")
    assert ImageChops.difference(original_image, translated_image).getbbox() is None
    assert "/FontFile2" in str(after.pages[0]["/Resources"]["/Font"].get_object()) or any(
        "/FontFile2" in str(font.get_object().get("/FontDescriptor", {}).get_object())
        for font in after.pages[0]["/Resources"]["/Font"].values()
        if font.get_object().get("/FontDescriptor")
    )


def test_long_translation_is_complete_in_appendix_and_leaves_source_untouched():
    source, metadata, segments = fixture()
    text = "海湾沿岸的交流与历史。" * 350 + "结束标记。"
    segments[0] = replace(segments[0], machine_translation=text)
    result, report = compose_pdf(source, metadata, segments)
    after = PdfReader(io.BytesIO(result))
    assert report["overflow_count"] == 1
    assert len(after.pages) > 2
    assert "The Gulf connects" in after.pages[0].extract_text()
    text_exported = "".join(
        "".join(page.extract_text().splitlines()[2:]) for page in after.pages[2:]
    )
    assert text in text_exported
    assert image_hashes(after)[:2] == image_hashes(PdfReader(io.BytesIO(source)))


def test_bilingual_pairs_original_and_translated_pages():
    source, metadata, segments = fixture()
    segments[0] = replace(segments[0], machine_translation="海湾地区的历史交流。")
    result, _ = compose_pdf(source, metadata, segments, bilingual=True)
    reader = PdfReader(io.BytesIO(result))
    assert len(reader.pages) == 4
    assert "The Gulf connects" in reader.pages[0].extract_text()
    assert "海湾地区" in reader.pages[1].extract_text()
    assert image_hashes(reader)[0] == image_hashes(reader)[1]
    assert image_hashes(reader)[2] == image_hashes(reader)[3]


def test_stale_source_mapping_falls_back_without_discarding_translation():
    source, metadata, segments = fixture()
    segments[0] = replace(
        segments[0], source_text="Manually edited text.", machine_translation="修改后的原文译文。"
    )
    result, report = compose_pdf(source, metadata, segments)
    assert report["overflow_count"] == 1
    assert "修改后的原文译文" in PdfReader(io.BytesIO(result)).pages[-1].extract_text()


def test_centered_title_stays_centered_after_translation():
    import pdfplumber

    stream = io.BytesIO()
    canvas = Canvas(stream, pagesize=(400, 600))
    canvas.setFont("Helvetica", 14)
    canvas.drawCentredString(200, 500, "A History of the Gulf and Its People")
    canvas.save()
    source = stream.getvalue()
    book = extract_pdf(source)
    segments = segments_from_blocks("test", list(book.blocks))
    segments[0] = replace(segments[0], machine_translation="海湾与人民的历史")
    result, report = compose_pdf(source, {"layout": book.layout}, segments)
    assert report["replaced_blocks"] == 1
    with pdfplumber.open(io.BytesIO(result)) as pdf:
        chars = pdf.pages[0].chars
        center = (min(c["x0"] for c in chars) + max(c["x1"] for c in chars)) / 2
        assert abs(center - 200) < 1


def _book_fixture(draw):
    stream = io.BytesIO()
    canvas = Canvas(stream, pagesize=(400, 600))
    canvas.setFont("Helvetica", 12)
    draw(canvas)
    canvas.save()
    source = stream.getvalue()
    book = extract_pdf(source)
    return source, {"layout": book.layout}, segments_from_blocks("test", list(book.blocks))


def test_indentation_and_trailing_spaces_do_not_break_object_matching():
    def draw(canvas):
        canvas.drawString(35, 530, "    Includes bibliographical references and index.    ")
        canvas.drawString(35, 500, "Another original paragraph stays here.")

    source, metadata, segments = _book_fixture(draw)
    segments[0] = replace(segments[0], machine_translation="附有参考文献和索引。")
    result, report = compose_pdf(source, metadata, segments)
    assert report["overflow_count"] == 0
    text = PdfReader(io.BytesIO(result)).pages[0].extract_text()
    assert "附有参考文献和索引" in text
    assert "Includes bibliographical" not in text
    assert "Another original paragraph" in text


def test_longer_centered_credit_uses_available_space_on_the_same_page():
    def draw(canvas):
        canvas.drawCentredString(200, 530, "Edited by")
        canvas.drawCentredString(200, 500, "Lawrence G. Potter, the editor of this history")

    source, metadata, segments = _book_fixture(draw)
    segments[0] = replace(segments[0], machine_translation="劳伦斯·G·波特 编")
    result, report = compose_pdf(source, metadata, segments)
    assert report["overflow_count"] == 0
    assert len(PdfReader(io.BytesIO(result)).pages) == 1
    assert "劳伦斯" in PdfReader(io.BytesIO(result)).pages[0].extract_text()


def test_duplicate_source_text_maps_to_its_own_occurrence():
    import pdfplumber

    def draw(canvas):
        canvas.drawString(35, 530, "Shared note about the Gulf and its people.")
        canvas.drawString(35, 470, "Shared note about the Gulf and its people.")

    source, metadata, segments = _book_fixture(draw)
    assert len(segments) == 2
    segments[1] = replace(segments[1], machine_translation="第二处注释。")
    result, report = compose_pdf(source, metadata, segments)
    assert report["overflow_count"] == 0
    with pdfplumber.open(io.BytesIO(result)) as pdf:
        chinese = [c for c in pdf.pages[0].chars if c["text"] == "第"]
        assert len(chinese) == 1
        assert 120 < chinese[0]["top"] < 140
        assert pdf.pages[0].extract_text().count("Shared note") == 1


def test_cross_page_preview_matches_export_without_spurious_continuation():
    def draw(canvas):
        canvas.bookmarkPage("trade")
        canvas.addOutlineEntry("Trade", "trade")
        canvas.drawString(35, 530, "The communities of the Persian Gulf and inter-")
        canvas.showPage()
        canvas.setFont("Helvetica", 12)
        canvas.drawString(35, 530, "national trade throughout its long coastal history.")

    source, metadata, segments = _book_fixture(draw)
    assert len(segments) == 1
    assert len(metadata["layout"][0]["regions"]) == 2
    target = "海湾沿岸居民通过海上贸易保持联系，形成了跨越国界的共同历史与文化传统。"
    segments[0] = replace(segments[0], machine_translation=target)
    full, report = compose_pdf(source, metadata, segments)
    assert report["overflow_count"] == 0
    full_reader = PdfReader(io.BytesIO(full))
    for number in (1, 2):
        preview, notes = compose_pdf(source, metadata, segments, only_page=number)
        assert notes["overflow_count"] == 0
        assert (
            PdfReader(io.BytesIO(preview)).pages[0].extract_text()
            == full_reader.pages[number - 1].extract_text()
        )
        assert render_pdf_page(preview, 1, 400) == render_pdf_page(full, number, 400)
    assert target == "".join(p.extract_text() for p in full_reader.pages).replace("\n", "")
