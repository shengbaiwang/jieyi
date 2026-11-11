"""Layout safety across region widths, native backgrounds and physical capacity."""

import io
from dataclasses import replace

import pdfplumber
import pytest
from PIL import Image, ImageChops
from pypdf import PdfReader, PdfWriter
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen.canvas import Canvas

from jieyi.ingestion.pdf import GEOMETRY_VERSION, extract_pdf, render_pdf_page
from jieyi.ingestion.pdf_export import _fit, _font, _inline_text, _StyledText, compose_pdf
from jieyi.ingestion.plaintext import segments_from_blocks
from jieyi.ingestion.typography import FontStyle


def _region(rect, *, size=12):
    return {"page": 1, "font_size": size, "font_name": "Times-Roman", "rect": rect, "lines": [rect]}


def test_flow_rewraps_original_text_when_next_region_is_wider():
    text = "甲乙丙丁戊己庚辛壬癸" * 6
    regions = [_region([30, 548, 78, 560]), _region([30, 300, 330, 530])]
    fitted = _fit(text, regions, _font(), shrink=False)
    assert fitted is not None
    assert [size for _, _, size in fitted] == [12, 12]
    first, second = [lines for _, lines, _ in fitted]
    assert len(first) == 1
    assert max(map(len, second)) > len(first[0]) * 3
    assert "".join(str(line) for _, lines, _ in fitted for line in lines) == text


def test_flow_preserves_hard_breaks_and_font_ownership_across_regions():
    first = "甲乙丙丁戊己庚辛壬癸" * 3
    second = "子丑寅卯辰巳午未申酉" * 2
    text = first + "\n" + second
    serif, bold = _font(), _font(FontStyle(bold=True))
    fonts = [serif] * len(first) + [None] + [bold] * len(second)
    styled = _StyledText(text, fonts)
    fitted = _fit(
        styled, [_region([30, 548, 78, 560]), _region([30, 250, 330, 530])], serif, shrink=False
    )
    assert fitted is not None
    lines = [line for _, lines, _ in fitted for line in lines if line]
    assert all(isinstance(line, _StyledText) for line in lines)
    assert "".join(lines) == first + second
    assert all(not ("癸" in line and "子" in line) for line in lines)
    assert tuple(face for line in lines for face in line.fonts) == tuple(
        fonts[: len(first)] + fonts[len(first) + 1 :]
    )
    # A wider following region still uses its available width, even with styles.
    assert max(map(len, lines[1:])) > len(lines[0]) * 3


def _native_background_fixture(
    *, color=(1, 1, 1), stroke=False, after=False, underlay=None, alpha=1
):
    stream = io.BytesIO()
    canvas = Canvas(stream, pagesize=(400, 600))
    canvas.setFont("Times-Roman", 12)
    if underlay == "image":
        picture = Image.new("RGB", (60, 20), (30, 100, 200))
        canvas.drawImage(ImageReader(picture), 30, 540, 300, 30)
    elif underlay == "path":
        canvas.setFillColorRGB(0.2, 0.4, 0.8)
        canvas.rect(30, 540, 300, 30, fill=1, stroke=0)

    def background():
        canvas.saveState()
        canvas.setFillAlpha(alpha)
        canvas.setFillColorRGB(*color)
        canvas.setStrokeColorRGB(0.1, 0.2, 0.3)
        canvas.rect(30, 548, 300, 15, fill=1, stroke=int(stroke))
        canvas.restoreState()
        canvas.setFillColorRGB(0, 0, 0)

    if not after:
        background()
    canvas.drawString(30, 550, "A paragraph with a neutral native background.")
    if after:
        background()
    # A colored illustration away from the text must remain unchanged.
    canvas.setFillColorRGB(0.2, 0.4, 0.8)
    canvas.rect(30, 100, 180, 200, fill=1, stroke=0)
    canvas.save()
    source = stream.getvalue()
    book = extract_pdf(source)
    segments = segments_from_blocks("background", list(book.blocks))
    translated = [
        replace(segment, machine_translation="背景上的译文内容。") for segment in segments
    ]
    return source, {"layout": book.layout, "geometry_version": GEOMETRY_VERSION}, translated


@pytest.mark.parametrize("color", [(1, 1, 1), (0.96, 0.96, 0.96)])
def test_neutral_background_is_preserved_without_unnecessary_reflow(color):
    source, metadata, segments = _native_background_fixture(color=color)
    result, report = compose_pdf(source, metadata, segments)
    assert report["reflowed_pages"] == []
    assert report["replaced_blocks"] == 1
    with pdfplumber.open(io.BytesIO(result)) as pdf:
        translated = [char for char in pdf.pages[0].chars if "\u4e00" <= char["text"] <= "\u9fff"]
        assert translated
        assert all(char["size"] == pytest.approx(12) for char in translated)
        assert "A paragraph" not in pdf.pages[0].extract_text()
        result_rects = pdf.pages[0].rects
    with pdfplumber.open(io.BytesIO(source)) as pdf:
        for original, actual in zip(pdf.pages[0].rects, result_rects, strict=True):
            for key in ("x0", "x1", "top", "bottom", "fill", "stroke", "non_stroking_color"):
                assert actual[key] == pytest.approx(original[key])
    before = Image.open(io.BytesIO(render_pdf_page(source, 1, 400))).convert("RGB")
    after = Image.open(io.BytesIO(render_pdf_page(result, 1, 400))).convert("RGB")
    assert (
        ImageChops.difference(
            before.crop((0, 250, 400, 600)), after.crop((0, 250, 400, 600))
        ).getbbox()
        is None
    )


@pytest.mark.parametrize(
    "options",
    [
        {"after": True},
        {"color": (0.2, 0.4, 0.8)},
        {"stroke": True},
    ],
)
def test_overlapping_foreground_and_illustration_are_not_treated_as_background(options):
    source, metadata, segments = _native_background_fixture(**options)
    with pytest.raises(ValueError, match="PDF|排版|图形|障碍|遮挡"):
        compose_pdf(source, metadata, segments)


def test_excess_translation_cannot_be_hidden_by_unreadable_font_size():
    source, metadata, segments = _native_background_fixture()
    segments = [
        replace(segment, machine_translation="译文内容必须完整而且清晰可读。" * 12)
        for segment in segments
    ]
    with pytest.raises(ValueError, match="排版|容量|字号|可读"):
        compose_pdf(source, metadata, segments)


def test_source_footnotes_are_not_enlarged_by_body_text_readability_floor():
    text = "小字脚注内容。"
    fitted = _fit(text, [_region([30, 100, 200, 107], size=7)], _font())
    assert fitted is not None
    assert fitted[0][2] == 7


def test_flow_rewraps_original_text_when_next_region_is_narrower():
    text = "甲乙丙丁戊己庚辛壬癸" * 4
    fitted = _fit(
        text,
        [_region([30, 548, 270, 560]), _region([30, 200, 78, 530])],
        _font(),
        shrink=False,
    )
    assert fitted is not None
    first, second = [lines for _, lines, _ in fitted]
    assert len(first) == 1
    assert len(first[0]) > max(map(len, second)) * 3
    assert all(size == 12 for _, _, size in fitted)
    assert "".join(line for _, lines, _ in fitted for line in lines) == text


def test_explicit_blank_line_survives_a_region_boundary():
    fitted = _fit(
        "甲乙丙丁\n\n子丑寅卯",
        [_region([30, 548, 330, 560]), _region([30, 300, 330, 530])],
        _font(),
        shrink=False,
    )
    assert fitted is not None
    assert fitted[0][1] == ["甲乙丙丁"]
    assert fitted[1][1] == ["", "子丑寅卯"]


@pytest.mark.parametrize(
    "options",
    [
        {"underlay": "image"},
        {"underlay": "path"},
        {"alpha": 0.5},
    ],
)
def test_neutral_plate_over_artwork_or_with_transparency_remains_an_obstacle(options):
    source, metadata, segments = _native_background_fixture(**options)
    with pytest.raises(ValueError, match="PDF|排版|图形|障碍|遮挡"):
        compose_pdf(source, metadata, segments)


def test_overlong_column_cannot_borrow_another_column_or_heading():
    from jieyi.domain.models import Segment, SegmentKind

    stream = io.BytesIO()
    canvas = Canvas(stream, pagesize=(420, 600))
    canvas.setFont("Times-Roman", 12)
    canvas.drawString(30, 570, "Heading")
    for x, lines in [
        (30, ["Left passage first line.", "Left passage next line.", "Left passage last line."]),
        (
            240,
            ["Right passage first line.", "Right passage next line.", "Right passage last line."],
        ),
    ]:
        for y, line in zip([530, 510, 490], lines, strict=True):
            canvas.drawString(x, y, line)
    canvas.save()
    source = stream.getvalue()
    # Supply explicit column ownership: PDF text extraction may join equal-height
    # columns into a single physical line, which is a separate ingestion concern.
    with pdfplumber.open(io.BytesIO(source)) as pdf:
        chars = pdf.pages[0].chars
        heading = [c for c in chars if c["y0"] > 550]
        left = [c for c in chars if c["y0"] < 550 and c["x0"] < 200]
        right = [c for c in chars if c["y0"] < 550 and c["x0"] >= 200]
        regions = []
        for owned in (heading, left, right):
            rect = [
                min(c["x0"] for c in owned),
                min(c["y0"] for c in owned),
                max(c["x1"] for c in owned),
                max(c["y1"] for c in owned),
            ]
            lines = []
            for y in sorted({c["y0"] for c in owned}, reverse=True):
                row = [c for c in owned if c["y0"] == y]
                lines.append(
                    [
                        min(c["x0"] for c in row),
                        y,
                        max(c["x1"] for c in row),
                        max(c["y1"] for c in row),
                    ]
                )
            regions.append({**_region(rect), "lines": lines})
    texts = [
        "Heading",
        "Left passage first line. Left passage next line. Left passage last line.",
        "Right passage first line. Right passage next line. Right passage last line.",
    ]
    targets = ["标题", "左" * 130, "右栏正文"]
    segments = [
        Segment(
            id=f"segment-{index}",
            document_id="columns",
            stable_key=str(index),
            ordinal=index,
            kind=SegmentKind.HEADING if index == 0 else SegmentKind.PARAGRAPH,
            source_text=text,
            machine_translation=target,
            source_refs=("pdf:page:1",),
        )
        for index, (text, target) in enumerate(zip(texts, targets, strict=True))
    ]
    metadata = {
        "geometry_version": GEOMETRY_VERSION,
        "layout": [
            {"source_text": text, "regions": [region]}
            for text, region in zip(texts, regions, strict=True)
        ],
    }
    try:
        result, _ = compose_pdf(source, metadata, segments)
    except ValueError:
        return
    with pdfplumber.open(io.BytesIO(result)) as pdf:
        chars = pdf.pages[0].chars
        translated = [c for c in chars if c["text"] in "标题左右栏正文"]
        assert translated
        assert all(c["size"] >= 8.4 - 0.01 for c in translated)
        assert all(c["x1"] < 200 and 487 <= c["y0"] <= 540 for c in chars if c["text"] == "左")
        assert sum(c["text"] == "左" for c in chars) == 130
        assert all(
            c["x0"] >= 240 and 487 <= c["y0"] <= 540 for c in chars if c["text"] in "右栏正文"
        )
        assert all(c["x0"] < 200 and c["y0"] > 550 for c in chars if c["text"] in "标题")


@pytest.mark.parametrize("underline", [False, True])
def test_toc_elastic_leader_preserves_label_and_folio_at_original_size(underline):
    stream = io.BytesIO()
    canvas = Canvas(stream, pagesize=(400, 600))
    canvas.setFont("Times-Roman", 12)
    canvas.drawString(30, 550, "Chapter One " + "." * 55 + " 17")
    if underline:
        canvas.rect(
            30, 549, pdfmetrics.stringWidth("Chapter One", "Times-Roman", 12), 0.5, fill=1, stroke=0
        )
    canvas.save()
    source = stream.getvalue()
    book = extract_pdf(source)
    segments = segments_from_blocks("contents", list(book.blocks))
    assert len(segments) == 1
    label = "第一章：章节标题"
    segments[0] = replace(segments[0], machine_translation=label + "…" * 150 + " 17")
    result, report = compose_pdf(
        source, {"layout": book.layout, "geometry_version": GEOMETRY_VERSION}, segments
    )
    assert report["reflowed_pages"] == []
    assert report["replaced_blocks"] == 1
    with pdfplumber.open(io.BytesIO(result)) as pdf:
        page = pdf.pages[0]
        text = page.extract_text().strip()
        assert len(page.rects) == int(underline)
        if underline:
            assert page.rects[0]["height"] == pytest.approx(0.5)
            assert page.rects[0]["y0"] == pytest.approx(549)
        assert text.startswith(label)
        assert text.endswith("17")
        assert "..." in text
        assert "…" not in text
        assert "Chapter One" not in text
        assert all(char["size"] == pytest.approx(12) for char in page.chars)
        assert (
            max(char["x1"] for char in page.chars) <= book.layout[0]["regions"][0]["rect"][2] + 0.01
        )


def test_rotated_fragment_line_recovers_native_size_and_source_word_boundaries():
    stream = io.BytesIO()
    canvas = Canvas(stream, pagesize=(400, 600))
    canvas.setFont("Helvetica", 12)
    canvas.drawString(35, 530, "A map caption describing the Persian Gulf cultural world.")
    canvas.save()
    source_reader = PdfReader(io.BytesIO(stream.getvalue()))
    source_reader.pages[0].rotate(90)
    writer = PdfWriter()
    writer.add_page(source_reader.pages[0])
    stream = io.BytesIO()
    writer.write(stream)
    source = stream.getvalue()
    book = extract_pdf(source)
    segments = segments_from_blocks("rotated-fragments", list(book.blocks))
    assert [segment.source_text for segment in segments] == [
        "A",
        "map caption describing the Persian",
        "G",
        "ulf",
        "cultural world.",
    ]
    targets = ["Map:", "Persian", "G", "ulf", "region."]
    translated = [
        replace(segment, machine_translation=text)
        for segment, text in zip(segments, targets, strict=True)
    ]
    result, report = compose_pdf(
        source, {"layout": book.layout, "geometry_version": GEOMETRY_VERSION}, translated
    )
    reader = PdfReader(io.BytesIO(result))
    assert len(reader.pages) == 1
    assert reader.pages[0].rotation == 90
    assert report["replaced_blocks"] == len(segments)
    assert reader.pages[0].extract_text().strip() == "Map: Persian Gulf region."
    # Unrotate for font-size inspection: pdfminer reports glyph widths as size
    # for 90-degree glyphs, which is the exact source of the former tiny-font bug.
    reader.pages[0].rotation = 0
    writer = PdfWriter()
    writer.add_page(reader.pages[0])
    stream = io.BytesIO()
    writer.write(stream)
    with pdfplumber.open(io.BytesIO(stream.getvalue())) as pdf:
        assert all(char["size"] == pytest.approx(12) for char in pdf.pages[0].chars)


def test_adjacent_row_backplates_tolerate_subpixel_ink_edge_overlap():
    import pypdfium2 as pdfium

    text = "A paragraph with gyp descenders."
    stream = io.BytesIO()
    canvas = Canvas(stream, pagesize=(400, 600))
    canvas.setFont("Times-Roman", 12)
    canvas.drawString(30, 540, text)
    canvas.save()
    with pdfium.PdfDocument(stream.getvalue()) as pdf:
        page = pdf[0]
        textpage = page.get_textpage()
        try:
            descent = min(textpage.get_charbox(i)[1] for i in range(textpage.count_chars())) - 540
        finally:
            textpage.close()
            page.close()
    stream = io.BytesIO()
    canvas = Canvas(stream, pagesize=(400, 600))
    canvas.setFont("Times-Roman", 12)
    for y in [540, 527, 514, 501]:
        canvas.setFillColorRGB(1, 1, 1)
        # The next plate reaches 0.01 pt into the preceding row's visible ink,
        # as happens after native-coordinate rounding in real PDF producers.
        canvas.rect(30, y - 3, 300, 16 + descent + 0.01, fill=1, stroke=0)
        canvas.setFillColorRGB(0, 0, 0)
        canvas.drawString(30, y, text)
    canvas.save()
    source = stream.getvalue()
    book = extract_pdf(source)
    segments = [
        replace(segment, machine_translation="正常可读的译文。")
        for segment in segments_from_blocks("row-plates", list(book.blocks))
    ]
    result, report = compose_pdf(
        source, {"layout": book.layout, "geometry_version": GEOMETRY_VERSION}, segments
    )
    assert report["reflowed_pages"] == []
    assert report["replaced_blocks"] == len(segments)
    with pdfplumber.open(io.BytesIO(result)) as pdf:
        assert len(pdf.pages[0].rects) == 4
        assert all(char["size"] == pytest.approx(12) for char in pdf.pages[0].chars)


def test_dark_horizontal_bar_through_source_text_remains_an_obstacle():
    stream = io.BytesIO()
    canvas = Canvas(stream, pagesize=(400, 600))
    canvas.setFont("Times-Roman", 12)
    text = "A source passage with a horizontal bar."
    canvas.drawString(30, 550, text)
    canvas.rect(30, 554, pdfmetrics.stringWidth(text, "Times-Roman", 12), 0.5, fill=1, stroke=0)
    canvas.save()
    source = stream.getvalue()
    book = extract_pdf(source)
    segments = [
        replace(segment, machine_translation="横线遮挡的正文。")
        for segment in segments_from_blocks("occluded-bar", list(book.blocks))
    ]
    with pytest.raises(ValueError, match="PDF|排版|图形|障碍|遮挡"):
        compose_pdf(source, {"layout": book.layout, "geometry_version": GEOMETRY_VERSION}, segments)


def _masked_toc_folio_fixture(*, separate_objects=True):
    stream = io.BytesIO()
    canvas = Canvas(stream, pagesize=(400, 600))
    canvas.setFont("Times-Roman", 12)
    prefix = "Chapter One " + "." * 55 + " "
    folio_x = 30 + pdfmetrics.stringWidth(prefix, "Times-Roman", 12)
    if separate_objects:
        canvas.drawString(30, 550, prefix)
        canvas.drawString(folio_x, 550, "4")
    else:
        canvas.drawString(30, 550, prefix + "4")
    canvas.setFillColorRGB(1, 1, 1)
    canvas.rect(folio_x - 1, 547, 8, 14, fill=1, stroke=0)
    canvas.setFillColorRGB(0, 0, 0)
    canvas.drawString(folio_x, 550, "6")
    canvas.save()
    source = stream.getvalue()
    book = extract_pdf(source)
    segments = segments_from_blocks("masked-contents", list(book.blocks))
    assert len(segments) == 1
    # Extraction retains both the concealed old number and the later visible
    # replacement. Preserve that exact source suffix; its visual truth lives in
    # the native number objects and white mask rather than a guessed text digit.
    target = segments[0].source_text.replace("Chapter One", "第一章：章节标题")
    translated = [replace(segments[0], machine_translation=target)]
    return (
        source,
        {"layout": book.layout, "geometry_version": GEOMETRY_VERSION},
        translated,
        folio_x,
    )


@pytest.mark.parametrize("separate_objects", [True, False])
def test_masked_toc_folio_retains_native_number_only_with_separable_objects(separate_objects):
    source, metadata, segments, folio_x = _masked_toc_folio_fixture(
        separate_objects=separate_objects
    )
    if not separate_objects:
        with pytest.raises(ValueError, match="归属不完整|无法安全排版"):
            compose_pdf(source, metadata, segments)
        return
    result, report = compose_pdf(source, metadata, segments)
    assert report["reflowed_pages"] == []
    assert report["replaced_blocks"] == 1
    with (
        pdfplumber.open(io.BytesIO(source)) as original,
        pdfplumber.open(io.BytesIO(result)) as after,
    ):
        rects_before, rects_after = original.pages[0].rects, after.pages[0].rects
        assert len(rects_before) == len(rects_after) == 1
        for key in ("x0", "x1", "y0", "y1", "non_stroking_color"):
            assert rects_after[0][key] == pytest.approx(rects_before[0][key])
        text = after.pages[0].extract_text()
        assert "第一章：章节标题" in text
        assert "Chapter One" not in text
        assert all(char["size"] == pytest.approx(12) for char in after.pages[0].chars)
    before = Image.open(io.BytesIO(render_pdf_page(source, 1, 400))).convert("RGB")
    after = Image.open(io.BytesIO(render_pdf_page(result, 1, 400))).convert("RGB")
    numeric_strip = (int(folio_x - 1), 38, int(folio_x + 10), 55)
    assert (
        ImageChops.difference(before.crop(numeric_strip), after.crop(numeric_strip)).getbbox()
        is None
    )


def test_masked_toc_folio_rejects_changed_target_page_number():
    import re

    source, metadata, segments, _ = _masked_toc_folio_fixture()
    target = re.sub(r"\d+$", "99", segments[0].machine_translation)
    segments = [replace(segments[0], machine_translation=target)]
    with pytest.raises(ValueError, match="目录|页码"):
        compose_pdf(source, metadata, segments)


@pytest.mark.parametrize(
    ("first", "second", "word_break", "expected"),
    [
        ("café", "chaud", True, "café chaud"),
        ("καλή", "μέρα", True, "καλή μέρα"),
        ("добрый", "день", True, "добрый день"),
        ("좋은", "하루", True, "좋은 하루"),
        ("中文", "正文", True, "中文正文"),
        ("日本", "かな", True, "日本かな"),
        ("G", "ulf", False, "Gulf"),
    ],
)
def test_shared_native_line_preserves_multilingual_word_boundaries(
    first, second, word_break, expected
):
    runs = [_StyledText(first, ["first-face"] * len(first)),
            _StyledText(second, ["second-face"] * len(second))]
    result = _inline_text(runs, [word_break])
    assert result == expected
    assert result.fonts[:len(first)] == ("first-face",) * len(first)
    assert result.fonts[-len(second):] == ("second-face",) * len(second)
    assert len(result.fonts) == len(result)


def test_masked_toc_folio_cannot_lose_native_number_ownership_during_reflow():
    source, metadata, segments, _ = _masked_toc_folio_fixture()
    # The ordinary paragraph below provides enough same-column capacity for a
    # generic fallback to succeed. A clipped TOC must not borrow that capacity
    # and redraw its already retained native folio in the paragraph below.
    line = "Another ordinary full-width source paragraph."
    stream = io.BytesIO()
    canvas = Canvas(stream, pagesize=(400, 600))
    canvas.setFont("Times-Roman", 12)
    for y in (520, 505, 490):
        canvas.drawString(30, y, line)
    canvas.save()
    writer = PdfWriter(clone_from=io.BytesIO(source))
    writer.pages[0].merge_page(PdfReader(io.BytesIO(stream.getvalue())).pages[0])
    stream = io.BytesIO()
    writer.write(stream)
    source = stream.getvalue()
    with pdfplumber.open(io.BytesIO(source)) as pdf:
        chars = [char for char in pdf.pages[0].chars if 480 < char["y0"] < 535]
        rect = [min(char["x0"] for char in chars), min(char["y0"] for char in chars),
                max(char["x1"] for char in chars), max(char["y1"] for char in chars)]
        lines = []
        for y in sorted({char["y0"] for char in chars}, reverse=True):
            row = [char for char in chars if char["y0"] == y]
            lines.append([min(char["x0"] for char in row), y,
                          max(char["x1"] for char in row), max(char["y1"] for char in row)])
    body_source = " ".join([line] * 3)
    body_region = {**_region(rect), "lines": lines}
    metadata = {**metadata, "layout": [*metadata["layout"],
                {"source_text": body_source, "regions": [body_region]}]}
    long_target = "第一章" + "长" * 25 + " " + "." * 55 + "46"
    translated = [
        replace(segments[0], machine_translation=long_target),
        replace(segments[0], id="body-below-masked-toc", stable_key="body-below-masked-toc",
                ordinal=segments[0].ordinal + 1, source_text=body_source,
                machine_translation="第二行"),
    ]
    with pytest.raises(ValueError, match="目录|页码|无法安全排版"):
        compose_pdf(source, metadata, translated)
