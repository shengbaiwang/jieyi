"""Source text color and per-character ownership survive safe PDF reflow."""

import io
from dataclasses import replace

import pdfplumber
import pytest
from PIL import Image, ImageChops
from reportlab.pdfgen.canvas import Canvas

from jieyi.domain.models import Segment, SegmentKind
from jieyi.ingestion.pdf import GEOMETRY_VERSION, extract_pdf, render_pdf_page
from jieyi.ingestion.pdf_export import (
    _fit,
    _font,
    _inline_text,
    _join_lines,
    _slice_text,
    _split_lines,
    _StyledText,
    compose_pdf,
)
from jieyi.ingestion.plaintext import segments_from_blocks
from jieyi.ingestion.typography import FontStyle


@pytest.mark.parametrize("text_color", [(1, 1, 1), (1, 1, 0.75)], ids=["white", "bright"])
def test_dark_banner_keeps_visible_source_text_color_and_native_graphics(text_color):
    stream = io.BytesIO()
    canvas = Canvas(stream, pagesize=(400, 600))
    canvas.setFillColorRGB(0.08, 0.12, 0.22)
    canvas.rect(30, 540, 340, 26, fill=1, stroke=0)
    canvas.setFillColorRGB(*text_color)
    canvas.setFont("Times-Roman", 12)
    canvas.drawString(40, 549, "Bright native text on a dark banner.")
    canvas.setFillColorRGB(0.25, 0.5, 0.75)
    canvas.rect(30, 100, 180, 150, fill=1, stroke=0)
    canvas.save()
    source = stream.getvalue()
    book = extract_pdf(source)
    segments = [
        replace(segment, machine_translation="横条内的译文清晰可见。")
        for segment in segments_from_blocks("dark-banner", list(book.blocks))
    ]
    result, report = compose_pdf(
        source, {"layout": book.layout, "geometry_version": GEOMETRY_VERSION}, segments
    )
    assert report["reflowed_pages"] == []
    assert report["replaced_blocks"] == 1
    with pdfplumber.open(io.BytesIO(source)) as pdf:
        original_rects = pdf.pages[0].rects
    with pdfplumber.open(io.BytesIO(result)) as pdf:
        page = pdf.pages[0]
        translated = [char for char in page.chars if "\u4e00" <= char["text"] <= "\u9fff"]
        assert translated
        for char in translated:
            assert char["size"] == pytest.approx(12)
            assert char["non_stroking_color"] == pytest.approx(text_color, abs=1 / 255)
        assert "Bright native" not in page.extract_text()
        assert len(page.rects) == len(original_rects) == 2
        for original, actual in zip(original_rects, page.rects, strict=True):
            for key in ("x0", "x1", "top", "bottom", "fill", "stroke", "non_stroking_color"):
                assert actual[key] == pytest.approx(original[key])
    before = Image.open(io.BytesIO(render_pdf_page(source, 1, 400))).convert("RGB")
    after = Image.open(io.BytesIO(render_pdf_page(result, 1, 400))).convert("RGB")
    assert (
        ImageChops.difference(
            before.crop((0, 200, 400, 600)), after.crop((0, 200, 400, 600))
        ).getbbox()
        is None
    )
    assert (
        ImageChops.difference(before.crop((31, 35, 38, 59)), after.crop((31, 35, 38, 59))).getbbox()
        is None
    )
    pixels = after.crop((40, 39, 340, 57)).tobytes()
    assert sum(sum(pixels[index : index + 3]) > 450 for index in range(0, len(pixels), 3)) > 20


def test_style_slice_join_inline_and_rewrap_preserve_color_and_font_ownership():
    first = "甲乙丙丁戊己庚辛壬癸" * 4
    second = "子丑寅卯辰巳午未申酉" * 3
    serif, bold = _font(), _font(FontStyle(bold=True))
    red, blue = (0.8, 0.1, 0.1), (0.1, 0.2, 0.8)
    runs = [
        _StyledText(first, [serif] * len(first), [red] * len(first)),
        _StyledText(second, [bold] * len(second), [blue] * len(second)),
    ]
    text = _join_lines(runs)
    assert isinstance(text, _StyledText)
    assert text == first + "\n" + second
    assert text.colors == tuple([red] * len(first) + [None] + [blue] * len(second))
    assert _slice_text(text, len(first) + 1).colors == (blue,) * len(second)
    split = _split_lines(text)
    assert [line.colors for line in split] == [run.colors for run in runs]
    assert [line.fonts for line in split] == [run.fonts for run in runs]
    regions = [
        {"page": 1, "font_size": 12, "font_name": "Times-Roman", "rect": [30, 548, 78, 560]},
        {"page": 1, "font_size": 12, "font_name": "Times-Roman", "rect": [30, 300, 330, 530]},
    ]
    fitted = _fit(text, regions, serif, shrink=False)
    assert fitted is not None
    lines = [line for _, lines, _ in fitted for line in lines]
    assert all(isinstance(line, _StyledText) for line in lines)
    actual = [
        (char, font, color)
        for line in lines
        for char, font, color in zip(line, line.fonts, line.colors, strict=True)
    ]
    expected = [
        (char, font, color)
        for run in runs
        for char, font, color in zip(run, run.fonts, run.colors, strict=True)
    ]
    assert actual == expected
    assert max(map(len, lines[1:])) > len(lines[0]) * 3
    assert all(size == 12 for _, _, size in fitted)
    ascii_runs = [
        _StyledText("Red", [serif] * 3, [red] * 3),
        _StyledText("Blue", [bold] * 4, [blue] * 4),
    ]
    inline = _inline_text(ascii_runs, [True])
    assert inline == "Red Blue"
    assert inline.fonts == (serif,) * 3 + (None,) + (bold,) * 4
    assert inline.colors == (red,) * 3 + (None,) + (blue,) * 4


def test_actual_paragraph_fallback_preserves_each_sources_color_and_font():
    stream = io.BytesIO()
    canvas = Canvas(stream, pagesize=(400, 600))
    sources = [
        ("First.", "Times-Bold", (0.8, 0.1, 0.1), 530),
        (
            "Second colored paragraph fills the shared column width.",
            "Helvetica",
            (0.1, 0.2, 0.8),
            512,
        ),
    ]
    for text, font, color, y in sources:
        canvas.setFont(font, 12)
        canvas.setFillColorRGB(*color)
        canvas.drawString(30, y, text)
    # Block upward expansion so this fixture still exercises paragraph fallback.
    canvas.rect(28, 540, 330, .4, fill=1, stroke=0)
    canvas.save()
    source = stream.getvalue()
    with pdfplumber.open(io.BytesIO(source)) as pdf:
        chars = pdf.pages[0].chars
        owned = [[char for char in chars if char["fontname"] == font] for _, font, _, _ in sources]
        regions = []
        for row, (_, font, _, _) in zip(owned, sources, strict=True):
            rect = [
                min(c["x0"] for c in row),
                min(c["y0"] for c in row),
                max(c["x1"] for c in row),
                max(c["y1"] for c in row),
            ]
            regions.append(
                {"page": 1, "font_size": 12, "font_name": font, "rect": rect, "lines": [rect]}
            )
    targets = ["甲" * 45, "乙" * 5]
    segments = [
        Segment(
            id=f"color-{index}",
            document_id="colored-fallback",
            stable_key=str(index),
            ordinal=index,
            kind=SegmentKind.PARAGRAPH,
            source_text=text,
            machine_translation=target,
            source_refs=("pdf:page:1",),
        )
        for index, ((text, _, _, _), target) in enumerate(zip(sources, targets, strict=True))
    ]
    metadata = {
        "geometry_version": GEOMETRY_VERSION,
        "layout": [
            {"source_text": text, "regions": [region]}
            for (text, _, _, _), region in zip(sources, regions, strict=True)
        ],
    }
    result, report = compose_pdf(source, metadata, segments)
    assert report["reflowed_pages"] == [1]
    assert report["replaced_blocks"] == 2
    with pdfplumber.open(io.BytesIO(result)) as pdf:
        chars = pdf.pages[0].chars
        first = [char for char in chars if char["text"] == "甲"]
        second = [char for char in chars if char["text"] == "乙"]
        assert len(first) == 45
        assert len(second) == 5
        for row, (_, _, expected, _) in zip([first, second], sources, strict=True):
            assert all(
                char["non_stroking_color"] == pytest.approx(expected, abs=1 / 255) for char in row
            )
            assert all(char["size"] >= 8.4 - 0.01 for char in row)
        assert {char["fontname"] for char in first}.isdisjoint(
            {char["fontname"] for char in second}
        )
        assert "First." not in pdf.pages[0].extract_text()
        assert "Second colored" not in pdf.pages[0].extract_text()


@pytest.mark.parametrize(
    "background,text_color",
    [((0.08, 0.12, 0.22), (1, 1, 1)), ((244 / 255, 233 / 255, 233 / 255), (0, 0, 0))],
    ids=["dark", "light"],
)
@pytest.mark.parametrize("foreground", [False, True], ids=["nested-shading", "foreground-mask"])
def test_nested_same_color_shading_preserves_small_header_and_empty_padding(
    background, text_color, foreground
):
    stream = io.BytesIO()
    canvas = Canvas(stream, pagesize=(400, 600))
    canvas.setFillColorRGB(*background)
    # A verified multiline parent, a tall child around a tiny heading, a row fill,
    # and a padding strip containing no source glyphs are distinct PDF objects.
    for rect in [(30, 340, 340, 130), (30, 435, 340, 30), (30, 395, 340, 24), (30, 340, 340, 15)]:
        canvas.rect(*rect, fill=1, stroke=0)
    canvas.setFillColorRGB(*text_color)
    canvas.setFont("Times-Roman", 6)
    canvas.drawString(40, 450, "A small heading in a generously padded row")
    canvas.setFont("Times-Roman", 12)
    for y, line in zip(
        (420, 402, 384, 366),
        [
            "An expression acquires meaning through common usage.",
            "Its wording belongs to the surrounding language and culture.",
            "The source presents its explanation as a multiline callout.",
            "Each translated line should keep the intended text color.",
        ],
        strict=True,
    ):
        canvas.drawString(40, y, line)
    if foreground:
        canvas.setFillColorRGB(*background)
        canvas.rect(30, 395, 340, 24, fill=1, stroke=0)
    canvas.save()
    source = stream.getvalue()
    book = extract_pdf(source)
    segments = [
        replace(segment, machine_translation="文字色保持清晰。")
        for segment in segments_from_blocks("nested-shading", list(book.blocks))
    ]
    metadata = {"layout": book.layout, "geometry_version": GEOMETRY_VERSION}
    if foreground:
        with pytest.raises(ValueError, match="PDF|图形|遮挡|排版"):
            compose_pdf(source, metadata, segments)
        return
    result, report = compose_pdf(source, metadata, segments)
    assert report["reflowed_pages"] == []
    assert report["replaced_blocks"] == len(segments)
    with pdfplumber.open(io.BytesIO(result)) as pdf:
        page = pdf.pages[0]
        assert len(page.rects) == 4
        assert all(
            rect["non_stroking_color"] == pytest.approx(background, abs=1 / 255)
            for rect in page.rects
        )
        translated = [char for char in page.chars if "\u4e00" <= char["text"] <= "\u9fff"]
        assert translated
        assert {round(char["size"], 3) for char in translated} == {6, 12}
        assert all(
            char["non_stroking_color"] == pytest.approx(text_color, abs=1 / 255)
            for char in translated
        )
    before = Image.open(io.BytesIO(render_pdf_page(source, 1, 400))).convert("RGB")
    after = Image.open(io.BytesIO(render_pdf_page(result, 1, 400))).convert("RGB")
    assert (
        ImageChops.difference(
            before.crop((31, 130, 38, 260)), after.crop((31, 130, 38, 260))
        ).getbbox()
        is None
    )


def test_long_dark_banner_title_cannot_expand_into_wider_body_column():
    stream = io.BytesIO()
    canvas = Canvas(stream, pagesize=(400, 600))
    canvas.setFillColorRGB(0.08, 0.12, 0.22)
    canvas.rect(30, 540, 200, 26, fill=1, stroke=0)
    canvas.setFillColorRGB(1, 1, 1)
    canvas.setFont("Times-Bold", 12)
    canvas.drawString(40, 549, "A compact title.")
    canvas.setFillColorRGB(0, 0, 0)
    canvas.setFont("Times-Roman", 12)
    canvas.drawString(40, 500, "A wider paragraph remains in the body of the page.")
    canvas.save()
    source = stream.getvalue()
    book = extract_pdf(source)
    segments = segments_from_blocks("bounded-banner", list(book.blocks))
    assert len(segments) == 2
    segments = [
        replace(
            segment,
            kind=SegmentKind.HEADING if index == 0 else SegmentKind.PARAGRAPH,
            machine_translation="甲" * 55 if index == 0 else "乙" * 8,
        )
        for index, segment in enumerate(segments)
    ]
    try:
        result, report = compose_pdf(
            source, {"layout": book.layout, "geometry_version": GEOMETRY_VERSION}, segments
        )
    except ValueError as error:
        assert any(word in str(error) for word in ("排版", "容器边界", "可读字号"))
        return
    assert report["replaced_blocks"] == 2
    with pdfplumber.open(io.BytesIO(result)) as pdf:
        heading = [char for char in pdf.pages[0].chars if char["text"] == "甲"]
        body = [char for char in pdf.pages[0].chars if char["text"] == "乙"]
        assert len(heading) == 55
        assert len(body) == 8
        for char in heading:
            assert char["x0"] >= 30 - 0.01
            assert char["x1"] <= 230 + 0.01
            assert char["y0"] >= 540 - 0.01
            assert char["y1"] <= 566 + 0.01
            assert char["non_stroking_color"] == pytest.approx((1, 1, 1))
        assert all(char["y1"] < 540 for char in body)
