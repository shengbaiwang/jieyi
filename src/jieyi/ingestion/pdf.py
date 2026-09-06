"""PDF text ingestion with page provenance; document content is never executable."""

from __future__ import annotations

import io
import re
import threading
import unicodedata
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from html import unescape
from itertools import pairwise
from statistics import median

import pdfplumber
import pypdfium2 as pdfium
from pypdf import PdfReader

from jieyi.domain.models import SegmentKind
from jieyi.ingestion.plaintext import ParsedBlock

VERSION = "pdf-layout-v2"
# PDFium is not thread-safe, including separate documents.
_RENDER_LOCK = threading.Lock()


@dataclass(frozen=True)
class PdfBook:
    title: str
    blocks: tuple[ParsedBlock, ...]
    pages: tuple[dict, ...]
    navigation: tuple[dict, ...]
    warnings: tuple[str, ...]
    layout: tuple[dict, ...] = field(default_factory=tuple)


def _line_text(chars: list[dict]) -> str:
    # A ligature may occur twice in pdfplumber's expanded word character list.
    # Reconstruct from glyph positions in linear time instead of quadratic deduplication.
    seen = set()
    output = ""
    previous = None
    for char in sorted(chars, key=lambda item: item["x0"]):
        key = (char["text"], round(char["x0"], 2), round(char["top"], 2))
        if key in seen:
            continue
        seen.add(key)
        if previous and char["x0"] - previous["x1"] > max(1, char["size"] * 0.2):
            output += " "
        output += unicodedata.normalize("NFKC", char["text"])
        previous = char
    return " ".join(output.split())


def _join(left: str, right: str) -> str:
    if re.search(r"[a-z]{2,}-$", left) and re.match(r"^[a-z]", right):
        return left[:-1] + right
    return left + " " + right


def _margin_key(text: str) -> str:
    return re.sub(r"\b(?:\d+|[ivxlcdm]+)\b", "#", text.casefold()).strip()


def _text_line(chars: list[dict]) -> dict:
    return {
        "text": _line_text(chars),
        "size": median(c["size"] for c in chars),
        "font": Counter(c["fontname"] for c in chars).most_common(1)[0][0],
        "x0": min(c["x0"] for c in chars),
        "x1": max(c["x1"] for c in chars),
        "top": min(c["top"] for c in chars),
        "bottom": max(c["bottom"] for c in chars),
        "chars": chars,
    }


def _page_lines(page) -> list[dict]:
    # Ornament fonts often map flourishes to ASCII digits. Remove those glyphs
    # before line detection: their tall bounds can swallow the chapter label.
    def keep_glyph(obj):
        if obj.get("object_type") != "char":
            return True
        if re.search(r"dingbats|wingdings|webdings", obj.get("fontname", ""), re.IGNORECASE):
            return False
        # A landscape insert may retain the portrait page's folio/production
        # marks, now vertical at its outer edges. They are not caption text.
        return not (
            page.rotation in {90, 270}
            and not obj.get("upright", True)
            and (obj["x1"] < page.width * 0.05 or obj["x0"] > page.width * 0.95)
        )

    filtered = page.filter(keep_glyph)
    lines = []
    for raw in filtered.extract_text_lines(x_tolerance=2, y_tolerance=3):
        chars = raw["chars"]
        if not chars:
            continue
        size = median(c["size"] for c in chars)
        # PDF extractors sometimes put a sunken initial on the second line.
        # Isolate oversized leading glyphs before assigning them to a body line.
        initials = [
            c
            for c in chars
            if c["size"] >= size * 1.5
            and re.fullmatch(r"[A-ZÀ-ÖØ-Þ]", c["text"])
            and c["x0"] <= min(g["x0"] for g in chars) + 1
        ]
        if len(initials) == 1 and len(chars) > 1:
            initial = initials[0]
            lines.append(_text_line([initial]))
            chars = [c for c in chars if c is not initial]
        lines.append(_text_line(chars))
    # A raised/drop initial shares vertical space with the first body line,
    # even when its baseline and font size differ. Attach to the nearest line
    # to its right, preserving the actual word space and the full glyph bounds.
    removed = set()
    for index, initial in enumerate(lines):
        if not re.fullmatch(r"[A-ZÀ-ÖØ-Þ]", initial["text"]):
            continue
        candidates = [
            (other_index, line)
            for other_index, line in enumerate(lines)
            if other_index != index
            and len(line["text"]) > 1
            and initial["size"] >= line["size"] * 1.5
            and -1 <= line["x0"] - initial["x1"] <= line["size"] * 0.8
            and initial["top"] <= line["top"] + line["size"] * 0.65
            and min(initial["bottom"], line["bottom"]) - max(initial["top"], line["top"])
            >= line["size"] * 0.45
        ]
        if candidates:
            target, line = min(candidates, key=lambda item: item[1]["top"])
            merged = _text_line(initial["chars"] + line["chars"])
            # Paragraph spacing must use the body baseline, not the initial.
            merged["flow_top"] = line["top"]
            merged["flow_bottom"] = line["bottom"]
            merged["drop_bottom"] = initial["bottom"]
            lines[target] = merged
            removed.add(index)
    result = []
    for index, line in enumerate(lines):
        if index not in removed:
            line.pop("chars")
            result.append(line)
    return result


def _is_heading(line: dict, body_size: float) -> bool:
    font = line["font"].casefold()
    if len(line["text"]) >= 300:
        return False
    if re.search(r"italic|oblique", font) and line["size"] <= body_size * 1.5:
        return False
    return line["size"] > body_size * 1.18 or (
        line["size"] >= body_size * 0.95
        and bool(re.search(r"bold|black|smallcaps|(?:^|[-_])[^-_]*sc$", font))
    )


def _heading_continuation(previous: dict, line: dict, body_size: float) -> bool:
    size = max(previous["size"], line["size"])
    aligned = (
        min(
            abs(line["x0"] - previous["x0"]),
            abs(line["x1"] - previous["x1"]),
            abs((line["x0"] + line["x1"] - previous["x0"] - previous["x1"]) / 2),
        )
        <= body_size
    )
    return (
        previous["font"] == line["font"]
        and abs(previous["size"] - line["size"]) <= size * 0.12
        and -size * 0.2 <= line["top"] - previous["bottom"] <= size * 0.9
        and aligned
    )


def _paragraph_break(
    group: list[dict], line: dict, *, body_size: float, left: float, right: float, normal_gap: float
) -> bool:
    previous = group[-1]
    previous_heading = _is_heading(previous, body_size)
    heading = _is_heading(line, body_size)
    if previous_heading or heading:
        return not (
            previous_heading and heading and _heading_continuation(previous, line, body_size)
        )
    gap = line.get("flow_top", line["top"]) - previous.get("flow_bottom", previous["bottom"])
    if gap > max(body_size * 0.65, normal_gap * 1.6):
        return True
    if abs(line["size"] - previous["size"]) > body_size * 0.15:
        return True
    numbered = r"^\d+(?:[.)]\s|\s+\D)"
    marker = re.match(numbered, line["text"])
    if (
        marker
        and re.match(numbered, group[0]["text"])
        and abs(line["x0"] - group[0]["x0"]) < body_size * 0.5
    ):
        return True
    # Indentation is relative to the body margin, excluding headings, folios,
    # ornaments and printer marks. Lines wrapping around a drop cap stay together.
    indent = line["x0"] - left
    wraps_initial = line["top"] < group[0].get("drop_bottom", -float("inf"))
    # A hanging-indent continuation starts after an incomplete first line.
    hanging = len(group) == 1 and bool(re.match(numbered, previous["text"]))
    if (
        re.match(numbered, group[0]["text"])
        and previous["font"] != line["font"]
        and re.search(r"italic|oblique", line["font"], re.IGNORECASE)
    ):
        return True
    if (
        not wraps_initial
        and not hanging
        and 4 < indent < body_size * 3.2
        and previous["x0"] < line["x0"] - 3
    ):
        return True
    short_end = previous["x1"] < right - max(25, body_size * 3)
    return bool(short_end and re.search(r'[.!?。！？][\d"”’)]*$', previous["text"]))


def extract_pdf(data: bytes, progress: Callable[[int, int], None] | None = None) -> PdfBook:
    if not data.lstrip().startswith(b"%PDF-"):
        raise ValueError("这不是有效的 PDF 文件，请重新选择。")
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise ValueError("PDF 有密码保护，请解锁后重新导入。")
        count = len(reader.pages)
        if not 0 < count <= 2000:
            raise ValueError("PDF 页数须在 1–2000 页之间，请拆分后导入。")
        title = str((reader.metadata or {}).get("/Title") or "Untitled PDF").strip()
        navigation: list[dict] = []

        def outline(items, level=0):
            for item in items:
                if isinstance(item, list):
                    outline(item, level + 1)
                else:
                    try:
                        page = reader.get_destination_page_number(item)
                        if page is not None and 0 <= page < count:
                            navigation.append(
                                {
                                    "title": unescape(str(item.title)),
                                    "page": page + 1,
                                    "level": level,
                                }
                            )
                    except (ValueError, KeyError, TypeError):
                        continue

        try:
            outline(reader.outline)
        except (ValueError, KeyError, TypeError):
            pass
        navigation.sort(key=lambda item: item["page"])
        pages: list[dict] = []
        margins: Counter = Counter()
        sizes: Counter = Counter()
        with pdfplumber.open(io.BytesIO(data), unicode_norm="NFC") as pdf:
            for index, page in enumerate(pdf.pages):
                lines = _page_lines(page)
                for line in lines:
                    sizes[round(line["size"], 1)] += len(line["text"])
                # Coordinates may be negative in PDFs with displaced media boxes.
                top = min((line["top"] for line in lines), default=0)
                bottom = max((line["bottom"] for line in lines), default=page.height)
                keys = {
                    _margin_key(line["text"])
                    for line in lines
                    if line["top"] <= top + 12 or line["bottom"] >= bottom - 12
                }
                margins.update(keys)
                pages.append(
                    {
                        "number": index + 1,
                        "width": page.width,
                        "height": page.height,
                        "lines": lines,
                    }
                )
                page.close()  # Release glyph caches while processing long books.
                if progress:
                    progress(index + 1, count)
        body_size = sizes.most_common(1)[0][0] if sizes else 10
        repeated = {key for key, hits in margins.items() if hits >= max(3, count * 0.015)}
        blocks: list[ParsedBlock] = []
        layout: list[dict] = []
        heading = ""
        page_info = []
        warnings = []
        for page in pages:
            number = page["number"]
            entries = [item for item in navigation if item["page"] == number]
            if entries:
                heading = entries[-1]["title"]
            lines = page["lines"]
            first_top = min((line["top"] for line in lines), default=0)
            last_bottom = max((line["bottom"] for line in lines), default=0)
            printed_label = None
            kept = []
            for line in lines:
                text = line["text"]
                edge = line["top"] <= first_top + 12 or line["bottom"] >= last_bottom - 12
                # Running heads can sit below printer marks and may occur on only
                # one contents page. A folio plus a short top line is structural.
                running_head = (
                    line["top"] <= first_top + 3
                    and line["size"] <= body_size * 1.1
                    and line["bottom"] < page["height"] * 0.15
                    and bool(re.match(r"^(?:\d+|[ivxlcdm]+)\s+\D|^.+\s+\d+$", text))
                    and len(text) < 120
                    and len(lines) > 1
                    and lines[1]["top"] - line["bottom"] > body_size * 0.5
                )
                if edge and (
                    (
                        _margin_key(text) in repeated
                        and (
                            not _is_heading(line, body_size) or line["top"] < page["height"] * 0.08
                        )
                    )
                    or re.fullmatch(r"\d+|[ivxlcdm]+", text)
                    or running_head
                ):
                    match = re.search(r"(?:^|\s)(\d+|[ivxlcdm]+)(?:\s|$)", text)
                    if match:
                        printed_label = match.group(1)
                    continue
                if text:
                    kept.append(line)
            page_info.append(
                {
                    "number": number,
                    "label": printed_label or str(number),
                    "width": page["width"],
                    "height": page["height"],
                    "text_chars": sum(len(line["text"]) for line in kept),
                }
            )
            if not kept:
                continue
            body_lines = [
                line
                for line in kept
                if not _is_heading(line, body_size)
                and abs(line["size"] - body_size) < body_size * 0.15
            ]
            flow_lines = body_lines or kept
            left = Counter(round(line["x0"]) for line in flow_lines).most_common(1)[0][0]
            right = max(line["x1"] for line in flow_lines)
            gaps = [
                b["top"] - a.get("flow_bottom", a["bottom"])
                for a, b in pairwise(flow_lines)
                if 0 <= b["top"] - a.get("flow_bottom", a["bottom"]) <= body_size * 2
            ]
            normal_gap = median(gaps) if len(gaps) >= 3 else body_size * 0.2
            group: list[dict] = []
            page_blocks: list[ParsedBlock] = []
            page_layout: list[dict] = []

            def flush(
                group=group,
                page_blocks=page_blocks,
                heading=heading,
                number=number,
                page_layout=page_layout,
                height=page["height"],
            ):
                if not group:
                    return
                text = group[0]["text"]
                for line in group[1:]:
                    text = _join(text, line["text"])
                size = median(line["size"] for line in group)
                kind = SegmentKind.PARAGRAPH
                if all(_is_heading(line, body_size) for line in group) and len(text) < 300:
                    kind = SegmentKind.HEADING
                elif size < body_size * 0.88 and re.match(r"^\d+[.\s]", text):
                    kind = SegmentKind.FOOTNOTE
                elif re.match(r"^(?:Figure|Map|Table|Fig\.)\s+\d", text):
                    kind = SegmentKind.CAPTION
                page_blocks.append(
                    ParsedBlock(
                        kind,
                        text,
                        heading or f"第 {number} 页",
                        (f"pdf:page:{number}",),
                        0.85,
                        "pdf_layout",
                        VERSION,
                    )
                )
                page_layout.append(
                    {
                        "source_text": text,
                        "regions": [
                            {
                                "page": number,
                                "font_size": size,
                                "rect": [
                                    min(line["x0"] for line in group),
                                    height - max(line["bottom"] for line in group),
                                    max(line["x1"] for line in group),
                                    height - min(line["top"] for line in group),
                                ],
                                "lines": [
                                    [
                                        line["x0"],
                                        height - line["bottom"],
                                        line["x1"],
                                        height - line["top"],
                                    ]
                                    for line in group
                                ],
                            }
                        ],
                    }
                )
                group.clear()

            for line in kept:
                if group and (
                    _paragraph_break(
                        group,
                        line,
                        body_size=body_size,
                        left=left,
                        right=right,
                        normal_gap=normal_gap,
                    )
                    or sum(len(x["text"]) for x in group) > 3500
                ):
                    flush()
                group.append(line)
            flush()
            # Merge only a clear continuation, preserving every source page reference.
            if blocks and page_blocks and not entries:
                prev, first = blocks[-1], page_blocks[0]
                if (
                    prev.kind == first.kind == SegmentKind.PARAGRAPH
                    and prev.heading_path == first.heading_path
                    and prev.source_refs[-1] == f"pdf:page:{number - 1}"
                    and not re.search(r"[.!?。！？][\d\"”’]*$", prev.text)
                    and re.match(r"^[a-z]", first.text)
                    and len(prev.text + first.text) < 4500
                ):
                    blocks[-1] = replace(
                        prev,
                        text=_join(prev.text, first.text),
                        source_refs=prev.source_refs + first.source_refs,
                    )
                    layout[-1]["source_text"] = blocks[-1].text
                    layout[-1]["regions"].extend(page_layout.pop(0)["regions"])
                    page_blocks.pop(0)
            blocks.extend(page_blocks)
            layout.extend(page_layout)
        text_chars = sum(len(block.text) for block in blocks)
        if text_chars < 30:
            raise ValueError(
                "未检测到可翻译的文字层。这可能是扫描版 PDF，请先进行 OCR 识别后再导入。"
            )
        empty = [p["number"] for p in page_info if p["text_chars"] < 20]
        if empty:
            warnings.append(
                f"{len(empty)} 页文字较少或没有文字层（如封面、地图或扫描页），原页已保留；图片中的文字尚未识别。"
            )
        warnings.append("复杂表格、多栏和脚注请结合原页核对；可在工作台编辑、拆分或合并原文。")
        return PdfBook(
            title,
            tuple(blocks),
            tuple(page_info),
            tuple(navigation),
            tuple(warnings),
            tuple(layout),
        )
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("PDF 解析失败，文件可能损坏或使用了不受支持的编码。") from exc


def render_pdf_page(data: bytes, page_number: int, width: int = 1200) -> bytes:
    with _RENDER_LOCK, pdfium.PdfDocument(data) as pdf:
        if not 1 <= page_number <= len(pdf):
            raise ValueError("PDF 页码超出范围")
        page = pdf[page_number - 1]
        try:
            # Cap pixel area as well as width for unusually tall pages.
            scale = min(
                width / max(page.get_width(), 1),
                (8_000_000 / max(page.get_width() * page.get_height(), 1)) ** 0.5,
            )
            bitmap = page.render(scale=scale)
            try:
                output = io.BytesIO()
                bitmap.to_pil().save(output, format="PNG")
                return output.getvalue()
            finally:
                bitmap.close()
        finally:
            page.close()
