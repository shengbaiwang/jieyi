"""PDF text ingestion with page provenance; document content is never executable."""

from __future__ import annotations

import io
import re
import threading
import unicodedata
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, replace
from html import unescape
from statistics import median

import pdfplumber
import pypdfium2 as pdfium
from pypdf import PdfReader

from jieyi.domain.models import SegmentKind
from jieyi.ingestion.plaintext import ParsedBlock

VERSION = "pdf-layout-v1"
# PDFium is not thread-safe, including separate documents.
_RENDER_LOCK = threading.Lock()


@dataclass(frozen=True)
class PdfBook:
    title: str
    blocks: tuple[ParsedBlock, ...]
    pages: tuple[dict, ...]
    navigation: tuple[dict, ...]
    warnings: tuple[str, ...]


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
                raw_lines = page.extract_text_lines(x_tolerance=2, y_tolerance=3)
                lines = []
                for line in raw_lines:
                    chars = line.pop("chars")
                    line["size"] = median(c["size"] for c in chars) if chars else 10
                    line["text"] = _line_text(chars)
                    sizes[round(line["size"], 1)] += len(line["text"])
                    lines.append(line)
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
                if edge and (
                    _margin_key(text) in repeated or re.fullmatch(r"\d+|[ivxlcdm]+", text)
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
            left = min(line["x0"] for line in kept)
            right = max(line["x1"] for line in kept)
            group: list[dict] = []
            page_blocks: list[ParsedBlock] = []

            def flush(group=group, page_blocks=page_blocks, heading=heading, number=number):
                if not group:
                    return
                text = group[0]["text"]
                for line in group[1:]:
                    text = _join(text, line["text"])
                size = median(line["size"] for line in group)
                kind = SegmentKind.PARAGRAPH
                if size > body_size * 1.18 and len(text) < 300:
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
                group.clear()

            for line in kept:
                if group:
                    prev = group[-1]
                    gap = line["top"] - prev["bottom"]
                    indent = line["x0"] - left
                    size_change = abs(line["size"] - prev["size"]) > body_size * 0.15
                    new_paragraph = 4 < indent < 30 and prev["x0"] < line["x0"] - 3
                    short_end = prev["x1"] < right - 35 and re.search(
                        r"[.!?。！？][\d\"”’]*$", prev["text"]
                    )
                    if (
                        gap > body_size * 0.65
                        or size_change
                        or new_paragraph
                        or short_end
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
                    page_blocks.pop(0)
            blocks.extend(page_blocks)
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
        return PdfBook(title, tuple(blocks), tuple(page_info), tuple(navigation), tuple(warnings))
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
