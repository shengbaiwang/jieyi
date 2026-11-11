"""Translate native PDF text objects while retaining original graphics and page geometry."""

from __future__ import annotations

import hashlib
import io
import os
import re
import unicodedata
from collections import defaultdict
from copy import copy
from ctypes import byref, c_float, c_int, c_uint, c_void_p, cast
from dataclasses import is_dataclass, replace
from functools import lru_cache
from itertools import pairwise
from pathlib import Path
from statistics import median
from threading import RLock

import pypdfium2 as pdfium
from pypdf import PdfReader, PdfWriter, Transformation
from pypdf.generic import NameObject, RectangleObject, StreamObject
from pypdfium2 import raw
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFError, TTFont
from reportlab.pdfgen.canvas import Canvas

from jieyi.ingestion.pdf import (
    _RENDER_LOCK,
    GEOMETRY_VERSION,
    enrich_layout_fonts,
    extract_pdf,
    repair_layout_geometry,
)
from jieyi.ingestion.typography import FontStyle, pdf_font_style

PDF_LAYOUT_VERSION = "direct-flow-v4"
_EXPORT_LOCK = RLock()
_SUPERSCRIPT_DIGITS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")


@lru_cache(maxsize=1)
def _installed_kai_fonts():
    """Use an installed Kai face without copying personal font files into books."""
    candidates = []
    directory = Path.home() / "Library" / "Fonts"
    if directory.is_dir():
        for candidate in sorted(directory.glob("*.ttf")):
            if re.search(r"wenkai|kaiti|ukai|楷", candidate.name, re.IGNORECASE):
                candidates.append((str(candidate), 0))
    return candidates


def _font(style=None):
    style = style or FontStyle()
    role = style.chinese_family
    suffix = "_BOLD" if style.bold else ""
    override = os.getenv(f"JIEYI_PDF_FONT_{role.upper()}{suffix}", "")
    if not override:
        override = os.getenv("JIEYI_PDF_FONT", "")
    # Include the configured path so switching fonts does not reuse stale metrics.
    name = f"JieyiCJK_{role}{suffix}_{hashlib.sha256(override.encode()).hexdigest()[:8]}"
    if name in pdfmetrics.getRegisteredFontNames():
        return name
    serif = [
        ("/System/Library/Fonts/Supplemental/Songti.ttc", 1 if style.bold else 6),
        ("/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc" if style.bold
         else "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc", 2),
        ("C:/Windows/Fonts/simsun.ttc", 0),
    ]
    sans = [
        ("/System/Library/Fonts/STHeiti Medium.ttc" if style.bold
         else "/System/Library/Fonts/STHeiti Light.ttc", 1),
        ("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf", 0),
        ("C:/Windows/Fonts/msyhbd.ttc" if style.bold else "C:/Windows/Fonts/msyh.ttc", 0),
    ]
    kai = [
        ("/System/Library/Fonts/Supplemental/Kaiti.ttc", 0),
        ("/System/Library/Fonts/Supplemental/STKaiti.ttf", 0),
        ("/usr/share/fonts/truetype/arphic/ukai.ttc", 0),
        ("C:/Windows/Fonts/simkai.ttf", 0),
    ] + _installed_kai_fonts() + [
        # A lighter Song face retains a distinct emphasis when Kai is unavailable.
        ("/System/Library/Fonts/Supplemental/Songti.ttc", 1 if style.bold else 3),
    ]
    candidates = ([(override, 0)] if override else [])
    if role == "kai":
        candidates += kai + serif + sans
    elif role in {"sans", "mono"}:
        candidates += sans + serif
    else:
        candidates += serif + sans
    for candidate, index in candidates:
        if candidate and Path(candidate).is_file():
            try:
                pdfmetrics.registerFont(TTFont(name, candidate, subfontIndex=index))
                return name
            except (ValueError, OSError, TTFError):
                continue
    raise ValueError("未找到可嵌入的中文字体，请通过 JIEYI_PDF_FONT 指定中文 TTF/TTC 字体。")


def _region_font(region):
    return _font(pdf_font_style(region.get("font_name", "")))


_LEADER = re.compile(r"^([^\n]+?)[.·…⋯]{5,}\s*(\d[\d ,–-]*)$")
_COMPOUND_LEADER = re.compile(
    r"^(?P<leader>[^\n]+?[.·…⋯]{5,}\s*(?P<folio>\d[\d ,–-]*))"
    r"(?P<separator>\s+)(?P<suffix>[^\n]+)$"
)


class _LeaderText(str):
    """TOC dots are elastic spacing, not translated paragraph content."""
    def __new__(cls, text):
        value = super().__new__(cls, text)
        match = _LEADER.fullmatch(text)
        value.label, value.folio = match.group(1).rstrip(), match.group(2).strip()
        return value


def _unspaced_script(char):
    name = unicodedata.name(char, "")
    return any(script in name for script in ("CJK", "HIRAGANA", "KATAKANA", "IDEOGRAPHIC", "BOPOMOFO"))


def _inline_text(runs, word_breaks):
    parts, fonts, colors = [], [], []
    previous = ""
    for index, text in enumerate(runs):
        # Preserve source word boundaries without splitting extracted G/ulf.
        if (previous and text and word_breaks[index - 1]
                and not (_unspaced_script(previous[-1]) or _unspaced_script(text[0]))
                and not previous[-1].isspace() and not text[0].isspace()):
            parts.append(" ")
            fonts.append(None)
            colors.append(None)
        parts.append(str(text))
        fonts.extend(text.fonts)
        colors.extend(text.colors)
        previous = text
    return _StyledText("".join(parts), fonts, colors)


class _StyledText(str):
    """Character font ownership survives wrapping and page-local reflow."""
    def __new__(cls, text, fonts, colors=None):
        value = super().__new__(cls, text)
        value.fonts = tuple(fonts)
        value.colors = tuple(colors) if colors is not None else (None,) * len(text)
        return value


def _slice_text(text, start, end=None):
    value = text[start:end]
    return (_StyledText(value, text.fonts[start:end], text.colors[start:end])
            if isinstance(text, _StyledText) else value)


def _split_lines(text):
    lines, offset = [], 0
    for part in text.splitlines(keepends=True):
        length = len(part.rstrip("\r\n"))
        lines.append(_slice_text(text, offset, offset + length))
        offset += len(part)
    return lines or [text]


def _join_lines(lines):
    text = "\n".join(lines)
    if not any(isinstance(line, _StyledText) for line in lines):
        return text
    fonts, colors = [], []
    for index, line in enumerate(lines):
        if index:
            fonts.append(None)
            colors.append(None)
        fonts.extend(line.fonts if isinstance(line, _StyledText) else [None] * len(line))
        colors.extend(line.colors if isinstance(line, _StyledText) else [None] * len(line))
    return _StyledText(text, fonts, colors)


def _text_width(text, font, size):
    if not isinstance(text, _StyledText):
        return pdfmetrics.stringWidth(text, font, size)
    return sum(pdfmetrics.stringWidth(char, name or font, size)
               for char, name in zip(text, text.fonts))


def _translation(segment):
    return (
        (
            segment.accepted_translation
            or segment.reviewed_translation
            or segment.edited_translation
            or segment.machine_translation
            or ""
        )
        .strip()
        .translate(_SUPERSCRIPT_DIGITS)
    )


def _inside(inner, outer, tolerance=1.2):
    return all(
        (
            inner[0] >= outer[0] - tolerance,
            inner[1] >= outer[1] - tolerance,
            inner[2] <= outer[2] + tolerance,
            inner[3] <= outer[3] + tolerance,
        )
    )


def _overlaps(a, b):
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def _fill_rgb(obj):
    rgba = [c_uint() for _ in range(4)]
    if (raw.FPDFPageObj_GetFillColor(obj, *(byref(v) for v in rgba))
            and rgba[3].value == 255):
        return tuple(v.value / 255 for v in rgba[:3])
    return None


def _text_em(obj):
    size = c_float()
    if raw.FPDFTextObj_GetFontSize(obj, byref(size)):
        matrix = obj.get_matrix()
        origin, vertical = matrix.on_point(0, 0), matrix.on_point(0, 1)
        scale = ((vertical[0] - origin[0]) ** 2 + (vertical[1] - origin[1]) ** 2) ** .5
        return size.value * scale
    return 0


def _region_text_color(region, objects, geometry):
    colors = defaultdict(int)
    for obj in objects:
        if obj.type != raw.FPDF_PAGEOBJ_TEXT:
            continue
        count = sum(_inside(box, region["rect"], region["font_size"] * .3)
                    for box in geometry.get(_object_key(obj), []))
        if count:
            color = _fill_rgb(obj)
            if color is not None:
                colors[color] += count
    return max(colors, key=colors.get) if colors and max(colors.values()) else (.08, .08, .08)


def _background_objects(page, objects, geometry):
    """Recognize text backplates by geometry, paint order and glyph ownership.

    Keep every path in the PDF. Only simple, opaque, light text backplates
    behind source text are excluded from collision tests. Foreground masks,
    frames, translucent fills and plates over artwork remain obstacles.
    """
    backgrounds = set()
    plate_colors = {}
    space_bridges = []
    frame_rules = None

    def framed_padding(bounds, parent):
        nonlocal frame_rules
        # Word can extend the top/bottom fill to the centres of separately
        # drawn frame edges. This is stronger ownership evidence than a larger
        # global tolerance: an unframed spill must still remain an obstacle.
        if (bounds[3] - bounds[1] > min(24, (parent[3] - parent[1]) * .5)
                or bounds[2] - bounds[0] < (parent[2] - parent[0]) * .9
                or bounds[1] < parent[1] - .3 or bounds[3] > parent[3] + .3
                or min(abs(bounds[1] - parent[1]), abs(bounds[3] - parent[3])) > .3
                or parent[0] - bounds[0] > 1.5 or bounds[2] - parent[2] > 1.5):
            return False
        if frame_rules is None:
            frame_rules = []
            for rule in objects:
                if rule.type != raw.FPDF_PAGEOBJ_PATH:
                    continue
                box = _bounds(rule, geometry)
                if not (0 < box[2] - box[0] <= 1.5
                        and box[3] - box[1] > (box[2] - box[0]) * 2):
                    continue
                fill, stroke = c_int(), c_int()
                color = _fill_rgb(rule)
                if (color is None or max(color) > .85
                        or raw.FPDFPath_CountSegments(rule) not in (4, 5)
                        or not raw.FPDFPath_GetDrawMode(rule, byref(fill), byref(stroke))
                        or not fill.value or stroke.value):
                    continue
                frame_rules.append((box, color))
        rows = [(box, color) for box, color in frame_rules
                if box[1] <= bounds[1] + .3 and box[3] >= bounds[3] - .3]
        left = [(box, color) for box, color in rows
                if box[0] <= bounds[0] + .05 and box[2] >= parent[0] - .3
                and box[0] <= parent[0] and box[2] <= parent[0] + .3]
        right = [(box, color) for box, color in rows
                 if box[2] >= bounds[2] - .05 and box[0] <= parent[2] + .3
                 and box[2] >= parent[2] and box[0] >= parent[2] - .3]
        return any(a_color == b_color for _, a_color in left for _, b_color in right)
    for index, obj in enumerate(objects):
        if obj.type != raw.FPDF_PAGEOBJ_PATH:
            continue
        fill, stroke = c_int(), c_int()
        rgba = [c_uint() for _ in range(4)]
        if not raw.FPDFPath_GetDrawMode(obj, byref(fill), byref(stroke)):
            continue
        if not fill.value or stroke.value:
            continue
        if not raw.FPDFPageObj_GetFillColor(obj, *(byref(v) for v in rgba)):
            continue
        r, g, b, alpha = (v.value for v in rgba)
        luminance = r * .299 + g * .587 + b * .114
        if alpha != 255:
            continue
        light = (min(r, g, b) >= 220 and luminance >= 225
                 and max(r, g, b) - min(r, g, b) <= 35)
        count = raw.FPDFPath_CountSegments(obj)
        if count not in (4, 5):
            continue
        matrix, points = obj.get_matrix(), []
        for n in range(count):
            segment = raw.FPDFPath_GetPathSegment(obj, n)
            if raw.FPDFPathSegment_GetType(segment) != (
                raw.FPDF_SEGMENT_MOVETO if n == 0 else raw.FPDF_SEGMENT_LINETO
            ):
                break
            x, y = c_float(), c_float()
            if not raw.FPDFPathSegment_GetPoint(segment, byref(x), byref(y)):
                break
            points.append(matrix.on_point(x.value, y.value))
        if len(points) != count:
            continue
        corners = {(round(x, 2), round(y, 2)) for x, y in points}
        xs, ys = {x for x, _ in corners}, {y for _, y in corners}
        if len(corners) != 4 or len(xs) != 2 or len(ys) != 2:
            continue
        # Four corners must form a rectangle, not a bow-tie or an open polyline.
        ring = points if len(points) == 5 else points + points[:1]
        if any(abs(a[0] - b[0]) > .02 and abs(a[1] - b[1]) > .02
               for a, b in pairwise(ring)):
            continue
        bounds = _bounds(obj, geometry)
        earlier_ink = any(
            min(_bounds(other, geometry)[2], bounds[2])
            - max(_bounds(other, geometry)[0], bounds[0]) > .05
            and min(_bounds(other, geometry)[3], bounds[3])
            - max(_bounds(other, geometry)[1], bounds[1]) > .05
            and (other.type == raw.FPDF_PAGEOBJ_TEXT
                 or (_object_key(other) not in backgrounds
                     and _graphic_intersects(other, bounds)))
            for other in objects[:index]
        )
        if earlier_ink:
            continue
        # Paragraph shading can be split into nested row/padding rectangles.
        # An identical child of an already verified plate adds no new artwork;
        # the earlier-ink test still prevents treating a foreground mask this way.
        if any(_object_key(parent) in backgrounds
               and plate_colors.get(_object_key(parent)) == (r, g, b)
               and (_inside(bounds, _bounds(parent, geometry), .3)
                    or framed_padding(bounds, _bounds(parent, geometry)))
               for parent in objects[:index]):
            backgrounds.add(_object_key(obj))
            plate_colors[_object_key(obj)] = (r, g, b)
            continue
        boxes = [box for other in objects[index + 1:]
                 if other.type == raw.FPDF_PAGEOBJ_TEXT
                 for box in geometry.get(_object_key(other), [])
                 if _inside(box, bounds, .8)]
        if not boxes:
            if 0 < bounds[2] - bounds[0] <= (bounds[3] - bounds[1]) * .5:
                space_bridges.append((obj, bounds, (r, g, b)))
            continue
        bright_owners = 0
        for other in objects[index + 1:]:
            if other.type != raw.FPDF_PAGEOBJ_TEXT:
                continue
            color = _fill_rgb(other)
            text_luminance = (sum(c * w for c, w in zip(color, (.299, .587, .114))) * 255
                              if color is not None else 0)
            if text_luminance >= 150 and text_luminance - luminance >= 100:
                bright_owners += sum(_inside(box, bounds, .8)
                                     for box in geometry.get(_object_key(other), []))
        if not light and bright_owners < len(boxes) * .8:
            continue
        glyph_height = max(b[3] - b[1] for b in boxes)
        em = max([glyph_height] + [_text_em(other) for other in objects[index + 1:]
                                    if other.type == raw.FPDF_PAGEOBJ_TEXT
                                    and any(_inside(box, bounds, .8)
                                            for box in geometry.get(_object_key(other), []))])
        row_plate = bounds[3] - bounds[1] <= em * 3.5
        page_plate = _inside(page.get_bbox(), bounds, 1)
        # Multi-line callouts use one pastel rectangle behind a whole paragraph.
        # Require substantial owned text, rather than treating a pale illustration
        # with one small label as a paragraph backplate. Independent border paths
        # remain obstacles and still constrain the translated text to the card.
        text_height = max(b[3] for b in boxes) - min(b[1] for b in boxes)
        # Verse may occupy a narrow strip of a broad container. Vertical text
        # ownership establishes the paragraph without demanding prose-like width.
        multiline_plate = (len(boxes) >= 8 and text_height > glyph_height * 1.5
                           and text_height >= (bounds[3] - bounds[1]) * .4)
        if row_plate or page_plate or multiline_plate:
            backgrounds.add(_object_key(obj))
            plate_colors[_object_key(obj)] = (r, g, b)
    # Word can shade one space as its own rectangle between two text-owned run
    # fills. Resolve these only after both neighbours have been verified. An
    # isolated white patch, wide blank, foreground mask or artwork has no such
    # two-sided row ownership and stays an obstacle.
    if space_bridges:
        rows = []
        for obj in objects:
            key = _object_key(obj)
            if key not in backgrounds:
                continue
            box = _bounds(obj, geometry)
            color = plate_colors[key]
            if not any(color == bridge_color
                       and abs(box[1] - bridge[1]) <= .05
                       and abs(box[3] - bridge[3]) <= .05
                       for _, bridge, bridge_color in space_bridges):
                continue
            em = max([0] + [_text_em(other) for other in objects
                            if other.type == raw.FPDF_PAGEOBJ_TEXT
                            and any(_inside(ink, box, .8)
                                    for ink in geometry.get(_object_key(other), []))])
            if em and box[3] - box[1] <= em * 3.5:
                rows.append((box, color, em))
        for obj, bounds, color in space_bridges:
            matching = [(box, em) for box, row_color, em in rows
                        if row_color == color and abs(box[1] - bounds[1]) <= .05
                        and abs(box[3] - bounds[3]) <= .05
                        and bounds[2] - bounds[0] <= em * .6]
            if (any(abs(box[2] - bounds[0]) <= .05 for box, _ in matching)
                    and any(abs(box[0] - bounds[2]) <= .05 for box, _ in matching)):
                backgrounds.add(_object_key(obj))
    return backgrounds


def _text_decorations(objects, geometry):
    """Thin source-owned underlines are typography, not illustration obstacles."""
    decorations = set()
    glyphs = [box for obj in objects if obj.type == raw.FPDF_PAGEOBJ_TEXT
              for box in geometry.get(_object_key(obj), [])]
    for obj in objects:
        if obj.type != raw.FPDF_PAGEOBJ_PATH:
            continue
        bounds = _bounds(obj, geometry)
        width, height = bounds[2] - bounds[0], bounds[3] - bounds[1]
        if height > .9 or width < 6:
            continue
        fill, stroke = c_int(), c_int()
        if not raw.FPDFPath_GetDrawMode(obj, byref(fill), byref(stroke)):
            continue
        if not fill.value or stroke.value or raw.FPDFPath_CountSegments(obj) not in (4, 5):
            continue
        rgba = [c_uint() for _ in range(4)]
        if not raw.FPDFPageObj_GetFillColor(obj, *(byref(v) for v in rgba)):
            continue
        luminance = sum(v.value * weight for v, weight in zip(rgba[:3], (.299, .587, .114)))
        if rgba[3].value != 255 or luminance > 180:
            continue
        # Every owned glyph sits above the underline; an interior black bar or
        # unrelated rule must not qualify merely because it is thin and dark.
        owners = [box for box in glyphs
                  if box[0] >= bounds[0] - 1 and box[2] <= bounds[2] + 1
                  and -(box[3] - box[1]) * .25 <= box[1] - bounds[3]
                      <= (box[3] - box[1]) * .55]
        if (len(owners) >= 3
                and max(box[2] for box in owners) - min(box[0] for box in owners) >= width * .8
                # Descenders can cross a real underline, but most of the
                # source row must sit above it rather than behind a black bar.
                and sum(_overlaps(box, bounds) for box in glyphs)
                    <= max(1, len(owners) * .35)):
            decorations.add(_object_key(obj))
    return decorations


def _graphic_intersects(obj, rect, backgrounds=()):
    if _object_key(obj) in backgrounds:
        return False
    if obj.type != raw.FPDF_PAGEOBJ_PATH:
        return True
    fill, stroke = c_int(), c_int()
    # A filled shape's interior is occupied even when no edge crosses the text.
    if not raw.FPDFPath_GetDrawMode(obj, byref(fill), byref(stroke)) or fill.value:
        return True
    matrix = obj.get_matrix()
    previous = None
    for index in range(raw.FPDFPath_CountSegments(obj)):
        segment = raw.FPDFPath_GetPathSegment(obj, index)
        x, y = c_float(), c_float()
        if not raw.FPDFPathSegment_GetPoint(segment, x, y):
            return True
        point = matrix.on_point(x.value, y.value)
        if previous is not None and raw.FPDFPathSegment_GetType(segment) != raw.FPDF_SEGMENT_MOVETO:
            bounds = (
                min(previous[0], point[0]) - 0.5,
                min(previous[1], point[1]) - 0.5,
                max(previous[0], point[0]) + 0.5,
                max(previous[1], point[1]) + 0.5,
            )
            if _overlaps(bounds, rect):
                return True
        previous = point
    return False


def _object_key(obj):
    # Page-object helpers stay alive for the entire page operation. Retain their
    # native key so repeated geometry checks need not cross ctypes each time.
    if isinstance(obj, pdfium.PdfObject):
        key = getattr(obj, "_jieyi_object_key", None)
        if key is None:
            key = cast(obj, c_void_p).value
            obj._jieyi_object_key = key
        return key
    return cast(obj, c_void_p).value


class _TextGeometry(defaultdict):
    """Page-local glyphs and immutable source-object bounds."""

    def __init__(self):
        super().__init__(list)
        self.bounds = {}


def _text_geometry(page):
    """Use visible glyphs, excluding PDF spacing/indentation from object bounds."""
    geometry = _TextGeometry()
    textpage = page.get_textpage()
    try:
        for index in range(textpage.count_chars()):
            code = raw.FPDFText_GetUnicode(textpage, index)
            if not code or chr(code).isspace():
                continue
            obj = raw.FPDFText_GetTextObject(textpage, index)
            if obj:
                geometry[_object_key(obj)].append(textpage.get_charbox(index))
    finally:
        textpage.close()
    return geometry


def _bounds(obj, geometry):
    key = _object_key(obj)
    cache = getattr(geometry, "bounds", None)
    if cache is not None and key in cache:
        return cache[key]
    boxes = geometry.get(key)
    if not boxes:
        bounds = obj.get_bounds()
    else:
        bounds = (
            min(b[0] for b in boxes),
            min(b[1] for b in boxes),
            max(b[2] for b in boxes),
            max(b[3] for b in boxes),
        )
    if cache is not None:
        cache[key] = bounds
    return bounds


def _belongs(obj, region, geometry):
    boxes = geometry.get(_object_key(obj), [obj.get_bounds()])
    # PDFium and pdfplumber derive glyph bounds independently. Type1 overhangs
    # routinely differ by a little more than one point, which used to make an
    # otherwise exact paragraph fall back to the untranslated source.
    tolerance = max(2.0, float(region["font_size"]) * 0.3)
    return all(any(_inside(box, line, tolerance) for line in region["lines"]) for box in boxes)


def _native_region(region, page):
    """Map displayed pdfplumber coordinates back to native PDF object space."""
    if region.get("native_space"):
        return region
    rotation = (page.rotation if hasattr(page, "rotation") else page.get_rotation()) % 360
    if not rotation:
        return region
    left, bottom, right, top = (tuple(map(float, page.mediabox))
                                if hasattr(page, "mediabox") else page.get_bbox())

    def point(x, y):
        if rotation == 90:
            return right + left - y, x
        if rotation == 180:
            return right + left - x, top + bottom - y
        if rotation == 270:
            return y, top + bottom - x
        raise ValueError(f"不支持的 PDF 页面旋转角度：{rotation}")

    def rectangle(rect):
        points = [
            point(rect[0], rect[1]),
            point(rect[0], rect[3]),
            point(rect[2], rect[1]),
            point(rect[2], rect[3]),
        ]
        return [
            min(p[0] for p in points),
            min(p[1] for p in points),
            max(p[0] for p in points),
            max(p[1] for p in points),
        ]

    native = {**region, "rect": rectangle(region["rect"])}
    native["lines"] = [rectangle(line) for line in region["lines"]]
    native["native_space"] = True
    if rotation in (90, 270):
        # pdfminer reports rotated glyph width as size. Native line height
        # recovers the typographic size without shrinking narrow letters.
        native["font_size"] = median(line[3] - line[1] for line in native["lines"])
    if "draw_rect" in region:
        native["draw_rect"] = rectangle(region["draw_rect"])
    return native


def _outside_visible_page(region, page, objects, geometry):
    """Ignore legacy printer text only when all its native ink is invisible."""
    native = _native_region(region, page)
    bbox = page.get_bbox()
    if _overlaps(native["rect"], bbox):
        return False
    owned = [box for obj in objects if obj.type == raw.FPDF_PAGEOBJ_TEXT
             and _belongs(obj, native, geometry)
             for box in geometry.get(_object_key(obj), [])]
    return bool(owned) and all(not _overlaps(box, bbox) for box in owned)


def _centered(region, bbox):
    return (
        not region.get("force_left")
        and region["rect"][2] - region["rect"][0] < (bbox[2] - bbox[0]) * 0.85
        and all(abs((line[0] + line[2] - bbox[0] - bbox[2]) / 2) < 4 for line in region["lines"])
    )


def _container_bounds(region, page, objects, geometry, backgrounds=()):
    """Return owned paragraph containers, excluding row plates and page fills."""
    containers = []
    for obj in objects:
        if _object_key(obj) not in backgrounds:
            continue
        bounds = _bounds(obj, geometry)
        color = _fill_rgb(obj)
        colored = (color is not None
                   and (max(color) - min(color) > .03 or max(color) < .7))
        if ((colored or bounds[3] - bounds[1] > float(region["font_size"]) * 2.5)
                and _inside(region["rect"], bounds, .1)
                and not _inside(page.get_bbox(), bounds, 1)):
            containers.append(bounds)
    return containers


def _expanded_region(region, neighbors, page, objects, geometry, backgrounds=()):
    """Use whitespace within the original column, stopping before all other content."""
    x0, y0, x1, y1 = region["rect"]
    bbox = page.get_bbox()
    size = region["font_size"]
    centered = _centered(region, bbox)
    if centered:
        margin = (bbox[2] - bbox[0]) * 0.08
        left, right = bbox[0] + margin, bbox[2] - margin
    else:
        # A short line is not a narrow column. Recover the column from aligned
        # neighboring paragraphs, without spanning a differently aligned column.
        aligned = [
            r["rect"]
            for r in neighbors
            if abs(r["rect"][0] - x0) <= size * 3 and r["rect"][0] < x1 and r["rect"][2] > x0
        ]
        left = x0
        right = min(bbox[2] - 8, max([x1] + [r[2] for r in aligned]))
    containers = _container_bounds(region, page, objects, geometry, backgrounds)
    if containers and not centered:
        # An owned card supplies a reliable column even when its short source
        # label has no wider aligned neighbor. Retain its existing left padding.
        edge = min(container[2] - max(1, x0 - container[0]) for container in containers)
        right = max(right, edge)
    bottom = max(bbox[1] + 8, y0 - size * 3)
    for obj in objects:
        bounds = _bounds(obj, geometry)
        if obj.type == raw.FPDF_PAGEOBJ_TEXT and _belongs(obj, region, geometry):
            continue
        candidate = (left, bottom, right, y1)
        if not _overlaps(bounds, candidate):
            continue
        if obj.type != raw.FPDF_PAGEOBJ_TEXT and not _graphic_intersects(obj, candidate, backgrounds):
            continue
        # Keep a gap to the next paragraph; never move text or cover an illustration.
        if bounds[3] <= y0 + 1:
            bottom = max(bottom, bounds[3] + 1)
        elif bounds[0] >= x1 - 1:
            right = min(right, (bounds[0] + x1) / 2 - 0.5)
        elif bounds[2] <= x0 + 1:
            left = max(left, (bounds[2] + x0) / 2 + 0.5)
    # Also reserve the full typographic boxes of neighboring paragraphs. Ink bounds
    # are smaller and would otherwise let two translated paragraphs touch.
    for other in neighbors:
        if other is region:
            continue
        bounds = other["rect"]
        if bounds[3] <= y0 + 0.1 and min(right, bounds[2]) > max(left, bounds[0]):
            bottom = max(bottom, bounds[3] + 1)
    if centered:
        center = (x0 + x1) / 2
        radius = max((x1 - x0) / 2, min(center - left, right - center))
        left, right = center - radius, center + radius
    for container in containers:
        bottom = max(bottom, container[1])
    # Spend the same three-em growth budget above the region when the space
    # below is occupied or cropped. Measure against the untouched source.
    remaining_growth = max(0, size * 3 - max(0, y0 - bottom))
    top = max(y1, min(bbox[3] - 8, y1 + remaining_growth))
    for obj in objects:
        bounds = _bounds(obj, geometry)
        if obj.type == raw.FPDF_PAGEOBJ_TEXT and _belongs(obj, region, geometry):
            continue
        if not _overlaps(bounds, (left, y1, right, top)):
            continue
        if (obj.type != raw.FPDF_PAGEOBJ_TEXT
                and not _graphic_intersects(obj, (left, y1, right, top), backgrounds)):
            continue
        top = min(top, bounds[1] - 1) if bounds[1] >= y1 - 1 else y1
    for other in neighbors:
        bounds = other["rect"]
        if (other is not region and bounds[1] >= y1 - .1
                and min(right, bounds[2]) > max(left, bounds[0])):
            top = min(top, bounds[1] - 1)
    draw_rect = [min(left, x0), min(bottom, y0), max(right, x1), max(top, y1)]
    for container in containers:
        draw_rect = [max(draw_rect[0], container[0]), max(draw_rect[1], container[1]),
                     min(draw_rect[2], container[2]), min(draw_rect[3], container[3])]
    return {**region, "draw_rect": draw_rect}


def _paint_lines(canvas, region, lines, size, bbox):
    color = region.get("text_color", (.08, .08, .08))
    canvas.setFillColorRGB(*color)
    face = _region_font(region)
    canvas.setFont(face, size)
    x0, _, _, y1 = region.get("draw_rect", region["rect"])
    centered = _centered(region, bbox)
    for index, line in enumerate(lines):
        y = y1 - size * 0.85 - index * size * region.get("line_spacing", 1.22)
        x = ((region["rect"][0] + region["rect"][2]) / 2
             - _text_width(line, face, size) / 2) if centered else x0
        if isinstance(line, _StyledText):
            text_object = canvas.beginText(x, y)
            previous = previous_color = None
            for char, char_font, char_color in zip(line, line.fonts, line.colors):
                char_font = char_font or face
                char_color = char_color or color
                if char_color != previous_color:
                    text_object.setFillColorRGB(*char_color)
                    previous_color = char_color
                if char_font != previous:
                    text_object.setFont(char_font, size)
                    previous = char_font
                text_object.textOut(char)
            canvas.drawText(text_object)
        else:
            canvas.drawString(x, y, line)


def _measured_lines(region, lines, size, bbox):
    """Measure the exact overlay glyphs without rendering or acquiring render locks."""
    stream = io.BytesIO()
    canvas = Canvas(stream, pagesize=(max(bbox[2], 1), max(bbox[3], 1)))
    _paint_lines(canvas, region, lines, size, bbox)
    canvas.save()
    with pdfium.PdfDocument(stream.getvalue()) as document:
        geometry = _text_geometry(document[0])
        return [box for boxes in geometry.values() for box in boxes]


def _raised_baseline(region, lines, size, page, objects, geometry, backgrounds,
                     neighbors, previous_draws):
    """Clear a bottom rule only when source ink proves the row belongs above it."""
    if (page.get_rotation() or len(lines) != 1 or len(region["lines"]) != 1
            or abs(size - float(region["font_size"])) > .01):
        return region
    rect = region.get("draw_rect", region["rect"])
    obstacles = [obj for obj in objects if obj.type != raw.FPDF_PAGEOBJ_TEXT
                 and _overlaps(_bounds(obj, geometry), rect)
                 and _graphic_intersects(obj, rect, backgrounds)]
    if not obstacles:
        return region
    bounds = [_bounds(obj, geometry) for obj in obstacles]
    # Do not relocate text around an interior illustration or a rule through the row.
    if any(obj.type != raw.FPDF_PAGEOBJ_PATH for obj in obstacles) or any(
        box[3] > rect[1] + size * .15 or box[1] > rect[1] + size * .05
        for box in bounds
    ):
        return region
    owners = [obj for obj in objects if obj.type == raw.FPDF_PAGEOBJ_TEXT
              and geometry.get(_object_key(obj)) and _belongs(obj, region, geometry)
              and _overlaps(_bounds(obj, geometry), region["rect"])]
    source_boxes = [box for obj in owners for box in geometry[_object_key(obj)]]
    if not source_boxes or min(box[1] for box in source_boxes) <= max(box[3] for box in bounds):
        return region
    for obj in owners:
        matrix = obj.get_matrix()
        origin, horizontal = matrix.on_point(0, 0), matrix.on_point(1, 0)
        if abs(horizontal[1] - origin[1]) > .001 or horizontal[0] <= origin[0]:
            return region
    # A table's interior crossbar has vertical rules continuing on both sides.
    # A card's top frame only continues below its top rule.
    for obj in objects:
        if obj.type != raw.FPDF_PAGEOBJ_PATH or _object_key(obj) in backgrounds:
            continue
        box = _bounds(obj, geometry)
        if box[2] - box[0] > size * .2:
            continue
        if any(box[0] <= rule[2] + .1 and box[2] >= rule[0] - .1
               and box[1] < rule[1] - size * .1 and box[3] > rule[3] + size * .3
               for rule in bounds):
            return region
    bbox = tuple(map(float, page.get_bbox()))
    try:
        target_boxes = _measured_lines(region, lines, size, bbox)
    except (ValueError, OSError, RuntimeError, TTFError, pdfium.PdfiumError):
        # An unreadable font or unsupported PDF measurement never justifies moving text.
        return region
    collisions = [box for box in bounds if any(_overlaps(ink, box) for ink in target_boxes)]
    if not target_boxes or not collisions:
        return region
    # The measured ink and the conservative draw rectangle must both clear the
    # rule; some sans faces place their ink above the typographic lower edge.
    bottom = min(rect[1], min(box[1] for box in target_boxes))
    shift = max(box[3] for box in collisions) + size * .04 - bottom
    if not 0 < shift <= size * .3:
        return region
    adjusted = {**region, "draw_rect": [rect[0], rect[1] + shift, rect[2], rect[3] + shift]}
    moved = adjusted["draw_rect"]
    ink = [min(box[0] for box in target_boxes), min(box[1] for box in target_boxes) + shift,
           max(box[2] for box in target_boxes), max(box[3] for box in target_boxes) + shift]
    if (not _inside(moved, bbox, .1) or not _inside(ink, bbox, .01)
            or any(not _inside(moved, container, .1)
                   for container in _container_bounds(region, page, objects, geometry, backgrounds))
            or any(other["rect"] != region["rect"] and _overlaps(other["rect"], moved)
                   for other in neighbors)
            or any(_overlaps(old.get("draw_rect", old["rect"]), moved)
                   for _, old, _, _ in previous_draws)):
        return region
    owned_ids = {_object_key(obj) for obj in owners}
    for obj in objects:
        if obj.type == raw.FPDF_PAGEOBJ_TEXT and _object_key(obj) in owned_ids:
            continue
        box = _bounds(obj, geometry)
        if not (_overlaps(box, moved) or _overlaps(box, ink)):
            continue
        if (obj.type == raw.FPDF_PAGEOBJ_TEXT
                or (_overlaps(box, moved) and _graphic_intersects(obj, moved, backgrounds))
                or (_overlaps(box, ink) and _graphic_intersects(obj, ink, backgrounds))):
            return region
    return adjusted


def _text_key(text):
    return "".join(unicodedata.normalize("NFKC", text).split())


def _wrapped_lines(text, font, size, width):
    """Yield display lines with offsets in the original text, never soft breaks."""
    offset = 0
    for part in text.splitlines(keepends=True) or [text]:
        length = len(part.rstrip("\r\n"))
        paragraph = _slice_text(text, offset, offset + length)
        start, line_width = 0, 0
        for index, char in enumerate(paragraph):
            face = paragraph.fonts[index] or font if isinstance(paragraph, _StyledText) else font
            char_width = pdfmetrics.stringWidth(char, face, size)
            if index > start and line_width + char_width > width:
                line = paragraph[start:index]
                space = line.rfind(" ")
                if char.isalpha() and not _unspaced_script(char) and space > len(line) // 2:
                    end = start + space
                    yield _slice_text(paragraph, start, end), offset + end + 1
                    start = end + 1
                else:
                    yield _slice_text(paragraph, start, index), offset + index
                    start = index
                line_width = _text_width(_slice_text(paragraph, start, index), font, size)
            line_width += char_width
        yield _slice_text(paragraph, start), offset + len(part)
        offset += len(part)


def _wrap(text, font, size, width):
    return [line for line, _ in _wrapped_lines(text, font, size, width)]


def _toc_folio_regions(target, regions, page, objects, geometry):
    """Preserve native folios when a PDF has masked/replaced its old numbers.

    Such foreground masks must never become ignorable backgrounds. Translate
    only the label/leader on their left, retaining native number text and mask.
    """
    if len(regions) != 1:
        return target, regions
    region = regions[0]
    x0, y0, x1, y1 = region["rect"]
    width, size = x1 - x0, float(region["font_size"])
    candidates = []
    for obj in objects:
        if obj.type != raw.FPDF_PAGEOBJ_PATH:
            continue
        bounds = _bounds(obj, geometry)
        if (bounds[0] < x0 + width * .85 or bounds[2] < x1 - size
                or bounds[3] - bounds[1] > size * 2
                or not _overlaps(bounds, region["rect"])):
            continue
        fill, stroke = c_int(), c_int()
        rgba = [c_uint() for _ in range(4)]
        if (not raw.FPDFPath_GetDrawMode(obj, byref(fill), byref(stroke))
                or not fill.value or stroke.value
                or raw.FPDFPath_CountSegments(obj) not in (4, 5)
                or not raw.FPDFPageObj_GetFillColor(obj, *(byref(v) for v in rgba))
                or rgba[3].value != 255 or min(v.value for v in rgba[:3]) < 235):
            continue
        candidates.append(bounds)
    if not candidates:
        return target, regions
    # Confirm that the reserved strip contains numeric folio ink, not a figure.
    cutoff = min(x0 + width * .85, min(bounds[0] for bounds in candidates) - size)
    textpage = page.get_textpage()
    try:
        folio_chars = [(raw.FPDFText_GetUnicode(textpage, i), textpage.get_charbox(i))
                       for i in range(textpage.count_chars())
                       if _inside(textpage.get_charbox(i),
                                  [cutoff, y0 - size, x1 + size, y1 + size], .1)]
        codes = [code for code, _ in folio_chars]
        if not codes or any(code and not (chr(code).isdigit() or chr(code).isspace()
                                         or chr(code) in ".·…⋯") for code in codes):
            raise ValueError("PDF 目录页码区包含无法确认的遮挡内容，已停止排版。")
        number_boxes = [box for code, box in folio_chars if code and chr(code).isdigit()]
        if not number_boxes:
            raise ValueError("PDF 目录页码区缺少可确认的数字，已停止排版。")
        cutoff = min(min(bounds[0] for bounds in candidates) - .05,
                     min(box[0] for box in number_boxes) - .2)
    finally:
        textpage.close()
    adjusted = {**region, "rect": [x0, y0, cutoff, y1],
                "lines": [[line[0], line[1], min(cutoff, line[2]), line[3]]
                          for line in region["lines"] if line[0] < cutoff]}
    clipped = _LeaderText(str(target))
    clipped.folio = ""
    return clipped, [adjusted]



def _compound_toc_plans(segment, target, regions, page):
    """Split one verified contents row followed by a heading, never its mask.

    Text extraction may join these physical rows into one segment. Require
    matching source/target folios and exact native glyph ownership for both
    rows before constructing temporary plans with the original segment ID.
    """
    source = _COMPOUND_LEADER.fullmatch(segment.source_text)
    translated = _COMPOUND_LEADER.fullmatch(target)
    if (not source or not translated
            or not source.group("suffix").strip() or not translated.group("suffix").strip()
            or source.group("folio").strip() != translated.group("folio").strip()
            or len(regions) != 1 or len(regions[0]["lines"]) != 2):
        raise ValueError("PDF 复合目录条目缺少明确的两行映射或一致页码，已停止排版。")
    region = regions[0]
    lines = sorted(region["lines"], key=lambda line: line[3], reverse=True)
    if lines[0][1] < lines[1][3] - .1:
        raise ValueError("PDF 复合目录条目的文字行重叠，已停止排版。")
    source_parts = [source.group("leader"), source.group("suffix")]
    target_parts = [_LeaderText(translated.group("leader")), translated.group("suffix")]
    textpage = page.get_textpage()
    try:
        glyphs = [(textpage.get_charbox(i), raw.FPDFText_GetUnicode(textpage, i))
                  for i in range(textpage.count_chars())]
        for line, expected in zip(lines, source_parts, strict=True):
            owned = [(box, code) for box, code in glyphs
                     if code and _inside(box, line, region["font_size"] * .3)]
            actual = "".join(chr(code) for _, code in sorted(owned, key=lambda item: item[0][0]))
            if _text_key(actual) != _text_key(expected):
                raise ValueError("PDF 复合目录条目的原生文字归属不明确，已停止排版。")
    finally:
        textpage.close()
    plans = []
    for source_part, target_part, line in zip(source_parts, target_parts, lines, strict=True):
        changes = {"source_text": source_part, "accepted_translation": str(target_part)}
        if is_dataclass(segment):
            part = replace(segment, **changes)
        else:
            part = copy(segment)
            for name, value in changes.items():
                setattr(part, name, value)
        plans.append((part, target_part, [{**region, "rect": list(line), "lines": [list(line)]}]))
    return plans


def _readable_floor(base):
    # Preserve small source footnotes; never turn ordinary body text into microtext.
    return min(base, max(8.0, base * 0.7))


def _fit(text, regions, font, *, shrink=True, compact=False, minimum=None):
    base = min(float(region["font_size"]) for region in regions)
    if compact and not isinstance(text, _LeaderText):
        text = _join_lines([line for line in _split_lines(text) if line.strip()])
    floor = _readable_floor(base) if shrink else base
    if minimum is not None:
        floor = max(floor, min(base, float(minimum)))
    leading = 1.08 if compact else 1.22
    sizes = [base - step * 0.5 for step in range(max(1, int((base - floor) * 2) + 1))]
    if sizes[-1] > floor + 0.01:
        sizes.append(floor)
    for size in sizes:
        remaining, fitted = text, []
        for region in regions:
            if not remaining:
                fitted.append(({**region, "line_spacing": leading}, [], size))
                continue
            x0, y0, x1, y1 = region.get("draw_rect", region["rect"])
            capacity = max(0, int((y1 - y0 - size + .001) / (size * leading)) + 1)
            # Latin numerals and punctuation can be much narrower than one em.
            # The actual styled glyph widths below determine whether a line fits.
            if capacity == 0 or x1 <= x0:
                break
            face = _region_font(region)
            if isinstance(remaining, _LeaderText):
                label = remaining.label + " "
                folio = " " + remaining.folio
                available = x1 - x0 - _text_width(label + folio, face, size)
                count = int(available / _text_width(".", face, size))
                if count < 3 or len(regions) != 1:
                    break
                fitted.append(({**region, "line_spacing": leading},
                               [label + "." * count + folio], size))
                remaining = ""
                continue
            selected = []
            consumed = 0
            for line, end in _wrapped_lines(remaining, face, size, x1 - x0):
                if _text_width(line, face, size) > x1 - x0 + .01:
                    break
                selected.append(line)
                consumed = end
                if len(selected) == capacity:
                    break
            if not selected:
                break
            fitted.append(({**region, "line_spacing": leading}, selected, size))
            remaining = _slice_text(remaining, consumed)
        if not remaining and len(fitted) == len(regions):
            return fitted
    return None


def ensure_pdf_layout(store, document_id):
    with _EXPORT_LOCK:
        metadata = store.get_pdf_metadata(document_id)
        if not metadata.get("layout"):
            book = extract_pdf(store.get_original_pdf(document_id))
            metadata["layout"] = book.layout
            metadata["geometry_version"] = GEOMETRY_VERSION
            store.update_pdf_metadata(document_id, metadata)
        if metadata.get("geometry_version") != GEOMETRY_VERSION:
            metadata["layout"] = repair_layout_geometry(
                store.get_original_pdf(document_id), metadata["layout"]
            )
            metadata["geometry_version"] = GEOMETRY_VERSION
            store.update_pdf_metadata(document_id, metadata)
        if any(not region.get("font_name") for item in metadata["layout"] for region in item["regions"]):
            metadata["layout"] = enrich_layout_fonts(store.get_original_pdf(document_id), metadata["layout"])
            store.update_pdf_metadata(document_id, metadata)
        return metadata


def _spread_transform(page):
    """Place the visible, rotated crop at the origin without scaling its contents."""
    box = page.cropbox
    transform = Transformation().rotate(-page.rotation)
    corners = [transform.apply_on(point) for point in (
        box.lower_left, box.lower_right, box.upper_left, box.upper_right,
    )]
    left = min(point[0] for point in corners)
    bottom = min(point[1] for point in corners)
    width = max(point[0] for point in corners) - left
    height = max(point[1] for point in corners) - bottom
    return transform.translate(-left, -bottom), width, height


def _bilingual_spreads(writer, original):
    """Keep each page reference, so bookmarks and links still target the same pair."""
    gutter = 24
    for index, page in enumerate(writer.pages):
        source = original.pages[index]
        translated = copy(page)
        if "/Annots" in source:
            # Use the original reader's destinations for both halves. Cloning links
            # from this writer into itself can duplicate their target pages.
            translated[NameObject("/Annots")] = source["/Annots"]
        left, left_width, left_height = _spread_transform(source)
        right, right_width, right_height = _spread_transform(translated)
        height = max(left_height, right_height)
        # The existing page becomes the spread; retain catalog metadata, page labels,
        # outline hierarchy and destinations rather than importing a second page tree.
        for key in ("/Contents", "/Resources", "/Annots"):
            page.pop(NameObject(key), None)
        page.rotation = 0
        bounds = (0, 0, left_width + gutter + right_width, height)
        for key in ("/MediaBox", "/CropBox", "/TrimBox", "/BleedBox", "/ArtBox"):
            page[NameObject(key)] = RectangleObject(bounds)
        page.merge_transformed_page(source, left.translate(0, height - left_height))
        page.merge_transformed_page(
            translated, right.translate(left_width + gutter, height - right_height),
        )
    writer.page_layout = "/SinglePage"


def _fit_direct(text, regions, font):
    """Keep all text on its source regions, even when conservative fitting fails."""
    text = str(text)  # A directory's dots/page numbers must not prevent export.
    regions = [{**region, "font_size": max(float(region["font_size"]), .5),
                "draw_rect": [region["rect"][0], region["rect"][1],
                              max(region["rect"][2], region["rect"][0] + 1),
                              max(region["rect"][3], region["rect"][1] + 1)]}
               for region in regions]
    fitted = _fit(text, regions, font, compact=True)
    if fitted is not None:
        return fitted
    # Find a fitting scale, then recover as much of the original size as possible.
    # There is deliberately no readability gate in this last-resort export path.
    upper, scale = 1.0, 0.5

    def at_scale(value):
        scaled = [{**region, "font_size": float(region["font_size"]) * value}
                  for region in regions]
        return _fit(text, scaled, font, shrink=False, compact=True)

    fitted = at_scale(scale)
    while fitted is None:
        upper, scale = scale, scale / 2
        fitted = at_scale(scale)
    lower = scale
    for _ in range(7):
        middle = (lower + upper) / 2
        candidate = at_scale(middle)
        if candidate is None:
            upper = middle
        else:
            lower, fitted = middle, candidate
    return fitted


def compose_pdf(source: bytes, metadata: dict, segments, *, bilingual=False, only_page=None,
                deleted_segments=(), strict=True):
    """Return (PDF bytes, layout report). No images/paths/forms are ever deleted or masked."""
    with _EXPORT_LOCK, _RENDER_LOCK:
        font = _font()
        original = PdfReader(io.BytesIO(source))
        deleted_segments = list(deleted_segments)
        deleted_ids = {segment.id for segment in deleted_segments}
        segments = list(segments) + deleted_segments
        if metadata.get("geometry_version") != GEOMETRY_VERSION:
            metadata = {**metadata, "layout": repair_layout_geometry(source, metadata["layout"])}
        metadata = {**metadata, "layout": enrich_layout_fonts(source, metadata["layout"])}
        metadata = {**metadata, "layout": [
            {**item, "regions": [_native_region(r, original.pages[r["page"] - 1])
                                  for r in item["regions"]]}
            for item in metadata["layout"]
        ]}
        layouts = defaultdict(list)
        by_page = defaultdict(list)
        page_local_regions = defaultdict(list)
        for index, item in enumerate(metadata["layout"]):
            layouts[_text_key(item["source_text"])].append((index, item))
            item_pages = list(dict.fromkeys(region["page"] for region in item["regions"]))
            for region in item["regions"]:
                by_page[region["page"]].append(region)
            if len(item_pages) == 1:
                page_local_regions[item_pages[0]].extend(item["regions"])
        plans = []
        unmatched = []
        failed = []
        consumed = set()
        item_segments = {}
        replaced_segment_ids = set()
        reflowed_pages = set()
        fallback_pages = set()
        fallback_reasons = {}
        required_pages = {only_page} if only_page else set()
        if only_page:
            connections = [{r["page"] for r in item["regions"]}
                           for item in metadata["layout"]]
            connections.extend({int(ref.rsplit(":", 1)[-1]) for ref in segment.source_refs
                                if ref.startswith("pdf:page:")} for segment in segments)
            while True:
                connected = required_pages | set().union(
                    *(numbers for numbers in connections if numbers & required_pages))
                if connected == required_pages:
                    break
                required_pages = connected

        def defer_layout(numbers, exc):
            if strict:
                raise exc
            for number in numbers:
                fallback_pages.add(number)
                fallback_reasons.setdefault(number, str(exc))

        for segment in segments:
            refs = list(
                dict.fromkeys(
                    int(ref.rsplit(":", 1)[-1])
                    for ref in segment.source_refs
                    if ref.startswith("pdf:page:")
                )
            )
            candidates = layouts.get(_text_key(segment.source_text), [])
            match = next(
                (
                    (index, item)
                    for index, item in candidates
                    if index not in consumed
                    and list(dict.fromkeys(r["page"] for r in item["regions"])) == refs
                ),
                None,
            )
            if match:
                consumed.add(match[0])
                item_segments[match[0]] = segment
            target = "" if segment.id in deleted_ids else _translation(segment)
            target_pages = set(refs)
            if not target_pages and metadata["layout"]:
                anchor = (candidates[0][1] if candidates else
                          metadata["layout"][min(segment.ordinal, len(metadata["layout"]) - 1)])
                target_pages = {r["page"] for r in anchor["regions"]}
            if ((not target and segment.id not in deleted_ids)
                    or (only_page and not required_pages & target_pages)):
                continue
            if not match:
                unmatched.append((segment, target, refs))
                continue
            # Fit the complete segment even for a one-page preview. The final page
            # selection happens after typesetting, exactly as in the exported PDF.
            if _LEADER.fullmatch(segment.source_text) and _LEADER.fullmatch(target):
                target = _LeaderText(target)
            plans.append((segment, target, match[1]["regions"]))
        # Source edits or missing references may break the exact mapping. Reserve
        # an unclaimed original block for each such edit so its source and target
        # are never both drawn in the same fallback region.
        fallback_assignments = defaultdict(list)
        fallback_anchors = {}
        for segment, target, refs in unmatched:
            available = [index for index in range(len(metadata["layout"]))
                         if index not in consumed and index not in fallback_assignments]
            choices = available or list(range(len(metadata["layout"])))
            if not choices:
                continue
            def anchor_rank(index, owner=segment, page_refs=frozenset(refs)):
                item = metadata["layout"][index]
                pages = {r["page"] for r in item["regions"]}
                return (_text_key(item["source_text"]) != _text_key(owner.source_text),
                        pages != page_refs, not bool(pages & page_refs),
                        abs(index - owner.ordinal))
            index = min(choices, key=anchor_rank)
            fallback_assignments[index].append((segment, target))
            fallback_anchors[segment.id] = index
        pinned_rows = {segment.id for segment, target, _ in plans if isinstance(target, _LeaderText)}
        # Decide which paragraphs need whitespace before removing any source
        # objects. Short translations can use their source boxes directly.
        expansion_needed = {
            id(region)
            for _, target, regions in plans
            if not _fit(target, regions, font, shrink=False)
            for region in regions
        }
        overlays = defaultdict(list)
        with pdfium.PdfDocument(source) as document:
            pages = {}
            objects = {}
            geometry = {}
            backgrounds = {}
            expanded_regions = {}

            def load_page(number):
                if number in pages:
                    return
                pages[number] = document[number - 1]
                objects[number] = list(pages[number].get_objects(max_depth=1))
                geometry[number] = _text_geometry(pages[number])
                backgrounds[number] = _background_objects(
                    pages[number], objects[number], geometry[number]
                ) | _text_decorations(objects[number], geometry[number])
                # Always measure expansion on the untouched page so preview and
                # export make identical layout decisions.
                for original_region in by_page[number]:
                    original_region["text_color"] = _region_text_color(
                        original_region, objects[number], geometry[number]
                    )
                    if id(original_region) in expansion_needed:
                        expanded_regions[id(original_region)] = _expanded_region(
                            original_region,
                            by_page[number],
                            pages[number],
                            objects[number],
                            geometry[number],
                            backgrounds[number],
                        )

            try:
                resolved_plans = []
                for segment, target, regions in plans:
                    try:
                        if (not _LEADER.fullmatch(segment.source_text)
                                and _COMPOUND_LEADER.fullmatch(segment.source_text)):
                            if not regions:
                                raise ValueError("PDF 复合目录条目缺少原页映射，已停止排版。")
                            for region in regions:
                                load_page(region["page"])
                            resolved_plans.extend(_compound_toc_plans(
                                segment, target, regions, pages[regions[0]["page"]]
                            ))
                        else:
                            resolved_plans.append((segment, target, regions))
                    except ValueError as exc:
                        defer_layout({r["page"] for r in regions}, exc)
                plans = resolved_plans
                pinned_rows = {segment.id for segment, target, _ in plans
                               if isinstance(target, _LeaderText)}
                visible_plans = []
                for segment, target, regions in plans:
                    if any(r["page"] in fallback_pages for r in regions):
                        continue
                    try:
                        for region in regions:
                            number = region["page"]
                            load_page(number)
                        if regions and all(_outside_visible_page(
                            region, pages[region["page"]], objects[region["page"]],
                            geometry[region["page"]]
                        ) for region in regions):
                            continue
                        visible_plans.append((segment, target, regions))
                        if isinstance(target, _LeaderText):
                            source_folio = _LEADER.fullmatch(segment.source_text).group(2).strip()
                            if target.folio != source_folio:
                                raise ValueError(f"PDF 第 {number} 页目录译文的页码与原文不一致。")
                            target, regions = _toc_folio_regions(
                                target, regions, pages[number], objects[number], geometry[number]
                            )
                        fitted = _fit(target, regions, font, shrink=False)
                        initial_fit = bool(fitted)
                        if not fitted:
                            expanded = [expanded_regions.get(id(r), r) for r in regions]
                            fitted = _fit(target, expanded, font)
                        if not fitted:
                            fitted = _fit(target, expanded, font, compact=True)
                        if not fitted:
                            if segment.id in pinned_rows:
                                raise ValueError(f"PDF 第 {number} 页目录条目无法在保留页码及可读字号下安全排版，请缩短目录译文。")
                            failed.append((segment, target, regions))
                            continue
                        if initial_fit and len(fitted) == 1 and segment.id not in pinned_rows:
                            region, lines, size = fitted[0]
                            number = region["page"]
                            region = _raised_baseline(
                                region, lines, size, pages[number], objects[number], geometry[number],
                                backgrounds[number], by_page[number], overlays[number]
                            )
                            fitted = [(region, lines, size)]
                        removals = []
                        safe = True
                        for region, lines, size in fitted:
                            number = region["page"]
                            page = pages[number]
                            native_region = _native_region(region, page)
                            rect = native_region["rect"]
                            draw_rect = native_region.get("draw_rect", rect)
                            selected = []
                            if not _inside(draw_rect, page.get_bbox(), 0.1):
                                safe = False
                                break
                            for obj in objects[number]:
                                bounds = _bounds(obj, geometry[number])
                                if obj.type == raw.FPDF_PAGEOBJ_TEXT:
                                    if _overlaps(bounds, rect) and _belongs(
                                        obj, native_region, geometry[number]
                                    ):
                                        selected.append(obj)
                                    elif _overlaps(bounds, draw_rect):
                                        safe = False
                                        break
                                elif _overlaps(bounds, draw_rect) and _graphic_intersects(
                                    obj, draw_rect, backgrounds[number]
                                ):
                                    safe = False
                                    break
                            if not safe or not selected:
                                safe = False
                                break
                            removals.append((number, selected))
                        if not safe:
                            if segment.id in pinned_rows:
                                raise ValueError(f"PDF 第 {number} 页目录文字与页码或遮挡对象重叠，无法安全排版。")
                            failed.append((segment, target, regions))
                            continue
                        for number, selected in removals:
                            for obj in selected:
                                pages[number].remove_obj(obj)
                                objects[number].remove(obj)
                                obj.close()
                        for region, lines, size in fitted:
                            overlays[region["page"]].append((segment.id, region, lines, size))
                        replaced_segment_ids.add(segment.id)
                    except ValueError as exc:
                        defer_layout({r["page"] for r in regions}, exc)
                plans = visible_plans
                # A paragraph may still fail because a translated citation no longer
                # fits the single source line. Reflow related text inside the same
                # page/column instead of retaining English or adding continuation pages.
                failed_by_page = defaultdict(list)
                cross_page_failed = []
                for item in failed:
                    item_pages = list(dict.fromkeys(r["page"] for r in item[2]))
                    if len(item_pages) == 1:
                        failed_by_page[item_pages[0]].append(item)
                    else:
                        cross_page_failed.append(item)

                def remove_group_text(number, regions):
                    load_page(number)
                    native_regions = [_native_region(region, pages[number]) for region in regions]
                    lines = [line for region in native_regions for line in region["lines"]]
                    tolerance = max(
                        2.0, max(float(region["font_size"]) for region in regions) * 0.3
                    )
                    selected = []
                    for obj in objects[number]:
                        if obj.type != raw.FPDF_PAGEOBJ_TEXT:
                            continue
                        boxes = geometry[number].get(_object_key(obj), [obj.get_bounds()])
                        if all(
                            any(_inside(box, line, tolerance) for line in lines) for box in boxes
                        ):
                            selected.append(obj)
                    if not selected:
                        raise ValueError(
                            f"PDF 第 {number} 页原文对象无法定位，已停止导出以避免漏译。"
                        )
                    for obj in selected:
                        pages[number].remove_obj(obj)
                        objects[number].remove(obj)
                        obj.close()

                def group_splits_text_object(number, regions):
                    """Detect layout regions that cover only part of one PDF text object."""
                    load_page(number)
                    native_regions = [_native_region(region, pages[number]) for region in regions]
                    lines = [line for region in native_regions for line in region["lines"]]
                    tolerance = max(
                        2.0, max(float(region["font_size"]) for region in regions) * 0.3
                    )
                    for obj in objects[number]:
                        if obj.type != raw.FPDF_PAGEOBJ_TEXT:
                            continue
                        boxes = geometry[number].get(_object_key(obj), [obj.get_bounds()])
                        contained = [
                            any(_inside(box, line, tolerance) for line in lines) for box in boxes
                        ]
                        if any(contained) and not all(contained):
                            return True
                    return False

                # Multi-page paragraphs retain their own page regions. An extremely
                # long target must still satisfy the same readability floor.
                for segment, target, regions in cross_page_failed:
                    if any(r["page"] in fallback_pages for r in regions):
                        continue
                    try:
                        fitted = _fit(target, regions, font, compact=True)
                        if not fitted:
                            raise ValueError(
                                f"第 {segment.ordinal + 1} 段无法在可读字号下完成原页排版，已停止导出。"
                            )
                        for number in dict.fromkeys(r["page"] for r in regions):
                            remove_group_text(number, [r for r in regions if r["page"] == number])
                        for region, lines, size in fitted:
                            overlays[region["page"]].append((segment.id, region, lines, size))
                            reflowed_pages.add(region["page"])
                        replaced_segment_ids.add(segment.id)
                    except ValueError as exc:
                        defer_layout({r["page"] for r in regions}, exc)

                full_reflow_pages = set()
                for segment, _, refs in unmatched:
                    if len(refs) != 1:
                        anchor = metadata["layout"][fallback_anchors[segment.id]]
                        numbers = {r["page"] for r in anchor["regions"]}
                        defer_layout(numbers, ValueError(
                            f"第 {segment.ordinal + 1} 段缺少唯一原页映射。"
                        ))
                        continue
                    full_reflow_pages.add(refs[0])

                def region_bounds(regions):
                    return [min(r["rect"][0] for r in regions),
                            min(r["rect"][1] for r in regions),
                            max(r["rect"][2] for r in regions),
                            max(r["rect"][3] for r in regions)]

                def same_column(a, b):
                    left, right = region_bounds(a), region_bounds(b)
                    size = min(float(r["font_size"]) for r in a + b)
                    overlap = min(left[2], right[2]) - max(left[0], right[0])
                    width = min(left[2] - left[0], right[2] - right[0])
                    gap = max(0, left[1] - right[3], right[1] - left[3])
                    return (abs(left[0] - right[0]) <= size * 2
                            and overlap >= width * .7 and gap <= size * 4)

                def adjacent(a, b):
                    if not same_column(a, b):
                        return False
                    left, right = region_bounds(a), region_bounds(b)
                    gap = [max(left[0], right[0]), min(left[3], right[3]),
                           min(left[2], right[2]), max(left[1], right[1])]
                    if gap[3] <= gap[1]:
                        return True
                    own_ids = {id(r) for r in a + b}
                    return not any(id(other) not in own_ids and _overlaps(other["rect"], gap)
                                   for other in by_page[number])

                def page_segments(number):
                    return [segment for segment in segments if [
                        int(ref.rsplit(":", 1)[-1]) for ref in segment.source_refs
                        if ref.startswith("pdf:page:")
                    ] == [number]]

                pages_to_reflow = sorted(set(failed_by_page) | full_reflow_pages)
                for number in pages_to_reflow:
                    if number in fallback_pages:
                        continue
                    try:
                        load_page(number)
                        local_plans = [plan for plan in plans
                                       if list(dict.fromkeys(r["page"] for r in plan[2])) == [number]]
                        if number in full_reflow_pages:
                            # Edited source may lose its original segment mapping. Only
                            # a single continuous column with uniform size is unambiguous.
                            regions = list(page_local_regions[number])
                            sizes = [float(r["font_size"]) for r in regions]
                            if (not regions or max(sizes) > min(sizes) * 1.12
                                    or any(not same_column([a], [b])
                                           for a, b in pairwise(regions))):
                                raise ValueError(
                                    f"PDF 第 {number} 页段落映射不明确，无法安全重排多栏或标题。"
                                )
                            groups = [(page_segments(number), regions, False)]
                        else:
                            # Grow only continuous same-kind blocks in one column.
                            # Similar font size alone never proves shared reading order.
                            groups, assigned = [], set()
                            container_owners = {
                                plan[0].id: frozenset(
                                    tuple(bounds)
                                    for region in plan[2]
                                    for bounds in _container_bounds(
                                        region, pages[number], objects[number], geometry[number],
                                        backgrounds[number]
                                    )
                                ) for plan in local_plans
                            }
                            for seed in failed_by_page[number]:
                                if seed[0].id in assigned:
                                    continue
                                selected = [seed]
                                base = min(float(r["font_size"]) for r in seed[2])
                                changed = True
                                while changed and seed[0].kind != "heading":
                                    changed = False
                                    for plan in local_plans:
                                        if (plan[0].id in assigned or plan[0].id in pinned_rows
                                                or any(plan[0].id == item[0].id
                                                       for item in selected)):
                                            continue
                                        if (plan[0].kind == seed[0].kind
                                                and container_owners[plan[0].id]
                                                == container_owners[seed[0].id]
                                                and abs(min(float(r["font_size"]) for r in plan[2])
                                                        - base) <= base * .12
                                                and any(adjacent(plan[2], item[2])
                                                        for item in selected)):
                                            selected.append(plan)
                                            changed = True
                                selected.sort(key=lambda plan: plan[0].ordinal)
                                group_segments = [plan[0] for plan in selected]
                                group_regions = [r for plan in selected for r in plan[2]]
                                inline = False
                                if group_splits_text_object(number, group_regions):
                                    # A rotated/native source line can be split into word
                                    # segments by extraction. Join only its exact object
                                    # owners; never expand this to a page or another column.
                                    native_lines = [line for r in group_regions for line in r["lines"]]
                                    ownership = []
                                    for obj in objects[number]:
                                        if obj.type != raw.FPDF_PAGEOBJ_TEXT:
                                            continue
                                        boxes = geometry[number].get(_object_key(obj), [])
                                        inside = [any(_inside(box, line, base * .3)
                                                      for line in native_lines) for box in boxes]
                                        if not any(inside) or all(inside):
                                            continue
                                        bounds = _bounds(obj, geometry[number])
                                        ordered = sorted(boxes, key=lambda box: box[0])
                                        if (bounds[3] - bounds[1] > base * 1.5
                                                or any(b[0] - a[2] > base * 3
                                                       for a, b in pairwise(ordered))):
                                            raise ValueError(
                                                f"PDF 第 {number} 页文字对象跨越段落或栏区，无法安全排版。"
                                            )
                                        ownership.extend(plan for plan in local_plans
                                                         if any(_inside(box, line, base * .3)
                                                                for r in plan[2] for line in r["lines"]
                                                                for box in boxes))
                                    for plan in ownership:
                                        if not any(plan[0].id == item[0].id for item in selected):
                                            selected.append(plan)
                                    selected.sort(key=lambda plan: plan[0].ordinal)
                                    group_segments = [plan[0] for plan in selected]
                                    group_regions = [r for plan in selected for r in plan[2]]
                                    if group_splits_text_object(number, group_regions):
                                        raise ValueError(f"PDF 第 {number} 页文字对象归属不完整，无法安全排版。")
                                    union = region_bounds(group_regions)
                                    if union[3] - union[1] > base * 1.5:
                                        raise ValueError(f"PDF 第 {number} 页文字对象跨越段落，无法安全排版。")
                                    inline = True
                                assigned.update(segment.id for segment in group_segments)
                                groups.append((group_segments, group_regions, inline))
                        for group_segments, group_regions, inline in groups:
                            if any(segment.id in pinned_rows for segment in group_segments):
                                raise ValueError(f"PDF 第 {number} 页重排会移动目录页码，已停止排版。")
                            if not group_regions or not group_segments:
                                raise ValueError(f"PDF 第 {number} 页缺少可重排文字区。")
                            group_segments.sort(key=lambda segment: segment.ordinal)
                            flow_size = min(float(r["font_size"]) for r in group_regions)
                            styled, word_breaks = [], []
                            previous_bounds = None
                            for segment in group_segments:
                                if segment.id in deleted_ids:
                                    continue
                                matching = next((plan[2] for plan in local_plans
                                                 if plan[0].id == segment.id), group_regions)
                                face = _region_font(matching[0])
                                text = _translation(segment) or segment.source_text
                                current_bounds = region_bounds(matching)
                                if previous_bounds is not None:
                                    word_breaks.append(current_bounds[0] - previous_bounds[2]
                                                       > flow_size * .12)
                                previous_bounds = current_bounds
                                color = matching[0].get("text_color", (.08, .08, .08))
                                styled.append(_StyledText(text, [face] * len(text), [color] * len(text)))
                            target = (_inline_text(styled, word_breaks)
                                      if inline else _join_lines(styled))
                            rect = region_bounds(group_regions)
                            if any(not _inside(rect, container, .1)
                                   for region in group_regions
                                   for container in _container_bounds(
                                       region, pages[number], objects[number], geometry[number],
                                       backgrounds[number]
                                   )):
                                raise ValueError(
                                    f"PDF 第 {number} 页重排会跨越文字容器边界，已停止排版。"
                                )
                            flow_region = {
                                "page": number,
                                "font_size": min(float(r["font_size"]) for r in group_regions),
                                "rect": rect,
                                "lines": [line for r in group_regions for line in r["lines"]],
                                "force_left": True,
                                "native_space": all(r.get("native_space") for r in group_regions),
                            }
                            group_ids = {id(region) for region in group_regions}
                            native_flow_rect = _native_region(flow_region, pages[number])["rect"]
                            graphic_blocked = any(
                                obj.type != raw.FPDF_PAGEOBJ_TEXT
                                and _overlaps(_bounds(obj, geometry[number]), native_flow_rect)
                                and _graphic_intersects(obj, native_flow_rect, backgrounds[number])
                                for obj in objects[number]
                            )
                            blocked = graphic_blocked or any(
                                id(other) not in group_ids and _overlaps(other["rect"], rect)
                                for other in by_page[number]
                            )
                            fitted = _fit(target, group_regions if blocked else [flow_region],
                                          font, compact=True)
                            if not fitted:
                                if graphic_blocked:
                                    raise ValueError(
                                        f"PDF 第 {number} 页段落跨越表格线、图形或遮挡，无法安全重排，请调整段落结构。"
                                    )
                                raise ValueError(
                                    f"PDF 第 {number} 页译文无法在可读字号下完成排版，"
                                    "请缩短译文或调整段落。"
                                )
                            remove_group_text(number, group_regions)
                            segment_ids = {segment.id for segment in group_segments}
                            overlays[number] = [draw for draw in overlays[number]
                                                if draw[0] not in segment_ids]
                            for region, lines, size in fitted:
                                overlays[number].append((f"page-flow:{number}", region, lines, size))
                            replaced_segment_ids.update(segment.id for segment in group_segments
                                                        if _translation(segment))
                            reflowed_pages.add(number)
                    except ValueError as exc:
                        defer_layout([number], exc)

                for number, draws in overlays.items():
                    if number in fallback_pages:
                        continue
                    try:
                        for _, region, _, _ in draws:
                            native = _native_region(region, pages[number])
                            rect = native.get("draw_rect", native["rect"])
                            if any(obj.type != raw.FPDF_PAGEOBJ_TEXT
                                   and _overlaps(_bounds(obj, geometry[number]), rect)
                                   and _graphic_intersects(obj, rect, backgrounds[number])
                                   for obj in objects[number]):
                                raise ValueError(
                                    f"PDF 第 {number} 页译文区域与图形或遮挡重叠，已停止排版。"
                                )
                            if any(obj.type == raw.FPDF_PAGEOBJ_TEXT
                                   and geometry[number].get(_object_key(obj))
                                   and _overlaps(_bounds(obj, geometry[number]), rect)
                                   for obj in objects[number]):
                                raise ValueError(
                                    f"PDF 第 {number} 页译文区域仍有原文对象，已停止输出以避免叠字。"
                                )
                        pages[number].gen_content()
                    except ValueError as exc:
                        defer_layout([number], exc)
                if fallback_pages:
                    # A source paragraph spanning pages must use the same split on
                    # every page, including when only one side needed direct fitting.
                    changed = True
                    while changed:
                        changed = False
                        for item in metadata["layout"]:
                            numbers = {r["page"] for r in item["regions"]}
                            if numbers & fallback_pages and not numbers <= fallback_pages:
                                fallback_pages.update(numbers)
                                changed = True
                    for number in sorted(fallback_pages):
                        load_page(number)
                        overlays[number] = []
                    direct_items = []
                    for index, item in enumerate(metadata["layout"]):
                        regions = item["regions"]
                        numbers = {r["page"] for r in regions}
                        if not numbers & fallback_pages:
                            continue
                        segment = item_segments.get(index)
                        assigned = fallback_assignments.get(index, [])
                        if assigned:
                            parts = ([_translation(segment) or segment.source_text]
                                     if segment and segment.id not in deleted_ids else [])
                            parts.extend(target for owner, target in assigned
                                         if owner.id not in deleted_ids)
                            text = "\n".join(parts)
                            replaced_segment_ids.update(owner.id for owner, target in assigned
                                                        if target)
                        else:
                            text = ("" if segment and segment.id in deleted_ids else
                                    (_translation(segment) or segment.source_text) if segment else
                                    item["source_text"])
                        direct_items.append((segment, text, regions))
                    for segment, text, regions in direct_items:
                        visible = []
                        for region in regions:
                            number = region["page"]
                            load_page(number)
                            box = pages[number].get_bbox()
                            rect = region["rect"]
                            if not _overlaps(rect, box):
                                continue
                            visible.append({**region, "rect": [max(rect[0], box[0]),
                                            max(rect[1], box[1]), min(rect[2], box[2]),
                                            min(rect[3], box[3])], "native_space": True})
                        if not visible:
                            continue
                        for region, lines, size in _fit_direct(text, visible, font):
                            number = region["page"]
                            if number in fallback_pages:
                                overlays[number].append(("direct", region, lines, size))
                        if segment and _translation(segment):
                            replaced_segment_ids.add(segment.id)
                    for number in sorted(fallback_pages):
                        rects = [r["rect"] for r in by_page[number]]
                        for obj in list(objects[number]):
                            if (obj.type == raw.FPDF_PAGEOBJ_TEXT
                                    and any(_overlaps(_bounds(obj, geometry[number]), rect)
                                            for rect in rects)):
                                pages[number].remove_obj(obj)
                                objects[number].remove(obj)
                                obj.close()
                        pages[number].gen_content()
                        reflowed_pages.add(number)
                edited_stream = io.BytesIO()
                if overlays:
                    document.save(edited_stream)
                    edited = PdfReader(edited_stream)
                else:
                    edited = original
            finally:
                for page in pages.values():
                    page.close()
        # Preserve the original catalog/bookmarks. Only replace content of modified pages.
        writer = PdfWriter()
        writer.clone_document_from_reader(original)
        # One overlay document shares its embedded font subsets across pages.
        # Separate canvases used to embed the same Chinese face once per page.
        overlay_stream = io.BytesIO()
        canvas = Canvas(overlay_stream) if overlays else None
        for number, draws in overlays.items():
            page = edited.pages[number - 1]
            box = page.mediabox
            canvas.setPageSize((max(float(box.right), 1), max(float(box.top), 1)))
            for _, region, lines, size in draws:
                canvas.saveState()
                if not region.get("native_space"):
                    rotation = page.rotation % 360
                    if rotation == 90:
                        canvas.transform(0, 1, -1, 0, float(box.right) + float(box.left), 0)
                    elif rotation == 180:
                        canvas.transform(
                            -1,
                            0,
                            0,
                            -1,
                            float(box.right) + float(box.left),
                            float(box.top) + float(box.bottom),
                        )
                    elif rotation == 270:
                        canvas.transform(0, -1, 1, 0, 0, float(box.top) + float(box.bottom))
                _paint_lines(canvas, region, lines, size, tuple(float(v) for v in box))
                canvas.restoreState()
            canvas.showPage()
        if overlays:
            canvas.save()
            overlay_reader = PdfReader(overlay_stream)
        for overlay_index, number in enumerate(overlays):
            edited_page = edited.pages[number - 1]
            page = writer.pages[number - 1]
            # Replacing only page content/resources keeps links, geometry and outline targets.
            for key in ("/Contents", "/Resources"):
                if key not in edited_page:
                    page.pop(NameObject(key), None)
                    continue
                value = edited_page.raw_get(key).clone(writer)
                # PDF streams must be indirect objects; a direct stream silently blanks
                # the page in PDFium/Poppler even when text extraction still succeeds.
                if isinstance(value, StreamObject):
                    value = writer._add_object(value)
                page[NameObject(key)] = value
            page.merge_page(overlay_reader.pages[overlay_index])
        report = {
            "layout_version": PDF_LAYOUT_VERSION,
            "min_font_size": min((size for draws in overlays.values()
                                  for _, _, lines, size in draws if lines), default=None),
            "replaced_blocks": len(replaced_segment_ids),
            "overflow_count": 0,
            "overflow": [],
            "reflowed_pages": sorted(reflowed_pages),
            "direct_pages": sorted(fallback_pages),
            "layout_notes": fallback_reasons,
        }
        if only_page:
            single = PdfWriter()
            single.add_page(writer.pages[only_page - 1])
            writer = single
        elif bilingual:
            _bilingual_spreads(writer, original)
        output = io.BytesIO()
        writer.write(output)
        return output.getvalue(), report


def export_translated_pdf(store, document_id, *, bilingual=False):
    metadata = ensure_pdf_layout(store, document_id)
    return compose_pdf(
        store.get_original_pdf(document_id),
        metadata,
        store.list_segments(document_id),
        bilingual=bilingual,
        deleted_segments=store.list_deleted_source_segments(document_id),
        strict=False,
    )[0]
