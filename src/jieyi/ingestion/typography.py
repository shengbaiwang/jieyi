"""Map source type families to a restrained Chinese editorial palette."""

from __future__ import annotations

import re
from dataclasses import dataclass

TYPOGRAPHY_VERSION = "editorial-fonts-v1"

CJK_FAMILIES = {
    "serif": '"Songti SC", "Noto Serif CJK SC", "Source Han Serif SC", SimSun, serif',
    "sans": '"PingFang SC", "Noto Sans CJK SC", "Source Han Sans SC", "Microsoft YaHei", sans-serif',
    "kai": '"Jieyi Kai", "Kaiti SC", STKaiti, KaiTi, "LXGW WenKai", "楷体", "Noto Serif CJK SC", serif',
    "mono": '"SFMono-Regular", Consolas, "Noto Sans Mono CJK SC", monospace',
}


def font_family_role(family: str, *, default: str = "serif") -> str:
    """Use the first recognizable family, respecting CSS fallback order."""
    for name in family.casefold().split(","):
        if re.search(r"mono|courier|consolas|menlo|typewriter|等宽", name):
            return "mono"
        if re.search(r"kaiti|stkaiti|kai\s*ti|楷", name):
            return "kai"
        if re.search(
            r"sans|arial|helvetica|gothic|heiti|pingfang|yahei|system-ui|"
            r"calibri|aptos|avenir|roboto|verdana|tahoma|futura|frutiger|univers|"
            r"trebuchet|gotham|黑|苹方",
            name,
        ):
            return "sans"
        if re.search(
            r"serif|times|georgia|garamond|baskerville|cambria|palatino|minion|caslon|"
            r"constantia|didot|bodoni|charter|song|simsun|ming|宋|明朝",
            name,
        ):
            return "serif"
    return default


@dataclass(frozen=True)
class FontStyle:
    family: str = "serif"
    bold: bool = False
    italic: bool = False

    @property
    def chinese_family(self) -> str:
        # Chinese emphasis uses a different ductus rather than synthetic slanting.
        return "kai" if self.italic and self.family != "mono" else self.family


def pdf_font_style(font_name: str) -> FontStyle:
    return FontStyle(
        family=font_family_role(font_name),
        bold=bool(re.search(r"bold|black|heavy|semibold|demi", font_name, re.IGNORECASE)),
        italic=bool(re.search(r"italic|oblique|slanted", font_name, re.IGNORECASE)),
    )


# An italic local-face alias gives Chinese emphasis upright Kai glyphs while
# leaving Latin italics intact. Without Kai, the browser retains visible italic
# emphasis in its fallback instead of silently rendering ordinary body text.
TRANSLATION_TYPOGRAPHY_CSS = """
@font-face {
  font-family: "Jieyi Kai";
  src: local("Kaiti SC"), local("STKaiti"), local("KaiTi"), local("LXGW WenKai");
  font-style: italic;
  font-weight: 400;
  unicode-range: U+3000-303F, U+3100-312F, U+3400-4DBF, U+4E00-9FFF,
    U+F900-FAFF, U+FF00-FFEF, U+20000-2FFFF;
}
"""
