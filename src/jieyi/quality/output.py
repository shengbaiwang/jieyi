"""Conservative detection of model-added response framing, never destructive cleanup."""
from __future__ import annotations

import json
import re

_MARKER = re.compile(r"\[\[JY_PH_[^]]+\]\]")
_LABEL = re.compile(
    r"(?im)(?:^\s*(?:[>*#-]\s*)*(?:\*\*)?|\*\*)"
    r"(翻译|译文|中文翻译|说明|解释|注释|Translation|Explanation|Notes?)"
    r"(?:\*\*)?\s*[:：]"
)
_CJK = re.compile(r"[\u3400-\u9fff]")


def output_problems(source: str, target: str) -> tuple[str, ...]:
    source = _MARKER.sub("", source).strip()
    target = _MARKER.sub("", target).strip()
    # Structured repair values must satisfy the same content contract as plain text.
    try:
        fragments = json.loads(target)
    except (ValueError, TypeError):
        fragments = None
    if isinstance(fragments, dict) and all(isinstance(v, str) for v in fragments.values()):
        target = "\n".join(fragments.values())
    problems = []
    source_labels = {m.group(1).casefold() for m in _LABEL.finditer(source)}
    added_labels = {m.group(1).casefold() for m in _LABEL.finditer(target)} - source_labels
    # Single labels can be legitimate translated headings. Require combined framing
    # or Markdown response formatting to avoid treating book prose as instructions.
    if len(added_labels) >= 2 or (added_labels and "**" in target and "**" not in source):
        problems.append("model_commentary")
    plain_source = re.sub(r"\s+", " ", source).strip()
    plain_target = re.sub(r"\s+", " ", target).strip()
    if (len(re.findall(r"[A-Za-z]+", plain_source)) >= 6
            and not _CJK.search(source) and len(_CJK.findall(target)) >= 6
            and plain_target.startswith(plain_source)):
        problems.append("source_echo")
    return tuple(problems)


OUTPUT_RETRY_RULE = (
    "The previous response violated the translation-only output contract. "
    "Translate the SOURCE again. Return only its translation; no repeated source, "
    "translation labels, explanations, analysis, or added Markdown. "
    "Preserve all required placeholders and the required fragment JSON schema."
)


class TranslationOutputError(ValueError):
    def __init__(self, problems, results):
        self.problems = tuple(problems)
        self.results = list(results)
        super().__init__("译文混入原文复述或解释，自动重试后仍未通过：" + ", ".join(problems))
