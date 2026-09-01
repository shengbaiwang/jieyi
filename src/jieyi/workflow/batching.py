from __future__ import annotations

import json
import re

from jieyi.domain.models import CandidateStage, TranslationRequest

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def build_batch_messages(requests: list[TranslationRequest]) -> list[dict[str, str]]:
    if not requests:
        raise ValueError("batch requests cannot be empty")
    first = requests[0]
    stage = first.task
    if stage is CandidateStage.REVIEW:
        instruction = (
            "Review each existing translation for fidelity, terminology, tone, citations and "
            "format. Revise only where needed. If current_translation is empty because the draft "
            "model refused the source, translate that source faithfully from scratch."
        )
    else:
        instruction = (
            "Translate every source faithfully into the target language. Preserve meaning, "
            "modality, attribution, paragraph boundaries and approved terminology."
        )
    system = (
        f"{instruction}\n"
        "Every placeholder shaped like [[JY_PH_0000]] must appear exactly once, unchanged. "
        "They may encode inline formatting and SourceAtom boundaries; keep translated text inside "
        "the same protected boundary pairs and never merge DOM fragments.\n"
        "Return one valid JSON object only. Its keys must be the supplied segment IDs and each "
        "value must be only that segment's translation. Do not add markdown or commentary.\n\n"
        f"{first.context}"
    )
    items = []
    for request in requests:
        item = {
            "id": request.segment.id,
            "heading": request.segment.heading_path,
            "source": request.segment.source_text,
        }
        if stage is CandidateStage.REVIEW:
            item["current_translation"] = request.existing_translation or ""
            item["known_issues"] = (
                request.issue_summary or "Independent sampled fidelity review."
            )
        items.append(item)
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": json.dumps(
                {"segments": items},
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        },
    ]


def parse_batch_translations(text: str, expected_ids: set[str]) -> dict[str, str]:
    cleaned = _FENCE.sub("", text.strip())
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError("batch provider did not return valid JSON") from exc

    if isinstance(value, dict) and isinstance(value.get("translations"), dict):
        value = value["translations"]
    if isinstance(value, list):
        value = {
            str(item.get("id", "")): item.get("text", "")
            for item in value
            if isinstance(item, dict)
        }
    if not isinstance(value, dict):
        raise TypeError("batch provider returned an unsupported JSON shape")

    translations = {
        str(key): str(item).strip()
        for key, item in value.items()
        if str(key) in expected_ids and isinstance(item, str) and item.strip()
    }
    missing = expected_ids - translations.keys()
    extra = translations.keys() - expected_ids
    if missing or extra:
        raise ValueError(
            f"batch response IDs mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
        )
    return translations
