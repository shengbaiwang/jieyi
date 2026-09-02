from __future__ import annotations

from jieyi.domain.models import CandidateStage, TranslationRequest
from jieyi.protection import extract_placeholder_tokens


def build_system_prompt(request: TranslationRequest) -> str:
    placeholder_tokens = extract_placeholder_tokens(request.segment.source_text)
    if placeholder_tokens:
        invariant_rule = (
            "Copy these source placeholder tokens exactly once, unchanged, and in this order: "
            + ", ".join(placeholder_tokens)
            + ". Do not create any other placeholder tokens."
        )
    else:
        invariant_rule = (
            "This source contains no placeholder tokens. Do not introduce any JY_PH marker "
            "or double-bracket placeholder."
        )
    if request.task is CandidateStage.REVIEW:
        if request.existing_translation:
            instruction = (
                "Independently compare the source and current translation sentence by sentence; "
                "do not assume the draft is correct. Check for omissions, additions, mistranslations, "
                "distorted modality or attribution, terminology errors, and unnatural target-language "
                "wording. Revise only where justified, preserve genuine source ambiguity, and actively "
                "correct every clear issue listed below. Return the complete revised translation first. "
                "Only when a material ambiguity or factual decision cannot be resolved from the source "
                "and supplied context, append a blank line, the exact line JY_REVIEW_ISSUES:, and one "
                "concise bullet per question for a human. Omit this appendix when no human decision is needed."
            )
        else:
            instruction = (
                "No draft translation is available because the earlier draft was refused or did "
                "not pass structural validation. Translate the source faithfully now. "
                "Preserve meaning, modality, attribution and every mandatory terminology "
                "constraint. Return only the translation."
            )
    elif request.task is CandidateStage.REPAIR:
        instruction = (
            "Repair placeholder integrity in the current translation. Keep its translated wording "
            "and use the masked source only to recover the exact placeholder positions. Do not "
            "retranslate, summarize, or return a plain-text version. Change only placeholder tokens "
            "unless a minimal surrounding whitespace change is required. Before responding, verify "
            "that every required placeholder appears exactly once and in the listed order. Return "
            "only the complete repaired translation, including all placeholders."
        )
    else:
        instruction = (
            "Translate the source faithfully. Do not add commentary or omit content. Preserve "
            "meaning, modality, attribution and every mandatory terminology constraint. "
            "Treat approved terminology as binding, not advisory. Return only the translation."
        )
    structure_rule = (
        "Placeholder pairs may represent protected inline formatting, links, notes, line breaks, "
        "and original SourceAtom boundaries. Keep translated text inside the corresponding pairs; "
        "do not merge, reorder, or move text across DOM fragments."
    )
    return f"{instruction}\n{invariant_rule}\n{structure_rule}\n\n{request.context}"


def build_user_prompt(request: TranslationRequest) -> str:
    segment_context = request.segment_context.strip()
    if request.task is CandidateStage.REVIEW:
        prompt = (
            f"SOURCE:\n{request.segment.source_text}\n\n"
            f"CURRENT TRANSLATION:\n{request.existing_translation or ''}\n\n"
            f"KNOWN ISSUES:\n{request.issue_summary or 'Perform an independent fidelity review.'}"
        )
    elif request.task is CandidateStage.REPAIR:
        prompt = (
            f"MASKED SOURCE:\n{request.segment.source_text}\n\n"
            f"BROKEN TRANSLATION:\n{request.existing_translation or ''}\n\n"
            f"PLACEHOLDER ERROR:\n{request.issue_summary}"
        )
    elif segment_context:
        prompt = f"SOURCE:\n{request.segment.source_text}"
    else:
        prompt = request.segment.source_text
    if segment_context:
        return f"{segment_context}\n\n{prompt}"
    return prompt


def build_messages(request: TranslationRequest) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": build_system_prompt(request)},
        {"role": "user", "content": build_user_prompt(request)},
    ]
