from __future__ import annotations

import json

from jieyi.domain.models import CandidateStage, TranslationRequest
from jieyi.protection import extract_placeholder_tokens


def build_system_prompt(request: TranslationRequest) -> str:
    placeholder_tokens = extract_placeholder_tokens(request.segment.source_text)
    if request.task is CandidateStage.REPAIR and request.atom_boundaries:
        boundary_tokens = {token for pair in request.atom_boundaries for token in pair}
        inline_tokens = [token for token in placeholder_tokens if token not in boundary_tokens]
        invariant_rule = (
            "Return a JSON object with exactly these keys: "
            + ", ".join(opening for opening, _ in request.atom_boundaries)
            + ". Each value must contain only that fragment's translated text. "
            "Do not put atom boundary tokens in values; the program restores them. "
            "Preserve each fragment's inline placeholders exactly once and in source order: "
            + (", ".join(inline_tokens) or "none") + "."
        )
    elif placeholder_tokens:
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
    if request.task is CandidateStage.REPAIR and request.atom_boundaries:
        instruction = (
            "Repair the EPUB translation by returning one translated value per source fragment. "
            "Read all fragments together as one sentence, then align each value to its source. "
            "You may minimally rephrase the draft to fit target-language grammar and the original "
            "fragment boundaries. Preserve every source claim, attribution, name and title; never "
            "omit a trailing clause merely to make the structure valid. Keep a book-title fragment "
            "limited to the book title and express surrounding clauses in their source fragment. "
            "Return only the complete JSON object, with no commentary or XML tags."
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
            "Follow mandatory terminology and conditional terminology when its sense applies. "
            "Reference terminology is optional guidance: adapt it to context without requiring "
            "identical wording or consistency. Return only the translation."
        )
    structure_rule = (
        "Placeholder pairs may represent protected inline formatting, links, notes, line breaks, "
        "and original SourceAtom boundaries. Keep translated text inside the corresponding pairs; "
        "do not merge, reorder, or move text across DOM fragments."
    )
    if request.task is CandidateStage.REPAIR and request.atom_boundaries:
        structure_rule = "The program owns all SourceAtom wrappers. Follow the supplied JSON fragment schema."
    elif request.atom_boundaries:
        structure_rule += "\nSourceAtom boundary pairs (open → close): " + "; ".join(
            f"{opening} → {closing}" for opening, closing in request.atom_boundaries
        ) + ". No translated text may appear before, between, or after these pairs."
    return f"{instruction}\n{invariant_rule}\n{structure_rule}\n\n{request.context}"


def build_user_prompt(request: TranslationRequest) -> str:
    segment_context = request.segment_context.strip()
    if request.task is CandidateStage.REPAIR and request.atom_boundaries:
        fragments = {
            opening: request.segment.source_text.split(opening, 1)[1].split(closing, 1)[0]
            for opening, closing in request.atom_boundaries
        }
        prompt = (
            "SOURCE FRAGMENTS (same keys required in the response):\n"
            + json.dumps(fragments, ensure_ascii=False)
            + f"\n\nBROKEN TRANSLATION:\n{request.existing_translation or ''}"
            + f"\n\nSTRUCTURE ERROR:\n{request.issue_summary}"
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
