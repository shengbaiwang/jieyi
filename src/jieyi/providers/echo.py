from __future__ import annotations

import json

from jieyi.domain.models import ModelSpec, TranslationRequest, TranslationResult


class EchoProvider:
    """Deterministic dry-run provider used for local workflow verification."""

    @staticmethod
    def _prefix_inside_structure(source: str, prefix: str) -> str:
        if source.startswith("[[JY_PH_"):
            end = source.find("]]")
            if end >= 0:
                return source[: end + 2] + prefix + source[end + 2 :]
        return prefix + source

    async def translate(self, request: TranslationRequest, model: ModelSpec) -> TranslationResult:
        if request.existing_translation:
            text = request.existing_translation
        else:
            text = self._prefix_inside_structure(
                request.segment.source_text,
                f"【{request.project.target_lang}】",
            )
        return TranslationResult(text=text)

    async def complete(
        self,
        messages: list[dict[str, str]],
        model: ModelSpec,
        *,
        thinking: bool | None = None,
        reasoning_effort: str | None = None,
        compute_mode: str | None = None,
        max_tokens: int | None = None,
    ) -> TranslationResult:
        del model, thinking, reasoning_effort, compute_mode, max_tokens
        content = messages[-1]["content"]
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict) and isinstance(payload.get("segments"), list):
            translations = {}
            for item in payload["segments"]:
                translations[item["id"]] = item.get("current_translation") or (
                    "translated:" + item["source"]
                )
            text = json.dumps(translations, ensure_ascii=False)
        else:
            source = content
            if "\n\nSOURCE:\n" in source:
                source = source.rsplit("\n\nSOURCE:\n", 1)[1]
            elif source.startswith("SOURCE:\n"):
                source = source[len("SOURCE:\n") :]
            current = ""
            if "\n\nCURRENT TRANSLATION:\n" in source:
                source, remainder = source.split("\n\nCURRENT TRANSLATION:\n", 1)
                current = remainder.split("\n\nKNOWN ISSUES:\n", 1)[0]
            text = current or self._prefix_inside_structure(source, "translated:")
        return TranslationResult(
            text=text,
            prompt_tokens=max(1, sum(len(message["content"]) for message in messages) // 4),
            completion_tokens=max(1, len(text) // 4),
        )
