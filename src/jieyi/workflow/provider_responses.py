from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from jieyi.domain.models import TranslationResult

_FILTER_REASONS = {
    "blocked",
    "content_filter",
    "content_filtered",
    "content_policy",
    "refusal",
    "safety",
}
_BUDGET_REASONS = {
    "length",
    "max_output_tokens",
    "max_tokens",
    "token_limit",
}

_CONTENT_FILTER_MARKERS = (
    "code 1301",
    "代码 1301",
    "content_filter",
    "content filtered",
    "content policy",
    "content safety",
    "safety policy",
    "unsafe or sensitive content",
    "不安全或敏感内容",
    "安全或敏感内容",
)


@dataclass(frozen=True, slots=True)
class EmptyResponseAttempt:
    attempt: int
    max_tokens: int
    kind: str
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    finish_reason: str
    response_id: str
    message_keys: tuple[str, ...]
    refusal: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def inspect_empty_result(
    result: TranslationResult,
    *,
    attempt: int,
    max_tokens: int,
) -> EmptyResponseAttempt:
    """Extract a small diagnostic envelope without persisting full model output."""
    payload: dict[str, Any] = {}
    if result.raw_response:
        try:
            decoded = json.loads(result.raw_response)
            if isinstance(decoded, dict):
                payload = decoded
        except json.JSONDecodeError:
            pass
    choices = payload.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices else {}
    if not isinstance(choice, dict):
        choice = {}
    message = choice.get("message")
    if not isinstance(message, dict):
        message = {}
    finish_reason = str(choice.get("finish_reason") or payload.get("finish_reason") or "")
    refusal_value = message.get("refusal") or choice.get("refusal") or payload.get("refusal")
    refusal = str(refusal_value or "").strip()[:300]
    normalized_reason = finish_reason.strip().lower().replace("-", "_")
    if refusal or normalized_reason in _FILTER_REASONS:
        kind = "content_filtered"
    elif normalized_reason in _BUDGET_REASONS or (
        result.completion_tokens > 0 or result.reasoning_tokens > 0
    ):
        kind = "output_budget_exhausted"
    else:
        kind = "upstream_empty_response"
    return EmptyResponseAttempt(
        attempt=attempt,
        max_tokens=max_tokens,
        kind=kind,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        reasoning_tokens=result.reasoning_tokens,
        finish_reason=finish_reason,
        response_id=str(payload.get("id") or ""),
        message_keys=tuple(sorted(str(key) for key in message)),
        refusal=refusal,
    )


def should_expand_output_budget(attempt: EmptyResponseAttempt) -> bool:
    return attempt.kind == "output_budget_exhausted"


def is_content_filtered_error(error: BaseException) -> bool:
    """Identify provider safety refusals that should be deferred to another model."""
    if isinstance(error, EmptyProviderResponseError):
        return error.kind == "content_filtered"
    normalized = " ".join(str(error).casefold().replace("（", "(").split())
    return any(marker in normalized for marker in _CONTENT_FILTER_MARKERS)


def content_filter_audit_payload(
    error: BaseException,
    *,
    stage: str,
    provider: str,
    model: str,
    segment_ordinal: int,
) -> dict[str, Any]:
    """Build a source-free audit record for a segment deferred after a refusal."""
    if isinstance(error, EmptyProviderResponseError):
        return error.audit_payload()
    return {
        "kind": "content_filtered",
        "job_stage": stage,
        "provider": provider,
        "model": model,
        "segment_ordinal": segment_ordinal,
        "message": str(error)[:1_000],
    }


def deferred_translation_message(store, segment_id: str) -> str:
    """Return the latest local failure note when a segment needs a safe retry or review."""
    events = store.list_audit_events("segment", segment_id)
    for event in reversed(events):
        payload = event.get("payload") or {}
        if (
            event.get("action") == "provider_failure"
            and (
                payload.get("kind") == "content_filtered"
                or payload.get("deferred") is True
            )
            and payload.get("job_stage") in {"draft", "repair"}
        ):
            return str(payload.get("message") or "该段草译未通过结构或内容验收。")
    return ""


def deferred_content_filter_message(store, segment_id: str) -> str:
    """Backward-compatible alias for callers that need any deferred draft note."""
    return deferred_translation_message(store, segment_id)


class EmptyProviderResponseError(RuntimeError):
    """A model call succeeded at the transport layer but produced no visible text."""

    def __init__(
        self,
        *,
        segment_id: str,
        segment_ordinal: int,
        stage: str,
        provider: str,
        model: str,
        attempts: list[EmptyResponseAttempt],
        results: list[TranslationResult],
    ):
        self.segment_id = segment_id
        self.segment_ordinal = segment_ordinal
        self.stage = stage
        self.provider = provider
        self.model = model
        self.attempts = tuple(attempts)
        self.results = tuple(results)
        kinds = {item.kind for item in attempts}
        if "content_filtered" in kinds:
            self.kind = "content_filtered"
        elif "upstream_empty_response" in kinds:
            self.kind = "upstream_empty_response"
        else:
            self.kind = "output_budget_exhausted"
        super().__init__(self._message())

    def _message(self) -> str:
        location = f"第 {self.segment_ordinal + 1} 段（{self.segment_id}）"
        last = self.attempts[-1]
        if self.kind == "content_filtered":
            detail = "上游模型拒绝或过滤了该内容"
            guidance = "请更换模型或人工处理该段。"
        elif self.kind == "output_budget_exhausted":
            detail = "模型的推理/输出预算用尽，仍未产生可见译文"
            guidance = "请增加输出额度、降低推理强度或重试该段。"
        else:
            detail = "上游返回了零 token 空响应，未提供明确原因"
            guidance = "请稍后重试该段或检查模型服务。"
        finish = f"，finish_reason={last.finish_reason}" if last.finish_reason else ""
        return (
            f"{detail}：{location}；模型 {self.model}，"
            f"completion_tokens={last.completion_tokens}，"
            f"reasoning_tokens={last.reasoning_tokens}{finish}。"
            f"已保留诊断记录；{guidance}"
        )

    def usage_result(self) -> TranslationResult:
        return TranslationResult(
            text="",
            prompt_tokens=sum(item.prompt_tokens for item in self.results),
            completion_tokens=sum(item.completion_tokens for item in self.results),
            reasoning_tokens=sum(item.reasoning_tokens for item in self.results),
            prompt_cache_hit_tokens=sum(
                item.prompt_cache_hit_tokens for item in self.results
            ),
            prompt_cache_miss_tokens=sum(
                item.prompt_cache_miss_tokens for item in self.results
            ),
            cost_usd=sum(item.cost_usd for item in self.results),
        )

    def audit_payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "job_stage": self.stage,
            "provider": self.provider,
            "model": self.model,
            "segment_ordinal": self.segment_ordinal,
            "message": str(self),
            "attempts": [item.to_dict() for item in self.attempts],
        }
