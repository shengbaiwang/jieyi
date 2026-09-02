from __future__ import annotations

import asyncio
import http.client
import json
import math
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable

from jieyi.domain.models import ModelSpec, TranslationRequest, TranslationResult
from jieyi.domain.reasoning import resolve_reasoning_control
from jieyi.prompting import build_messages

_OPENROUTER_TITLE = "Jieyi"


class ProviderError(RuntimeError):
    pass


def _message_text(payload: object) -> str:
    """Normalize text-only and content-block chat completion responses."""
    try:
        content = payload["choices"][0]["message"].get("content")  # type: ignore[index]
    except (KeyError, IndexError, TypeError, AttributeError) as exc:
        raise ProviderError("Provider returned an unsupported response shape") from exc
    if content is None:
        # Reasoning APIs commonly use null when the generation budget ends
        # before a visible answer. The workflow can retry this empty result
        # with a larger output budget while retaining usage diagnostics.
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts).strip()
    raise ProviderError("Provider returned an unsupported response shape")


def _required_temperature(detail: str) -> float | None:
    """Extract a provider-mandated temperature from a sanitized 400 detail."""
    normalized = " ".join(detail.lower().split())
    if "temperature" not in normalized:
        return None
    number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
    patterns = (
        rf"\bonly\s+({number})\s+(?:is\s+)?allowed\b",
        (
            rf"\btemperature\s*(?:=|:)?\s*"
            rf"(?:must\s+(?:be|equal)|should\s+be|required(?:\s+to\s+be)?)"
            rf"\s*({number})\b"
        ),
        rf"\btemperature\s*=\s*({number})\b",
    )
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match is None:
            continue
        value = float(match.group(1))
        if math.isfinite(value) and 0.0 <= value <= 2.0:
            return value
    return None


def _unsupported_reasoning_effort(detail: str) -> bool:
    normalized = " ".join(detail.lower().replace("-", "_").split())
    names_effort = "reasoning_effort" in normalized or "reasoning effort" in normalized
    english_rejection = names_effort and any(
        marker in normalized
        for marker in (
            "unsupported",
            "not supported",
            "invalid",
            "unknown",
            "unrecognized",
            "not permitted",
            "allowed values",
            "extra inputs",
        )
    )
    chinese_rejection = (
        "不支持关闭思考" in normalized
        or "不支持关闭深度思考" in normalized
        or (
            ("思考" in normalized or "推理" in normalized)
            and "请使用" in normalized
            and any(level in normalized for level in ("low", "medium", "high", "max"))
        )
    )
    return english_rejection or chinese_rejection


def _http_error_detail(exc: urllib.error.HTTPError) -> str:
    """Return a useful provider error without exposing request headers or credentials."""
    try:
        raw = exc.read().decode("utf-8", errors="replace").strip()
    except (OSError, ValueError):
        raw = ""
    if not raw:
        return str(exc.reason or "Bad Request")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return " ".join(raw.split())[:1_000]
    if isinstance(payload, dict):
        error = payload.get("error", payload)
        if isinstance(error, dict):
            message = error.get("message") or error.get("msg") or payload.get("message")
            code = error.get("code") or payload.get("code")
            if message:
                suffix = f"（代码 {code}）" if code else ""
                return f"{message}{suffix}"[:1_000]
    return " ".join(raw.split())[:1_000]


class OpenAICompatibleProvider:
    """Small SDK-free adapter for cloud or local OpenAI-compatible endpoints."""

    def __init__(
        self,
        base_url: str = "",
        api_key: str = "",
        timeout_seconds: int = 180,
        *,
        chat_endpoint: str = "",
        protocol: str = "chat_completions",
        capabilities: Iterable[str] = (),
    ):
        normalized = (chat_endpoint or base_url).rstrip("/")
        self.protocol = protocol.strip().lower() or "chat_completions"
        if self.protocol == "responses":
            self.endpoint = normalized if normalized.endswith("/responses") else f"{normalized}/responses"
        elif self.protocol == "anthropic_messages":
            self.endpoint = normalized if normalized.endswith("/messages") else f"{normalized}/messages"
        elif self.protocol == "gemini_generate_content":
            self.endpoint = normalized
        else:
            self.endpoint = (
                normalized
                if normalized.endswith("/chat/completions")
                else f"{normalized}/chat/completions"
            )
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.capabilities = frozenset(capabilities)
        self.is_openrouter = "openrouter.ai" in self.endpoint.lower()
        self._temperature_overrides: dict[str, float] = {}
        self._reasoning_effort_overrides: dict[tuple[str, str], str | None] = {}

    async def translate(self, request: TranslationRequest, model: ModelSpec) -> TranslationResult:
        return await asyncio.to_thread(self._translate_sync, request, model)

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
        return await asyncio.to_thread(
            self._complete_sync,
            messages,
            model,
            thinking,
            reasoning_effort,
            max_tokens,
            compute_mode,
        )

    @staticmethod
    def _plain_message_text(messages: list[dict[str, str]]) -> str:
        return "\n\n".join(
            f"{item.get('role', 'user')}: {item.get('content', '')}" for item in messages
            if item.get("content")
        )

    def _json_request(self, payload: dict[str, object], headers: dict[str, str], endpoint: str | None = None) -> dict[str, object]:
        request = urllib.request.Request(
            endpoint or self.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                value = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise ProviderError(f"Provider request failed: HTTP {exc.code}: {_http_error_detail(exc)}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise ProviderError(f"Provider request failed: {exc}") from exc
        if not isinstance(value, dict):
            raise ProviderError("Provider returned an unsupported response shape")
        return value

    def _complete_responses_sync(self, messages, model, reasoning_effort, max_tokens):
        payload: dict[str, object] = {"model": model.model, "input": messages}
        if max_tokens is not None:
            payload["max_output_tokens"] = max_tokens
        if reasoning_effort:
            payload["reasoning"] = {"effort": reasoning_effort}
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
            headers["x-api-key"] = self.api_key
        result = self._json_request(payload, headers)
        text = result.get("output_text")
        if not isinstance(text, str):
            parts = []
            for item in result.get("output", []) if isinstance(result.get("output"), list) else []:
                for block in item.get("content", []) if isinstance(item, dict) and isinstance(item.get("content"), list) else []:
                    if isinstance(block, dict) and isinstance(block.get("text"), str): parts.append(block["text"])
            text = "\n".join(parts)
        usage = result.get("usage") if isinstance(result.get("usage"), dict) else {}
        return TranslationResult(text=text.strip(), prompt_tokens=int(usage.get("input_tokens") or 0), completion_tokens=int(usage.get("output_tokens") or 0), raw_response=json.dumps(result, ensure_ascii=False))

    def _complete_anthropic_sync(self, messages, model, max_tokens):
        system = "\n".join(item["content"] for item in messages if item.get("role") == "system")
        items = [{"role": item.get("role", "user"), "content": item.get("content", "")} for item in messages if item.get("role") != "system"]
        payload: dict[str, object] = {"model": model.model, "messages": items, "max_tokens": max_tokens or 4096}
        if system: payload["system"] = system
        headers = {"Content-Type": "application/json", "Accept": "application/json", "anthropic-version": "2023-06-01"}
        if self.api_key: headers["x-api-key"] = self.api_key
        result = self._json_request(payload, headers)
        parts = [item.get("text", "") for item in result.get("content", []) if isinstance(item, dict) and isinstance(item.get("text"), str)]
        usage = result.get("usage") if isinstance(result.get("usage"), dict) else {}
        return TranslationResult(text="\n".join(parts).strip(), prompt_tokens=int(usage.get("input_tokens") or 0), completion_tokens=int(usage.get("output_tokens") or 0), raw_response=json.dumps(result, ensure_ascii=False))

    def _complete_gemini_sync(self, messages, model, max_tokens):
        endpoint = self.endpoint
        separator = "&" if "?" in endpoint else "?"
        endpoint = f"{endpoint}{separator}key={urllib.parse.quote(self.api_key)}" if self.api_key else endpoint
        contents = [{"role": "model" if item.get("role") == "assistant" else "user", "parts": [{"text": item.get("content", "")}]} for item in messages if item.get("role") != "system"]
        payload: dict[str, object] = {"contents": contents}
        if max_tokens is not None: payload["generationConfig"] = {"maxOutputTokens": max_tokens, "temperature": model.temperature}
        result = self._json_request(payload, {"Content-Type": "application/json", "Accept": "application/json"}, endpoint)
        parts = []
        for candidate in result.get("candidates", []) if isinstance(result.get("candidates"), list) else []:
            content = candidate.get("content", {}) if isinstance(candidate, dict) else {}
            for part in content.get("parts", []) if isinstance(content, dict) and isinstance(content.get("parts"), list) else []:
                if isinstance(part, dict) and isinstance(part.get("text"), str): parts.append(part["text"])
        usage = result.get("usageMetadata") if isinstance(result.get("usageMetadata"), dict) else {}
        return TranslationResult(text="\n".join(parts).strip(), prompt_tokens=int(usage.get("promptTokenCount") or 0), completion_tokens=int(usage.get("candidatesTokenCount") or 0), raw_response=json.dumps(result, ensure_ascii=False))

    def _translate_sync(self, request: TranslationRequest, model: ModelSpec) -> TranslationResult:
        return self._complete_sync(build_messages(request), model, None, None, None, None)

    def _complete_sync(
        self,
        messages: list[dict[str, str]],
        model: ModelSpec,
        thinking: bool | None,
        reasoning_effort: str | None,
        max_tokens: int | None,
        compute_mode: str | None = None,
        _effort_candidates: tuple[str | None, ...] | None = None,
    ) -> TranslationResult:
        if self.protocol == "responses":
            effort = reasoning_effort
            if compute_mode:
                control = resolve_reasoning_control(model.model, self.capabilities, compute_mode)
                effort = control.effort_candidates[0] if control.effort_candidates else None
            return self._complete_responses_sync(messages, model, effort, max_tokens)
        if self.protocol == "anthropic_messages":
            return self._complete_anthropic_sync(messages, model, max_tokens)
        if self.protocol == "gemini_generate_content":
            return self._complete_gemini_sync(messages, model, max_tokens)
        effective_thinking = thinking
        effort_candidates = _effort_candidates
        if compute_mode and effort_candidates is None:
            control = resolve_reasoning_control(model.model, self.capabilities, compute_mode)
            effective_thinking = control.thinking
            effort_candidates = control.effort_candidates
            override_key = (model.model, control.mode)
            if override_key in self._reasoning_effort_overrides:
                effort_candidates = (self._reasoning_effort_overrides[override_key],)
        if effort_candidates is None:
            effort_candidates = (reasoning_effort,)
        effective_effort = effort_candidates[0]
        payload: dict[str, object] = {"model": model.model, "messages": messages}
        if effective_thinking is not None and "thinking" in self.capabilities:
            payload["thinking"] = {"type": "enabled" if effective_thinking else "disabled"}
        if effective_effort and "reasoning_effort" in self.capabilities:
            payload["reasoning_effort"] = effective_effort
        if not effective_thinking:
            payload["temperature"] = self._temperature_overrides.get(model.model, model.temperature)
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        request_temperature = payload.get("temperature")
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.is_openrouter:
            headers["HTTP-Referer"] = "http://localhost:3000"
            # urllib/http.client serializes header values as Latin-1. Keep the
            # optional OpenRouter attribution title ASCII-safe; the Chinese UI
            # name belongs in content, where the request body is UTF-8.
            headers["X-OpenRouter-Title"] = _OPENROUTER_TITLE
        http_request = urllib.request.Request(
            self.endpoint, data=body, headers=headers, method="POST"
        )

        payload = None
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(http_request, timeout=self.timeout_seconds) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                detail = _http_error_detail(exc)
                required_temperature = _required_temperature(detail)
                if (
                    exc.code == 400
                    and required_temperature is not None
                    and request_temperature != required_temperature
                ):
                    self._temperature_overrides[model.model] = required_temperature
                    return self._complete_sync(
                        messages,
                        model,
                        thinking,
                        reasoning_effort,
                        max_tokens,
                        compute_mode,
                        effort_candidates,
                    )
                if (
                    exc.code == 400
                    and compute_mode
                    and len(effort_candidates) > 1
                    and _unsupported_reasoning_effort(detail)
                ):
                    return self._complete_sync(
                        messages,
                        model,
                        thinking,
                        reasoning_effort,
                        max_tokens,
                        compute_mode,
                        effort_candidates[1:],
                    )
                last_error = ProviderError(f"HTTP {exc.code}: {detail}")
                if exc.code not in {429, 500, 502, 503, 504} or attempt == 2:
                    break
                retry_after = exc.headers.get("Retry-After", "")
                delay = float(retry_after) if retry_after.isdigit() else float(2**attempt)
                time.sleep(min(delay, 8.0))
            except (
                urllib.error.URLError,
                TimeoutError,
                json.JSONDecodeError,
                http.client.IncompleteRead,
                ConnectionError,
            ) as exc:
                last_error = exc
                if attempt == 2:
                    break
                time.sleep(float(2**attempt))
        if payload is None:
            raise ProviderError(f"Provider request failed: {last_error}") from last_error
        if compute_mode:
            control = resolve_reasoning_control(model.model, self.capabilities, compute_mode)
            self._reasoning_effort_overrides[(model.model, control.mode)] = effective_effort

        text = _message_text(payload)

        usage = payload.get("usage") or {}
        completion_details = usage.get("completion_tokens_details") or {}
        prompt_details = (
            usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
        )
        prompt_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        cache_hit_tokens = int(
            usage.get("prompt_cache_hit_tokens")
            or usage.get("cache_read_input_tokens")
            or prompt_details.get("cached_tokens")
            or 0
        )
        explicit_cache_miss = (
            usage.get("prompt_cache_miss_tokens")
            or usage.get("cache_creation_input_tokens")
            or prompt_details.get("cache_write_tokens")
        )
        cache_miss_tokens = int(
            explicit_cache_miss
            if explicit_cache_miss is not None
            else max(0, prompt_tokens - cache_hit_tokens)
        )
        return TranslationResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=int(
                usage.get("completion_tokens") or usage.get("output_tokens") or 0
            ),
            cost_usd=float(usage.get("cost") or 0),
            reasoning_tokens=int(completion_details.get("reasoning_tokens") or 0),
            prompt_cache_hit_tokens=cache_hit_tokens,
            prompt_cache_miss_tokens=cache_miss_tokens,
            raw_response=json.dumps(payload, ensure_ascii=False),
        )
