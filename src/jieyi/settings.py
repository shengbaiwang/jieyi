from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from jieyi.domain.reasoning import (
    COMPUTE_MODES,
    LEGACY_EFFORTS,
    legacy_effort_for_mode,
    normalize_compute_mode,
)
from jieyi.providers.catalog import (
    PROVIDER_PRESETS,
    get_provider_preset,
    infer_provider_type,
    join_endpoint,
    validate_http_url,
)

SETTINGS_VERSION = 4
KEYCHAIN_SERVICE = "org.jieyi.translation.providers"
LEGACY_KEYCHAIN_SERVICE = "org.jieyi.translation.openai-compatible"
LEGACY_KEYCHAIN_ACCOUNT = "default"
PROFILE_ID_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")


@dataclass(slots=True)
class ProviderProfile:
    id: str = "default"
    name: str = "OpenAI"
    provider_type: str = "openai"
    base_url: str = "https://api.openai.com/v1"
    chat_path: str = "chat/completions"
    models_path: str = "models"
    protocol: str = "chat_completions"
    auth_required: bool = True
    capabilities: tuple[str, ...] = ()

    @property
    def registry_name(self) -> str:
        return f"profile:{self.id}"

    @property
    def chat_endpoint(self) -> str:
        return join_endpoint(self.base_url, self.chat_path)

    @property
    def models_endpoint(self) -> str:
        return join_endpoint(self.base_url, self.models_path)


@dataclass(slots=True)
class ModelBinding:
    profile_id: str = "default"
    model: str = ""
    compute_mode: str = "economy"


def _current_capabilities(provider_type: str, stored: tuple[str, ...]) -> tuple[str, ...]:
    """Merge newly supported preset features into already-saved provider profiles."""
    preset = get_provider_preset(provider_type)
    if provider_type == "custom" or preset.id != provider_type:
        return stored
    return tuple(dict.fromkeys((*stored, *preset.capabilities)))


def _default_profile() -> ProviderProfile:
    preset = get_provider_preset("openai")
    return ProviderProfile(
        id="default",
        name=preset.name,
        provider_type=preset.id,
        base_url=preset.base_url,
        chat_path=preset.chat_path,
        models_path=preset.models_path,
        auth_required=preset.auth_required,
        capabilities=preset.capabilities,
    )


@dataclass(slots=True)
class ProviderSettings:
    version: int = SETTINGS_VERSION
    profiles: tuple[ProviderProfile, ...] = field(default_factory=lambda: (_default_profile(),))
    draft: ModelBinding = field(default_factory=ModelBinding)
    term_discovery: ModelBinding | None = None

    def __post_init__(self):
        if self.term_discovery is None:
            self.term_discovery = ModelBinding(
                self.draft.profile_id, self.draft.model, self.draft.compute_mode
            )

    def profile(self, profile_id: str) -> ProviderProfile | None:
        return next((item for item in self.profiles if item.id == profile_id), None)


@dataclass(frozen=True, slots=True)
class SecretWriteResult:
    source: str
    warning: str = ""


class LocalSecretStore:
    """Store one credential per profile, with environment and session fallbacks."""

    def __init__(self):
        self._session_keys: dict[str, str] = {}

    def get(self, profile_id: str) -> tuple[str, str]:
        environment_key = os.getenv(self._environment_name(profile_id), "").strip()
        if not environment_key and profile_id == "default":
            environment_key = os.getenv("JIEYI_OPENAI_API_KEY", "").strip()
        if environment_key:
            return environment_key, "environment"
        if profile_id in self._session_keys:
            return self._session_keys[profile_id], "session"
        if not self._keychain_available():
            return "", "none"
        key = self._read_keychain(KEYCHAIN_SERVICE, profile_id)
        if not key and profile_id == "default":
            key = self._read_keychain(LEGACY_KEYCHAIN_SERVICE, LEGACY_KEYCHAIN_ACCOUNT)
        return (key, "keychain") if key else ("", "none")

    def set(self, profile_id: str, api_key: str) -> SecretWriteResult:
        value = api_key.strip()
        if not value:
            return SecretWriteResult(self.get(profile_id)[1])
        if self._keychain_available():
            result = subprocess.run(
                [
                    "security",
                    "add-generic-password",
                    "-U",
                    "-s",
                    KEYCHAIN_SERVICE,
                    "-a",
                    profile_id,
                    "-w",
                    value,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                self._session_keys.pop(profile_id, None)
                return SecretWriteResult("keychain")
            self._session_keys[profile_id] = value
            detail = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "授权失败"
            return SecretWriteResult(
                "session",
                f"macOS 钥匙串暂不可写，密钥仅在本次运行有效（{detail}）",
            )
        self._session_keys[profile_id] = value
        return SecretWriteResult("session")

    @staticmethod
    def _environment_name(profile_id: str) -> str:
        suffix = re.sub(r"[^A-Za-z0-9]", "_", profile_id).upper()
        return f"JIEYI_API_KEY_{suffix}"

    @staticmethod
    def _keychain_available() -> bool:
        return (
            sys.platform == "darwin"
            and os.getenv("JIEYI_DISABLE_KEYCHAIN", "") != "1"
            and shutil.which("security") is not None
        )

    @staticmethod
    def _read_keychain(service: str, account: str) -> str:
        result = subprocess.run(
            ["security", "find-generic-password", "-s", service, "-a", account, "-w"],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else ""


class LocalSettingsStore:
    """Persist versioned provider profiles and keep their API keys outside the config file."""

    def __init__(self, path: str | Path, secret_store: LocalSecretStore | None = None):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.secrets = secret_store or LocalSecretStore()

    def load(self) -> ProviderSettings:
        if not self.path.exists():
            return ProviderSettings()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return ProviderSettings()
        if payload.get("version") in {2, 3, SETTINGS_VERSION} and isinstance(
            payload.get("profiles"), list
        ):
            return self._load_v2(payload)
        return self._migrate_legacy(payload)

    def save(self, settings: ProviderSettings) -> ProviderSettings:
        validate_settings(settings)
        temporary = self.path.with_suffix(f"{self.path.suffix}.tmp")
        temporary.write_text(
            json.dumps(asdict(settings), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.chmod(0o600)
        temporary.replace(self.path)
        return settings

    def get_api_key(self, profile_id: str = "default") -> tuple[str, str]:
        return self.secrets.get(profile_id)

    def set_api_key(self, api_key: str, profile_id: str = "default") -> SecretWriteResult:
        return self.secrets.set(profile_id, api_key)

    @staticmethod
    def _load_v2(payload: dict[str, Any]) -> ProviderSettings:
        profiles = tuple(
            ProviderProfile(
                id=str(item.get("id", "")),
                name=str(item.get("name", "")),
                provider_type=str(item.get("provider_type", "custom")),
                base_url=str(item.get("base_url", "")),
                chat_path=str(item.get("chat_path", "chat/completions")),
                models_path=str(item.get("models_path", "models")),
                protocol=str(item.get("protocol", "chat_completions")),
                auth_required=bool(item.get("auth_required", True)),
                capabilities=_current_capabilities(
                    str(item.get("provider_type", "custom")),
                    tuple(str(value) for value in item.get("capabilities", [])),
                ),
            )
            for item in payload.get("profiles", [])
            if isinstance(item, dict)
        )
        draft_payload = payload.get("draft") or {}
        discovery_payload = payload.get("term_discovery")
        first_id = profiles[0].id if profiles else "default"
        settings = ProviderSettings(
            profiles=profiles or (_default_profile(),),
            draft=ModelBinding(
                profile_id=str(draft_payload.get("profile_id", first_id)),
                model=str(draft_payload.get("model", "")),
                compute_mode=normalize_compute_mode(
                    str(
                        draft_payload.get("compute_mode")
                        or draft_payload.get("reasoning_effort")
                        or ""
                    ),
                    "economy",
                ),
            ),
            term_discovery=ModelBinding(
                profile_id=str(discovery_payload.get("profile_id", first_id)),
                model=str(discovery_payload.get("model", "")),
                compute_mode=normalize_compute_mode(
                    discovery_payload.get("compute_mode"), "balanced"
                ),
            ) if isinstance(discovery_payload, dict) else None,
        )
        try:
            validate_settings(settings)
        except ValueError:
            return ProviderSettings()
        return settings

    @staticmethod
    def _migrate_legacy(payload: dict[str, Any]) -> ProviderSettings:
        base_url = str(payload.get("base_url") or "https://api.openai.com/v1").strip()
        legacy_type = str(payload.get("provider_type") or "custom")
        provider_type = infer_provider_type(base_url, legacy_type)
        preset = get_provider_preset(provider_type)
        profile = ProviderProfile(
            id="default",
            name=preset.name,
            provider_type=provider_type,
            base_url=base_url.rstrip("/"),
            chat_path=preset.chat_path,
            models_path=preset.models_path,
            auth_required=preset.auth_required,
            capabilities=preset.capabilities,
        )
        return ProviderSettings(
            profiles=(profile,),
            draft=ModelBinding("default", str(payload.get("draft_model") or ""), "economy"),
        )


def profile_from_preset(
    profile_id: str,
    provider_type: str,
    *,
    name: str = "",
    base_url: str = "",
    chat_path: str = "",
    models_path: str = "",
    protocol: str = "",
    auth_required: bool | None = None,
    capabilities: tuple[str, ...] | None = None,
) -> ProviderProfile:
    preset = get_provider_preset(provider_type)
    return ProviderProfile(
        id=profile_id.strip(),
        name=name.strip() or preset.name,
        provider_type=provider_type.strip() or "custom",
        base_url=(base_url.strip() or preset.base_url).rstrip("/"),
        chat_path=chat_path.strip() or preset.chat_path,
        models_path=models_path.strip() or preset.models_path,
        protocol=protocol.strip() or getattr(preset, "protocol", "chat_completions"),
        auth_required=preset.auth_required if auth_required is None else auth_required,
        capabilities=preset.capabilities if capabilities is None else capabilities,
    )


def validate_settings(settings: ProviderSettings) -> None:
    if not settings.profiles:
        raise ValueError("至少需要一个模型服务连接")
    ids: set[str] = set()
    for profile in settings.profiles:
        if not PROFILE_ID_PATTERN.fullmatch(profile.id):
            raise ValueError(f"连接 ID 不合法：{profile.id}")
        if profile.id in ids:
            raise ValueError(f"连接 ID 重复：{profile.id}")
        ids.add(profile.id)
        validate_http_url(profile.base_url, f"{profile.name} 的 API 地址")
        validate_http_url(profile.chat_endpoint, f"{profile.name} 的 Chat 地址")
        validate_http_url(profile.models_endpoint, f"{profile.name} 的 Models 地址")
    if settings.draft.profile_id not in ids:
        raise ValueError("草译模型绑定的连接不存在")
    if settings.term_discovery.profile_id not in ids:
        raise ValueError("术语发现模型绑定的连接不存在")
    for role, binding in (
        ("草译", settings.draft),
        ("术语发现", settings.term_discovery),
    ):
        if binding.compute_mode not in COMPUTE_MODES:
            raise ValueError(f"{role}计算策略不合法：{binding.compute_mode}")


def provider_public_payload(store: LocalSettingsStore) -> dict[str, Any]:
    settings = store.load()
    profiles: list[dict[str, Any]] = []
    for profile in settings.profiles:
        api_key, key_source = store.get_api_key(profile.id)
        profiles.append(
            {
                **asdict(profile),
                "chat_endpoint": profile.chat_endpoint,
                "models_endpoint": profile.models_endpoint,
                "api_key_configured": bool(api_key),
                "key_source": key_source,
            }
        )
    draft_profile = settings.profile(settings.draft.profile_id) or settings.profiles[0]
    discovery_profile = settings.profile(settings.term_discovery.profile_id) or draft_profile
    draft_key, draft_key_source = store.get_api_key(draft_profile.id)
    return {
        "version": SETTINGS_VERSION,
        "profiles": profiles,
        "presets": [preset.public_payload() for preset in PROVIDER_PRESETS],
        "draft_profile_id": settings.draft.profile_id,
        "draft_provider": draft_profile.registry_name,
        "draft_model": settings.draft.model,
        "draft_compute_mode": settings.draft.compute_mode,
        "draft_reasoning_effort": legacy_effort_for_mode(settings.draft.compute_mode),
        "term_discovery_profile_id": settings.term_discovery.profile_id,
        "term_discovery_provider": discovery_profile.registry_name,
        "term_discovery_model": settings.term_discovery.model,
        "term_discovery_compute_mode": settings.term_discovery.compute_mode,
        "warnings": configuration_warnings(settings),
        "provider_type": draft_profile.provider_type,
        "base_url": draft_profile.base_url,
        "api_key_configured": bool(draft_key),
        "key_source": draft_key_source,
    }


def configuration_warnings(settings: ProviderSettings) -> list[str]:
    warnings: list[str] = []
    bindings = (
        ("草译", settings.draft),
        ("术语发现", settings.term_discovery),
    )
    for role, binding in bindings:
        if not binding.model:
            continue
        profile = settings.profile(binding.profile_id)
        if profile is None:
            continue
        model = binding.model.lower()
        if profile.provider_type.startswith("glm") and not model.startswith("glm-"):
            warnings.append(f"{role}模型 {binding.model} 与 GLM 连接可能不匹配")
        elif profile.provider_type == "deepseek" and not model.startswith("deepseek-"):
            warnings.append(f"{role}模型 {binding.model} 与 DeepSeek 连接可能不匹配")
        elif profile.provider_type == "kimi-coding" and not model.startswith(
            ("k3", "kimi-for-coding")
        ):
            warnings.append(f"{role}模型 {binding.model} 与 Kimi Coding 连接可能不匹配")
        elif profile.provider_type == "kimi-platform" and not model.startswith("kimi-"):
            warnings.append(f"{role}模型 {binding.model} 与 Kimi 开放平台连接可能不匹配")
    return warnings


def test_openai_compatible_connection(
    base_url: str,
    api_key: str,
    timeout_seconds: int = 20,
    *,
    models_endpoint: str = "",
    required_models: Iterable[str] = (),
    protocol: str = "chat_completions",
) -> dict[str, Any]:
    normalized = validate_http_url(base_url)
    if protocol == "anthropic_messages":
        return {
            "ok": True,
            "models": [],
            "models_url": "",
            "stages": [
                {"id": "url", "ok": True, "message": "地址格式正确"},
                {"id": "models", "ok": True, "message": "Anthropic 接口未提供通用模型列表，请手动输入模型 ID 后实测"},
            ],
            "notes": ["请使用服务端提供的 Claude 模型 ID，并通过“实测能力”验证 /v1/messages。"],
        }
    models_url = validate_http_url(
        models_endpoint or join_endpoint(normalized, "models"), "Models 地址"
    )
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(models_url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        if exc.code in {401, 403}:
            raise RuntimeError(f"认证失败（HTTP {exc.code}），请检查该连接对应的 API Key") from exc
        raise RuntimeError(f"模型列表接口返回 HTTP {exc.code}：{detail}") from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"无法连接模型服务：{exc}") from exc

    model_items = payload.get("data") if isinstance(payload, dict) else None
    model_ids = [str(item.get("id")) for item in model_items or [] if isinstance(item, dict)]
    required = list(dict.fromkeys(model.strip() for model in required_models if model.strip()))
    missing = [model for model in required if model_ids and model not in model_ids]
    if missing:
        raise RuntimeError(
            f"连接与认证正常，但模型列表中找不到：{'、'.join(missing)}。"
            "请从服务端返回的模型列表中重新选择。"
        )
    return {
        "ok": True,
        "models": model_ids[:100],
        "models_url": models_url,
        "stages": [
            {"id": "url", "ok": True, "message": "地址格式正确"},
            {"id": "auth", "ok": True, "message": "服务已接受认证"},
            {"id": "models", "ok": True, "message": f"读取到 {len(model_ids)} 个模型"},
            *([{"id": "binding", "ok": True, "message": "已绑定模型可用"}] if required else []),
        ],
    }


def _probe_error_detail(exc: urllib.error.HTTPError) -> str:
    try:
        raw = exc.read().decode("utf-8", errors="replace").strip()
    except (OSError, ValueError):
        raw = ""
    if not raw:
        return str(exc.reason or "请求失败")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return " ".join(raw.split())[:600]
    if isinstance(payload, dict):
        error = payload.get("error", payload)
        if isinstance(error, dict):
            message = error.get("message") or error.get("msg") or payload.get("message")
            if message:
                return str(message)[:600]
    return " ".join(raw.split())[:600]


def _probe_chat_completion(
    chat_endpoint: str,
    api_key: str,
    model: str,
    timeout_seconds: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": "Translate into Chinese. Return only the translation: The book is open.",
            }
        ],
        "max_tokens": 128,
    }
    payload.update(extra or {})
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        chat_endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = _probe_error_detail(exc)
        if exc.code in {400, 404, 405, 409, 422}:
            return {
                "accepted": False,
                "status": exc.code,
                "detail": detail,
                "latency_ms": round((time.perf_counter() - started) * 1000),
                "tokens": 0,
                "visible_output": False,
            }
        if exc.code in {401, 403}:
            raise RuntimeError(f"模型请求认证失败（HTTP {exc.code}），请检查 API Key") from exc
        raise RuntimeError(f"模型能力探测返回 HTTP {exc.code}：{detail}") from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"无法完成模型能力探测：{exc}") from exc

    choices = result.get("choices") if isinstance(result, dict) else None
    content: object = None
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        message = choices[0].get("message")
        if isinstance(message, dict):
            content = message.get("content")
    visible_output = bool(content.strip()) if isinstance(content, str) else bool(content)
    usage = result.get("usage") if isinstance(result, dict) else {}
    if not isinstance(usage, dict):
        usage = {}
    tokens = int(
        usage.get("total_tokens")
        or (
            int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
            + int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        )
    )
    return {
        "accepted": True,
        "status": 200,
        "detail": "",
        "latency_ms": round((time.perf_counter() - started) * 1000),
        "tokens": tokens,
        "visible_output": visible_output,
    }


def _mode_mapping(
    supported_efforts: list[str], thinking_states: list[str]
) -> tuple[str, dict[str, str]]:
    if supported_efforts:
        ordered = [item for item in LEGACY_EFFORTS if item in supported_efforts]
        balanced = "medium" if "medium" in ordered else ordered[len(ordered) // 2]
        return "effort", {
            "economy": f"强度 {ordered[0]}",
            "balanced": f"强度 {balanced}",
            "performance": f"强度 {ordered[-1]}",
        }
    if {"disabled", "enabled"}.issubset(thinking_states):
        return "thinking", {
            "economy": "关闭思考",
            "balanced": "开启思考",
            "performance": "开启思考",
        }
    return "default", {
        "economy": "服务端默认",
        "balanced": "服务端默认",
        "performance": "服务端默认",
    }


def test_openai_compatible_model(
    chat_endpoint: str,
    api_key: str,
    model: str,
    timeout_seconds: int = 30,
    protocol: str = "chat_completions",
) -> dict[str, Any]:
    """Probe one selected model with tiny requests and report only verified controls.

    A valid value is considered verified only when the same endpoint rejects a deliberately
    invalid value. This prevents OpenAI-compatible proxies that silently discard unknown
    fields from being reported as supporting every reasoning level.
    """
    endpoint = validate_http_url(chat_endpoint, "Chat 地址")
    model_id = model.strip()
    if not model_id:
        raise ValueError("请先选择要测试的模型")

    if protocol != "chat_completions":
        from jieyi.domain.models import ModelSpec
        from jieyi.providers.openai_compatible import OpenAICompatibleProvider
        started = time.perf_counter()
        provider = OpenAICompatibleProvider(api_key=api_key, chat_endpoint=endpoint, protocol=protocol)
        try:
            result = provider._complete_sync([{"role": "user", "content": "Translate into Chinese. Return only the translation: The book is open."}], ModelSpec("probe", model_id), None, None, None, None)
        except Exception as exc:
            raise RuntimeError(f"模型 {model_id} 无法完成短译测试：{exc}") from exc
        return {"ok": True, "model": model_id, "baseline": {"visible_output": bool(result.text), "latency_ms": round((time.perf_counter() - started) * 1000)}, "reasoning": {"kind": "default", "verification": "unverified", "supported_efforts": [], "accepted_efforts": [], "empty_efforts": [], "thinking_states": []}, "mode_mapping": {"economy": "服务端默认", "balanced": "服务端默认", "performance": "服务端默认"}, "requests": 1, "total_tokens": result.prompt_tokens + result.completion_tokens, "duration_ms": round((time.perf_counter() - started) * 1000), "notes": ["该协议使用其原生请求格式，思考档位由服务端默认行为决定。"]}

    attempts: list[dict[str, Any]] = []

    def run(extra: dict[str, Any] | None = None) -> dict[str, Any]:
        result = _probe_chat_completion(endpoint, api_key, model_id, timeout_seconds, extra)
        attempts.append(result)
        return result

    baseline = run()
    if not baseline["accepted"]:
        raise RuntimeError(
            f"模型 {model_id} 无法完成短译测试（HTTP {baseline['status']}）：{baseline['detail']}"
        )

    effort_results = {effort: run({"reasoning_effort": effort}) for effort in LEGACY_EFFORTS}
    accepted_efforts = [effort for effort, result in effort_results.items() if result["accepted"]]
    empty_efforts = [
        effort for effort in accepted_efforts if not effort_results[effort]["visible_output"]
    ]
    invalid_effort = run({"reasoning_effort": "__jieyi_invalid__"})
    effort_verified = not invalid_effort["accepted"]
    supported_efforts = accepted_efforts if effort_verified else []

    accepted_thinking = [
        state for state in ("disabled", "enabled") if run({"thinking": {"type": state}})["accepted"]
    ]
    invalid_thinking = run({"thinking": {"type": "__jieyi_invalid__"}})
    thinking_verified = not invalid_thinking["accepted"]
    thinking_states = accepted_thinking if thinking_verified else []

    kind, mode_mapping = _mode_mapping(supported_efforts, thinking_states)
    notes: list[str] = []
    if accepted_efforts and not effort_verified:
        notes.append("接口连无效 reasoning_effort 也接受，可能会静默忽略该参数，未标记为已支持。")
    if accepted_thinking and not thinking_verified:
        notes.append("接口连无效 thinking 类型也接受，可能会静默忽略该参数，未标记为已支持。")
    if empty_efforts:
        notes.append(
            f"强度 {'、'.join(empty_efforts)} 接受参数但短测没有可见文本；"
            "正式翻译需提高输出 token 上限，或改用更低强度。"
        )
    if not baseline["visible_output"]:
        notes.append("短译请求成功但没有可见文本；这个模型不适合直接开始批量翻译。")
    if kind == "default":
        notes.append("未验证到可调思考控制，三档模式将使用服务端默认行为。")

    return {
        "ok": True,
        "model": model_id,
        "baseline": {
            "visible_output": baseline["visible_output"],
            "latency_ms": baseline["latency_ms"],
        },
        "reasoning": {
            "kind": kind,
            "verification": "verified" if kind != "default" else "unverified",
            "supported_efforts": supported_efforts,
            "accepted_efforts": accepted_efforts,
            "empty_efforts": empty_efforts,
            "thinking_states": thinking_states,
        },
        "mode_mapping": mode_mapping,
        "requests": len(attempts),
        "total_tokens": sum(int(item["tokens"]) for item in attempts),
        "duration_ms": sum(int(item["latency_ms"]) for item in attempts),
        "notes": notes,
    }
