from __future__ import annotations

from dataclasses import asdict, dataclass
from urllib.parse import urlparse


@dataclass(frozen=True, slots=True)
class ProviderPreset:
    id: str
    name: str
    note: str
    base_url: str
    chat_path: str = "chat/completions"
    models_path: str = "models"
    protocol: str = "chat_completions"
    auth_required: bool = True
    capabilities: tuple[str, ...] = ()
    default_models: tuple[str, ...] = ()

    def public_payload(self) -> dict[str, object]:
        return asdict(self)


PROVIDER_PRESETS: tuple[ProviderPreset, ...] = (
    ProviderPreset(
        id="openai",
        name="OpenAI",
        note="官方 API",
        base_url="https://api.openai.com/v1",
        capabilities=("reasoning_effort", "tools", "vision"),
    ),
    ProviderPreset(
        id="openrouter",
        name="OpenRouter",
        note="多模型聚合",
        base_url="https://openrouter.ai/api/v1",
        capabilities=("reasoning_effort", "tools", "vision"),
    ),
    ProviderPreset(
        id="deepseek",
        name="DeepSeek",
        note="官方兼容接口",
        base_url="https://api.deepseek.com/v1",
        capabilities=("thinking", "reasoning_effort", "tools"),
        default_models=("deepseek-v4-flash", "deepseek-v4-pro"),
    ),
    ProviderPreset(
        id="kimi-coding",
        name="Kimi Coding",
        note="会员编程 API",
        base_url="https://api.kimi.com/coding/v1",
        capabilities=("reasoning_effort", "tools", "vision"),
        default_models=("k3", "k3-256k", "kimi-for-coding"),
    ),
    ProviderPreset(
        id="kimi-platform",
        name="Kimi 开放平台",
        note="按量付费 API",
        base_url="https://api.moonshot.cn/v1",
        capabilities=("reasoning_effort", "tools", "vision"),
        default_models=("kimi-k3",),
    ),
    ProviderPreset(
        id="glm-cn",
        name="智谱 GLM",
        note="国内通用 API",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        capabilities=("thinking", "reasoning_effort", "tools", "vision"),
        default_models=("glm-5.2", "glm-4.7-flash"),
    ),
    ProviderPreset(
        id="glm-coding-cn",
        name="GLM Coding",
        note="国内编程套餐",
        base_url="https://open.bigmodel.cn/api/coding/paas/v4",
        capabilities=("thinking", "reasoning_effort", "tools", "vision"),
        default_models=("glm-5.2", "glm-4.7-flash"),
    ),
    ProviderPreset(
        id="glm-global",
        name="Z.ai GLM",
        note="国际通用 API",
        base_url="https://api.z.ai/api/paas/v4",
        capabilities=("thinking", "reasoning_effort", "tools", "vision"),
        default_models=("glm-5.2", "glm-4.7-flash"),
    ),
    ProviderPreset(
        id="glm-coding-global",
        name="Z.ai Coding",
        note="国际编程套餐",
        base_url="https://api.z.ai/api/coding/paas/v4",
        capabilities=("thinking", "reasoning_effort", "tools", "vision"),
        default_models=("glm-5.2", "glm-4.7-flash"),
    ),
    ProviderPreset(
        id="ollama",
        name="Ollama",
        note="本地模型",
        base_url="http://127.0.0.1:11434/v1",
        auth_required=False,
        capabilities=("tools", "vision"),
    ),
    ProviderPreset(
        id="lmstudio",
        name="LM Studio",
        note="本地模型",
        base_url="http://127.0.0.1:1234/v1",
        auth_required=False,
        capabilities=("tools", "vision"),
    ),
    ProviderPreset(
        id="custom",
        name="自定义",
        note="显式 OpenAI 兼容端点",
        base_url="",
        capabilities=("tools",),
    ),
)

PRESETS_BY_ID = {item.id: item for item in PROVIDER_PRESETS}


def get_provider_preset(provider_type: str) -> ProviderPreset:
    return PRESETS_BY_ID.get(provider_type, PRESETS_BY_ID["custom"])


def join_endpoint(base_url: str, path: str) -> str:
    """Join an explicitly configured base URL and endpoint path without guessing versions."""
    normalized_path = path.strip()
    if normalized_path.startswith(("http://", "https://")):
        return normalized_path.rstrip("/")
    return f"{base_url.strip().rstrip('/')}/{normalized_path.lstrip('/')}"


def validate_http_url(value: str, label: str = "API 地址") -> str:
    normalized = value.strip().rstrip("/")
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{label}必须是完整的 http:// 或 https:// 地址")
    return normalized


def infer_provider_type(base_url: str, fallback: str = "custom") -> str:
    value = base_url.lower()
    if "api.kimi.com/coding" in value:
        return "kimi-coding"
    if "api.moonshot.cn" in value or "api.moonshot.ai" in value:
        return "kimi-platform"
    if "open.bigmodel.cn/api/coding" in value:
        return "glm-coding-cn"
    if "open.bigmodel.cn/api/paas" in value:
        return "glm-cn"
    if "api.z.ai/api/coding" in value:
        return "glm-coding-global"
    if "api.z.ai/api/paas" in value:
        return "glm-global"
    if "openrouter.ai" in value:
        return "openrouter"
    if "deepseek.com" in value:
        return "deepseek"
    if "11434" in value:
        return "ollama"
    if "1234" in value:
        return "lmstudio"
    if "api.openai.com" in value:
        return "openai"
    return fallback if fallback in PRESETS_BY_ID else "custom"
