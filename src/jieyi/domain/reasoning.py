from __future__ import annotations

from dataclasses import dataclass

COMPUTE_MODES = ("economy", "balanced", "performance")
LEGACY_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")


def normalize_compute_mode(value: str | None, default: str = "balanced") -> str:
    """Convert old provider-specific effort values to a stable user intent."""
    normalized = (value or "").strip().lower()
    if normalized in COMPUTE_MODES:
        return normalized
    if normalized in {"none", "minimal", "low"}:
        return "economy"
    if normalized == "medium":
        return "balanced"
    if normalized in {"high", "xhigh", "max"}:
        return "performance"
    if default not in COMPUTE_MODES:
        raise ValueError(f"unknown compute mode: {default}")
    return default


@dataclass(frozen=True, slots=True)
class ReasoningControl:
    mode: str
    thinking: bool | None
    effort_candidates: tuple[str | None, ...]
    source: str


def _known_efforts(model: str) -> tuple[str, ...] | None:
    """Return model-specific levels only where the model ID makes them unambiguous."""
    name = model.strip().lower()
    if name.startswith("glm-5.3"):
        return ("low", "high", "max")
    if name.startswith("gpt-5.6"):
        return ("none", "low", "medium", "high", "xhigh", "max")
    if name.startswith(("gpt-5.5-pro", "gpt-5.4-pro")):
        return ("medium", "high", "xhigh")
    if name.startswith(("gpt-5.3-codex", "gpt-5.2-codex", "gpt-5.1-codex")):
        return ("low", "medium", "high", "xhigh")
    if name.startswith("gpt-5.1"):
        return ("none", "low", "medium", "high")
    if name.startswith(("gpt-oss-", "o3", "o4-mini")):
        return ("low", "medium", "high")
    return None


def _pick_known(levels: tuple[str, ...], mode: str) -> str:
    if mode == "economy":
        return levels[0]
    if mode == "performance":
        return levels[-1]
    return "medium" if "medium" in levels else levels[len(levels) // 2]


def resolve_reasoning_control(
    model: str,
    capabilities: frozenset[str] | set[str] | tuple[str, ...],
    compute_mode: str,
) -> ReasoningControl:
    """Map a portable compute intent to native controls with safe fallbacks.

    Unknown OpenAI-compatible models get conservative candidate sequences. A provider
    adapter may retry the next candidate after an explicit unsupported-parameter 400.
    The final None candidate means to use the model's own default.
    """
    mode = normalize_compute_mode(compute_mode)
    features = frozenset(capabilities)
    supports_effort = "reasoning_effort" in features
    supports_thinking = "thinking" in features
    # Do not send two competing native controls when an endpoint advertises both.
    thinking = (mode != "economy") if supports_thinking and not supports_effort else None

    if not supports_effort:
        source = "thinking" if supports_thinking else "default"
        return ReasoningControl(mode, thinking, (None,), source)

    known = _known_efforts(model)
    if known:
        return ReasoningControl(mode, thinking, (_pick_known(known, mode), None), "model")

    candidates: dict[str, tuple[str | None, ...]] = {
        "economy": ("none", "minimal", "low", None),
        "balanced": ("medium", "low", "high", None),
        # High is the most portable quality-first level. Vendor-specific levels are
        # selected only when the model matrix above establishes that they exist.
        "performance": ("high", "xhigh", "max", "medium", None),
    }
    return ReasoningControl(mode, thinking, candidates[mode], "adaptive")


def legacy_effort_for_mode(mode: str) -> str:
    """Compatibility value for old clients and previously persisted job schemas."""
    return {"economy": "low", "balanced": "medium", "performance": "high"}[
        normalize_compute_mode(mode)
    ]
