from __future__ import annotations

from jieyi.domain.ports import TranslationProvider


class ProviderRegistry:
    def __init__(self):
        self._providers: dict[str, TranslationProvider] = {}

    def register(self, name: str, provider: TranslationProvider) -> None:
        if not name.strip():
            raise ValueError("Provider name cannot be empty")
        self._providers[name] = provider

    def unregister(self, name: str) -> None:
        self._providers.pop(name, None)

    def unregister_matching(self, prefix: str) -> None:
        for name in tuple(self._providers):
            if name.startswith(prefix):
                self._providers.pop(name, None)

    def get(self, name: str) -> TranslationProvider:
        try:
            return self._providers[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._providers)) or "none"
            raise KeyError(f"Unknown provider '{name}'. Available: {available}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._providers))
