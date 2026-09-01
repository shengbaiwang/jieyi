from __future__ import annotations

from typing import Protocol

from .models import ModelSpec, TranslationRequest, TranslationResult


class TranslationProvider(Protocol):
    """A narrow provider port. Vendor SDK details stay outside the workflow."""

    async def translate(
        self, request: TranslationRequest, model: ModelSpec
    ) -> TranslationResult: ...

