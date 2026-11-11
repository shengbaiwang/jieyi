"""Direct text requests use configured models without touching the book library."""
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from jieyi.api.app import create_app
from jieyi.domain.models import TranslationResult
from jieyi.providers import EchoProvider
from jieyi.settings import LocalSettingsStore, ModelBinding, ProviderSettings, profile_from_preset


@pytest.fixture
def text_client():
    with tempfile.TemporaryDirectory() as directory:
        settings_path = Path(directory) / "settings.json"
        LocalSettingsStore(settings_path).save(ProviderSettings(
            profiles=(profile_from_preset("local", "ollama", selected_models=("test-model",)),),
            draft=ModelBinding("local", "test-model", "economy"),
            term_discovery=ModelBinding("local", "test-model", "economy"),
        ))
        with patch("jieyi.settings.LocalSecretStore._keychain_available", return_value=False):
            app = create_app(str(Path(directory) / "api.db"), str(settings_path))
            app.state.providers.register("profile:local", EchoProvider())
            with TestClient(app) as client:
                yield client


def request(**changes):
    return {"text": "First paragraph.\n\nSecond paragraph. https://example.com `code` [12]",
            "profile_id": "local", "model": "test-model", "source_lang": "auto",
            "target_lang": "zh-CN", "style_guide": "Use a natural literary voice.", **changes}


def test_text_translation_preserves_structure_without_creating_books(text_client):
    result = text_client.post("/translate/text", json=request())
    assert result.status_code == 200, result.text
    payload = result.json()
    assert "\n\nSecond paragraph." in payload["text"]
    assert "https://example.com `code` [12]" in payload["text"]
    assert "JY_PH" not in payload["text"]
    assert "raw_response" not in payload
    assert payload["prompt_tokens"] > 0
    assert text_client.get("/projects").json() == []
    store = text_client.app.state.store
    for table in ("projects", "documents", "segments", "jobs", "candidates"):
        with store._connect() as connection:
            assert connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0


def test_text_translation_passes_languages_style_model_and_compute_mode(text_client):
    calls = []

    class RecordingProvider:
        async def complete(self, messages, model, **options):
            calls.append((messages, model, options))
            return TranslationResult(text="Bonjour le monde.", prompt_tokens=20, completion_tokens=5)

    text_client.app.state.providers.register("profile:local", RecordingProvider())
    response = text_client.post("/translate/text", json=request(
        text="Hello world.", source_lang="en", target_lang="fr", compute_mode="performance",
        model="custom-model",
    ))
    assert response.status_code == 200, response.text
    messages, model, options = calls[0]
    prompt = "\n".join(item["content"] for item in messages)
    assert "Source language: en" in prompt
    assert "Target language: fr" in prompt
    assert "Use a natural literary voice." in prompt
    assert "Hello world." in prompt
    assert options["compute_mode"] == "performance"
    assert model.model == "custom-model"
    assert response.json()["text"] == "Bonjour le monde."


@pytest.mark.parametrize("changes", [
    {"text": "   "}, {"text": "x" * 12_001}, {"profile_id": "missing"},
    {"model": " "}, {"target_lang": " "}, {"compute_mode": "invalid"},
])
def test_text_translation_rejects_invalid_requests(text_client, changes):
    response = text_client.post("/translate/text", json=request(**changes))
    assert response.status_code == 422


def test_missing_placeholder_returns_actionable_error(text_client):
    class BrokenProvider:
        async def complete(self, *_args, **_options):
            return TranslationResult(text="链接被删掉了。")

    text_client.app.state.providers.register("profile:local", BrokenProvider())
    response = text_client.post("/translate/text", json=request(text="Read https://example.com"))
    assert response.status_code == 502
    assert "更换模型" in response.json()["detail"]


def test_upstream_failure_does_not_leak_secrets(text_client):
    class FailingProvider:
        async def complete(self, *_args, **_options):
            raise RuntimeError("upstream secret-api-key")

    text_client.app.state.providers.register("profile:local", FailingProvider())
    response = text_client.post("/translate/text", json=request())
    assert response.status_code == 502
    assert "secret-api-key" not in response.text


def test_truncated_translation_is_not_returned_as_complete(text_client):
    class TruncatedProvider:
        async def complete(self, *_args, **_options):
            return TranslationResult(text="译文不完整", raw_response='{"finish_reason":"length"}')

    text_client.app.state.providers.register("profile:local", TruncatedProvider())
    response = text_client.post("/translate/text", json=request(text="Hello."))
    assert response.status_code == 502
