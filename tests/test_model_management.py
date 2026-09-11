"""Model library persistence and migration without real provider requests or keys."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from jieyi.api.app import create_app
from jieyi.settings import LocalSettingsStore, ModelBinding, ProviderSettings, profile_from_preset
from jieyi.settings import test_openai_compatible_connection as check_connection


class ModelManagementTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.keychain = patch("jieyi.settings.LocalSecretStore._keychain_available", return_value=False)
        self.keychain.start()
        self.addCleanup(self.keychain.stop)
        self.path = Path(self.directory.name) / "models.settings.json"
        self.store = LocalSettingsStore(self.path)
        self.store.save(ProviderSettings(
            profiles=(
                profile_from_preset("first", "ollama", selected_models=("draft-a", "other-a")),
                profile_from_preset("second", "ollama", selected_models=("terms-b",)),
            ),
            draft=ModelBinding("first", "draft-a"),
            term_discovery=ModelBinding("second", "terms-b"),
        ))
        self.client = TestClient(create_app(str(Path(self.directory.name) / "models.db")))
        self.addCleanup(self.client.close)

    def test_model_library_is_normalized_persisted_and_scoped_to_connection(self):
        payload = self.client.get("/settings/provider").json()
        payload["profiles"][0]["selected_models"] = [" draft-a ", "new-model", "new-model", " "]
        result = self.client.patch("/settings/provider", json=payload)
        self.assertEqual(result.status_code, 200, result.text)
        self.assertEqual(result.json()["profiles"][0]["selected_models"], ["draft-a", "new-model"])
        loaded = self.store.load()
        self.assertEqual(loaded.profiles[0].selected_models, ("draft-a", "new-model"))
        self.assertEqual(loaded.profiles[1].selected_models, ("terms-b",))
        self.assertEqual(loaded.draft.model, "draft-a")
        self.assertEqual(loaded.term_discovery.model, "terms-b")
        self.assertNotIn('"api_key"', self.path.read_text())

    def test_older_client_does_not_erase_model_library(self):
        payload = self.client.get("/settings/provider").json()
        for profile in payload["profiles"]:
            del profile["selected_models"]
        payload["profiles"][0]["name"] = "Renamed service"
        result = self.client.patch("/settings/provider", json=payload)
        self.assertEqual(result.status_code, 200, result.text)
        self.assertEqual(self.store.load().profiles[0].selected_models, ("draft-a", "other-a"))
        self.assertEqual(self.store.load().profiles[0].name, "Renamed service")

    def test_explicit_empty_library_remains_empty_after_reload(self):
        payload = self.client.get("/settings/provider").json()
        payload["profiles"][0]["selected_models"] = []
        response = self.client.patch("/settings/provider", json=payload)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.store.load().profiles[0].selected_models, ())
        self.assertEqual(self.client.get("/settings/provider").json()["profiles"][0]["selected_models"], [])

    def test_legacy_file_keeps_bindings_without_inventing_added_models(self):
        payload = json.loads(self.path.read_text())
        for profile in payload["profiles"]:
            del profile["selected_models"]
        self.path.write_text(json.dumps(payload))
        loaded = self.store.load()
        self.assertIsNone(loaded.profiles[0].selected_models)
        self.assertEqual(loaded.draft.model, "draft-a")
        self.assertEqual(loaded.term_discovery.profile_id, "second")

    def test_model_discovery_is_not_silently_truncated_at_100(self):
        models = [{"id": f"model-{index}"} for index in range(150)]
        models.extend([{}, {"id": None}, {"id": " "}, {"id": " model-149 "}])
        with patch("urllib.request.urlopen") as request:
            request.return_value.__enter__.return_value.read.return_value = json.dumps({"data": models}).encode()
            result = check_connection("http://localhost:11434/v1", "")
        self.assertEqual(len(result["models"]), 150)
        self.assertEqual(result["models"][-1], "model-149")
