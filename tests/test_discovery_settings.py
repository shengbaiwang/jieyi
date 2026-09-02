import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from jieyi.api.app import create_app
from jieyi.domain.models import TranslationResult
from jieyi.settings import LocalSettingsStore, ModelBinding, ProviderSettings, profile_from_preset


class DiscoverySettingsTests(unittest.TestCase):
    def test_old_settings_copy_draft_once_then_round_trip_independently(self):
        for version in (2, 3):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "settings.json"
                payload = asdict(ProviderSettings(
                    profiles=(profile_from_preset("local", "ollama"),),
                    draft=ModelBinding("local", "old-draft", "performance"),
                ))
                payload["version"] = version
                del payload["term_discovery"]
                path.write_text(json.dumps(payload))
                store = LocalSettingsStore(path)
                settings = store.load()
                self.assertEqual(settings.term_discovery, settings.draft)
                self.assertIsNot(settings.term_discovery, settings.draft)

                settings.draft.model = "new-draft"
                store.save(settings)
                loaded = store.load()
                self.assertEqual(loaded.version, 4)
                self.assertEqual(loaded.draft.model, "new-draft")
                self.assertEqual(loaded.term_discovery, ModelBinding(
                    "local", "old-draft", "performance"
                ))

    def test_legacy_settings_and_explicit_empty_model(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            path.write_text(json.dumps({"draft_model": "legacy-model"}))
            store = LocalSettingsStore(path)
            settings = store.load()
            self.assertEqual(settings.term_discovery.model, "legacy-model")
            settings.term_discovery.model = ""
            store.save(settings)
            self.assertEqual(store.load().term_discovery.model, "")


class DiscoverySettingsApiTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.keychain = patch("jieyi.settings.LocalSecretStore._keychain_available", return_value=False)
        self.keychain.start()
        self.addCleanup(self.keychain.stop)
        self.path = Path(self.tempdir.name) / "api.settings.json"
        LocalSettingsStore(self.path).save(ProviderSettings(
            profiles=(
                profile_from_preset("draft", "ollama"),
                profile_from_preset("terms", "ollama"),
            ),
            draft=ModelBinding("draft", "draft-model", "economy"),
            term_discovery=ModelBinding("terms", "term-model", "balanced"),
        ))
        self.app = create_app(str(Path(self.tempdir.name) / "api.db"))
        self.client = TestClient(self.app)
        self.addCleanup(self.client.close)

    def test_partial_updates_preserve_other_roles_and_persist(self):
        response = self.client.patch("/settings/provider", json={
            "term_discovery_model": "specialized-terms",
            "term_discovery_compute_mode": "performance",
        })
        self.assertEqual(response.status_code, 200, response.text)
        saved = response.json()
        self.assertEqual(saved["term_discovery_provider"], "profile:terms")
        self.assertEqual(saved["term_discovery_model"], "specialized-terms")
        self.assertEqual(saved["term_discovery_compute_mode"], "performance")
        self.assertEqual(saved["draft_model"], "draft-model")

        response = self.client.patch("/settings/provider", json={
            "draft_model": "changed-draft",
        })
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["term_discovery_model"], "specialized-terms")
        loaded = LocalSettingsStore(self.path).load()
        self.assertEqual(loaded.term_discovery, ModelBinding(
            "terms", "specialized-terms", "performance"
        ))
        self.assertEqual(loaded.draft.model, "changed-draft")

    def test_full_form_and_older_client_preserve_discovery_binding(self):
        payload = self.client.get("/settings/provider").json()
        payload["draft_model"] = "updated-draft"
        response = self.client.patch("/settings/provider", json=payload)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["term_discovery_model"], "term-model")
        for key in list(payload):
            if key.startswith("term_discovery_"):
                del payload[key]
        response = self.client.patch("/settings/provider", json=payload)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["term_discovery_model"], "term-model")

    def test_invalid_connection_does_not_overwrite_saved_settings(self):
        before = self.path.read_bytes()
        response = self.client.patch("/settings/provider", json={
            "term_discovery_profile_id": "missing",
        })
        self.assertEqual(response.status_code, 422)
        self.assertIn("术语发现", response.json()["detail"])
        self.assertEqual(self.path.read_bytes(), before)

    def test_scanning_uses_discovery_provider_model_and_compute_mode(self):
        calls = []

        class AnalysisProvider:
            async def complete(self, messages, model, **kwargs):
                calls.append((model.provider, model.model, kwargs["compute_mode"]))
                cards = json.loads(messages[-1]["content"])["candidates"]
                return TranslationResult(text=json.dumps({"proposals": [{
                    "candidate_id": card["candidate_id"],
                    "keep": False,
                    "evidence_ids": [card["evidence"][0]["evidence_id"]],
                    "rationale": "No uniform translation needed.",
                    "confidence": 0.9,
                } for card in cards]}))

        self.app.state.providers.register("profile:terms", AnalysisProvider())
        project = self.client.post("/projects", json={
            "name": "Terms", "source_lang": "en", "target_lang": "zh-CN",
        }).json()
        document = self.client.post(f"/projects/{project['id']}/documents", json={
            "title": "Agency", "source_format": "txt",
            "text": "Agency is defined as a capacity to act.\n\n"
                    "Agency shapes social power. Social power affects agency.",
        }).json()
        saved = self.client.get("/settings/provider").json()
        response = self.client.post(f"/documents/{document['id']}/term-discovery-runs", json={
            "provider": saved["term_discovery_provider"],
            "model": saved["term_discovery_model"],
            "compute_mode": saved["term_discovery_compute_mode"],
        })
        self.assertEqual(response.status_code, 201, response.text)
        self.assertTrue(calls)
        self.assertEqual(set(calls), {("profile:terms", "term-model", "balanced")})
        self.assertEqual(response.json()["model"], "term-model")

        # Clearing this role remains independent of the configured draft model.
        saved = self.client.patch("/settings/provider", json={"term_discovery_model": ""}).json()
        self.assertEqual(saved["term_discovery_model"], "")
        self.assertEqual(self.client.get("/settings/provider").json()["term_discovery_model"], "")
        count = len(calls)
        response = self.client.post(f"/documents/{document['id']}/term-discovery-runs", json={
            "provider": "", "model": "", "max_model_candidates": 0,
        })
        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual(len(calls), count)
        self.assertEqual(response.json()["model"], "")
