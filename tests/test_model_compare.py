import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from jieyi.api.app import create_app


class ModelCompareApiTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        app = create_app(str(Path(self.tempdir.name) / "compare.db"))
        self.client = TestClient(app)

    def tearDown(self):
        self.client.close()
        self.tempdir.cleanup()

    def _seed_segment(self):
        project = self.client.post(
            "/projects",
            json={"name": "Compare", "source_lang": "en", "target_lang": "zh-CN"},
        ).json()
        document = self.client.post(
            f"/projects/{project['id']}/documents",
            json={
                "title": "Compare Book",
                "text": "Agency matters. Still.\n\nSecond paragraph for context.",
                "source_format": "txt",
            },
        ).json()
        segment = self.client.get(f"/documents/{document['id']}/segments").json()[0]
        return project, document, segment

    def test_compare_runs_models_without_overwriting_translations(self):
        _, document, segment = self._seed_segment()
        response = self.client.post(
            f"/segments/{segment['id']}/model-compare",
            json={
                "models": [
                    {"provider": "echo", "model": "dry-run-a"},
                    {"provider": "echo", "model": "dry-run-b"},
                ],
                "compute_mode": "economy",
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["segment_id"], segment["id"])
        self.assertEqual(len(payload["results"]), 2)
        models = {item["model"] for item in payload["results"]}
        self.assertEqual(models, {"dry-run-a", "dry-run-b"})
        for item in payload["results"]:
            self.assertTrue(item["ok"])
            self.assertTrue(item["text"].startswith("translated:") or "Agency matters" in item["text"])

        candidates = self.client.get(f"/segments/{segment['id']}/candidates").json()
        self.assertEqual(len(candidates), 2)
        self.assertEqual({item["stage"] for item in candidates}, {"compare"})

        reloaded = self.client.get(
            f"/documents/{document['id']}/segments/page?offset=0&limit=1"
        ).json()["items"][0]
        self.assertIsNone(reloaded["machine_translation"])
        self.assertEqual(reloaded["status"], "source")

    def test_compare_rejects_unknown_provider(self):
        _, _, segment = self._seed_segment()
        response = self.client.post(
            f"/segments/{segment['id']}/model-compare",
            json={"models": [{"provider": "missing", "model": "x"}]},
        )
        self.assertEqual(response.status_code, 422)

    def test_compare_rejects_duplicate_models(self):
        _, _, segment = self._seed_segment()
        response = self.client.post(
            f"/segments/{segment['id']}/model-compare",
            json={
                "models": [
                    {"provider": "echo", "model": "same"},
                    {"provider": "echo", "model": "same"},
                ]
            },
        )
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
