import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from test_epub import build_epub

from jieyi.api.app import create_app


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        app = create_app(str(Path(self.tempdir.name) / "api.db"))
        self.client = TestClient(app)

    def tearDown(self):
        self.client.close()
        self.tempdir.cleanup()

    def test_human_review_queue_combines_required_checks_and_distributed_sampling(self):
        project = self.client.post(
            "/projects",
            json={"name": "Review flow", "source_lang": "en", "target_lang": "zh-CN"},
        ).json()
        document = self.client.post(
            f"/projects/{project['id']}/documents",
            json={
                "title": "Three paragraphs",
                "text": "One paragraph.\n\nTwo paragraphs in 2020.\n\nThree paragraphs.",
                "source_format": "txt",
            },
        ).json()
        draft_job = self.client.post(
            f"/documents/{document['id']}/jobs",
            json={"draft_provider": "echo", "draft_model": "draft", "review_policy": "never"},
        ).json()
        self.assertEqual(self.client.post(f"/jobs/{draft_job['id']}/run").status_code, 200)

        review_job = self.client.post(
            f"/documents/{document['id']}/jobs",
            json={
                "draft_provider": "echo",
                "draft_model": "unused",
                "reviewer_provider": "echo",
                "reviewer_model": "reviewer",
                "task_mode": "review",
                "review_policy": "all",
                "review_sample_rate": 0,
            },
        ).json()
        self.assertEqual(self.client.post(f"/jobs/{review_job['id']}/run").status_code, 200)

        overview = self.client.get(f"/documents/{document['id']}/overview").json()
        self.assertEqual(overview["reviewed_count"], 3)
        queue = self.client.get(
            f"/documents/{document['id']}/human-review-queue",
            params={"sample_rate": 0.34},
        ).json()
        self.assertEqual(len(queue), 2)
        self.assertEqual({item["reason"] for item in queue}, {"sample"})
        self.assertEqual([item["ordinal"] for item in queue], sorted(item["ordinal"] for item in queue))

        confirmed = queue[0]
        response = self.client.patch(
            f"/segments/{confirmed['segment_id']}/confirm",
            json={"translation": confirmed["translation"], "rationale": "Human sample passed"},
        )
        self.assertEqual(response.status_code, 200)
        remaining = self.client.get(
            f"/documents/{document['id']}/human-review-queue",
            params={"sample_rate": 0.34},
        ).json()
        self.assertNotIn(confirmed["segment_id"], {item["segment_id"] for item in remaining})

        segments = self.client.get(f"/documents/{document['id']}/segments").json()
        segment = next(item for item in segments if item["ordinal"] == 1)
        edited = self.client.patch(
            f"/segments/{segment['id']}/draft",
            json={"translation": "人工改写但漏掉了数字。"},
        )
        self.assertEqual(edited.status_code, 200)
        required = self.client.get(
            f"/documents/{document['id']}/human-review-queue",
            params={"sample_rate": 0},
        ).json()
        self.assertEqual(required, [])

    def test_epub_overview_uses_embedded_navigation_boundaries(self):
        project = self.client.post(
            "/projects",
            json={"name": "EPUB TOC", "source_lang": "en", "target_lang": "zh-CN"},
        ).json()
        document = self.client.post(
            f"/projects/{project['id']}/documents/epub",
            content=build_epub(),
            headers={"Content-Type": "application/epub+zip"},
        ).json()

        listed = self.client.get(f"/projects/{project['id']}/documents").json()
        self.assertTrue(listed[0]["cover_url"].endswith("/OPS/cover.png"))
        cover = self.client.get(listed[0]["cover_url"])
        self.assertEqual(cover.status_code, 200)
        self.assertTrue(cover.content.startswith(b"\x89PNG"))
        overview = self.client.get(f"/documents/{document['id']}/overview")
        self.assertEqual(overview.status_code, 200)
        chapters = overview.json()["chapters"]
        self.assertEqual([item["title"] for item in chapters], ["Second from TOC", "First from TOC"])
        self.assertEqual([item["start_ordinal"] for item in chapters], [0, 4])
        self.assertEqual([item["segment_count"] for item in chapters], [4, 2])
        self.assertEqual([item["level"] for item in chapters], [0, 0])

        rendered = self.client.get(
            f"/documents/{document['id']}/epub/spine/0",
            params={"mode": "original", "layout": "faithful"},
        )
        self.assertEqual(rendered.status_code, 200)
        self.assertIn("data-jy-reader-resize", rendered.text)
        self.assertIn("jy-epub-resize", rendered.text)
        self.assertIn("font-size: 18px", rendered.text)
        csp = rendered.headers["content-security-policy"]
        self.assertIn("script-src 'sha256-", csp)
        self.assertIn("sandbox allow-scripts", csp)

    def test_dry_run_translation_and_human_confirmation(self):
        health = self.client.get("/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["api_version"], 2)
        self.assertEqual(health.json()["quality_detector_version"], "5")
        self.assertTrue(health.json()["db_path"].endswith("api.db"))
        self.assertIn("echo", health.json()["providers"])

        project = self.client.post(
            "/projects",
            json={"name": "Theory", "source_lang": "en", "target_lang": "zh-CN"},
        ).json()
        projects = self.client.get("/projects")
        self.assertEqual(projects.status_code, 200)
        self.assertEqual(projects.json()[0]["id"], project["id"])
        document_response = self.client.post(
            f"/projects/{project['id']}/documents",
            json={"title": "Chapter", "text": "Agency changed in 2020.", "source_format": "txt"},
        )
        self.assertEqual(document_response.status_code, 201)
        document = document_response.json()

        overview = self.client.get(f"/documents/{document['id']}/overview")
        self.assertEqual(overview.status_code, 200)
        self.assertEqual(overview.json()["segment_count"], 1)
        page = self.client.get(f"/documents/{document['id']}/segments/page?limit=1")
        self.assertEqual(page.status_code, 200)
        self.assertEqual(page.json()["total"], 1)
        self.assertEqual(page.json()["items"][0]["source_text"], "Agency changed in 2020.")
        search = self.client.get(f"/documents/{document['id']}/search", params={"q": "Agency"})
        self.assertEqual(search.status_code, 200)
        self.assertEqual(search.json()[0]["ordinal"], 0)

        term_response = self.client.post(
            f"/projects/{project['id']}/terms",
            json={
                "source": "Agency",
                "target": "能动性",
                "status": "approved",
                "aliases": ["agentic capacity"],
                "context_keywords": ["changed"],
                "sense": "行动能力",
                "disambiguation": "不是政府机构",
            },
        )
        self.assertEqual(term_response.status_code, 201)
        terms = self.client.get(f"/projects/{project['id']}/terms")
        self.assertEqual(terms.status_code, 200)
        self.assertEqual(terms.json()[0]["target"], "能动性")
        self.assertEqual(terms.json()[0]["aliases"], ["agentic capacity"])
        self.assertEqual(terms.json()[0]["sense"], "行动能力")

        job_response = self.client.post(
            f"/documents/{document['id']}/jobs",
            json={
                "draft_provider": "echo",
                "draft_model": "dry-run",
                "review_policy": "never",
            },
        )
        self.assertEqual(job_response.status_code, 201)
        job = job_response.json()
        jobs = self.client.get(f"/documents/{document['id']}/jobs")
        self.assertEqual(jobs.status_code, 200)
        self.assertEqual(jobs.json()[0]["id"], job["id"])
        segments_before_run = self.client.get(
            f"/documents/{document['id']}/segments"
        ).json()
        preview = self.client.get(
            f"/jobs/{job['id']}/segments/{segments_before_run[0]['id']}/prompt-preview"
        )
        self.assertEqual(preview.status_code, 200)
        self.assertIn("APPROVED TERMINOLOGY", preview.json()["messages"][0]["content"])
        self.assertIn("placeholder token", preview.json()["messages"][0]["content"])

        run = self.client.post(f"/jobs/{job['id']}/run")
        self.assertEqual(run.status_code, 200)
        self.assertEqual(run.json()["status"], "completed")

        segments = self.client.get(f"/documents/{document['id']}/segments").json()
        self.assertEqual(segments[0]["status"], "machine_translated")
        draft = self.client.patch(
            f"/segments/{segments[0]['id']}/draft",
            json={"translation": "人工编辑中的草稿。"},
        )
        self.assertEqual(draft.status_code, 200)
        self.assertEqual(draft.json()["edited_translation"], "人工编辑中的草稿。")
        candidates = self.client.get(f"/segments/{segments[0]['id']}/candidates")
        self.assertEqual(candidates.status_code, 200)
        self.assertEqual(len(candidates.json()), 1)
        confirm = self.client.patch(
            f"/segments/{segments[0]['id']}/confirm",
            json={"translation": "能动性在 2020 年发生了变化。", "rationale": "Manual review"},
        )
        self.assertEqual(confirm.status_code, 200)
        self.assertEqual(confirm.json()["status"], "human_confirmed")
        current_issues = self.client.get(f"/documents/{document['id']}/issues")
        self.assertEqual(current_issues.status_code, 200)
        self.assertEqual(current_issues.json(), [])

        tm = self.client.get(f"/projects/{project['id']}/translation-memory")
        self.assertEqual(tm.status_code, 200)
        self.assertEqual(tm.json()[0]["target_text"], "能动性在 2020 年发生了变化。")

        exported = self.client.get(f"/documents/{document['id']}/export")
        self.assertIn("能动性在 2020 年发生了变化。", exported.text)

        style = self.client.patch(
            f"/projects/{project['id']}/style",
            json={"style_guide": "采用清晰自然的通俗表达。"},
        )
        self.assertEqual(style.status_code, 200)
        self.assertEqual(style.json()["style_guide"], "采用清晰自然的通俗表达。")
        self.assertEqual(
            self.client.get("/projects").json()[0]["style_guide"],
            "采用清晰自然的通俗表达。",
        )

    def test_document_can_be_deleted_with_its_book_data(self):
        project = self.client.post(
            "/projects",
            json={"name": "Disposable", "source_lang": "en", "target_lang": "zh-CN"},
        ).json()
        document = self.client.post(
            f"/projects/{project['id']}/documents",
            json={"title": "Delete me", "text": "One paragraph.", "source_format": "txt"},
        ).json()

        deleted = self.client.delete(f"/documents/{document['id']}")
        self.assertEqual(deleted.status_code, 204)
        self.assertEqual(self.client.get(f"/projects/{project['id']}/documents").json(), [])
        self.assertEqual(self.client.get(f"/documents/{document['id']}/segments").status_code, 404)
        self.assertEqual(self.client.delete(f"/documents/{document['id']}").status_code, 404)

    def test_provider_settings_can_be_managed_without_exposing_a_key(self):
        initial = self.client.get("/settings/provider")
        self.assertEqual(initial.status_code, 200)
        self.assertNotIn("api_key", initial.json())

        updated = self.client.patch(
            "/settings/provider",
            json={
                "provider_type": "ollama",
                "base_url": "http://127.0.0.1:11434/v1",
                "draft_model": "local-model",
                "reviewer_model": "",
                "review_enabled": False,
            },
        )
        self.assertEqual(updated.status_code, 200)
        self.assertEqual(updated.json()["provider_type"], "ollama")
        self.assertEqual(updated.json()["draft_model"], "local-model")
        self.assertNotIn("api_key", updated.json())

        health = self.client.get("/health")
        self.assertIn("openai-compatible", health.json()["providers"])

    def test_draft_and_reviewer_can_use_independent_provider_profiles(self):
        updated = self.client.patch(
            "/settings/provider",
            json={
                "version": 2,
                "profiles": [
                    {
                        "id": "kimi",
                        "name": "Kimi 草译",
                        "provider_type": "kimi-coding",
                        "base_url": "https://api.kimi.com/coding/v1",
                    },
                    {
                        "id": "glm",
                        "name": "GLM 审校",
                        "provider_type": "glm-cn",
                        "base_url": "https://open.bigmodel.cn/api/paas/v4",
                    },
                ],
                "draft_profile_id": "kimi",
                "draft_model": "k3",
                "draft_reasoning_effort": "low",
                "reviewer_profile_id": "glm",
                "reviewer_model": "glm-5.3-flash",
                "reviewer_reasoning_effort": "xhigh",
                "review_enabled": True,
            },
        )

        self.assertEqual(updated.status_code, 200)
        payload = updated.json()
        self.assertEqual(payload["draft_provider"], "profile:kimi")
        self.assertEqual(payload["reviewer_provider"], "profile:glm")
        self.assertEqual(payload["draft_compute_mode"], "economy")
        self.assertEqual(payload["reviewer_compute_mode"], "performance")
        self.assertEqual(payload["draft_reasoning_effort"], "low")
        self.assertEqual(payload["reviewer_reasoning_effort"], "high")
        self.assertEqual(len(payload["profiles"]), 2)
        self.assertNotIn("api_key", payload["profiles"][0])
        providers = self.client.get("/health").json()["providers"]
        self.assertIn("profile:kimi", providers)
        self.assertIn("profile:glm", providers)

    def test_epub_can_be_uploaded_as_raw_request_body(self):
        project = self.client.post(
            "/projects",
            json={"name": "EPUB", "source_lang": "en", "target_lang": "zh-CN"},
        ).json()
        response = self.client.post(
            f"/projects/{project['id']}/documents/epub",
            content=build_epub(),
            headers={"Content-Type": "application/epub+zip"},
        )
        self.assertEqual(response.status_code, 201)
        document = response.json()
        self.assertEqual(document["title"], "Test Theory Book")
        self.assertEqual(document["source_format"], "epub")
        segments = self.client.get(f"/documents/{document['id']}/segments").json()
        self.assertEqual(segments[0]["source_text"], "Second Chapter")

    def test_epub_can_be_inspected_before_import(self):
        response = self.client.post(
            "/imports/epub/inspect",
            content=build_epub(),
            headers={"Content-Type": "application/epub+zip"},
        )
        self.assertEqual(response.status_code, 200)
        inspection = response.json()
        self.assertEqual(inspection["title"], "Test Theory Book")
        self.assertGreater(inspection["block_count"], 0)
        self.assertGreater(inspection["chapter_count"], 0)
        self.assertEqual(inspection["preview"][0]["text"], "Second Chapter")

    def test_same_source_can_define_multiple_contextual_senses(self):
        project = self.client.post(
            "/projects",
            json={"name": "Polysemy", "source_lang": "en", "target_lang": "zh-CN"},
        ).json()
        financial = {
            "source": "bank",
            "target": "银行",
            "status": "approved",
            "sense": "金融机构",
            "context_keywords": ["loan", "credit"],
        }
        river = {
            "source": "bank",
            "target": "河岸",
            "status": "approved",
            "sense": "河流边缘",
            "context_keywords": ["river", "flood"],
        }
        first = self.client.post(f"/projects/{project['id']}/terms", json=financial)
        second = self.client.post(f"/projects/{project['id']}/terms", json=river)
        duplicate = self.client.post(f"/projects/{project['id']}/terms", json=financial)

        self.assertEqual(first.status_code, 201)
        self.assertEqual(second.status_code, 201)
        self.assertEqual(duplicate.status_code, 409)
        terms = self.client.get(f"/projects/{project['id']}/terms").json()
        self.assertEqual({item["target"] for item in terms}, {"银行", "河岸"})


if __name__ == "__main__":
    unittest.main()
