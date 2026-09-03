import asyncio
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from fastapi.testclient import TestClient

from jieyi.api.app import create_app
from jieyi.domain.models import JobStatus, TermEntry, TermStatus, TranslationResult
from jieyi.prompting import build_messages
from jieyi.providers import ProviderRegistry
from jieyi.quality import run_deterministic_checks
from jieyi.quality.terminology_review import TerminologyReviewRepository, review_items
from jieyi.terminology import matching_terms, render_terminology_constraints, resolve_terminology
from jieyi.workflow import TranslationEngine, create_document, create_job, create_project


class ReferenceTranslationProvider:
    def __init__(self):
        self.messages = []

    async def translate(self, request, model):
        return await self.complete(build_messages(request), model)

    async def complete(self, messages, model, **kwargs):
        self.messages.append(messages)
        return TranslationResult(text="主体可以自主行动。", prompt_tokens=100, completion_tokens=10)


class ReferenceTermTests(unittest.TestCase):
    def setUp(self):
        self.term = TermEntry(
            id="reference", project_id="project", source="agency", target="能动性",
            status=TermStatus.APPROVED, enforcement="reference", aliases=("agentic capacity",),
            sense="行动能力", disambiguation="根据上下文灵活处理", context_keywords=("personal",),
            rationale="供译者参考", forbidden_targets=("自主行动",),
        )

    def test_references_match_aliases_and_keep_guidance_without_consistency_findings(self):
        source = "Personal agentic capacity matters."
        resolution = resolve_terminology(source, [self.term])
        self.assertEqual(resolution.matched_terms, (self.term,))
        self.assertEqual(resolution.references[0].matched_forms, ("agentic capacity",))
        self.assertFalse(resolution.enforced)
        self.assertFalse(resolution.ambiguous)
        self.assertEqual(run_deterministic_checks(source, "自主行动很重要。", [self.term]), [])
        prompt = render_terminology_constraints(source, [self.term])
        for text in ("[REFERENCE]", "OPTIONAL GUIDANCE", self.term.sense,
                     self.term.disambiguation, self.term.rationale, "keyword hints: personal"):
            self.assertIn(text, prompt)
        self.assertNotIn("[MANDATORY]", prompt)
        self.assertNotIn("[CONDITIONAL]", prompt)
        self.assertNotIn("forbidden", prompt)
        self.assertNotIn("never use", prompt)
        self.assertEqual(
            [issue.code for issue in run_deterministic_checks(source + " [^1]", "译文", [self.term])],
            ["footnote_mismatch"],
        )

    def test_only_approved_and_matching_references_enter_context(self):
        self.assertEqual(matching_terms("Unrelated text.", [self.term]), [])
        for status in (TermStatus.PROPOSED, TermStatus.DEPRECATED, TermStatus.FORBIDDEN):
            self.assertEqual(matching_terms("Agency matters.", [replace(self.term, status=status)]), [])

    def test_overlapping_references_never_weaken_mandatory_or_conditional_terms(self):
        mandatory = replace(
            self.term, id="mandatory", enforcement="global", target="行动力", forbidden_targets=(),
        )
        for source, reference_source in (
            ("Personal agency matters.", "personal agency"),
            ("Agency matters.", "agency"),
        ):
            for enforcement in ("global", "contextual"):
                with self.subTest(reference_source=reference_source, enforcement=enforcement):
                    binding = replace(mandatory, enforcement=enforcement)
                    reference = replace(self.term, source=reference_source)
                    resolution = resolve_terminology(source, [reference, binding])
                    self.assertEqual(len(resolution.references), 1)
                    issues = run_deterministic_checks(source, "主体可以自主行动。", [reference, binding])
                    self.assertEqual(
                        [issue.code for issue in issues],
                        ["approved_term_missing" if enforcement == "global" else "terminology_pending"],
                    )
                    if enforcement == "global":
                        self.assertEqual(resolution.enforced[0].term.id, "mandatory")
                        self.assertFalse(resolution.ambiguous)
                    else:
                        self.assertEqual(resolution.ambiguous[0].candidates[0].term.id, "mandatory")
                        self.assertEqual(len(resolution.ambiguous[0].candidates), 1)

    def test_longer_mandatory_phrase_still_owns_its_span_with_references_present(self):
        phrase = replace(self.term, id="phrase", source="personal agency", enforcement="global")
        short = replace(self.term, id="short", enforcement="global")
        resolution = resolve_terminology("Personal agency matters.", [short, self.term, phrase])
        self.assertEqual([item.term.id for item in resolution.enforced], ["phrase"])
        self.assertEqual([item.term.id for item in resolution.references], ["reference"])


class ReferenceTermIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.app = create_app(str(Path(self.temp.name) / "terms.db"))
        self.client = TestClient(self.app)
        self.store = self.app.state.store
        self.project = create_project(self.store, name="Reference terms", source_lang="en", target_lang="zh")
        self.document = create_document(
            self.store, project_id=self.project.id, title="Chapter",
            text="Personal agency matters.", source_format="txt",
        )
        self.segment = self.store.list_segments(self.document.id)[0]
        self.terms_url = f"/projects/{self.project.id}/terms"
        response = self.client.post(self.terms_url, json={
            "source": "agency", "target": "能动性", "enforcement": "reference",
            "aliases": ["agentic capacity"], "sense": "行动能力", "context_keywords": ["personal"],
            "disambiguation": "根据语境选择", "forbidden_targets": ["自主行动"],
        })
        self.assertEqual(response.status_code, 201)
        self.term_json = response.json()
        self.term_url = f"{self.terms_url}/{self.term_json['id']}"
        self.job = create_job(
            self.store, document_id=self.document.id, draft_provider="test", draft_model="stub",
            tm_enabled=False,
        )

    def tearDown(self):
        self.client.close()
        self.temp.cleanup()

    def test_reference_reaches_both_translation_runners_and_does_not_flag_alternate_wording(self):
        provider = ReferenceTranslationProvider()
        registry = ProviderRegistry()
        registry.register("test", provider)
        engine = TranslationEngine(self.store, registry)
        preview = engine.preview(self.job.id, self.segment.id)
        self.assertEqual(preview["relevant_terms"][0]["enforcement"], "reference")
        for runner in (engine.run, engine.run_optimized):
            with self.subTest(runner=runner.__name__):
                document = create_document(
                    self.store, project_id=self.project.id, title=runner.__name__,
                    text="Personal agency matters.", source_format="txt",
                )
                segment = self.store.list_segments(document.id)[0]
                job = create_job(
                    self.store, document_id=document.id, draft_provider="test", draft_model="stub",
                    tm_enabled=False,
                )
                provider.messages.clear()
                completed = asyncio.run(runner(job.id))
                self.assertEqual(completed.status, JobStatus.COMPLETED)
                self.assertTrue(provider.messages)
                messages = "\n".join(item["content"] for item in provider.messages[-1])
                self.assertIn("agency | agentic capacity -> 能动性 [REFERENCE]", messages)
                self.assertIn("Reference terminology is optional guidance", messages)
                self.assertNotIn("Treat approved terminology as binding", messages)
                self.assertEqual(self.store.get_segment(segment.id).machine_translation, "主体可以自主行动。")
                self.assertEqual(self.client.get(f"/documents/{document.id}/issues").json(), [])
                self.assertEqual(TerminologyReviewRepository(self.store).snapshot(document.id), [])

    def test_switching_enforcement_persists_and_reindexes_existing_translation(self):
        self.store.set_machine_translation(self.segment.id, "主体可以自主行动。")
        issues_url = f"/documents/{self.document.id}/issues"
        queue_url = f"/documents/{self.document.id}/human-review-queue"
        for mode, expected in (
            ("global", {"approved_term_missing", "forbidden_term_used"}),
            ("reference", set()),
            ("contextual", {"terminology_pending"}),
            ("reference", set()),
            ("global", {"approved_term_missing", "forbidden_term_used"}),
        ):
            with self.subTest(mode=mode):
                result = self.client.patch(self.term_url, json={"enforcement": mode})
                self.assertEqual(result.status_code, 200)
                self.assertEqual(result.json(), self.term_json | {"enforcement": mode})
                self.assertEqual(self.client.get(self.terms_url).json()[0], result.json())
                self.assertEqual({item["code"] for item in self.client.get(issues_url).json()}, expected)
                self.assertEqual(bool(self.client.get(queue_url).json()), mode == "global")
                items = TerminologyReviewRepository(self.store).snapshot(self.document.id)
                self.assertEqual(bool(items), mode == "contextual")
        events = self.store.list_audit_events("term", self.term_json["id"])
        self.assertEqual(events[-1]["action"], "enforcement_updated")

    def test_update_validates_mode_and_project_ownership_without_mutating_term(self):
        self.assertEqual(self.client.patch(self.term_url, json={"enforcement": "invalid"}).status_code, 422)
        self.assertEqual(self.client.patch(self.term_url, json={}).status_code, 422)
        other = create_project(self.store, name="Other", source_lang="en", target_lang="zh")
        self.assertEqual(self.client.patch(
            f"/projects/{other.id}/terms/{self.term_json['id']}", json={"enforcement": "global"},
        ).status_code, 404)
        self.assertEqual(self.client.patch(
            f"{self.terms_url}/missing", json={"enforcement": "global"},
        ).status_code, 404)
        self.assertEqual(self.client.get(self.terms_url).json()[0], self.term_json)
        default = self.client.post(self.terms_url, json={"source": "book", "target": "书"})
        self.assertEqual(default.json()["enforcement"], "auto")

    def test_reference_does_not_invalidate_cached_checks_for_binding_terms(self):
        reference = self.store.list_terms(self.project.id)[0]
        conditional = replace(reference, id="conditional", enforcement="contextual")
        without_reference = review_items(self.segment, [conditional], self.project)
        with_reference = review_items(self.segment, [conditional, reference], self.project)
        self.assertEqual(len(without_reference), 1)
        self.assertEqual(with_reference, without_reference)
        self.assertEqual(review_items(self.segment, [reference], self.project), [])
