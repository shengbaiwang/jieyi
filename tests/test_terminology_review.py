import asyncio
import json
import sqlite3
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from fastapi.testclient import TestClient

from jieyi.api.app import create_app
from jieyi.domain.models import (
    IssueSeverity,
    ModelSpec,
    QualityIssue,
    TermEntry,
    TermStatus,
    TranslationResult,
)
from jieyi.persistence.sqlite import SQLiteStore
from jieyi.providers import ProviderRegistry
from jieyi.quality import refresh_segment_quality, reindex_all_quality, run_deterministic_checks
from jieyi.quality.service import document_issues
from jieyi.quality.terminology_review import (
    TerminologyReviewManager,
    review_items,
    validate_verdict,
)
from jieyi.terminology import render_terminology_constraints, resolve_terminology, term_spans
from jieyi.workflow.services import create_document, create_job, create_project

# Small human-labelled regression corpus: not model-generated ground truth.
SENSE_CASES = [
    ("Life in an original State of Nature was a state of war.", "原初的自然状态是一种战争状态。"),
    ("Governments discuss the state of nature.", "各国政府讨论自然状态。"),
    ("They lived in Washington State.", "他们住在华盛顿州。"),
    ("It was a city-state in the Mexican state of Puebla.", "那是墨西哥普埃布拉州的一座城邦。"),
    ("To state matters clearly.", "把问题陈述清楚。"),
]


def verdict(item, status="not_applicable", quote="", reason="此处是状态义，不是政治实体。"):
    start = item["translation"].find(quote) if quote else 0
    return {
        "id": item["id"],
        "status": status,
        "reason": reason,
        "source_quote": item["source"][item["start"] : item["end"]],
        "source_start": item["start"],
        "source_end": item["end"],
        "target_quote": quote,
        "target_start": start,
        "target_end": start + len(quote),
        "omission": False,
    }


class ScriptedProvider:
    def __init__(self, decide=verdict):
        self.decide = decide
        self.calls = 0

    async def complete(self, messages, model, **kwargs):
        self.calls += 1
        items = json.loads(messages[-1]["content"])
        return TranslationResult(
            text=json.dumps({"verdicts": [self.decide(item) for item in items]}),
            prompt_tokens=120,
            completion_tokens=80,
            reasoning_tokens=10,
            cost_usd=0.001,
        )


class TerminologyReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.app = create_app(str(Path(self.temp.name) / "qa.db"))
        self.client = TestClient(self.app)
        self.store = self.app.state.store
        self.project = create_project(
            self.store, name="Polysemy", source_lang="en", target_lang="zh"
        )
        self.term = TermEntry(
            "state",
            self.project.id,
            "state",
            "国家",
            status=TermStatus.APPROVED,
            sense="a sovereign political organization",
            disambiguation="Not a condition, province, or verb.",
        )
        self.store.add_term(self.term)
        self.registry = ProviderRegistry()
        self.provider = ScriptedProvider()
        self.registry.register("test", self.provider)
        self.manager = TerminologyReviewManager(self.store, self.registry)
        self.repo = self.manager.repository
        self.model = ModelSpec("test", "qa-v1", 0)

    def tearDown(self):
        self.client.close()
        self.temp.cleanup()

    def book(self, pairs):
        document = create_document(
            self.store,
            project_id=self.project.id,
            title="Regression cases",
            text="\n\n".join(source for source, _ in pairs),
            source_format="txt",
        )
        segments = self.store.list_segments(document.id)
        job = create_job(
            self.store,
            document_id=document.id,
            draft_provider="echo",
            draft_model="draft",
        )
        for segment, (_, target) in zip(segments, pairs, strict=True):
            self.store.set_machine_translation(segment.id, target)
            refresh_segment_quality(self.store, segment.id, job_id=job.id)
        return document, self.store.list_segments(document.id), job

    def run_review(self, document, budget=200_000):
        run = self.repo.create_run(document.id, self.model, "balanced", budget)
        asyncio.run(self.manager.run(run, self.provider))
        return self.repo.runs(document.id)[0]

    def test_actual_polysemy_cases_never_receive_a_global_constraint(self):
        for source, target in SENSE_CASES:
            with self.subTest(source=source):
                findings = run_deterministic_checks(source, target, [self.term])
                self.assertTrue(findings)
                self.assertTrue(all(item.severity is IssueSeverity.INFO for item in findings))
                prompt = render_terminology_constraints(source, [self.term])
                self.assertIn("Not a condition, province, or verb.", prompt)
                self.assertIn("NOT APPLICABLE", prompt)
                self.assertNotIn("[MANDATORY]", prompt)
        with_hint = replace(self.term, context_keywords=("Governments",))
        self.assertFalse(resolve_terminology(SENSE_CASES[1][0], [with_hint]).enforced)

    def test_global_fixed_terms_and_longer_phrases_keep_strict_checks(self):
        city = replace(
            self.term,
            id="city",
            source="city-state",
            target="城邦",
            enforcement="global",
            sense="",
            disambiguation="",
        )
        plain = replace(self.term, enforcement="global")
        source = "A city-state became a state."
        resolution = resolve_terminology(source, [city, plain])
        state_hit = next(item for item in resolution.enforced if item.term.id == "state")
        self.assertEqual([hit.start for hit in state_hit.occurrences], [22])
        findings = run_deterministic_checks(source, "城邦变成了一个状态。", [city, plain])
        self.assertEqual([item.code for item in findings], ["approved_term_missing"])

    def test_original_offsets_survive_casefold_and_unicode(self):
        text = "Straße ﬂow ＡＩ café"
        for term, expected in [("STRASSE", "Straße"), ("flow", "ﬂow"), ("AI", "ＡＩ")]:
            spans = term_spans(text, term)
            self.assertEqual(len(spans), 1)
            self.assertEqual(text[slice(*spans[0])], expected)
        self.assertFalse(term_spans("café", "caf"))
        self.assertFalse(term_spans("interagency", "agency"))

    def test_pending_is_visible_but_not_human_required_or_a_clean_sample(self):
        document, segments, _ = self.book(SENSE_CASES[:1])
        self.store.set_reviewed_translation(segments[0].id, segments[0].machine_translation)
        findings = self.client.get(f"/documents/{document.id}/issues").json()
        self.assertEqual([item["code"] for item in findings], ["terminology_pending"])
        queue = self.client.get(f"/documents/{document.id}/human-review-queue?sample_rate=1").json()
        self.assertEqual(queue, [])
        summary = self.client.get(f"/documents/{document.id}/terminology-review").json()
        self.assertEqual(summary["pending_segments"], 1)
        self.assertEqual(summary["pending"], 2)

    def test_positive_error_and_correct_target_elsewhere_use_occurrence_evidence(self):
        document, _, _ = self.book(
            [
                (
                    "The state taxes citizens; their state of mind varies.",
                    "这个状态向公民征税；他们的国家各异。",
                ),
            ]
        )
        self.provider.decide = lambda item: (
            verdict(item, "inconsistent", "状态", "此处是征税的政治实体，应译为国家。")
            if item["start"] == 4
            else verdict(item)
        )
        run = self.run_review(document)
        self.assertEqual(run["status"], "completed")
        issues = document_issues(self.store, document.id)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0]["code"], "terminology_inconsistent")
        self.assertEqual(issues[0]["severity"], "warning")
        self.assertEqual(issues[0]["details"]["target_quote"], "状态")
        queue = self.client.get(f"/documents/{document.id}/human-review-queue").json()
        self.assertEqual(queue[0]["reason"], "warning")

    def test_consistent_result_requires_real_approved_target_evidence(self):
        _, segments, _ = self.book([("The state taxes citizens.", "国家征税。")])
        item = review_items(segments[0], [self.term], self.project)[0]
        good = verdict(item, "consistent", "国家", "政治实体译法正确。")
        self.assertIsNotNone(validate_verdict(item, good))
        self.assertIsNone(validate_verdict(item, good | {"target_quote": "不存在"}))
        self.assertIsNone(validate_verdict(item, good | {"source_start": 0}))
        self.assertIsNone(validate_verdict(item, good | {"id": "invented"}))
        self.assertIsNone(validate_verdict(item, verdict(item, "consistent")))

    def test_not_applicable_clears_pending_without_changing_any_translation(self):
        document, segments, _ = self.book(SENSE_CASES)
        before = [as_tuple(segment) for segment in segments]
        run = self.run_review(document)
        self.assertEqual(run["status"], "completed")
        self.assertEqual(document_issues(self.store, document.id), [])
        self.assertEqual(
            before, [as_tuple(segment) for segment in self.store.list_segments(document.id)]
        )
        self.assertEqual(self.repo.summary(document.id)["pending"], 0)
        calls = self.provider.calls
        self.run_review(document)
        self.assertEqual(self.provider.calls, calls)

    def test_source_target_and_glossary_changes_invalidate_cache(self):
        document, segments, _ = self.book(SENSE_CASES[:1])
        self.run_review(document)
        self.store.save_segment_draft(segments[0].id, "修改后的自然状态和战争状态。")
        self.assertEqual(self.repo.summary(document.id)["pending"], 2)
        self.run_review(document)
        with self.store._connect() as connection:
            connection.execute(
                "UPDATE segments SET source_text=source_text || ' More context.' WHERE id=?",
                (segments[0].id,),
            )
        self.assertEqual(self.repo.summary(document.id)["pending"], 2)
        self.run_review(document)
        self.store.add_term(replace(self.term, id="condition", target="状态", sense="condition"))
        self.assertEqual(self.repo.summary(document.id)["pending"], 4)

    def test_reindex_preserves_other_review_sources_and_human_decisions(self):
        document, segments, job = self.book(SENSE_CASES[:1])
        segment = segments[0]
        self.store.replace_issues(
            job.id,
            segment.id,
            [
                QualityIssue("reviewer_attention", "指代需确认", IssueSeverity.WARNING),
                QualityIssue("translation_deferred", "旧任务失败记录", IssueSeverity.ERROR),
                QualityIssue("approved_term_missing", "旧版误报", IssueSeverity.ERROR),
            ],
            target_text=segment.machine_translation,
            detector_version="5",
        )
        reindex_all_quality(self.store, force=True)
        reindex_all_quality(self.store, force=True)
        issues = self.store.list_issues(document.id)
        codes = [item["code"] for item in issues]
        self.assertEqual(codes.count("reviewer_attention"), 1)
        self.assertEqual(codes.count("translation_deferred"), 1)
        self.assertEqual(codes.count("terminology_pending"), 1)
        self.assertNotIn("approved_term_missing", codes)
        self.store.confirm_segment(segment.id, segment.machine_translation, "确认译文")
        before = self.store.get_segment(segment.id)
        refresh_segment_quality(self.store, segment.id)
        self.assertEqual(self.store.get_segment(segment.id), before)

    def test_invalid_model_output_retries_but_never_claims_success(self):
        document, _, _ = self.book(SENSE_CASES[:1])
        self.provider.decide = lambda item: {"id": item["id"], "status": "consistent"}
        run = self.run_review(document)
        self.assertEqual(run["status"], "partial")
        self.assertEqual(self.provider.calls, 2)
        self.assertEqual(self.repo.summary(document.id)["pending"], 2)
        self.assertEqual(run["usage"]["prompt_tokens"], 240)
        self.assertEqual(run["usage"]["verified"], 0)

    def test_quote_only_response_is_located_by_code_and_repeated_targets_need_an_index(self):
        _, segments, _ = self.book([("The state has power.", "国家拥有权力，国家行使权力。")])
        item = review_items(segments[0], [self.term], self.project)[0]
        value = verdict(item, "consistent", "国家", "政治实体。")
        for key in ("source_start", "source_end", "target_start", "target_end"):
            value.pop(key)
        self.assertIsNone(validate_verdict(item, value))
        located = validate_verdict(item, value | {"target_occurrence": 1})
        self.assertEqual(located["target_start"], 7)
        self.assertEqual(located["source_start"], 4)

    def test_conflicting_model_senses_are_human_questions(self):
        document, _, _ = self.book([("The state has power.", "国家与政府有权力。")])
        self.store.add_term(replace(self.term, id="government", target="政府", sense="government"))
        self.provider.decide = lambda item: verdict(item, "consistent", item["term"]["target"])
        self.run_review(document)
        findings = document_issues(self.store, document.id)
        self.assertEqual({item["code"] for item in findings}, {"terminology_uncertain"})
        self.assertEqual(self.repo.summary(document.id)["counts"], {"uncertain": 2})

    def test_legacy_term_schema_migrates_before_verdict_foreign_keys_are_created(self):
        path = str(Path(self.temp.name) / "legacy.db")
        with sqlite3.connect(path) as connection:
            connection.execute("""CREATE TABLE terms (
                id TEXT PRIMARY KEY, project_id TEXT, source TEXT, target TEXT, status TEXT,
                scope TEXT, domain TEXT, rationale TEXT, forbidden_targets_json TEXT,
                created_at TEXT, UNIQUE(project_id,source,scope))""")
        store = SQLiteStore(path)
        store.migrate()
        with store._connect() as connection:
            targets = {
                row["table"]
                for row in connection.execute("PRAGMA foreign_key_list(terminology_verdicts)")
            }
        self.assertIn("terms", targets)
        self.assertNotIn("terms_legacy", targets)

    def test_budget_limit_and_interruption_keep_remaining_work(self):
        document, _, _ = self.book(SENSE_CASES[:1])
        run = self.run_review(document, budget=1)
        self.assertEqual(run["status"], "partial")
        self.assertEqual(self.provider.calls, 0)
        active = self.repo.create_run(document.id, self.model, "balanced", 10000)
        self.repo.fail_interrupted()
        self.assertEqual(self.repo.runs(document.id)[0]["id"], active["id"])
        self.assertEqual(self.repo.runs(document.id)[0]["status"], "failed")
        self.assertGreater(self.repo.summary(document.id)["pending"], 0)

    def test_model_failure_is_separate_from_a_clean_result(self):
        document, _, _ = self.book(SENSE_CASES[:1])

        async def failed(*args, **kwargs):
            raise RuntimeError("private provider details")

        self.provider.complete = failed
        run = self.run_review(document)
        self.assertEqual(run["status"], "failed")
        self.assertNotIn("private provider details", run["error"])
        self.assertEqual(self.repo.summary(document.id)["pending"], 2)

    def test_missing_model_is_actionable_without_mutating_pending_work(self):
        document, _, _ = self.book(SENSE_CASES[:1])
        result = self.client.post(f"/documents/{document.id}/terminology-review", json={})
        self.assertEqual(result.status_code, 422)
        self.assertGreater(self.repo.summary(document.id)["pending"], 0)


def as_tuple(segment):
    return (
        segment.source_text,
        segment.machine_translation,
        segment.edited_translation,
        segment.reviewed_translation,
        segment.accepted_translation,
        segment.status,
    )
