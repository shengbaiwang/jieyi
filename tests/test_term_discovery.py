import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from jieyi.api.app import create_app
from jieyi.api.term_routes import _translated_state_hash
from jieyi.domain.models import ModelSpec, TranslationResult
from jieyi.ingestion import segments_from_text
from jieyi.persistence.sqlite import SQLiteStore
from jieyi.term_discovery import (
    DiscoveryConfig,
    discovery_fingerprint,
    enrich_candidates,
    mine_term_candidates,
    new_discovery_run,
)
from jieyi.term_repository import TermRepository


class GroundedProvider:
    async def complete(self, messages, model, **_kwargs):
        del model
        request = json.loads(messages[-1]["content"])
        card = request["candidates"][0]
        evidence_id = card["evidence"][0]["evidence_id"]
        return TranslationResult(
            text=json.dumps(
                {
                    "proposals": [
                        {
                            "candidate_id": card["candidate_id"],
                            "keep": True,
                            "sense_key": "capacity",
                            "sense": "capacity to act",
                            "concept_definition": "A capacity described in the evidence.",
                            "target": "能动性",
                            "rationale": "A recurring conceptual distinction.",
                            "disambiguation": "Use for capacity, not an institution.",
                            "confidence": 0.9,
                            "evidence_ids": [evidence_id],
                        },
                        {
                            "candidate_id": "invented-candidate",
                            "keep": True,
                            "target": "幻觉",
                            "evidence_ids": ["invented-evidence"],
                        },
                    ]
                },
                ensure_ascii=False,
            ),
            prompt_tokens=100,
            completion_tokens=40,
        )


class OmittingProvider:
    async def complete(self, messages, model, **_kwargs):
        del model
        request = json.loads(messages[-1]["content"])
        card = request["candidates"][0]
        return TranslationResult(
            text=json.dumps(
                {
                    "proposals": [
                        {
                            "candidate_id": card["candidate_id"],
                            "keep": False,
                            "sense_key": "omit",
                            "sense": "",
                            "concept_definition": "",
                            "target": "",
                            "rationale": "Ordinary wording does not require a stable translation.",
                            "disambiguation": "",
                            "confidence": 0.95,
                            "evidence_ids": [card["evidence"][0]["evidence_id"]],
                        }
                    ]
                }
            ),
            prompt_tokens=10,
            completion_tokens=5,
        )


class TermDiscoveryAlgorithmTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.segments = segments_from_text(
            "doc_1",
            "Agency is defined as the capacity to act.\n\n"
            "The theory of agency differs from a government agency.\n\n"
            "Distributed cognition changes agency.",
            "txt",
        )

    def test_same_quoted_span_is_counted_once_and_keeps_all_extraction_signals(self):
        segments = segments_from_text(
            "doc_quoted",
            "The author calls this \u201cepistemic friction\u201d.",
            "txt",
        )
        candidates, _ = mine_term_candidates(
            segments,
            DiscoveryConfig(max_candidates=100, min_score=0.1),
        )
        phrase = next(item for item in candidates if item["lexeme_key"] == "epistemic friction")
        self.assertEqual(phrase["frequency"], 1)
        self.assertIn("quoted", phrase["extraction_methods"])

    def test_full_scan_preserves_exact_evidence_and_filters_phrase_noise(self):
        candidates, coverage = mine_term_candidates(
            self.segments,
            DiscoveryConfig(max_candidates=100, min_score=0.1),
        )
        self.assertEqual(coverage["segments_scanned"], len(self.segments))
        self.assertEqual(coverage["segments_total"], len(self.segments))
        agency = next(item for item in candidates if item["lexeme_key"] == "agency")
        self.assertEqual(agency["frequency"], 4)
        by_id = {segment.id: segment for segment in self.segments}
        for evidence in agency["evidence"]:
            source = by_id[evidence["segment_id"]].source_text
            self.assertEqual(
                source[evidence["start_offset"] : evidence["end_offset"]],
                evidence["source_form"],
            )
        self.assertNotIn(
            "defined as the capacity to",
            {item["lexeme_key"] for item in candidates},
        )

    def test_cjk_documents_use_cjk_candidate_generation(self):
        segments = segments_from_text(
            "doc_zh",
            '"\u77e5\u8bc6\u56fe\u8c31"\u7528\u4e8e\u8868\u793a\u6982\u5ff5\u4e4b\u95f4\u7684\u5173\u7cfb\u3002'
            "\u8be5\u7cfb\u7edf\u53cd\u590d\u4f7f\u7528\u77e5\u8bc6\u56fe\u8c31\u3002",
            "txt",
        )
        candidates, coverage = mine_term_candidates(
            segments,
            DiscoveryConfig(max_candidates=100, min_score=0.1),
            source_lang="zh-CN",
        )
        self.assertEqual(coverage["language_profile"], "cjk")
        self.assertIn(
            "\u77e5\u8bc6\u56fe\u8c31",
            {candidate["lexeme_key"] for candidate in candidates},
        )

    def test_french_profile_corrects_bad_metadata_and_blocks_stopword_noise(self):
        segments = segments_from_text(
            "doc_fr",
            "Un homme passe devant le café et regarde la place Saint-Sulpice.\n\n"
            "Le café est plein, des gens sont dans la rue et un autobus passe.\n\n"
            "Sur le terre-plein de la place Saint-Sulpice, les pigeons sont là.\n\n"
            "Un autre homme traverse le terre-plein et va vers Saint-Sulpice.\n\n"
            "Dans la rue, le café et les gens sont encore là, mais rien ne change.",
            "txt",
        )
        candidates, coverage = mine_term_candidates(
            segments,
            DiscoveryConfig(max_candidates=100, min_score=0.1),
            source_lang="en",
        )
        keys = {candidate["lexeme_key"] for candidate in candidates}
        self.assertEqual(coverage["language_profile"], "fr")
        self.assertLessEqual(len(candidates), coverage["review_queue_ceiling"])
        self.assertTrue({"saint-sulpice", "terre-plein"} & keys)
        self.assertTrue({"de", "des", "le", "un"}.isdisjoint(keys))
        self.assertFalse(any("," in candidate["canonical_form"] for candidate in candidates))

    async def test_model_retries_only_missing_candidate_decisions(self):
        retry_segments = segments_from_text(
            "doc_retry",
            'The author contrasts "epistemic friction" with "distributed cognition".',
            "txt",
        )
        candidates, _ = mine_term_candidates(
            retry_segments,
            DiscoveryConfig(max_candidates=100, min_score=0.1),
        )
        self.assertGreaterEqual(len(candidates), 2)
        enriched, usage = await enrich_candidates(
            candidates,
            provider=OmittingProvider(),
            model=ModelSpec(provider="fake", model="omitter"),
            source_lang="en",
            target_lang="zh-CN",
            config=DiscoveryConfig(
                max_candidates=100,
                max_model_candidates=2,
                model_batch_size=2,
                min_score=0.1,
            ),
        )
        self.assertEqual(usage["model_decisions"], 2)
        self.assertEqual(usage["missing_decisions"], 0)
        self.assertEqual(usage["model_calls"], 2)
        self.assertTrue(
            all(enriched[index]["senses"][0]["ai_recommended"] is False for index in range(2))
        )

    async def test_model_can_only_enrich_known_candidates_and_evidence(self):
        candidates, _ = mine_term_candidates(
            self.segments,
            DiscoveryConfig(max_candidates=100, min_score=0.1),
        )
        enriched, usage = await enrich_candidates(
            candidates,
            provider=GroundedProvider(),
            model=ModelSpec(provider="fake", model="grounded"),
            source_lang="en",
            target_lang="zh-CN",
            config=DiscoveryConfig(
                max_candidates=100,
                max_model_candidates=1,
                model_batch_size=1,
                min_score=0.1,
            ),
        )
        sense = enriched[0]["senses"][0]
        self.assertEqual(sense["proposed_target"], "能动性")
        self.assertEqual(sense["sense"], "capacity to act")
        self.assertEqual(usage["invalid_proposals"], 1)
        valid_evidence = {item["id"] for item in enriched[0]["evidence"]}
        self.assertTrue(set(sense["evidence_ids"]).issubset(valid_evidence))


class TermDiscoveryApiTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.database = Path(self.tempdir.name) / "api.db"
        app = create_app(str(self.database))
        self.client = TestClient(app)

    def tearDown(self):
        self.client.close()
        self.tempdir.cleanup()

    def test_duplicate_running_scan_is_reused_and_restart_closes_orphan(self):
        project = self.client.post(
            "/projects",
            json={"name": "Runs", "source_lang": "en", "target_lang": "zh-CN"},
        ).json()
        document = self.client.post(
            f"/projects/{project['id']}/documents",
            json={
                "title": "Run states",
                "text": "Agency is defined as a capacity to act.",
                "source_format": "txt",
            },
        ).json()
        store = SQLiteStore(self.database)
        repository = TermRepository(store)
        config = DiscoveryConfig()
        stored_document = store.get_document(document["id"])
        segments = store.list_segments(document["id"])
        fingerprint = discovery_fingerprint(
            f"{stored_document.source_hash}:{_translated_state_hash(segments)}:::balanced",
            config,
        )
        run = new_discovery_run(
            document_id=document["id"],
            fingerprint=fingerprint,
            config=config,
            provider="",
            model="",
        )
        repository.create_run(run)

        duplicate = self.client.post(f"/documents/{document['id']}/term-discovery-runs", json={})
        self.assertEqual(duplicate.status_code, 201, duplicate.text)
        self.assertEqual(duplicate.json()["id"], run["id"])
        self.assertEqual(len(repository.list_runs(document["id"])), 1)

        restarted_client = TestClient(create_app(str(self.database)))
        restarted_client.close()
        interrupted = repository.get_run(run["id"])
        self.assertEqual(interrupted["status"], "failed")
        self.assertIn("restart", interrupted["error"])
        self.assertTrue(interrupted["completed_at"])

    def test_candidate_requires_human_approval_then_checks_existing_translation(self):
        project = self.client.post(
            "/projects",
            json={"name": "Concepts", "source_lang": "en", "target_lang": "zh-CN"},
        ).json()
        document = self.client.post(
            f"/projects/{project['id']}/documents",
            json={
                "title": "Agency",
                "text": (
                    "Agency is defined as the capacity to act.\n\n"
                    "The theory of agency differs from a government agency.\n\n"
                    "Distributed cognition changes agency."
                ),
                "source_format": "txt",
            },
        ).json()
        job = self.client.post(
            f"/documents/{document['id']}/jobs",
            json={
                "draft_provider": "echo",
                "draft_model": "draft",
            },
        ).json()
        self.assertEqual(self.client.post(f"/jobs/{job['id']}/run").status_code, 200)

        run_response = self.client.post(
            f"/documents/{document['id']}/term-discovery-runs",
            json={"max_candidates": 100, "min_score": 0.1},
        )
        self.assertEqual(run_response.status_code, 201, run_response.text)
        run = run_response.json()
        self.assertEqual(run["coverage"]["segments_scanned"], 3)
        candidates = self.client.get(f"/documents/{document['id']}/term-candidates").json()
        agency = next(item for item in candidates if item["lexeme_key"] == "agency")
        self.assertIn(agency["candidate_type"], {"concept", "lexical_risk"})
        self.assertGreater(agency["boundary_confidence"], 0)
        candidate_sense = agency["senses"][0]
        self.assertEqual(candidate_sense["status"], "pending")
        self.assertEqual(self.client.get(f"/projects/{project['id']}/terms").json(), [])

        approval = self.client.post(
            f"/term-candidate-senses/{candidate_sense['id']}/approve",
            json={
                "target": "能动性",
                "sense": "行动主体采取行动的能力",
                "rationale": "人工根据全书证据确认",
                "context_keywords": ["capacity", "act"],
                "disambiguation": "不用于政府机构义项",
            },
        )
        self.assertEqual(approval.status_code, 200, approval.text)
        payload = approval.json()
        self.assertEqual(payload["term"]["status"], "approved")
        self.assertGreater(payload["impact"]["translated_occurrences_checked"], 0)
        self.assertEqual(payload["impact"]["segments_needing_revision"], 0)
        self.assertGreater(payload["impact"]["segments_pending_verification"], 0)
        self.assertEqual(payload["term"]["enforcement"], "contextual")
        terms = self.client.get(f"/projects/{project['id']}/terms").json()
        self.assertEqual(terms[0]["target"], "能动性")
        self.assertEqual(terms[0]["context_keywords"], ["capacity", "act"])
        approved_candidates = self.client.get(
            f"/documents/{document['id']}/term-candidates",
            params={"status": "approved"},
        ).json()
        self.assertEqual(approved_candidates[0]["senses"][0]["status"], "approved")

        # The approval can be withdrawn without rewriting any translation or evidence.
        store = SQLiteStore(self.database)
        before = self.client.get(f"/documents/{document['id']}/segments").json()
        term_id = payload["term"]["id"]
        other = self.client.post(
            f"/projects/{project['id']}/terms",
            json={"source": "agency", "target": "机构", "sense": "government institution"},
        )
        self.assertEqual(other.status_code, 201, other.text)
        self.assertTrue(any(issue["code"] == "terminology_pending"
                            for issue in store.list_issues(document["id"])))
        forbidden_edit = self.client.patch(
            f"/term-candidate-senses/{candidate_sense['id']}", json={"status": "pending"},
        )
        self.assertEqual(forbidden_edit.status_code, 422)

        # Existing databases also retain the keywords entered before this migration.
        with store._connect() as connection:
            connection.execute("ALTER TABLE term_candidate_senses DROP COLUMN context_keywords_json")
        store.migrate()
        repository = TermRepository(store)
        self.assertEqual(repository.get_sense(candidate_sense["id"])["context_keywords"],
                         ["capacity", "act"])

        revoked = self.client.post(f"/term-candidate-senses/{candidate_sense['id']}/revoke", json={})
        self.assertEqual(revoked.status_code, 200, revoked.text)
        self.assertEqual(revoked.json()["removed_term_id"], term_id)
        restored = revoked.json()["candidate"]
        self.assertEqual(restored["status"], "pending")
        self.assertIsNone(restored["approved_term_id"])
        self.assertEqual(restored["proposed_target"], "能动性")
        self.assertEqual(restored["context_keywords"], ["capacity", "act"])
        self.assertEqual(self.client.get(f"/documents/{document['id']}/segments").json(), before)
        self.assertEqual([term.id for term in store.list_terms(project["id"])],
                         [other.json()["id"]])
        self.assertTrue(all(term_id not in json.dumps(issue.get("details", {}))
                            for issue in store.list_issues(document["id"])))
        audit = store.list_audit_events("term_candidate_sense", candidate_sense["id"])
        self.assertEqual(sum(event["action"] == "approval_revoked" for event in audit), 1)
        repeated = self.client.post(f"/term-candidate-senses/{candidate_sense['id']}/revoke", json={})
        self.assertEqual(repeated.status_code, 200)
        self.assertIsNone(repeated.json()["removed_term_id"])
        reapproved = self.client.post(
            f"/term-candidate-senses/{candidate_sense['id']}/approve",
            json={"target": "主体能动性", "sense": restored["sense"],
                  "context_keywords": restored["context_keywords"]},
        )
        self.assertEqual(reapproved.status_code, 200, reapproved.text)
        self.assertNotEqual(reapproved.json()["term"]["id"], term_id)
        self.assertEqual(len(store.list_terms(project["id"])), 2)
        self.assertEqual(self.client.post("/term-candidate-senses/missing/revoke", json={}).status_code,
                         404)

    def test_candidate_reference_and_fixed_choices_persist_and_apply_distinct_rules(self):
        project = self.client.post("/projects", json={
            "name": "Candidate choices", "source_lang": "en", "target_lang": "zh-CN",
        }).json()
        document = self.client.post(f"/projects/{project['id']}/documents", json={
            "title": "Agency", "source_format": "txt",
            "text": "Agency is defined as the capacity to act. Agency matters.",
        }).json()
        job = self.client.post(f"/documents/{document['id']}/jobs", json={
            "draft_provider": "echo", "draft_model": "draft",
        }).json()
        self.assertEqual(self.client.post(f"/jobs/{job['id']}/run").status_code, 200)
        segments_url = f"/documents/{document['id']}/segments"
        before = self.client.get(segments_url).json()
        response = self.client.post(f"/documents/{document['id']}/term-discovery-runs", json={
            "max_candidates": 100, "min_score": 0.1, "max_model_candidates": 0,
        })
        self.assertEqual(response.status_code, 201, response.text)
        candidates_url = f"/documents/{document['id']}/term-candidates"
        def candidate_sense():
            candidates = self.client.get(candidates_url).json()
            return next(item for item in candidates if item["lexeme_key"] == "agency")["senses"][0]

        sense_id = candidate_sense()["id"]
        approval_url = f"/term-candidate-senses/{sense_id}/approve"
        invalid = self.client.post(approval_url, json={"target": "能动性", "enforcement": "invalid"})
        self.assertEqual(invalid.status_code, 422)
        self.assertEqual(candidate_sense()["status"], "pending")
        store = SQLiteStore(self.database)
        for mode, marker in (("reference", "[REFERENCE]"), ("global", "[MANDATORY]")):
            with self.subTest(mode=mode):
                response = self.client.post(approval_url, json={
                    "target": "能动性", "enforcement": mode, "sense": "行动能力",
                    "context_keywords": ["capacity"], "disambiguation": "由译者根据证据选择",
                })
                self.assertEqual(response.status_code, 200, response.text)
                payload = response.json()
                self.assertEqual(payload["term"]["enforcement"], mode)
                self.assertEqual(candidate_sense()["approved_enforcement"], mode)
                self.assertEqual(TermRepository(store).get_sense(sense_id)["approved_enforcement"], mode)
                terms = self.client.get(f"/projects/{project['id']}/terms").json()
                self.assertEqual(terms[0]["enforcement"], mode)
                preview = self.client.get(
                    f"/jobs/{job['id']}/segments/{before[0]['id']}/prompt-preview",
                ).json()
                self.assertIn(marker, preview["messages"][0]["content"])
                issues = self.client.get(f"/documents/{document['id']}/issues").json()
                if mode == "reference":
                    self.assertEqual(issues, [])
                    self.assertEqual(payload["impact"]["translated_occurrences_checked"], 0)
                    self.assertEqual(payload["impact"]["segments"], [])
                else:
                    self.assertEqual({item["code"] for item in issues}, {"approved_term_missing"})
                    self.assertGreater(payload["impact"]["segments_needing_revision"], 0)
                audit = store.list_audit_events("term", payload["term"]["id"])
                approved = next(item for item in audit if item["action"] == "approved_from_candidate")
                self.assertEqual(approved["payload"]["enforcement"], mode)
                self.assertEqual(self.client.get(segments_url).json(), before)
                revoked = self.client.post(f"/term-candidate-senses/{sense_id}/revoke", json={})
                self.assertEqual(revoked.status_code, 200)
                self.assertEqual(candidate_sense()["status"], "pending")
                self.assertIsNone(candidate_sense()["approved_enforcement"])

    def test_rejection_is_auditable_and_does_not_create_term(self):
        project = self.client.post(
            "/projects",
            json={"name": "Review", "source_lang": "en", "target_lang": "zh-CN"},
        ).json()
        document = self.client.post(
            f"/projects/{project['id']}/documents",
            json={
                "title": "Terms",
                "text": "Agency means capacity. Agency shapes practice. Agency matters.",
                "source_format": "txt",
            },
        ).json()
        self.client.post(
            f"/documents/{document['id']}/term-discovery-runs",
            json={"max_candidates": 100, "min_score": 0.1},
        )
        candidate = self.client.get(f"/documents/{document['id']}/term-candidates").json()[0]
        sense = candidate["senses"][0]
        rejected = self.client.patch(
            f"/term-candidate-senses/{sense['id']}",
            json={"status": "rejected", "rationale": "这里不是需约束的术语"},
        )
        self.assertEqual(rejected.status_code, 200, rejected.text)
        self.assertEqual(rejected.json()["status"], "rejected")
        self.assertEqual(self.client.post(f"/term-candidate-senses/{sense['id']}/revoke", json={}).status_code, 422)
        self.assertEqual(self.client.get(f"/projects/{project['id']}/terms").json(), [])


if __name__ == "__main__":
    unittest.main()
