import asyncio
import json
import tempfile
import unittest
from concurrent.futures import CancelledError as FutureCancelledError
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from jieyi.api.app import create_app
from jieyi.domain.models import ModelSpec, TranslationResult
from jieyi.ingestion import segments_from_text
from jieyi.term_discovery import DiscoveryConfig, enrich_candidates, mine_term_candidates
from jieyi.term_repository import TermRepository

SOURCE = '\n\n'.join(
    f'The author defines "{term}" as a distinct concept in this theory.'
    for term in (
        'epistemic friction', 'distributed cognition', 'social agency', 'symbolic capital',
        'collective memory', 'cognitive dissonance', 'institutional power', 'cultural mediation',
    )
)


def reply(cards, **overrides):
    return TranslationResult(
        text=json.dumps({'proposals': [
            {'candidate_id': c['candidate_id'], 'keep': True, 'target': '术语译法',
             'criteria': {
                 'stable_concept': True,
                 'book_significant': True,
                 'consistency_needed': True,
                 'specialized_usage': True,
             },
             'confidence': 0.9,
             'evidence_ids': [c['evidence'][0]['evidence_id']], **overrides}
            for c in cards
        ]}),
        prompt_tokens=100, completion_tokens=40, reasoning_tokens=10, cost_usd=0.01,
    )


class RecordingProvider:
    def __init__(self, handler=None):
        self.calls = []
        self.handler = handler

    async def complete(self, messages, model, **kwargs):
        cards = json.loads(messages[-1]['content'])['candidates']
        self.calls.append({'ids': [c['candidate_id'] for c in cards], 'model': model.model, **kwargs})
        return self.handler(cards) if self.handler else reply(cards)


class RecoveryAlgorithmTests(unittest.IsolatedAsyncioTestCase):
    async def test_truncated_batches_shrink_and_only_missing_ids_repeat(self):
        candidates, _ = mine_term_candidates(
            segments_from_text('doc', SOURCE, 'txt'), DiscoveryConfig(min_score=0.1),
        )
        self.assertGreaterEqual(len(candidates), 4)
        first_id = candidates[0]['id']

        def respond(cards):
            if len(provider.calls) == 1:
                return reply(cards[:1])
            if len(cards) > 1:
                return TranslationResult(
                    text='{"proposals": [', completion_tokens=4000,
                    raw_response=json.dumps({'choices': [{'finish_reason': 'length'}]}),
                )
            return reply(cards)

        provider = RecordingProvider(respond)
        checkpoints = []
        _, usage = await enrich_candidates(
            candidates, provider=provider, model=ModelSpec('test', 'test'),
            source_lang='en', target_lang='zh-CN',
            config=DiscoveryConfig(max_model_candidates=4),
            checkpoint=lambda items, current: checkpoints.append(current['model_decisions']),
        )
        self.assertEqual(usage['missing_decisions'], 0)
        self.assertEqual(usage['model_kept'], 4)
        self.assertEqual(sum(first_id in c['ids'] for c in provider.calls), 1)
        self.assertEqual(checkpoints[0], 1)
        self.assertEqual(checkpoints[-1], 4)
        self.assertTrue(any(c['max_tokens'] == 8000 for c in provider.calls[1:]))
        self.assertTrue(any(d['finish_reason'] == 'length' for d in usage['diagnostics']))

    async def test_invalid_decisions_remain_pending_with_bounded_retries(self):
        for bad_fields in (
            {'keep': 'false'},
            {'target': ''},
            {'evidence_ids': ['invented']},
            {'criteria': {}},
        ):
            with self.subTest(fields=bad_fields):
                candidates, _ = mine_term_candidates(segments_from_text('doc', SOURCE, 'txt'))
                provider = RecordingProvider(lambda cards, fields=bad_fields: reply(cards, **fields))
                _, usage = await enrich_candidates(
                    candidates, provider=provider, model=ModelSpec('test', 'test'),
                    source_lang='en', target_lang='zh-CN',
                    config=DiscoveryConfig(max_model_candidates=1),
                )
                self.assertEqual(len(provider.calls), 4)
                self.assertEqual(usage['missing_decisions'], 1)
                self.assertEqual(usage['model_decisions'], 0)
                self.assertEqual(usage['invalid_proposals'], 4)
                self.assertIsNone(candidates[0]['senses'][0]['ai_recommended'])


class RecoveryApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = str(Path(self.temp.name) / 'recovery.db')
        self.app = create_app(self.path)
        self.client = TestClient(self.app)
        self.repository = TermRepository(self.app.state.store)
        self.provider = RecordingProvider()
        self.app.state.providers.register('recovery', self.provider)
        self.project = self.client.post('/projects', json={
            'name': 'Recovery', 'source_lang': 'en', 'target_lang': 'zh-CN',
        }).json()
        self.document = self.client.post(f"/projects/{self.project['id']}/documents", json={
            'title': 'Recovery', 'text': SOURCE, 'source_format': 'txt',
        }).json()
        self.base = f"/documents/{self.document['id']}"
        self.options = {'provider': 'recovery', 'model': 'model-a', 'model_batch_size': 2,
                        'max_model_candidates': 8, 'min_score': 0.1}

    def tearDown(self):
        self.client.close()
        self.temp.cleanup()

    def start(self, **options):
        response = self.client.post(self.base + '/term-discovery-runs', json=self.options | options)
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()

    def retry(self, run, **options):
        response = self.client.post(self.base + f"/term-discovery-runs/{run['id']}/retry", json=options)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def candidates(self):
        return self.client.get(self.base + '/term-candidates').json()

    def fail_after_first_batch(self, cards):
        if len(self.provider.calls) > 1:
            raise TimeoutError('test provider timed out')
        return reply(cards)

    def test_partial_resume_keeps_ids_usage_and_never_scans_again(self):
        self.provider.handler = self.fail_after_first_batch
        partial = self.start()
        self.assertEqual(partial['status'], 'partial')
        self.assertEqual(partial['coverage']['model_decisions'], 2)
        self.assertEqual(partial['coverage']['missing_model_decisions'], 4)
        self.assertEqual(partial['prompt_tokens'], 100)
        before = self.candidates()
        completed_ids = {c['id'] for c in before if c['senses'][0]['ai_recommended'] is not None}
        self.provider.handler = None
        self.provider.calls.clear()
        with patch('jieyi.api.term_routes.mine_term_candidates', side_effect=AssertionError('rescanned')):
            finished = self.retry(partial, model='model-b', provider='recovery')
            self.assertEqual(finished['status'], 'completed')
            self.assertEqual(finished['coverage']['missing_model_decisions'], 0)
            self.assertEqual(finished['prompt_tokens'], 300)
            self.assertEqual(finished['completion_tokens'], 120)
            self.assertEqual(finished['reasoning_tokens'], 30)
            self.assertAlmostEqual(finished['cost_usd'], 0.03)
            self.assertEqual(finished['coverage']['model_calls'], 4)
            self.assertEqual(len(self.provider.calls), 2)
            self.assertTrue(completed_ids.isdisjoint(i for c in self.provider.calls for i in c['ids']))
            self.assertTrue(all(c['model'] == 'model-b' for c in self.provider.calls))
            after = self.candidates()
            self.assertEqual([(c['id'], c['evidence']) for c in before],
                             [(c['id'], c['evidence']) for c in after])
            self.assertEqual([c['senses'] for c in before if c['id'] in completed_ids],
                             [c['senses'] for c in after if c['id'] in completed_ids])
            again = self.retry(finished)
            self.assertEqual(again['prompt_tokens'], finished['prompt_tokens'])
            self.assertEqual(len(self.provider.calls), 2)
            # Translation/model changes must not trigger a new full scan.
            segment = self.client.get(self.base + '/segments').json()[0]
            self.client.patch(f"/segments/{segment['id']}/confirm", json={'translation': '已确认译文'})
            reused = self.start(model='model-c', compute_mode='performance', model_batch_size=4)
            self.assertEqual(reused['id'], partial['id'])
            self.assertEqual(len(self.repository.list_runs(self.document['id'])), 1)

    def test_human_edits_approvals_and_rejections_survive_resume(self):
        local = self.start(provider='', model='')
        candidates = self.candidates()
        self.assertGreaterEqual(len(candidates), 6)
        protected_ids = {c['id'] for c in candidates[:3]}
        for index, c in enumerate(candidates[:3]):
            sense_id = c['senses'][0]['id']
            if index == 0:
                response = self.client.post(f'/term-candidate-senses/{sense_id}/approve', json={'target': '人工批准'})
            else:
                response = self.client.patch(f'/term-candidate-senses/{sense_id}', json={
                    'status': 'rejected' if index == 1 else 'pending', 'proposed_target': '人工编辑',
                })
            self.assertEqual(response.status_code, 200, response.text)
        protected_before = [c['senses'] for c in self.candidates()[:3]]
        finished = self.retry(local, provider='recovery', model='model-a')
        self.assertEqual(finished['status'], 'completed')
        self.assertTrue(protected_ids.isdisjoint(i for c in self.provider.calls for i in c['ids']))
        self.assertEqual([c['senses'] for c in self.candidates()[:3]], protected_before)
        self.assertEqual(len(self.app.state.store.list_terms(self.project['id'])), 1)

    def test_human_edit_during_model_call_wins_over_late_response(self):
        local = self.start(provider='', model='')
        first = self.candidates()[0]
        sense_id = first['senses'][0]['id']

        def edit_then_reply(cards):
            if len(self.provider.calls) == 1:
                self.repository.review_sense(
                    sense_id, status='pending', proposed_target='人工刚修改', sense='人工义项',
                    rationale='保留', disambiguation='', actor='human',
                )
            return reply(cards)

        self.provider.handler = edit_then_reply
        finished = self.retry(local, provider='recovery', model='model-a')
        self.assertEqual(finished['status'], 'completed')
        stored = self.repository.get_sense(sense_id)
        self.assertEqual(stored['proposed_target'], '人工刚修改')
        self.assertIsNone(stored['ai_recommended'])

    def test_restart_recovers_saved_partial_run_and_legacy_completed_error(self):
        self.provider.handler = self.fail_after_first_batch
        partial = self.start()
        before = self.candidates()
        # Simulate death after a checkpoint but before the run's final status update.
        self.repository.update_run(partial['id'], status='running', error='')
        app = create_app(self.path)
        app.state.providers.register('recovery', RecordingProvider())
        restarted = TestClient(app)
        with patch('jieyi.api.term_routes.mine_term_candidates', side_effect=AssertionError('rescanned')):
            restored = self.repository.get_run(partial['id'])
            self.assertEqual(restored['status'], 'partial')
            response = restarted.post(self.base + f"/term-discovery-runs/{partial['id']}/retry", json={})
            self.assertEqual(response.json()['status'], 'completed', response.text)
        self.assertEqual([c['id'] for c in self.candidates()], [c['id'] for c in before])
        restarted.close()
        # Historical screenshot records use completed+error and lack the new coverage fields.
        legacy = self.repository.get_run(partial['id'])
        coverage = {k: v for k, v in legacy['coverage'].items()
                    if k not in {'source_hash', 'scan_completed', 'review_provider', 'review_model'}}
        self.repository.update_run(legacy['id'], status='completed', coverage=coverage,
                                   error='Model review incomplete after retry: 8 candidate decision(s) missing')
        with patch('jieyi.api.term_routes.mine_term_candidates', side_effect=AssertionError('rescanned')):
            self.assertEqual(self.start()['id'], legacy['id'])
            self.assertEqual(self.retry(legacy)['error'], '')

    def test_legacy_completed_with_missing_decisions_resumes_without_replacing_candidates(self):
        self.provider.handler = self.fail_after_first_batch
        partial = self.start()
        before = self.candidates()
        coverage = {k: v for k, v in partial['coverage'].items()
                    if k not in {'source_hash', 'scan_completed', 'review_provider', 'review_model',
                                 'review_compute_mode', 'review_limit'}}
        self.repository.update_run(partial['id'], status='completed', coverage=coverage,
                                   error='Model review incomplete after retry: 6 candidate decision(s) missing')
        self.provider.handler = None
        self.provider.calls.clear()
        with patch('jieyi.api.term_routes.mine_term_candidates', side_effect=AssertionError('rescanned')):
            self.assertEqual(self.start()['id'], partial['id'])
            finished = self.retry(partial)
        self.assertEqual(finished['status'], 'completed')
        self.assertEqual(finished['error'], '')
        self.assertEqual(len(self.provider.calls), 2)
        self.assertEqual([c['id'] for c in self.candidates()], [c['id'] for c in before])
        self.assertEqual([c['senses'] for c in self.candidates()[:2]], [c['senses'] for c in before[:2]])

    def test_scan_is_durable_before_first_model_call_and_claim_is_atomic(self):
        def inspect_scan(cards):
            run = self.repository.list_runs(self.document['id'])[0]
            saved = self.repository.list_candidates(self.document['id'])
            self.assertTrue(run['coverage']['scan_completed'])
            self.assertTrue({c['candidate_id'] for c in cards}.issubset({c['id'] for c in saved}))
            return reply(cards)

        self.provider.handler = inspect_scan
        run = self.start()
        with ThreadPoolExecutor(max_workers=4) as pool:
            claims = list(pool.map(lambda _: self.repository.claim_review(run['id']), range(4)))
        self.assertEqual(sum(claims), 1)
        calls = len(self.provider.calls)
        self.assertEqual(self.retry(run)['status'], 'running')
        self.assertEqual(self.start()['id'], run['id'])
        self.assertEqual(len(self.provider.calls), calls)

    def test_cancellation_retains_checkpoint_and_can_resume(self):
        class CancellingProvider(RecordingProvider):
            async def complete(inner, messages, model, **kwargs):
                if inner.calls:
                    raise asyncio.CancelledError()
                return await super().complete(messages, model, **kwargs)

        self.app.state.providers.register('recovery', CancellingProvider())
        with self.assertRaises((asyncio.CancelledError, FutureCancelledError)):
            self.start()
        run = self.repository.list_runs(self.document['id'])[0]
        self.assertEqual(run['status'], 'partial')
        self.assertEqual(run['coverage']['model_decisions'], 2)
        self.assertEqual(run['prompt_tokens'], 100)
        self.app.state.providers.register('recovery', self.provider)
        self.assertEqual(self.retry(run)['status'], 'completed')

    def test_retry_checks_document_and_missing_scan(self):
        run = self.start(provider='', model='')
        response = self.client.post(self.base + '/term-discovery-runs/missing/retry', json={})
        self.assertEqual(response.status_code, 404)
        other = self.client.post(f"/projects/{self.project['id']}/documents", json={
            'title': 'Other', 'text': 'Other source', 'source_format': 'txt',
        }).json()
        response = self.client.post(
            f"/documents/{other['id']}/term-discovery-runs/{run['id']}/retry", json={},
        )
        self.assertEqual(response.status_code, 404)
        self.repository.update_run(run['id'], coverage={}, status='failed')
        response = self.client.post(self.base + f"/term-discovery-runs/{run['id']}/retry", json={})
        self.assertEqual(response.status_code, 409)
