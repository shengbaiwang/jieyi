import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from xml.etree import ElementTree as ET

from test_epub_roundtrip import build_roundtrip_epub
from test_epub_structure import build_structural_epub
from test_workflow import EmptyUntilBudgetGrowsProvider

from jieyi.domain.models import JobStatus, TermEntry, TermStatus, TranslationResult, new_id
from jieyi.ingestion.epub_roundtrip import parse_structured_translation
from jieyi.persistence import SQLiteStore
from jieyi.protection import PlaceholderIntegrityError, ProtectedTextCodec
from jieyi.providers import EchoProvider, ProviderRegistry
from jieyi.workflow import (
    TranslationEngine,
    create_document,
    create_epub_document,
    create_job,
    create_project,
)
from jieyi.workflow.provider_responses import is_incomplete_result


class LocalAtomEnvelopeTests(unittest.TestCase):
    def test_only_outer_envelope_is_removed_and_inline_values_roundtrip(self):
        source = '<jy-atom data-jy-id="a">See <em>the idea</em> (Smith 2020) [^2].</jy-atom>'
        original = ProtectedTextCodec().encode(source)
        compact = original.compact_single_atom()
        self.assertEqual(len(compact.tokens), len(original.tokens) - 2)
        self.assertFalse(compact.atom_boundaries)
        self.assertEqual(compact.restore(compact.masked), source)
        self.assertEqual(compact.mask_translation(source), compact.masked)
        self.assertEqual(compact.compact_single_atom(), compact)
        with self.assertRaises(PlaceholderIntegrityError):
            compact.restore(compact.masked.replace(compact.tokens[0], ''))
        with self.assertRaises(PlaceholderIntegrityError):
            compact.restore(compact.masked + original.tokens[0])
        with self.assertRaises(PlaceholderIntegrityError):
            compact.mask_translation('missing envelope')

    def test_multi_atom_boundaries_remain_strict(self):
        source = '<jy-atom data-jy-id="a">First.</jy-atom><jy-atom data-jy-id="b">Second.</jy-atom>'
        original = ProtectedTextCodec().encode(source)
        self.assertIs(original.compact_single_atom(), original)
        with self.assertRaises(PlaceholderIntegrityError):
            original.restore(original.masked.replace(original.tokens[1], ''))

    def test_literal_markers_and_dom_injection_still_checked(self):
        source = '<jy-atom data-jy-id="a">Literal [[JY_PH_0000]].</jy-atom>'
        compact = ProtectedTextCodec().encode(source).compact_single_atom()
        self.assertEqual(compact.restore(compact.masked), source)
        plain, _ = parse_structured_translation(compact.restore(compact.masked), ('a',))
        self.assertIn('[[JY_PH_0000]]', plain)
        with self.assertRaises(ValueError):
            parse_structured_translation(
                compact.restore('<jy-atom data-jy-id="evil">' + compact.masked + '</jy-atom>'),
                ('a',),
            )


class RecordingEcho(EchoProvider):
    def __init__(self):
        self.calls = []

    async def complete(self, messages, model, **kwargs):
        self.calls.append((messages, kwargs))
        return await super().complete(messages, model, **kwargs)


class TokenEfficiencyWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = SQLiteStore(Path(self.directory.name) / 'test.db')
        self.store.migrate()
        self.project = create_project(
            self.store, name='Book', source_lang='en', target_lang='zh-CN',
            style_guide='Keep all attributions and qualifications.',
        )

    def run_with(self, document, provider, **config):
        registry = ProviderRegistry()
        registry.register('test', provider)
        engine = TranslationEngine(self.store, registry)
        job = create_job(
            self.store, document_id=document.id, draft_provider='test', draft_model='test',
            concurrency=1, max_concurrency=1, **config,
        )
        return engine, job

    def test_optimized_preview_is_exact_and_epub_mapping_survives(self):
        document = create_epub_document(
            self.store, project_id=self.project.id, file_data=build_roundtrip_epub(),
        )
        self.store.add_term(TermEntry(
            id=new_id('term'), project_id=self.project.id,
            source='Agency', target='能动性', status=TermStatus.APPROVED,
        ))
        segments = self.store.list_segments(document.id)
        current = next(segment for segment in segments if 'Agency' in segment.source_text)
        provider = RecordingEcho()
        engine, job = self.run_with(document, provider, segment_ranges=[(current.ordinal, current.ordinal)])
        preview = engine.preview(job.id, current.id, optimized=True)
        self.assertIsNotNone(preview['local_wrapper'])
        self.assertEqual(preview['relevant_terms'][0]['target'], '能动性')
        completed = asyncio.run(engine.run_optimized(job.id))
        self.assertEqual(completed.status, JobStatus.COMPLETED)
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(provider.calls[0][0], preview['messages'])
        self.assertEqual(provider.calls[0][1]['max_tokens'], preview['max_output_tokens'])
        saved = self.store.epub_structured_translation(current.id)
        self.assertIn('<jy-atom data-jy-id=', saved)
        strong = next(node for node in ET.fromstring(saved).iter()
                      if node.tag.rsplit('}', 1)[-1] == 'strong')
        self.assertEqual(strong.text, 'matters')
        self.assertTrue(self.store.get_segment(current.id).machine_translation)
        self.assertIn('Keep all attributions', preview['messages'][0]['content'])
        self.assertIn('Agency -> 能动性', preview['messages'][1]['content'])
        for ordinal in [current.ordinal - 1, current.ordinal + 1]:
            if 0 <= ordinal < len(segments):
                self.assertIn(segments[ordinal].source_text, preview['messages'][1]['content'])
        progress = self.store.job_progress(job.id)
        self.assertEqual(progress['model_call_count'], 1)
        self.assertEqual(progress['repair_call_count'], 0)

    def test_multi_atom_epub_preview_and_execution_keep_boundaries(self):
        document = create_epub_document(
            self.store, project_id=self.project.id, file_data=build_structural_epub(),
        )
        current = next(s for s in self.store.list_segments(document.id) if len(s.source_refs) > 1)
        provider = RecordingEcho()
        engine, job = self.run_with(document, provider, segment_ranges=[(current.ordinal, current.ordinal)])
        preview = engine.preview(job.id, current.id, optimized=True)
        self.assertIsNone(preview['local_wrapper'])
        self.assertIn('SourceAtom boundary pairs', preview['messages'][0]['content'])
        asyncio.run(engine.run_optimized(job.id))
        structured = self.store.epub_structured_translation(current.id)
        self.assertEqual(structured.count('<jy-atom '), len(current.source_refs))
        self.assertEqual(provider.calls[0][0], preview['messages'])

    def test_balanced_short_source_avoids_paid_empty_retry_without_lowering_effort(self):
        document = create_document(
            self.store, project_id=self.project.id, title='Short', text='Bref.', source_format='txt',
        )
        provider = EmptyUntilBudgetGrowsProvider()
        engine, job = self.run_with(document, provider, draft_compute_mode='balanced')
        asyncio.run(engine.run_optimized(job.id))
        self.assertEqual(provider.max_tokens, [2560])
        self.assertEqual(provider.reasoning_efforts, ['medium'])
        progress = self.store.job_progress(job.id)
        self.assertEqual(progress['total_tokens'], 220)
        self.assertEqual(progress['model_call_count'], 1)
        self.assertEqual(progress['retry_call_count'], 0)

    def test_truncated_visible_response_is_retried_and_usage_is_counted_once(self):
        class TruncatedOnce(RecordingEcho):
            async def complete(self, messages, model, **kwargs):
                if not self.calls:
                    self.calls.append((messages, kwargs))
                    return TranslationResult(
                        text='partial translation', prompt_tokens=100, completion_tokens=512,
                        raw_response=json.dumps({'choices': [{'finish_reason': 'length'}]}),
                    )
                return await super().complete(messages, model, **kwargs)

        document = create_document(
            self.store, project_id=self.project.id, title='Short', text='Bref.', source_format='txt',
        )
        provider = TruncatedOnce()
        engine, job = self.run_with(document, provider)
        asyncio.run(engine.run_optimized(job.id))
        segment = self.store.list_segments(document.id)[0]
        self.assertEqual(segment.machine_translation, 'translated:Bref.')
        self.assertEqual([call[1]['max_tokens'] for call in provider.calls], [512, 2048])
        calls = [event['payload'] for event in self.store.list_audit_events('job', job.id)
                 if event['action'] == 'model_call']
        self.assertEqual([call['outcome'] for call in calls], ['output_budget_exhausted', 'complete'])
        progress = self.store.job_progress(job.id)
        self.assertEqual(progress['model_call_count'], 2)
        self.assertEqual(progress['retry_call_count'], 1)
        self.assertEqual(progress['total_tokens'], sum(
            call['prompt_tokens'] + call['completion_tokens'] for call in calls
        ))
        self.assertNotIn('partial translation', json.dumps(calls))

    def test_truncated_response_at_cap_is_deferred_without_saving_partial_text(self):
        class AlwaysTruncated(RecordingEcho):
            async def complete(self, messages, model, **kwargs):
                return TranslationResult(
                    text='partial', prompt_tokens=100, completion_tokens=512,
                    raw_response=json.dumps({'choices': [{'finish_reason': 'length'}]}),
                )

        document = create_document(
            self.store, project_id=self.project.id, title='Short', text='Bref.', source_format='txt',
        )
        engine, job = self.run_with(document, AlwaysTruncated(), max_output_tokens=512)
        asyncio.run(engine.run_optimized(job.id))
        self.assertIsNone(self.store.list_segments(document.id)[0].machine_translation)
        progress = self.store.job_progress(job.id)
        self.assertEqual(progress['deferred_segments'], 1)
        self.assertEqual(progress['total_tokens'], 612)
        self.assertEqual(progress['model_call_count'], 1)

    def test_native_protocol_truncation_flags(self):
        for payload in (
            {'choices': [{'finish_reason': 'length'}]},
            {'status': 'incomplete', 'incomplete_details': {'reason': 'max_output_tokens'}},
            {'stop_reason': 'max_tokens'},
            {'candidates': [{'finishReason': 'MAX_TOKENS'}]},
        ):
            with self.subTest(payload=payload):
                self.assertTrue(is_incomplete_result(TranslationResult(
                    text='partial', raw_response=json.dumps(payload),
                )))
        self.assertFalse(is_incomplete_result(TranslationResult(
            text='complete', raw_response='{"choices":[{"finish_reason":"stop"}]}',
        )))
