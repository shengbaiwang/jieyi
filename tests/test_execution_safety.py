"""All providers in this module are local fakes; no network or model billing."""
import asyncio
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from unittest.mock import patch

import pytest

from jieyi.domain.models import CandidateStage, JobStatus, SegmentStatus, TranslationResult, new_id
from jieyi.persistence import SQLiteStore
from jieyi.persistence.execution import DocumentBusyError
from jieyi.providers import EchoProvider, ProviderRegistry
from jieyi.workflow import TranslationEngine, create_document, create_job, create_project
from jieyi.workflow.jobs import JobManager


@pytest.fixture
def book(tmp_path):
    store = SQLiteStore(tmp_path / 'test.db')
    store.migrate()
    project = create_project(store, name='Safety', source_lang='en', target_lang='zh-CN')
    document = create_document(store, project_id=project.id, title='Safety', source_format='txt',
                               text='First passage.\n\nSecond passage.\n\nThird passage.')
    return store, document


def job_for(store, document, **kwargs):
    return create_job(store, document_id=document.id, draft_provider='fake', draft_model='test',
                      **kwargs)


def engine_for(store, provider):
    registry = ProviderRegistry()
    registry.register('fake', provider)
    return TranslationEngine(store, registry)


async def until(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0.002)


def test_concurrent_equivalent_submissions_reuse_one_job(book):
    store, document = book
    alternatives = [[(0, 1), (1, 2), (0, 1)], [(0, 2)], [(2, 2), (0, 0), (1, 1)]]
    with ThreadPoolExecutor(max_workers=8) as pool:
        jobs = list(pool.map(lambda i: job_for(store, document,
                    segment_ranges=alternatives[i % len(alternatives)]), range(24)))
    assert len({job.id for job in jobs}) == 1
    assert jobs[0].recipe.segment_ranges == ((0, 2),)
    assert job_for(store, document, segment_ranges=[(1, 1)]).id != jobs[0].id
    store.set_job_status(jobs[0].id, JobStatus.CANCELLED)
    assert job_for(store, document, segment_ranges=[(0, 2)]).id != jobs[0].id


def test_document_lock_shared_across_processes_and_aliases(book, tmp_path):
    store, document = book
    alias = tmp_path / 'alias.db'
    alias.symlink_to(store.path)
    code = '''from jieyi.persistence.execution import DocumentRunLock
import sys
with DocumentRunLock(sys.argv[1], sys.argv[2]):
    print('ready', flush=True)
    sys.stdin.readline()
'''
    child = subprocess.Popen([sys.executable, '-c', code, str(alias), document.id],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == 'ready'
        with pytest.raises(DocumentBusyError), store.reserve_document(document.id):
            pass
        with store.reserve_document('another-document'):
            pass
    finally:
        child.communicate('\n', timeout=3)
    with store.reserve_document(document.id):
        pass


def test_recovery_does_not_pause_another_live_process(book):
    store, document = book
    job = job_for(store, document)
    store.set_job_status(job.id, JobStatus.RUNNING)
    with store.reserve_document(document.id):
        store.pause_interrupted_jobs()
        assert store.get_job(job.id).status is JobStatus.RUNNING
    store.pause_interrupted_jobs()
    assert store.get_job(job.id).status is JobStatus.PAUSED


@pytest.mark.parametrize('optimized', [False, True])
@pytest.mark.parametrize('translation_type', ['machine', 'edited', 'reviewed', 'confirmed'])
def test_all_execution_paths_skip_existing_translations(book, optimized, translation_type):
    store, document = book
    first = store.list_segments(document.id)[0]
    methods = {'machine': store.set_machine_translation, 'edited': store.save_segment_draft,
               'reviewed': store.set_reviewed_translation, 'confirmed': store.confirm_segment}
    methods[translation_type](first.id, '保留的译文')
    provider = EchoProvider()
    calls = []
    original_translate, original_complete = provider.translate, provider.complete

    async def translate(request, model):
        calls.append(request.segment.id)
        return await original_translate(request, model)

    async def complete(messages, model, **kwargs):
        calls.append(messages)
        return await original_complete(messages, model, **kwargs)

    provider.translate, provider.complete = translate, complete
    engine = engine_for(store, provider)
    runner = engine.run_optimized if optimized else engine.run
    job = job_for(store, document)
    assert asyncio.run(runner(job.id)).status is JobStatus.COMPLETED
    assert len(calls) == 2
    assert asyncio.run(runner(job_for(store, document).id)).status is JobStatus.COMPLETED
    assert len(calls) == 2
    assert getattr(store.get_segment(first.id),
                   {'machine': 'machine_translation', 'edited': 'edited_translation',
                    'reviewed': 'reviewed_translation', 'confirmed': 'accepted_translation'}[
                        translation_type]) == '保留的译文'


def test_checkpoint_is_atomic_and_does_not_replace_human_work(book):
    store, document = book
    job = job_for(store, document)
    segment = store.list_segments(document.id)[0]
    kwargs = {'job': job, 'segment': segment, 'text': '机器译文', 'issues': [],
                  'detector_version': 'test', 'stage': CandidateStage.DRAFT}
    with (patch.object(store, 'replace_issues', side_effect=RuntimeError('disk failure')),
          pytest.raises(RuntimeError, match='disk failure')):
        store.commit_generated_translation(**kwargs)
    assert not store.get_segment(segment.id).machine_translation
    with store._connect() as conn:
        assert conn.execute('SELECT COUNT(*) FROM candidates').fetchone()[0] == 0
    store.confirm_segment(segment.id, '人工确认的译文')
    assert not store.commit_generated_translation(**kwargs)
    assert store.get_segment(segment.id).status is SegmentStatus.HUMAN_CONFIRMED
    assert store.get_segment(segment.id).accepted_translation == '人工确认的译文'
    second = store.list_segments(document.id)[1]
    kwargs['segment'] = second
    assert store.commit_generated_translation(**kwargs)
    assert not store.commit_generated_translation(**kwargs)
    with store._connect() as conn:
        assert conn.execute('SELECT COUNT(*) FROM candidates').fetchone()[0] == 1


class SlowProvider(EchoProvider):
    """Simulate an already-issued blocking HTTP request, never use the network."""
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    async def translate(self, request, model):
        await self.wait_for_response()
        return await super().translate(request, model)

    async def complete(self, messages, model, **kwargs):
        await self.wait_for_response()
        return await super().complete(messages, model, **kwargs)

    async def wait_for_response(self):
        self.calls += 1
        if self.calls == 2:
            def fake_http():
                self.started.set()
                assert self.release.wait(3), 'test must release its simulated request'
            await asyncio.to_thread(fake_http)


def test_pause_saves_issued_results_and_blocks_overlapping_resume(book):
    store, document = book
    provider = SlowProvider()
    engine = engine_for(store, provider)
    manager = JobManager(store, engine)
    job = job_for(store, document, batch_size=3, concurrency=1, max_concurrency=1)
    first = store.list_segments(document.id)[0]

    async def scenario():
        manager.start(job.id)
        await until(provider.started.is_set)
        await until(lambda: bool(store.get_segment(first.id).machine_translation))
        progress = await manager.stop(job.id, JobStatus.PAUSED)
        assert progress['draining']
        with pytest.raises(DocumentBusyError):
            manager.start(job.id)
        other = store.create_job(replace(job, id=new_id('job')))
        with pytest.raises(DocumentBusyError):
            await engine.run(other.id)
        provider.release.set()
        await until(lambda: not manager.running(job.id))
        assert provider.calls == 2  # Third request was queued but never issued.
        assert store.get_job(job.id).status is JobStatus.PAUSED
        manager.start(job.id)
        await until(lambda: not manager.running(job.id))
        assert store.get_job(job.id).status is JobStatus.COMPLETED
        assert provider.calls == 3
        await manager.shutdown()

    try:
        asyncio.run(scenario())
    finally:
        provider.release.set()


def test_same_text_in_distinct_locations_is_not_collapsed(book):
    store, document = book
    document = create_document(store, project_id=document.project_id, title='Repeated text',
                               text='Identical passage.\n\nIdentical passage.', source_format='txt')
    engine = engine_for(store, EchoProvider())
    asyncio.run(engine.run_optimized(job_for(store, document).id))
    segments = store.list_segments(document.id)
    assert len(segments) == 2 and segments[0].id != segments[1].id
    assert all(segment.machine_translation for segment in segments)


@pytest.mark.parametrize('optimized', [False, True])
@pytest.mark.parametrize('repeat_cancel', [False, True])
def test_cancellation_keeps_ownership_until_issued_request_is_saved(book, optimized, repeat_cancel):
    store, document = book
    provider = SlowProvider()
    engine = engine_for(store, provider)
    job = job_for(store, document, batch_size=3, concurrency=1, max_concurrency=1)

    async def scenario():
        runner = engine.run_optimized if optimized else engine.run
        task = asyncio.create_task(runner(job.id))
        try:
            await until(provider.started.is_set)
            task.cancel()
            await until(lambda: store.get_job(job.id).status is JobStatus.PAUSED)
            if repeat_cancel:
                task.cancel()
                await asyncio.sleep(0)
            with pytest.raises(DocumentBusyError), store.reserve_document(document.id):
                pass
            provider.release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert provider.calls == 2
            assert store.list_segments(document.id)[1].machine_translation
            await runner(job.id)
            assert provider.calls == 3
        finally:
            provider.release.set()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


@pytest.mark.parametrize('response', ['', 'broken [[JY_PH_9999]]'])
def test_pause_prevents_empty_retry_and_placeholder_repair(book, response):
    store, document = book
    job = job_for(store, document, concurrency=1, max_concurrency=1)

    class PauseOnResponse(EchoProvider):
        calls = 0

        async def complete(self, messages, model, **kwargs):
            self.calls += 1
            store.set_job_status(job.id, JobStatus.PAUSED)
            return TranslationResult(text=response, prompt_tokens=3, completion_tokens=1)

    provider = PauseOnResponse()
    completed = asyncio.run(engine_for(store, provider).run_optimized(job.id))
    assert completed.status is JobStatus.PAUSED
    assert provider.calls == 1
    assert completed.next_ordinal == 0


def test_api_returns_conflict_without_starting_duplicate_run(book):
    from fastapi.testclient import TestClient

    from jieyi.api.app import create_app

    store, document = book
    job = job_for(store, document)
    with TestClient(create_app(str(store.path))) as client, store.reserve_document(document.id):
        for action in ('start', 'run', 'pause', 'cancel'):
            response = client.post(f'/jobs/{job.id}/{action}')
            assert response.status_code == 409
            assert store.get_job(job.id).status is JobStatus.PENDING


@pytest.mark.parametrize('optimized', [False, True])
def test_epub_late_result_preserves_human_text_and_structure(book, optimized):
    from test_epub import build_epub

    from jieyi.workflow import create_epub_document

    store, document = book
    document = create_epub_document(store, project_id=document.project_id, file_data=build_epub())
    segment = store.list_segments(document.id)[0]
    original = store.epub_translation_source(segment.id)
    plain = store.capture_epub_translation(segment.id, original, 'draft', persist=False)
    job = job_for(store, document, segment_ranges=[(0, 0)])
    kwargs = {'job': job, 'segment': segment, 'text': plain, 'structured_value': original,
                  'issues': [], 'detector_version': 'test', 'stage': CandidateStage.DRAFT}
    with (patch.object(store, 'replace_issues', side_effect=RuntimeError('disk failure')),
          pytest.raises(RuntimeError)):
        store.commit_generated_translation(**kwargs)
    with store._connect() as conn:
        assert conn.execute('SELECT COUNT(*) FROM epub_atom_translations').fetchone()[0] == 0

    class HumanEditsBeforeResponse(EchoProvider):
        async def translate(self, request, model):
            store.save_segment_draft(segment.id, '人工刚保存的内容')
            return await super().translate(request, model)

        async def complete(self, messages, model, **kwargs):
            store.save_segment_draft(segment.id, '人工刚保存的内容')
            return await super().complete(messages, model, **kwargs)

    engine = engine_for(store, HumanEditsBeforeResponse())
    asyncio.run((engine.run_optimized if optimized else engine.run)(job.id))
    assert store.get_segment(segment.id).edited_translation == '人工刚保存的内容'
    assert not store.get_segment(segment.id).machine_translation
    with store._connect() as conn:
        assert conn.execute('SELECT COUNT(*) FROM epub_atom_translations').fetchone()[0] == 0


@pytest.mark.parametrize('optimized', [False, True])
def test_changed_source_is_left_for_resume_not_reported_complete(book, optimized):
    store, document = book
    job = job_for(store, document, segment_ranges=[(0, 0)])
    segment = store.list_segments(document.id)[0]

    class ChangeSourceBeforeResponse(EchoProvider):
        async def translate(self, request, model):
            store.update_segment_source(segment.id, 'Corrected source.')
            return await super().translate(request, model)

        async def complete(self, messages, model, **kwargs):
            store.update_segment_source(segment.id, 'Corrected source.')
            return await super().complete(messages, model, **kwargs)

    runner = 'run_optimized' if optimized else 'run'
    paused = asyncio.run(getattr(engine_for(store, ChangeSourceBeforeResponse()), runner)(job.id))
    assert paused.status is JobStatus.PAUSED
    assert paused.next_ordinal == 0
    assert not store.get_segment(segment.id).machine_translation
    completed = asyncio.run(getattr(engine_for(store, EchoProvider()), runner)(job.id))
    assert completed.status is JobStatus.COMPLETED
    assert 'Corrected source.' in store.get_segment(segment.id).machine_translation


@pytest.mark.parametrize('wrapper_started', [False, True])
def test_pause_before_background_task_starts_dispatches_nothing(book, wrapper_started):
    store, document = book
    provider = SlowProvider()
    manager = JobManager(store, engine_for(store, provider))
    job = job_for(store, document)

    async def scenario():
        manager.start(job.id)
        if wrapper_started:
            await asyncio.sleep(0)
        await manager.stop(job.id, JobStatus.PAUSED)
        await until(lambda: not manager.running(job.id))
        assert store.get_job(job.id).status is JobStatus.PAUSED
        assert provider.calls == 0
        await manager.shutdown()

    asyncio.run(scenario())


@pytest.mark.parametrize('timeout', [False, True])
def test_fatal_failure_drains_issued_sibling_before_unlocking(book, timeout):
    store, document = book
    job = job_for(store, document, batch_size=1, concurrency=2, max_concurrency=2)

    class FailingAndSlowProvider(EchoProvider):
        def __init__(self):
            self.started = threading.Event()
            self.release = threading.Event()
            self.calls = 0

        async def complete(self, messages, model, **kwargs):
            self.calls += 1
            if self.calls == 1:
                await until(self.started.is_set)
                raise RuntimeError('first request failed')

            def fake_http():
                self.started.set()
                assert self.release.wait(3)
                if timeout:
                    raise TimeoutError('simulated provider timeout')

            await asyncio.to_thread(fake_http)
            result = await super().complete(messages, model, **kwargs)
            return replace(result, cost_usd=0.25)

    provider = FailingAndSlowProvider()
    engine = engine_for(store, provider)

    async def scenario():
        task = asyncio.create_task(engine.run_optimized(job.id))
        try:
            await until(lambda: store.get_job(job.id).status is JobStatus.FAILED)
            with pytest.raises(DocumentBusyError), store.reserve_document(document.id):
                pass
            assert not task.done()
            provider.release.set()
            with pytest.raises(RuntimeError, match='first request failed'):
                await task
            assert provider.calls == 2
            second = store.list_segments(document.id)[1]
            assert bool(second.machine_translation) is not timeout
            assert store.get_job(job.id).total_cost_usd == (0 if timeout else 0.25)
            with store.reserve_document(document.id):
                pass
            assert (await engine_for(store, EchoProvider()).run_optimized(job.id)).status is JobStatus.COMPLETED
        finally:
            provider.release.set()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


def test_shutdown_preserves_cancelled_status_and_finishes_saved_response(book):
    store, document = book
    provider = SlowProvider()
    manager = JobManager(store, engine_for(store, provider))
    job = job_for(store, document, concurrency=1, max_concurrency=1)

    async def scenario():
        manager.start(job.id)
        await until(provider.started.is_set)
        await manager.stop(job.id, JobStatus.CANCELLED)
        shutdown = asyncio.create_task(manager.shutdown())
        try:
            await asyncio.sleep(0)
            assert store.get_job(job.id).status is JobStatus.CANCELLED
            assert not shutdown.done()
            with pytest.raises(DocumentBusyError), store.reserve_document(document.id):
                pass
            provider.release.set()
            await shutdown
            assert provider.calls == 2
            assert store.list_segments(document.id)[1].machine_translation
            assert store.get_job(job.id).status is JobStatus.CANCELLED
            with pytest.raises(ValueError, match='Cancelled'):
                manager.start(job.id)
        finally:
            provider.release.set()
            await shutdown

    asyncio.run(scenario())


@pytest.mark.parametrize('optimized', [False, True])
def test_cancelled_job_is_not_changed_to_failed_on_late_timeout(book, optimized):
    store, document = book
    job = job_for(store, document, concurrency=1, max_concurrency=1)

    class CancelledTimeout(EchoProvider):
        def fail(self):
            store.set_job_status(job.id, JobStatus.CANCELLED)
            raise TimeoutError('simulated late timeout')

        async def translate(self, request, model):
            self.fail()

        async def complete(self, messages, model, **kwargs):
            self.fail()

    engine = engine_for(store, CancelledTimeout())
    with pytest.raises(TimeoutError):
        asyncio.run((engine.run_optimized if optimized else engine.run)(job.id))
    assert store.get_job(job.id).status is JobStatus.CANCELLED
    with store.reserve_document(document.id):
        pass


@pytest.mark.parametrize('optimized', [False, True])
def test_cancel_before_worker_dispatch_makes_no_calls(book, optimized):
    store, document = book
    provider = SlowProvider()
    engine = engine_for(store, provider)
    job = job_for(store, document)

    async def scenario():
        task = asyncio.create_task((engine.run_optimized if optimized else engine.run)(job.id))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert provider.calls == 0
        assert store.get_job(job.id).status is JobStatus.PAUSED
        with store.reserve_document(document.id):
            pass

    asyncio.run(scenario())


def test_source_change_cannot_turn_cancelled_standard_job_into_paused(book):
    store, document = book
    segment = store.list_segments(document.id)[0]
    job = job_for(store, document)

    class CancelAndEdit(EchoProvider):
        async def translate(self, request, model):
            store.set_job_status(job.id, JobStatus.CANCELLED)
            store.update_segment_source(segment.id, 'Corrected source.')
            return await super().translate(request, model)

    result = asyncio.run(engine_for(store, CancelAndEdit()).run(job.id))
    assert result.status is JobStatus.CANCELLED
    assert not store.get_segment(segment.id).machine_translation
