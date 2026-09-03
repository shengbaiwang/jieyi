import asyncio
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock

from test_epub import build_epub

from jieyi.domain.models import (
    CandidateStage,
    JobStatus,
    SegmentKind,
    SegmentStatus,
    TermEntry,
    TermStatus,
    TranslationResult,
    new_id,
)
from jieyi.persistence import SQLiteStore
from jieyi.providers import EchoProvider, ProviderRegistry
from jieyi.workflow import (
    TranslationEngine,
    create_document,
    create_epub_document,
    create_job,
    create_project,
)


def _sources_from_messages(messages) -> list[str]:
    content = messages[-1]["content"]
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict) and isinstance(payload.get("segments"), list):
        return [
            str(item.get("source", "")) for item in payload["segments"] if isinstance(item, dict)
        ]
    if "\n\nSOURCE:\n" in content:
        content = content.rsplit("\n\nSOURCE:\n", 1)[1]
    elif content.startswith("SOURCE:\n"):
        content = content[len("SOURCE:\n") :]
    if "\n\nCURRENT TRANSLATION:\n" in content:
        content = content.split("\n\nCURRENT TRANSLATION:\n", 1)[0]
    return [content]


class FailOnceProvider:
    def __init__(self):
        self.calls = 0

    async def translate(self, request, model):
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("simulated provider outage")
        return TranslationResult(text=f"translated:{request.segment.source_text}")


class PlaceholderRepairProvider:
    async def translate(self, request, model):
        if request.task is CandidateStage.REPAIR:
            return TranslationResult(text=request.segment.source_text)
        broken = request.segment.source_text.replace("[[JY_PH_0000]]", "", 1)
        return TranslationResult(text=broken)


class OptimizedPlaceholderRepairProvider:
    def __init__(self):
        self.draft_calls = 0
        self.repair_calls = 0

    async def complete(self, messages, model, **kwargs):
        del model, kwargs
        content = messages[-1]["content"]
        if "\n\nBROKEN TRANSLATION:\n" in content:
            self.repair_calls += 1
            broken = content.split("\n\nBROKEN TRANSLATION:\n", 1)[1]
            broken = broken.split("\n\nPLACEHOLDER ERROR:\n", 1)[0]
            return TranslationResult(
                text=broken.replace("[[JY_PH_9999]]", "").strip(),
                prompt_tokens=12,
                completion_tokens=4,
            )
        self.draft_calls += 1
        source = _sources_from_messages(messages)[0]
        return TranslationResult(
            text=f"translated:{source} [[JY_PH_9999]]",
            prompt_tokens=10,
            completion_tokens=5,
        )


class RetryPlaceholderRepairProvider:
    """Fail two repair validations before returning a valid marker-preserving result."""

    def __init__(self):
        self.draft_calls = 0
        self.repair_calls = 0
        self.broken_repair_inputs: list[str] = []

    async def complete(self, messages, model, **kwargs):
        del model, kwargs
        content = messages[-1]["content"]
        if "\n\nBROKEN TRANSLATION:\n" in content:
            self.repair_calls += 1
            broken = content.split("\n\nBROKEN TRANSLATION:\n", 1)[1]
            broken = broken.split("\n\nPLACEHOLDER ERROR:\n", 1)[0].strip()
            self.broken_repair_inputs.append(broken)
            if self.repair_calls < 3:
                return TranslationResult(
                    text="repair stripped every marker",
                    prompt_tokens=7,
                    completion_tokens=3,
                )
            source = content.split("MASKED SOURCE:\n", 1)[1]
            source = source.split("\n\nBROKEN TRANSLATION:\n", 1)[0].strip()
            return TranslationResult(text=source, prompt_tokens=8, completion_tokens=4)
        self.draft_calls += 1
        source = _sources_from_messages(messages)[0]
        return TranslationResult(
            text=source.replace("[[JY_PH_0000]]", "", 1),
            prompt_tokens=10,
            completion_tokens=5,
        )


class DuplicatePlaceholderProvider:
    async def complete(self, messages, model, **kwargs):
        del model, kwargs
        content = messages[-1]["content"]
        if "\n\nBROKEN TRANSLATION:\n" in content:
            raise AssertionError("duplicate-only damage should be repaired locally")
        source = _sources_from_messages(messages)[0]
        token = "[[JY_PH_0000]]"
        return TranslationResult(
            text=source.replace(token, f"{token}重复文本{token}", 1),
            prompt_tokens=10,
            completion_tokens=5,
        )


class CrossWiringBatchProvider:
    """Return a wrong source whenever a prompt contains more than one segment."""

    def __init__(self):
        self.max_sources = 0

    async def complete(self, messages, model, **kwargs):
        sources = _sources_from_messages(messages)
        self.max_sources = max(self.max_sources, len(sources))
        if len(sources) != 1:
            return TranslationResult(text=f"translated:{sources[-1]}")
        return TranslationResult(
            text=f"translated:{sources[0]}", prompt_tokens=10, completion_tokens=5
        )


class ConcurrentProbeProvider:
    def __init__(self):
        self.active = 0
        self.max_active = 0
        self.calls: list[tuple[list[dict[str, str]], list[str]]] = []

    async def complete(self, messages, model, **kwargs):
        del model, kwargs
        sources = _sources_from_messages(messages)
        self.calls.append((messages, sources))
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(0.02)
            source = sources[0]
            return TranslationResult(
                text=f"translated:{source}",
                prompt_tokens=sum(len(item["content"]) for item in messages),
                completion_tokens=max(1, len(source) // 4),
            )
        finally:
            self.active -= 1


class ContinuousQueueProvider:
    """The third segment must start before the deliberately slow first one can finish."""

    def __init__(self):
        self.third_started = asyncio.Event()
        self.started: list[str] = []

    async def complete(self, messages, model, **kwargs):
        del model, kwargs
        source = _sources_from_messages(messages)[0]
        self.started.append(source)
        if "Third segment" in source:
            self.third_started.set()
        if "First slow segment" in source:
            await asyncio.wait_for(self.third_started.wait(), timeout=1)
        await asyncio.sleep(0.01)
        return TranslationResult(
            text=f"translated:{source}",
            prompt_tokens=10,
            completion_tokens=5,
        )


class PermanentlyBrokenPlaceholderProvider:
    """Keep one protected segment invalid while translating its siblings normally."""

    async def complete(self, messages, model, **kwargs):
        del model, kwargs
        content = messages[-1]["content"]
        if "\n\nBROKEN TRANSLATION:\n" in content:
            return TranslationResult(text="invalid repair", prompt_tokens=8, completion_tokens=3)
        source = _sources_from_messages(messages)[0]
        if "[[JY_PH_0000]]" in source:
            return TranslationResult(
                text=source.replace("[[JY_PH_0000]]", "", 1),
                prompt_tokens=10,
                completion_tokens=5,
            )
        return TranslationResult(
            text=f"translated:{source}",
            prompt_tokens=10,
            completion_tokens=5,
        )


class EmptyUntilBudgetGrowsProvider:
    """Mimic a reasoning model that spends a small output budget on reasoning."""

    def __init__(self):
        self.max_tokens: list[int] = []
        self.reasoning_efforts: list[str | None] = []

    async def complete(self, messages, model, **kwargs):
        del model
        budget = kwargs["max_tokens"]
        self.max_tokens.append(budget)
        self.reasoning_efforts.append(kwargs.get("reasoning_effort"))
        if budget < 2_048:
            return TranslationResult(
                text="",
                prompt_tokens=100,
                completion_tokens=budget,
                reasoning_tokens=budget,
            )
        source = _sources_from_messages(messages)[0]
        return TranslationResult(
            text=f"translated:{source}",
            prompt_tokens=100,
            completion_tokens=120,
            reasoning_tokens=80,
        )


class ZeroTokenEmptyProvider:
    """Return a transport-success response with no output or usage tokens."""

    def __init__(self, *, finish_reason: str = ""):
        self.finish_reason = finish_reason
        self.max_tokens: list[int] = []

    async def complete(self, messages, model, **kwargs):
        del messages, model
        self.max_tokens.append(kwargs["max_tokens"])
        return TranslationResult(
            text="",
            raw_response=json.dumps(
                {
                    "id": "response-empty-1",
                    "choices": [
                        {
                            "message": {"content": None},
                            "finish_reason": self.finish_reason or None,
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                    },
                }
            ),
        )


class SelectiveEmptyProvider(ZeroTokenEmptyProvider):
    """Return empty results for one segment, while its siblings succeed."""

    def __init__(self):
        super().__init__()
        self.sources: list[str] = []

    async def complete(self, messages, model, **kwargs):
        source = _sources_from_messages(messages)[0]
        self.sources.append(source)
        if "changed in 2020" in source:
            return await super().complete(messages, model, **kwargs)
        return TranslationResult(text=f"translated:{source}", prompt_tokens=10, completion_tokens=5)


class SelectiveContentFilterProvider:
    """Filter one segment while translating its batch siblings successfully."""

    def __init__(self):
        self.filtered_calls = 0

    async def complete(self, messages, model, **kwargs):
        del model, kwargs
        source = _sources_from_messages(messages)[0]
        if "changed in 2020" in source:
            self.filtered_calls += 1
            return TranslationResult(
                text="",
                raw_response=json.dumps(
                    {
                        "id": "response-filtered-1",
                        "choices": [
                            {
                                "message": {"content": None},
                                "finish_reason": "content_filter",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                        },
                    }
                ),
            )
        return TranslationResult(
            text=f"translated:{source}",
            prompt_tokens=10,
            completion_tokens=5,
        )


class SelectiveHttpSafetyProvider:
    """Raise the HTTP 400 / 1301 error shape shown by GLM for one segment."""

    async def translate(self, request, model):
        del model
        if "changed in 2020" in request.segment.source_text:
            raise RuntimeError(
                "Provider request failed: HTTP 400: "
                "系统检测到输入或生成内容可能包含不安全或敏感内容（代码 1301）"
            )
        return TranslationResult(text=f"translated:{request.segment.source_text}")


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.store = SQLiteStore(Path(self.tempdir.name) / "jieyi.db")
        self.store.migrate()
        self.project = create_project(
            self.store,
            name="A Theory Book",
            source_lang="en",
            target_lang="zh-CN",
            style_guide="Use restrained academic Chinese.",
        )
        self.document = create_document(
            self.store,
            project_id=self.project.id,
            title="Chapter One",
            text="Agency matters.\n\nIt changed in 2020.\n\nThe conclusion follows.",
            source_format="txt",
        )
        registry = ProviderRegistry()
        registry.register("echo", EchoProvider())
        self.engine = TranslationEngine(self.store, registry)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_job_pauses_and_resumes_from_segment_checkpoint(self):
        job = create_job(
            self.store,
            document_id=self.document.id,
            draft_provider="echo",
            draft_model="dry-run",
        )
        paused = asyncio.run(self.engine.run(job.id, max_segments=1))
        self.assertEqual(paused.status, JobStatus.PAUSED)
        self.assertEqual(paused.next_ordinal, 1)

        completed = asyncio.run(self.engine.run(job.id))
        self.assertEqual(completed.status, JobStatus.COMPLETED)
        self.assertEqual(completed.next_ordinal, 3)
        self.assertTrue(
            all(
                item.status is SegmentStatus.MACHINE_TRANSLATED
                for item in self.store.list_segments(self.document.id)
            )
        )

    def test_optimized_runner_batches_and_resumes_with_usage_metrics(self):
        job = create_job(
            self.store,
            document_id=self.document.id,
            draft_provider="echo",
            draft_model="dry-run",
            batch_size=2,
            concurrency=2,
            draft_thinking=False,
        )
        paused = asyncio.run(self.engine.run_optimized(job.id, max_batches=1))
        self.assertEqual(paused.status, JobStatus.PAUSED)
        self.assertEqual(paused.next_ordinal, 2)
        progress = self.store.job_progress(job.id)
        self.assertEqual(progress["batch_count"], 1)
        self.assertGreater(progress["total_tokens"], 0)
        self.assertEqual(progress["reasoning_tokens"], 0)

        completed = asyncio.run(self.engine.run_optimized(job.id))
        self.assertEqual(completed.status, JobStatus.COMPLETED)
        self.assertEqual(completed.next_ordinal, 3)
        progress = self.store.job_progress(job.id)
        self.assertEqual(progress["batch_count"], 2)
        self.assertTrue(
            all(item.machine_translation for item in self.store.list_segments(self.document.id))
        )

    def test_optimized_runner_opens_a_fresh_token_budget_when_resumed(self):
        job = create_job(
            self.store,
            document_id=self.document.id,
            draft_provider="echo",
            draft_model="dry-run",
            batch_size=1,
            concurrency=1,
            max_concurrency=1,
            token_budget=1,
        )

        first_pause = asyncio.run(self.engine.run_optimized(job.id))
        self.assertEqual(first_pause.status, JobStatus.PAUSED)
        self.assertEqual(first_pause.next_ordinal, 1)
        first_total = self.store.job_progress(job.id)["total_tokens"]

        second_pause = asyncio.run(self.engine.run_optimized(job.id))
        self.assertEqual(second_pause.status, JobStatus.PAUSED)
        self.assertEqual(second_pause.next_ordinal, 2)
        self.assertGreater(self.store.job_progress(job.id)["total_tokens"], first_total)

        completed = asyncio.run(self.engine.run_optimized(job.id))
        self.assertEqual(completed.status, JobStatus.COMPLETED)
        self.assertEqual(completed.next_ordinal, 3)

    def test_optimized_runner_honors_selected_segment_ranges(self):
        job = create_job(
            self.store,
            document_id=self.document.id,
            draft_provider="echo",
            draft_model="dry-run",
            segment_ranges=[(1, 1)],
        )

        completed = asyncio.run(self.engine.run_optimized(job.id))

        self.assertEqual(completed.status, JobStatus.COMPLETED)
        segments = self.store.list_segments(self.document.id)
        self.assertIsNone(segments[0].machine_translation)
        self.assertIsNotNone(segments[1].machine_translation)
        self.assertIsNone(segments[2].machine_translation)
        progress = self.store.job_progress(job.id)
        self.assertEqual(progress["total_segments"], 1)
        self.assertEqual(progress["processed_segments"], 1)
        self.assertEqual(progress["recipe"]["segment_ranges"], ((1, 1),))

    def test_optimized_runner_isolates_sources_to_prevent_cross_wiring(self):
        registry = ProviderRegistry()
        provider = CrossWiringBatchProvider()
        registry.register("cross-wiring", provider)
        engine = TranslationEngine(self.store, registry)
        job = create_job(
            self.store,
            document_id=self.document.id,
            draft_provider="cross-wiring",
            draft_model="test",
            batch_size=3,
            concurrency=1,
        )

        completed = asyncio.run(engine.run_optimized(job.id))

        self.assertEqual(completed.status, JobStatus.COMPLETED)
        for segment in self.store.list_segments(self.document.id):
            self.assertEqual(
                segment.machine_translation,
                f"translated:{segment.source_text}",
            )

        self.assertEqual(provider.max_sources, 1)

    def test_optimized_runner_parallelizes_isolated_cache_ready_prompts(self):
        self.store.add_term(
            TermEntry(
                id=new_id("term"),
                project_id=self.project.id,
                source="Agency",
                target="能动性",
                status=TermStatus.APPROVED,
            )
        )
        self.store.add_term(
            TermEntry(
                id=new_id("term"),
                project_id=self.project.id,
                source="conclusion",
                target="结论",
                status=TermStatus.APPROVED,
            )
        )
        provider = ConcurrentProbeProvider()
        registry = ProviderRegistry()
        registry.register("concurrent", provider)
        engine = TranslationEngine(self.store, registry)
        job = create_job(
            self.store,
            document_id=self.document.id,
            draft_provider="concurrent",
            draft_model="test",
            batch_size=3,
            concurrency=3,
        )

        completed = asyncio.run(engine.run_optimized(job.id))

        self.assertEqual(completed.status, JobStatus.COMPLETED)
        self.assertEqual(provider.max_active, 3)
        self.assertEqual(len(provider.calls), 3)
        self.assertTrue(all(len(sources) == 1 for _, sources in provider.calls))
        system_prompts = {messages[0]["content"] for messages, _ in provider.calls}
        self.assertEqual(len(system_prompts), 1)
        rendered_prompts = "\n".join(
            message["content"] for messages, _ in provider.calls for message in messages
        )
        self.assertEqual(rendered_prompts.count("Agency -> 能动性"), 1)
        self.assertEqual(rendered_prompts.count("conclusion -> 结论"), 1)
        for segment in self.store.list_segments(self.document.id):
            self.assertEqual(
                segment.machine_translation,
                f"translated:{segment.source_text}",
            )

    def test_optimized_draft_includes_location_and_neighbors_outside_selected_range(self):
        document = create_document(
            self.store,
            project_id=self.project.id,
            title="Context book",
            text="# Chapter One\n\nEarlier argument.\n\nThis approach works."
                 "\n\nFollowing evidence.\n\nDistant conclusion.",
            source_format="markdown",
        )
        segments = self.store.list_segments(document.id)
        self.store.confirm_segment(segments[1].id, "不应进入初译参考的旧译文。")
        provider = ConcurrentProbeProvider()
        registry = ProviderRegistry()
        registry.register("probe", provider)
        engine = TranslationEngine(self.store, registry)
        job = create_job(
            self.store, document_id=document.id,
            draft_provider="probe", draft_model="test",
            segment_ranges=[(2, 2)],
        )

        completed = asyncio.run(engine.run_optimized(job.id))

        self.assertEqual(completed.status, JobStatus.COMPLETED)
        self.assertEqual(len(provider.calls), 1)
        messages, sources = provider.calls[0]
        self.assertEqual(sources, ["This approach works."])
        context = messages[-1]["content"].split("\n\nSOURCE:\n", 1)[0]
        self.assertIn("# LOCATION\nChapter One", context)
        self.assertIn("# PREVIOUS SOURCE (segment 1)\nEarlier argument.", context)
        self.assertIn("# FOLLOWING SOURCE (segment 3)\nFollowing evidence.", context)
        self.assertIn("only the current source is the output target", context)
        self.assertNotIn("Distant conclusion.", context)
        self.assertNotIn("旧译文", context)
        self.assertNotIn("Earlier argument.", messages[0]["content"])
        self.assertIsNone(self.store.get_segment(segments[3].id).machine_translation)
        self.assertEqual(
            self.store.get_segment(segments[2].id).machine_translation,
            "translated:This approach works.",
        )

    def test_optimized_context_honors_neighbor_radius(self):
        for radius in (0, 2):
            with self.subTest(radius=radius):
                document = create_document(
                    self.store, project_id=self.project.id, title="Radius",
                    text="\n\n".join(f"Passage {index}." for index in range(5)),
                    source_format="txt",
                )
                provider = ConcurrentProbeProvider()
                registry = ProviderRegistry()
                registry.register("probe", provider)
                engine = TranslationEngine(self.store, registry)
                job = create_job(
                    self.store, document_id=document.id,
                    draft_provider="probe", draft_model="test",
                    segment_ranges=[(2, 2)],
                )
                job = self.store.create_job(replace(
                    job, id=new_id("job"), recipe=replace(job.recipe, neighbor_radius=radius),
                ))
                completed = asyncio.run(engine.run_optimized(job.id))
                self.assertEqual(completed.status, JobStatus.COMPLETED)
                messages, sources = provider.calls[0]
                self.assertEqual(sources, ["Passage 2."])
                prompt = messages[-1]["content"]
                for index in (0, 1, 3, 4):
                    self.assertEqual(f"Passage {index}." in prompt, radius == 2)

    def test_optimized_runner_continuously_refills_available_slots(self):
        document = create_document(
            self.store,
            project_id=self.project.id,
            title="Continuous queue",
            text=(
                "First slow segment.\n\n"
                "Second quick segment.\n\n"
                "Third segment."
            ),
            source_format="txt",
        )
        provider = ContinuousQueueProvider()
        registry = ProviderRegistry()
        registry.register("continuous", provider)
        engine = TranslationEngine(self.store, registry)
        job = create_job(
            self.store,
            document_id=document.id,
            draft_provider="continuous",
            draft_model="test",
            batch_size=1,
            concurrency=2,
            max_concurrency=2,
        )

        completed = asyncio.run(
            asyncio.wait_for(engine.run_optimized(job.id), timeout=2)
        )

        self.assertEqual(completed.status, JobStatus.COMPLETED)
        self.assertEqual(len(provider.started), 3)
        self.assertTrue(
            all(segment.machine_translation for segment in self.store.list_segments(document.id))
        )

    def test_optimized_runner_adapts_from_safe_start_to_configured_maximum(self):
        document = create_document(
            self.store,
            project_id=self.project.id,
            title="Adaptive concurrency",
            text="\n\n".join(f"Segment {index}." for index in range(12)),
            source_format="txt",
        )
        provider = ConcurrentProbeProvider()
        registry = ProviderRegistry()
        registry.register("adaptive", provider)
        engine = TranslationEngine(self.store, registry)
        job = create_job(
            self.store,
            document_id=document.id,
            draft_provider="adaptive",
            draft_model="test",
            batch_size=1,
            concurrency=2,
            max_concurrency=4,
        )

        completed = asyncio.run(engine.run_optimized(job.id))

        self.assertEqual(completed.status, JobStatus.COMPLETED)
        self.assertEqual(provider.max_active, 4)

    def test_optimized_runner_quarantines_invalid_segment_without_discarding_siblings(self):
        document = create_document(
            self.store,
            project_id=self.project.id,
            title="Quarantined segment",
            text=(
                "First ordinary paragraph.\n\n"
                "The argument follows (Smith 2020).\n\n"
                "Third ordinary paragraph."
            ),
            source_format="txt",
        )
        registry = ProviderRegistry()
        registry.register("permanently-broken", PermanentlyBrokenPlaceholderProvider())
        registry.register("echo", EchoProvider())
        engine = TranslationEngine(self.store, registry)
        job = create_job(
            self.store,
            document_id=document.id,
            draft_provider="permanently-broken",
            draft_model="test",
            batch_size=3,
            concurrency=3,
            max_concurrency=3,
        )

        completed = asyncio.run(engine.run_optimized(job.id))

        self.assertEqual(completed.status, JobStatus.COMPLETED)
        segments = self.store.list_segments(document.id)
        self.assertIsNotNone(segments[0].machine_translation)
        self.assertIsNone(segments[1].machine_translation)
        self.assertIsNotNone(segments[2].machine_translation)
        deferred = [
            issue for issue in self.store.list_issues(document.id)
            if issue["code"] == "translation_deferred"
        ]
        self.assertEqual(len(deferred), 1)
        self.assertEqual(deferred[0]["severity"], "error")
        self.assertEqual(self.store.job_progress(job.id)["deferred_segments"], 1)
        failure = self.store.list_audit_events("segment", segments[1].id)[-1]
        self.assertIs(failure["payload"]["deferred"], True)

    def test_optimized_runner_retries_empty_reasoning_response_with_larger_budget(self):
        document = create_document(
            self.store,
            project_id=self.project.id,
            title="Short source",
            text="Bref.",
            source_format="txt",
        )
        provider = EmptyUntilBudgetGrowsProvider()
        registry = ProviderRegistry()
        registry.register("budget-sensitive", provider)
        engine = TranslationEngine(self.store, registry)
        job = create_job(
            self.store,
            document_id=document.id,
            draft_provider="budget-sensitive",
            draft_model="reasoning-model",
            concurrency=1,
            draft_reasoning_effort="low",
        )

        completed = asyncio.run(engine.run_optimized(job.id))

        self.assertEqual(completed.status, JobStatus.COMPLETED)
        segment = self.store.list_segments(document.id)[0]
        self.assertEqual(segment.machine_translation, "translated:Bref.")
        self.assertEqual(provider.max_tokens, [512, 2_048])
        self.assertEqual(provider.reasoning_efforts, ["low", "low"])
        progress = self.store.job_progress(job.id)
        self.assertEqual(progress["prompt_tokens"], 200)
        self.assertEqual(progress["completion_tokens"], 632)
        self.assertEqual(progress["reasoning_tokens"], 592)

    def test_zero_token_empty_response_retries_once_without_budget_growth(self):
        document = create_document(
            self.store,
            project_id=self.project.id,
            title="Empty upstream response",
            text="Bref.",
            source_format="txt",
        )
        provider = ZeroTokenEmptyProvider()
        registry = ProviderRegistry()
        registry.register("empty", provider)
        engine = TranslationEngine(self.store, registry)
        job = create_job(
            self.store,
            document_id=document.id,
            draft_provider="empty",
            draft_model="empty-model",
            concurrency=1,
        )

        completed = asyncio.run(engine.run_optimized(job.id))

        self.assertEqual(provider.max_tokens, [512, 512])
        self.assertEqual(completed.status, JobStatus.COMPLETED)
        self.assertEqual(completed.next_ordinal, 1)
        self.assertIsNone(completed.last_error)
        segment = self.store.list_segments(document.id)[0]
        self.assertIsNone(segment.machine_translation)
        self.assertEqual(self.store.list_candidates(segment.id), [])
        self.assertEqual(self.store.job_progress(job.id)["deferred_segments"], 1)
        failures = [
            event
            for event in self.store.list_audit_events("segment", segment.id)
            if event["action"] == "provider_failure"
        ]
        self.assertEqual(failures[-1]["payload"]["kind"], "upstream_empty_response")
        self.assertEqual(len(failures[-1]["payload"]["attempts"]), 2)
        self.assertIs(failures[-1]["payload"]["deferred"], True)
        self.assertNotIn("内容策略", failures[-1]["payload"]["message"])
        issue = self.store.list_issues(document.id)[0]
        self.assertEqual(issue["code"], "translation_deferred")
        self.assertEqual(issue["severity"], "error")
        self.assertIn("重试", issue["message"])

    def test_empty_segment_preserves_siblings_and_resume_checkpoint(self):
        provider = SelectiveEmptyProvider()
        registry = ProviderRegistry()
        registry.register("empty", provider)
        engine = TranslationEngine(self.store, registry)
        job = create_job(
            self.store, document_id=self.document.id,
            draft_provider="empty", draft_model="test", batch_size=2, concurrency=3,
        )

        paused = asyncio.run(engine.run_optimized(job.id, max_batches=1))
        self.assertEqual(paused.status, JobStatus.PAUSED)
        self.assertEqual(paused.next_ordinal, 2)
        segments = self.store.list_segments(self.document.id)
        self.assertEqual(segments[0].machine_translation, "translated:Agency matters.")
        self.assertIsNone(segments[1].machine_translation)
        self.assertIsNone(segments[2].machine_translation)
        self.assertEqual(len(provider.max_tokens), 2)

        completed = asyncio.run(engine.run_optimized(job.id))
        self.assertEqual(completed.status, JobStatus.COMPLETED)
        self.assertEqual(completed.next_ordinal, 3)
        self.assertEqual(len(provider.max_tokens), 2)
        self.assertEqual(provider.sources.count("Agency matters."), 1)
        self.assertEqual(
            self.store.get_segment(segments[2].id).machine_translation,
            "translated:The conclusion follows.",
        )
        progress = self.store.job_progress(job.id)
        self.assertEqual(progress["deferred_segments"], 1)
        self.assertEqual(progress["total_tokens"], 30)
        self.assertEqual(progress["processed_segments"], 3)

    def test_transient_empty_response_succeeds_without_deferral(self):
        provider = ZeroTokenEmptyProvider()
        provider.complete = AsyncMock(side_effect=[
            TranslationResult(text="", prompt_tokens=7),
            TranslationResult(text="translated:Agency matters.", prompt_tokens=10, completion_tokens=5),
        ])
        registry = ProviderRegistry()
        registry.register("empty", provider)
        engine = TranslationEngine(self.store, registry)
        job = create_job(
            self.store, document_id=self.document.id,
            draft_provider="empty", draft_model="test", segment_ranges=[(0, 0)],
        )
        completed = asyncio.run(engine.run_optimized(job.id))
        self.assertEqual(completed.status, JobStatus.COMPLETED)
        calls = provider.complete.call_args_list
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0].kwargs["max_tokens"], calls[1].kwargs["max_tokens"])
        progress = self.store.job_progress(job.id)
        self.assertEqual(progress["deferred_segments"], 0)
        self.assertEqual(progress["total_tokens"], 22)
        self.assertEqual(
            self.store.list_segments(self.document.id)[0].machine_translation,
            "translated:Agency matters.",
        )

    def test_optimized_transport_failure_still_stops_for_retry(self):
        provider = ZeroTokenEmptyProvider()
        provider.complete = AsyncMock(side_effect=RuntimeError("HTTP 401: invalid API key"))
        registry = ProviderRegistry()
        registry.register("broken", provider)
        engine = TranslationEngine(self.store, registry)
        job = create_job(
            self.store, document_id=self.document.id,
            draft_provider="broken", draft_model="test", concurrency=1, max_concurrency=1,
        )
        with self.assertRaisesRegex(RuntimeError, "HTTP 401"):
            asyncio.run(engine.run_optimized(job.id))
        failed = self.store.get_job(job.id)
        self.assertEqual(failed.status, JobStatus.FAILED)
        self.assertEqual(failed.next_ordinal, 0)
        self.assertEqual(self.store.job_progress(job.id)["deferred_segments"], 0)

    def test_content_filter_is_deferred_without_discarding_batch_siblings(self):
        provider = SelectiveContentFilterProvider()
        registry = ProviderRegistry()
        registry.register("selective-filter", provider)
        registry.register("echo", EchoProvider())
        engine = TranslationEngine(self.store, registry)
        job = create_job(
            self.store,
            document_id=self.document.id,
            draft_provider="selective-filter",
            draft_model="filter-model",
            batch_size=3,
            concurrency=3,
        )

        completed = asyncio.run(engine.run_optimized(job.id))

        self.assertEqual(completed.status, JobStatus.COMPLETED)
        self.assertEqual(completed.next_ordinal, 3)
        self.assertEqual(provider.filtered_calls, 1)
        segments = self.store.list_segments(self.document.id)
        self.assertEqual(
            segments[0].machine_translation,
            f"translated:{segments[0].source_text}",
        )
        self.assertIsNone(segments[1].machine_translation)
        self.assertEqual(
            segments[2].machine_translation,
            f"translated:{segments[2].source_text}",
        )
        failures = [
            event
            for event in self.store.list_audit_events("segment", segments[1].id)
            if event["action"] == "provider_failure"
        ]
        payload = failures[-1]["payload"]
        self.assertEqual(payload["kind"], "content_filtered")
        self.assertEqual(payload["attempts"][0]["finish_reason"], "content_filter")
        self.assertEqual(payload["attempts"][0]["response_id"], "response-filtered-1")
        self.assertNotIn(segments[1].source_text, json.dumps(payload))

    def test_human_confirmation_is_separate_and_audited(self):
        job = create_job(
            self.store,
            document_id=self.document.id,
            draft_provider="echo",
            draft_model="dry-run",
        )
        asyncio.run(self.engine.run(job.id))
        segment = self.store.list_segments(self.document.id)[0]
        self.store.confirm_segment(segment.id, "能动性很重要。", "Terminology approved")
        confirmed = self.store.get_segment(segment.id)
        self.assertEqual(confirmed.status, SegmentStatus.HUMAN_CONFIRMED)
        self.assertEqual(confirmed.accepted_translation, "能动性很重要。")
        events = self.store.list_audit_events("segment", segment.id)
        self.assertEqual(events[-1]["action"], "confirmed")

    def test_only_relevant_approved_terms_enter_the_context(self):
        self.store.add_term(
            TermEntry(
                id=new_id("term"),
                project_id=self.project.id,
                source="Agency",
                target="能动性",
                status=TermStatus.APPROVED,
            )
        )
        self.store.add_term(
            TermEntry(
                id=new_id("term"),
                project_id=self.project.id,
                source="unrelated",
                target="无关",
                status=TermStatus.APPROVED,
            )
        )
        segment = self.store.list_segments(self.document.id)[0]
        context, terms = self.engine.context_compiler.compile(
            self.project, segment, neighbor_radius=1, max_chars=10_000
        )
        self.assertIn("Agency -> 能动性", context)
        self.assertNotIn("unrelated -> 无关", context)
        self.assertEqual([term.source for term in terms], ["Agency"])

    def test_alias_and_context_sense_are_rendered_as_conditional_constraints(self):
        self.store.add_term(
            TermEntry(
                id=new_id("term"),
                project_id=self.project.id,
                source="artificial intelligence",
                target="人工智能",
                aliases=("AI",),
                sense="技术领域",
                context_keywords=("model",),
                disambiguation="仅指计算机科学中的 AI",
                status=TermStatus.APPROVED,
            )
        )
        document = create_document(
            self.store,
            project_id=self.project.id,
            title="AI chapter",
            text="An AI model shows agency.",
            source_format="txt",
        )
        segment = self.store.list_segments(document.id)[0]
        context, terms = self.engine.context_compiler.compile(
            self.project, segment, neighbor_radius=1, max_chars=10_000
        )
        self.assertIn("CONDITIONALLY APPROVED TERMINOLOGY", context)
        self.assertIn("NOT APPLICABLE", context)
        self.assertIn("仅指计算机科学中的 AI", context)
        self.assertIn("artificial intelligence | AI -> 人工智能 [CONDITIONAL]", context)
        self.assertEqual([term.source for term in terms], ["artificial intelligence"])

    def test_provider_failure_preserves_checkpoint_for_retry(self):
        provider = FailOnceProvider()
        registry = ProviderRegistry()
        registry.register("flaky", provider)
        engine = TranslationEngine(self.store, registry)
        job = create_job(
            self.store,
            document_id=self.document.id,
            draft_provider="flaky",
            draft_model="test",
        )

        with self.assertRaisesRegex(RuntimeError, "simulated provider outage"):
            asyncio.run(engine.run(job.id))
        failed = self.store.get_job(job.id)
        self.assertEqual(failed.status, JobStatus.FAILED)
        self.assertEqual(failed.next_ordinal, 1)

        completed = asyncio.run(engine.run(job.id))
        self.assertEqual(completed.status, JobStatus.COMPLETED)
        first_segment = self.store.list_segments(self.document.id)[0]
        self.assertEqual(len(self.store.list_candidates(first_segment.id)), 1)

    def test_placeholder_damage_triggers_focused_repair(self):
        document = create_document(
            self.store,
            project_id=self.project.id,
            title="Citations",
            text="The argument follows (Smith 2020) [^2].",
            source_format="txt",
        )
        registry = ProviderRegistry()
        registry.register("repairing", PlaceholderRepairProvider())
        engine = TranslationEngine(self.store, registry)
        job = create_job(
            self.store,
            document_id=document.id,
            draft_provider="repairing",
            draft_model="test",
        )
        completed = asyncio.run(engine.run(job.id))
        self.assertEqual(completed.status, JobStatus.COMPLETED)
        segment = self.store.list_segments(document.id)[0]
        self.assertEqual(
            segment.source_text, self.store.get_segment(segment.id).machine_translation
        )
        stages = [item["stage"] for item in self.store.list_candidates(segment.id)]
        self.assertEqual(stages, ["draft", "repair"])

    def test_optimized_runner_repairs_hallucinated_placeholder_per_segment(self):
        document = create_document(
            self.store,
            project_id=self.project.id,
            title="No placeholders",
            text="Ordinary source text.",
            source_format="txt",
        )
        provider = OptimizedPlaceholderRepairProvider()
        registry = ProviderRegistry()
        registry.register("optimized-repairing", provider)
        engine = TranslationEngine(self.store, registry)
        job = create_job(
            self.store,
            document_id=document.id,
            draft_provider="optimized-repairing",
            draft_model="test",
        )

        completed = asyncio.run(engine.run_optimized(job.id))

        self.assertEqual(completed.status, JobStatus.COMPLETED)
        segment = self.store.list_segments(document.id)[0]
        self.assertEqual(segment.machine_translation, "translated:Ordinary source text.")
        stages = [item["stage"] for item in self.store.list_candidates(segment.id)]
        self.assertEqual(stages, ["draft", "repair"])
        self.assertEqual(provider.draft_calls, 1)
        self.assertEqual(provider.repair_calls, 1)

    def test_optimized_runner_retries_invalid_repairs_from_original_draft(self):
        document = create_document(
            self.store,
            project_id=self.project.id,
            title="Repeated placeholder repair",
            text="The argument follows (Smith 2020).",
            source_format="txt",
        )
        provider = RetryPlaceholderRepairProvider()
        registry = ProviderRegistry()
        registry.register("retry-repairing", provider)
        engine = TranslationEngine(self.store, registry)
        job = create_job(
            self.store,
            document_id=document.id,
            draft_provider="retry-repairing",
            draft_model="test",
        )

        completed = asyncio.run(engine.run_optimized(job.id))

        self.assertEqual(completed.status, JobStatus.COMPLETED)
        segment = self.store.list_segments(document.id)[0]
        self.assertEqual(segment.machine_translation, segment.source_text)
        self.assertEqual(provider.draft_calls, 1)
        self.assertEqual(provider.repair_calls, 3)
        self.assertEqual(len(set(provider.broken_repair_inputs)), 1)
        stages = [item["stage"] for item in self.store.list_candidates(segment.id)]
        self.assertEqual(stages, ["draft", "repair"])

    def test_optimized_runner_removes_surplus_duplicate_without_model_repair(self):
        document = create_document(
            self.store,
            project_id=self.project.id,
            title="Duplicate placeholder",
            text="The argument follows (Smith 2020).",
            source_format="txt",
        )
        registry = ProviderRegistry()
        registry.register("duplicate-placeholder", DuplicatePlaceholderProvider())
        engine = TranslationEngine(self.store, registry)
        job = create_job(
            self.store,
            document_id=document.id,
            draft_provider="duplicate-placeholder",
            draft_model="test",
        )

        completed = asyncio.run(engine.run_optimized(job.id))

        self.assertEqual(completed.status, JobStatus.COMPLETED)
        segment = self.store.list_segments(document.id)[0]
        self.assertEqual(
            segment.machine_translation,
            "The argument follows (Smith 2020)重复文本.",
        )
        stages = [item["stage"] for item in self.store.list_candidates(segment.id)]
        self.assertEqual(stages, ["draft"])

    def test_confirmed_translation_becomes_fuzzy_tm_context(self):
        first = self.store.list_segments(self.document.id)[0]
        self.store.confirm_segment(first.id, "能动性很重要。", "Approved wording")
        other = create_document(
            self.store,
            project_id=self.project.id,
            title="Related",
            text="Agency really matters.",
            source_format="txt",
        )
        segment = self.store.list_segments(other.id)[0]
        matches = self.store.search_translation_memory(
            self.project.id, segment.source_text, threshold=0.6, limit=3
        )
        self.assertEqual(matches[0].target_text, "能动性很重要。")
        self.assertLess(matches[0].similarity, 1.0)

        context, _ = self.engine.context_compiler.compile(
            self.project,
            segment,
            neighbor_radius=0,
            max_chars=10_000,
            tm_threshold=0.6,
        )
        self.assertIn("TRANSLATION MEMORY", context)
        self.assertIn("能动性很重要。", context)

    def test_epub_import_uses_metadata_title_and_canonical_segments(self):
        document = create_epub_document(
            self.store,
            project_id=self.project.id,
            file_data=build_epub(),
        )
        self.assertEqual(document.title, "Test Theory Book")
        self.assertEqual(document.source_format, "epub")
        segments = self.store.list_segments(document.id)
        self.assertEqual(segments[0].source_text, "Second Chapter")
        self.assertEqual(segments[3].kind, SegmentKind.FOOTNOTE)


if __name__ == "__main__":
    unittest.main()
