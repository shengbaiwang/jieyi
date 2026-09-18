import asyncio
import json

import pytest

from jieyi.domain.models import JobStatus, TranslationResult
from jieyi.persistence import SQLiteStore
from jieyi.providers import ProviderRegistry
from jieyi.quality.output import output_problems
from jieyi.quality.service import refresh_segment_quality
from jieyi.workflow import TranslationEngine, create_document, create_job, create_project

SOURCE = "That to the highth of  this great Argument"
BAD = ('That to the highth of this great Argument **翻译：** 以阐明这伟大论证的极致， '
       '**说明：** - "highth" 是 "height" 的古体拼写。')
GOOD = "使我得以登临这宏大诗题的高处，"


def test_screenshot_and_structured_repair_are_detected():
    assert set(output_problems(SOURCE, BAD)) == {"source_echo", "model_commentary"}
    assert output_problems(SOURCE, json.dumps({"[[JY_PH_0000]]": BAD}))
    assert output_problems("Nor the deep Tract of Hell, say first what cause",
                           "Nor the deep Tract of Hell, say first what cause 先说是什么原因")


@pytest.mark.parametrize("source,target", [
    (SOURCE, GOOD),
    ("Explanation: the result is uncertain.", "说明：结果尚不确定。"),
    ("Translation: a study", "翻译：一项研究"),
    ("**Important**", "**重要**"),
    ("Eternal Providence", "永恒的天意（Eternal Providence）"),
    ("The poem quotes a phrase.", "诗中引用了一个短语。"),
])
def test_legitimate_content_is_preserved(source, target):
    assert not output_problems(source, target)


@pytest.mark.parametrize("optimized", [False, True])
@pytest.mark.parametrize("exhausted", [False, True])
def test_retry_quarantine_cost_and_quality_refresh(tmp_path, optimized, exhausted):
    store = SQLiteStore(tmp_path / "test.db")
    store.migrate()
    project = create_project(store, name="Poetry", source_lang="en", target_lang="zh-CN")
    document = create_document(store, project_id=project.id, title="Poetry",
                               source_format="txt", text=SOURCE)
    job = create_job(store, document_id=document.id, draft_provider="fake", draft_model="test")

    class Provider:
        calls = 0

        async def translate(self, request, model):
            if self.calls:
                assert "previous response violated" in request.context
            return self.result()

        async def complete(self, messages, model, **kwargs):
            if self.calls:
                assert "previous response violated" in messages[0]["content"]
            return self.result()

        def result(self):
            self.calls += 1
            return TranslationResult(text=BAD if exhausted or self.calls == 1 else GOOD,
                                     prompt_tokens=10, completion_tokens=20, cost_usd=0.01)

    provider = Provider()
    registry = ProviderRegistry()
    registry.register("fake", provider)
    engine = TranslationEngine(store, registry)
    asyncio.run(engine.run_optimized(job.id) if optimized else engine.run(job.id))
    segment = store.list_segments(document.id)[0]
    saved_job = store.get_job(job.id)
    assert saved_job.status is JobStatus.COMPLETED
    assert provider.calls == (3 if exhausted else 2)
    assert saved_job.total_cost_usd == pytest.approx(provider.calls * 0.01)
    assert segment.machine_translation == (None if exhausted else GOOD)
    if exhausted:
        assert any(i["code"] == "translation_deferred" for i in store.list_issues(document.id))
    # Historical pollution is flagged; a later correction clears just these findings.
    store.save_segment_draft(segment.id, BAD)
    refresh_segment_quality(store, segment.id)
    assert any(i["code"] == "source_echo" for i in store.list_issues(document.id))
    store.save_segment_draft(segment.id, GOOD)
    refresh_segment_quality(store, segment.id)
    assert not any(i["code"] in {"source_echo", "model_commentary"}
                   for i in store.list_issues(document.id))
