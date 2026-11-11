import asyncio

import pytest
from fastapi.testclient import TestClient

from jieyi.api.app import create_app
from jieyi.domain.models import IssueSeverity, JobStatus, QualityIssue, TranslationResult
from jieyi.persistence import SQLiteStore
from jieyi.providers import ProviderRegistry
from jieyi.quality import refresh_segment_quality, reindex_all_quality
from jieyi.quality.checks import DETECTOR_VERSION
from jieyi.workflow import TranslationEngine, create_document, create_job, create_project

SOURCE = "How bow'd the woods beneath their sturdy stroke!"
TRANSLATION = "林木在他们有力的砍伐下纷纷低垂！"


def create_book(store, *, source_lang="en", target_lang="zh-CN"):
    project = create_project(
        store, name="Poetry", source_lang=source_lang, target_lang=target_lang,
    )
    document = create_document(
        store, project_id=project.id, title="Elegy", source_format="txt", text=SOURCE,
    )
    job = create_job(
        store, document_id=document.id, draft_provider="copy", draft_model="test",
    )
    return document, store.list_segments(document.id)[0], job


class CopyProvider:
    async def translate(self, request, model):
        return TranslationResult(text=request.segment.source_text)

    async def complete(self, messages, model, **kwargs):
        return TranslationResult(text=SOURCE)


@pytest.mark.parametrize("optimized", [False, True])
@pytest.mark.parametrize("correction", ["draft", "confirm", "reviewed"])
def test_copied_translation_enters_review_and_correction_clears_it(
    tmp_path, optimized, correction,
):
    app = create_app(str(tmp_path / "book.db"))
    store = app.state.store
    document, segment, job = create_book(store)
    providers = ProviderRegistry()
    providers.register("copy", CopyProvider())
    engine = TranslationEngine(store, providers)

    result = asyncio.run(engine.run_optimized(job.id) if optimized else engine.run(job.id))
    assert result.status is JobStatus.COMPLETED
    assert store.get_segment(segment.id).machine_translation == SOURCE

    with TestClient(app) as client:
        issue_url = f"/documents/{document.id}/issues"
        queue_url = f"/documents/{document.id}/human-review-queue"
        issues = client.get(issue_url).json()
        issue = next(item for item in issues if item["code"] == "untranslated_text")
        assert issue["severity"] == "error"
        assert issue["details"]["detector_version"] == DETECTOR_VERSION
        queue = client.get(queue_url).json()
        assert len(queue) == 1
        assert queue[0]["segment_id"] == segment.id
        assert queue[0]["reason"] == "error"
        assert queue[0]["translation"] == SOURCE

        if correction == "reviewed":
            store.set_reviewed_translation(segment.id, TRANSLATION)
            refresh_segment_quality(store, segment.id)
        else:
            response = client.patch(
                f"/segments/{segment.id}/{correction}", json={"translation": TRANSLATION},
            )
            assert response.status_code == 200
        assert client.get(queue_url).json() == []
        assert not any(item["code"] == "untranslated_text" for item in client.get(issue_url).json())


def test_startup_rechecks_historical_copies_and_preserves_other_findings(tmp_path):
    path = tmp_path / "historical.db"
    store = SQLiteStore(path)
    store.migrate()
    document, segment, job = create_book(store)
    store.set_machine_translation(segment.id, SOURCE)
    store.replace_issues(
        job.id, segment.id,
        [QualityIssue("reviewer_attention", "核实诗句的意象。", IssueSeverity.WARNING)],
        target_text=SOURCE, detector_version="7",
    )
    store.set_meta("quality_detector_version", "7")

    app = create_app(str(path))
    with TestClient(app) as client:
        issues = client.get(f"/documents/{document.id}/issues").json()
        assert {item["code"] for item in issues} == {"untranslated_text", "reviewer_attention"}
        assert app.state.store.get_meta("quality_detector_version") == DETECTOR_VERSION
        queue = client.get(f"/documents/{document.id}/human-review-queue").json()
        assert [item["segment_id"] for item in queue] == [segment.id]
        assert queue[0]["reason"] == "error"
        before = app.state.store.get_segment(segment.id)
        assert reindex_all_quality(app.state.store) == 0
        assert app.state.store.get_segment(segment.id) == before
        assert len(app.state.store.list_issues(document.id)) == 2


@pytest.mark.parametrize("optimized", [False, True])
def test_same_language_variants_do_not_report_untranslated_text(tmp_path, optimized):
    app = create_app(str(tmp_path / "same-language.db"))
    store = app.state.store
    document, _, job = create_book(store, source_lang="en-US", target_lang="en-GB")
    providers = ProviderRegistry()
    providers.register("copy", CopyProvider())
    engine = TranslationEngine(store, providers)
    asyncio.run(engine.run_optimized(job.id) if optimized else engine.run(job.id))
    with TestClient(app) as client:
        assert client.get(f"/documents/{document.id}/issues").json() == []
        assert client.get(f"/documents/{document.id}/human-review-queue").json() == []
