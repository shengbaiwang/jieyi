from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
from threading import Event

import pytest

from jieyi.domain.models import Segment, SegmentKind
from jieyi.ingestion.pdf_export_cache import PDFExportCache


class Store:
    def __init__(self):
        self.metadata = {"layout": [{"source_text": "Source", "regions": []}]}
        self.source = b"%PDF-original"
        self.segments = [Segment(
            id="segment", document_id="book", stable_key="source", ordinal=0,
            kind=SegmentKind.PARAGRAPH, source_text="Source", source_refs=("pdf:page:1",),
            machine_translation="初稿", edited_translation="编辑", reviewed_translation="审校",
            accepted_translation="确认",
        )]
        self.deleted = []

    def get_pdf_metadata(self, _):
        return deepcopy(self.metadata)

    def get_original_pdf(self, _):
        return self.source

    def list_segments(self, _):
        return list(self.segments)

    def list_deleted_source_segments(self, _):
        return list(self.deleted)


def test_same_content_hits_and_bilingual_has_its_own_result():
    calls = []

    def exporter(snapshot, document_id, *, bilingual):
        calls.append(bilingual)
        return b"bilingual" if bilingual else b"translated"

    store, cache = Store(), PDFExportCache(exporter=exporter)
    assert cache.export(store, "book") == b"translated"
    assert cache.export(store, "book") == b"translated"
    assert cache.export(store, "book", bilingual=True) == b"bilingual"
    assert cache.export(store, "book", bilingual=True) == b"bilingual"
    assert calls == [False, True]


def test_non_exported_draft_changes_do_not_rebuild_an_accepted_translation():
    calls = []

    def exporter(*args, **kwargs):
        calls.append(True)
        return b"accepted result"

    store, cache = Store(), PDFExportCache(exporter=exporter)
    cache.export(store, "book")
    store.segments[0] = replace(store.segments[0], machine_translation="新机器初稿")
    assert cache.export(store, "book") == b"accepted result"
    assert len(calls) == 1


@pytest.mark.parametrize("change", [
    "accepted", "reviewed", "edited", "machine", "source_text", "source_refs", "kind",
    "deleted", "layout", "source_pdf", "font",
])
def test_output_changes_invalidate_the_cache(change, monkeypatch):
    calls = []

    def exporter(*args, **kwargs):
        calls.append(True)
        return str(len(calls)).encode()

    store = Store()
    # Test each translation state as the effective, highest-priority value.
    if change == "reviewed":
        store.segments[0] = replace(store.segments[0], accepted_translation=None)
    elif change == "edited":
        store.segments[0] = replace(
            store.segments[0], accepted_translation=None, reviewed_translation=None,
        )
    elif change == "machine":
        store.segments[0] = replace(
            store.segments[0], accepted_translation=None, reviewed_translation=None,
            edited_translation=None,
        )
    cache = PDFExportCache(exporter=exporter)
    assert cache.export(store, "book") == b"1"
    if change in {"accepted", "reviewed", "edited", "machine"}:
        store.segments[0] = replace(store.segments[0], **{change + "_translation": "修改"})
    elif change == "source_text":
        store.segments[0] = replace(store.segments[0], source_text="Edited source")
    elif change == "source_refs":
        store.segments[0] = replace(store.segments[0], source_refs=("pdf:page:2",))
    elif change == "kind":
        store.segments[0] = replace(store.segments[0], kind=SegmentKind.HEADING)
    elif change == "deleted":
        store.deleted = store.segments
        store.segments = []
    elif change == "layout":
        store.metadata["geometry_version"] = "new"
    elif change == "source_pdf":
        store.source = b"%PDF-replaced"
    else:
        monkeypatch.setenv("JIEYI_PDF_FONT", "/custom/font.ttf")
    assert cache.export(store, "book") == b"2"
    assert cache.export(store, "book") == b"2"


def test_concurrent_downloads_share_one_build():
    entered, release = Event(), Event()
    calls = []

    def exporter(*args, **kwargs):
        calls.append(True)
        entered.set()
        assert release.wait(5)
        return b"result"

    store, cache = Store(), PDFExportCache(exporter=exporter)
    with ThreadPoolExecutor(max_workers=3) as pool:
        first = pool.submit(cache.export, store, "book")
        assert entered.wait(5)
        second = pool.submit(cache.export, store, "book")
        third = pool.submit(cache.export, store, "book")
        release.set()
        assert [future.result(5) for future in (first, second, third)] == [b"result"] * 3
    assert len(calls) == 1


def test_failed_build_is_not_cached():
    calls = []

    def exporter(*args, **kwargs):
        calls.append(True)
        if len(calls) == 1:
            raise ValueError("invalid source")
        return b"fixed"

    store, cache = Store(), PDFExportCache(exporter=exporter)
    with pytest.raises(ValueError, match="invalid source"):
        cache.export(store, "book")
    assert cache.export(store, "book") == b"fixed"
    assert cache.export(store, "book") == b"fixed"
    assert len(calls) == 2


def test_byte_limit_evicts_older_results_and_skips_oversize():
    calls = []

    def exporter(snapshot, *args, **kwargs):
        calls.append(snapshot.source)
        return b"huge" * 8 if snapshot.source == b"large" else b"1234"

    store, cache = Store(), PDFExportCache(max_bytes=6, exporter=exporter)
    cache.export(store, "book")
    store.source = b"second"
    cache.export(store, "book")
    store.source = b"%PDF-original"
    cache.export(store, "book")
    assert len(calls) == 3  # Both four-byte entries cannot remain in six bytes.
    store.source = b"large"
    cache.export(store, "book")
    cache.export(store, "book")
    assert len(calls) == 5
    assert cache._bytes <= 6


def test_entry_limit_is_lru():
    calls = []

    def exporter(snapshot, *args, **kwargs):
        calls.append(snapshot.source)
        return snapshot.source

    store, cache = Store(), PDFExportCache(max_entries=2, exporter=exporter)
    for source in (b"a", b"b", b"a", b"c", b"a", b"b"):
        store.source = source
        cache.export(store, "book")
    assert calls == [b"a", b"b", b"c", b"b"]


def test_exporter_reads_the_revision_used_for_its_key():
    store = Store()

    def exporter(snapshot, *args, **kwargs):
        store.source = b"new source"
        store.segments[0] = replace(store.segments[0], accepted_translation="新译文")
        return snapshot.get_original_pdf("book")

    cache = PDFExportCache(exporter=exporter)
    assert cache.export(store, "book") == b"%PDF-original"
