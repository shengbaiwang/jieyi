"""Bounded PDF downloads with content-based invalidation and shared in-flight work."""

from __future__ import annotations

import hashlib
import json
import os
from collections import OrderedDict
from concurrent.futures import Future
from copy import deepcopy
from threading import Lock

from jieyi.ingestion import pdf_export
from jieyi.ingestion.pdf import GEOMETRY_VERSION
from jieyi.ingestion.typography import TYPOGRAPHY_VERSION


class _ExportSnapshot:
    """Keep the cache fingerprint and the exported content on the same revision."""

    def __init__(self, store, document_id):
        self.store = store
        self.document_id = document_id
        self.metadata = store.get_pdf_metadata(document_id)
        self.source = store.get_original_pdf(document_id)
        self.segments = store.list_segments(document_id)
        self.deleted = store.list_deleted_source_segments(document_id)

    def get_pdf_metadata(self, document_id):
        return deepcopy(self.metadata)

    def get_original_pdf(self, document_id):
        return self.source

    def list_segments(self, document_id):
        return self.segments

    def list_deleted_source_segments(self, document_id):
        return self.deleted

    def update_pdf_metadata(self, document_id, metadata):
        # Preserve the existing one-time layout migration performed by the exporter.
        self.store.update_pdf_metadata(document_id, metadata)
        self.metadata = deepcopy(metadata)

    def fingerprint(self, bilingual):
        def segment_content(segment):
            return (
                segment.id, segment.ordinal, segment.kind, segment.source_text,
                segment.source_refs, pdf_export._translation(segment),
            )

        digest = hashlib.sha256(self.source)
        digest.update(json.dumps(
            (
                self.document_id, bilingual, self.metadata,
                [segment_content(segment) for segment in self.segments],
                [segment_content(segment) for segment in self.deleted],
                pdf_export.PDF_LAYOUT_VERSION, GEOMETRY_VERSION, TYPOGRAPHY_VERSION,
                sorted((key, value) for key, value in os.environ.items()
                       if key.startswith("JIEYI_PDF_FONT")),
            ),
            ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ).encode())
        return digest.digest()


class PDFExportCache:
    """An app-local LRU; large downloads are shared only while they are building."""

    def __init__(self, *, max_bytes=64 * 1024 * 1024, max_entries=4, exporter=None):
        self.max_bytes = max_bytes
        self.max_entries = max_entries
        self._exporter = exporter
        self._lock = Lock()
        self._entries = OrderedDict()
        self._bytes = 0
        self._pending: dict[bytes, Future] = {}

    def export(self, store, document_id, *, bilingual=False):
        snapshot = _ExportSnapshot(store, document_id)
        key = snapshot.fingerprint(bilingual)
        with self._lock:
            if key in self._entries:
                self._entries.move_to_end(key)
                return self._entries[key]
            future = self._pending.get(key)
            owner = future is None
            if owner:
                future = self._pending[key] = Future()
        if not owner:
            return future.result()

        try:
            exporter = self._exporter or pdf_export.export_translated_pdf
            result = exporter(snapshot, document_id, bilingual=bilingual)
        except BaseException as exc:
            with self._lock:
                self._pending.pop(key, None)
                future.set_exception(exc)
            raise

        with self._lock:
            if self.max_entries > 0 and len(result) <= self.max_bytes:
                while self._entries and (
                    len(self._entries) >= self.max_entries
                    or self._bytes + len(result) > self.max_bytes
                ):
                    _, removed = self._entries.popitem(last=False)
                    self._bytes -= len(removed)
                self._entries[key] = result
                self._bytes += len(result)
            self._pending.pop(key, None)
            future.set_result(result)
        return result
