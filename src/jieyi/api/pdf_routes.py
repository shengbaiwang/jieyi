"""Bounded, asynchronous PDF inspection and passive raster previews."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
from collections import OrderedDict
from dataclasses import asdict

from fastapi import HTTPException, Query, Request, Response

from jieyi.domain.models import new_id
from jieyi.ingestion.pdf import GEOMETRY_VERSION, extract_pdf, render_pdf_page
from jieyi.ingestion.pdf_export import PDF_LAYOUT_VERSION, compose_pdf, ensure_pdf_layout
from jieyi.ingestion.pdf_reading import pdf_page_interactions, pdf_page_text, pdf_search_pages
from jieyi.ingestion.typography import TYPOGRAPHY_VERSION
from jieyi.workflow.services import create_pdf_document

MAX_BYTES = 128 * 1024 * 1024


async def _body(request: Request) -> bytes:
    data = bytearray()
    async for chunk in request.stream():
        if len(data) + len(chunk) > MAX_BYTES:
            raise HTTPException(413, "PDF 最大支持 128 MB，请拆分后导入。")
        data.extend(chunk)
    if not data:
        raise HTTPException(422, "PDF 文件为空。")
    return bytes(data)


def install_pdf_routes(app, store):
    inspections: dict[str, dict] = {}
    tasks: set[asyncio.Task] = set()
    app.state.pdf_tasks = tasks
    parsing = asyncio.Semaphore(2)
    images: OrderedDict = OrderedDict()
    search_indexes: OrderedDict = OrderedDict()
    indexing: dict[str, asyncio.Task] = {}
    text_pages: OrderedDict = OrderedDict()
    interactions: OrderedDict = OrderedDict()
    pending: dict[tuple, asyncio.Task] = {}
    rendering = asyncio.Semaphore(2)

    def public(item):
        return {key: value for key, value in item.items() if key not in {"book", "hash", "created"}}

    @app.post("/imports/pdf/inspect", status_code=202)
    async def inspect(request: Request):
        now = time.monotonic()
        for key, item in list(inspections.items()):
            if item["status"] != "processing" and now - item["created"] > 1800:
                inspections.pop(key, None)
        if sum(item["status"] == "processing" for item in inspections.values()) >= 2:
            raise HTTPException(429, "正在解析其他 PDF，请稍后重试。")
        data = await _body(request)
        digest = hashlib.sha256(data).hexdigest()
        for key, item in inspections.items():
            if item["hash"] == digest and item["status"] != "failed":
                return {"id": key, **public(item)}
        # Cache at most four full text models; source bytes are released after parsing.
        while len(inspections) >= 4:
            key = next(
                (key for key, item in inspections.items() if item["status"] != "processing"), None
            )
            if key is None:
                break
            inspections.pop(key)
        if sum(item["status"] == "processing" for item in inspections.values()) >= 2:
            raise HTTPException(429, "正在解析其他 PDF，请稍后重试。")
        identifier = new_id("pdf")
        state = {
            "status": "processing",
            "completed_pages": 0,
            "total_pages": 0,
            "created": now,
            "hash": digest,
        }
        inspections[identifier] = state

        async def run():
            def progress(done, total):
                state.update(completed_pages=done, total_pages=total)

            try:
                async with parsing:
                    book = await asyncio.to_thread(extract_pdf, data, progress)
                state.update(
                    status="ready",
                    book=book,
                    title=book.title,
                    block_count=len(book.blocks),
                    page_count=len(book.pages),
                    chapter_count=len(book.navigation),
                    warnings=book.warnings,
                    preview=[
                        {
                            "text": block.text,
                            "kind": block.kind.value,
                            "heading_path": block.heading_path,
                        }
                        for block in book.blocks[:8]
                    ],
                )
            except ValueError as exc:
                state.update(status="failed", detail=str(exc))
            except Exception:
                logging.getLogger(__name__).exception("PDF inspection failed")
                state.update(status="failed", detail="PDF 解析失败，请重新选择文件。")

        task = asyncio.create_task(run())
        tasks.add(task)
        task.add_done_callback(tasks.discard)
        return {"id": identifier, **public(state)}

    @app.get("/imports/pdf/inspect/{identifier}")
    async def inspection_status(identifier: str):
        state = inspections.get(identifier)
        if state is None:
            raise HTTPException(404, "解析预览已过期，请重新选择 PDF。")
        return {"id": identifier, **public(state)}

    @app.post("/projects/{project_id}/documents/pdf", status_code=201)
    async def import_pdf(
        project_id: str, request: Request, title: str | None = Query(default=None, max_length=500)
    ):
        store.get_project(project_id)
        data = await _body(request)
        digest = hashlib.sha256(data).hexdigest()
        book = next(
            (
                item["book"]
                for item in inspections.values()
                if item["hash"] == digest and item["status"] == "ready"
            ),
            None,
        )
        try:
            async with parsing:
                document = await asyncio.to_thread(
                    create_pdf_document,
                    store,
                    project_id=project_id,
                    file_data=data,
                    title=title,
                    book=book,
                )
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return asdict(document)

    @app.get("/documents/{document_id}/pdf")
    async def manifest(document_id: str):
        return {
            "document_id": document_id,
            **{
                key: value
                for key, value in store.get_pdf_metadata(document_id).items()
                if key != "layout"
            },
        }

    @app.get("/documents/{document_id}/pdf/original")
    async def original(document_id: str):
        return Response(
            store.get_original_pdf(document_id),
            media_type="application/pdf",
            headers={
                "Content-Disposition": 'attachment; filename="original.pdf"',
                "X-Content-Type-Options": "nosniff",
            },
        )

    async def page_variant(document_id, page_number, width, mode):
        metadata = store.get_pdf_metadata(document_id)
        if not 1 <= page_number <= len(metadata["pages"]):
            raise HTTPException(404, "PDF 页码超出范围")
        segments = store.list_segments(document_id) if mode == "translated" else []
        deleted = store.list_deleted_source_segments(document_id) if mode == "translated" else []
        revision = (hashlib.sha256(repr((segments, deleted, GEOMETRY_VERSION, TYPOGRAPHY_VERSION, PDF_LAYOUT_VERSION)).encode()).hexdigest()
                    if mode == "translated" else "original")
        key = (document_id, page_number, width, mode, revision)
        if key in images:
            images.move_to_end(key)
            return images[key]

        def build():
            source = store.get_original_pdf(document_id)
            report = {"overflow": [], "overflow_count": 0}
            number = page_number
            if mode == "translated":
                layout = ensure_pdf_layout(store, document_id)
                source, report = compose_pdf(source, layout, segments, only_page=page_number,
                                             deleted_segments=deleted, strict=False)
                number = 1
            return render_pdf_page(source, number, width), report, (source if mode == "translated" else b""), number

        async def render():
            try:
                async with rendering:
                    result = await asyncio.to_thread(build)
                images[key] = result
                while len(images) > 12:
                    images.popitem(last=False)
                return result
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
            except Exception as exc:
                raise HTTPException(422, "此页暂时无法渲染，请下载原 PDF 查看。") from exc
            finally:
                pending.pop(key, None)

        if key not in pending:
            task = asyncio.create_task(render())
            pending[key] = task
            tasks.add(task)
            task.add_done_callback(tasks.discard)
        # The image and continuation notes share one render; navigation cancellation
        # must not cancel a result still needed by another request.
        return await asyncio.shield(pending[key])

    @app.get("/documents/{document_id}/pdf/pages/{page_number}")
    async def page(
        document_id: str,
        page_number: int,
        width: int = Query(default=1200, ge=320, le=1800),
        mode: str = Query(default="original", pattern="^(original|translated)$"),
    ):
        image, report, _, _ = await page_variant(document_id, page_number, width, mode)
        return Response(
            image,
            media_type="image/png",
            headers={
                "Cache-Control": "no-cache" if mode == "translated" else "private, max-age=3600",
                "X-Content-Type-Options": "nosniff",
                "X-PDF-Overflow": str(report["overflow_count"]),
            },
        )

    @app.get("/documents/{document_id}/pdf/pages/{page_number}/translation-notes")
    async def page_notes(document_id: str, page_number: int):
        _, report, _, _ = await page_variant(document_id, page_number, 1200, "translated")
        return report


    @app.get("/documents/{document_id}/pdf/pages/{page_number}/text")
    async def page_text(
        document_id: str, page_number: int,
        mode: str = Query(default="original", pattern="^(original|translated)$"),
    ):
        metadata = store.get_pdf_metadata(document_id)
        if not 1 <= page_number <= len(metadata["pages"]):
            raise HTTPException(404, "PDF 页码超出范围")
        if mode == "translated":
            _, _, source, number = await page_variant(document_id, page_number, 1200, mode)
        else:
            source, number = store.get_original_pdf(document_id), page_number
        key = (document_id, page_number, mode, hashlib.sha256(source).hexdigest())
        if key not in text_pages:
            async with rendering:
                text_pages[key] = await asyncio.to_thread(pdf_page_text, source, number)
            while len(text_pages) > 24:
                text_pages.popitem(last=False)
        text_pages.move_to_end(key)
        result = {"page": page_number, "mode": mode, **text_pages[key]}
        # Original destinations cannot be laid over reflowed translation geometry.
        if mode == "original":
            link_key = (document_id, page_number)
            if link_key not in interactions:
                interactions[link_key] = await asyncio.to_thread(
                    pdf_page_interactions, source, page_number,
                )
                while len(interactions) > 24:
                    interactions.popitem(last=False)
            result.update(interactions[link_key])
        else:
            result.update(links=[], annotations=[])
        return result

    @app.get("/documents/{document_id}/pdf/search")
    async def search(
        document_id: str, q: str = Query(min_length=1, max_length=500),
        mode: str = Query(default="original", pattern="^(original|translated)$"),
        offset: int = Query(default=0, ge=0), limit: int = Query(default=50, ge=1, le=100),
    ):
        store.get_pdf_metadata(document_id)
        query = q.strip()
        if not query:
            raise HTTPException(422, "请输入搜索文字")
        # Whitespace follows PDF line breaks; user text is always a literal query.
        pattern = re.compile(r"\s+".join(re.escape(part) for part in query.split()), re.IGNORECASE)
        if mode == "original":
            if document_id not in search_indexes:
                async def build_index():
                    try:
                        async with parsing:
                            result = await asyncio.to_thread(
                                pdf_search_pages, store.get_original_pdf(document_id),
                            )
                        search_indexes[document_id] = result
                        while len(search_indexes) > 3:
                            search_indexes.popitem(last=False)
                        return result
                    finally:
                        indexing.pop(document_id, None)
                if document_id not in indexing:
                    task = asyncio.create_task(build_index())
                    indexing[document_id] = task
                    tasks.add(task)
                    task.add_done_callback(tasks.discard)
                pages = await asyncio.shield(indexing[document_id])
            else:
                search_indexes.move_to_end(document_id)
                pages = search_indexes[document_id]
        else:
            pages = []
            for segment in store.list_segments(document_id):
                text = (segment.accepted_translation or segment.reviewed_translation
                        or segment.edited_translation or segment.machine_translation
                        or segment.source_text)
                page_refs = [int(ref.split(":")[2]) for ref in segment.source_refs
                             if re.fullmatch(r"pdf:page:\d+", ref)]
                pages.append({"page": page_refs[0] if page_refs else 1, "text": text})
        results, total = [], 0
        for item in pages:
            for match in pattern.finditer(item["text"]):
                if offset <= total < offset + limit:
                    start, end = match.span()
                    results.append({"page": item["page"], "quote": match.group(),
                                    "snippet": item["text"][max(0, start - 70):end + 100],
                                    "start": len(item["text"][:start].encode("utf-16-le")) // 2,
                                    "end": len(item["text"][:end].encode("utf-16-le")) // 2,
                                    "mode": mode})
                total += 1
        return {"results": results, "total": total, "offset": offset, "limit": limit}
