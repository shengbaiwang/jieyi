"""Bounded, asynchronous PDF inspection and passive raster previews."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections import OrderedDict
from dataclasses import asdict

from fastapi import HTTPException, Query, Request, Response

from jieyi.domain.models import new_id
from jieyi.ingestion.pdf import extract_pdf, render_pdf_page
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
        return {"document_id": document_id, **store.get_pdf_metadata(document_id)}

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

    @app.get("/documents/{document_id}/pdf/pages/{page_number}")
    async def page(
        document_id: str, page_number: int, width: int = Query(default=1200, ge=320, le=1800)
    ):
        metadata = store.get_pdf_metadata(document_id)
        if not 1 <= page_number <= len(metadata["pages"]):
            raise HTTPException(404, "PDF 页码超出范围")
        key = (document_id, page_number, width)
        if key in images:
            images.move_to_end(key)
            image = images[key]
        else:
            try:
                image = await asyncio.to_thread(
                    render_pdf_page, store.get_original_pdf(document_id), page_number, width
                )
            except Exception as exc:
                raise HTTPException(422, "此页暂时无法渲染，请下载原 PDF 查看。") from exc
            images[key] = image
            while len(images) > 12:
                images.popitem(last=False)
        return Response(
            image,
            media_type="image/png",
            headers={"Cache-Control": "private, max-age=3600", "X-Content-Type-Options": "nosniff"},
        )
