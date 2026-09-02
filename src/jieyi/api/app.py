from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from dataclasses import asdict
from math import ceil
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from jieyi.api.term_routes import install_term_routes
from jieyi.domain.models import JobStatus, ModelSpec, TermEntry, TermStatus, new_id
from jieyi.domain.reasoning import normalize_compute_mode
from jieyi.ingestion import extract_epub, take_distributed_sample
from jieyi.ingestion.epub_navigation import (
    parse_epub_navigation,
    parse_xml_resource,
    position_navigation,
)
from jieyi.ingestion.epub_reader import (
    export_translated_epub,
    reader_csp,
    render_content_path,
    render_spine,
    safe_resource,
)
from jieyi.ingestion.epub_roundtrip import parse_epub_archive
from jieyi.persistence.sqlite import NotFoundError, SQLiteStore
from jieyi.providers import EchoProvider, OpenAICompatibleProvider, ProviderRegistry
from jieyi.quality import (
    DETECTOR_VERSION,
    refresh_segment_quality,
    reindex_all_quality,
    reindex_project_quality,
)
from jieyi.quality.service import document_issues, requires_human_review
from jieyi.quality.terminology_review import TerminologyReviewManager
from jieyi.settings import (
    LocalSettingsStore,
    ModelBinding,
    ProviderSettings,
    profile_from_preset,
    provider_public_payload,
    test_openai_compatible_connection,
    test_openai_compatible_model,
)
from jieyi.workflow import (
    TranslationEngine,
    create_document,
    create_epub_document,
    create_job,
    create_project,
)
from jieyi.workflow.jobs import JobManager


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1)
    source_lang: str = Field(min_length=1)
    target_lang: str = Field(min_length=1)
    domain: str = "humanities_and_social_sciences"
    style_guide: str = ""
    quote_policy: str = "preserve_source_citations"


class ProjectStyleUpdate(BaseModel):
    style_guide: str = Field(default="", max_length=12_000)


class DocumentCreate(BaseModel):
    title: str = Field(min_length=1)
    text: str = Field(min_length=1)
    source_format: str = "markdown"


class TermCreate(BaseModel):
    source: str = Field(min_length=1)
    target: str = Field(min_length=1)
    status: TermStatus = TermStatus.APPROVED
    scope: str = "project"
    domain: str = ""
    rationale: str = ""
    forbidden_targets: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)
    context_keywords: list[str] = Field(default_factory=list)
    sense: str = ""
    disambiguation: str = ""
    enforcement: Literal["auto", "global", "contextual"] = "auto"


class TerminologyReviewCreate(BaseModel):
    token_budget: int = Field(default=200_000, ge=10_000, le=2_000_000)


class JobCreate(BaseModel):
    draft_provider: str = "echo"
    draft_model: str = "dry-run"
    reviewer_provider: str | None = None
    reviewer_model: str | None = None
    task_mode: str = "draft"
    review_policy: str = "on_issue"
    tm_enabled: bool = True
    tm_threshold: float = Field(default=0.78, ge=0, le=1)
    tm_max_results: int = Field(default=3, ge=0, le=10)
    batch_size: int = Field(default=10, ge=1, le=20)
    concurrency: int = Field(default=3, ge=1, le=12)
    max_concurrency: int = Field(default=5, ge=1, le=12)
    max_batch_chars: int = Field(default=4_000, ge=500, le=12_000)
    draft_thinking: bool = False
    review_thinking: bool = True
    draft_compute_mode: str | None = None
    review_compute_mode: str | None = None
    draft_reasoning_effort: str | None = None
    review_reasoning_effort: str | None = None
    review_sample_rate: float = Field(default=0.08, ge=0, le=1)
    max_output_tokens: int = Field(default=6_000, ge=256, le=32_000)
    token_budget: int = Field(default=2_000_000, ge=1)
    segment_ranges: list[tuple[int, int]] = Field(default_factory=list)


class ConfirmSegment(BaseModel):
    translation: str = Field(min_length=1)
    rationale: str = ""
    actor: str = "human"


class SaveSegmentDraft(BaseModel):
    translation: str = ""


class ProviderProfileUpdate(BaseModel):
    id: str = Field(min_length=1, max_length=64)
    name: str = ""
    provider_type: str = Field(default="custom", min_length=1)
    base_url: str = ""
    chat_path: str = ""
    models_path: str = ""
    protocol: str = "chat_completions"
    auth_required: bool | None = None
    capabilities: list[str] | None = None
    api_key: str = ""


class ProviderSettingsUpdate(BaseModel):
    version: int = 3
    profiles: list[ProviderProfileUpdate] | None = None
    draft_profile_id: str = ""
    draft_model: str = ""
    draft_compute_mode: str | None = None
    draft_reasoning_effort: str | None = None
    reviewer_profile_id: str = ""
    reviewer_model: str = ""
    reviewer_compute_mode: str | None = None
    reviewer_reasoning_effort: str | None = None
    review_enabled: bool = False
    # Legacy single-provider fields remain accepted during migration.
    provider_type: str = "openai"
    base_url: str = ""
    api_key: str = ""


class ProviderConnectionTest(BaseModel):
    profile_id: str = ""
    provider_type: str = "custom"
    base_url: str = ""
    models_path: str = ""
    protocol: str = "chat_completions"
    api_key: str = ""
    required_models: list[str] = Field(default_factory=list)


class ModelCapabilityTest(BaseModel):
    profile_id: str = ""
    provider_type: str = "custom"
    base_url: str = ""
    chat_path: str = ""
    protocol: str = "chat_completions"
    api_key: str = ""
    model: str = Field(min_length=1)


def _sync_provider_registry(
    registry: ProviderRegistry,
    settings_store: LocalSettingsStore,
) -> None:
    registry.unregister_matching("profile:")
    registry.unregister("openai-compatible")
    settings = settings_store.load()
    draft_provider: OpenAICompatibleProvider | None = None
    for profile in settings.profiles:
        api_key, _ = settings_store.get_api_key(profile.id)
        provider = OpenAICompatibleProvider(
            api_key=api_key,
            chat_endpoint=profile.chat_endpoint,
            protocol=profile.protocol,
            capabilities=profile.capabilities,
        )
        registry.register(profile.registry_name, provider)
        if profile.id == settings.draft.profile_id:
            draft_provider = provider
    if draft_provider is not None:
        registry.register("openai-compatible", draft_provider)


def _provider_registry(settings_store: LocalSettingsStore) -> ProviderRegistry:
    registry = ProviderRegistry()
    registry.register("echo", EchoProvider())
    _sync_provider_registry(registry, settings_store)
    return registry


def create_app(db_path: str | None = None, settings_path: str | None = None) -> FastAPI:
    resolved_db_path = db_path or os.getenv("JIEYI_DB", "jieyi.db")
    store = SQLiteStore(resolved_db_path)
    store.migrate()
    reindex_all_quality(store)
    store.pause_interrupted_jobs()
    resolved_settings_path = settings_path or os.getenv("JIEYI_CONFIG", "").strip()
    if not resolved_settings_path:
        resolved_settings_path = str(Path(resolved_db_path).with_suffix(".settings.json"))
    settings_store = LocalSettingsStore(resolved_settings_path)
    providers = _provider_registry(settings_store)
    engine = TranslationEngine(store, providers)
    manager = JobManager(store, engine)
    terminology_manager = TerminologyReviewManager(store, providers)
    terminology_manager.repository.fail_interrupted()

    def start_terminology_review(document_id, token_budget=200_000):
        settings = settings_store.load()
        binding = settings.reviewer if settings.reviewer.model else settings.draft
        profile = settings.profile(binding.profile_id)
        if not binding.model or profile is None:
            raise ValueError("请先在设置中配置审校模型或草译模型。")
        return terminology_manager.start(
            document_id, ModelSpec(profile.registry_name, binding.model, 0.0),
            binding.compute_mode, token_budget,
        )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        await manager.shutdown()
        await terminology_manager.shutdown()

    app = FastAPI(title="介译 API", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_origin_regex=r"http://(?:localhost|127\.0\.0\.1):\d+",
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.store = store
    app.state.providers = providers
    app.state.engine = engine
    app.state.job_manager = manager
    app.state.settings_store = settings_store
    app.state.terminology_manager = terminology_manager

    @app.exception_handler(NotFoundError)
    async def not_found_handler(_, exc: NotFoundError):
        return PlainTextResponse(str(exc), status_code=404)

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "api_version": 2,
            "quality_detector_version": DETECTOR_VERSION,
            "db_path": resolved_db_path
            if resolved_db_path == ":memory:"
            else str(Path(resolved_db_path).resolve()),
            "providers": providers.names(),
        }

    @app.post("/projects", status_code=201)
    async def post_project(body: ProjectCreate):
        return asdict(create_project(store, **body.model_dump()))

    @app.get("/projects")
    async def get_projects():
        return [asdict(item) for item in store.list_projects()]

    @app.patch("/projects/{project_id}/style")
    async def patch_project_style(project_id: str, body: ProjectStyleUpdate):
        return asdict(store.update_project_style(project_id, body.style_guide.strip()))

    @app.get("/projects/{project_id}/documents")
    async def get_project_documents(project_id: str):
        store.get_project(project_id)
        documents = []
        for item in store.list_documents(project_id):
            payload = asdict(item) | {"cover_url": None}
            if item.source_format == "epub" and store.has_epub_archive(item.id):
                package = store.get_epub_package(item.id)
                cover_path = package.get("cover_path")
                if cover_path:
                    payload["cover_url"] = f"/documents/{item.id}/epub/resources/{cover_path}"
            documents.append(payload)
        return documents

    @app.post("/projects/{project_id}/documents", status_code=201)
    async def post_document(project_id: str, body: DocumentCreate):
        try:
            document = create_document(store, project_id=project_id, **body.model_dump())
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return asdict(document)

    @app.post("/projects/{project_id}/documents/epub", status_code=201)
    async def post_epub_document(
        project_id: str,
        request: Request,
        title: str | None = Query(default=None, max_length=500),
    ):
        file_data = await request.body()
        if not file_data:
            raise HTTPException(status_code=422, detail="EPUB request body is empty")
        if len(file_data) > 128 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="EPUB file exceeds the 128 MB upload limit")
        try:
            document = create_epub_document(
                store,
                project_id=project_id,
                file_data=file_data,
                title=title,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return asdict(document)

    @app.delete("/documents/{document_id}", status_code=204)
    async def delete_document(document_id: str):
        store.delete_document(document_id)
        return Response(status_code=204)

    @app.post("/imports/epub/inspect")
    async def inspect_epub(request: Request):
        file_data = await request.body()
        if not file_data:
            raise HTTPException(status_code=422, detail="EPUB request body is empty")
        if len(file_data) > 128 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="EPUB file exceeds the 128 MB upload limit")
        try:
            book = extract_epub(file_data)
            archive = parse_epub_archive(file_data, book.source_atoms)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "title": book.title,
            "block_count": len(book.blocks),
            "chapter_count": len(book.navigation) or len(book.spine_items),
            "resource_count": len(archive.resources),
            "cover_path": archive.cover_path,
            "nav_path": archive.nav_path,
            "rendition_layout": archive.rendition_layout,
            "fixed_layout": any(item.fixed_layout for item in archive.spine),
            "preview": [
                {
                    "kind": block.kind.value,
                    "text": block.text,
                    "heading_path": block.heading_path,
                }
                for block in book.blocks[:8]
            ],
        }

    @app.put("/documents/{document_id}/epub/source")
    async def supplement_epub_source(document_id: str, request: Request):
        file_data = await request.body()
        if not file_data:
            raise HTTPException(status_code=422, detail="EPUB request body is empty")
        if len(file_data) > 128 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="EPUB file exceeds the 128 MB upload limit")
        document = store.get_document(document_id)
        try:
            book = extract_epub(file_data)
            archive = parse_epub_archive(file_data, book.source_atoms)
            store.attach_epub_archive(document.id, archive)
        except ValueError as exc:
            status = 409 if "SHA-256" in str(exc) else 422
            raise HTTPException(status_code=status, detail=str(exc)) from exc
        return store.get_epub_package(document_id)

    @app.get("/documents/{document_id}/epub")
    async def get_epub_reader_manifest(document_id: str):
        document = store.get_document(document_id)
        if document.source_format != "epub":
            raise HTTPException(status_code=409, detail="Document is not an EPUB")
        package = store.get_epub_package(document_id)
        spine = store.list_epub_spine(document_id)
        cover_path = package.get("cover_path")
        return package | {
            "spine": spine,
            "segment_locations": store.list_epub_segment_locations(document_id),
            "modes": ["original", "translated", "bilingual"],
            "layout_strategies": ["faithful", "comfort"],
            "cover_url": (
                f"/documents/{document_id}/epub/resources/{cover_path}" if cover_path else None
            ),
        }

    @app.get("/documents/{document_id}/epub/original")
    async def download_original_epub(document_id: str):
        store.get_document(document_id)
        return Response(
            content=store.get_original_epub(document_id),
            media_type="application/epub+zip",
            headers={
                "Content-Disposition": 'attachment; filename="original.epub"',
                "X-Content-Type-Options": "nosniff",
            },
        )

    @app.get("/documents/{document_id}/epub/mappings")
    async def get_epub_mappings(document_id: str):
        store.get_document(document_id)
        return store.list_epub_mappings(document_id)

    @app.get("/documents/{document_id}/epub/spine/{spine_index}")
    async def get_epub_spine_document(
        document_id: str,
        spine_index: int,
        request: Request,
        mode: str = Query(default="original", pattern="^(original|translated|bilingual)$"),
        layout: str = Query(default="faithful", pattern="^(faithful|comfort)$"),
    ):
        store.get_document(document_id)
        try:
            payload, media_type = render_spine(
                store,
                document_id,
                spine_index,
                mode=mode,
                layout=layout,
            )
        except (ValueError, KeyError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return Response(
            content=payload,
            media_type=media_type,
            headers={
                "Content-Security-Policy": reader_csp(
                    str(request.base_url), allow_resize_script=True
                ),
                "Referrer-Policy": "no-referrer",
                "X-Content-Type-Options": "nosniff",
                "Cache-Control": "private, no-store",
            },
        )

    @app.get("/documents/{document_id}/epub/content/{content_path:path}")
    async def get_epub_linked_content(document_id: str, content_path: str, request: Request):
        store.get_document(document_id)
        try:
            payload, media_type = render_content_path(store, document_id, content_path)
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        return Response(
            content=payload,
            media_type=media_type,
            headers={
                "Content-Security-Policy": reader_csp(str(request.base_url)),
                "Referrer-Policy": "no-referrer",
                "X-Content-Type-Options": "nosniff",
                "Cache-Control": "private, no-store",
            },
        )

    @app.get("/documents/{document_id}/epub/resources/{resource_path:path}")
    async def get_epub_render_resource(document_id: str, resource_path: str, request: Request):
        store.get_document(document_id)
        try:
            payload, media_type = safe_resource(store, document_id, resource_path)
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        return Response(
            content=payload,
            media_type=media_type,
            headers={
                "Content-Security-Policy": reader_csp(str(request.base_url)),
                "Referrer-Policy": "no-referrer",
                "X-Content-Type-Options": "nosniff",
                "Cache-Control": "private, max-age=3600",
            },
        )

    @app.get("/documents/{document_id}/segments")
    async def get_segments(document_id: str):
        store.get_document(document_id)
        return [asdict(item) for item in store.list_segments(document_id)]

    @app.get("/documents/{document_id}/segments/page")
    async def get_segment_page(
        document_id: str,
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=100, ge=1, le=250),
    ):
        document = store.get_document(document_id)
        project = store.get_project(document.project_id)
        total, items = store.list_segments_page(document_id, offset=offset, limit=limit)
        return {
            "document": asdict(document),
            "project": asdict(project),
            "total": total,
            "offset": offset,
            "limit": limit,
            "items": [asdict(item) for item in items],
        }

    @app.get("/documents/{document_id}/overview")
    async def get_document_overview(document_id: str):
        document = store.get_document(document_id)
        project = store.get_project(document.project_id)
        segments = store.list_segments(document_id)
        chapters: list[dict] = []

        def chapter_row(title: str, start: int, end: int, level: int = 0) -> dict:
            members = [item for item in segments if start <= item.ordinal <= end]
            return {
                "title": title,
                "level": level,
                "start_ordinal": start,
                "end_ordinal": end,
                "segment_count": len(members),
                "translated_count": sum(
                    bool(
                        item.machine_translation
                        or item.edited_translation
                        or item.reviewed_translation
                        or item.accepted_translation
                    )
                    for item in members
                ),
                "confirmed_count": sum(item.status.value == "human_confirmed" for item in members),
            }

        if document.source_format == "epub" and store.has_epub_archive(document_id):
            try:
                package = store.get_epub_package(document_id)
                nav_path = package.get("nav_path")
                if nav_path:
                    nav_resource = store.get_epub_resource(document_id, nav_path)
                    navigation = parse_epub_navigation(nav_resource["data"], nav_path)
                    roots = {}
                    for path in {item.path for item in navigation}:
                        try:
                            resource = store.get_epub_resource(document_id, path)
                            roots[path] = parse_xml_resource(resource["data"], path)
                        except (NotFoundError, ValueError):
                            continue
                    atoms = store.list_epub_mappings(document_id)["atoms"]
                    positioned = position_navigation(navigation, roots, atoms)
                    segment_ordinals = {item.id: item.ordinal for item in segments}
                    navigation_starts = []
                    seen = set()
                    for order, item in enumerate(positioned):
                        start = segment_ordinals.get(item.atom["segment_id"])
                        key = (start, item.entry.label, item.entry.level)
                        if start is None or key in seen:
                            continue
                        seen.add(key)
                        navigation_starts.append((start, order, item.entry))
                    navigation_starts.sort(key=lambda item: (item[0], item[1]))
                    if navigation_starts:
                        first_start = navigation_starts[0][0]
                        if first_start > 0:
                            chapters.append(chapter_row("正文", 0, first_start - 1))
                        for index, (start, _, entry) in enumerate(navigation_starts):
                            later = next(
                                (
                                    candidate[0]
                                    for candidate in navigation_starts[index + 1 :]
                                    if candidate[0] > start
                                ),
                                len(segments),
                            )
                            chapters.append(chapter_row(entry.label, start, later - 1, entry.level))
            except (NotFoundError, ValueError):
                chapters = []

        if not chapters:
            for segment in segments:
                starts_chapter = segment.kind.value == "heading"
                if not chapters or starts_chapter:
                    title = (
                        segment.source_text
                        if starts_chapter
                        else segment.heading_path.split(" / ", 1)[0].strip() or "正文"
                    )
                    level = max(0, segment.heading_path.count(" / ")) if starts_chapter else 0
                    chapters.append(
                        {
                            "title": title,
                            "level": level,
                            "start_ordinal": segment.ordinal,
                            "end_ordinal": segment.ordinal,
                            "segment_count": 0,
                            "translated_count": 0,
                            "confirmed_count": 0,
                        }
                    )
                chapter = chapters[-1]
                chapter["end_ordinal"] = segment.ordinal
                chapter["segment_count"] += 1
                if (
                    segment.machine_translation
                    or segment.edited_translation
                    or segment.reviewed_translation
                    or segment.accepted_translation
                ):
                    chapter["translated_count"] += 1
                if segment.status.value == "human_confirmed":
                    chapter["confirmed_count"] += 1
        return {
            "document": asdict(document),
            "project": asdict(project),
            "segment_count": len(segments),
            "translated_count": sum(
                bool(
                    item.machine_translation
                    or item.edited_translation
                    or item.reviewed_translation
                    or item.accepted_translation
                )
                for item in segments
            ),
            "reviewed_count": sum(bool(item.reviewed_translation) for item in segments),
            "confirmed_count": sum(item.status.value == "human_confirmed" for item in segments),
            "chapters": chapters,
        }

    @app.get("/documents/{document_id}/sample")
    async def sample_document(
        document_id: str,
        budget: int = Query(default=6_000, ge=200, le=100_000),
        count: int = Query(default=8, ge=1, le=50),
    ):
        store.get_document(document_id)
        source = "\n\n".join(item.source_text for item in store.list_segments(document_id))
        return asdict(take_distributed_sample(source, total_budget=budget, excerpt_count=count))

    @app.get("/documents/{document_id}/search")
    async def search_document(
        document_id: str,
        q: str = Query(min_length=1, max_length=200),
        limit: int = Query(default=30, ge=1, le=100),
    ):
        store.get_document(document_id)
        return [asdict(item) for item in store.search_segments(document_id, q, limit=limit)]

    @app.post("/projects/{project_id}/terms", status_code=201)
    async def post_term(project_id: str, body: TermCreate):
        store.get_project(project_id)
        value = body.model_dump()
        value["source"] = value["source"].strip()
        value["target"] = value["target"].strip()
        if not value["source"] or not value["target"]:
            raise HTTPException(status_code=422, detail="Source and target terms cannot be blank")
        for field_name in ("forbidden_targets", "aliases", "context_keywords"):
            value[field_name] = tuple(
                dict.fromkeys(
                    item.strip() for item in value[field_name] if item.strip()
                )
            )
        source_key = value["source"].casefold()
        value["aliases"] = tuple(
            item for item in value["aliases"] if item.casefold() != source_key
        )
        for field_name in ("scope", "domain", "rationale", "sense", "disambiguation"):
            value[field_name] = value[field_name].strip()
        term = TermEntry(id=new_id("term"), project_id=project_id, **value)
        try:
            created = store.add_term(term)
            reindex_project_quality(store, project_id)
            return asdict(created)
        except Exception as exc:
            if "UNIQUE constraint failed" in str(exc):
                raise HTTPException(
                    status_code=409, detail="Term sense already exists in this scope"
                ) from exc
            raise

    @app.get("/projects/{project_id}/terms")
    async def get_terms(project_id: str):
        store.get_project(project_id)
        return [asdict(item) for item in store.list_terms(project_id)]

    @app.post("/documents/{document_id}/jobs", status_code=201)
    async def post_job(document_id: str, body: JobCreate):
        if body.draft_provider not in providers.names():
            raise HTTPException(
                status_code=422,
                detail=f"Provider is not configured: {body.draft_provider}",
            )
        if body.reviewer_provider and body.reviewer_provider not in providers.names():
            raise HTTPException(
                status_code=422,
                detail=f"Provider is not configured: {body.reviewer_provider}",
            )
        try:
            job = create_job(store, document_id=document_id, **body.model_dump())
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return asdict(job)

    @app.post("/jobs/{job_id}/start")
    async def start_job(job_id: str):
        try:
            return manager.start(job_id)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/jobs/{job_id}/pause")
    async def pause_job(job_id: str):
        return await manager.stop(job_id, JobStatus.PAUSED)

    @app.post("/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str):
        return await manager.stop(job_id, JobStatus.CANCELLED)

    @app.post("/jobs/{job_id}/run")
    async def run_job(job_id: str, max_segments: int | None = Query(default=None, ge=1)):
        try:
            return asdict(await engine.run(job_id, max_segments=max_segments))
        except KeyError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/jobs/{job_id}")
    async def get_job(job_id: str):
        return store.job_progress(job_id)

    @app.get("/documents/{document_id}/jobs")
    async def get_document_jobs(document_id: str):
        store.get_document(document_id)
        return [store.job_progress(item.id) for item in store.list_jobs(document_id)]

    @app.get("/jobs/{job_id}/segments/{segment_id}/prompt-preview")
    async def preview_prompt(job_id: str, segment_id: str):
        try:
            return engine.preview(job_id, segment_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.patch("/segments/{segment_id}/confirm")
    async def confirm_segment(segment_id: str, body: ConfirmSegment):
        store.confirm_segment(segment_id, **body.model_dump())
        refresh_segment_quality(store, segment_id)
        return asdict(store.get_segment(segment_id))

    @app.patch("/segments/{segment_id}/draft")
    async def save_segment_draft(segment_id: str, body: SaveSegmentDraft):
        store.save_segment_draft(segment_id, body.translation)
        refresh_segment_quality(store, segment_id)
        return asdict(store.get_segment(segment_id))

    @app.get("/segments/{segment_id}/candidates")
    async def get_segment_candidates(segment_id: str):
        store.get_segment(segment_id)
        return store.list_candidates(segment_id)

    @app.get("/segments/{segment_id}/provider-failures")
    async def get_segment_provider_failures(segment_id: str):
        store.get_segment(segment_id)
        return [
            event
            for event in store.list_audit_events("segment", segment_id)
            if event["action"] == "provider_failure"
        ]

    @app.get("/documents/{document_id}/issues")
    async def get_issues(document_id: str):
        store.get_document(document_id)
        return document_issues(store, document_id)

    @app.get("/documents/{document_id}/human-review-queue")
    async def get_human_review_queue(
        document_id: str,
        sample_rate: float = Query(default=0.08, ge=0, le=1),
    ):
        """Return unresolved findings plus a reproducible sample of clean AI-reviewed text."""
        store.get_document(document_id)
        segments = store.list_segments(document_id)
        issues_by_segment: dict[str, list[dict]] = {}
        all_findings = document_issues(store, document_id)
        pending_ids = {issue["segment_id"] for issue in all_findings
                       if not requires_human_review(issue)}
        for issue in all_findings:
            if requires_human_review(issue):
                issues_by_segment.setdefault(issue["segment_id"], []).append(issue)

        unconfirmed = [
            segment for segment in segments if segment.status.value != "human_confirmed"
        ]
        clean = [
            segment
            for segment in unconfirmed
            if segment.reviewed_translation and segment.id not in issues_by_segment
            and segment.id not in pending_ids
        ]
        sample_count = min(len(clean), ceil(len(clean) * sample_rate)) if sample_rate else 0
        sampled_ids: set[str] = set()
        if sample_count:
            # Midpoints of equal ordinal intervals give a stable, book-wide sample instead
            # of over-representing the opening pages.
            sampled_ids = {
                clean[min(len(clean) - 1, int((index + 0.5) * len(clean) / sample_count))].id
                for index in range(sample_count)
            }

        result = []
        for segment in unconfirmed:
            findings = issues_by_segment.get(segment.id, [])
            if not findings and segment.id not in sampled_ids:
                continue
            has_error = any(item["severity"] == "error" for item in findings)
            reason = "error" if has_error else "warning" if findings else "sample"
            result.append(
                {
                    "segment_id": segment.id,
                    "ordinal": segment.ordinal,
                    "reason": reason,
                    "issue_count": len(findings),
                    "source_text": segment.source_text,
                    "translation": (
                        segment.reviewed_translation
                        or segment.edited_translation
                        or segment.machine_translation
                        or ""
                    ),
                }
            )
        return result

    @app.get("/projects/{project_id}/translation-memory")
    async def get_translation_memory(project_id: str):
        store.get_project(project_id)
        return [asdict(item) for item in store.list_translation_memory(project_id)]

    @app.get("/documents/{document_id}/export")
    async def export_document(
        document_id: str,
        bilingual: bool = False,
        format: str = Query(default="text", pattern="^(text|book)$"),
    ):
        document = store.get_document(document_id)
        if format == "book" and document.source_format == "epub":
            return Response(
                content=export_translated_epub(store, document_id),
                media_type="application/epub+zip",
                headers={
                    "Content-Disposition": 'attachment; filename="translated.epub"',
                    "X-Content-Type-Options": "nosniff",
                },
            )
        media_type = (
            "text/markdown; charset=utf-8"
            if document.source_format == "markdown"
            else "text/plain; charset=utf-8"
        )
        return Response(
            content=store.export_document(document_id, bilingual=bilingual),
            media_type=media_type,
            headers={"Content-Disposition": 'attachment; filename="translated.txt"'},
        )

    @app.get("/settings/provider")
    async def get_provider_settings():
        return provider_public_payload(settings_store)

    @app.patch("/settings/provider")
    async def patch_provider_settings(body: ProviderSettingsUpdate):
        try:
            if body.profiles is not None:
                profiles = tuple(
                    profile_from_preset(
                        item.id,
                        item.provider_type,
                        name=item.name,
                        base_url=item.base_url,
                        chat_path=item.chat_path,
                        models_path=item.models_path,
                        protocol=item.protocol,
                        auth_required=item.auth_required,
                        capabilities=(
                            tuple(item.capabilities) if item.capabilities is not None else None
                        ),
                    )
                    for item in body.profiles
                )
                first_id = profiles[0].id if profiles else ""
                settings = ProviderSettings(
                    profiles=profiles,
                    draft=ModelBinding(
                        body.draft_profile_id or first_id,
                        body.draft_model.strip(),
                        normalize_compute_mode(
                            body.draft_compute_mode or body.draft_reasoning_effort,
                            "economy",
                        ),
                    ),
                    reviewer=ModelBinding(
                        body.reviewer_profile_id or body.draft_profile_id or first_id,
                        body.reviewer_model.strip(),
                        normalize_compute_mode(
                            body.reviewer_compute_mode or body.reviewer_reasoning_effort,
                            "performance",
                        ),
                    ),
                    review_enabled=body.review_enabled,
                )
                key_updates = [(item.id, item.api_key) for item in body.profiles]
            else:
                profile = profile_from_preset(
                    "default",
                    body.provider_type,
                    base_url=body.base_url,
                )
                settings = ProviderSettings(
                    profiles=(profile,),
                    draft=ModelBinding(
                        "default",
                        body.draft_model.strip(),
                        normalize_compute_mode(
                            body.draft_compute_mode or body.draft_reasoning_effort,
                            "economy",
                        ),
                    ),
                    reviewer=ModelBinding(
                        "default",
                        body.reviewer_model.strip(),
                        normalize_compute_mode(
                            body.reviewer_compute_mode or body.reviewer_reasoning_effort,
                            "performance",
                        ),
                    ),
                    review_enabled=body.review_enabled,
                )
                key_updates = [("default", body.api_key)]
            settings_store.save(settings)
            warnings: list[str] = []
            for profile_id, api_key in key_updates:
                if api_key.strip():
                    result = settings_store.set_api_key(api_key, profile_id)
                    if result.warning:
                        warnings.append(result.warning)
            _sync_provider_registry(providers, settings_store)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except OSError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        payload = provider_public_payload(settings_store)
        payload["warnings"] = [*payload.get("warnings", []), *warnings]
        return payload

    @app.post("/settings/provider/test")
    async def test_provider_settings(body: ProviderConnectionTest):
        try:
            current = settings_store.load()
            stored_profile = current.profile(body.profile_id) if body.profile_id else None
            if body.base_url:
                profile = profile_from_preset(
                    body.profile_id or "test",
                    body.provider_type,
                    base_url=body.base_url,
                    models_path=body.models_path,
                    protocol=body.protocol,
                )
            elif stored_profile is not None:
                profile = stored_profile
            else:
                raise ValueError("找不到要测试的连接")
            existing_key, _ = settings_store.get_api_key(profile.id)
            return await asyncio.to_thread(
                test_openai_compatible_connection,
                profile.base_url,
                body.api_key.strip() or existing_key,
                models_endpoint=profile.models_endpoint,
                required_models=body.required_models,
                protocol=profile.protocol,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.post("/settings/provider/model-test")
    async def test_provider_model(body: ModelCapabilityTest):
        try:
            current = settings_store.load()
            stored_profile = current.profile(body.profile_id) if body.profile_id else None
            if body.base_url:
                profile = profile_from_preset(
                    body.profile_id or "test",
                    body.provider_type,
                    base_url=body.base_url,
                    chat_path=body.chat_path,
                    protocol=body.protocol,
                )
            elif stored_profile is not None:
                profile = stored_profile
            else:
                raise ValueError("找不到要测试的模型连接")
            existing_key, _ = settings_store.get_api_key(profile.id)
            return await asyncio.to_thread(
                test_openai_compatible_model,
                profile.chat_endpoint,
                body.api_key.strip() or existing_key,
                body.model,
                protocol=profile.protocol,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.get("/documents/{document_id}/terminology-review")
    async def get_terminology_review(document_id: str):
        summary = terminology_manager.repository.summary(document_id)
        settings = settings_store.load()
        binding = settings.reviewer if settings.reviewer.model else settings.draft
        profile = settings.profile(binding.profile_id)
        return summary | {
            "configured_model": binding.model,
            "service_host": urlsplit(profile.base_url).hostname if profile else "",
        }

    @app.post("/documents/{document_id}/terminology-review", status_code=202)
    async def post_terminology_review(document_id: str, body: TerminologyReviewCreate):
        store.get_document(document_id)
        try:
            return start_terminology_review(document_id, body.token_budget)
        except (ValueError, KeyError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    install_term_routes(app, store, providers)
    return app
