from __future__ import annotations

import asyncio
import hashlib
import sqlite3
from dataclasses import asdict, replace

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from jieyi.domain.models import IssueSeverity, ModelSpec, TermEnforcement, utc_now
from jieyi.quality import reindex_project_quality, run_deterministic_checks
from jieyi.quality.service import visible_translation
from jieyi.term_discovery import (
    DiscoveryConfig,
    discovery_fingerprint,
    enrich_candidates,
    mine_term_candidates,
    model_review_counts,
    new_discovery_run,
)
from jieyi.term_repository import TermRepository
from jieyi.terminology import term_appears


class TermDiscoveryCreate(BaseModel):
    provider: str = ""
    model: str = ""
    compute_mode: str = "balanced"
    max_candidates: int = Field(default=40, ge=5, le=500)
    max_evidence_per_candidate: int = Field(default=6, ge=1, le=20)
    max_model_candidates: int = Field(default=40, ge=0, le=200)
    model_batch_size: int = Field(default=4, ge=1, le=20)
    min_score: float = Field(default=0.34, ge=0, le=1)


class TermDiscoveryRetry(BaseModel):
    provider: str = ""
    model: str = ""
    compute_mode: str | None = None
    max_model_candidates: int | None = Field(default=None, ge=1, le=200)


class TermCandidateReview(BaseModel):
    status: str = "pending"
    proposed_target: str = ""
    sense: str = ""
    rationale: str = ""
    disambiguation: str = ""
    actor: str = "human"


class TermCandidateRevocation(BaseModel):
    actor: str = "human"


class TermCandidateApproval(BaseModel):
    target: str = Field(min_length=1)
    enforcement: TermEnforcement = "contextual"
    sense: str = ""
    rationale: str = ""
    context_keywords: list[str] = Field(default_factory=list)
    disambiguation: str = ""
    actor: str = "human"


def _translated_state_hash(segments) -> str:
    digest = hashlib.sha256()
    for segment in segments:
        digest.update(segment.id.encode("utf-8"))
        digest.update(visible_translation(segment).encode("utf-8"))
    return digest.hexdigest()


def _record_impact(store, term_id: str, payload: dict) -> None:
    with store._connect() as connection:
        store._audit(connection, "term", term_id, "existing_translations_checked", payload)


def install_term_routes(app: FastAPI, store, providers) -> None:
    repository = TermRepository(store)
    repository.fail_orphaned_runs()

    tasks: set[asyncio.Task] = set()
    app.state.term_discovery_tasks = tasks

    def validate_provider(provider: str, model: str) -> None:
        if bool(provider) != bool(model):
            raise HTTPException(status_code=422, detail="provider and model must be supplied together")
        if provider and provider not in providers.names():
            raise HTTPException(status_code=422, detail=f"Provider is not configured: {provider}")

    async def review_saved_run(run, provider_name, model_name, compute_mode, limit):
        document = store.get_document(run["document_id"])
        project = store.get_project(document.project_id)
        config = replace(DiscoveryConfig(**run["config"]), max_model_candidates=limit)
        candidates = repository.list_candidates(document.id, run_id=run["id"], limit=2_000)
        coverage = dict(run["coverage"]) | {
            "scan_completed": True, "source_hash": document.source_hash,
            "review_provider": provider_name, "review_model": model_name,
            "review_compute_mode": compute_mode, "review_limit": limit,
        }
        # Save the selected model before any network call, including for a restarted retry.
        repository.update_run(run["id"], coverage=coverage)

        def checkpoint(items, usage):
            repository.checkpoint_review(
                run["id"], items, usage, baseline=run, coverage=coverage, limit=limit,
            )

        try:
            _, usage = await enrich_candidates(
                candidates, provider=providers.get(provider_name),
                model=ModelSpec(provider=provider_name, model=model_name, temperature=0.0),
                source_lang=str(coverage.get("language_profile") or project.source_lang),
                target_lang=project.target_lang, config=config, compute_mode=compute_mode,
                checkpoint=checkpoint,
            )
            checkpoint(candidates, usage)
            missing = coverage["missing_model_decisions"]
            error = usage.get("model_error", "")
            if missing and not error:
                error = f"仍有 {missing} 项未取得有效模型判断；已保存进度，可仅继续这些候选的复核。"
            coverage["model_error"] = error
            return repository.update_run(
                run["id"], status="partial" if missing or error else "completed",
                coverage=coverage, error=error, completed_at=utc_now(),
            )
        except BaseException as exc:
            # Includes cancellation at shutdown. Each previous response is already durable.
            error = (
                "模型复核已中断；已保存进度，可继续复核。"
                if isinstance(exc, asyncio.CancelledError) else str(exc)[:1_000]
            )
            current = repository.get_run(run["id"])
            repository.update_run(
                run["id"], status="partial", coverage=current["coverage"] | {"model_error": error},
                error=error, completed_at=utc_now(),
            )
            raise

    async def review_in_task(*args):
        # A closed browser/request must not throw away the current model response.
        task = asyncio.create_task(review_saved_run(*args))
        tasks.add(task)
        task.add_done_callback(tasks.discard)
        return await asyncio.shield(task)

    @app.post("/documents/{document_id}/term-discovery-runs", status_code=201)
    async def create_term_discovery_run(document_id: str, body: TermDiscoveryCreate):
        document = store.get_document(document_id)
        project = store.get_project(document.project_id)
        validate_provider(body.provider, body.model)
        config = DiscoveryConfig(
            max_candidates=body.max_candidates,
            max_evidence_per_candidate=body.max_evidence_per_candidate,
            max_model_candidates=body.max_model_candidates,
            model_batch_size=body.model_batch_size, min_score=body.min_score,
        )
        # Scan identity does not change with translations or model settings.
        fingerprint = discovery_fingerprint(
            document.source_hash,
            replace(config, max_model_candidates=0, model_batch_size=0),
        )
        proposed = new_discovery_run(
            document_id=document_id, fingerprint=fingerprint, config=config,
            provider=body.provider, model=body.model,
        )
        proposed["coverage"] = {"source_hash": document.source_hash, "scan_completed": False}
        run = repository.create_run(proposed, reuse_scan=True)
        if run["id"] != proposed["id"]:
            return run
        try:
            candidates, coverage = mine_term_candidates(
                store.list_segments(document_id), config, source_lang=project.source_lang,
            )
            counts = model_review_counts(candidates, body.max_model_candidates)
            coverage |= {
                "source_hash": document.source_hash, "scan_completed": True,
                "model_candidates_requested": min(len(candidates), body.max_model_candidates),
                "missing_model_decisions": counts["missing_decisions"],
                "review_compute_mode": body.compute_mode,
            }
            # Persist the entire scan before the first model request.
            repository.replace_candidates(run["id"], candidates, coverage=coverage)
            run = repository.get_run(run["id"])
            if body.provider and body.max_model_candidates:
                return await review_in_task(
                    run, body.provider, body.model, body.compute_mode, body.max_model_candidates,
                )
            return repository.update_run(run["id"], status="completed", completed_at=utc_now())
        except Exception as exc:
            current = repository.get_run(run["id"])
            repository.update_run(
                run["id"], status="partial" if current["coverage"].get("scan_completed") else "failed",
                error=str(exc)[:1_000], completed_at=utc_now(),
            )
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/documents/{document_id}/term-discovery-runs/{run_id}/retry")
    async def retry_term_discovery_run(document_id: str, run_id: str, body: TermDiscoveryRetry):
        document = store.get_document(document_id)
        try:
            run = repository.get_run(run_id)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if run["document_id"] != document_id:
            raise HTTPException(status_code=404, detail="Term discovery run not found in this document")
        coverage = run["coverage"]
        if run["status"] == "running":
            return run
        if not (coverage.get("scan_completed") or (
            coverage.get("segments_scanned", 0) > 0
            and coverage.get("segments_scanned") == coverage.get("segments_total")
        )):
            raise HTTPException(status_code=409, detail="全文扫描尚未完成，暂无可复用的候选。")
        if coverage.get("source_hash", document.source_hash) != document.source_hash:
            raise HTTPException(status_code=409, detail="原文已变化，需要重新生成候选。")
        validate_provider(body.provider, body.model)
        provider = body.provider or coverage.get("review_provider") or run["provider"]
        model = body.model or coverage.get("review_model") or run["model"]
        validate_provider(provider, model)
        if not provider:
            raise HTTPException(status_code=422, detail="请先配置术语发现模型。")
        mode = body.compute_mode or coverage.get("review_compute_mode") or "balanced"
        limit = body.max_model_candidates or coverage.get("review_limit") or (
            run["config"].get("max_model_candidates") or 40
        )
        if not repository.claim_review(run_id):
            return repository.get_run(run_id)
        return await review_in_task(run, provider, model, mode, limit)

    @app.get("/documents/{document_id}/term-discovery-runs")
    async def list_term_discovery_runs(document_id: str):
        store.get_document(document_id)
        return repository.list_runs(document_id)

    @app.get("/documents/{document_id}/term-candidates")
    async def list_term_candidates(
        document_id: str,
        run_id: str = "",
        status: str = "",
        limit: int = Query(default=500, ge=1, le=2_000),
    ):
        store.get_document(document_id)
        if status and status not in {"pending", "approved", "rejected"}:
            raise HTTPException(status_code=422, detail="Unsupported candidate status")
        return repository.list_candidates(document_id, run_id=run_id, status=status, limit=limit)

    @app.patch("/term-candidate-senses/{sense_id}")
    async def review_term_candidate_sense(sense_id: str, body: TermCandidateReview):
        try:
            return repository.review_sense(
                sense_id,
                status=body.status,
                proposed_target=body.proposed_target.strip(),
                sense=body.sense.strip(),
                rationale=body.rationale.strip(),
                disambiguation=body.disambiguation.strip(),
                actor=body.actor.strip() or "human",
            )
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/term-candidate-senses/{sense_id}/revoke")
    async def revoke_term_candidate_approval(sense_id: str, body: TermCandidateRevocation):
        try:
            term_id, project_id = repository.revoke_approval(
                sense_id, actor=body.actor.strip() or "human",
            )
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        reindex_project_quality(store, project_id)
        return {"removed_term_id": term_id, "candidate": repository.get_sense(sense_id)}

    @app.post("/term-candidate-senses/{sense_id}/approve")
    async def approve_term_candidate_sense(sense_id: str, body: TermCandidateApproval):
        try:
            term, candidate = repository.approve_sense(
                sense_id,
                target=body.target,
                enforcement=body.enforcement,
                sense=body.sense,
                rationale=body.rationale,
                context_keywords=tuple(
                    dict.fromkeys(item.strip() for item in body.context_keywords if item.strip())
                ),
                disambiguation=body.disambiguation,
                actor=body.actor.strip() or "human",
            )
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except sqlite3.IntegrityError as exc:
            if "UNIQUE constraint failed" in str(exc):
                raise HTTPException(
                    status_code=409,
                    detail="Term sense already exists in this scope",
                ) from exc
            raise
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        affected = []
        if term.enforcement != "reference":
            project_terms = store.list_terms(term.project_id)
            for document in store.list_documents(term.project_id):
                for segment in store.list_segments(document.id):
                    if not any(
                        term_appears(segment.source_text, form) for form in (term.source, *term.aliases)
                    ):
                        continue
                    target = visible_translation(segment)
                    if not target:
                        continue
                    issues = run_deterministic_checks(
                        segment.source_text, target, project_terms, segment_kind=segment.kind
                    )
                    affected.append(
                        {
                            "document_id": document.id,
                            "segment_id": segment.id,
                            "ordinal": segment.ordinal,
                            "issue_codes": [issue.code for issue in issues],
                            "needs_revision": any(issue.severity is IssueSeverity.ERROR for issue in issues),
                            "pending_verification": any(issue.code == "terminology_pending"
                                                        for issue in issues),
                        }
                    )
        reindex_project_quality(store, term.project_id)
        impact = {
            "translated_occurrences_checked": len(affected),
            "segments_needing_revision": sum(item["needs_revision"] for item in affected),
            "segments_pending_verification": sum(item["pending_verification"] for item in affected),
            "segments": affected,
            "candidate_sense_id": sense_id,
            "run_id": candidate["run_id"],
        }
        _record_impact(store, term.id, impact)
        return {"term": asdict(term), "impact": impact}
