from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import asdict

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from jieyi.domain.models import ModelSpec, utc_now
from jieyi.quality import reindex_project_quality, run_deterministic_checks
from jieyi.quality.service import visible_translation
from jieyi.term_discovery import (
    DiscoveryConfig,
    discovery_fingerprint,
    enrich_candidates,
    mine_term_candidates,
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
    model_batch_size: int = Field(default=8, ge=1, le=20)
    min_score: float = Field(default=0.34, ge=0, le=1)


class TermCandidateReview(BaseModel):
    status: str = "pending"
    proposed_target: str = ""
    sense: str = ""
    rationale: str = ""
    disambiguation: str = ""
    actor: str = "human"


class TermCandidateApproval(BaseModel):
    target: str = Field(min_length=1)
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

    @app.post("/documents/{document_id}/term-discovery-runs", status_code=201)
    async def create_term_discovery_run(document_id: str, body: TermDiscoveryCreate):
        document = store.get_document(document_id)
        project = store.get_project(document.project_id)
        segments = store.list_segments(document_id)
        if bool(body.provider) != bool(body.model):
            raise HTTPException(
                status_code=422,
                detail="provider and model must be supplied together",
            )
        if body.provider not in providers.names() and body.provider:
            raise HTTPException(
                status_code=422,
                detail=f"Provider is not configured: {body.provider}",
            )
        config = DiscoveryConfig(
            max_candidates=body.max_candidates,
            max_evidence_per_candidate=body.max_evidence_per_candidate,
            max_model_candidates=body.max_model_candidates,
            model_batch_size=body.model_batch_size,
            min_score=body.min_score,
        )
        fingerprint = discovery_fingerprint(
            f"{document.source_hash}:{_translated_state_hash(segments)}:"
            f"{body.provider}:{body.model}:{body.compute_mode}",
            config,
        )
        running = repository.find_running_run(document_id, fingerprint)
        if running is not None:
            return running
        run = new_discovery_run(
            document_id=document_id,
            fingerprint=fingerprint,
            config=config,
            provider=body.provider,
            model=body.model,
        )
        repository.create_run(run)
        try:
            candidates, coverage = mine_term_candidates(
                segments, config, source_lang=project.source_lang
            )
            usage = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "reasoning_tokens": 0,
                "cost_usd": 0.0,
                "model_calls": 0,
                "invalid_proposals": 0,
                "model_decisions": 0,
                "missing_decisions": 0,
                "model_kept": 0,
                "model_omitted": 0,
            }
            model_error = ""
            if body.provider and body.max_model_candidates:
                provider = providers.get(body.provider)
                complete = getattr(provider, "complete", None)
                if complete is None:
                    model_error = "Configured provider does not support analysis completions"
                else:
                    try:
                        candidates, usage = await enrich_candidates(
                            candidates,
                            provider=provider,
                            model=ModelSpec(
                                provider=body.provider,
                                model=body.model,
                                temperature=0.0,
                            ),
                            source_lang=str(
                                coverage.get("language_profile") or project.source_lang
                            ),
                            target_lang=project.target_lang,
                            config=config,
                            compute_mode=body.compute_mode,
                        )
                    except Exception as exc:  # noqa: BLE001 — third-party adapter boundary
                        model_error = str(exc)[:1_000]
            if usage["missing_decisions"] and not model_error:
                model_error = (
                    f"Model review incomplete after retry: "
                    f"{usage['missing_decisions']} candidate decision(s) missing"
                )
            coverage = coverage | {
                "model_candidates_requested": min(len(candidates), body.max_model_candidates),
                "model_calls": usage["model_calls"],
                "invalid_model_proposals": usage["invalid_proposals"],
                "model_decisions": usage["model_decisions"],
                "missing_model_decisions": usage["missing_decisions"],
                "model_kept": usage["model_kept"],
                "model_omitted": usage["model_omitted"],
                "model_error": model_error,
            }
            repository.replace_candidates(run["id"], candidates)
            completed = repository.update_run(
                run["id"],
                status="completed",
                coverage=coverage,
                prompt_tokens=usage["prompt_tokens"],
                completion_tokens=usage["completion_tokens"],
                reasoning_tokens=usage["reasoning_tokens"],
                cost_usd=usage["cost_usd"],
                error=model_error,
                completed_at=utc_now(),
            )
            return completed
        except Exception as exc:
            repository.update_run(
                run["id"],
                status="failed",
                error=str(exc)[:1_000],
                completed_at=utc_now(),
            )
            raise HTTPException(status_code=500, detail=str(exc)) from exc

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

    @app.post("/term-candidate-senses/{sense_id}/approve")
    async def approve_term_candidate_sense(sense_id: str, body: TermCandidateApproval):
        try:
            term, candidate = repository.approve_sense(
                sense_id,
                target=body.target,
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
                    segment.source_text, target, [term], segment_kind=segment.kind
                )
                affected.append(
                    {
                        "document_id": document.id,
                        "segment_id": segment.id,
                        "ordinal": segment.ordinal,
                        "issue_codes": [issue.code for issue in issues],
                        "needs_revision": bool(issues),
                    }
                )
        reindex_project_quality(store, term.project_id)
        impact = {
            "translated_occurrences_checked": len(affected),
            "segments_needing_revision": sum(item["needs_revision"] for item in affected),
            "segments": affected,
            "candidate_sense_id": sense_id,
            "run_id": candidate["run_id"],
        }
        _record_impact(store, term.id, impact)
        return {"term": asdict(term), "impact": impact}
