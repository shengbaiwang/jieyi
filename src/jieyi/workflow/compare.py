from __future__ import annotations

from dataclasses import asdict, replace

from jieyi.domain.models import (
    CandidateStage,
    Job,
    JobStatus,
    ModelSpec,
    TranslationRequest,
    TranslationResult,
    new_id,
)
from jieyi.prompting import build_messages
from jieyi.protection import PlaceholderIntegrityError
from jieyi.quality.checks import run_deterministic_checks
from jieyi.workflow.optimized import _complete_one, _strip_fence, _sum_results
from jieyi.workflow.requests import initial_output_budget, prepare_translation, project_context

_MAX_MODELS = 6
_REPAIR_ATTEMPTS = 2


class ModelCompareError(ValueError):
    """Local, user-facing failure for one compare request."""


def _visible_candidate_payload(candidate_id: str, result: TranslationResult, *, issues, error: str | None) -> dict:
    return {
        "candidate_id": candidate_id,
        "text": result.text if error is None else "",
        "error": error,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "reasoning_tokens": result.reasoning_tokens,
        "cost_usd": result.cost_usd,
        "issues": [
            {"code": issue.code, "message": issue.message, "severity": issue.severity.value}
            for issue in issues
        ],
    }


async def compare_segment_models(
    engine,
    segment_id: str,
    models: list[dict],
    *,
    compute_mode: str = "economy",
) -> dict:
    """Run several models on one segment and store their candidates.

    Results never overwrite machine/edited/accepted translations. The caller
    (human) decides which draft, if any, enters the working copy.
    """
    if not models:
        raise ModelCompareError("请至少选择一个模型。")
    if len(models) > _MAX_MODELS:
        raise ModelCompareError(f"一次最多对比 {_MAX_MODELS} 个模型。")

    unique_keys: list[tuple[str, str]] = []
    specs: list[ModelSpec] = []
    for item in models:
        provider = str(item.get("provider") or "").strip()
        model = str(item.get("model") or "").strip()
        if not provider or not model:
            raise ModelCompareError("每个对比项都要指定连接与模型 ID。")
        key = (provider, model)
        if key in unique_keys:
            raise ModelCompareError(f"模型重复：{provider} / {model}")
        unique_keys.append(key)
        try:
            engine.providers.get(provider)
        except KeyError as exc:
            raise ModelCompareError(f"Provider is not configured: {provider}") from exc
        specs.append(ModelSpec(provider=provider, model=model))

    segment = engine.store.get_segment(segment_id)
    document = engine.store.get_document(segment.document_id)
    project = engine.store.get_project_for_document(segment.document_id)
    all_segments = engine.store.list_segments(segment.document_id)
    segments_by_ordinal = {item.ordinal: item for item in all_segments}
    approved_terms = [
        term for term in engine.store.list_terms(project.id) if term.status.value == "approved"
    ]

    recipe = replace(
        _default_recipe(specs[0]),
        draft_compute_mode=compute_mode,
        segment_ranges=((segment.ordinal, segment.ordinal),),
    )
    job = engine.store.create_job(
        Job(id=new_id("job"), document_id=segment.document_id, recipe=recipe),
        reuse_unfinished=False,
    )
    reservation = engine.reserve(job.id)
    try:
        engine.store.set_job_status(job.id, JobStatus.RUNNING)
        shared_context = project_context(project, recipe.max_context_chars)
        prepared = prepare_translation(
            engine.store,
            engine.protected_text_codec,
            project,
            document,
            segment,
            recipe,
            approved_terms,
            segments_by_ordinal,
            shared_context=shared_context,
        )
        results: list[dict] = []
        total_cost = 0.0

        async def run_one(spec: ModelSpec) -> dict:
            nonlocal total_cost
            request = replace(prepared.request, task=CandidateStage.COMPARE)
            provider = engine.providers.get(spec.provider)
            usage_parts: list[TranslationResult] = []
            last_text = ""
            try:
                completed = await _complete_one(
                    provider,
                    request,
                    spec,
                    thinking=compute_mode != "economy",
                    reasoning_effort=recipe.draft_reasoning_effort,
                    compute_mode=compute_mode,
                    max_tokens=initial_output_budget(
                        request, compute_mode, recipe.max_output_tokens
                    ),
                    max_output_tokens=recipe.max_output_tokens,
                    record_call=lambda payload: engine.store.record_model_call(job.id, {
                        **payload,
                        "purpose": "model_compare",
                    }),
                )
                usage_parts.append(completed)
                last_text = _strip_fence(completed.text)
                restored_text, usage_parts = await _restore_with_optional_repair(
                    engine,
                    job,
                    provider,
                    prepared,
                    request,
                    spec,
                    last_text,
                    compute_mode=compute_mode,
                    recipe=recipe,
                    usage_parts=usage_parts,
                )
                if usage_parts:
                    last_text = usage_parts[-1].text
                merged = _sum_results(usage_parts, restored_text)
                issues = run_deterministic_checks(
                    segment.source_text,
                    restored_text,
                    prepared.terms,
                    segment_kind=segment.kind,
                )
                candidate_id = engine.store.record_candidate(
                    job_id=job.id,
                    segment_id=segment.id,
                    stage=CandidateStage.COMPARE,
                    provider=spec.provider,
                    model=spec.model,
                    result=merged,
                )
                total_cost += merged.cost_usd
                payload = _visible_candidate_payload(candidate_id, merged, issues=issues, error=None)
                payload.update({"provider": spec.provider, "model": spec.model, "compute_mode": compute_mode, "ok": True})
                return payload
            except Exception as exc:  # noqa: BLE001 -- isolate one model failure from siblings
                cost = sum(item.cost_usd for item in usage_parts)
                total_cost += cost
                raw_text = last_text or (usage_parts[-1].text if usage_parts else "")
                failed = TranslationResult(
                    text=raw_text,
                    prompt_tokens=sum(item.prompt_tokens for item in usage_parts),
                    completion_tokens=sum(item.completion_tokens for item in usage_parts),
                    reasoning_tokens=sum(item.reasoning_tokens for item in usage_parts),
                    cost_usd=cost,
                )
                candidate_id = engine.store.record_candidate(
                    job_id=job.id,
                    segment_id=segment.id,
                    stage=CandidateStage.COMPARE,
                    provider=spec.provider,
                    model=spec.model,
                    result=failed,
                )
                payload = _visible_candidate_payload(candidate_id, failed, issues=[], error=str(exc)[:500])
                payload.update({
                    "provider": spec.provider,
                    "model": spec.model,
                    "compute_mode": compute_mode,
                    "ok": False,
                    "raw_text": raw_text,
                    "text": raw_text,
                })
                return payload

        # Sequential keeps cost predictable and avoids stampeding one endpoint.
        for spec in specs:
            results.append(await run_one(spec))

        current = engine.store.get_job(job.id)
        engine.store.save_job(replace(
            current,
            status=JobStatus.COMPLETED,
            next_ordinal=segment.ordinal + 1,
            total_cost_usd=current.total_cost_usd + total_cost,
        ))
        return {
            "segment_id": segment.id,
            "document_id": segment.document_id,
            "ordinal": segment.ordinal,
            "source_text": segment.source_text,
            "compute_mode": compute_mode,
            "job_id": job.id,
            "total_cost_usd": total_cost,
            "results": results,
        }
    except Exception:
        try:
            current = engine.store.get_job(job.id)
            engine.store.save_job(replace(current, status=JobStatus.FAILED))
        except Exception:  # noqa: BLE001 -- keep the original failure
            pass
        raise
    finally:
        reservation.release()


async def _restore_with_optional_repair(
    engine,
    job,
    provider,
    prepared,
    request: TranslationRequest,
    spec: ModelSpec,
    text: str,
    *,
    compute_mode: str,
    recipe,
    usage_parts: list[TranslationResult],
) -> tuple[str, list[TranslationResult]]:
    protected = prepared.protected
    structured = prepared.structured
    try:
        return _restore_text(engine, protected, structured, request.segment.id, text), usage_parts
    except (PlaceholderIntegrityError, ValueError) as initial_error:
        deterministic = protected.repair_surplus_placeholders(text)
        if deterministic is not None:
            try:
                return _restore_text(
                    engine, protected, structured, request.segment.id, deterministic
                ), usage_parts
            except (PlaceholderIntegrityError, ValueError):
                pass
        error = initial_error
        for attempt in range(1, _REPAIR_ATTEMPTS + 1):
            repair_request = replace(
                request,
                task=CandidateStage.REPAIR,
                existing_translation=text,
                issue_summary=(
                    f"Repair attempt {attempt} of {_REPAIR_ATTEMPTS}. "
                    f"Previous validation error: {error}"
                ),
            )
            repaired = await _complete_one(
                provider,
                repair_request,
                spec,
                thinking=compute_mode != "economy",
                reasoning_effort=recipe.draft_reasoning_effort,
                compute_mode=compute_mode,
                max_tokens=initial_output_budget(
                    repair_request, compute_mode, recipe.max_output_tokens
                ),
                max_output_tokens=recipe.max_output_tokens,
            )
            usage_parts.append(repaired)
            candidate_text = _strip_fence(repaired.text)
            try:
                restored = _restore_text(
                    engine, protected, structured, request.segment.id, candidate_text
                )
                return restored, usage_parts
            except (PlaceholderIntegrityError, ValueError) as repair_error:
                error = repair_error
        raise PlaceholderIntegrityError(
            f"占位符修复失败：{error}",
            missing=getattr(error, "missing", ()),
            extra=getattr(error, "extra", ()),
        ) from error


def _restore_text(engine, protected, structured: bool, segment_id: str, text: str) -> str:
    restored_value = protected.restore(text)
    if structured:
        return engine.store.capture_epub_translation(
            segment_id, restored_value, CandidateStage.COMPARE.value, persist=False
        )
    return restored_value


def _default_recipe(spec: ModelSpec):
    from jieyi.domain.models import TranslationRecipe

    return TranslationRecipe(
        draft=spec,
        batch_size=1,
        concurrency=1,
        max_concurrency=1,
        draft_compute_mode="economy",
        draft_reasoning_effort="none",
    )


def build_compare_prompt_preview(engine, segment_id: str, models: list[dict], *, compute_mode: str = "economy") -> dict:
    """Shared prompt shape for compare and tests; no model call."""
    if not models:
        raise ModelCompareError("请至少选择一个模型。")
    segment = engine.store.get_segment(segment_id)
    document = engine.store.get_document(segment.document_id)
    project = engine.store.get_project_for_document(segment.document_id)
    all_segments = engine.store.list_segments(segment.document_id)
    segments_by_ordinal = {item.ordinal: item for item in all_segments}
    approved_terms = [
        term for term in engine.store.list_terms(project.id) if term.status.value == "approved"
    ]
    spec = ModelSpec(
        provider=str(models[0].get("provider") or "echo"),
        model=str(models[0].get("model") or "dry-run"),
    )
    recipe = replace(
        _default_recipe(spec),
        draft_compute_mode=compute_mode,
        segment_ranges=((segment.ordinal, segment.ordinal),),
    )
    prepared = prepare_translation(
        engine.store,
        engine.protected_text_codec,
        project,
        document,
        segment,
        recipe,
        approved_terms,
        segments_by_ordinal,
        shared_context=project_context(project, recipe.max_context_chars),
    )
    request = replace(prepared.request, task=CandidateStage.COMPARE)
    return {
        "messages": build_messages(request),
        "protected_spans": [asdict(span) for span in prepared.protected.spans],
        "compute_mode": compute_mode,
        "models": [{"provider": item.get("provider"), "model": item.get("model")} for item in models],
    }
