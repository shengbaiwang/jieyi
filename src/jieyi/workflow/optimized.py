from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import replace

from jieyi.domain.models import (
    CandidateStage,
    IssueSeverity,
    Job,
    JobStatus,
    QualityIssue,
    Segment,
    SegmentStatus,
    TermEntry,
    TranslationRequest,
    TranslationResult,
)
from jieyi.prompting import build_messages, is_minimal_prompt_request
from jieyi.protection import PlaceholderIntegrityError, ProtectedText
from jieyi.providers.openai_compatible import ReasoningMandatoryError
from jieyi.quality.checks import (
    DETECTOR_VERSION,
    run_deterministic_checks,
)
from jieyi.quality.output import OUTPUT_RETRY_RULE, TranslationOutputError, output_problems
from jieyi.workflow.cohesion import (
    build_cohesion_messages,
    parse_cohesion_revisions,
    plan_cohesion_groups,
)
from jieyi.workflow.provider_responses import (
    EmptyProviderResponseError,
    content_filter_audit_payload,
    inspect_empty_result,
    is_content_filtered_error,
    is_incomplete_result,
    response_stop_reason,
    should_expand_output_budget,
)
from jieyi.workflow.requests import initial_output_budget, prepare_translation, project_context

_PLACEHOLDER_REPAIR_ATTEMPTS = 3


class _SegmentChanged(Exception):
    """A queued request became unnecessary before it obtained a provider slot."""

    def __init__(self, results=()):
        self.results = results


class _AdaptiveConcurrencyLimiter:
    """A conservative additive-increase/multiplicative-decrease request gate."""

    def __init__(self, initial: int, maximum: int):
        self.initial = max(1, initial)
        self.maximum = max(self.initial, maximum)
        self.limit = self.initial
        self.active = 0
        self._successes = 0
        self._condition = asyncio.Condition()

    async def acquire(self) -> None:
        async with self._condition:
            await self._condition.wait_for(lambda: self.active < self.limit)
            self.active += 1

    async def release(self, *, success: bool) -> None:
        async with self._condition:
            self.active = max(0, self.active - 1)
            if success:
                self._successes += 1
                if self.limit < self.maximum and self._successes >= self.limit:
                    self.limit += 1
                    self._successes = 0
            else:
                self.limit = max(1, self.limit // 2)
                self._successes = 0
            self._condition.notify_all()

    async def complete(self, operation):
        await self.acquire()
        try:
            result = await operation()
        except BaseException:
            await self.release(success=False)
            raise
        await self.release(success=True)
        return result


def _visible_translation(segment: Segment) -> str:
    return (
        segment.accepted_translation
        or segment.reviewed_translation
        or segment.edited_translation
        or segment.machine_translation
        or ""
    )


def _groups(segments: list[Segment], batch_size: int, max_chars: int) -> list[list[Segment]]:
    groups: list[list[Segment]] = []
    current: list[Segment] = []
    current_chars = 0
    for segment in segments:
        size = len(segment.source_text) + len(_visible_translation(segment))
        if current and (len(current) >= batch_size or current_chars + size > max_chars):
            groups.append(current)
            current = []
            current_chars = 0
        current.append(segment)
        current_chars += size
    if current:
        groups.append(current)
    return groups


def _sum_results(results: list[TranslationResult], text: str) -> TranslationResult:
    return TranslationResult(
        text=text,
        prompt_tokens=sum(item.prompt_tokens for item in results),
        completion_tokens=sum(item.completion_tokens for item in results),
        cost_usd=sum(item.cost_usd for item in results),
        reasoning_tokens=sum(item.reasoning_tokens for item in results),
        prompt_cache_hit_tokens=sum(item.prompt_cache_hit_tokens for item in results),
        prompt_cache_miss_tokens=sum(item.prompt_cache_miss_tokens for item in results),
        raw_response=results[-1].raw_response if results else None,
    )


def _strip_fence(text: str) -> str:
    value = text.strip()
    lines = value.splitlines()
    if len(lines) >= 3 and lines[0].strip().startswith("```") and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return value


async def _complete_one(
    provider,
    request: TranslationRequest,
    model_spec,
    *,
    thinking: bool,
    reasoning_effort: str,
    compute_mode: str,
    max_tokens: int,
    max_output_tokens: int,
    record_call: Callable[[dict[str, object]], None] | None = None,
    can_dispatch: Callable[[], bool] | None = None,
) -> TranslationResult:
    """Retry only failures that diagnostics identify as recoverable."""
    results: list[TranslationResult] = []
    attempts = []
    budget = min(max_tokens, max_output_tokens)
    messages = build_messages(request)
    # Minimal-tier requests drop project/neighbor context that was empty
    # anyway; still count it explicitly so audits stay comparable.
    minimal_tier = is_minimal_prompt_request(request)
    for attempt_number in range(1, 4):
        if can_dispatch is not None and not can_dispatch():
            raise _SegmentChanged(results)
        started = time.monotonic()
        result = await provider.complete(
            messages,
            model_spec,
            thinking=thinking,
            reasoning_effort=reasoning_effort,
            compute_mode=compute_mode,
            max_tokens=budget,
        )
        results.append(result)
        text = _strip_fence(result.text)
        problems = output_problems(request.segment.source_text, text)
        stop_reason = response_stop_reason(result)
        attempt = inspect_empty_result(
            result,
            attempt=attempt_number,
            max_tokens=budget,
        )
        if record_call is not None:
            record_call({
                "segment_id": request.segment.id,
                "stage": request.task.value,
                "attempt": attempt_number,
                "provider": model_spec.provider,
                "model": model_spec.model,
                "compute_mode": compute_mode,
                "max_tokens": budget,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "reasoning_tokens": result.reasoning_tokens,
                "cache_hit_tokens": result.prompt_cache_hit_tokens,
                "cache_miss_tokens": result.prompt_cache_miss_tokens,
                "cost_usd": result.cost_usd,
                "source_chars": len(request.segment.source_text),
                "context_chars": len(request.context) + len(request.segment_context),
                "prompt_chars": sum(len(message["content"]) for message in messages),
                "output_chars": len(text),
                "finish_reason": stop_reason,
                "minimal_prompt": minimal_tier,
                "outcome": (
                    "output_contamination" if problems else
                    "complete" if text and not is_incomplete_result(result)
                    and attempt.kind != "content_filtered" else attempt.kind
                ),
                "elapsed_seconds": round(time.monotonic() - started, 3),
            })
        if text and not is_incomplete_result(result) and attempt.kind != "content_filtered":
            if problems:
                if attempt_number == 3:
                    raise TranslationOutputError(problems, results)
                messages = build_messages(request)
                messages[0]["content"] += "\n\n" + OUTPUT_RETRY_RULE
                continue
            return _sum_results(results, text)
        attempts.append(attempt)
        if (
            should_expand_output_budget(attempt)
            and attempt_number < 3
            and budget < max_output_tokens
        ):
            budget = min(max_output_tokens, max(2_048, budget * 2))
            continue
        if attempt.kind == "upstream_empty_response" and attempt_number == 1:
            # A single same-budget retry covers transient 200/empty gateway responses.
            continue
        break

    raise EmptyProviderResponseError(
        segment_id=request.segment.id,
        segment_ordinal=request.segment.ordinal,
        stage=request.task.value,
        provider=model_spec.provider,
        model=model_spec.model,
        attempts=attempts,
        results=results,
    )


async def _translate_group(engine, job, project, document, segments, **options):
    # Validate and checkpoint each segment immediately. A slow sibling must not
    # make a completed result disappear when the enclosing batch is paused.
    completed = {}
    started = time.monotonic()

    async def one(segment):
        result = await _translate_segment(engine, job, project, document, [segment], **options)
        completed[segment.id] = result
        return result

    tasks = [asyncio.create_task(one(segment)) for segment in segments]
    try:
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if completed:
            usage = _sum_results([result[4] for result in completed.values()], "")
            engine.store.record_batch(
                job_id=job.id, stage=CandidateStage.DRAFT,
                start_ordinal=segments[0].ordinal, end_ordinal=segments[-1].ordinal,
                segment_count=sum(len(result[2]) for result in completed.values()),
                result=usage, elapsed_seconds=time.monotonic() - started,
            )
            current = engine.store.get_job(job.id)
            engine.store.save_job(replace(current, total_cost_usd=current.total_cost_usd + usage.cost_usd))
        raise
    translations, stages, terms, failures, usages = {}, {}, {}, {}, []
    cohesion_committed: list[Segment] = []
    for segment, outcome in zip(segments, outcomes, strict=True):
        if isinstance(outcome, BaseException):
            failures[segment.id] = outcome
            terms[segment.id] = []
            continue
        _, _, text, candidate_stages, usage, _, segment_terms, segment_failures, committed = outcome
        translations.update(text)
        stages.update(candidate_stages)
        terms.update(segment_terms)
        failures.update(segment_failures)
        usages.append(usage)
        cohesion_committed.extend(committed)
    if cohesion_committed:
        try:
            cohesion_result = await _run_cohesion_for_group(
                engine, job, project, document, cohesion_committed,
                shared_context=options.get("project_context", ""),
                approved_terms=options.get("approved_terms", []),
                limiter=options["limiter"],
            )
            if cohesion_result is not None:
                usages.append(cohesion_result)
        except _SegmentChanged as stopped:
            usages.extend(stopped.results)
    return (CandidateStage.DRAFT, job.recipe.draft, translations, stages,
            _sum_results(usages, ""), time.monotonic() - started, terms, failures)


async def _translate_segment(
    engine,
    job: Job,
    project,
    document,
    segments: list[Segment],
    *,
    project_context: str,
    approved_terms: list[TermEntry],
    segments_by_ordinal: dict[int, Segment],
    limiter: _AdaptiveConcurrencyLimiter,
):
    stage = CandidateStage.DRAFT
    model_spec = job.recipe.draft
    provider = engine.providers.get(model_spec.provider)
    prepared = [
        prepare_translation(
            engine.store, engine.protected_text_codec, project, document, segment,
            job.recipe, approved_terms, segments_by_ordinal, shared_context=project_context,
        )
        for segment in segments
    ]
    requests = [item.request for item in prepared]
    protected_by_id = {item.request.segment.id: item.protected for item in prepared}
    structured_by_id = {item.request.segment.id: item.structured for item in prepared}
    terms_by_id = {item.request.segment.id: item.terms for item in prepared}
    context_by_id = {item.request.segment.id: item.request.segment_context for item in prepared}
    started = time.monotonic()
    translations: dict[str, str] = {}
    usage_results: list[TranslationResult] = []
    thinking = job.recipe.draft_thinking
    reasoning_effort = job.recipe.draft_reasoning_effort
    compute_mode = job.recipe.draft_compute_mode

    def can_dispatch(segment_id):
        current = engine.store.get_segment(segment_id)
        original = next(item for item in segments if item.id == segment_id)
        return (
            engine.store.get_job(job.id).status is JobStatus.RUNNING
            and not _visible_translation(current)
            and current.status is not SegmentStatus.HUMAN_CONFIRMED
            and current.source_text == original.source_text
        )

    async def translate_one(request: TranslationRequest) -> tuple[str, TranslationResult]:
        async def operation():
            return await _complete_one(
                provider,
                request,
                model_spec,
                thinking=thinking,
                reasoning_effort=reasoning_effort,
                compute_mode=compute_mode,
                max_tokens=initial_output_budget(request, compute_mode, job.recipe.max_output_tokens),
                max_output_tokens=job.recipe.max_output_tokens,
                record_call=lambda payload: engine.store.record_model_call(job.id, payload),
                can_dispatch=lambda: can_dispatch(request.segment.id),
            )
        result = await limiter.complete(operation)
        return request.segment.id, result

    # Resolve every segment independently so one provider refusal does not discard
    # successful siblings from the same batch.
    failures: dict[str, BaseException] = {}
    completed = await asyncio.gather(
        *(translate_one(request) for request in requests),
        return_exceptions=True,
    )
    for request, outcome in zip(requests, completed, strict=True):
        if isinstance(outcome, _SegmentChanged):
            usage_results.extend(outcome.results)
            continue
        if isinstance(outcome, BaseException):
            failures[request.segment.id] = outcome
            if isinstance(outcome, EmptyProviderResponseError):
                usage_results.append(outcome.usage_result())
            elif isinstance(outcome, TranslationOutputError):
                usage_results.extend(outcome.results)
            continue
        segment_id, result = outcome
        translations[segment_id] = result.text
        usage_results.append(result)
    restored: dict[str, str] = {}
    candidate_stages: dict[str, CandidateStage] = {}
    structured_values: dict[str, str] = {}
    for segment in segments:
        if segment.id not in translations:
            continue
        text = translations[segment.id]
        protected = protected_by_id[segment.id]
        try:
            restored_value = protected.restore(text)
            restored[segment.id] = (
                engine.store.capture_epub_translation(
                    segment.id,
                    restored_value,
                    "draft", persist=False,
                )
                if structured_by_id[segment.id]
                else restored_value
            )
            structured_values[segment.id] = restored_value
            candidate_stages[segment.id] = stage
            continue
        except (PlaceholderIntegrityError, ValueError) as initial_error:
            repair_error: BaseException = initial_error
            deterministic = protected.repair_surplus_placeholders(text)
            if deterministic is not None:
                try:
                    restored_value = protected.restore(deterministic)
                    restored[segment.id] = (
                        engine.store.capture_epub_translation(
                            segment.id,
                            restored_value,
                            "draft", persist=False,
                        )
                        if structured_by_id[segment.id]
                        else restored_value
                    )
                    structured_values[segment.id] = restored_value
                    candidate_stages[segment.id] = stage
                    continue
                except (PlaceholderIntegrityError, ValueError) as deterministic_error:
                    repair_error = deterministic_error

            # Keep the broken output for diagnosis. Usage is aggregated in the
            # batch record, so this snapshot intentionally has no call metrics.
            engine.store.record_candidate(
                job_id=job.id,
                segment_id=segment.id,
                stage=stage,
                provider=model_spec.provider,
                model=model_spec.model,
                result=TranslationResult(text=text),
            )
            source_chars = len(protected.masked) + len(text)
            for repair_attempt in range(1, _PLACEHOLDER_REPAIR_ATTEMPTS + 1):
                repair_request = TranslationRequest(
                    project=project,
                    document=document,
                    segment=replace(segment, source_text=protected.masked),
                    atom_boundaries=protected.atom_boundaries if structured_by_id[segment.id] else (),
                    context=project_context,
                    segment_context=context_by_id[segment.id],
                    task=CandidateStage.REPAIR,
                    # Always repair the original draft. A failed repair commonly strips every
                    # marker, so chaining repair outputs destroys useful placement information.
                    existing_translation=text,
                    issue_summary=(
                        f"Repair attempt {repair_attempt} of "
                        f"{_PLACEHOLDER_REPAIR_ATTEMPTS}. Previous validation error: "
                        f"{repair_error}"
                    ),
                )
                try:
                    async def repair_operation(
                        repair_request=repair_request,
                        source_chars=source_chars,
                    ):
                        return await _complete_one(
                            provider,
                            repair_request,
                            model_spec,
                            thinking=False,
                            reasoning_effort="none",
                            compute_mode="economy",
                            max_tokens=min(
                                job.recipe.max_output_tokens,
                                max(512, source_chars * 2),
                            ),
                            max_output_tokens=job.recipe.max_output_tokens,
                            record_call=lambda payload: engine.store.record_model_call(job.id, payload),
                            can_dispatch=lambda: can_dispatch(repair_request.segment.id),
                        )
                    repaired = await limiter.complete(repair_operation)
                    usage_results.append(repaired)
                    restored_value = protected.restore(
                        protected.assemble_atom_repair(repaired.text)
                        if structured_by_id[segment.id] else repaired.text
                    )
                    restored[segment.id] = (
                        engine.store.capture_epub_translation(segment.id, restored_value, "repair", persist=False)
                        if structured_by_id[segment.id]
                        else restored_value
                    )
                    structured_values[segment.id] = restored_value
                    candidate_stages[segment.id] = CandidateStage.REPAIR
                    break
                except _SegmentChanged as stopped:
                    usage_results.extend(stopped.results)
                    break
                except TranslationOutputError as exc:
                    usage_results.extend(exc.results)
                    failures[segment.id] = exc
                    break
                except EmptyProviderResponseError as exc:
                    usage_results.append(exc.usage_result())
                    repair_error = exc
                except (
                    PlaceholderIntegrityError,
                    RuntimeError,
                    ValueError,
                ) as exc:
                    repair_error = exc
            else:
                if isinstance(repair_error, EmptyProviderResponseError):
                    failures[segment.id] = repair_error
                else:
                    failures[segment.id] = PlaceholderIntegrityError(
                        "Placeholder repair failed after "
                        f"{_PLACEHOLDER_REPAIR_ATTEMPTS} attempts for segment "
                        f"{segment.ordinal + 1} ({segment.id}): {repair_error}",
                        missing=getattr(repair_error, "missing", ()),
                        extra=getattr(repair_error, "extra", ()),
                    )
    for segment in segments:
        if segment.id not in restored:
            continue
        text = restored[segment.id]
        issues = run_deterministic_checks(
            segment.source_text, text, terms_by_id[segment.id], segment_kind=segment.kind,
            source_lang=project.source_lang, target_lang=project.target_lang,
        )
        if not engine.store.commit_generated_translation(
            job=job, segment=segment, text=text, issues=issues,
            detector_version=DETECTOR_VERSION, stage=candidate_stages[segment.id],
            structured_value=structured_values.get(segment.id) if structured_by_id[segment.id] else None,
        ):
            restored.pop(segment.id)
    usage = _sum_results(usage_results, "")
    elapsed = time.monotonic() - started
    # Fresh translations committed by this batch: the only input eligible for
    # the group cohesion pass. Deferred/failed segments are never polished.
    # The cohesion pass runs inside the same dispatch group so batch counts,
    # checkpoints, and max_batches semantics stay unchanged.
    cohesion_committed: list[Segment] = []
    if getattr(job.recipe, "cohesion_enabled", False):
        cohesion_committed = [
            segment for segment in segments
            if segment.id in restored
            and engine.store.get_segment(segment.id).source_text == segment.source_text
        ]
    return (
        stage,
        model_spec,
        restored,
        candidate_stages,
        usage,
        elapsed,
        terms_by_id,
        failures,
        cohesion_committed,
    )


def _can_defer_segment_failure(error: BaseException) -> bool:
    """Quarantine exhausted segment responses; transport/configuration errors still stop."""
    if isinstance(error, ReasoningMandatoryError):
        return False
    if is_content_filtered_error(error):
        return True
    if isinstance(error, (PlaceholderIntegrityError, TranslationOutputError)):
        return True
    # Empty responses have already exhausted their bounded retries in _complete_one.
    # Preserve their diagnostics and continue, just as for other unusable responses.
    return isinstance(error, EmptyProviderResponseError)


def _record_deferred_segment(
    engine,
    job: Job,
    segment: Segment,
    *,
    stage: CandidateStage,
    model_spec,
    error: BaseException,
    terms: list[TermEntry],
) -> None:
    """Persist a visible hard warning without accepting an invalid translation."""
    if is_content_filtered_error(error):
        payload = content_filter_audit_payload(
            error,
            stage=stage.value,
            provider=model_spec.provider,
            model=model_spec.model,
            segment_ordinal=segment.ordinal,
        )
    elif isinstance(error, EmptyProviderResponseError):
        payload = error.audit_payload()
    else:
        payload = {
            "kind": "segment_validation_failure",
            "job_stage": stage.value,
            "provider": model_spec.provider,
            "model": model_spec.model,
            "segment_ordinal": segment.ordinal,
            "message": str(error)[:1_000],
        }
    payload.update({"job_id": job.id, "deferred": True})
    engine.store.record_provider_failure(segment.id, payload)

    visible = _visible_translation(segment)
    project = engine.store.get_project_for_document(segment.document_id)
    issues = (
        run_deterministic_checks(
            segment.source_text,
            visible,
            terms,
            segment_kind=segment.kind,
            source_lang=project.source_lang,
            target_lang=project.target_lang,
        )
        if visible
        else []
    )
    if isinstance(error, TranslationOutputError):
        failure_message = "模型反复混入原文或解释，本段已隔离且未保存。可重试草译或更换模型。"
    elif isinstance(error, PlaceholderIntegrityError):
        reason = "EPUB 文本片段对齐失败" if "SourceAtom" in str(error) else "译文格式标记校验失败"
        failure_message = f"{reason}，自动修复 {_PLACEHOLDER_REPAIR_ATTEMPTS} 次仍未通过。本段尚未保存，可重试草译或更换模型。"
    elif is_content_filtered_error(error):
        failure_message = "模型拒绝生成本段译文。本段尚未保存，可更换模型或人工翻译。"
    elif isinstance(error, EmptyProviderResponseError):
        if error.kind == "upstream_empty_response":
            failure_message = "上游返回空响应，重试后仍无译文。本段已隔离待复核，可稍后重试草译或检查模型服务。"
        else:
            failure_message = "模型未返回可用译文。本段尚未保存，请检查输出额度或重试草译。"
    else:
        failure_message = "本段译文未通过验收，尚未保存。请重试草译或查看失败记录。"
    issues.append(
        QualityIssue(
            code="translation_deferred",
            message=failure_message,
            severity=IssueSeverity.ERROR,
            details={
                "kind": payload.get("kind", "segment_validation_failure"),
                "error": str(error)[:1_000],
                "job_stage": stage.value,
                "provider": model_spec.provider,
                "model": model_spec.model,
                "detector_version": DETECTOR_VERSION,
                "confidence": "high",
            },
        )
    )
    engine.store.replace_issues(
        job.id,
        segment.id,
        issues,
        target_text=visible,
        detector_version=DETECTOR_VERSION,
    )


async def _run_cohesion_for_group(
    engine,
    job: Job,
    project,
    document,
    committed: list[Segment],
    *,
    shared_context: str,
    approved_terms: list[TermEntry],
    limiter: _AdaptiveConcurrencyLimiter,
) -> TranslationResult | None:
    """Plan cohesion groups from fresh drafts and run one pass per group.

    Usage is returned to the caller so the cohesion pass stays inside the same
    dispatch-group batch record: batch counts, checkpoints, and max_batches
    semantics are unchanged. Cost still lands in job totals via that batch.
    """
    if not getattr(job.recipe, "cohesion_enabled", False):
        return None
    if engine.store.get_job(job.id).status is not JobStatus.RUNNING:
        return None
    fresh = [
        engine.store.get_segment(segment.id)
        for segment in sorted(committed, key=lambda item: item.ordinal)
    ]
    fresh = [
        segment for segment in fresh
        if segment.status is not SegmentStatus.HUMAN_CONFIRMED
        and not (segment.accepted_translation or segment.edited_translation)
        and _visible_translation(segment)
    ]
    if len(fresh) < max(2, int(getattr(job.recipe, "cohesion_min_segments", 2) or 2)):
        return None
    collected: list[TranslationResult] = []
    for cohesion_group in plan_cohesion_groups(
        fresh,
        min_segments=int(getattr(job.recipe, "cohesion_min_segments", 2) or 2),
        max_segments=int(getattr(job.recipe, "cohesion_max_segments", 8) or 8),
        max_group_chars=int(getattr(job.recipe, "cohesion_max_group_chars", 6_000) or 6_000),
    ):
        if engine.store.get_job(job.id).status is not JobStatus.RUNNING:
            break
        try:
            usage, _applied, _skipped = await _run_cohesion_group(
                engine, job, project, document, list(cohesion_group.segments),
                shared_context=shared_context,
                approved_terms=approved_terms,
                limiter=limiter,
            )
        except _SegmentChanged as stopped:
            collected.extend(stopped.results)
            break
        except (EmptyProviderResponseError, TranslationOutputError) as exc:
            collected.extend(list(exc.results) if hasattr(exc, "results") else [])
            continue
        except (PlaceholderIntegrityError, RuntimeError, ValueError) as exc:
            engine.store.record_provider_failure(
                cohesion_group.segments[0].id,
                {
                    "kind": "cohesion_group_failure",
                    "job_stage": CandidateStage.REVIEW.value,
                    "provider": job.recipe.draft.provider,
                    "model": job.recipe.draft.model,
                    "segment_ordinal": cohesion_group.segments[0].ordinal,
                    "cohesion_group": list(cohesion_group.ordinals),
                    "message": str(exc)[:1_000],
                    "job_id": job.id,
                },
            )
            continue
        collected.append(usage)
    if not collected:
        return None
    return _sum_results(collected, "")


async def _run_cohesion_group(
    engine,
    job: Job,
    project,
    document,
    segments: list[Segment],
    *,
    shared_context: str,
    approved_terms: list[TermEntry],
    limiter: _AdaptiveConcurrencyLimiter,
) -> tuple[TranslationResult, int, int]:
    """Run one group cohesion pass; bad segments fall back to their drafts."""
    provider = engine.providers.get(job.recipe.draft.provider)
    ordered = sorted(segments, key=lambda item: item.ordinal)
    expected_ids = {segment.id for segment in ordered}
    ordered_ids = [segment.id for segment in ordered]
    sources: dict[str, str] = {}
    drafts: dict[str, str] = {}
    protected_by_id: dict[str, ProtectedText] = {}
    terms_by_id: dict[str, list[TermEntry]] = {}
    structured_by_id: dict[str, bool] = {}
    approved_ids = {term.id for term in approved_terms}
    for segment in ordered:
        current = engine.store.get_segment(segment.id)
        visible = _visible_translation(current)
        if not visible:
            raise ValueError(f"Cohesion group lost its draft: {segment.id}")
        structured_source = engine.store.epub_translation_source(segment.id)
        protected = engine.protected_text_codec.encode(structured_source or current.source_text)
        if structured_source:
            protected = protected.compact_single_atom()
        sources[segment.id] = current.source_text
        drafts[segment.id] = visible
        protected_by_id[segment.id] = protected
        structured_by_id[segment.id] = bool(structured_source)
        terms_by_id[segment.id] = [term for term in approved_terms if term.id in approved_ids]
    messages = build_cohesion_messages(
        context=shared_context, sources=sources, drafts=drafts, ordered_ids=ordered_ids,
    )
    prompt_chars = sum(len(item["content"]) for item in messages)
    group_chars = sum(len(sources[item]) + len(drafts[item]) for item in ordered_ids)
    budget = min(
        max(512, job.recipe.cohesion_max_output_tokens),
        max(512, group_chars * 2),
        job.recipe.max_output_tokens,
    )

    async def operation() -> TranslationResult:
        call_started = time.monotonic()
        result = await provider.complete(
            messages,
            job.recipe.draft,
            thinking=False,
            reasoning_effort="none",
            compute_mode="economy",
            max_tokens=budget,
        )
        engine.store.record_model_call(job.id, {
            "segment_id": ordered_ids[0],
            "stage": CandidateStage.REVIEW.value,
            "attempt": 1,
            "provider": job.recipe.draft.provider,
            "model": job.recipe.draft.model,
            "compute_mode": "economy",
            "max_tokens": budget,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "reasoning_tokens": result.reasoning_tokens,
            "cache_hit_tokens": result.prompt_cache_hit_tokens,
            "cache_miss_tokens": result.prompt_cache_miss_tokens,
            "cost_usd": result.cost_usd,
            "source_chars": group_chars,
            "context_chars": len(shared_context),
            "prompt_chars": prompt_chars,
            "output_chars": len(result.text or ""),
            "finish_reason": response_stop_reason(result),
            "cohesion_group": [segment.ordinal for segment in ordered],
            "outcome": "complete" if (result.text or "").strip() else "upstream_empty_response",
            "elapsed_seconds": round(time.monotonic() - call_started, 3),
        })
        return result

    result = await limiter.complete(operation)
    usage_results = [result]
    try:
        revisions = parse_cohesion_revisions(result.text or "", expected_ids)
    except (ValueError, TypeError):
        engine.store.record_model_call(job.id, {
            "segment_id": ordered_ids[0],
            "stage": CandidateStage.REVIEW.value,
            "attempt": 1,
            "provider": job.recipe.draft.provider,
            "model": job.recipe.draft.model,
            "compute_mode": "economy",
            "max_tokens": budget,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
            "cache_hit_tokens": 0,
            "cache_miss_tokens": 0,
            "cost_usd": 0.0,
            "source_chars": group_chars,
            "context_chars": len(shared_context),
            "prompt_chars": prompt_chars,
            "output_chars": len(result.text or ""),
            "finish_reason": "",
            "cohesion_group": [segment.ordinal for segment in ordered],
            "outcome": "cohesion_invalid_json",
            "elapsed_seconds": 0.0,
        })
        return _sum_results(usage_results, ""), 0, len(ordered)
    applied = 0
    skipped = 0
    for segment in ordered:
        revised = revisions[segment.id]
        protected = protected_by_id[segment.id]
        current = engine.store.get_segment(segment.id)
        # Human work or source edits during the pass win; keep the draft.
        if (
            current.source_text != segment.source_text
            or current.status is SegmentStatus.HUMAN_CONFIRMED
            or (current.accepted_translation or current.edited_translation)
        ):
            skipped += 1
            continue
        try:
            restored_value = protected.restore(revised)
        except (PlaceholderIntegrityError, ValueError):
            skipped += 1
            continue
        if revised.strip() == drafts[segment.id].strip():
            applied += 0
            skipped += 1
            continue
        structured_value: str | None = None
        if structured_by_id[segment.id]:
            try:
                structured_value = engine.store.capture_epub_translation(
                    segment.id, restored_value, CandidateStage.REVIEW.value, persist=False,
                )
            except ValueError:
                skipped += 1
                continue
        issues = run_deterministic_checks(
            current.source_text, restored_value, terms_by_id[segment.id],
            segment_kind=current.kind,
            source_lang=project.source_lang,
            target_lang=project.target_lang,
        )
        if any(issue.code == "approved_term_missing" for issue in issues):
            skipped += 1
            continue
        engine.store.record_candidate(
            job_id=job.id, segment_id=segment.id, stage=CandidateStage.REVIEW,
            provider=job.recipe.draft.provider, model=job.recipe.draft.model,
            result=TranslationResult(text=restored_value),
        )
        engine.store.set_reviewed_translation(segment.id, restored_value)
        if structured_value is not None:
            engine.store.capture_epub_translation(
                segment.id, restored_value, CandidateStage.REVIEW.value,
            )
        engine.store.replace_issues(
            job.id, segment.id, issues, target_text=restored_value,
            detector_version=DETECTOR_VERSION,
        )
        applied += 1
    return _sum_results(usage_results, ""), applied, skipped


async def run_optimized(engine, job_id: str, *, max_batches: int | None = None) -> Job:
    job = engine.store.get_job(job_id)
    if job.status is JobStatus.COMPLETED:
        return job
    run_started_tokens = int(engine.store.job_progress(job.id)["total_tokens"])
    project = engine.store.get_project_for_document(job.document_id)
    document = engine.store.get_document(job.document_id)
    all_segments = engine.store.list_segments(job.document_id)
    # A stable reference snapshot includes neighbors outside the selected range so
    # concurrent batch results cannot change another request's context.
    segments_by_ordinal = {segment.ordinal: segment for segment in all_segments}
    selected_ranges = job.recipe.segment_ranges
    pending = [
        segment
        for segment in all_segments
        if segment.ordinal >= job.next_ordinal
        and (
            not selected_ranges
            or any(start <= segment.ordinal <= end for start, end in selected_ranges)
        )
    ]
    approved_terms = [
        term for term in engine.store.list_terms(project.id) if term.status.value == "approved"
    ]
    eligible = [
        segment
        for segment in pending
        if segment.status is not SegmentStatus.HUMAN_CONFIRMED
        and not _visible_translation(segment)
    ]
    groups = _groups(eligible, job.recipe.batch_size, job.recipe.max_batch_chars)
    shared_context = project_context(project, job.recipe.max_context_chars)
    limiter = _AdaptiveConcurrencyLimiter(
        job.recipe.concurrency,
        job.recipe.max_concurrency,
    )
    job = engine.store.save_job(replace(job, status=JobStatus.RUNNING, last_error=None))
    completed_batches = 0
    next_group_index = 0
    checkpoint_group_index = 0
    resolved_groups: set[int] = set()
    pending_tasks: dict[asyncio.Task, tuple[int, list[Segment]]] = {}

    async def finish_pending() -> None:
        tasks = list(pending_tasks)
        if tasks:
            outcomes = await asyncio.gather(*tasks, return_exceptions=True)
            for task, outcome in zip(tasks, outcomes, strict=True):
                if isinstance(outcome, BaseException):
                    continue
                _, group = pending_tasks[task]
                stage, _, translations, _, usage, elapsed, _, _ = outcome
                engine.store.record_batch(
                    job_id=job.id, stage=stage, start_ordinal=group[0].ordinal,
                    end_ordinal=group[-1].ordinal, segment_count=len(translations),
                    result=usage, elapsed_seconds=elapsed,
                )
                current = engine.store.get_job(job.id)
                engine.store.save_job(replace(
                    current, total_cost_usd=current.total_cost_usd + usage.cost_usd,
                ))
        pending_tasks.clear()

    def schedule_group(group_index: int) -> None:
        group = groups[group_index]
        task = asyncio.create_task(
            _translate_group(
                engine,
                job,
                project,
                document,
                group,
                project_context=shared_context,
                approved_terms=approved_terms,
                segments_by_ordinal=segments_by_ordinal,
                limiter=limiter,
            ),
            name=f"jieyi-{job.id}-group-{group_index}",
        )
        pending_tasks[task] = (group_index, group)

    try:
        while next_group_index < len(groups) or pending_tasks:
            current = engine.store.get_job(job.id)
            stopping = current.status in {JobStatus.PAUSED, JobStatus.CANCELLED}
            if stopping and not pending_tasks:
                return current
            progress = engine.store.job_progress(job.id)
            run_tokens = max(0, int(progress["total_tokens"]) - run_started_tokens)
            if not stopping and run_tokens >= job.recipe.token_budget:
                engine.store.save_job(
                    replace(
                        current,
                        status=JobStatus.PAUSED,
                        last_error=(
                            f"Run token budget reached: {run_tokens:,} / "
                            f"{job.recipe.token_budget:,} "
                            f"(cumulative: {progress['total_tokens']:,}). "
                            "Continue to open a fresh budget window."
                        ),
                    )
                )
                continue
            if not stopping and max_batches is not None and completed_batches >= max_batches:
                engine.store.save_job(replace(current, status=JobStatus.PAUSED))
                continue

            group_window = job.recipe.max_concurrency
            while not stopping and next_group_index < len(groups) and len(pending_tasks) < group_window:
                if (
                    max_batches is not None
                    and completed_batches + len(pending_tasks) >= max_batches
                ):
                    break
                schedule_group(next_group_index)
                next_group_index += 1

            if not pending_tasks:
                continue

            done, _ = await asyncio.wait(
                pending_tasks,
                return_when=asyncio.FIRST_COMPLETED,
            )
            fatal_failures: list[tuple[int, BaseException]] = []
            for task in sorted(done, key=lambda item: pending_tasks[item][0]):
                group_index, group = pending_tasks.pop(task)
                task_failure = task.exception()
                if task_failure is not None:
                    fatal_failures.append((group[0].ordinal, task_failure))
                    continue
                result = task.result()
                (
                    stage,
                    model_spec,
                    translations,
                    _candidate_stages,
                    usage,
                    elapsed,
                    terms_by_id,
                    segment_failures,
                ) = result
                engine.store.record_batch(
                    job_id=job.id,
                    stage=stage,
                    start_ordinal=group[0].ordinal,
                    end_ordinal=group[-1].ordinal,
                    segment_count=len(translations),
                    result=usage,
                    elapsed_seconds=elapsed,
                )
                completed_batches += 1
                segments_by_id = {segment.id: segment for segment in group}
                blocking_segment_failures: list[tuple[int, BaseException]] = []
                for segment_id, segment_error in segment_failures.items():
                    failed_segment = segments_by_id[segment_id]
                    if _can_defer_segment_failure(segment_error):
                        _record_deferred_segment(
                            engine,
                            job,
                            failed_segment,
                            stage=stage,
                            model_spec=model_spec,
                            error=segment_error,
                            terms=terms_by_id[segment_id],
                        )
                        continue
                    if isinstance(segment_error, EmptyProviderResponseError):
                        payload = segment_error.audit_payload()
                    else:
                        payload = {
                            "kind": "segment_processing_failure",
                            "job_stage": stage.value,
                            "provider": model_spec.provider,
                            "model": model_spec.model,
                            "segment_ordinal": failed_segment.ordinal,
                            "message": str(segment_error),
                        }
                    payload["job_id"] = job.id
                    engine.store.record_provider_failure(segment_id, payload)
                    blocking_segment_failures.append(
                        (failed_segment.ordinal, segment_error)
                    )
                if blocking_segment_failures:
                    fatal_failures.extend(blocking_segment_failures)
                elif all(
                    _visible_translation(engine.store.get_segment(segment.id))
                    or segment.id in segment_failures
                    for segment in group
                ):
                    resolved_groups.add(group_index)

                checkpoint = engine.store.get_job(job.id).next_ordinal
                while checkpoint_group_index in resolved_groups:
                    checkpoint = max(
                        checkpoint,
                        groups[checkpoint_group_index][-1].ordinal + 1,
                    )
                    checkpoint_group_index += 1
                current = engine.store.get_job(job.id)
                job = engine.store.save_job(
                    replace(
                        current,
                        next_ordinal=checkpoint,
                        total_cost_usd=current.total_cost_usd + usage.cost_usd,
                    )
                )

            if fatal_failures:
                _, failure = min(fatal_failures, key=lambda item: item[0])
                raise failure

        current = engine.store.get_job(job.id)
        if current.status in {JobStatus.PAUSED, JobStatus.CANCELLED}:
            return current
        if checkpoint_group_index < len(groups):
            return engine.store.save_job(replace(
                current, status=JobStatus.PAUSED,
                last_error="执行期间原文或译文发生变化，请检查后继续未完成段落。",
            ))
        return engine.store.save_job(
            replace(current, status=JobStatus.COMPLETED, next_ordinal=len(all_segments))
        )
    except Exception as exc:
        current = engine.store.get_job(job.id)
        status = current.status if current.status in {
            JobStatus.PAUSED, JobStatus.CANCELLED,
        } else JobStatus.FAILED
        engine.store.save_job(replace(current, status=status, last_error=str(exc)))
        raise
    finally:
        await finish_pending()
