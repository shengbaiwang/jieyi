from __future__ import annotations

import asyncio
import time
from dataclasses import replace

from jieyi.context.compiler import compile_neighbor_context
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
from jieyi.prompting import build_messages
from jieyi.protection import PlaceholderIntegrityError
from jieyi.quality.checks import (
    DETECTOR_VERSION,
    run_deterministic_checks,
)
from jieyi.terminology import matching_terms, render_terminology_constraints
from jieyi.workflow.provider_responses import (
    EmptyProviderResponseError,
    content_filter_audit_payload,
    inspect_empty_result,
    is_content_filtered_error,
    should_expand_output_budget,
)

_PLACEHOLDER_REPAIR_ATTEMPTS = 3


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


def _project_context(project, max_chars: int) -> str:
    lines = [
        "# PROJECT",
        f"Source language: {project.source_lang}",
        f"Target language: {project.target_lang}",
        f"Domain: {project.domain}",
        f"Quote policy: {project.quote_policy}",
    ]
    if project.style_guide.strip():
        lines.extend(["", "# STYLE GUIDE", project.style_guide.strip()])
    value = "\n".join(lines)
    if len(value) > max_chars:
        return value[: max(0, max_chars - 40)] + "\n[context truncated by budget]"
    return value


def _term_context(source: str, terms: list[TermEntry]) -> str:
    return render_terminology_constraints(source, terms)


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
) -> TranslationResult:
    """Retry only failures that diagnostics identify as recoverable."""
    results: list[TranslationResult] = []
    attempts = []
    budget = min(max_tokens, max_output_tokens)
    messages = build_messages(request)
    for attempt_number in range(1, 4):
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
        if text:
            return _sum_results(results, text)

        attempt = inspect_empty_result(
            result,
            attempt=attempt_number,
            max_tokens=budget,
        )
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


async def _translate_group(
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
    source_by_id: dict[str, str] = {}
    structured_by_id: dict[str, bool] = {}
    protected_by_id = {}
    for segment in segments:
        epub_source = engine.store.epub_translation_source(segment.id)
        use_structure = bool(epub_source)
        source = epub_source if use_structure else segment.source_text
        source_by_id[segment.id] = source
        structured_by_id[segment.id] = use_structure
        protected_by_id[segment.id] = engine.protected_text_codec.encode(source)
    requests: list[TranslationRequest] = []
    terms_by_id: dict[str, list[TermEntry]] = {}
    context_by_id: dict[str, str] = {}
    for segment in segments:
        protected = protected_by_id[segment.id]
        relevant = matching_terms(segment.source_text, approved_terms)
        terms_by_id[segment.id] = relevant
        term_context = _term_context(segment.source_text, relevant)
        radius = max(0, job.recipe.neighbor_radius)
        neighbors = [
            segments_by_ordinal[ordinal]
            for ordinal in range(max(0, segment.ordinal - radius), segment.ordinal + radius + 1)
            if ordinal != segment.ordinal and ordinal in segments_by_ordinal
        ]
        neighbor_context = compile_neighbor_context(
            segment,
            neighbors,
            max_chars=max(
                0, job.recipe.max_context_chars - len(project_context) - len(term_context) - 4
            ),
            include_translations=False,
        )
        context_by_id[segment.id] = "\n\n".join(filter(None, [term_context, neighbor_context]))
        requests.append(
            TranslationRequest(
                project=project,
                document=document,
                segment=replace(segment, source_text=protected.masked),
                atom_boundaries=protected.atom_boundaries if structured_by_id[segment.id] else (),
                context=project_context,
                segment_context=context_by_id[segment.id],
                task=stage,
                existing_translation=None,
                issue_summary="",
            )
        )
    started = time.monotonic()
    translations: dict[str, str] = {}
    usage_results: list[TranslationResult] = []
    thinking = job.recipe.draft_thinking
    reasoning_effort = job.recipe.draft_reasoning_effort
    compute_mode = job.recipe.draft_compute_mode

    async def translate_one(request: TranslationRequest) -> tuple[str, TranslationResult]:
        source_chars = len(request.segment.source_text) + len(request.existing_translation or "")
        async def operation():
            return await _complete_one(
                provider,
                request,
                model_spec,
                thinking=thinking,
                reasoning_effort=reasoning_effort,
                compute_mode=compute_mode,
                max_tokens=min(
                    job.recipe.max_output_tokens,
                    max(512, source_chars * 2),
                ),
                max_output_tokens=job.recipe.max_output_tokens,
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
        if isinstance(outcome, BaseException):
            failures[request.segment.id] = outcome
            if isinstance(outcome, EmptyProviderResponseError):
                usage_results.append(outcome.usage_result())
            continue
        segment_id, result = outcome
        translations[segment_id] = result.text
        usage_results.append(result)
    restored: dict[str, str] = {}
    candidate_stages: dict[str, CandidateStage] = {}
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
                    "draft",
                )
                if structured_by_id[segment.id]
                else restored_value
            )
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
                            "draft",
                        )
                        if structured_by_id[segment.id]
                        else restored_value
                    )
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
                        )
                    repaired = await limiter.complete(repair_operation)
                    usage_results.append(repaired)
                    restored_value = protected.restore(
                        protected.assemble_atom_repair(repaired.text)
                        if structured_by_id[segment.id] else repaired.text
                    )
                    restored[segment.id] = (
                        engine.store.capture_epub_translation(segment.id, restored_value, "repair")
                        if structured_by_id[segment.id]
                        else restored_value
                    )
                    candidate_stages[segment.id] = CandidateStage.REPAIR
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
    usage = _sum_results(usage_results, "")
    elapsed = time.monotonic() - started
    return (
        stage,
        model_spec,
        restored,
        candidate_stages,
        usage,
        elapsed,
        terms_by_id,
        failures,
    )


def _can_defer_segment_failure(error: BaseException) -> bool:
    """Quarantine exhausted segment responses; transport/configuration errors still stop."""
    if is_content_filtered_error(error):
        return True
    if isinstance(error, PlaceholderIntegrityError):
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
    issues = (
        run_deterministic_checks(
            segment.source_text,
            visible,
            terms,
            segment_kind=segment.kind,
        )
        if visible
        else []
    )
    if isinstance(error, PlaceholderIntegrityError):
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
    project_context = _project_context(project, job.recipe.max_context_chars)
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

    async def cancel_pending() -> None:
        tasks = list(pending_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
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
                project_context=project_context,
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
            if current.status in {JobStatus.PAUSED, JobStatus.CANCELLED}:
                await cancel_pending()
                return current
            progress = engine.store.job_progress(job.id)
            run_tokens = max(0, int(progress["total_tokens"]) - run_started_tokens)
            if run_tokens >= job.recipe.token_budget:
                await cancel_pending()
                return engine.store.save_job(
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
            if max_batches is not None and completed_batches >= max_batches:
                await cancel_pending()
                return engine.store.save_job(replace(current, status=JobStatus.PAUSED))

            group_window = job.recipe.max_concurrency
            while next_group_index < len(groups) and len(pending_tasks) < group_window:
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
                    candidate_stages,
                    usage,
                    elapsed,
                    terms_by_id,
                    segment_failures,
                ) = result
                for segment in group:
                    if segment.id not in translations:
                        continue
                    text = translations[segment.id]
                    engine.store.record_candidate(
                        job_id=job.id,
                        segment_id=segment.id,
                        stage=candidate_stages[segment.id],
                        provider=model_spec.provider,
                        model=model_spec.model,
                        result=TranslationResult(text=text),
                    )
                    issues = run_deterministic_checks(
                        segment.source_text,
                        text,
                        terms_by_id[segment.id],
                        segment_kind=segment.kind,
                    )
                    engine.store.replace_issues(
                        job.id,
                        segment.id,
                        issues,
                        target_text=text,
                        detector_version=DETECTOR_VERSION,
                    )
                    engine.store.set_machine_translation(segment.id, text)
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
                else:
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
                await cancel_pending()
                _, failure = min(fatal_failures, key=lambda item: item[0])
                raise failure

        return engine.store.save_job(
            replace(job, status=JobStatus.COMPLETED, next_ordinal=len(all_segments))
        )
    except Exception as exc:
        engine.store.save_job(
            replace(engine.store.get_job(job.id), status=JobStatus.FAILED, last_error=str(exc))
        )
        raise
