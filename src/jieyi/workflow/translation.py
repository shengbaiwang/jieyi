from __future__ import annotations

from dataclasses import asdict, replace

from jieyi.context.compiler import ContextCompiler
from jieyi.domain.models import (
    CandidateStage,
    Document,
    IssueSeverity,
    Job,
    JobStatus,
    ModelSpec,
    Project,
    Segment,
    SegmentStatus,
    TranslationRequest,
    TranslationResult,
)
from jieyi.prompting import build_messages
from jieyi.protection import PlaceholderIntegrityError, ProtectedText, ProtectedTextCodec
from jieyi.providers.registry import ProviderRegistry
from jieyi.quality.checks import (
    DETECTOR_VERSION,
    reviewer_attention_issues,
    run_deterministic_checks,
)
from jieyi.workflow.provider_responses import (
    EmptyProviderResponseError,
    content_filter_audit_payload,
    deferred_content_filter_message,
    inspect_empty_result,
    is_content_filtered_error,
    parse_review_response,
)

_PLACEHOLDER_REPAIR_ATTEMPTS = 3


class TranslationEngine:
    """Durable segment workflow. Every completed segment is its own checkpoint."""

    def __init__(self, store, providers: ProviderRegistry):
        self.store = store
        self.providers = providers
        self.context_compiler = ContextCompiler(store)
        self.protected_text_codec = ProtectedTextCodec()

    async def run(self, job_id: str, *, max_segments: int | None = None) -> Job:
        job = self.store.get_job(job_id)
        if job.status is JobStatus.COMPLETED:
            return job

        project = self.store.get_project_for_document(job.document_id)
        document = self.store.get_document(job.document_id)
        segments = self.store.list_segments(job.document_id)
        job = self.store.save_job(replace(job, status=JobStatus.RUNNING, last_error=None))
        processed = 0

        try:
            for segment in segments:
                if segment.ordinal < job.next_ordinal:
                    continue
                if job.recipe.segment_ranges and not any(
                    start <= segment.ordinal <= end for start, end in job.recipe.segment_ranges
                ):
                    continue
                if max_segments is not None and processed >= max_segments:
                    return self.store.save_job(replace(job, status=JobStatus.PAUSED))

                if segment.status is SegmentStatus.HUMAN_CONFIRMED:
                    job = self.store.save_job(replace(job, next_ordinal=segment.ordinal + 1))
                    continue

                context, terms = self.context_compiler.compile(
                    project,
                    segment,
                    neighbor_radius=job.recipe.neighbor_radius,
                    max_chars=job.recipe.max_context_chars,
                    tm_enabled=job.recipe.tm_enabled,
                    tm_threshold=job.recipe.tm_threshold,
                    tm_max_results=job.recipe.tm_max_results,
                )
                structured_source = self.store.epub_translation_source(segment.id)
                structured = bool(structured_source)
                protected = self.protected_text_codec.encode(
                    structured_source or segment.source_text
                )

                if job.recipe.task_mode == "review":
                    existing = (
                        segment.reviewed_translation
                        or segment.edited_translation
                        or segment.machine_translation
                        or segment.accepted_translation
                    )
                    deferred_message = deferred_content_filter_message(self.store, segment.id)
                    if not existing and not deferred_message:
                        job = self.store.save_job(replace(job, next_ordinal=segment.ordinal + 1))
                        continue
                    if (
                        structured
                        and existing
                        and not self.store.epub_structured_translation(segment.id)
                    ):
                        structured = False
                        protected = self.protected_text_codec.encode(segment.source_text)
                    review_spec = job.recipe.reviewer
                    if review_spec is None:
                        raise ValueError("A reviewer model is required for review jobs")
                    first_issues = run_deterministic_checks(
                        segment.source_text,
                        existing or "",
                        terms,
                        segment_kind=segment.kind,
                    )
                    reviewed, review_cost = await self._translate_protected(
                        job=job,
                        project=project,
                        document=document,
                        segment=segment,
                        protected=protected,
                        structured=structured,
                        context=context,
                        stage=CandidateStage.REVIEW,
                        model_spec=review_spec,
                        existing_translation=existing,
                        issue_summary="\n".join(
                            filter(
                                None,
                                [
                                    deferred_message,
                                    *(issue.message for issue in first_issues),
                                ],
                            )
                        ),
                    )
                    final_issues = run_deterministic_checks(
                        segment.source_text,
                        reviewed.text,
                        terms,
                        segment_kind=segment.kind,
                    )
                    final_issues.extend(
                        reviewer_attention_issues(reviewed.review_findings)
                    )
                    self.store.replace_issues(
                        job.id,
                        segment.id,
                        final_issues,
                        target_text=reviewed.text,
                        detector_version=DETECTOR_VERSION,
                    )
                    self.store.set_reviewed_translation(segment.id, reviewed.text)
                    job = self.store.save_job(
                        replace(
                            job,
                            next_ordinal=segment.ordinal + 1,
                            total_cost_usd=job.total_cost_usd + review_cost,
                        )
                    )
                    processed += 1
                    continue

                draft_spec = job.recipe.draft
                try:
                    draft, draft_cost = await self._translate_protected(
                        job=job,
                        project=project,
                        document=document,
                        segment=segment,
                        protected=protected,
                        structured=structured,
                        context=context,
                        stage=CandidateStage.DRAFT,
                        model_spec=draft_spec,
                    )
                except Exception as exc:
                    if not is_content_filtered_error(exc):
                        raise
                    if not isinstance(exc, EmptyProviderResponseError):
                        payload = content_filter_audit_payload(
                            exc,
                            stage=CandidateStage.DRAFT.value,
                            provider=draft_spec.provider,
                            model=draft_spec.model,
                            segment_ordinal=segment.ordinal,
                        )
                        payload["job_id"] = job.id
                        self.store.record_provider_failure(segment.id, payload)
                    job = self.store.save_job(replace(job, next_ordinal=segment.ordinal + 1))
                    processed += 1
                    continue

                final = draft
                segment_cost = draft_cost
                first_issues = run_deterministic_checks(
                    segment.source_text,
                    draft.text,
                    terms,
                    segment_kind=segment.kind,
                )
                should_review = (
                    job.recipe.reviewer is not None
                    and job.recipe.review_policy != "never"
                    and (
                        job.recipe.review_policy == "all"
                        or any(issue.severity is IssueSeverity.ERROR for issue in first_issues)
                    )
                )
                if should_review:
                    review_spec = job.recipe.reviewer
                    assert review_spec is not None
                    reviewed, review_cost = await self._translate_protected(
                        job=job,
                        project=project,
                        document=document,
                        segment=segment,
                        protected=protected,
                        structured=structured,
                        context=context,
                        stage=CandidateStage.REVIEW,
                        model_spec=review_spec,
                        existing_translation=draft.text,
                        issue_summary="\n".join(issue.message for issue in first_issues),
                    )
                    if reviewed.text.strip():
                        final = reviewed
                        segment_cost += review_cost

                final_issues = run_deterministic_checks(
                    segment.source_text,
                    final.text,
                    terms,
                    segment_kind=segment.kind,
                )
                final_issues.extend(
                    reviewer_attention_issues(final.review_findings)
                )
                self.store.replace_issues(
                    job.id,
                    segment.id,
                    final_issues,
                    target_text=final.text,
                    detector_version=DETECTOR_VERSION,
                )
                self.store.set_machine_translation(segment.id, final.text)

                job = self.store.save_job(
                    replace(
                        job,
                        next_ordinal=segment.ordinal + 1,
                        total_cost_usd=job.total_cost_usd + segment_cost,
                    )
                )
                processed += 1

        except Exception as exc:
            failed = replace(job, status=JobStatus.FAILED, last_error=str(exc))
            self.store.save_job(failed)
            raise

        return self.store.save_job(
            replace(job, status=JobStatus.COMPLETED, next_ordinal=len(segments))
        )

    async def run_optimized(self, job_id: str, *, max_batches: int | None = None) -> Job:
        from jieyi.workflow.optimized import run_optimized

        return await run_optimized(self, job_id, max_batches=max_batches)

    def preview(self, job_id: str, segment_id: str) -> dict:
        """Return the exact draft messages and protected spans without calling a model."""
        job = self.store.get_job(job_id)
        segment = self.store.get_segment(segment_id)
        if segment.document_id != job.document_id:
            raise ValueError("Segment does not belong to the job document")
        project = self.store.get_project_for_document(job.document_id)
        document = self.store.get_document(job.document_id)
        context, terms = self.context_compiler.compile(
            project,
            segment,
            neighbor_radius=job.recipe.neighbor_radius,
            max_chars=job.recipe.max_context_chars,
            tm_enabled=job.recipe.tm_enabled,
            tm_threshold=job.recipe.tm_threshold,
            tm_max_results=job.recipe.tm_max_results,
        )
        structured_source = self.store.epub_translation_source(segment.id)
        protected = self.protected_text_codec.encode(structured_source or segment.source_text)
        request = TranslationRequest(
            project=project,
            document=document,
            segment=replace(segment, source_text=protected.masked),
            context=context,
            task=CandidateStage.DRAFT,
        )
        return {
            "job_id": job.id,
            "segment_id": segment.id,
            "provider": job.recipe.draft.provider,
            "model": job.recipe.draft.model,
            "messages": build_messages(request),
            "protected_spans": [asdict(span) for span in protected.spans],
            "relevant_terms": [asdict(term) for term in terms],
        }

    def _raise_empty_result(
        self,
        *,
        job: Job,
        segment: Segment,
        stage: CandidateStage,
        model_spec: ModelSpec,
        result: TranslationResult,
    ) -> None:
        error = EmptyProviderResponseError(
            segment_id=segment.id,
            segment_ordinal=segment.ordinal,
            stage=stage.value,
            provider=model_spec.provider,
            model=model_spec.model,
            attempts=[inspect_empty_result(result, attempt=1, max_tokens=0)],
            results=[result],
        )
        payload = error.audit_payload()
        payload["job_id"] = job.id
        self.store.record_provider_failure(segment.id, payload)
        raise error

    async def _translate_protected(
        self,
        *,
        job: Job,
        project: Project,
        document: Document,
        segment: Segment,
        protected: ProtectedText,
        structured: bool,
        context: str,
        stage: CandidateStage,
        model_spec: ModelSpec,
        existing_translation: str | None = None,
        issue_summary: str = "",
    ) -> tuple[TranslationResult, float]:
        provider = self.providers.get(model_spec.provider)
        existing_for_prompt = (
            self.store.epub_structured_translation(segment.id)
            if structured and existing_translation
            else existing_translation
        )
        masked_existing = (
            protected.mask_translation(existing_for_prompt) if existing_for_prompt else None
        )
        request = TranslationRequest(
            project=project,
            document=document,
            segment=replace(segment, source_text=protected.masked),
            context=context,
            task=stage,
            existing_translation=masked_existing,
            issue_summary=issue_summary,
        )
        result = await provider.translate(request, model_spec)
        if not result.text.strip():
            self._raise_empty_result(
                job=job,
                segment=segment,
                stage=stage,
                model_spec=model_spec,
                result=result,
            )
        if stage is CandidateStage.REVIEW:
            raw_review_text = result.text
            reviewed_text, findings = parse_review_response(raw_review_text)
            result = replace(
                result,
                text=reviewed_text,
                raw_response=result.raw_response or raw_review_text,
                review_findings=findings,
            )
            if not result.text.strip():
                self._raise_empty_result(
                    job=job,
                    segment=segment,
                    stage=stage,
                    model_spec=model_spec,
                    result=result,
                )

        try:
            restored_value = protected.restore(result.text)
            restored = replace(
                result,
                text=(
                    self.store.capture_epub_translation(segment.id, restored_value, stage.value)
                    if structured
                    else restored_value
                ),
            )
        except (PlaceholderIntegrityError, ValueError) as initial_error:
            repair_error: BaseException = initial_error
            deterministic = protected.repair_surplus_placeholders(result.text)
            if deterministic is not None:
                try:
                    restored_value = protected.restore(deterministic)
                    restored = replace(
                        result,
                        text=(
                            self.store.capture_epub_translation(
                                segment.id, restored_value, stage.value
                            )
                            if structured
                            else restored_value
                        ),
                    )
                except (PlaceholderIntegrityError, ValueError) as deterministic_error:
                    repair_error = deterministic_error
                else:
                    self.store.record_candidate(
                        job_id=job.id,
                        segment_id=segment.id,
                        stage=stage,
                        provider=model_spec.provider,
                        model=model_spec.model,
                        result=restored,
                    )
                    return restored, result.cost_usd

            self.store.record_candidate(
                job_id=job.id,
                segment_id=segment.id,
                stage=stage,
                provider=model_spec.provider,
                model=model_spec.model,
                result=result,
            )
            repair_cost = 0.0
            for repair_attempt in range(1, _PLACEHOLDER_REPAIR_ATTEMPTS + 1):
                repair_request = TranslationRequest(
                    project=project,
                    document=document,
                    segment=replace(segment, source_text=protected.masked),
                    context=context,
                    task=CandidateStage.REPAIR,
                    # Keep retry attempts independent. A rejected repair can contain fewer
                    # correct markers than the original draft and must not become the next base.
                    existing_translation=result.text,
                    issue_summary=(
                        f"Repair attempt {repair_attempt} of "
                        f"{_PLACEHOLDER_REPAIR_ATTEMPTS}. Previous validation error: "
                        f"{repair_error}"
                    ),
                )
                repaired = await provider.translate(repair_request, model_spec)
                repair_cost += repaired.cost_usd
                if not repaired.text.strip():
                    repair_error = EmptyProviderResponseError(
                        segment_id=segment.id,
                        segment_ordinal=segment.ordinal,
                        stage=CandidateStage.REPAIR.value,
                        provider=model_spec.provider,
                        model=model_spec.model,
                        attempts=[inspect_empty_result(repaired, attempt=1, max_tokens=0)],
                        results=[repaired],
                    )
                    continue
                try:
                    restored_value = protected.restore(repaired.text)
                    restored = replace(
                        repaired,
                        text=(
                            self.store.capture_epub_translation(
                                segment.id, restored_value, CandidateStage.REPAIR.value
                            )
                            if structured
                            else restored_value
                        ),
                        review_findings=result.review_findings,
                    )
                    break
                except (PlaceholderIntegrityError, ValueError) as repair_exc:
                    repair_error = repair_exc
            else:
                if isinstance(repair_error, EmptyProviderResponseError):
                    payload = repair_error.audit_payload()
                    payload["job_id"] = job.id
                    self.store.record_provider_failure(segment.id, payload)
                    raise repair_error
                raise PlaceholderIntegrityError(
                    "Placeholder repair failed after "
                    f"{_PLACEHOLDER_REPAIR_ATTEMPTS} attempts for segment "
                    f"{segment.ordinal + 1} ({segment.id}): {repair_error}",
                    missing=getattr(repair_error, "missing", ()),
                    extra=getattr(repair_error, "extra", ()),
                ) from repair_error
            self.store.record_candidate(
                job_id=job.id,
                segment_id=segment.id,
                stage=CandidateStage.REPAIR,
                provider=model_spec.provider,
                model=model_spec.model,
                result=restored,
            )
            return restored, result.cost_usd + repair_cost

        self.store.record_candidate(
            job_id=job.id,
            segment_id=segment.id,
            stage=stage,
            provider=model_spec.provider,
            model=model_spec.model,
            result=restored,
        )
        return restored, result.cost_usd
