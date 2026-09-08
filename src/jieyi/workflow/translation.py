from __future__ import annotations

import asyncio
from dataclasses import asdict, replace

from jieyi.context.compiler import ContextCompiler
from jieyi.domain.models import (
    CandidateStage,
    Document,
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
    run_deterministic_checks,
)
from jieyi.workflow.provider_responses import (
    EmptyProviderResponseError,
    content_filter_audit_payload,
    inspect_empty_result,
    is_content_filtered_error,
)

_PLACEHOLDER_REPAIR_ATTEMPTS = 3


class _RunStopped(Exception):
    def __init__(self, cost: float):
        self.cost = cost


class TranslationEngine:
    """Durable segment workflow. Every completed segment is its own checkpoint."""

    def __init__(self, store, providers: ProviderRegistry):
        self.store = store
        self.providers = providers
        self.context_compiler = ContextCompiler(store)
        self.protected_text_codec = ProtectedTextCodec()

    def reserve(self, job_id: str):
        job = self.store.get_job(job_id)
        if job.status is JobStatus.CANCELLED:
            raise ValueError("Cancelled jobs cannot be resumed")
        return self.store.reserve_document(job.document_id)

    async def _execute(self, job_id, operation, reservation):
        # A cancelled coroutine cannot stop an issued blocking HTTP request.
        # Retain ownership until its response is saved. PAUSED stops dispatch.
        with reservation:
            stop_requested = False
            owner = asyncio.current_task()
            starting_status = self.store.get_job(job_id).status

            async def execute_operation():
                if (stop_requested or (owner is not None and owner.cancelling())
                    or self.store.get_job(job_id).status is not starting_status):
                    return self.store.get_job(job_id)
                return await operation()

            work = asyncio.create_task(execute_operation())
            try:
                return await asyncio.shield(work)
            except asyncio.CancelledError:
                stop_requested = True
                if self.store.get_job(job_id).status in {JobStatus.PENDING, JobStatus.RUNNING}:
                    self.store.set_job_status(job_id, JobStatus.PAUSED)
                while not work.done():
                    try:
                        await asyncio.shield(work)
                    except asyncio.CancelledError:
                        continue
                    except Exception:  # noqa: BLE001 -- worker recorded failure; preserve cancellation
                        break
                if not work.cancelled():
                    work.exception()
                raise

    async def run(self, job_id: str, *, max_segments: int | None = None) -> Job:
        return await self._execute(
            job_id, lambda: self._run(job_id, max_segments=max_segments), self.reserve(job_id),
        )

    async def _run(self, job_id: str, *, max_segments: int | None = None) -> Job:
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
                job = self.store.get_job(job_id)
                if job.status is not JobStatus.RUNNING:
                    return job
                if segment.ordinal < job.next_ordinal:
                    continue
                if job.recipe.segment_ranges and not any(
                    start <= segment.ordinal <= end for start, end in job.recipe.segment_ranges
                ):
                    continue
                if max_segments is not None and processed >= max_segments:
                    return self.store.save_job(replace(job, status=JobStatus.PAUSED))

                segment = self.store.get_segment(segment.id)
                if segment.status is SegmentStatus.HUMAN_CONFIRMED or any((
                    segment.accepted_translation, segment.reviewed_translation,
                    segment.edited_translation, segment.machine_translation,
                )):
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

                if structured:
                    protected = protected.compact_single_atom()

                draft_spec = job.recipe.draft
                try:
                    draft, draft_cost, structured_value = await self._translate_protected(
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
                except _RunStopped as stopped:
                    current = self.store.get_job(job_id)
                    return self.store.save_job(replace(
                        current, total_cost_usd=current.total_cost_usd + stopped.cost,
                    ))
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
                    job = self.store.save_job(replace(
                        self.store.get_job(job_id), next_ordinal=segment.ordinal + 1,
                    ))
                    processed += 1
                    continue

                final = draft
                segment_cost = draft_cost
                final_issues = run_deterministic_checks(
                    segment.source_text,
                    final.text,
                    terms,
                    segment_kind=segment.kind,
                )
                committed = self.store.commit_generated_translation(
                    job=job, segment=segment, text=final.text, issues=final_issues,
                    detector_version=DETECTOR_VERSION, stage=CandidateStage.DRAFT,
                    record_candidate=False, structured_value=structured_value,
                )
                current_segment = self.store.get_segment(segment.id)
                if not committed and not any((
                    current_segment.accepted_translation, current_segment.reviewed_translation,
                    current_segment.edited_translation, current_segment.machine_translation,
                )) and current_segment.status is not SegmentStatus.HUMAN_CONFIRMED:
                    current = self.store.get_job(job_id)
                    return self.store.save_job(replace(
                        current,
                        status=JobStatus.CANCELLED if current.status is JobStatus.CANCELLED else JobStatus.PAUSED,
                        total_cost_usd=current.total_cost_usd + segment_cost,
                        last_error="执行期间原文发生变化，请检查后继续本段。",
                    ))

                job = self.store.get_job(job_id)
                job = self.store.save_job(
                    replace(
                        job,
                        next_ordinal=segment.ordinal + 1,
                        total_cost_usd=job.total_cost_usd + segment_cost,
                    )
                )
                processed += 1

        except Exception as exc:
            current = self.store.get_job(job_id)
            status = current.status if current.status in {
                JobStatus.PAUSED, JobStatus.CANCELLED,
            } else JobStatus.FAILED
            failed = replace(current, status=status, last_error=str(exc))
            self.store.save_job(failed)
            raise

        job = self.store.get_job(job_id)
        if job.status is not JobStatus.RUNNING:
            return job
        return self.store.save_job(
            replace(job, status=JobStatus.COMPLETED, next_ordinal=len(segments))
        )

    async def run_optimized(self, job_id: str, *, max_batches: int | None = None,
                            _reservation=None) -> Job:
        from jieyi.workflow.optimized import run_optimized

        reservation = _reservation or self.reserve(job_id)
        if _reservation and self.store.get_job(job_id).status is not JobStatus.RUNNING:
            reservation.release()
            return self.store.get_job(job_id)
        return await self._execute(
            job_id, lambda: run_optimized(self, job_id, max_batches=max_batches), reservation,
        )

    def preview(self, job_id: str, segment_id: str, *, optimized: bool = False) -> dict:
        """Return the exact draft messages and protected spans without calling a model."""
        job = self.store.get_job(job_id)
        segment = self.store.get_segment(segment_id)
        if segment.document_id != job.document_id:
            raise ValueError("Segment does not belong to the job document")
        project = self.store.get_project_for_document(job.document_id)
        document = self.store.get_document(job.document_id)
        from jieyi.workflow.requests import initial_output_budget, prepare_translation

        if optimized:
            prepared = prepare_translation(
                self.store, self.protected_text_codec, project, document, segment, job.recipe,
                [term for term in self.store.list_terms(project.id) if term.status.value == "approved"],
                {item.ordinal: item for item in self.store.list_segments(document.id)},
            )
            request, protected, terms = prepared.request, prepared.protected, prepared.terms
        else:
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
            if structured_source:
                protected = protected.compact_single_atom()
            request = TranslationRequest(
                project=project,
                document=document,
                segment=replace(segment, source_text=protected.masked),
                atom_boundaries=protected.atom_boundaries if bool(structured_source) else (),
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
            "local_wrapper": protected.local_wrapper,
            "execution_mode": "optimized" if optimized else "standard",
            "max_output_tokens": initial_output_budget(
                request, job.recipe.draft_compute_mode, job.recipe.max_output_tokens,
            ) if optimized else None,
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
    ) -> tuple[TranslationResult, float, str | None]:
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
            atom_boundaries=protected.atom_boundaries if structured else (),
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
        try:
            restored_value = protected.restore(result.text)
            restored = replace(
                result,
                text=(
                    self.store.capture_epub_translation(segment.id, restored_value, stage.value, persist=False)
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
                                segment.id, restored_value, stage.value, persist=False,
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
                    return restored, result.cost_usd, restored_value if structured else None

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
                    atom_boundaries=protected.atom_boundaries if structured else (),
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
                if self.store.get_job(job.id).status is not JobStatus.RUNNING:
                    raise _RunStopped(result.cost_usd + repair_cost)
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
                    restored_value = protected.restore(
                        protected.assemble_atom_repair(repaired.text)
                        if structured else repaired.text
                    )
                    restored = replace(
                        repaired,
                        text=(
                            self.store.capture_epub_translation(
                                segment.id, restored_value, CandidateStage.REPAIR.value, persist=False,
                            )
                            if structured
                            else restored_value
                        ),
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
            return restored, result.cost_usd + repair_cost, restored_value if structured else None

        self.store.record_candidate(
            job_id=job.id,
            segment_id=segment.id,
            stage=stage,
            provider=model_spec.provider,
            model=model_spec.model,
            result=restored,
        )
        return restored, result.cost_usd, restored_value if structured else None
