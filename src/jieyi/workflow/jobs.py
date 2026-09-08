from __future__ import annotations

import asyncio

from jieyi.domain.models import JobStatus
from jieyi.persistence.execution import DocumentBusyError


class JobManager:
    """Own in-process job tasks while SQLite remains the durable source of truth."""

    def __init__(self, store, engine):
        self.store = store
        self.engine = engine
        self._tasks: dict[str, asyncio.Task] = {}

    def running(self, job_id: str) -> bool:
        task = self._tasks.get(job_id)
        return bool(task and not task.done())

    def start(self, job_id: str) -> dict:
        job = self.store.get_job(job_id)
        if job.status is JobStatus.COMPLETED:
            return self.store.job_progress(job_id)
        if job.status is JobStatus.CANCELLED:
            raise ValueError("Cancelled jobs cannot be resumed")
        if self.running(job_id):
            if job.status is not JobStatus.RUNNING:
                raise DocumentBusyError("已停止派发新请求，正在保存已发请求的结果，请稍后继续。")
            return self.store.job_progress(job_id)

        reservation = self.engine.reserve(job_id)
        try:
            self.store.set_job_status(job_id, JobStatus.RUNNING)
            task = asyncio.create_task(
                self.engine.run_optimized(job_id, _reservation=reservation), name=f"jieyi-{job_id}"
            )
        except BaseException:
            reservation.release()
            raise
        self._tasks[job_id] = task

        def discard(completed: asyncio.Task) -> None:
            reservation.release()  # Also handles cancellation before the coroutine starts.
            if self._tasks.get(job_id) is completed:
                self._tasks.pop(job_id, None)
            if not completed.cancelled():
                completed.exception()

        task.add_done_callback(discard)
        return self.store.job_progress(job_id)

    async def stop(self, job_id: str, status: JobStatus) -> dict:
        job = self.store.get_job(job_id)
        if job.status in {JobStatus.COMPLETED, JobStatus.CANCELLED}:
            return self.store.job_progress(job_id)
        task = self._tasks.get(job_id)
        if task and not task.done():
            # urllib/to_thread cancellation does not stop the remote request.
            # Stop dispatch and keep ownership while issued requests finish/save.
            self.store.set_job_status(job_id, status)
        else:
            with self.engine.reserve(job_id):
                self.store.set_job_status(job_id, status)
        return self.store.job_progress(job_id) | {"draining": bool(task and not task.done())}

    async def shutdown(self) -> None:
        tasks = [task for task in self._tasks.values() if not task.done()]
        for job_id, task in self._tasks.items():
            if not task.done() and self.store.get_job(job_id).status is JobStatus.RUNNING:
                self.store.set_job_status(job_id, JobStatus.PAUSED)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()

