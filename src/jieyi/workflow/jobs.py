from __future__ import annotations

import asyncio

from jieyi.domain.models import JobStatus


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
            return self.store.job_progress(job_id)

        self.store.set_job_status(job_id, JobStatus.RUNNING)
        task = asyncio.create_task(self.engine.run_optimized(job_id), name=f"jieyi-{job_id}")
        self._tasks[job_id] = task

        def discard(completed: asyncio.Task) -> None:
            if self._tasks.get(job_id) is completed:
                self._tasks.pop(job_id, None)
            if not completed.cancelled():
                completed.exception()

        task.add_done_callback(discard)
        return self.store.job_progress(job_id)

    async def stop(self, job_id: str, status: JobStatus) -> dict:
        self.store.get_job(job_id)
        task = self._tasks.pop(job_id, None)
        if task and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self.store.set_job_status(job_id, status)
        return self.store.job_progress(job_id)

    async def shutdown(self) -> None:
        tasks = [task for task in self._tasks.values() if not task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()

