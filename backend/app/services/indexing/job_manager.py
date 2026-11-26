from __future__ import annotations

import logging
import threading
import uuid
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from app.services.indexing.indexing_service import (
    IndexingReport,
    IndexingService,
    IndexProgress,
)

logger = logging.getLogger(__name__)


class JobOperation(str, Enum):
    INDEX = "index"
    REMOVE = "remove"


class JobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class IndexJob:
    job_id: str
    operation: JobOperation
    directory: Path
    state: JobState
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    processed: int = 0
    total: int = 0
    current_path: Path | None = None
    report: IndexingReport | None = None
    removed_paths: tuple[Path, ...] = ()
    error: str | None = None


class IndexJobManager:
    """Runs indexing jobs one at a time on a background worker thread.

    A single worker serializes every write to Chroma, SQLite and the model
    interpreters, while search requests keep being served by the API.
    """

    def __init__(
        self,
        indexing_service: IndexingService,
        max_history: int = 50,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if max_history <= 0:
            raise ValueError("max_history must be greater than zero")

        self._service = indexing_service
        self._max_history = max_history
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="indexer",
        )
        self._jobs: OrderedDict[str, IndexJob] = OrderedDict()
        self._futures: dict[str, Future[None]] = {}
        self._lock = threading.Lock()

    def submit(
        self,
        directory: str | Path,
        operation: JobOperation = JobOperation.INDEX,
    ) -> IndexJob:
        job = IndexJob(
            job_id=uuid.uuid4().hex,
            operation=operation,
            directory=Path(directory).expanduser().resolve(),
            state=JobState.QUEUED,
            created_at=self._clock(),
        )

        with self._lock:
            self._jobs[job.job_id] = job
            self._trim_history()
            self._futures[job.job_id] = self._executor.submit(self._run, job.job_id)

        return job

    def get(self, job_id: str) -> IndexJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self) -> list[IndexJob]:
        with self._lock:
            return list(reversed(self._jobs.values()))

    def active_job(self) -> IndexJob | None:
        with self._lock:
            for job in reversed(self._jobs.values()):
                if job.state in {JobState.QUEUED, JobState.RUNNING}:
                    return job

        return None

    def wait(self, job_id: str, timeout: float | None = None) -> IndexJob | None:
        """Block until a job finishes. Intended for tests and scripts."""

        with self._lock:
            future = self._futures.get(job_id)

        if future is not None:
            future.result(timeout=timeout)

        return self.get(job_id)

    def shutdown(self, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=not wait)

    def _run(self, job_id: str) -> None:
        self._update(job_id, state=JobState.RUNNING, started_at=self._clock())
        job = self.get(job_id)
        assert job is not None

        try:
            if job.operation is JobOperation.INDEX:
                report = self._service.index_directory(
                    job.directory,
                    progress=lambda progress: self._on_progress(job_id, progress),
                )
                self._update(job_id, report=report)
            else:
                removed = self._service.remove_directory(job.directory)
                self._update(job_id, removed_paths=tuple(removed))
        except Exception as exc:
            logger.exception("Index job %s failed", job_id)
            self._update(
                job_id,
                state=JobState.FAILED,
                error=f"{type(exc).__name__}: {exc}",
                finished_at=self._clock(),
            )
            return

        self._update(
            job_id,
            state=JobState.SUCCEEDED,
            current_path=None,
            finished_at=self._clock(),
        )

    def _on_progress(self, job_id: str, progress: IndexProgress) -> None:
        self._update(
            job_id,
            processed=progress.processed,
            total=progress.total,
            current_path=progress.current_path,
        )

    def _update(self, job_id: str, **changes: object) -> None:
        with self._lock:
            job = self._jobs.get(job_id)

            if job is not None:
                self._jobs[job_id] = replace(job, **changes)  # type: ignore[arg-type]

    def _trim_history(self) -> None:
        while len(self._jobs) > self._max_history:
            oldest_id, oldest = next(iter(self._jobs.items()))

            if oldest.state in {JobState.QUEUED, JobState.RUNNING}:
                break

            del self._jobs[oldest_id]
            self._futures.pop(oldest_id, None)
