import threading
from pathlib import Path

from app.services.indexing.indexing_service import IndexingReport, IndexProgress
from app.services.indexing.job_manager import IndexJobManager, JobOperation, JobState


class FakeIndexingService:
    def __init__(self) -> None:
        self.release = threading.Event()
        self.release.set()
        self.fail = False

    def index_directory(self, directory, progress=None):
        self.release.wait(timeout=5)

        if self.fail:
            raise RuntimeError("boom")

        if progress is not None:
            progress(IndexProgress(1, 2, Path(directory) / "a.txt"))
            progress(IndexProgress(2, 2, None))

        return IndexingReport(root=Path(directory), scanned=2)

    def remove_directory(self, directory):
        return [Path(directory) / "a.txt"]


def test_index_job_succeeds_with_progress_and_report(tmp_path: Path) -> None:
    manager = IndexJobManager(FakeIndexingService())

    job = manager.submit(tmp_path)
    finished = manager.wait(job.job_id, timeout=5)

    assert finished is not None
    assert finished.state is JobState.SUCCEEDED
    assert finished.processed == 2
    assert finished.total == 2
    assert finished.report is not None and finished.report.scanned == 2
    assert finished.started_at is not None and finished.finished_at is not None
    manager.shutdown()


def test_remove_job_records_removed_paths(tmp_path: Path) -> None:
    manager = IndexJobManager(FakeIndexingService())

    job = manager.submit(tmp_path, JobOperation.REMOVE)
    finished = manager.wait(job.job_id, timeout=5)

    assert finished is not None
    assert finished.removed_paths == (tmp_path.resolve() / "a.txt",)
    manager.shutdown()


def test_failed_job_keeps_error_message(tmp_path: Path) -> None:
    service = FakeIndexingService()
    service.fail = True
    manager = IndexJobManager(service)

    job = manager.submit(tmp_path)
    finished = manager.wait(job.job_id, timeout=5)

    assert finished is not None
    assert finished.state is JobState.FAILED
    assert finished.error == "RuntimeError: boom"
    manager.shutdown()


def test_jobs_run_one_at_a_time_and_active_job_is_reported(tmp_path: Path) -> None:
    service = FakeIndexingService()
    service.release.clear()
    manager = IndexJobManager(service)

    first = manager.submit(tmp_path / "one")
    second = manager.submit(tmp_path / "two")

    assert manager.get(second.job_id).state is JobState.QUEUED
    assert manager.active_job() is not None

    service.release.set()
    manager.wait(first.job_id, timeout=5)
    manager.wait(second.job_id, timeout=5)

    assert manager.active_job() is None
    assert [job.job_id for job in manager.list_jobs()] == [second.job_id, first.job_id]
    manager.shutdown()


def test_history_is_trimmed(tmp_path: Path) -> None:
    manager = IndexJobManager(FakeIndexingService(), max_history=2)

    jobs = [manager.submit(tmp_path) for _ in range(3)]

    for job in jobs:
        manager.wait(job.job_id, timeout=5)

    extra = manager.submit(tmp_path)
    manager.wait(extra.job_id, timeout=5)

    assert len(manager.list_jobs()) <= 3
    assert manager.get(extra.job_id) is not None
    assert manager.get("missing") is None
    manager.shutdown()
