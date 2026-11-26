from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from app.models import IngestionErrorRecord
from app.services.indexing.indexing_service import IndexingReport
from app.services.indexing.job_manager import IndexJob, JobOperation, JobState
from app.services.retrieval.ranking import SearchMode
from app.services.retrieval.retrieval_service import (
    MAX_QUERY_LENGTH,
    MAX_TOP_K,
    SearchResponse,
    SearchResult,
)


class DirectoryRequest(BaseModel):
    directory: str = Field(min_length=1, description="Absolute local folder path")


class ErrorRecordModel(BaseModel):
    path: str
    error_type: str
    message: str

    @classmethod
    def from_record(cls, record: IngestionErrorRecord) -> ErrorRecordModel:
        return cls(
            path=str(record.path),
            error_type=record.error_type,
            message=record.message,
        )


class IndexingReportModel(BaseModel):
    root: str
    scanned: int
    indexed: int
    unchanged: int
    duplicates: int
    removed: int
    skipped_unsupported: int
    vector_count: int
    errors: list[ErrorRecordModel]

    @classmethod
    def from_report(cls, report: IndexingReport) -> IndexingReportModel:
        return cls(
            root=str(report.root),
            scanned=report.scanned,
            indexed=len(report.indexed),
            unchanged=len(report.unchanged),
            duplicates=len(report.duplicates),
            removed=len(report.removed),
            skipped_unsupported=report.skipped_unsupported,
            vector_count=report.vector_count,
            errors=[ErrorRecordModel.from_record(error) for error in report.errors],
        )


class JobModel(BaseModel):
    job_id: str
    operation: JobOperation
    directory: str
    state: JobState
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    processed: int
    total: int
    current_path: str | None
    report: IndexingReportModel | None
    removed_count: int
    error: str | None

    @classmethod
    def from_job(cls, job: IndexJob) -> JobModel:
        return cls(
            job_id=job.job_id,
            operation=job.operation,
            directory=str(job.directory),
            state=job.state,
            created_at=job.created_at,
            started_at=job.started_at,
            finished_at=job.finished_at,
            processed=job.processed,
            total=job.total,
            current_path=str(job.current_path) if job.current_path else None,
            report=IndexingReportModel.from_report(job.report) if job.report else None,
            removed_count=len(job.removed_paths),
            error=job.error,
        )


class IndexStatusModel(BaseModel):
    roots: list[str]
    file_count: int
    document_count: int
    text_file_count: int
    image_file_count: int
    text_vector_count: int
    image_vector_count: int
    active_job: JobModel | None


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=MAX_QUERY_LENGTH)
    top_k: int = Field(default=10, ge=1, le=MAX_TOP_K)
    modes: list[SearchMode] | None = Field(
        default=None,
        description="Channels to use; all channels when omitted",
    )
    extensions: list[str] | None = Field(
        default=None,
        description="File extensions such as 'pdf' or '.png'",
    )
    path_prefix: str | None = None
    modified_after: datetime | None = None
    modified_before: datetime | None = None


class ChannelScoreModel(BaseModel):
    raw: float
    normalized: float
    weight: float
    contribution: float
    rank: int


class SearchResultModel(BaseModel):
    document_id: str
    kind: str
    file_name: str
    extension: str
    paths: list[str]
    score: float
    channel_scores: dict[str, ChannelScoreModel]
    snippet: str | None
    page_number: int | None
    chunk_id: str | None

    @classmethod
    def from_result(cls, result: SearchResult) -> SearchResultModel:
        return cls(
            document_id=result.document_id,
            kind=result.kind.value,
            file_name=result.file_name,
            extension=result.extension,
            paths=[str(path) for path in result.paths],
            score=round(result.score, 6),
            channel_scores={
                mode.value: ChannelScoreModel(
                    raw=score.raw,
                    normalized=score.normalized,
                    weight=score.weight,
                    contribution=score.contribution,
                    rank=score.rank,
                )
                for mode, score in result.channel_scores.items()
            },
            snippet=result.snippet,
            page_number=result.page_number,
            chunk_id=result.chunk_id,
        )


class SearchResponseModel(BaseModel):
    query: str
    took_ms: float
    results: list[SearchResultModel]
    warnings: list[str]

    @classmethod
    def from_response(cls, response: SearchResponse) -> SearchResponseModel:
        return cls(
            query=response.query,
            took_ms=round(response.took_ms, 3),
            results=[SearchResultModel.from_result(item) for item in response.results],
            warnings=response.warnings,
        )


class ReadinessModel(BaseModel):
    status: str
    components: dict[str, str]
