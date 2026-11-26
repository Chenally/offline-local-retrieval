from __future__ import annotations

import threading
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, status

from app.api.schemas import (
    DirectoryRequest,
    IndexStatusModel,
    JobModel,
    ReadinessModel,
    SearchRequest,
    SearchResponseModel,
)
from app.config import AppSettings
from app.container import ServiceContainer, build_container
from app.services.embeddings.errors import EmbeddingConfigurationError
from app.services.embeddings.types import EmbeddingSpace
from app.services.indexing.job_manager import JobOperation
from app.services.retrieval.ranking import SearchMode
from app.services.retrieval.retrieval_service import InvalidQueryError, SearchQuery
from app.services.storage.errors import StorageError

ContainerFactory = Callable[[], ServiceContainer]


def _default_factory() -> ServiceContainer:
    return build_container(AppSettings.from_env())


def create_app(container_factory: ContainerFactory | None = None) -> FastAPI:
    """Build the FastAPI application.

    Services are created lazily on first use, so the liveness probe works
    even before model files are provisioned. Tests inject a factory that
    builds services with fake embedding backends.
    """

    factory = container_factory or _default_factory
    container_lock = threading.Lock()
    holder: dict[str, ServiceContainer] = {}

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield

        with container_lock:
            container = holder.pop("container", None)

        if container is not None:
            container.close()

    app = FastAPI(
        title="Offline Local Retrieval Service",
        version="0.4.0",
        lifespan=lifespan,
    )

    def get_container() -> ServiceContainer:
        with container_lock:
            if "container" not in holder:
                try:
                    holder["container"] = factory()
                except (EmbeddingConfigurationError, StorageError) as exc:
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail=f"Retrieval service is not ready: {exc}",
                    ) from exc

            return holder["container"]

    @app.get("/health/live")
    def liveness() -> dict[str, str]:
        return {
            "status": "ok",
        }

    @app.get("/health/ready", response_model=ReadinessModel)
    def readiness(
        container: ServiceContainer = Depends(get_container),
    ) -> ReadinessModel:
        components = dict(container.component_status)
        degraded = any(value != "ready" for value in components.values())
        return ReadinessModel(
            status="degraded" if degraded else "ready",
            components=components,
        )

    @app.post(
        "/index",
        response_model=JobModel,
        status_code=status.HTTP_202_ACCEPTED,
    )
    def start_indexing(
        request: DirectoryRequest,
        container: ServiceContainer = Depends(get_container),
    ) -> JobModel:
        directory = _existing_directory(request.directory)
        job = container.job_manager.submit(directory, JobOperation.INDEX)
        return JobModel.from_job(job)

    @app.post(
        "/index/remove",
        response_model=JobModel,
        status_code=status.HTTP_202_ACCEPTED,
    )
    def remove_folder(
        request: DirectoryRequest,
        container: ServiceContainer = Depends(get_container),
    ) -> JobModel:
        directory = Path(request.directory).expanduser().resolve()
        job = container.job_manager.submit(directory, JobOperation.REMOVE)
        return JobModel.from_job(job)

    @app.get("/index/jobs/{job_id}", response_model=JobModel)
    def get_job(
        job_id: str,
        container: ServiceContainer = Depends(get_container),
    ) -> JobModel:
        job = container.job_manager.get(job_id)

        if job is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Unknown job {job_id}",
            )

        return JobModel.from_job(job)

    @app.get("/index/status", response_model=IndexStatusModel)
    def index_status(
        container: ServiceContainer = Depends(get_container),
    ) -> IndexStatusModel:
        summary = container.manifest.summary()
        active = container.job_manager.active_job()

        return IndexStatusModel(
            roots=[str(root) for root in summary.roots],
            file_count=summary.file_count,
            document_count=summary.document_count,
            text_file_count=summary.text_file_count,
            image_file_count=summary.image_file_count,
            text_vector_count=container.vector_store.count(
                EmbeddingSpace.TEXT_SEMANTIC
            ),
            image_vector_count=container.vector_store.count(EmbeddingSpace.MULTIMODAL),
            active_job=JobModel.from_job(active) if active else None,
        )

    @app.post("/search", response_model=SearchResponseModel)
    def search(
        request: SearchRequest,
        container: ServiceContainer = Depends(get_container),
    ) -> SearchResponseModel:
        query = SearchQuery(
            text=request.query,
            top_k=request.top_k,
            modes=frozenset(request.modes) if request.modes else frozenset(SearchMode),
            extensions=frozenset(request.extensions) if request.extensions else None,
            path_prefix=Path(request.path_prefix) if request.path_prefix else None,
            modified_after=request.modified_after,
            modified_before=request.modified_before,
        )

        try:
            response = container.retrieval_service.search(query)
        except InvalidQueryError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
        except StorageError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc

        return SearchResponseModel.from_response(response)

    return app


def _existing_directory(value: str) -> Path:
    directory = Path(value).expanduser().resolve()

    if not directory.is_dir():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Directory does not exist: {directory}",
        )

    return directory


app = create_app()
