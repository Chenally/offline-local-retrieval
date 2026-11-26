from __future__ import annotations

from dataclasses import dataclass, field

from app.config import AppSettings
from app.services.chunking.text_chunker import TextChunker
from app.services.embeddings.base import EmbeddingBackend
from app.services.embeddings.errors import EmbeddingConfigurationError
from app.services.embeddings.service import EmbeddingService
from app.services.embeddings.synchronized import SynchronizedEmbeddingService
from app.services.embeddings.types import EmbeddingSpace
from app.services.indexing.indexing_service import IndexingService
from app.services.indexing.job_manager import IndexJobManager
from app.services.retrieval.ranking import HybridRanker
from app.services.retrieval.retrieval_service import RetrievalService
from app.services.storage.keyword_index import KeywordIndex
from app.services.storage.manifest import IndexManifest
from app.services.storage.vector_store import (
    ChromaVectorStore,
    CollectionSpec,
    VectorStore,
)

# Used when a backend does not report its own output dimension.
EMBEDDING_DIMENSION = 512


@dataclass
class ServiceContainer:
    """Wired application services shared by the API and scripts."""

    settings: AppSettings
    embedder: SynchronizedEmbeddingService
    vector_store: VectorStore
    keyword_index: KeywordIndex
    manifest: IndexManifest
    indexing_service: IndexingService
    retrieval_service: RetrievalService
    job_manager: IndexJobManager
    component_status: dict[str, str] = field(default_factory=dict)

    def close(self) -> None:
        self.job_manager.shutdown(wait=True)
        self.keyword_index.close()
        self.manifest.close()


def load_backends(
    settings: AppSettings,
) -> tuple[list[EmbeddingBackend], dict[str, str]]:
    """Load every LiteRT backend whose model files exist.

    Missing optional models disable one search channel instead of the
    whole service; the status map explains what is unavailable.
    """

    backends: list[EmbeddingBackend] = []
    status: dict[str, str] = {}

    try:
        from app.services.embeddings.sentence_backend import (
            SentenceEmbeddingBackend,
        )

        backends.append(
            SentenceEmbeddingBackend(
                model_path=settings.text_model_path,
                vocab_path=settings.text_vocab_path,
            )
        )
        status["text_semantic"] = "ready"
    except (EmbeddingConfigurationError, ImportError) as exc:
        status["text_semantic"] = f"unavailable: {exc}"

    try:
        from app.services.embeddings.mobileclip_backend import (
            MobileClipImageBackend,
            MobileClipTextBackend,
        )

        backends.append(
            MobileClipTextBackend(
                model_path=settings.mobileclip_text_model_path,
                tokenizer_path=settings.mobileclip_tokenizer_path,
            )
        )
        backends.append(
            MobileClipImageBackend(
                model_path=settings.mobileclip_image_model_path,
            )
        )
        status["mobileclip"] = "ready"
    except (EmbeddingConfigurationError, ImportError) as exc:
        backends = [
            backend
            for backend in backends
            if backend.space is not EmbeddingSpace.MULTIMODAL
        ]
        status["mobileclip"] = f"unavailable: {exc}"

    return backends, status


def build_container(
    settings: AppSettings,
    backends: list[EmbeddingBackend] | None = None,
    vector_store: VectorStore | None = None,
    embedding_dimension: int | None = None,
) -> ServiceContainer:
    status: dict[str, str] = {}

    if backends is None:
        backends, status = load_backends(settings)

    if not backends:
        raise EmbeddingConfigurationError(
            f"No embedding model could be loaded from {settings.models_dir}"
        )

    embedder = SynchronizedEmbeddingService(EmbeddingService(backends))

    if vector_store is None:
        specs = {
            backend.space: CollectionSpec(
                space=backend.space,
                model_id=backend.model_id,
                dimension=embedding_dimension
                or int(getattr(backend, "dimension", EMBEDDING_DIMENSION)),
            )
            for backend in backends
        }
        vector_store = ChromaVectorStore(settings.chroma_dir, specs.values())

    status["vector_store"] = "ready"

    keyword_index = KeywordIndex(settings.keyword_db_path)
    manifest = IndexManifest(settings.manifest_db_path)
    status["keyword_index"] = "ready"
    status["manifest"] = "ready"

    indexing_service = IndexingService(
        embedder=embedder,
        vector_store=vector_store,
        keyword_index=keyword_index,
        manifest=manifest,
        chunker=TextChunker(
            max_units=settings.chunk_max_units,
            overlap_units=settings.chunk_overlap_units,
        ),
    )

    retrieval_service = RetrievalService(
        embedder=embedder,
        vector_store=vector_store,
        keyword_index=keyword_index,
        manifest=manifest,
        ranker=HybridRanker(settings.ranking),
        candidate_multiplier=settings.candidate_multiplier,
    )

    return ServiceContainer(
        settings=settings,
        embedder=embedder,
        vector_store=vector_store,
        keyword_index=keyword_index,
        manifest=manifest,
        indexing_service=indexing_service,
        retrieval_service=retrieval_service,
        job_manager=IndexJobManager(indexing_service),
        component_status=status,
    )
