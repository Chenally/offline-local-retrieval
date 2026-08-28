from __future__ import annotations

from app.services.embeddings.errors import EmbeddingError
from app.services.embeddings.service import EmbeddingService
from app.services.embeddings.types import (
    EmbeddingModality,
    EmbeddingSpace,
    EmbeddingVector,
)
from app.services.retrieval.base import VectorStore
from app.services.retrieval.errors import (
    InvalidRetrievalInputError,
    SearchError,
)
from app.services.retrieval.types import (
    RetrievalCandidates,
)


class RetrievalService:
    """Collects retrieval candidates from isolated vector spaces."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
    ) -> None:
        self._embedding_service = embedding_service
        self._vector_store = vector_store

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 10,
    ) -> RetrievalCandidates:
        if (
            not isinstance(query, str)
            or not query.strip()
        ):
            raise InvalidRetrievalInputError(
                "Search query must not be blank"
            )

        if top_k <= 0:
            raise InvalidRetrievalInputError(
                "top_k must be greater than zero"
            )

        try:
            semantic_vectors = (
                self._embedding_service.embed_texts(
                    [query],
                    space=EmbeddingSpace.TEXT_SEMANTIC,
                )
            )
            multimodal_vectors = (
                self._embedding_service.embed_texts(
                    [query],
                    space=EmbeddingSpace.MULTIMODAL,
                )
            )
        except EmbeddingError as exc:
            raise SearchError(
                "Could not embed search query"
            ) from exc

        semantic_query = self._require_query_vector(
            semantic_vectors,
            space=EmbeddingSpace.TEXT_SEMANTIC,
        )
        multimodal_query = self._require_query_vector(
            multimodal_vectors,
            space=EmbeddingSpace.MULTIMODAL,
        )

        text_semantic = self._vector_store.query(
            semantic_query,
            top_k=top_k,
            modality=EmbeddingModality.TEXT,
        )
        multimodal_text = self._vector_store.query(
            multimodal_query,
            top_k=top_k,
            modality=EmbeddingModality.TEXT,
        )
        multimodal_images = self._vector_store.query(
            multimodal_query,
            top_k=top_k,
            modality=EmbeddingModality.IMAGE,
        )

        return RetrievalCandidates(
            text_semantic=tuple(text_semantic),
            multimodal_text=tuple(multimodal_text),
            multimodal_images=tuple(
                multimodal_images
            ),
        )

    @staticmethod
    def _require_query_vector(
        vectors: list[EmbeddingVector],
        *,
        space: EmbeddingSpace,
    ) -> EmbeddingVector:
        if len(vectors) != 1:
            raise SearchError(
                "Search embedding must contain "
                "exactly one vector"
            )

        vector = vectors[0]

        if (
            vector.modality
            is not EmbeddingModality.TEXT
            or vector.space is not space
        ):
            raise SearchError(
                "Search embedding does not match "
                "the requested vector space"
            )

        return vector
