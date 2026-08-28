from __future__ import annotations

from pathlib import Path

from app.models import ParsedDocument
from app.services.embeddings.errors import EmbeddingError
from app.services.embeddings.service import EmbeddingService
from app.services.embeddings.types import (
    EmbeddingModality,
    EmbeddingSpace,
    EmbeddingVector,
)
from app.services.retrieval.base import VectorStore
from app.services.retrieval.errors import (
    IndexingError,
    InvalidRetrievalInputError,
)
from app.services.retrieval.types import VectorRecord


class IndexingService:
    """Converts parsed local content into persistent vector records."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
    ) -> None:
        self._embedding_service = embedding_service
        self._vector_store = vector_store

    def index_documents(
        self,
        documents: list[ParsedDocument],
    ) -> int:
        if not documents:
            return 0

        self._validate_documents(documents)

        texts = [
            document.text
            for document in documents
        ]

        try:
            semantic_vectors = (
                self._embedding_service.embed_texts(
                    texts,
                    space=EmbeddingSpace.TEXT_SEMANTIC,
                )
            )
            multimodal_vectors = (
                self._embedding_service.embed_texts(
                    texts,
                    space=EmbeddingSpace.MULTIMODAL,
                )
            )
        except EmbeddingError as exc:
            raise IndexingError(
                "Could not embed parsed documents"
            ) from exc

        self._require_count(
            len(documents),
            semantic_vectors,
            "text-semantic",
        )
        self._require_count(
            len(documents),
            multimodal_vectors,
            "multimodal-text",
        )

        records: list[VectorRecord] = []

        for document, semantic, multimodal in zip(
            documents,
            semantic_vectors,
            multimodal_vectors,
            strict=True,
        ):
            self._require_contract(
                semantic,
                modality=EmbeddingModality.TEXT,
                space=EmbeddingSpace.TEXT_SEMANTIC,
            )
            self._require_contract(
                multimodal,
                modality=EmbeddingModality.TEXT,
                space=EmbeddingSpace.MULTIMODAL,
            )

            for embedding in (
                semantic,
                multimodal,
            ):
                records.append(
                    VectorRecord(
                        record_id=(
                            f"text:{document.document_id}"
                        ),
                        content=document.text,
                        source_path=document.metadata.path,
                        source_sha256=(
                            document.metadata.sha256
                        ),
                        embedding=embedding,
                    )
                )

        self._vector_store.upsert(records)

        return len(documents)

    def index_images(
        self,
        image_paths: list[str | Path],
    ) -> int:
        if not image_paths:
            return 0

        paths = [
            Path(path).expanduser().resolve()
            for path in image_paths
        ]

        try:
            vectors = self._embedding_service.embed_images(
                paths
            )
        except EmbeddingError as exc:
            raise IndexingError(
                "Could not embed local images"
            ) from exc

        self._require_count(
            len(paths),
            vectors,
            "multimodal-image",
        )

        records: list[VectorRecord] = []
        seen_content_ids: set[str] = set()

        for path, embedding in zip(
            paths,
            vectors,
            strict=True,
        ):
            self._require_contract(
                embedding,
                modality=EmbeddingModality.IMAGE,
                space=EmbeddingSpace.MULTIMODAL,
            )

            if embedding.content_id in seen_content_ids:
                continue

            seen_content_ids.add(
                embedding.content_id
            )

            records.append(
                VectorRecord(
                    record_id=(
                        f"image:{embedding.content_id}"
                    ),
                    content=str(path),
                    source_path=path,
                    source_sha256=(
                        embedding.content_id
                    ),
                    embedding=embedding,
                )
            )

        self._vector_store.upsert(records)

        return len(records)

    @staticmethod
    def _validate_documents(
        documents: list[ParsedDocument],
    ) -> None:
        document_ids: set[str] = set()

        for document in documents:
            if not document.document_id.strip():
                raise InvalidRetrievalInputError(
                    "Document ID must not be blank"
                )

            if not document.text.strip():
                raise InvalidRetrievalInputError(
                    "Document text must not be blank"
                )

            if not document.metadata.sha256.strip():
                raise InvalidRetrievalInputError(
                    "Document SHA-256 must not be blank"
                )

            if document.document_id in document_ids:
                raise InvalidRetrievalInputError(
                    "Document IDs must be unique "
                    "within one indexing request"
                )

            document_ids.add(
                document.document_id
            )

    @staticmethod
    def _require_count(
        expected: int,
        vectors: list[EmbeddingVector],
        label: str,
    ) -> None:
        if len(vectors) != expected:
            raise IndexingError(
                f"{label} embedding count does not "
                "match the input count"
            )

    @staticmethod
    def _require_contract(
        vector: EmbeddingVector,
        *,
        modality: EmbeddingModality,
        space: EmbeddingSpace,
    ) -> None:
        if (
            vector.modality is not modality
            or vector.space is not space
        ):
            raise IndexingError(
                "Embedding vector does not match "
                "the required indexing contract"
            )
