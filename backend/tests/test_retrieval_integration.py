from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pytest
from app.models import FileMetadata, ParsedDocument
from app.services.embeddings.types import (
    EmbeddingModality,
    EmbeddingSpace,
    EmbeddingVector,
)
from app.services.retrieval.chroma_store import (
    ChromaVectorStore,
)
from app.services.retrieval.indexing_service import (
    IndexingService,
)
from app.services.retrieval.retrieval_service import (
    RetrievalService,
)
from app.services.retrieval.search_service import (
    SearchService,
)


class DeterministicEmbeddingService:
    """Produces deterministic vectors for integration testing."""

    def embed_texts(
        self,
        texts: list[str],
        space: EmbeddingSpace = (
            EmbeddingSpace.TEXT_SEMANTIC
        ),
    ) -> list[EmbeddingVector]:
        return [
            EmbeddingVector(
                content_id=self._text_id(text),
                values=self._values(text),
                modality=EmbeddingModality.TEXT,
                space=space,
                model_id=(
                    "bert"
                    if space
                    is EmbeddingSpace.TEXT_SEMANTIC
                    else "mobileclip"
                ),
            )
            for text in texts
        ]

    def embed_images(
        self,
        paths: list[Path],
    ) -> list[EmbeddingVector]:
        return [
            EmbeddingVector(
                content_id=hashlib.sha256(
                    path.read_bytes()
                ).hexdigest(),
                values=self._values(path.name),
                modality=EmbeddingModality.IMAGE,
                space=EmbeddingSpace.MULTIMODAL,
                model_id="mobileclip",
            )
            for path in paths
        ]

    @staticmethod
    def _values(
        value: str,
    ) -> tuple[float, float]:
        if "cat" in value.casefold():
            return (1.0, 0.0)

        return (0.0, 1.0)

    @staticmethod
    def _text_id(text: str) -> str:
        return hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest()


def _document(
    tmp_path: Path,
    *,
    document_id: str,
    text: str,
) -> ParsedDocument:
    path = tmp_path / f"{document_id}.txt"
    path.write_text(
        text,
        encoding="utf-8",
    )

    content = path.read_bytes()

    return ParsedDocument(
        document_id=document_id,
        text=text,
        metadata=FileMetadata(
            path=path.resolve(),
            file_name=path.name,
            extension=".txt",
            mime_type="text/plain",
            size_bytes=len(content),
            modified_at=datetime(
                2026,
                1,
                1,
                tzinfo=timezone.utc,
            ),
            sha256=hashlib.sha256(
                content
            ).hexdigest(),
        ),
    )


def test_indexes_persists_retrieves_and_ranks(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "chroma"
    embeddings = DeterministicEmbeddingService()

    writer_store = ChromaVectorStore(
        database_path
    )
    indexer = IndexingService(
        embeddings,
        writer_store,
    )

    documents = [
        _document(
            tmp_path,
            document_id="cat-document",
            text="cat local guide",
        ),
        _document(
            tmp_path,
            document_id="dog-document",
            text="dog training notes",
        ),
    ]

    image_path = tmp_path / "cat-image.png"
    image_path.write_bytes(
        b"deterministic image fixture"
    )

    assert (
        indexer.index_documents(documents)
        == 2
    )
    assert (
        indexer.index_images([image_path])
        == 1
    )

    # A new store instance verifies persisted data,
    # rather than querying only in-memory state.
    reader_store = ChromaVectorStore(
        database_path
    )
    retrieval = RetrievalService(
        embeddings,
        reader_store,
    )
    search = SearchService(retrieval)

    results = search.search(
        "cat",
        limit=3,
    )

    image_id = hashlib.sha256(
        image_path.read_bytes()
    ).hexdigest()

    assert [
        result.record_id
        for result in results
    ] == [
        f"image:{image_id}",
        "text:cat-document",
        "text:dog-document",
    ]
    assert [
        result.modality
        for result in results
    ] == [
        EmbeddingModality.IMAGE,
        EmbeddingModality.TEXT,
        EmbeddingModality.TEXT,
    ]
    assert results[0].final_score == (
        pytest.approx(1.0)
    )
    assert results[1].final_score == (
        pytest.approx(1.0)
    )
    assert results[2].final_score == (
        pytest.approx(0.0)
    )
    assert (
        results[1].keyword_score
        == 1.0
    )
    assert (
        results[1].text_semantic_score
        == pytest.approx(1.0)
    )
    assert (
        results[1].multimodal_score
        == pytest.approx(1.0)
    )
