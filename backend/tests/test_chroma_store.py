from pathlib import Path

import pytest
from app.services.embeddings.types import (
    EmbeddingModality,
    EmbeddingSpace,
    EmbeddingVector,
)
from app.services.retrieval.chroma_store import (
    ChromaVectorStore,
)
from app.services.retrieval.errors import (
    InvalidRetrievalInputError,
    RetrievalConfigurationError,
)
from app.services.retrieval.types import VectorRecord


def _embedding(
    *,
    content_id: str,
    values: tuple[float, ...],
    modality: EmbeddingModality,
    space: EmbeddingSpace,
    model_id: str,
) -> EmbeddingVector:
    return EmbeddingVector(
        content_id=content_id,
        values=values,
        modality=modality,
        space=space,
        model_id=model_id,
    )


def _record(
    *,
    record_id: str,
    content: str,
    path: Path,
    source_sha256: str,
    embedding: EmbeddingVector,
) -> VectorRecord:
    return VectorRecord(
        record_id=record_id,
        content=content,
        source_path=path,
        source_sha256=source_sha256,
        embedding=embedding,
    )


def test_upsert_persists_and_updates_by_id(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "chroma"
    source_path = tmp_path / "document.txt"

    original = _record(
        record_id="document-1",
        content="original content",
        path=source_path,
        source_sha256="source-one",
        embedding=_embedding(
            content_id="content-one",
            values=(1.0, 0.0),
            modality=EmbeddingModality.TEXT,
            space=EmbeddingSpace.TEXT_SEMANTIC,
            model_id="bert-model",
        ),
    )

    ChromaVectorStore(database_path).upsert(
        [original]
    )

    reopened = ChromaVectorStore(database_path)

    first_matches = reopened.query(
        original.embedding,
        top_k=10,
        modality=EmbeddingModality.TEXT,
    )

    assert len(first_matches) == 1
    assert first_matches[0].record_id == "document-1"
    assert first_matches[0].content == "original content"
    assert first_matches[0].similarity == pytest.approx(
        1.0
    )

    updated = _record(
        record_id="document-1",
        content="updated content",
        path=source_path,
        source_sha256="source-two",
        embedding=_embedding(
            content_id="content-two",
            values=(0.0, 1.0),
            modality=EmbeddingModality.TEXT,
            space=EmbeddingSpace.TEXT_SEMANTIC,
            model_id="bert-model",
        ),
    )

    reopened.upsert([updated])

    updated_matches = reopened.query(
        updated.embedding,
        top_k=10,
        modality=EmbeddingModality.TEXT,
    )

    assert len(updated_matches) == 1
    assert updated_matches[0].content == "updated content"
    assert (
        updated_matches[0].source_sha256
        == "source-two"
    )


def test_query_isolates_space_model_and_modality(
    tmp_path: Path,
) -> None:
    store = ChromaVectorStore(
        tmp_path / "chroma"
    )

    semantic = _record(
        record_id="semantic-text",
        content="semantic document",
        path=tmp_path / "semantic.txt",
        source_sha256="semantic-sha",
        embedding=_embedding(
            content_id="semantic-content",
            values=(1.0, 0.0),
            modality=EmbeddingModality.TEXT,
            space=EmbeddingSpace.TEXT_SEMANTIC,
            model_id="bert-model",
        ),
    )

    multimodal_text = _record(
        record_id="multimodal-text",
        content="multimodal document",
        path=tmp_path / "multimodal.txt",
        source_sha256="multimodal-text-sha",
        embedding=_embedding(
            content_id="multimodal-text-content",
            values=(1.0, 0.0),
            modality=EmbeddingModality.TEXT,
            space=EmbeddingSpace.MULTIMODAL,
            model_id="mobileclip",
        ),
    )

    multimodal_image = _record(
        record_id="multimodal-image",
        content="/local/image.jpg",
        path=tmp_path / "image.jpg",
        source_sha256="multimodal-image-sha",
        embedding=_embedding(
            content_id="multimodal-image-content",
            values=(1.0, 0.0),
            modality=EmbeddingModality.IMAGE,
            space=EmbeddingSpace.MULTIMODAL,
            model_id="mobileclip",
        ),
    )

    store.upsert(
        [
            semantic,
            multimodal_text,
            multimodal_image,
        ]
    )

    semantic_matches = store.query(
        semantic.embedding,
        top_k=10,
    )

    assert [
        match.record_id
        for match in semantic_matches
    ] == ["semantic-text"]

    image_matches = store.query(
        multimodal_text.embedding,
        top_k=10,
        modality=EmbeddingModality.IMAGE,
    )

    assert [
        match.record_id
        for match in image_matches
    ] == ["multimodal-image"]

    wrong_model = _embedding(
        content_id="query",
        values=(1.0, 0.0),
        modality=EmbeddingModality.TEXT,
        space=EmbeddingSpace.MULTIMODAL,
        model_id="different-model",
    )

    assert store.query(
        wrong_model,
        top_k=10,
    ) == []


def test_delete_only_affects_requested_space(
    tmp_path: Path,
) -> None:
    store = ChromaVectorStore(
        tmp_path / "chroma"
    )

    semantic = _record(
        record_id="shared-id",
        content="semantic",
        path=tmp_path / "semantic.txt",
        source_sha256="semantic-sha",
        embedding=_embedding(
            content_id="semantic-content",
            values=(1.0, 0.0),
            modality=EmbeddingModality.TEXT,
            space=EmbeddingSpace.TEXT_SEMANTIC,
            model_id="bert-model",
        ),
    )

    multimodal = _record(
        record_id="shared-id",
        content="multimodal",
        path=tmp_path / "image.jpg",
        source_sha256="image-sha",
        embedding=_embedding(
            content_id="image-content",
            values=(1.0, 0.0),
            modality=EmbeddingModality.IMAGE,
            space=EmbeddingSpace.MULTIMODAL,
            model_id="mobileclip",
        ),
    )

    store.upsert(
        [
            semantic,
            multimodal,
        ]
    )

    store.delete(
        ["shared-id"],
        space=EmbeddingSpace.TEXT_SEMANTIC,
    )

    assert store.query(
        semantic.embedding,
        top_k=1,
    ) == []

    assert len(
        store.query(
            multimodal.embedding,
            top_k=1,
        )
    ) == 1


def test_rejects_invalid_inputs(
    tmp_path: Path,
) -> None:
    store = ChromaVectorStore(
        tmp_path / "chroma"
    )

    embedding = _embedding(
        content_id="content",
        values=(1.0, 0.0),
        modality=EmbeddingModality.TEXT,
        space=EmbeddingSpace.TEXT_SEMANTIC,
        model_id="bert-model",
    )

    invalid = _record(
        record_id="",
        content="content",
        path=tmp_path / "document.txt",
        source_sha256="source-sha",
        embedding=embedding,
    )

    with pytest.raises(
        InvalidRetrievalInputError
    ):
        store.upsert([invalid])

    with pytest.raises(
        InvalidRetrievalInputError
    ):
        store.query(
            embedding,
            top_k=0,
        )

    with pytest.raises(
        InvalidRetrievalInputError
    ):
        store.delete(
            [""],
            space=EmbeddingSpace.TEXT_SEMANTIC,
        )


def test_rejects_duplicate_ids_in_same_space(
    tmp_path: Path,
) -> None:
    store = ChromaVectorStore(
        tmp_path / "chroma"
    )

    embedding = _embedding(
        content_id="content",
        values=(1.0, 0.0),
        modality=EmbeddingModality.TEXT,
        space=EmbeddingSpace.TEXT_SEMANTIC,
        model_id="bert-model",
    )

    first = _record(
        record_id="same-id",
        content="first",
        path=tmp_path / "first.txt",
        source_sha256="first-sha",
        embedding=embedding,
    )

    second = _record(
        record_id="same-id",
        content="second",
        path=tmp_path / "second.txt",
        source_sha256="second-sha",
        embedding=embedding,
    )

    with pytest.raises(
        InvalidRetrievalInputError
    ):
        store.upsert(
            [
                first,
                second,
            ]
        )


def test_empty_operations_are_safe(
    tmp_path: Path,
) -> None:
    store = ChromaVectorStore(
        tmp_path / "chroma"
    )

    query = _embedding(
        content_id="query",
        values=(1.0, 0.0),
        modality=EmbeddingModality.TEXT,
        space=EmbeddingSpace.TEXT_SEMANTIC,
        model_id="bert-model",
    )

    store.upsert([])
    store.delete(
        [],
        space=EmbeddingSpace.TEXT_SEMANTIC,
    )

    assert store.query(
        query,
        top_k=5,
    ) == []


def test_rejects_database_path_that_is_a_file(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "not-a-directory"
    database_path.write_text(
        "file",
        encoding="utf-8",
    )

    with pytest.raises(
        RetrievalConfigurationError
    ):
        ChromaVectorStore(database_path)
