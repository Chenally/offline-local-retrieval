from datetime import datetime, timezone
from pathlib import Path

import pytest
from app.models import FileMetadata, ParsedDocument
from app.services.embeddings.errors import (
    EmbeddingBackendError,
)
from app.services.embeddings.types import (
    EmbeddingModality,
    EmbeddingSpace,
    EmbeddingVector,
)
from app.services.retrieval.errors import (
    IndexingError,
    InvalidRetrievalInputError,
)
from app.services.retrieval.indexing_service import (
    IndexingService,
)
from app.services.retrieval.types import VectorRecord


class FakeEmbeddingService:
    def __init__(
        self,
        *,
        image_content_ids: list[str] | None = None,
        short_text_batch: bool = False,
        fail: bool = False,
        wrong_contract: bool = False,
    ) -> None:
        self.text_calls: list[
            tuple[list[str], EmbeddingSpace]
        ] = []
        self.image_calls: list[list[Path]] = []
        self.image_content_ids = image_content_ids
        self.short_text_batch = short_text_batch
        self.fail = fail
        self.wrong_contract = wrong_contract

    def embed_texts(
        self,
        texts: list[str],
        space: EmbeddingSpace = (
            EmbeddingSpace.TEXT_SEMANTIC
        ),
    ) -> list[EmbeddingVector]:
        self.text_calls.append(
            (list(texts), space)
        )

        if self.fail:
            raise EmbeddingBackendError(
                "embedding failed"
            )

        vector_space = space

        if (
            self.wrong_contract
            and space is EmbeddingSpace.TEXT_SEMANTIC
        ):
            vector_space = EmbeddingSpace.MULTIMODAL

        vectors = [
            EmbeddingVector(
                content_id=f"text-{index}",
                values=(1.0, 0.0),
                modality=EmbeddingModality.TEXT,
                space=vector_space,
                model_id=(
                    "bert"
                    if space
                    is EmbeddingSpace.TEXT_SEMANTIC
                    else "mobileclip"
                ),
            )
            for index, _ in enumerate(texts)
        ]

        if self.short_text_batch:
            return vectors[:-1]

        return vectors

    def embed_images(
        self,
        paths: list[Path],
    ) -> list[EmbeddingVector]:
        self.image_calls.append(
            list(paths)
        )

        if self.fail:
            raise EmbeddingBackendError(
                "embedding failed"
            )

        content_ids = (
            self.image_content_ids
            if self.image_content_ids is not None
            else [
                f"image-{index}"
                for index, _ in enumerate(paths)
            ]
        )

        return [
            EmbeddingVector(
                content_id=content_id,
                values=(0.0, 1.0),
                modality=EmbeddingModality.IMAGE,
                space=EmbeddingSpace.MULTIMODAL,
                model_id="mobileclip",
            )
            for content_id in content_ids
        ]


class FakeVectorStore:
    def __init__(self) -> None:
        self.records: list[VectorRecord] = []

    def upsert(
        self,
        records: list[VectorRecord],
    ) -> None:
        self.records.extend(records)


def _document(
    tmp_path: Path,
    *,
    document_id: str = "document-one",
    text: str = "local retrieval",
    sha256: str = "document-sha",
) -> ParsedDocument:
    path = tmp_path / "document.txt"

    return ParsedDocument(
        document_id=document_id,
        text=text,
        metadata=FileMetadata(
            path=path,
            file_name=path.name,
            extension=".txt",
            mime_type="text/plain",
            size_bytes=16,
            modified_at=datetime(
                2026,
                1,
                1,
                tzinfo=timezone.utc,
            ),
            sha256=sha256,
        ),
    )


def test_indexes_documents_in_both_spaces(
    tmp_path: Path,
) -> None:
    embeddings = FakeEmbeddingService()
    store = FakeVectorStore()
    service = IndexingService(
        embeddings,
        store,
    )

    documents = [
        _document(
            tmp_path,
            document_id="one",
            text="first document",
            sha256="sha-one",
        ),
        _document(
            tmp_path,
            document_id="two",
            text="second document",
            sha256="sha-two",
        ),
    ]

    indexed = service.index_documents(
        documents
    )

    assert indexed == 2
    assert embeddings.text_calls == [
        (
            [
                "first document",
                "second document",
            ],
            EmbeddingSpace.TEXT_SEMANTIC,
        ),
        (
            [
                "first document",
                "second document",
            ],
            EmbeddingSpace.MULTIMODAL,
        ),
    ]
    assert len(store.records) == 4
    assert {
        (
            record.record_id,
            record.embedding.space,
        )
        for record in store.records
    } == {
        (
            "text:one",
            EmbeddingSpace.TEXT_SEMANTIC,
        ),
        (
            "text:one",
            EmbeddingSpace.MULTIMODAL,
        ),
        (
            "text:two",
            EmbeddingSpace.TEXT_SEMANTIC,
        ),
        (
            "text:two",
            EmbeddingSpace.MULTIMODAL,
        ),
    }


def test_indexes_unique_image_content_only(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"

    embeddings = FakeEmbeddingService(
        image_content_ids=[
            "same-image-sha",
            "same-image-sha",
        ]
    )
    store = FakeVectorStore()
    service = IndexingService(
        embeddings,
        store,
    )

    indexed = service.index_images(
        [first, second]
    )

    assert indexed == 1
    assert len(store.records) == 1

    record = store.records[0]

    assert record.record_id == (
        "image:same-image-sha"
    )
    assert record.source_path == first.resolve()
    assert record.source_sha256 == (
        "same-image-sha"
    )
    assert record.content == str(first.resolve())
    assert (
        record.embedding.modality
        is EmbeddingModality.IMAGE
    )
    assert (
        record.embedding.space
        is EmbeddingSpace.MULTIMODAL
    )


def test_empty_inputs_do_not_call_dependencies() -> None:
    embeddings = FakeEmbeddingService()
    store = FakeVectorStore()
    service = IndexingService(
        embeddings,
        store,
    )

    assert service.index_documents([]) == 0
    assert service.index_images([]) == 0
    assert embeddings.text_calls == []
    assert embeddings.image_calls == []
    assert store.records == []


@pytest.mark.parametrize(
    (
        "document_id",
        "text",
        "sha256",
    ),
    [
        ("", "content", "sha"),
        ("document", "   ", "sha"),
        ("document", "content", ""),
    ],
)
def test_rejects_invalid_documents(
    tmp_path: Path,
    document_id: str,
    text: str,
    sha256: str,
) -> None:
    service = IndexingService(
        FakeEmbeddingService(),
        FakeVectorStore(),
    )

    with pytest.raises(
        InvalidRetrievalInputError
    ):
        service.index_documents(
            [
                _document(
                    tmp_path,
                    document_id=document_id,
                    text=text,
                    sha256=sha256,
                )
            ]
        )


def test_rejects_duplicate_document_ids(
    tmp_path: Path,
) -> None:
    service = IndexingService(
        FakeEmbeddingService(),
        FakeVectorStore(),
    )

    with pytest.raises(
        InvalidRetrievalInputError
    ):
        service.index_documents(
            [
                _document(
                    tmp_path,
                    document_id="duplicate",
                    sha256="sha-one",
                ),
                _document(
                    tmp_path,
                    document_id="duplicate",
                    sha256="sha-two",
                ),
            ]
        )


def test_rejects_incomplete_embedding_batch(
    tmp_path: Path,
) -> None:
    store = FakeVectorStore()
    service = IndexingService(
        FakeEmbeddingService(
            short_text_batch=True
        ),
        store,
    )

    with pytest.raises(IndexingError):
        service.index_documents(
            [_document(tmp_path)]
        )

    assert store.records == []


def test_normalizes_embedding_failure(
    tmp_path: Path,
) -> None:
    service = IndexingService(
        FakeEmbeddingService(fail=True),
        FakeVectorStore(),
    )

    with pytest.raises(IndexingError):
        service.index_documents(
            [_document(tmp_path)]
        )


def test_rejects_wrong_embedding_contract(
    tmp_path: Path,
) -> None:
    store = FakeVectorStore()
    service = IndexingService(
        FakeEmbeddingService(
            wrong_contract=True
        ),
        store,
    )

    with pytest.raises(IndexingError):
        service.index_documents(
            [_document(tmp_path)]
        )

    assert store.records == []
