from pathlib import Path

import pytest

pytest.importorskip("chromadb")

from app.services.embeddings.types import (  # noqa: E402
    EmbeddingModality,
    EmbeddingSpace,
    EmbeddingVector,
)
from app.services.storage.errors import IncompatibleVectorError  # noqa: E402
from app.services.storage.vector_store import (  # noqa: E402
    ChromaVectorStore,
    CollectionSpec,
    VectorRecord,
    collection_name,
)

TEXT_SPEC = CollectionSpec(EmbeddingSpace.TEXT_SEMANTIC, "bert-test", 3)
CLIP_SPEC = CollectionSpec(EmbeddingSpace.MULTIMODAL, "clip-test", 3)


def vector(
    values: tuple[float, float, float],
    space: EmbeddingSpace = EmbeddingSpace.TEXT_SEMANTIC,
    model_id: str = "bert-test",
    modality: EmbeddingModality = EmbeddingModality.TEXT,
) -> EmbeddingVector:
    norm = sum(value * value for value in values) ** 0.5
    return EmbeddingVector(
        content_id="c",
        values=tuple(value / norm for value in values),
        modality=modality,
        space=space,
        model_id=model_id,
    )


def text_record(
    record_id: str,
    document_id: str,
    values: tuple[float, float, float],
    extension: str = ".txt",
) -> VectorRecord:
    return VectorRecord(
        record_id=record_id,
        document_id=document_id,
        embedding=vector(values),
        metadata={"extension": extension, "page_number": None, "chunk_index": 0},
        text=f"text of {record_id}",
    )


@pytest.fixture
def store(tmp_path: Path) -> ChromaVectorStore:
    return ChromaVectorStore(tmp_path / "chroma", [TEXT_SPEC, CLIP_SPEC])


def test_telemetry_is_disabled(store: ChromaVectorStore) -> None:
    assert store.telemetry_enabled is False


def test_upsert_query_returns_cosine_similarity(store: ChromaVectorStore) -> None:
    store.upsert(
        [
            text_record("a:0", "a", (1.0, 0.0, 0.0)),
            text_record("b:0", "b", (0.0, 1.0, 0.0)),
        ]
    )

    hits = store.query(vector((1.0, 0.1, 0.0)), n_results=5)

    assert [hit.record_id for hit in hits] == ["a:0", "b:0"]
    assert hits[0].similarity == pytest.approx(0.995, abs=1e-3)
    assert hits[0].document_id == "a"
    assert hits[0].text == "text of a:0"
    assert hits[0].metadata["model_id"] == "bert-test"
    assert "page_number" not in hits[0].metadata


def test_spaces_are_isolated(store: ChromaVectorStore) -> None:
    store.upsert([text_record("a:0", "a", (1.0, 0.0, 0.0))])
    store.upsert(
        [
            VectorRecord(
                record_id="img:image",
                document_id="img",
                embedding=vector(
                    (1.0, 0.0, 0.0),
                    space=EmbeddingSpace.MULTIMODAL,
                    model_id="clip-test",
                    modality=EmbeddingModality.IMAGE,
                ),
                metadata={"extension": ".png"},
            )
        ]
    )

    clip_hits = store.query(
        vector((1.0, 0.0, 0.0), space=EmbeddingSpace.MULTIMODAL, model_id="clip-test"),
        n_results=5,
    )

    assert [hit.document_id for hit in clip_hits] == ["img"]
    assert clip_hits[0].text is None
    assert store.count(EmbeddingSpace.TEXT_SEMANTIC) == 1
    assert store.count(EmbeddingSpace.MULTIMODAL) == 1


def test_rejects_vectors_from_another_model(store: ChromaVectorStore) -> None:
    with pytest.raises(IncompatibleVectorError):
        store.query(vector((1.0, 0.0, 0.0), model_id="other-model"), n_results=1)

    wrong_dimension = EmbeddingVector(
        content_id="c",
        values=(1.0, 0.0),
        modality=EmbeddingModality.TEXT,
        space=EmbeddingSpace.TEXT_SEMANTIC,
        model_id="bert-test",
    )

    with pytest.raises(IncompatibleVectorError):
        store.upsert(
            [
                VectorRecord(
                    record_id="x",
                    document_id="x",
                    embedding=wrong_dimension,
                )
            ]
        )


def test_extension_filter_and_delete(store: ChromaVectorStore) -> None:
    store.upsert(
        [
            text_record("a:0", "a", (1.0, 0.0, 0.0), ".txt"),
            text_record("a:1", "a", (0.9, 0.1, 0.0), ".txt"),
            text_record("b:0", "b", (1.0, 0.0, 0.1), ".pdf"),
        ]
    )

    pdf_hits = store.query(vector((1.0, 0.0, 0.0)), n_results=5, extensions={".pdf"})
    assert [hit.document_id for hit in pdf_hits] == ["b"]

    store.delete_document("a")

    assert store.count(EmbeddingSpace.TEXT_SEMANTIC) == 1


def test_query_on_empty_collection(store: ChromaVectorStore) -> None:
    assert store.query(vector((1.0, 0.0, 0.0)), n_results=3) == []


def test_upsert_is_idempotent_and_persistent(tmp_path: Path) -> None:
    path = tmp_path / "chroma"
    first = ChromaVectorStore(path, [TEXT_SPEC])
    first.upsert([text_record("a:0", "a", (1.0, 0.0, 0.0))])
    first.upsert([text_record("a:0", "a", (1.0, 0.0, 0.0))])

    reopened = ChromaVectorStore(path, [TEXT_SPEC])

    assert reopened.count(EmbeddingSpace.TEXT_SEMANTIC) == 1


def test_collection_name_is_chroma_safe() -> None:
    spec = CollectionSpec(EmbeddingSpace.MULTIMODAL, "mobileclip/s0 v1!", 512)

    assert collection_name(spec) == "multimodal__mobileclip_s0_v1"
