"""Contract and error-path tests for ChromaVectorStore using an injected
client, so they run without the chromadb package."""

import pytest
from app.services.embeddings.types import (
    EmbeddingModality,
    EmbeddingSpace,
    EmbeddingVector,
)
from app.services.storage.errors import IncompatibleVectorError, StorageError
from app.services.storage.vector_store import (
    ChromaVectorStore,
    CollectionSpec,
    VectorRecord,
)

SPEC = CollectionSpec(EmbeddingSpace.TEXT_SEMANTIC, "bert", 2)


class BrokenCollection:
    name = "broken"

    def __init__(self, metadata=None) -> None:
        self.metadata = metadata or {}

    def upsert(self, **kwargs):
        raise RuntimeError("disk full")

    def delete(self, **kwargs):
        raise RuntimeError("locked")

    def count(self):
        return 1

    def query(self, **kwargs):
        raise RuntimeError("index corrupted")


class FakeClient:
    def __init__(self, collection) -> None:
        self.collection = collection

    def get_or_create_collection(self, **kwargs):
        return self.collection


def embedding(space=EmbeddingSpace.TEXT_SEMANTIC, model_id="bert") -> EmbeddingVector:
    return EmbeddingVector(
        content_id="c",
        values=(1.0, 0.0),
        modality=EmbeddingModality.TEXT,
        space=space,
        model_id=model_id,
    )


def make_store(collection=None) -> ChromaVectorStore:
    return ChromaVectorStore(
        "unused",
        [SPEC],
        client=FakeClient(collection or BrokenCollection()),
    )


def test_backend_failures_become_storage_errors() -> None:
    store = make_store()
    record = VectorRecord(record_id="r", document_id="d", embedding=embedding())

    with pytest.raises(StorageError, match="disk full"):
        store.upsert([record])

    with pytest.raises(StorageError, match="locked"):
        store.delete_document("d")

    with pytest.raises(StorageError, match="index corrupted"):
        store.query(embedding(), n_results=1)


def test_rejects_existing_collection_from_another_model() -> None:
    with pytest.raises(IncompatibleVectorError, match="model_id"):
        make_store(BrokenCollection(metadata={"model_id": "older-model"}))


def test_rejects_invalid_specs() -> None:
    client = FakeClient(BrokenCollection())

    with pytest.raises(StorageError, match="Duplicate"):
        ChromaVectorStore("unused", [SPEC, SPEC], client=client)

    with pytest.raises(StorageError, match="positive"):
        ChromaVectorStore(
            "unused",
            [CollectionSpec(EmbeddingSpace.TEXT_SEMANTIC, "bert", 0)],
            client=client,
        )


def test_rejects_unknown_space_and_bad_limits() -> None:
    store = make_store()

    with pytest.raises(IncompatibleVectorError, match="No collection"):
        store.query(embedding(space=EmbeddingSpace.MULTIMODAL), n_results=1)

    with pytest.raises(ValueError):
        store.query(embedding(), n_results=0)

    assert store.count(EmbeddingSpace.MULTIMODAL) == 0


def test_telemetry_reports_false_without_settings_accessor() -> None:
    assert make_store().telemetry_enabled is False
