from __future__ import annotations

import re
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from app.services.embeddings.types import EmbeddingSpace, EmbeddingVector
from app.services.storage.errors import IncompatibleVectorError, StorageError

MetadataValue = str | int | float | bool

_UPSERT_BATCH_SIZE = 512


@dataclass(frozen=True, slots=True)
class CollectionSpec:
    """Model contract enforced for one embedding space."""

    space: EmbeddingSpace
    model_id: str
    dimension: int


@dataclass(frozen=True, slots=True)
class VectorRecord:
    """One vector together with the metadata needed to explain a hit."""

    record_id: str
    document_id: str
    embedding: EmbeddingVector
    metadata: Mapping[str, MetadataValue | None] = field(default_factory=dict)
    text: str | None = None


@dataclass(frozen=True, slots=True)
class VectorHit:
    record_id: str
    document_id: str
    similarity: float
    metadata: Mapping[str, MetadataValue]
    text: str | None


class VectorStore(Protocol):
    """Persistent storage and nearest-neighbour search for embeddings."""

    def upsert(self, records: Sequence[VectorRecord]) -> None: ...

    def delete_document(self, document_id: str) -> None: ...

    def query(
        self,
        embedding: EmbeddingVector,
        n_results: int,
        extensions: Collection[str] | None = None,
    ) -> list[VectorHit]: ...

    def count(self, space: EmbeddingSpace) -> int: ...


def collection_name(spec: CollectionSpec) -> str:
    """Build a Chroma-safe collection name that encodes space and model."""

    raw_name = f"{spec.space.value}__{spec.model_id}"
    name = re.sub(r"[^A-Za-z0-9._-]", "_", raw_name).strip("._-")

    return name[:512]


class ChromaVectorStore:
    """Stores each embedding space in its own persistent Chroma collection.

    Every collection is bound to exactly one ``(space, model_id, dimension)``
    contract. Writes and queries that do not satisfy the contract are
    rejected, so BERT and MobileCLIP vectors can never be compared.
    """

    def __init__(
        self,
        path: str | Path,
        specs: Iterable[CollectionSpec],
        client: Any | None = None,
    ) -> None:
        if client is None:
            import chromadb
            from chromadb.config import Settings

            Path(path).mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(
                path=str(path),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=False,
                ),
            )

        self._client: Any = client
        self._specs: dict[EmbeddingSpace, CollectionSpec] = {}
        self._collections: dict[EmbeddingSpace, Any] = {}

        for spec in specs:
            if spec.space in self._specs:
                raise StorageError(f"Duplicate collection spec for {spec.space.value}")

            if spec.dimension <= 0:
                raise StorageError("Collection dimension must be positive")

            collection = self._client.get_or_create_collection(
                name=collection_name(spec),
                metadata={
                    "hnsw:space": "cosine",
                    "embedding_space": spec.space.value,
                    "model_id": spec.model_id,
                    "dimension": spec.dimension,
                },
                embedding_function=None,
            )

            self._validate_collection(collection, spec)
            self._specs[spec.space] = spec
            self._collections[spec.space] = collection

    @property
    def telemetry_enabled(self) -> bool:
        settings = getattr(self._client, "get_settings", None)

        if settings is None:
            return False

        return bool(settings().anonymized_telemetry)

    def upsert(self, records: Sequence[VectorRecord]) -> None:
        by_space: dict[EmbeddingSpace, list[VectorRecord]] = {}

        for record in records:
            self._check_embedding(record.embedding)
            by_space.setdefault(record.embedding.space, []).append(record)

        for space, space_records in by_space.items():
            collection = self._collections[space]

            for start in range(0, len(space_records), _UPSERT_BATCH_SIZE):
                batch = space_records[start : start + _UPSERT_BATCH_SIZE]
                with_text = [record for record in batch if record.text is not None]
                without_text = [record for record in batch if record.text is None]

                try:
                    if with_text:
                        collection.upsert(
                            ids=[record.record_id for record in with_text],
                            embeddings=[
                                list(record.embedding.values) for record in with_text
                            ],
                            metadatas=[self._metadata(record) for record in with_text],
                            documents=[record.text for record in with_text],
                        )

                    if without_text:
                        collection.upsert(
                            ids=[record.record_id for record in without_text],
                            embeddings=[
                                list(record.embedding.values) for record in without_text
                            ],
                            metadatas=[
                                self._metadata(record) for record in without_text
                            ],
                        )
                except Exception as exc:
                    raise StorageError(
                        f"Could not write vectors to {space.value}: {exc}"
                    ) from exc

    def delete_document(self, document_id: str) -> None:
        for space, collection in self._collections.items():
            try:
                collection.delete(where={"document_id": document_id})
            except Exception as exc:
                raise StorageError(
                    f"Could not delete {document_id} from {space.value}: {exc}"
                ) from exc

    def query(
        self,
        embedding: EmbeddingVector,
        n_results: int,
        extensions: Collection[str] | None = None,
    ) -> list[VectorHit]:
        self._check_embedding(embedding)

        if n_results <= 0:
            raise ValueError("n_results must be greater than zero")

        collection = self._collections[embedding.space]

        try:
            available = collection.count()

            if available == 0:
                return []

            where = None

            if extensions:
                where = {"extension": {"$in": sorted(extensions)}}

            result = collection.query(
                query_embeddings=[list(embedding.values)],
                n_results=min(n_results, available),
                where=where,
                include=["embeddings", "metadatas", "documents"],
            )
        except Exception as exc:
            raise StorageError(
                f"Vector query failed for {embedding.space.value}: {exc}"
            ) from exc

        ids = result["ids"][0]

        if not ids:
            return []

        metadatas = (result.get("metadatas") or [[{}] * len(ids)])[0]
        documents = (result.get("documents") or [[None] * len(ids)])[0]
        stored = np.asarray(result["embeddings"][0], dtype=np.float32)
        query_values = np.asarray(embedding.values, dtype=np.float32)

        # Vectors are L2-normalized, so the dot product is the cosine
        # similarity regardless of the distance function the index uses.
        similarities = stored @ query_values

        hits = [
            VectorHit(
                record_id=record_id,
                document_id=str((metadata or {}).get("document_id", "")),
                similarity=float(similarity),
                metadata=dict(metadata or {}),
                text=document,
            )
            for record_id, metadata, document, similarity in zip(
                ids,
                metadatas,
                documents,
                similarities,
                strict=True,
            )
        ]

        hits.sort(key=lambda hit: (-hit.similarity, hit.record_id))
        return hits

    def count(self, space: EmbeddingSpace) -> int:
        collection = self._collections.get(space)

        if collection is None:
            return 0

        return int(collection.count())

    def _check_embedding(self, embedding: EmbeddingVector) -> None:
        spec = self._specs.get(embedding.space)

        if spec is None:
            raise IncompatibleVectorError(
                f"No collection is configured for {embedding.space.value}"
            )

        if embedding.model_id != spec.model_id:
            raise IncompatibleVectorError(
                f"{embedding.space.value} expects model {spec.model_id}, "
                f"got {embedding.model_id}"
            )

        if len(embedding.values) != spec.dimension:
            raise IncompatibleVectorError(
                f"{embedding.space.value} expects dimension {spec.dimension}, "
                f"got {len(embedding.values)}"
            )

    @staticmethod
    def _validate_collection(collection: Any, spec: CollectionSpec) -> None:
        metadata = collection.metadata or {}

        expected = {
            "embedding_space": spec.space.value,
            "model_id": spec.model_id,
            "dimension": spec.dimension,
        }

        for key, value in expected.items():
            if key in metadata and metadata[key] != value:
                raise IncompatibleVectorError(
                    f"Collection {collection.name} was created with "
                    f"{key}={metadata[key]!r}, expected {value!r}"
                )

    @staticmethod
    def _metadata(record: VectorRecord) -> dict[str, MetadataValue]:
        metadata: dict[str, MetadataValue] = {
            key: value for key, value in record.metadata.items() if value is not None
        }
        metadata["document_id"] = record.document_id
        metadata["space"] = record.embedding.space.value
        metadata["modality"] = record.embedding.modality.value
        metadata["model_id"] = record.embedding.model_id
        return metadata
