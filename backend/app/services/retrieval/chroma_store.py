from __future__ import annotations

from pathlib import Path

import chromadb
from app.services.embeddings.types import (
    EmbeddingModality,
    EmbeddingSpace,
    EmbeddingVector,
)
from app.services.retrieval.errors import (
    InvalidRetrievalInputError,
    RetrievalConfigurationError,
    RetrievalError,
    VectorStoreError,
)
from app.services.retrieval.types import (
    RetrievalMatch,
    VectorRecord,
)
from chromadb.api.models.Collection import Collection

_COLLECTION_NAMES = {
    EmbeddingSpace.TEXT_SEMANTIC: "text_semantic_vectors",
    EmbeddingSpace.MULTIMODAL: "multimodal_vectors",
}


class ChromaVectorStore:
    """Persists normalized local embeddings in isolated collections."""

    def __init__(
        self,
        database_path: str | Path,
    ) -> None:
        path = Path(database_path).expanduser().resolve()

        if path.exists() and not path.is_dir():
            raise RetrievalConfigurationError(
                f"Chroma database path is not a directory: {path}"
            )

        try:
            path.mkdir(
                parents=True,
                exist_ok=True,
            )
            self._client = chromadb.PersistentClient(
                path=str(path),
            )
        except Exception as exc:
            raise RetrievalConfigurationError(
                f"Could not open Chroma database: {path}"
            ) from exc

        self._collections: dict[
            EmbeddingSpace,
            Collection,
        ] = {}

    def upsert(
        self,
        records: list[VectorRecord],
    ) -> None:
        if not records:
            return

        self._validate_records(records)

        grouped: dict[
            EmbeddingSpace,
            list[VectorRecord],
        ] = {}

        for record in records:
            grouped.setdefault(
                record.embedding.space,
                [],
            ).append(record)

        try:
            for space, space_records in grouped.items():
                collection = self._collection(space)

                collection.upsert(
                    ids=[
                        record.record_id
                        for record in space_records
                    ],
                    embeddings=[
                        list(record.embedding.values)
                        for record in space_records
                    ],
                    documents=[
                        record.content
                        for record in space_records
                    ],
                    metadatas=[
                        self._metadata(record)
                        for record in space_records
                    ],
                )
        except RetrievalError:
            raise
        except Exception as exc:
            raise VectorStoreError(
                f"Could not upsert vector records: {exc}"
            ) from exc

    def query(
        self,
        query: EmbeddingVector,
        *,
        top_k: int,
        modality: EmbeddingModality | None = None,
    ) -> list[RetrievalMatch]:
        if top_k <= 0:
            raise InvalidRetrievalInputError(
                "top_k must be greater than zero"
            )

        if not query.values:
            raise InvalidRetrievalInputError(
                "Query embedding cannot be empty"
            )

        if not query.model_id.strip():
            raise InvalidRetrievalInputError(
                "Query model_id cannot be empty"
            )

        where: dict[str, object] = {
            "model_id": query.model_id,
        }

        if modality is not None:
            where = {
                "$and": [
                    {
                        "model_id": query.model_id,
                    },
                    {
                        "modality": modality.value,
                    },
                ]
            }

        try:
            collection = self._collection(query.space)

            if collection.count() == 0:
                return []

            result = collection.query(
                query_embeddings=[
                    list(query.values)
                ],
                n_results=top_k,
                where=where,
                include=[
                    "documents",
                    "metadatas",
                    "distances",
                ],
            )

            return self._matches(result)

        except RetrievalError:
            raise
        except Exception as exc:
            raise VectorStoreError(
                f"Could not query vector records: {exc}"
            ) from exc

    def delete(
        self,
        record_ids: list[str],
        *,
        space: EmbeddingSpace,
    ) -> None:
        if not record_ids:
            return

        if any(
            not record_id.strip()
            for record_id in record_ids
        ):
            raise InvalidRetrievalInputError(
                "Record IDs cannot be empty"
            )

        try:
            self._collection(space).delete(
                ids=record_ids,
            )
        except RetrievalError:
            raise
        except Exception as exc:
            raise VectorStoreError(
                f"Could not delete vector records: {exc}"
            ) from exc

    def _collection(
        self,
        space: EmbeddingSpace,
    ) -> Collection:
        collection = self._collections.get(space)

        if collection is not None:
            return collection

        collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAMES[space],
            embedding_function=None,
            configuration={
                "hnsw": {
                    "space": "cosine",
                }
            },
        )

        self._collections[space] = collection
        return collection

    @staticmethod
    def _metadata(
        record: VectorRecord,
    ) -> dict[str, str]:
        embedding = record.embedding

        return {
            "content_id": embedding.content_id,
            "source_path": str(record.source_path),
            "source_sha256": record.source_sha256,
            "modality": embedding.modality.value,
            "space": embedding.space.value,
            "model_id": embedding.model_id,
        }

    @staticmethod
    def _validate_records(
        records: list[VectorRecord],
    ) -> None:
        seen: set[
            tuple[EmbeddingSpace, str]
        ] = set()

        for record in records:
            if not record.record_id.strip():
                raise InvalidRetrievalInputError(
                    "Record ID cannot be empty"
                )

            if not record.content.strip():
                raise InvalidRetrievalInputError(
                    "Stored content cannot be empty"
                )

            if not record.source_sha256.strip():
                raise InvalidRetrievalInputError(
                    "Source SHA-256 cannot be empty"
                )

            if not record.embedding.values:
                raise InvalidRetrievalInputError(
                    "Stored embedding cannot be empty"
                )

            if not record.embedding.model_id.strip():
                raise InvalidRetrievalInputError(
                    "Stored model_id cannot be empty"
                )

            key = (
                record.embedding.space,
                record.record_id,
            )

            if key in seen:
                raise InvalidRetrievalInputError(
                    "Duplicate record ID in one embedding space: "
                    f"{record.record_id}"
                )

            seen.add(key)

    @staticmethod
    def _matches(
        result: dict[str, object],
    ) -> list[RetrievalMatch]:
        ids = result.get("ids")
        documents = result.get("documents")
        metadatas = result.get("metadatas")
        distances = result.get("distances")

        if not (
            isinstance(ids, list)
            and isinstance(documents, list)
            and isinstance(metadatas, list)
            and isinstance(distances, list)
        ):
            raise VectorStoreError(
                "Chroma returned an incomplete query result"
            )

        if not ids:
            return []

        result_ids = ids[0]
        result_documents = documents[0]
        result_metadatas = metadatas[0]
        result_distances = distances[0]

        if not (
            len(result_ids)
            == len(result_documents)
            == len(result_metadatas)
            == len(result_distances)
        ):
            raise VectorStoreError(
                "Chroma returned misaligned query columns"
            )

        matches: list[RetrievalMatch] = []

        for (
            record_id,
            document,
            metadata,
            distance,
        ) in zip(
            result_ids,
            result_documents,
            result_metadatas,
            result_distances,
            strict=True,
        ):
            if (
                not isinstance(record_id, str)
                or not isinstance(document, str)
                or not isinstance(metadata, dict)
            ):
                raise VectorStoreError(
                    "Chroma returned an invalid record"
                )

            try:
                numeric_distance = float(distance)

                matches.append(
                    RetrievalMatch(
                        record_id=record_id,
                        content=document,
                        source_path=Path(
                            str(metadata["source_path"])
                        ),
                        source_sha256=str(
                            metadata["source_sha256"]
                        ),
                        content_id=str(
                            metadata["content_id"]
                        ),
                        modality=EmbeddingModality(
                            str(metadata["modality"])
                        ),
                        space=EmbeddingSpace(
                            str(metadata["space"])
                        ),
                        model_id=str(
                            metadata["model_id"]
                        ),
                        distance=numeric_distance,
                        similarity=(
                            1.0 - numeric_distance
                        ),
                    )
                )
            except (
                KeyError,
                TypeError,
                ValueError,
            ) as exc:
                raise VectorStoreError(
                    "Stored vector metadata is invalid"
                ) from exc

        return matches
