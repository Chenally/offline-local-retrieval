from __future__ import annotations

from typing import Protocol

from app.services.embeddings.types import (
    EmbeddingModality,
    EmbeddingSpace,
    EmbeddingVector,
)
from app.services.retrieval.types import (
    RetrievalMatch,
    VectorRecord,
)


class VectorStore(Protocol):
    """Persistent storage boundary used by indexing and retrieval."""

    def upsert(
        self,
        records: list[VectorRecord],
    ) -> None:
        """Create or replace vector records by stable record ID."""
        ...

    def query(
        self,
        query: EmbeddingVector,
        *,
        top_k: int,
        modality: EmbeddingModality | None = None,
    ) -> list[RetrievalMatch]:
        """Return nearest records from the query embedding space."""
        ...

    def delete(
        self,
        record_ids: list[str],
        *,
        space: EmbeddingSpace,
    ) -> None:
        """Delete records from one explicit embedding space."""
        ...
