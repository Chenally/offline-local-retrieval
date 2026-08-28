from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.services.embeddings.types import (
    EmbeddingModality,
    EmbeddingSpace,
    EmbeddingVector,
)


@dataclass(frozen=True, slots=True)
class VectorRecord:
    """One local content item ready for persistent vector storage."""

    record_id: str
    content: str
    source_path: Path
    source_sha256: str
    embedding: EmbeddingVector


@dataclass(frozen=True, slots=True)
class RetrievalMatch:
    """One similarity-search result returned by a vector store."""

    record_id: str
    content: str
    source_path: Path
    source_sha256: str
    content_id: str
    modality: EmbeddingModality
    space: EmbeddingSpace
    model_id: str
    distance: float
    similarity: float

@dataclass(frozen=True, slots=True)
class RetrievalCandidates:
    """Candidate matches collected from separate embedding spaces."""

    text_semantic: tuple[RetrievalMatch, ...]
    multimodal_text: tuple[RetrievalMatch, ...]
    multimodal_images: tuple[RetrievalMatch, ...]

@dataclass(frozen=True, slots=True)
class RankedResult:
    """A retrieval result with explainable component scores."""

    record_id: str
    content: str
    source_path: Path
    source_sha256: str
    modality: EmbeddingModality
    keyword_score: float
    text_semantic_score: float
    multimodal_score: float
    final_score: float