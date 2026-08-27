from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class EmbeddingModality(str, Enum):
    TEXT = "text"
    IMAGE = "image"


class EmbeddingSpace(str, Enum):
    TEXT_SEMANTIC = "text_semantic"
    MULTIMODAL = "multimodal"


@dataclass(frozen=True, slots=True)
class EmbeddingVector:
    """Normalized embedding returned by the unified service."""

    content_id: str
    values: tuple[float, ...]
    modality: EmbeddingModality
    space: EmbeddingSpace
    model_id: str