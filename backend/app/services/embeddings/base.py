from __future__ import annotations

from pathlib import Path
from typing import Protocol, TypeAlias

import numpy as np
from app.services.embeddings.types import (
    EmbeddingModality,
    EmbeddingSpace,
    EmbeddingVector,
)
from numpy.typing import NDArray

EmbeddingInput: TypeAlias = str | Path


class EmbeddingBackend(Protocol):
    """Interface implemented by every local embedding backend."""

    model_id: str
    modality: EmbeddingModality
    space: EmbeddingSpace

    def embed_batch(
        self,
        inputs: list[EmbeddingInput],
    ) -> NDArray[np.float32]:
        """Return one row per input and one column per dimension."""
        ...

class Embedder(Protocol):
    """Text and image embedding operations used by indexing and search.

    Implemented by ``EmbeddingService`` and ``SynchronizedEmbeddingService``.
    """

    def embed_texts(
        self,
        texts: list[str],
        space: EmbeddingSpace = EmbeddingSpace.TEXT_SEMANTIC,
    ) -> list[EmbeddingVector]: ...

    def embed_images(
        self,
        image_paths: list[str | Path],
    ) -> list[EmbeddingVector]: ...
