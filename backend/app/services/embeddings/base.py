from __future__ import annotations

from pathlib import Path
from typing import Protocol, TypeAlias

import numpy as np
from app.services.embeddings.types import (
    EmbeddingModality,
    EmbeddingSpace,
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