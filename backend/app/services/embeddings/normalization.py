from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from app.services.embeddings.errors import InvalidEmbeddingVectorError


def l2_normalize(
    vector: Sequence[float] | np.ndarray,
) -> tuple[float, ...]:
    """Validate and L2-normalize one embedding vector."""

    values = np.asarray(vector, dtype=np.float32)

    if values.ndim != 1 or values.size == 0:
        raise InvalidEmbeddingVectorError(
            "Embedding must be a non-empty one-dimensional vector"
        )

    if not np.isfinite(values).all():
        raise InvalidEmbeddingVectorError(
            "Embedding contains NaN or infinite values"
        )

    norm = float(np.linalg.norm(values))

    if norm == 0.0:
        raise InvalidEmbeddingVectorError(
            "Embedding cannot be normalized because its L2 norm is zero"
        )

    normalized = values / norm
    return tuple(float(value) for value in normalized)