import numpy as np
import pytest
from app.services.embeddings.errors import InvalidEmbeddingVectorError
from app.services.embeddings.normalization import l2_normalize


def test_l2_normalize_returns_unit_vector() -> None:
    result = l2_normalize([3.0, 4.0])

    assert result == pytest.approx((0.6, 0.8))
    assert np.linalg.norm(result) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "vector",
    (
        [],
        [[1.0, 2.0]],
        [0.0, 0.0],
        [float("nan"), 1.0],
        [float("inf"), 1.0],
    ),
)
def test_l2_normalize_rejects_invalid_vector(
    vector: list[float] | list[list[float]],
) -> None:
    with pytest.raises(InvalidEmbeddingVectorError):
        l2_normalize(vector)