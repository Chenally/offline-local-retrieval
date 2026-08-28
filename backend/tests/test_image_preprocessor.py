from pathlib import Path

import numpy as np
import pytest
from app.services.embeddings.errors import (
    InvalidEmbeddingInputError,
)
from app.services.embeddings.image_preprocessor import (
    ImagePreprocessor,
)
from PIL import Image


def create_image(path: Path) -> None:
    Image.new(
        "RGB",
        (4, 2),
        color=(255, 128, 0),
    ).save(path)


def test_prepare_batch_returns_normalized_nhwc_tensor(
    tmp_path: Path,
) -> None:
    path = tmp_path / "image.png"
    create_image(path)

    preprocessor = ImagePreprocessor(
        width=2,
        height=2,
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
    )

    batch = preprocessor.prepare_batch([path])

    assert batch.shape == (1, 2, 2, 3)
    assert batch.dtype == np.float32

    assert batch[0, 0, 0] == pytest.approx(
        [1.0, 128.0 / 255.0, 0.0]
    )


def test_prepare_batch_supports_nchw_tensor(
    tmp_path: Path,
) -> None:
    path = tmp_path / "image.png"
    create_image(path)

    preprocessor = ImagePreprocessor(
        width=2,
        height=3,
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        channels_first=True,
    )

    batch = preprocessor.prepare_batch([path])

    assert batch.shape == (1, 3, 3, 2)


def test_prepare_batch_returns_empty_nhwc_tensor() -> None:
    preprocessor = ImagePreprocessor(
        width=2,
        height=3,
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
    )

    batch = preprocessor.prepare_batch([])

    assert batch.shape == (0, 3, 2, 3)


def test_rejects_missing_or_corrupted_image(
    tmp_path: Path,
) -> None:
    preprocessor = ImagePreprocessor(
        width=2,
        height=2,
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
    )

    with pytest.raises(
        InvalidEmbeddingInputError
    ):
        preprocessor.prepare_batch(
            [tmp_path / "missing.png"]
        )

    corrupted = tmp_path / "corrupted.png"
    corrupted.write_bytes(b"not an image")

    with pytest.raises(
        InvalidEmbeddingInputError
    ):
        preprocessor.prepare_batch([corrupted])


def test_rejects_invalid_image_configuration() -> None:
    with pytest.raises(
        ValueError,
        match="dimensions",
    ):
        ImagePreprocessor(
            width=0,
            height=2,
            mean=(0.0, 0.0, 0.0),
            std=(1.0, 1.0, 1.0),
        )

    with pytest.raises(
        ValueError,
        match="deviation",
    ):
        ImagePreprocessor(
            width=2,
            height=2,
            mean=(0.0, 0.0, 0.0),
            std=(1.0, 0.0, 1.0),
        )