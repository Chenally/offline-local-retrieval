from __future__ import annotations

from pathlib import Path

import numpy as np
from app.services.embeddings.errors import (
    InvalidEmbeddingInputError,
)
from numpy.typing import NDArray
from PIL import Image, ImageOps


class ImagePreprocessor:
    """Creates model-ready float32 image batches."""

    def __init__(
        self,
        width: int,
        height: int,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
        channels_first: bool = False,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError(
                "Image dimensions must be greater than zero"
            )

        if any(value <= 0 for value in std):
            raise ValueError(
                "Image standard deviation must be positive"
            )

        self._size = (width, height)

        self._mean = np.asarray(
            mean,
            dtype=np.float32,
        ).reshape(1, 1, 3)

        self._std = np.asarray(
            std,
            dtype=np.float32,
        ).reshape(1, 1, 3)

        self._channels_first = channels_first

    def prepare_batch(
        self,
        image_paths: list[str | Path],
    ) -> NDArray[np.float32]:
        images = [
            self._prepare_image(
                Path(path).expanduser().resolve()
            )
            for path in image_paths
        ]

        if not images:
            width, height = self._size

            shape = (
                (0, 3, height, width)
                if self._channels_first
                else (0, height, width, 3)
            )

            return np.empty(
                shape,
                dtype=np.float32,
            )

        return np.stack(images).astype(
            np.float32,
            copy=False,
        )

    def _prepare_image(
        self,
        path: Path,
    ) -> NDArray[np.float32]:
        if not path.is_file():
            raise InvalidEmbeddingInputError(
                f"Image file does not exist: {path}"
            )

        try:
            with Image.open(path) as image:
                rgb_image = image.convert("RGB")

                resized = ImageOps.fit(
                    rgb_image,
                    self._size,
                    method=Image.Resampling.BILINEAR,
                )

                values = (
                    np.asarray(
                        resized,
                        dtype=np.float32,
                    )
                    / 255.0
                )

        except OSError as exc:
            raise InvalidEmbeddingInputError(
                f"Could not preprocess image "
                f"{path}: {exc}"
            ) from exc

        normalized = (
            values - self._mean
        ) / self._std

        if self._channels_first:
            normalized = np.transpose(
                normalized,
                (2, 0, 1),
            )

        return normalized.astype(
            np.float32,
            copy=False,
        )