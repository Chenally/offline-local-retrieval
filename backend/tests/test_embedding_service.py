from pathlib import Path

import numpy as np
import pytest
from app.services.embeddings.base import EmbeddingInput
from app.services.embeddings.cache import EmbeddingCache
from app.services.embeddings.errors import (
    EmbeddingBackendError,
    EmbeddingConfigurationError,
    InvalidEmbeddingInputError,
    InvalidEmbeddingVectorError,
)
from app.services.embeddings.service import EmbeddingService
from app.services.embeddings.types import (
    EmbeddingModality,
    EmbeddingSpace,
)


class FakeBackend:
    def __init__(
        self,
        model_id: str,
        modality: EmbeddingModality,
        space: EmbeddingSpace,
    ) -> None:
        self.model_id = model_id
        self.modality = modality
        self.space = space
        self.batch_sizes: list[int] = []

    def embed_batch(
        self,
        inputs: list[EmbeddingInput],
    ) -> np.ndarray:
        self.batch_sizes.append(len(inputs))

        return np.asarray(
            [
                [
                    float(index + 1),
                    float(index + 2),
                ]
                for index, _ in enumerate(inputs)
            ],
            dtype=np.float32,
        )


def make_text_backend(
    space: EmbeddingSpace = (
        EmbeddingSpace.TEXT_SEMANTIC
    ),
) -> FakeBackend:
    return FakeBackend(
        model_id=f"text-{space.value}",
        modality=EmbeddingModality.TEXT,
        space=space,
    )


def test_embed_texts_batches_and_normalizes() -> None:
    backend = make_text_backend()
    service = EmbeddingService(
        [backend],
        batch_size=2,
    )

    results = service.embed_texts(
        ["first", "second", "third"]
    )

    assert backend.batch_sizes == [2, 1]
    assert len(results) == 3

    assert all(
        np.linalg.norm(result.values)
        == pytest.approx(1.0)
        for result in results
    )


def test_embed_texts_uses_cache() -> None:
    backend = make_text_backend()
    service = EmbeddingService(
        [backend],
        cache=EmbeddingCache(),
    )

    first = service.embed_texts(["same text"])
    second = service.embed_texts(["same text"])

    assert backend.batch_sizes == [1]
    assert first == second


def test_duplicate_inputs_are_inferred_once() -> None:
    backend = make_text_backend()
    service = EmbeddingService([backend])

    results = service.embed_texts(
        ["duplicate", "duplicate"]
    )

    assert backend.batch_sizes == [1]
    assert results[0] == results[1]


def test_routes_multimodal_text_and_image(
    tmp_path: Path,
) -> None:
    text_backend = make_text_backend(
        EmbeddingSpace.MULTIMODAL
    )

    image_backend = FakeBackend(
        model_id="mobileclip-image",
        modality=EmbeddingModality.IMAGE,
        space=EmbeddingSpace.MULTIMODAL,
    )

    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(b"image bytes")

    service = EmbeddingService(
        [text_backend, image_backend]
    )

    text_result = service.embed_texts(
        ["image query"],
        space=EmbeddingSpace.MULTIMODAL,
    )

    image_result = service.embed_images(
        [image_path]
    )

    assert (
        text_result[0].space
        is EmbeddingSpace.MULTIMODAL
    )

    assert (
        text_result[0].modality
        is EmbeddingModality.TEXT
    )

    assert (
        image_result[0].space
        is EmbeddingSpace.MULTIMODAL
    )

    assert (
        image_result[0].modality
        is EmbeddingModality.IMAGE
    )


def test_empty_batch_returns_empty_list() -> None:
    service = EmbeddingService(
        [make_text_backend()]
    )

    assert service.embed_texts([]) == []


@pytest.mark.parametrize(
    "text",
    ["", "   "],
)
def test_rejects_empty_text(text: str) -> None:
    service = EmbeddingService(
        [make_text_backend()]
    )

    with pytest.raises(
        InvalidEmbeddingInputError
    ):
        service.embed_texts([text])


def test_rejects_missing_image(
    tmp_path: Path,
) -> None:
    image_backend = FakeBackend(
        model_id="mobileclip-image",
        modality=EmbeddingModality.IMAGE,
        space=EmbeddingSpace.MULTIMODAL,
    )

    service = EmbeddingService(
        [image_backend]
    )

    with pytest.raises(
        InvalidEmbeddingInputError
    ):
        service.embed_images(
            [tmp_path / "missing.jpg"]
        )


def test_rejects_missing_backend() -> None:
    service = EmbeddingService(
        [make_text_backend()]
    )

    with pytest.raises(
        EmbeddingConfigurationError
    ):
        service.embed_texts(
            ["query"],
            space=EmbeddingSpace.MULTIMODAL,
        )


def test_rejects_duplicate_backend() -> None:
    with pytest.raises(
        EmbeddingConfigurationError
    ):
        EmbeddingService(
            [
                make_text_backend(),
                make_text_backend(),
            ]
        )


def test_rejects_empty_model_id() -> None:
    backend = make_text_backend()
    backend.model_id = " "

    with pytest.raises(
        EmbeddingConfigurationError
    ):
        EmbeddingService([backend])


def test_rejects_invalid_batch_size() -> None:
    with pytest.raises(
        ValueError,
        match="batch_size",
    ):
        EmbeddingService(
            [make_text_backend()],
            batch_size=0,
        )


def test_rejects_wrong_backend_output_shape() -> None:
    backend = make_text_backend()

    def wrong_shape(
        inputs: list[EmbeddingInput],
    ) -> np.ndarray:
        return np.ones(
            (len(inputs) + 1, 2),
            dtype=np.float32,
        )

    backend.embed_batch = wrong_shape  # type: ignore[method-assign]

    service = EmbeddingService([backend])

    with pytest.raises(
        InvalidEmbeddingVectorError
    ):
        service.embed_texts(["text"])


def test_normalizes_backend_failure() -> None:
    backend = make_text_backend()

    def fail(
        inputs: list[EmbeddingInput],
    ) -> np.ndarray:
        raise RuntimeError("model failed")

    backend.embed_batch = fail  # type: ignore[method-assign]

    service = EmbeddingService([backend])

    with pytest.raises(
        EmbeddingBackendError,
        match="model failed",
    ):
        service.embed_texts(["text"])