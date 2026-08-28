from pathlib import Path

import app.services.embeddings.mobileclip_backend as backend_module
import numpy as np
import pytest
from app.services.embeddings.errors import (
    EmbeddingBackendError,
    EmbeddingConfigurationError,
    InvalidEmbeddingInputError,
)
from app.services.embeddings.types import (
    EmbeddingModality,
    EmbeddingSpace,
)
from PIL import Image


class FakeTokenizer:
    def encode_batch(
        self,
        texts: list[str],
        max_length: int,
    ) -> list[list[int]]:
        assert max_length == 77

        return [
            [
                49406,
                10 + index,
                49407,
            ]
            for index, _ in enumerate(texts)
        ]


class FakeInterpreter:
    def __init__(
        self,
        kind: str,
        *,
        fail_inference: bool = False,
    ) -> None:
        self.kind = kind
        self.fail_inference = fail_inference
        self.values: dict[int, np.ndarray] = {}

    def allocate_tensors(self) -> None:
        pass

    def get_input_details(
        self,
    ) -> list[dict[str, object]]:
        if self.kind == "text":
            return [
                self._details(
                    "token_ids",
                    0,
                    [1, 77],
                    np.int32,
                ),
                self._details(
                    "eot_position",
                    1,
                    [1],
                    np.int32,
                ),
            ]

        return [
            self._details(
                "image",
                0,
                [1, 3, 256, 256],
                np.float32,
            )
        ]

    def get_output_details(
        self,
    ) -> list[dict[str, object]]:
        output_index = (
            2 if self.kind == "text" else 1
        )

        return [
            self._details(
                "embedding",
                output_index,
                [1, 512],
                np.float32,
            )
        ]

    def set_tensor(
        self,
        index: int,
        value: np.ndarray,
    ) -> None:
        self.values[index] = value

    def invoke(self) -> None:
        if self.fail_inference:
            raise RuntimeError("inference failed")

        output = np.zeros(
            (1, 512),
            dtype=np.float32,
        )

        if self.kind == "text":
            output[0, 0] = self.values[
                0
            ].sum()
            output[0, 1] = self.values[
                1
            ][0]
            self.values[2] = output
        else:
            output[0, 0] = self.values[
                0
            ].sum()
            self.values[1] = output

    def get_tensor(
        self,
        index: int,
    ) -> np.ndarray:
        return self.values[index]

    @staticmethod
    def _details(
        name: str,
        index: int,
        shape: list[int],
        dtype: object,
    ) -> dict[str, object]:
        return {
            "name": name,
            "index": index,
            "shape": np.asarray(
                shape,
                dtype=np.int32,
            ),
            "dtype": dtype,
        }


class FakeImageInterpreter(
    FakeInterpreter
):
    def __init__(
        self,
        model_path: str,
    ) -> None:
        super().__init__("image")


def _patch_text_backend(
    monkeypatch: pytest.MonkeyPatch,
    interpreter: FakeInterpreter,
) -> None:
    monkeypatch.setattr(
        backend_module,
        "_load_interpreter",
        lambda model_path: interpreter,
    )
    monkeypatch.setattr(
        backend_module,
        "MobileClipTokenizer",
        lambda tokenizer_path: FakeTokenizer(),
    )


def test_text_backend_embeds_multiple_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_text_backend(
        monkeypatch,
        FakeInterpreter("text"),
    )

    backend = (
        backend_module.MobileClipTextBackend(
            model_path="text.tflite",
            tokenizer_path="tokenizer.json",
        )
    )

    vectors = backend.embed_batch(
        ["first text", "second text"]
    )

    assert vectors.shape == (2, 512)
    assert vectors.dtype == np.float32
    assert vectors[:, 1].tolist() == [
        2.0,
        2.0,
    ]
    assert not np.array_equal(
        vectors[0],
        vectors[1],
    )
    assert (
        backend.modality
        == EmbeddingModality.TEXT
    )
    assert (
        backend.space
        == EmbeddingSpace.MULTIMODAL
    )


def test_image_backend_embeds_multiple_images(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "image.tflite"
    model_path.touch()

    monkeypatch.setattr(
        backend_module,
        "Interpreter",
        FakeImageInterpreter,
    )

    black_path = tmp_path / "black.png"
    white_path = tmp_path / "white.png"

    Image.new(
        "RGB",
        (32, 24),
        color=(0, 0, 0),
    ).save(black_path)

    Image.new(
        "RGB",
        (32, 24),
        color=(255, 255, 255),
    ).save(white_path)

    backend = (
        backend_module.MobileClipImageBackend(
            model_path=model_path
        )
    )

    vectors = backend.embed_batch(
        [black_path, white_path]
    )

    assert vectors.shape == (2, 512)
    assert vectors.dtype == np.float32
    assert not np.array_equal(
        vectors[0],
        vectors[1],
    )
    assert (
        backend.modality
        == EmbeddingModality.IMAGE
    )
    assert (
        backend.space
        == EmbeddingSpace.MULTIMODAL
    )


def test_backends_return_empty_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def load_interpreter(
        model_path: str,
    ) -> FakeInterpreter:
        kind = (
            "text"
            if "text" in str(model_path)
            else "image"
        )
        return FakeInterpreter(kind)

    monkeypatch.setattr(
        backend_module,
        "_load_interpreter",
        load_interpreter,
    )
    monkeypatch.setattr(
        backend_module,
        "MobileClipTokenizer",
        lambda tokenizer_path: FakeTokenizer(),
    )

    text_backend = (
        backend_module.MobileClipTextBackend(
            "text.tflite",
            "tokenizer.json",
        )
    )
    image_backend = (
        backend_module.MobileClipImageBackend(
            "image.tflite"
        )
    )

    assert text_backend.embed_batch(
        []
    ).shape == (0, 512)
    assert image_backend.embed_batch(
        []
    ).shape == (0, 512)


def test_text_backend_rejects_non_text_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_text_backend(
        monkeypatch,
        FakeInterpreter("text"),
    )

    backend = (
        backend_module.MobileClipTextBackend(
            "text.tflite",
            "tokenizer.json",
        )
    )

    with pytest.raises(
        InvalidEmbeddingInputError
    ):
        backend.embed_batch(
            [tmp_path / "image.png"]
        )


def test_rejects_missing_model(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        EmbeddingConfigurationError
    ):
        backend_module.MobileClipImageBackend(
            tmp_path / "missing.tflite"
        )


def test_normalizes_interpreter_load_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "broken.tflite"
    model_path.touch()

    class FailingInterpreter:
        def __init__(
            self,
            model_path: str,
        ) -> None:
            raise RuntimeError("load failed")

    monkeypatch.setattr(
        backend_module,
        "Interpreter",
        FailingInterpreter,
    )

    with pytest.raises(
        EmbeddingConfigurationError
    ):
        backend_module.MobileClipImageBackend(
            model_path
        )


def test_rejects_unexpected_tensor_signatures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        backend_module,
        "_load_interpreter",
        lambda model_path: FakeInterpreter(
            "image"
        ),
    )

    with pytest.raises(
        EmbeddingConfigurationError
    ):
        backend_module.MobileClipTextBackend(
            "text.tflite",
            "tokenizer.json",
        )

    monkeypatch.setattr(
        backend_module,
        "_load_interpreter",
        lambda model_path: FakeInterpreter(
            "text"
        ),
    )

    with pytest.raises(
        EmbeddingConfigurationError
    ):
        backend_module.MobileClipImageBackend(
            "image.tflite"
        )


def test_normalizes_inference_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_text_backend(
        monkeypatch,
        FakeInterpreter(
            "text",
            fail_inference=True,
        ),
    )

    text_backend = (
        backend_module.MobileClipTextBackend(
            "text.tflite",
            "tokenizer.json",
        )
    )

    with pytest.raises(
        EmbeddingBackendError
    ):
        text_backend.embed_batch(["text"])

    monkeypatch.setattr(
        backend_module,
        "_load_interpreter",
        lambda model_path: FakeInterpreter(
            "image",
            fail_inference=True,
        ),
    )

    image_path = tmp_path / "image.png"
    Image.new(
        "RGB",
        (8, 8),
    ).save(image_path)

    image_backend = (
        backend_module.MobileClipImageBackend(
            "image.tflite"
        )
    )

    with pytest.raises(
        EmbeddingBackendError
    ):
        image_backend.embed_batch(
            [image_path]
        )
