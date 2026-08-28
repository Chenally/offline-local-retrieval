from pathlib import Path

import numpy as np
import pytest
from app.services.embeddings.bert_backend import (
    BertLiteRTBackend,
)
from app.services.embeddings.errors import (
    EmbeddingConfigurationError,
    InvalidEmbeddingInputError,
)


class FakeInterpreter:
    def __init__(
        self,
        model_path: str,
    ) -> None:
        self.values: dict[
            int,
            np.ndarray,
        ] = {}
        self.batch_size = 1

    def allocate_tensors(self) -> None:
        pass

    def get_input_details(
        self,
    ) -> list[dict[str, object]]:
        return [
            self._details(
                "input_word_ids",
                0,
                128,
                np.int32,
            ),
            self._details(
                "input_mask",
                1,
                128,
                np.int32,
            ),
            self._details(
                "input_type_ids",
                2,
                128,
                np.int32,
            ),
        ]

    def get_output_details(
        self,
    ) -> list[dict[str, object]]:
        return [
            self._details(
                (
                    "module_apply_tokens/bert/"
                    "pooler/Squeeze"
                ),
                3,
                512,
                np.float32,
            )
        ]

    def resize_tensor_input(
        self,
        index: int,
        shape: list[int],
        strict: bool,
    ) -> None:
        self.batch_size = shape[0]

    def set_tensor(
        self,
        index: int,
        value: np.ndarray,
    ) -> None:
        self.values[index] = value

    def invoke(self) -> None:
        vectors = np.zeros(
            (
                self.batch_size,
                512,
            ),
            dtype=np.float32,
        )

        vectors[:, 0] = self.values[
            0
        ].sum(axis=1)
        vectors[:, 1] = self.values[
            1
        ].sum(axis=1)

        self.values[3] = vectors

    def get_tensor(
        self,
        index: int,
    ) -> np.ndarray:
        return self.values[index]

    def _details(
        self,
        name: str,
        index: int,
        width: int,
        dtype: type[np.generic],
    ) -> dict[str, object]:
        return {
            "name": name,
            "index": index,
            "shape": np.asarray(
                [
                    self.batch_size,
                    width,
                ]
            ),
            "shape_signature": np.asarray(
                [-1, width]
            ),
            "dtype": dtype,
        }


def _write_assets(
    tmp_path: Path,
) -> tuple[Path, Path]:
    model_path = tmp_path / "model.tflite"
    model_path.touch()

    vocab_path = tmp_path / "vocab.txt"
    vocab_path.write_text(
        (
            "[PAD]\n"
            "[UNK]\n"
            "[CLS]\n"
            "[SEP]\n"
            "[MASK]\n"
            "local\n"
            "retrieval\n"
        ),
        encoding="utf-8",
    )

    return model_path, vocab_path


def test_embed_batch_runs_matched_tensors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path, vocab_path = _write_assets(
        tmp_path
    )

    monkeypatch.setattr(
        (
            "app.services.embeddings."
            "bert_backend.Interpreter"
        ),
        FakeInterpreter,
    )

    backend = BertLiteRTBackend(
        model_path,
        vocab_path,
    )

    vectors = backend.embed_batch(
        [
            "local",
            "local retrieval",
        ]
    )

    assert vectors.shape == (2, 512)
    assert vectors.dtype == np.float32
    assert vectors[:, 1].tolist() == [
        3.0,
        4.0,
    ]


def test_embed_batch_rejects_non_text_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path, vocab_path = _write_assets(
        tmp_path
    )

    monkeypatch.setattr(
        (
            "app.services.embeddings."
            "bert_backend.Interpreter"
        ),
        FakeInterpreter,
    )

    backend = BertLiteRTBackend(
        model_path,
        vocab_path,
    )

    with pytest.raises(
        InvalidEmbeddingInputError
    ):
        backend.embed_batch(
            [tmp_path / "image.jpg"]
        )


def test_rejects_missing_model(
    tmp_path: Path,
) -> None:
    vocab_path = tmp_path / "vocab.txt"
    vocab_path.touch()

    with pytest.raises(
        EmbeddingConfigurationError
    ):
        BertLiteRTBackend(
            tmp_path / "missing.tflite",
            vocab_path,
        )
