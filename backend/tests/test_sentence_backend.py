from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("ai_edge_litert")

import app.services.embeddings.sentence_backend as backend_module  # noqa: E402
from app.services.embeddings.errors import (  # noqa: E402
    EmbeddingBackendError,
    EmbeddingConfigurationError,
    InvalidEmbeddingInputError,
)
from app.services.embeddings.sentence_backend import (  # noqa: E402
    MINILM_MODEL_ID,
    SentenceEmbeddingBackend,
)
from app.services.embeddings.types import EmbeddingSpace  # noqa: E402

DIMENSION = 384


class FakeInterpreter:
    """Returns [sum of token IDs, number of non-pad tokens, 0, ...]."""

    input_shape = (1, 128)
    input_dtype = np.int32
    output_shape = (1, DIMENSION)
    fail_load = False
    fail_invoke = False

    def __init__(self, model_path: str) -> None:
        if self.fail_load:
            raise RuntimeError("corrupt flatbuffer")
        self.values: dict[int, np.ndarray] = {}

    def allocate_tensors(self) -> None:
        pass

    def get_input_details(self) -> list[dict]:
        return [
            {
                "name": "args_0",
                "index": 0,
                "shape": np.asarray(self.input_shape),
                "dtype": self.input_dtype,
            }
        ]

    def get_output_details(self) -> list[dict]:
        return [
            {
                "name": "output_0",
                "index": 1,
                "shape": np.asarray(self.output_shape),
                "dtype": np.float32,
            }
        ]

    def set_tensor(self, index: int, value: np.ndarray) -> None:
        assert value.dtype == np.int32 and value.shape == (1, 128)
        self.values[index] = value

    def invoke(self) -> None:
        if self.fail_invoke:
            raise RuntimeError("delegate failure")
        tokens = self.values[0]
        output = np.zeros((1, DIMENSION), dtype=np.float32)
        output[0, 0] = tokens.sum()
        output[0, 1] = (tokens != 0).sum()
        self.values[1] = output

    def get_tensor(self, index: int) -> np.ndarray:
        return self.values[index]


@pytest.fixture
def assets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    model = tmp_path / "minilm.tflite"
    model.touch()
    vocab = tmp_path / "vocab.txt"
    vocab.write_text(
        "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nlocal\nretrieval\n",
        encoding="utf-8",
    )

    class Interpreter(FakeInterpreter):
        pass

    monkeypatch.setattr(backend_module, "Interpreter", Interpreter)
    return model, vocab


def test_embeds_each_text_and_reports_dimension(assets) -> None:
    backend = SentenceEmbeddingBackend(*assets)

    vectors = backend.embed_batch(["local", "local retrieval"])

    assert backend.model_id == MINILM_MODEL_ID
    assert backend.space is EmbeddingSpace.TEXT_SEMANTIC
    assert backend.dimension == DIMENSION
    assert vectors.shape == (2, DIMENSION)
    assert vectors.dtype == np.float32
    # [CLS]=2, local=5, retrieval=6, [SEP]=3
    assert vectors[:, 0].tolist() == [10.0, 16.0]
    assert vectors[:, 1].tolist() == [3.0, 4.0]


def test_empty_batch(assets) -> None:
    assert SentenceEmbeddingBackend(*assets).embed_batch([]).shape == (0, DIMENSION)


def test_rejects_non_text_input(assets, tmp_path: Path) -> None:
    with pytest.raises(InvalidEmbeddingInputError):
        SentenceEmbeddingBackend(*assets).embed_batch([tmp_path / "a.png"])


def test_rejects_missing_model(tmp_path: Path) -> None:
    with pytest.raises(EmbeddingConfigurationError):
        SentenceEmbeddingBackend(tmp_path / "missing.tflite", tmp_path / "vocab.txt")


def test_rejects_empty_model_id(assets) -> None:
    with pytest.raises(EmbeddingConfigurationError):
        SentenceEmbeddingBackend(*assets, model_id=" ")


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("input_shape", (1, 77)),
        ("input_dtype", np.int64),
        ("output_shape", (1, 4, DIMENSION)),
    ],
)
def test_rejects_unexpected_tensor_contract(
    assets, monkeypatch: pytest.MonkeyPatch, attribute: str, value: object
) -> None:
    monkeypatch.setattr(backend_module.Interpreter, attribute, value)

    with pytest.raises(EmbeddingConfigurationError, match="tensor configuration"):
        SentenceEmbeddingBackend(*assets)


def test_normalizes_load_and_inference_failures(
    assets, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(backend_module.Interpreter, "fail_load", True)

    with pytest.raises(EmbeddingConfigurationError, match="Could not load"):
        SentenceEmbeddingBackend(*assets)

    monkeypatch.setattr(backend_module.Interpreter, "fail_load", False)
    monkeypatch.setattr(backend_module.Interpreter, "fail_invoke", True)
    backend = SentenceEmbeddingBackend(*assets)

    with pytest.raises(EmbeddingBackendError):
        backend.embed_batch(["local"])
