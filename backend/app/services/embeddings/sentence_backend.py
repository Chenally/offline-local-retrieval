from __future__ import annotations

from pathlib import Path

import numpy as np
from ai_edge_litert.interpreter import Interpreter
from app.services.embeddings.base import EmbeddingInput
from app.services.embeddings.bert_tokenizer import BertTokenizer
from app.services.embeddings.errors import (
    EmbeddingBackendError,
    EmbeddingConfigurationError,
    InvalidEmbeddingInputError,
)
from app.services.embeddings.text_preprocessor import TextPreprocessor
from app.services.embeddings.types import EmbeddingModality, EmbeddingSpace
from numpy.typing import NDArray

MINILM_MODEL_ID = "all_minilm_l6_v2_float32_v1"
_CONTEXT_LENGTH = 128


class SentenceEmbeddingBackend:
    """Runs a BERT-family sentence-embedding model exported to LiteRT.

    Expected tensor contract (produced by ``tools/convert_minilm_l6.ipynb``):

    - one input: ``int32[1, 128]`` WordPiece token IDs, padded with 0
    - one output: ``float32[1, D]`` sentence embedding

    The attention mask, mean pooling and normalization are computed inside
    the exported graph, so only token IDs cross the interpreter boundary.
    The embedding dimension ``D`` is read from the model.

    This replaces the MediaPipe ``bert_embedder`` used in Week 3, whose raw
    BERT pooler output did not separate related from unrelated sentences.
    """

    modality = EmbeddingModality.TEXT
    space = EmbeddingSpace.TEXT_SEMANTIC

    def __init__(
        self,
        model_path: str | Path,
        vocab_path: str | Path,
        model_id: str = MINILM_MODEL_ID,
    ) -> None:
        path = Path(model_path).expanduser().resolve()

        if not path.is_file():
            raise EmbeddingConfigurationError(
                f"Sentence embedding model does not exist: {path}"
            )

        if not model_id.strip():
            raise EmbeddingConfigurationError("model_id cannot be empty")

        self.model_id = model_id
        self._preprocessor = TextPreprocessor(
            tokenizer=BertTokenizer(vocab_path),
            max_length=_CONTEXT_LENGTH,
            pad_token_id=0,
        )

        try:
            self._interpreter = Interpreter(model_path=str(path))
            self._interpreter.allocate_tensors()
        except Exception as exc:
            raise EmbeddingConfigurationError(
                f"Could not load sentence embedding model: {path}"
            ) from exc

        inputs = self._interpreter.get_input_details()
        outputs = self._interpreter.get_output_details()

        if (
            len(inputs) != 1
            or tuple(inputs[0]["shape"]) != (1, _CONTEXT_LENGTH)
            or np.dtype(inputs[0]["dtype"]) != np.int32
            or len(outputs) != 1
            or len(outputs[0]["shape"]) != 2
            or int(outputs[0]["shape"][0]) != 1
            or int(outputs[0]["shape"][1]) <= 0
            or np.dtype(outputs[0]["dtype"]) != np.float32
        ):
            raise EmbeddingConfigurationError(
                "Unexpected sentence embedding model tensor configuration"
            )

        self._input_index = inputs[0]["index"]
        self._output_index = outputs[0]["index"]
        self.dimension = int(outputs[0]["shape"][1])

    def embed_batch(
        self,
        inputs: list[EmbeddingInput],
    ) -> NDArray[np.float32]:
        if any(not isinstance(value, str) for value in inputs):
            raise InvalidEmbeddingInputError(
                "Sentence embedding backend accepts strings only"
            )

        if not inputs:
            return np.empty((0, self.dimension), dtype=np.float32)

        batch = self._preprocessor.prepare_batch([str(value) for value in inputs])
        vectors: list[NDArray[np.float32]] = []

        for token_ids in batch.input_ids:
            try:
                self._interpreter.set_tensor(
                    self._input_index,
                    token_ids.reshape(1, _CONTEXT_LENGTH).astype(np.int32),
                )
                self._interpreter.invoke()
                output = self._interpreter.get_tensor(self._output_index)
            except Exception as exc:
                raise EmbeddingBackendError(
                    "Sentence embedding inference failed"
                ) from exc

            vectors.append(np.asarray(output[0], dtype=np.float32))

        return np.stack(vectors)
