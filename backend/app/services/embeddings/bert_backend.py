from __future__ import annotations

from pathlib import Path

import numpy as np
from ai_edge_litert.interpreter import Interpreter
from app.services.embeddings.base import EmbeddingInput
from app.services.embeddings.bert_tokenizer import (
    BertTokenizer,
)
from app.services.embeddings.errors import (
    EmbeddingBackendError,
    EmbeddingConfigurationError,
    InvalidEmbeddingInputError,
)
from app.services.embeddings.text_preprocessor import (
    TextPreprocessor,
)
from app.services.embeddings.types import (
    EmbeddingModality,
    EmbeddingSpace,
)
from numpy.typing import NDArray

_MODEL_ID = "bert_embedder_float32_v1"
_EMBEDDING_DIMENSION = 512
_CONTEXT_LENGTH = 128


class BertLiteRTBackend:
    """Runs the BERT semantic text embedder with LiteRT."""

    model_id = _MODEL_ID
    modality = EmbeddingModality.TEXT
    space = EmbeddingSpace.TEXT_SEMANTIC

    def __init__(
        self,
        model_path: str | Path,
        vocab_path: str | Path,
    ) -> None:
        path = Path(
            model_path
        ).expanduser().resolve()

        if not path.is_file():
            raise EmbeddingConfigurationError(
                f"BERT model does not exist: {path}"
            )

        self._preprocessor = TextPreprocessor(
            tokenizer=BertTokenizer(
                vocab_path
            ),
            max_length=_CONTEXT_LENGTH,
            pad_token_id=0,
        )

        try:
            self._interpreter = Interpreter(
                model_path=str(path),
            )
            self._interpreter.allocate_tensors()
            self._refresh_tensor_details()
            self._validate_model()
        except EmbeddingConfigurationError:
            raise
        except Exception as exc:
            raise EmbeddingConfigurationError(
                "Could not load BERT LiteRT model"
            ) from exc

    def embed_batch(
        self,
        inputs: list[EmbeddingInput],
    ) -> NDArray[np.float32]:
        texts: list[str] = []

        for value in inputs:
            if not isinstance(value, str):
                raise InvalidEmbeddingInputError(
                    "BERT backend accepts strings only"
                )

            texts.append(value)

        if not texts:
            return np.empty(
                (
                    0,
                    _EMBEDDING_DIMENSION,
                ),
                dtype=np.float32,
            )

        batch = self._preprocessor.prepare_batch(
            texts
        )

        try:
            self._resize_batch(len(texts))

            self._interpreter.set_tensor(
                self._inputs[
                    "input_word_ids"
                ]["index"],
                batch.input_ids,
            )
            self._interpreter.set_tensor(
                self._inputs[
                    "input_mask"
                ]["index"],
                batch.attention_mask,
            )
            self._interpreter.set_tensor(
                self._inputs[
                    "input_type_ids"
                ]["index"],
                batch.token_type_ids,
            )

            self._interpreter.invoke()

            vectors = np.asarray(
                self._interpreter.get_tensor(
                    self._output["index"]
                ),
                dtype=np.float32,
            )
        except Exception as exc:
            raise EmbeddingBackendError(
                "BERT LiteRT inference failed"
            ) from exc

        expected_shape = (
            len(texts),
            _EMBEDDING_DIMENSION,
        )

        if vectors.shape != expected_shape:
            raise EmbeddingBackendError(
                "BERT LiteRT returned an "
                "unexpected output shape"
            )

        return vectors

    def _resize_batch(
        self,
        batch_size: int,
    ) -> None:
        current_size = int(
            self._inputs[
                "input_word_ids"
            ]["shape"][0]
        )

        if current_size == batch_size:
            return

        for details in self._inputs.values():
            self._interpreter.resize_tensor_input(
                details["index"],
                [
                    batch_size,
                    _CONTEXT_LENGTH,
                ],
                strict=True,
            )

        self._interpreter.allocate_tensors()
        self._refresh_tensor_details()

    def _refresh_tensor_details(
        self,
    ) -> None:
        self._inputs = {
            details["name"]: details
            for details
            in self._interpreter.get_input_details()
        }

        outputs = {
            details["name"]: details
            for details
            in self._interpreter.get_output_details()
        }

        output_name = (
            "module_apply_tokens/bert/"
            "pooler/Squeeze"
        )

        if output_name not in outputs:
            raise EmbeddingConfigurationError(
                "BERT output tensor is missing: "
                f"{output_name}"
            )

        self._output = outputs[output_name]

    def _validate_model(self) -> None:
        expected_inputs = {
            "input_word_ids",
            "input_mask",
            "input_type_ids",
        }

        if set(self._inputs) != expected_inputs:
            raise EmbeddingConfigurationError(
                "BERT model input tensors do "
                "not match the expected signature"
            )

        for details in self._inputs.values():
            if (
                details[
                    "shape_signature"
                ].tolist()
                != [
                    -1,
                    _CONTEXT_LENGTH,
                ]
                or details["dtype"]
                is not np.int32
            ):
                raise EmbeddingConfigurationError(
                    "BERT model input tensor has "
                    "an unexpected signature"
                )

        if (
            self._output[
                "shape_signature"
            ].tolist()
            != [
                -1,
                _EMBEDDING_DIMENSION,
            ]
            or self._output["dtype"]
            is not np.float32
        ):
            raise EmbeddingConfigurationError(
                "BERT model output tensor has "
                "an unexpected signature"
            )
