from __future__ import annotations

from pathlib import Path

import numpy as np
from ai_edge_litert.interpreter import Interpreter
from app.services.embeddings.base import EmbeddingInput
from app.services.embeddings.errors import (
    EmbeddingBackendError,
    EmbeddingConfigurationError,
    InvalidEmbeddingInputError,
)
from app.services.embeddings.image_preprocessor import (
    ImagePreprocessor,
)
from app.services.embeddings.mobileclip_tokenizer import (
    MobileClipTokenizer,
)
from app.services.embeddings.text_preprocessor import (
    TextPreprocessor,
)
from app.services.embeddings.types import (
    EmbeddingModality,
    EmbeddingSpace,
)
from numpy.typing import NDArray

_MODEL_ID = "mobileclip_s0"
_EMBEDDING_DIMENSION = 512
_TEXT_LENGTH = 77
_IMAGE_SIZE = 256


def _load_interpreter(
    model_path: str | Path,
) -> Interpreter:
    path = Path(
        model_path
    ).expanduser().resolve()

    if not path.is_file():
        raise EmbeddingConfigurationError(
            f"Model file does not exist: {path}"
        )

    try:
        interpreter = Interpreter(
            model_path=str(path)
        )
        interpreter.allocate_tensors()
    except Exception as exc:
        raise EmbeddingConfigurationError(
            f"Could not load LiteRT model: {path}"
        ) from exc

    return interpreter


class MobileClipTextBackend:
    """Runs the MobileCLIP-S0 text encoder locally."""

    model_id = _MODEL_ID
    modality = EmbeddingModality.TEXT
    space = EmbeddingSpace.MULTIMODAL

    def __init__(
        self,
        model_path: str | Path,
        tokenizer_path: str | Path,
    ) -> None:
        self._interpreter = _load_interpreter(
            model_path
        )

        input_details = (
            self._interpreter.get_input_details()
        )
        output_details = (
            self._interpreter.get_output_details()
        )

        token_inputs = [
            detail
            for detail in input_details
            if tuple(detail["shape"]) == (1, _TEXT_LENGTH)
            and detail["dtype"] == np.int32
        ]

        position_inputs = [
            detail
            for detail in input_details
            if tuple(detail["shape"]) == (1,)
            and detail["dtype"] == np.int32
        ]

        if (
            len(token_inputs) != 1
            or len(position_inputs) != 1
            or len(output_details) != 1
            or tuple(output_details[0]["shape"])
            != (1, _EMBEDDING_DIMENSION)
        ):
            raise EmbeddingConfigurationError(
                "Unexpected MobileCLIP text model "
                "tensor configuration"
            )

        self._token_input_index = token_inputs[0][
            "index"
        ]
        self._position_input_index = (
            position_inputs[0]["index"]
        )
        self._output_index = output_details[0][
            "index"
        ]

        self._preprocessor = TextPreprocessor(
            tokenizer=MobileClipTokenizer(
                tokenizer_path
            ),
            max_length=_TEXT_LENGTH,
            pad_token_id=0,
        )

    def embed_batch(
        self,
        inputs: list[EmbeddingInput],
    ) -> NDArray[np.float32]:
        if any(
            not isinstance(value, str)
            for value in inputs
        ):
            raise InvalidEmbeddingInputError(
                "Text backend accepts strings only"
            )

        if not inputs:
            return np.empty(
                (0, _EMBEDDING_DIMENSION),
                dtype=np.float32,
            )

        batch = self._preprocessor.prepare_batch(
            inputs
        )

        vectors: list[NDArray[np.float32]] = []

        for input_ids in batch.input_ids:
            token_ids = input_ids.reshape(
                1,
                _TEXT_LENGTH,
            )

            eot_position = np.argmax(
                token_ids,
                axis=1,
            ).astype(np.int32)

            try:
                self._interpreter.set_tensor(
                    self._token_input_index,
                    token_ids,
                )
                self._interpreter.set_tensor(
                    self._position_input_index,
                    eot_position,
                )
                self._interpreter.invoke()

                output = self._interpreter.get_tensor(
                    self._output_index
                )
            except Exception as exc:
                raise EmbeddingBackendError(
                    "MobileCLIP text inference failed"
                ) from exc

            vectors.append(
                np.asarray(
                    output[0],
                    dtype=np.float32,
                )
            )

        return np.stack(vectors)


class MobileClipImageBackend:
    """Runs the MobileCLIP-S0 image encoder locally."""

    model_id = _MODEL_ID
    modality = EmbeddingModality.IMAGE
    space = EmbeddingSpace.MULTIMODAL

    def __init__(
        self,
        model_path: str | Path,
    ) -> None:
        self._interpreter = _load_interpreter(
            model_path
        )

        input_details = (
            self._interpreter.get_input_details()
        )
        output_details = (
            self._interpreter.get_output_details()
        )

        if (
            len(input_details) != 1
            or len(output_details) != 1
            or tuple(input_details[0]["shape"])
            != (1, 3, _IMAGE_SIZE, _IMAGE_SIZE)
            or input_details[0]["dtype"] != np.float32
            or tuple(output_details[0]["shape"])
            != (1, _EMBEDDING_DIMENSION)
        ):
            raise EmbeddingConfigurationError(
                "Unexpected MobileCLIP image model "
                "tensor configuration"
            )

        self._input_index = input_details[0][
            "index"
        ]
        self._output_index = output_details[0][
            "index"
        ]

        self._preprocessor = ImagePreprocessor(
            width=_IMAGE_SIZE,
            height=_IMAGE_SIZE,
            mean=(0.0, 0.0, 0.0),
            std=(1.0, 1.0, 1.0),
            channels_first=True,
        )

    def embed_batch(
        self,
        inputs: list[EmbeddingInput],
    ) -> NDArray[np.float32]:
        if not inputs:
            return np.empty(
                (0, _EMBEDDING_DIMENSION),
                dtype=np.float32,
            )

        images = self._preprocessor.prepare_batch(
            inputs
        )

        vectors: list[NDArray[np.float32]] = []

        for image in images:
            model_input = image.reshape(
                1,
                3,
                _IMAGE_SIZE,
                _IMAGE_SIZE,
            )

            try:
                self._interpreter.set_tensor(
                    self._input_index,
                    model_input,
                )
                self._interpreter.invoke()

                output = self._interpreter.get_tensor(
                    self._output_index
                )
            except Exception as exc:
                raise EmbeddingBackendError(
                    "MobileCLIP image inference failed"
                ) from exc

            vectors.append(
                np.asarray(
                    output[0],
                    dtype=np.float32,
                )
            )

        return np.stack(vectors)
