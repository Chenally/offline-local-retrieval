from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from app.services.embeddings.errors import (
    InvalidEmbeddingInputError,
)
from numpy.typing import NDArray


class TextTokenizer(Protocol):
    """Interface implemented by a model-compatible tokenizer."""

    def encode_batch(
        self,
        texts: list[str],
        max_length: int,
    ) -> list[list[int]]:
        """Return token IDs including required special tokens."""
        ...


@dataclass(frozen=True, slots=True)
class TextBatch:
    input_ids: NDArray[np.int32]
    attention_mask: NDArray[np.int32]
    token_type_ids: NDArray[np.int32]


class TextPreprocessor:
    """Builds fixed-length integer tensors from local text."""

    def __init__(
        self,
        tokenizer: TextTokenizer,
        max_length: int,
        pad_token_id: int = 0,
    ) -> None:
        if max_length <= 0:
            raise ValueError(
                "max_length must be greater than zero"
            )

        self._tokenizer = tokenizer
        self._max_length = max_length
        self._pad_token_id = pad_token_id

    def prepare_batch(
        self,
        texts: list[str],
    ) -> TextBatch:
        if any(not text.strip() for text in texts):
            raise InvalidEmbeddingInputError(
                "Text preprocessing requires non-empty strings"
            )

        encoded = self._tokenizer.encode_batch(
            texts,
            self._max_length,
        )

        if len(encoded) != len(texts):
            raise InvalidEmbeddingInputError(
                "Tokenizer returned an unexpected "
                "number of sequences"
            )

        input_ids = np.full(
            (len(texts), self._max_length),
            self._pad_token_id,
            dtype=np.int32,
        )

        attention_mask = np.zeros_like(input_ids)

        for index, token_ids in enumerate(encoded):
            length = min(
                len(token_ids),
                self._max_length,
            )

            input_ids[
                index,
                :length,
            ] = token_ids[:length]

            attention_mask[
                index,
                :length,
            ] = 1

        return TextBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=np.zeros_like(input_ids),
        )