from __future__ import annotations

from pathlib import Path

from app.services.embeddings.errors import (
    EmbeddingConfigurationError,
    InvalidEmbeddingInputError,
)
from tokenizers import Tokenizer


class MobileClipTokenizer:
    """Loads the tokenizer bundled for MobileCLIP-S0."""

    def __init__(
        self,
        tokenizer_path: str | Path,
    ) -> None:
        path = Path(
            tokenizer_path
        ).expanduser().resolve()

        if not path.is_file():
            raise EmbeddingConfigurationError(
                f"Tokenizer file does not exist: {path}"
            )

        try:
            self._tokenizer = Tokenizer.from_file(
                str(path)
            )
        except Exception as exc:
            raise EmbeddingConfigurationError(
                f"Could not load tokenizer: {path}"
            ) from exc

        eot_token_id = self._tokenizer.token_to_id(
            "<|endoftext|>"
        )

        if eot_token_id is None:
            raise EmbeddingConfigurationError(
                "Tokenizer does not contain "
                "<|endoftext|>"
            )

        self._eot_token_id = eot_token_id

    def encode_batch(
        self,
        texts: list[str],
        max_length: int,
    ) -> list[list[int]]:
        if max_length < 2:
            raise ValueError(
                "max_length must be at least two"
            )

        if any(not text.strip() for text in texts):
            raise InvalidEmbeddingInputError(
                "Tokenization requires non-empty strings"
            )

        try:
            encodings = self._tokenizer.encode_batch(
                texts
            )
        except Exception as exc:
            raise InvalidEmbeddingInputError(
                "Could not tokenize text"
            ) from exc

        token_batches: list[list[int]] = []

        for encoding in encodings:
            token_ids = encoding.ids

            if len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
                token_ids[-1] = self._eot_token_id

            token_batches.append(token_ids)

        return token_batches
