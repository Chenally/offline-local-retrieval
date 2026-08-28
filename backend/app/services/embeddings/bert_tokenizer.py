from __future__ import annotations

from pathlib import Path

from app.services.embeddings.errors import (
    EmbeddingConfigurationError,
    InvalidEmbeddingInputError,
)
from tokenizers import BertWordPieceTokenizer


class BertTokenizer:
    """WordPiece tokenizer matched to the BERT model vocabulary."""

    def __init__(
        self,
        vocab_path: str | Path,
    ) -> None:
        path = Path(
            vocab_path
        ).expanduser().resolve()

        if not path.is_file():
            raise EmbeddingConfigurationError(
                f"BERT vocabulary does not exist: {path}"
            )

        self._tokenizer = BertWordPieceTokenizer(
            str(path),
            lowercase=True,
        )

    def encode_batch(
        self,
        texts: list[str],
        max_length: int,
    ) -> list[list[int]]:
        if max_length < 2:
            raise InvalidEmbeddingInputError(
                "BERT max_length must be at least two"
            )

        self._tokenizer.enable_truncation(
            max_length=max_length,
        )

        return [
            encoding.ids
            for encoding in self._tokenizer.encode_batch(
                texts
            )
        ]
