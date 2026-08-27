from pathlib import Path

import pytest
from app.services.embeddings.bert_tokenizer import (
    BertTokenizer,
)
from app.services.embeddings.errors import (
    EmbeddingConfigurationError,
    InvalidEmbeddingInputError,
)


def _write_vocab(path: Path) -> None:
    path.write_text(
        (
            "[PAD]\n"
            "[UNK]\n"
            "[CLS]\n"
            "[SEP]\n"
            "[MASK]\n"
            "off\n"
            "##line\n"
            "local\n"
            "retrieval\n"
        ),
        encoding="utf-8",
    )


def test_encode_batch_uses_wordpiece_and_special_tokens(
    tmp_path: Path,
) -> None:
    vocab_path = tmp_path / "vocab.txt"
    _write_vocab(vocab_path)

    tokenizer = BertTokenizer(vocab_path)

    assert tokenizer.encode_batch(
        ["Offline local retrieval"],
        max_length=8,
    ) == [[2, 5, 6, 7, 8, 3]]


def test_encode_batch_truncates_and_preserves_sep(
    tmp_path: Path,
) -> None:
    vocab_path = tmp_path / "vocab.txt"
    _write_vocab(vocab_path)

    tokenizer = BertTokenizer(vocab_path)

    token_ids = tokenizer.encode_batch(
        ["local local local local"],
        max_length=4,
    )[0]

    assert token_ids == [2, 7, 7, 3]


def test_rejects_missing_vocabulary(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        EmbeddingConfigurationError
    ):
        BertTokenizer(
            tmp_path / "missing.txt"
        )


def test_rejects_too_small_max_length(
    tmp_path: Path,
) -> None:
    vocab_path = tmp_path / "vocab.txt"
    _write_vocab(vocab_path)

    tokenizer = BertTokenizer(vocab_path)

    with pytest.raises(
        InvalidEmbeddingInputError
    ):
        tokenizer.encode_batch(
            ["local"],
            max_length=1,
        )
