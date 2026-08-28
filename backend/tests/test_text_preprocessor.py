import numpy as np
import pytest
from app.services.embeddings.errors import (
    InvalidEmbeddingInputError,
)
from app.services.embeddings.text_preprocessor import (
    TextPreprocessor,
)


class FakeTokenizer:
    def encode_batch(
        self,
        texts: list[str],
        max_length: int,
    ) -> list[list[int]]:
        return [
            [101, len(text), 102]
            for text in texts
        ]


def test_prepare_batch_pads_text_tensors() -> None:
    preprocessor = TextPreprocessor(
        tokenizer=FakeTokenizer(),
        max_length=5,
    )

    batch = preprocessor.prepare_batch(
        ["cat", "document"]
    )

    assert batch.input_ids.tolist() == [
        [101, 3, 102, 0, 0],
        [101, 8, 102, 0, 0],
    ]

    assert batch.attention_mask.tolist() == [
        [1, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
    ]

    assert (
        np.count_nonzero(
            batch.token_type_ids
        )
        == 0
    )

    assert batch.input_ids.dtype == np.int32


def test_prepare_batch_truncates_token_ids() -> None:
    preprocessor = TextPreprocessor(
        tokenizer=FakeTokenizer(),
        max_length=2,
    )

    batch = preprocessor.prepare_batch(["cat"])

    assert batch.input_ids.tolist() == [
        [101, 3]
    ]

    assert batch.attention_mask.tolist() == [
        [1, 1]
    ]


@pytest.mark.parametrize(
    "text",
    ["", "   "],
)
def test_prepare_batch_rejects_empty_text(
    text: str,
) -> None:
    preprocessor = TextPreprocessor(
        tokenizer=FakeTokenizer(),
        max_length=5,
    )

    with pytest.raises(
        InvalidEmbeddingInputError
    ):
        preprocessor.prepare_batch([text])


def test_rejects_invalid_max_length() -> None:
    with pytest.raises(
        ValueError,
        match="max_length",
    ):
        TextPreprocessor(
            tokenizer=FakeTokenizer(),
            max_length=0,
        )


def test_rejects_wrong_tokenizer_batch_size() -> None:
    class BrokenTokenizer:
        def encode_batch(
            self,
            texts: list[str],
            max_length: int,
        ) -> list[list[int]]:
            return []

    preprocessor = TextPreprocessor(
        tokenizer=BrokenTokenizer(),
        max_length=5,
    )

    with pytest.raises(
        InvalidEmbeddingInputError
    ):
        preprocessor.prepare_batch(["text"])