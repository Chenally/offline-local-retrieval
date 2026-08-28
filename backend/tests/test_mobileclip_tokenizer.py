from pathlib import Path

import app.services.embeddings.mobileclip_tokenizer as tokenizer_module
import pytest
from app.services.embeddings.errors import (
    EmbeddingConfigurationError,
    InvalidEmbeddingInputError,
)
from app.services.embeddings.mobileclip_tokenizer import (
    MobileClipTokenizer,
)


class FakeEncoding:
    def __init__(self, token_ids: list[int]) -> None:
        self.ids = token_ids


class FakeRuntimeTokenizer:
    def __init__(
        self,
        *,
        eot_token_id: int | None = 49407,
        fail_encoding: bool = False,
    ) -> None:
        self._eot_token_id = eot_token_id
        self._fail_encoding = fail_encoding

    def token_to_id(
        self,
        token: str,
    ) -> int | None:
        if token == "<|endoftext|>":
            return self._eot_token_id

        return None

    def encode_batch(
        self,
        texts: list[str],
    ) -> list[FakeEncoding]:
        if self._fail_encoding:
            raise RuntimeError("tokenization failed")

        encodings = {
            "short": [49406, 1, 49407],
            "long": [
                49406,
                1,
                2,
                3,
                4,
                49407,
            ],
        }

        return [
            FakeEncoding(encodings[text])
            for text in texts
        ]


def _write_tokenizer_asset(
    tmp_path: Path,
) -> Path:
    path = tmp_path / "tokenizer.json"
    path.write_text("{}", encoding="utf-8")
    return path


def _patch_loader(
    monkeypatch: pytest.MonkeyPatch,
    runtime_tokenizer: FakeRuntimeTokenizer,
) -> None:
    class FakeLoader:
        @staticmethod
        def from_file(
            path: str,
        ) -> FakeRuntimeTokenizer:
            return runtime_tokenizer

    monkeypatch.setattr(
        tokenizer_module,
        "Tokenizer",
        FakeLoader,
    )


def test_encode_batch_preserves_eot_when_truncated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write_tokenizer_asset(tmp_path)
    _patch_loader(
        monkeypatch,
        FakeRuntimeTokenizer(),
    )

    tokenizer = MobileClipTokenizer(path)

    assert tokenizer.encode_batch(
        ["short", "long"],
        max_length=4,
    ) == [
        [49406, 1, 49407],
        [49406, 1, 2, 49407],
    ]


def test_rejects_missing_tokenizer_asset(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        EmbeddingConfigurationError
    ):
        MobileClipTokenizer(
            tmp_path / "missing.json"
        )


def test_normalizes_tokenizer_load_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write_tokenizer_asset(tmp_path)

    class FailingLoader:
        @staticmethod
        def from_file(path: str) -> None:
            raise RuntimeError("invalid JSON")

    monkeypatch.setattr(
        tokenizer_module,
        "Tokenizer",
        FailingLoader,
    )

    with pytest.raises(
        EmbeddingConfigurationError
    ):
        MobileClipTokenizer(path)


def test_rejects_tokenizer_without_eot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write_tokenizer_asset(tmp_path)
    _patch_loader(
        monkeypatch,
        FakeRuntimeTokenizer(
            eot_token_id=None
        ),
    )

    with pytest.raises(
        EmbeddingConfigurationError
    ):
        MobileClipTokenizer(path)


def test_rejects_invalid_max_length(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write_tokenizer_asset(tmp_path)
    _patch_loader(
        monkeypatch,
        FakeRuntimeTokenizer(),
    )

    tokenizer = MobileClipTokenizer(path)

    with pytest.raises(ValueError):
        tokenizer.encode_batch(
            ["short"],
            max_length=1,
        )


def test_rejects_blank_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write_tokenizer_asset(tmp_path)
    _patch_loader(
        monkeypatch,
        FakeRuntimeTokenizer(),
    )

    tokenizer = MobileClipTokenizer(path)

    with pytest.raises(
        InvalidEmbeddingInputError
    ):
        tokenizer.encode_batch(
            ["   "],
            max_length=77,
        )


def test_normalizes_encoding_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write_tokenizer_asset(tmp_path)
    _patch_loader(
        monkeypatch,
        FakeRuntimeTokenizer(
            fail_encoding=True
        ),
    )

    tokenizer = MobileClipTokenizer(path)

    with pytest.raises(
        InvalidEmbeddingInputError
    ):
        tokenizer.encode_batch(
            ["short"],
            max_length=77,
        )
