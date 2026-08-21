from pathlib import Path

import pytest

from app.errors import CorruptedFileError, FilePermissionError
from app.services.metadata_extractor import MetadataExtractor
from app.services.parsers.txt_parser import TxtParser


def test_parse_valid_utf8_text_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sample.txt"
    expected_text = "Offline local retrieval.\nSecond line."

    path.write_text(
        expected_text,
        encoding="utf-8",
    )

    metadata = MetadataExtractor().extract(path)
    document = TxtParser().parse(path, metadata)

    assert document.document_id == metadata.sha256
    assert document.text == expected_text
    assert document.metadata == metadata
    assert document.page_count is None


def test_parse_rejects_invalid_utf8(
    tmp_path: Path,
) -> None:
    path = tmp_path / "broken.txt"
    path.write_bytes(b"\xff\xfe\x00")

    metadata = MetadataExtractor().extract(path)

    with pytest.raises(CorruptedFileError):
        TxtParser().parse(path, metadata)


def test_parse_normalizes_permission_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "blocked.txt"
    path.write_text("content", encoding="utf-8")

    metadata = MetadataExtractor().extract(path)

    def deny_read(
        *args: object,
        **kwargs: object,
    ) -> str:
        raise PermissionError

    monkeypatch.setattr(
        Path,
        "read_text",
        deny_read,
    )

    with pytest.raises(FilePermissionError):
        TxtParser().parse(path, metadata)
        