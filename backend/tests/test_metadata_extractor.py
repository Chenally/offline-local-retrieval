import hashlib
from pathlib import Path

import pytest

from app.errors import FileReadError
from app.services.metadata_extractor import MetadataExtractor


def test_extract_returns_metadata_and_sha256(
    tmp_path: Path,
) -> None:
    path = tmp_path / "example.txt"
    content = b"offline retrieval"
    path.write_bytes(content)

    metadata = MetadataExtractor(
        chunk_size=4,
    ).extract(path)

    assert metadata.path == path
    assert metadata.file_name == "example.txt"
    assert metadata.extension == ".txt"
    assert metadata.mime_type == "text/plain"
    assert metadata.size_bytes == len(content)
    assert metadata.sha256 == hashlib.sha256(
        content
    ).hexdigest()
    assert metadata.modified_at.tzinfo is not None


def test_rejects_non_positive_chunk_size() -> None:
    with pytest.raises(ValueError, match="chunk_size"):
        MetadataExtractor(chunk_size=0)


def test_extract_rejects_non_file(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileReadError):
        MetadataExtractor().extract(
            tmp_path / "missing.txt"
        )
        