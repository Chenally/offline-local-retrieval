from pathlib import Path

import pytest
from app.errors import CorruptedFileError, FilePermissionError
from app.models import DocumentKind
from app.services.metadata_extractor import MetadataExtractor
from app.services.parsers.image_parser import ImageParser

from support import write_png


def test_accepts_valid_png(tmp_path: Path) -> None:
    path = write_png(tmp_path / "red_car.png")

    parsed = ImageParser().parse(path, MetadataExtractor().extract(path))

    assert parsed.kind is DocumentKind.IMAGE
    assert parsed.text == ""


def test_accepts_valid_jpeg(tmp_path: Path) -> None:
    from PIL import Image

    path = tmp_path / "photo.JPG"
    Image.new("RGB", (4, 4)).save(path, format="JPEG")

    parsed = ImageParser().parse(path, MetadataExtractor().extract(path))

    assert parsed.kind is DocumentKind.IMAGE


def test_rejects_non_image_bytes(tmp_path: Path) -> None:
    path = tmp_path / "fake.png"
    path.write_bytes(b"definitely not an image")

    with pytest.raises(CorruptedFileError):
        ImageParser().parse(path, MetadataExtractor().extract(path))


def test_rejects_truncated_png(tmp_path: Path) -> None:
    valid = write_png(tmp_path / "valid.png")
    truncated = tmp_path / "truncated.png"
    truncated.write_bytes(valid.read_bytes()[:30])

    with pytest.raises(CorruptedFileError):
        ImageParser().parse(truncated, MetadataExtractor().extract(truncated))


def test_maps_permission_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = write_png(tmp_path / "blocked.png")
    metadata = MetadataExtractor().extract(path)

    def deny(*args: object, **kwargs: object) -> None:
        raise PermissionError

    monkeypatch.setattr("app.services.parsers.image_parser.Image.open", deny)

    with pytest.raises(FilePermissionError):
        ImageParser().parse(path, metadata)
