from datetime import datetime, timezone
from pathlib import Path

from app.models import FileMetadata
from app.services.duplicate_detector import DuplicateDetector


def make_metadata(
    path: Path,
    sha256: str,
) -> FileMetadata:
    return FileMetadata(
        path=path,
        file_name=path.name,
        extension=path.suffix,
        mime_type="text/plain",
        size_bytes=1,
        modified_at=datetime.now(timezone.utc),
        sha256=sha256,
    )


def test_register_returns_original_path_for_duplicate() -> None:
    detector = DuplicateDetector()

    original = make_metadata(
        Path("original.txt"),
        "same-hash",
    )
    duplicate = make_metadata(
        Path("duplicate.txt"),
        "same-hash",
    )

    assert detector.register(original) is None
    assert detector.register(duplicate) == original.path


def test_register_accepts_different_hashes() -> None:
    detector = DuplicateDetector()

    assert detector.register(
        make_metadata(Path("a.txt"), "hash-a")
    ) is None

    assert detector.register(
        make_metadata(Path("b.txt"), "hash-b")
    ) is None
    