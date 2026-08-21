from pathlib import Path

import pytest

from app.errors import InvalidDirectoryError
from app.services.file_scanner import FileScanner


def test_scan_returns_visible_regular_files_recursively(
    tmp_path: Path,
) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()

    first = tmp_path / "a.txt"
    second = nested / "b.pdf"
    hidden = tmp_path / ".hidden.txt"
    hidden_directory = tmp_path / ".private"
    hidden_directory.mkdir()
    hidden_nested = hidden_directory / "secret.txt"

    for path in (first, second, hidden, hidden_nested):
        path.write_text("content", encoding="utf-8")

    assert FileScanner().scan(tmp_path) == [
        first,
        second,
    ]


@pytest.mark.parametrize(
    "directory_name",
    ["missing", "plain-file"],
)
def test_scan_rejects_invalid_directory(
    tmp_path: Path,
    directory_name: str,
) -> None:
    path = tmp_path / directory_name

    if directory_name == "plain-file":
        path.write_text(
            "not a directory",
            encoding="utf-8",
        )

    with pytest.raises(InvalidDirectoryError):
        FileScanner().scan(path)