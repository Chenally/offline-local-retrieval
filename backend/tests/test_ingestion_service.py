from pathlib import Path

import pytest

from app.services.ingestion_service import IngestionService


def test_ingest_processes_batch_and_records_failures(
    tmp_path: Path,
) -> None:
    (tmp_path / "valid.txt").write_text(
        "unique text",
        encoding="utf-8",
    )

    (tmp_path / "duplicate.txt").write_text(
        "unique text",
        encoding="utf-8",
    )

    (tmp_path / "unsupported.docx").write_bytes(
        b"unsupported"
    )

    (tmp_path / "broken.pdf").write_bytes(
        b"not a PDF"
    )

    report = IngestionService().ingest(tmp_path)

    assert len(report.documents) == 1
    assert report.documents[0].text == "unique text"

    assert len(report.duplicates) == 1
    assert report.duplicates[0].name in {
        "duplicate.txt",
        "valid.txt",
    }

    assert {
        error.error_type
        for error in report.errors
    } == {
        "CorruptedFileError",
        "UnsupportedFileTypeError",
    }


def test_ingest_records_permission_error_and_continues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blocked = tmp_path / "blocked.txt"
    valid = tmp_path / "valid.txt"

    blocked.write_text(
        "blocked",
        encoding="utf-8",
    )

    valid.write_text(
        "valid",
        encoding="utf-8",
    )

    original_read_text = Path.read_text

    def conditional_read(
        path: Path,
        *args: object,
        **kwargs: object,
    ) -> str:
        if path == blocked:
            raise PermissionError

        return original_read_text(
            path,
            *args,
            **kwargs,
        )

    monkeypatch.setattr(
        Path,
        "read_text",
        conditional_read,
    )

    report = IngestionService().ingest(tmp_path)

    assert [
        document.metadata.path
        for document in report.documents
    ] == [valid]

    assert len(report.errors) == 1
    assert report.errors[0].path == blocked
    assert (
        report.errors[0].error_type
        == "FilePermissionError"
    )


def test_ingest_empty_directory_returns_empty_report(
    tmp_path: Path,
) -> None:
    report = IngestionService().ingest(tmp_path)

    assert report.documents == []
    assert report.duplicates == []
    assert report.errors == []


def test_ingest_invalid_directory_returns_error(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing"

    report = IngestionService().ingest(missing)

    assert report.documents == []
    assert len(report.errors) == 1
    assert report.errors[0].path == missing
    assert (
        report.errors[0].error_type
        == "InvalidDirectoryError"
    )