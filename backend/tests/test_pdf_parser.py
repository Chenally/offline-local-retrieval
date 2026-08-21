from pathlib import Path

import pytest

from app.errors import CorruptedFileError
from app.services.metadata_extractor import MetadataExtractor
from app.services.parsers.pdf_parser import PdfParser


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_PDF = (
    PROJECT_ROOT
    / "sample_data"
    / "pdf"
    / "sample.pdf"
)


def test_parse_valid_pdf() -> None:
    metadata = MetadataExtractor().extract(SAMPLE_PDF)

    document = PdfParser().parse(
        SAMPLE_PDF,
        metadata,
    )

    assert document.document_id == metadata.sha256
    assert document.text.strip()
    assert document.metadata == metadata
    assert document.page_count is not None
    assert document.page_count >= 1


def test_parse_rejects_corrupted_pdf(
    tmp_path: Path,
) -> None:
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"this is not a PDF")

    metadata = MetadataExtractor().extract(path)

    with pytest.raises(CorruptedFileError):
        PdfParser().parse(path, metadata)
        