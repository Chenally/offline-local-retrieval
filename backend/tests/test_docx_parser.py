from pathlib import Path

import pytest
from app.errors import CorruptedFileError
from app.services.metadata_extractor import MetadataExtractor
from app.services.parsers.docx_parser import DocxParser
from docx import Document

from support import write_docx


def test_parses_paragraphs_and_tables(tmp_path: Path) -> None:
    path = tmp_path / "report.docx"
    document = Document()
    document.add_paragraph("Quarterly revenue summary")
    document.add_paragraph("")
    table = document.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Region"
    table.rows[0].cells[1].text = "APAC"
    document.save(str(path))

    metadata = MetadataExtractor().extract(path)
    parsed = DocxParser().parse(path, metadata)

    assert parsed.document_id == metadata.sha256
    assert parsed.text == "Quarterly revenue summary\nRegion | APAC"


def test_rejects_corrupted_docx(tmp_path: Path) -> None:
    path = tmp_path / "broken.docx"
    path.write_bytes(b"not a zip archive")

    with pytest.raises(CorruptedFileError):
        DocxParser().parse(path, MetadataExtractor().extract(path))


def test_empty_docx_returns_empty_text(tmp_path: Path) -> None:
    path = write_docx(tmp_path / "empty.docx", [])

    parsed = DocxParser().parse(path, MetadataExtractor().extract(path))

    assert parsed.text == ""
