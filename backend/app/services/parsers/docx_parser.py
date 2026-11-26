from __future__ import annotations

import zipfile
from pathlib import Path

from docx import Document
from docx.opc.exceptions import PackageNotFoundError

from app.errors import (
    CorruptedFileError,
    FilePermissionError,
    FileReadError,
)
from app.models import FileMetadata, ParsedDocument


class DocxParser:
    """Extracts paragraph and table text from Office Open XML documents.

    ``python-docx`` is used instead of Apache Tika so that DOCX parsing
    needs neither a Java runtime nor a first-run network download.
    """

    supported_extensions = frozenset({".docx"})

    def parse(
        self,
        path: Path,
        metadata: FileMetadata,
    ) -> ParsedDocument:
        try:
            document = Document(str(path))
        except PermissionError as exc:
            raise FilePermissionError(path) from exc
        except (
            PackageNotFoundError,
            zipfile.BadZipFile,
            KeyError,
            ValueError,
        ) as exc:
            raise CorruptedFileError(
                path,
                f"Invalid DOCX package: {exc}",
            ) from exc
        except OSError as exc:
            raise FileReadError(path, str(exc)) from exc
        except Exception as exc:  # lxml parse errors and similar
            raise CorruptedFileError(path, str(exc)) from exc

        blocks: list[str] = [
            paragraph.text.strip() for paragraph in document.paragraphs
        ]

        for table in document.tables:
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells]
                blocks.append(" | ".join(cell for cell in cells if cell))

        text = "\n".join(block for block in blocks if block)

        return ParsedDocument(
            document_id=metadata.sha256,
            text=text,
            metadata=metadata,
        )
