from __future__ import annotations

from pathlib import Path

import pypdfium2 as pdfium

from app.errors import (
    CorruptedFileError,
    FilePermissionError,
    FileReadError,
)
from app.models import FileMetadata, ParsedDocument


class PdfParser:
    """Extracts text from PDF pages with PDFium."""

    supported_extensions = frozenset({".pdf"})

    def parse(
        self,
        path: Path,
        metadata: FileMetadata,
    ) -> ParsedDocument:
        document: pdfium.PdfDocument | None = None

        try:
            document = pdfium.PdfDocument(str(path))
            page_count = len(document)
            page_texts: list[str] = []

            for page_index in range(page_count):
                page = document[page_index]
                text_page = None

                try:
                    text_page = page.get_textpage()
                    page_texts.append(
                        text_page.get_text_range().strip()
                    )
                finally:
                    if text_page is not None:
                        text_page.close()

                    page.close()

        except PermissionError as exc:
            raise FilePermissionError(path) from exc
        except pdfium.PdfiumError as exc:
            raise CorruptedFileError(path, str(exc)) from exc
        except OSError as exc:
            raise FileReadError(path, str(exc)) from exc
        finally:
            if document is not None:
                document.close()

        text = "\n\n".join(
            page_text
            for page_text in page_texts
            if page_text
        )

        return ParsedDocument(
            document_id=metadata.sha256,
            text=text,
            metadata=metadata,
            page_count=page_count,
        )