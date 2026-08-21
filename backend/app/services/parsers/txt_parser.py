from __future__ import annotations

from pathlib import Path

from app.errors import (
    CorruptedFileError,
    FilePermissionError,
    FileReadError,
)
from app.models import FileMetadata, ParsedDocument


class TxtParser:
    """Parses UTF-8 plain-text documents."""

    supported_extensions = frozenset({".txt"})

    def parse(
        self,
        path: Path,
        metadata: FileMetadata,
    ) -> ParsedDocument:
        try:
            text = path.read_text(encoding="utf-8")
        except PermissionError as exc:
            raise FilePermissionError(path) from exc
        except UnicodeDecodeError as exc:
            raise CorruptedFileError(
                path,
                "File is not valid UTF-8",
            ) from exc
        except OSError as exc:
            raise FileReadError(path, str(exc)) from exc

        return ParsedDocument(
            document_id=metadata.sha256,
            text=text,
            metadata=metadata,
        )