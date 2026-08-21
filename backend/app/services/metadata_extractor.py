from __future__ import annotations

import hashlib
import mimetypes
from datetime import datetime, timezone
from pathlib import Path

from app.errors import FilePermissionError, FileReadError
from app.models import FileMetadata


class MetadataExtractor:
    """Extracts filesystem metadata and a content-based SHA-256 ID."""

    def __init__(self, chunk_size: int = 64 * 1024) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")

        self._chunk_size = chunk_size

    def extract(self, file_path: str | Path) -> FileMetadata:
        path = Path(file_path).expanduser().resolve()

        if not path.is_file():
            raise FileReadError(path, "Path is not a regular file")

        try:
            stat = path.stat()
            sha256 = self._calculate_sha256(path)
        except PermissionError as exc:
            raise FilePermissionError(path) from exc
        except OSError as exc:
            raise FileReadError(path, str(exc)) from exc

        mime_type = mimetypes.guess_type(path.name)[0]

        return FileMetadata(
            path=path,
            file_name=path.name,
            extension=path.suffix.lower(),
            mime_type=mime_type or "application/octet-stream",
            size_bytes=stat.st_size,
            modified_at=datetime.fromtimestamp(
                stat.st_mtime,
                tz=timezone.utc,
            ),
            sha256=sha256,
        )

    def _calculate_sha256(self, path: Path) -> str:
        digest = hashlib.sha256()

        with path.open("rb") as file:
            while chunk := file.read(self._chunk_size):
                digest.update(chunk)

        return digest.hexdigest()