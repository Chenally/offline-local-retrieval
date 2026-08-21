from __future__ import annotations

from pathlib import Path

from app.models import FileMetadata


class DuplicateDetector:
    """Detects duplicate content by SHA-256 within one ingestion run."""

    def __init__(self) -> None:
        self._original_paths: dict[str, Path] = {}

    def register(self, metadata: FileMetadata) -> Path | None:
        original_path = self._original_paths.get(metadata.sha256)

        if original_path is not None:
            return original_path

        self._original_paths[metadata.sha256] = metadata.path
        return None