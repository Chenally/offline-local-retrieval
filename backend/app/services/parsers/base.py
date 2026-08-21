from __future__ import annotations

from pathlib import Path
from typing import Protocol

from app.models import FileMetadata, ParsedDocument


class FileParser(Protocol):
    """Interface shared by every local document parser."""

    supported_extensions: frozenset[str]

    def parse(
        self,
        path: Path,
        metadata: FileMetadata,
    ) -> ParsedDocument:
        """Parse one file into the normalized document model."""
        ...