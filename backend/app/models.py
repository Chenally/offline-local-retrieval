from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True, slots=True)
class FileMetadata:
    """Metadata collected before a local file is parsed."""

    path: Path
    file_name: str
    extension: str
    mime_type: str
    size_bytes: int
    modified_at: datetime
    sha256: str


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    """Normalized result produced by every document parser."""

    document_id: str
    text: str
    metadata: FileMetadata
    page_count: int | None = None


@dataclass(frozen=True, slots=True)
class IngestionErrorRecord:
    """A recoverable file-level error from a batch ingestion run."""

    path: Path
    error_type: str
    message: str


@dataclass(slots=True)
class IngestionReport:
    """Complete result of ingesting one local directory."""

    documents: list[ParsedDocument] = field(default_factory=list)
    duplicates: list[Path] = field(default_factory=list)
    errors: list[IngestionErrorRecord] = field(default_factory=list)