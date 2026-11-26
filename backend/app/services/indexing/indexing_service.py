from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from app.errors import (
    FileIngestionError,
    FilePermissionError,
    FileReadError,
)
from app.models import (
    DocumentKind,
    FileMetadata,
    IngestionErrorRecord,
    ParsedDocument,
)
from app.services.chunking.text_chunker import TextChunker
from app.services.embeddings.base import Embedder
from app.services.embeddings.errors import EmbeddingError
from app.services.embeddings.types import EmbeddingSpace
from app.services.file_scanner import FileScanner
from app.services.metadata_extractor import MetadataExtractor
from app.services.parsers.registry import ParserRegistry
from app.services.storage.errors import StorageError
from app.services.storage.keyword_index import KeywordEntry, KeywordIndex
from app.services.storage.manifest import IndexManifest, ManifestRecord
from app.services.storage.vector_store import VectorRecord, VectorStore

logger = logging.getLogger(__name__)


class FileOutcome(str, Enum):
    INDEXED = "indexed"
    UNCHANGED = "unchanged"
    DUPLICATE = "duplicate"


@dataclass(frozen=True, slots=True)
class IndexProgress:
    processed: int
    total: int
    current_path: Path | None


@dataclass(slots=True)
class IndexingReport:
    """Result of synchronizing one directory with the local index."""

    root: Path
    scanned: int = 0
    indexed: list[Path] = field(default_factory=list)
    unchanged: list[Path] = field(default_factory=list)
    duplicates: list[Path] = field(default_factory=list)
    removed: list[Path] = field(default_factory=list)
    skipped_unsupported: int = 0
    vector_count: int = 0
    errors: list[IngestionErrorRecord] = field(default_factory=list)


ProgressCallback = Callable[[IndexProgress], None]


class IndexingService:
    """Keeps the vector store and keyword index in sync with local folders.

    Pipeline per file: scan -> stat/hash -> parse -> chunk -> embed ->
    write vectors and keywords -> record in the manifest. The manifest is
    written last, so a file whose indexing fails is retried next run.
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        keyword_index: KeywordIndex,
        manifest: IndexManifest,
        scanner: FileScanner | None = None,
        metadata_extractor: MetadataExtractor | None = None,
        parser_registry: ParserRegistry | None = None,
        chunker: TextChunker | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._keyword_index = keyword_index
        self._manifest = manifest
        self._scanner = scanner or FileScanner()
        self._metadata_extractor = metadata_extractor or MetadataExtractor()
        self._parser_registry = parser_registry or ParserRegistry()
        self._chunker = chunker or TextChunker()
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def index_directory(
        self,
        directory: str | Path,
        progress: ProgressCallback | None = None,
    ) -> IndexingReport:
        root = Path(directory).expanduser().resolve()
        report = IndexingReport(root=root)

        try:
            paths = self._scanner.scan(root)
        except FileIngestionError as exc:
            # A missing folder may be an unmounted drive: keep its index.
            self._record_error(report, root, exc)
            return report

        self._manifest.add_root(root, self._clock())

        supported = [path for path in paths if self._parser_registry.supports(path)]
        report.scanned = len(paths)
        report.skipped_unsupported = len(paths) - len(supported)

        current_paths = set(supported)

        for record in self._manifest.records_under(root):
            if record.path in current_paths:
                continue

            try:
                self._remove_record(record)
                report.removed.append(record.path)
            except StorageError as exc:
                self._record_error(report, record.path, exc)

        total = len(supported)

        for index, path in enumerate(supported):
            if progress is not None:
                progress(IndexProgress(index, total, path))

            try:
                outcome, vector_count = self._index_file(root, path)
            except (FileIngestionError, EmbeddingError, StorageError) as exc:
                self._record_error(report, path, exc)
                continue
            except PermissionError:
                self._record_error(report, path, FilePermissionError(path))
                continue
            except OSError as exc:
                self._record_error(report, path, FileReadError(path, str(exc)))
                continue
            except Exception as exc:  # keep one bad file from stopping a batch
                logger.exception("Unexpected failure while indexing %s", path)
                self._record_error(report, path, exc)
                continue

            if outcome is FileOutcome.INDEXED:
                report.indexed.append(path)
                report.vector_count += vector_count
            elif outcome is FileOutcome.DUPLICATE:
                report.duplicates.append(path)
            else:
                report.unchanged.append(path)

        if progress is not None:
            progress(IndexProgress(total, total, None))

        return report

    def remove_directory(self, directory: str | Path) -> list[Path]:
        """Forget a folder and purge vectors no other folder still uses."""

        root = Path(directory).expanduser().resolve()
        self._manifest.remove_root(root)
        other_roots = self._manifest.roots()
        removed: list[Path] = []

        for record in self._manifest.records_under(root):
            if any(_is_within(record.path, other) for other in other_roots):
                continue

            self._remove_record(record)
            removed.append(record.path)

        return removed

    # Per-file work ---------------------------------------------------------

    def _index_file(self, root: Path, path: Path) -> tuple[FileOutcome, int]:
        existing = self._manifest.get(path)
        stat = path.stat()
        modified_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)

        if (
            existing is not None
            and existing.size_bytes == stat.st_size
            and existing.modified_at == modified_at
        ):
            return FileOutcome.UNCHANGED, 0

        metadata = self._metadata_extractor.extract(path)

        if existing is not None and existing.document_id == metadata.sha256:
            # Touched but identical bytes: refresh the stat fingerprint only.
            self._manifest.upsert(
                self._manifest_record(
                    root, metadata, existing.kind, existing.record_count
                )
            )
            return FileOutcome.UNCHANGED, 0

        previous_document_id = existing.document_id if existing else None
        sibling = self._manifest.find_document(metadata.sha256)

        if sibling is not None:
            # Same bytes already indexed through another path: reuse vectors.
            self._manifest.upsert(
                self._manifest_record(
                    root, metadata, sibling.kind, sibling.record_count
                )
            )
            self._release(previous_document_id)
            return FileOutcome.DUPLICATE, 0

        parser = self._parser_registry.get_parser(path)
        document = parser.parse(path, metadata)
        vector_count = self._write_document(document)

        self._manifest.upsert(
            self._manifest_record(root, metadata, document.kind, vector_count)
        )
        self._release(previous_document_id)
        return FileOutcome.INDEXED, vector_count

    def _write_document(self, document: ParsedDocument) -> int:
        document_id = document.document_id
        metadata = document.metadata

        # Clear leftovers of an earlier interrupted attempt.
        self._purge(document_id)

        try:
            if document.kind is DocumentKind.IMAGE:
                records, entries = self._image_records(document)
            else:
                records, entries = self._text_records(document)

            if records:
                self._vector_store.upsert(records)

            self._keyword_index.add_document(
                document_id=document_id,
                kind=document.kind,
                extension=metadata.extension,
                file_name=metadata.file_name,
                entries=entries,
            )
        except Exception:
            self._purge(document_id)
            raise

        return len(records)

    def _text_records(
        self,
        document: ParsedDocument,
    ) -> tuple[list[VectorRecord], list[KeywordEntry]]:
        metadata = document.metadata
        chunks = self._chunker.chunk(document)

        if not chunks:
            # Nothing to embed (for example a scanned PDF without a text
            # layer); keep the file findable by its name.
            return [], [KeywordEntry(chunk_id=f"{document.document_id}:name", text="")]

        embeddings = self._embedder.embed_texts(
            [chunk.text for chunk in chunks],
            space=EmbeddingSpace.TEXT_SEMANTIC,
        )

        records = [
            VectorRecord(
                record_id=chunk.chunk_id,
                document_id=document.document_id,
                embedding=embedding,
                metadata={
                    "kind": document.kind.value,
                    "extension": metadata.extension,
                    "file_name": metadata.file_name,
                    "chunk_index": chunk.chunk_index,
                    "page_number": chunk.page_number,
                    "char_start": chunk.char_start,
                    "char_end": chunk.char_end,
                },
                text=chunk.text,
            )
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]

        entries = [
            KeywordEntry(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                page_number=chunk.page_number,
            )
            for chunk in chunks
        ]

        return records, entries

    def _image_records(
        self,
        document: ParsedDocument,
    ) -> tuple[list[VectorRecord], list[KeywordEntry]]:
        metadata = document.metadata
        [embedding] = self._embedder.embed_images([metadata.path])
        record_id = f"{document.document_id}:image"

        record = VectorRecord(
            record_id=record_id,
            document_id=document.document_id,
            embedding=embedding,
            metadata={
                "kind": document.kind.value,
                "extension": metadata.extension,
                "file_name": metadata.file_name,
            },
        )

        return [record], [KeywordEntry(chunk_id=record_id, text="")]

    # Bookkeeping -----------------------------------------------------------

    def _remove_record(self, record: ManifestRecord) -> None:
        self._manifest.delete(record.path)
        self._release(record.document_id)

    def _release(self, document_id: str | None) -> None:
        """Delete stored content once no path references it any more."""

        if document_id is None:
            return

        if self._manifest.find_document(document_id) is None:
            self._purge(document_id)

    def _purge(self, document_id: str) -> None:
        self._vector_store.delete_document(document_id)
        self._keyword_index.delete_document(document_id)

    def _manifest_record(
        self,
        root: Path,
        metadata: FileMetadata,
        kind: DocumentKind,
        record_count: int,
    ) -> ManifestRecord:
        return ManifestRecord(
            path=metadata.path,
            root=root,
            document_id=metadata.sha256,
            kind=kind,
            extension=metadata.extension,
            file_name=metadata.file_name,
            size_bytes=metadata.size_bytes,
            modified_at=metadata.modified_at,
            record_count=record_count,
            indexed_at=self._clock(),
        )

    @staticmethod
    def _record_error(
        report: IndexingReport,
        path: Path,
        error: Exception,
    ) -> None:
        report.errors.append(
            IngestionErrorRecord(
                path=path,
                error_type=type(error).__name__,
                message=str(error),
            )
        )


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False

    return True
