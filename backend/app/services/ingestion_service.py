from __future__ import annotations

from pathlib import Path

from app.errors import (
    FileIngestionError,
    FilePermissionError,
    FileReadError,
)
from app.models import IngestionErrorRecord, IngestionReport
from app.services.duplicate_detector import DuplicateDetector
from app.services.file_scanner import FileScanner
from app.services.metadata_extractor import MetadataExtractor
from app.services.parsers.registry import ParserRegistry


class IngestionService:
    """Coordinates discovery, deduplication, and local file parsing."""

    def __init__(
        self,
        scanner: FileScanner | None = None,
        metadata_extractor: MetadataExtractor | None = None,
        parser_registry: ParserRegistry | None = None,
    ) -> None:
        self._scanner = scanner or FileScanner()
        self._metadata_extractor = (
            metadata_extractor or MetadataExtractor()
        )
        self._parser_registry = parser_registry or ParserRegistry()

    def ingest(self, directory: str | Path) -> IngestionReport:
        report = IngestionReport()
        duplicate_detector = DuplicateDetector()

        try:
            paths = self._scanner.scan(directory)
        except FileIngestionError as exc:
            self._record_error(
                report,
                Path(directory),
                exc,
            )
            return report

        for path in paths:
            try:
                parser = self._parser_registry.get_parser(path)
                metadata = self._metadata_extractor.extract(path)
                original_path = duplicate_detector.register(metadata)

                if original_path is not None:
                    report.duplicates.append(path)
                    continue

                document = parser.parse(path, metadata)
                report.documents.append(document)

            except FileIngestionError as exc:
                self._record_error(report, path, exc)

            except PermissionError:
                self._record_error(
                    report,
                    path,
                    FilePermissionError(path),
                )

            except OSError as exc:
                self._record_error(
                    report,
                    path,
                    FileReadError(path, str(exc)),
                )

        return report

    @staticmethod
    def _record_error(
        report: IngestionReport,
        path: Path,
        error: FileIngestionError,
    ) -> None:
        report.errors.append(
            IngestionErrorRecord(
                path=path,
                error_type=type(error).__name__,
                message=str(error),
            )
        )