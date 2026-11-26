from __future__ import annotations

import os
import sqlite3
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from app.models import DocumentKind
from app.services.storage.errors import StorageError


@dataclass(frozen=True, slots=True)
class ManifestRecord:
    """Indexing state of one local file path."""

    path: Path
    root: Path
    document_id: str
    kind: DocumentKind
    extension: str
    file_name: str
    size_bytes: int
    modified_at: datetime
    record_count: int
    indexed_at: datetime


@dataclass(frozen=True, slots=True)
class ManifestSummary:
    roots: tuple[Path, ...]
    file_count: int
    document_count: int
    text_file_count: int
    image_file_count: int


class IndexManifest:
    """Persistent map from file paths to indexed content hashes.

    The manifest is what makes indexing incremental: unchanged files are
    skipped, changed files are re-indexed, deleted files are purged, and
    several paths with identical bytes share one set of vectors.
    """

    def __init__(self, path: str | Path) -> None:
        database = str(path)

        if database != ":memory:":
            Path(database).parent.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()

        try:
            self._connection = sqlite3.connect(
                database,
                check_same_thread=False,
            )
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS files (
                    path TEXT PRIMARY KEY,
                    root TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    extension TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    modified_at TEXT NOT NULL,
                    record_count INTEGER NOT NULL,
                    indexed_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS files_document_id
                    ON files (document_id);
                CREATE TABLE IF NOT EXISTS roots (
                    path TEXT PRIMARY KEY,
                    added_at TEXT NOT NULL
                );
                """
            )
            self._connection.commit()
        except sqlite3.Error as exc:
            raise StorageError(f"Could not open index manifest: {exc}") from exc

    # Roots -----------------------------------------------------------------

    def add_root(self, root: Path, added_at: datetime) -> None:
        self._write(
            "INSERT OR IGNORE INTO roots (path, added_at) VALUES (?, ?)",
            (str(root), added_at.isoformat()),
        )

    def remove_root(self, root: Path) -> None:
        self._write("DELETE FROM roots WHERE path = ?", (str(root),))

    def roots(self) -> list[Path]:
        rows = self._read("SELECT path FROM roots ORDER BY path")
        return [Path(row[0]) for row in rows]

    # Files -----------------------------------------------------------------

    def get(self, path: Path) -> ManifestRecord | None:
        rows = self._read("SELECT * FROM files WHERE path = ?", (str(path),))
        return self._record(rows[0]) if rows else None

    def upsert(self, record: ManifestRecord) -> None:
        self._write(
            """
            INSERT INTO files (
                path, root, document_id, kind, extension, file_name,
                size_bytes, modified_at, record_count, indexed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                root = excluded.root,
                document_id = excluded.document_id,
                kind = excluded.kind,
                extension = excluded.extension,
                file_name = excluded.file_name,
                size_bytes = excluded.size_bytes,
                modified_at = excluded.modified_at,
                record_count = excluded.record_count,
                indexed_at = excluded.indexed_at
            """,
            (
                str(record.path),
                str(record.root),
                record.document_id,
                record.kind.value,
                record.extension,
                record.file_name,
                record.size_bytes,
                record.modified_at.isoformat(),
                record.record_count,
                record.indexed_at.isoformat(),
            ),
        )

    def delete(self, path: Path) -> None:
        self._write("DELETE FROM files WHERE path = ?", (str(path),))

    def records_under(self, root: Path) -> list[ManifestRecord]:
        prefix = str(root).rstrip(os.sep) + os.sep
        upper_bound = prefix[:-1] + chr(ord(os.sep) + 1)
        rows = self._read(
            "SELECT * FROM files WHERE path >= ? AND path < ? ORDER BY path",
            (prefix, upper_bound),
        )
        return [self._record(row) for row in rows]

    def records_for_documents(
        self,
        document_ids: Iterable[str],
    ) -> dict[str, list[ManifestRecord]]:
        ids = sorted(set(document_ids))
        result: dict[str, list[ManifestRecord]] = {key: [] for key in ids}

        # Stay below SQLite's default host-parameter limit.
        for start in range(0, len(ids), 500):
            batch = ids[start : start + 500]
            placeholders = ", ".join("?" for _ in batch)
            rows = self._read(
                f"SELECT * FROM files WHERE document_id IN ({placeholders}) "
                "ORDER BY path",
                tuple(batch),
            )

            for row in rows:
                record = self._record(row)
                result[record.document_id].append(record)

        return result

    def find_document(self, document_id: str) -> ManifestRecord | None:
        rows = self._read(
            "SELECT * FROM files WHERE document_id = ? LIMIT 1",
            (document_id,),
        )
        return self._record(rows[0]) if rows else None

    def summary(self) -> ManifestSummary:
        rows = self._read(
            """
            SELECT
                COUNT(*),
                COUNT(DISTINCT document_id),
                SUM(CASE WHEN kind = 'text' THEN 1 ELSE 0 END),
                SUM(CASE WHEN kind = 'image' THEN 1 ELSE 0 END)
            FROM files
            """
        )
        file_count, document_count, text_count, image_count = rows[0]

        return ManifestSummary(
            roots=tuple(self.roots()),
            file_count=int(file_count or 0),
            document_count=int(document_count or 0),
            text_file_count=int(text_count or 0),
            image_file_count=int(image_count or 0),
        )

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    # Helpers ---------------------------------------------------------------

    def _read(
        self,
        sql: str,
        parameters: tuple[object, ...] = (),
    ) -> list[tuple[Any, ...]]:
        with self._lock:
            try:
                return self._connection.execute(sql, parameters).fetchall()
            except sqlite3.Error as exc:
                raise StorageError(f"Manifest read failed: {exc}") from exc

    def _write(self, sql: str, parameters: tuple[object, ...]) -> None:
        with self._lock:
            try:
                self._connection.execute(sql, parameters)
                self._connection.commit()
            except sqlite3.Error as exc:
                self._connection.rollback()
                raise StorageError(f"Manifest write failed: {exc}") from exc

    @staticmethod
    def _record(row: tuple[Any, ...]) -> ManifestRecord:
        (
            path,
            root,
            document_id,
            kind,
            extension,
            file_name,
            size_bytes,
            modified_at,
            record_count,
            indexed_at,
        ) = row

        return ManifestRecord(
            path=Path(str(path)),
            root=Path(str(root)),
            document_id=str(document_id),
            kind=DocumentKind(str(kind)),
            extension=str(extension),
            file_name=str(file_name),
            size_bytes=int(size_bytes),
            modified_at=datetime.fromisoformat(str(modified_at)),
            record_count=int(record_count),
            indexed_at=datetime.fromisoformat(str(indexed_at)),
        )
