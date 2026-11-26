from __future__ import annotations

import sqlite3
import threading
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from pathlib import Path

from app.models import DocumentKind
from app.services.storage.errors import StorageError
from app.services.text_units import (
    contains_cjk,
    iter_search_terms,
    space_cjk_characters,
)

# Column weights for bm25(): unindexed columns first, then file_name, body.
_BM25_WEIGHTS = "0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 1.0"
_MAX_QUERY_TERMS = 32


@dataclass(frozen=True, slots=True)
class KeywordEntry:
    """One searchable unit: a text chunk, or a file name for an image."""

    chunk_id: str
    text: str
    page_number: int | None = None


@dataclass(frozen=True, slots=True)
class KeywordHit:
    chunk_id: str
    document_id: str
    kind: DocumentKind
    score: float
    text: str
    page_number: int | None
    file_name: str = ""


class KeywordIndex:
    """Local BM25 full-text index backed by SQLite FTS5.

    Chroma can filter on substrings but does not score lexical relevance,
    so keyword ranking uses FTS5's built-in ``bm25()`` function. SQLite is
    part of the Python standard library, which keeps this channel offline
    and dependency-free.
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
            self._connection.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS keyword_chunks USING fts5(
                    chunk_id UNINDEXED,
                    document_id UNINDEXED,
                    kind UNINDEXED,
                    extension UNINDEXED,
                    page_number UNINDEXED,
                    raw_text UNINDEXED,
                    file_name,
                    body,
                    tokenize = 'unicode61 remove_diacritics 2'
                )
                """
            )
            self._connection.commit()
        except sqlite3.Error as exc:
            raise StorageError(f"Could not open keyword index: {exc}") from exc

    def add_document(
        self,
        document_id: str,
        kind: DocumentKind,
        extension: str,
        file_name: str,
        entries: Sequence[KeywordEntry],
    ) -> None:
        searchable_name = space_cjk_characters(
            file_name.replace("_", " ").replace("-", " ")
        )

        rows = [
            (
                entry.chunk_id,
                document_id,
                kind.value,
                extension,
                entry.page_number,
                entry.text,
                searchable_name,
                space_cjk_characters(entry.text),
            )
            for entry in entries
        ]

        with self._lock:
            try:
                self._connection.execute(
                    "DELETE FROM keyword_chunks WHERE document_id = ?",
                    (document_id,),
                )
                self._connection.executemany(
                    """
                    INSERT INTO keyword_chunks (
                        chunk_id, document_id, kind, extension, page_number,
                        raw_text, file_name, body
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
                self._connection.commit()
            except sqlite3.Error as exc:
                self._connection.rollback()
                raise StorageError(
                    f"Could not index keywords for {document_id}: {exc}"
                ) from exc

    def delete_document(self, document_id: str) -> None:
        with self._lock:
            try:
                self._connection.execute(
                    "DELETE FROM keyword_chunks WHERE document_id = ?",
                    (document_id,),
                )
                self._connection.commit()
            except sqlite3.Error as exc:
                self._connection.rollback()
                raise StorageError(
                    f"Could not delete keywords for {document_id}: {exc}"
                ) from exc

    def search(
        self,
        query: str,
        limit: int,
        extensions: Collection[str] | None = None,
    ) -> list[KeywordHit]:
        if limit <= 0:
            raise ValueError("limit must be greater than zero")

        match_expression = build_match_expression(query)

        if match_expression is None:
            return []

        sql = f"""
            SELECT chunk_id, document_id, kind, page_number, raw_text, file_name,
                   bm25(keyword_chunks, {_BM25_WEIGHTS}) AS rank
            FROM keyword_chunks
            WHERE keyword_chunks MATCH ?
        """
        parameters: list[object] = [match_expression]

        if extensions:
            ordered = sorted(extensions)
            placeholders = ", ".join("?" for _ in ordered)
            sql += f" AND extension IN ({placeholders})"
            parameters.extend(ordered)

        sql += " ORDER BY rank, chunk_id LIMIT ?"
        parameters.append(limit)

        with self._lock:
            try:
                rows = self._connection.execute(sql, parameters).fetchall()
            except sqlite3.Error as exc:
                raise StorageError(f"Keyword search failed: {exc}") from exc

        # FTS5 bm25() is negative; smaller means more relevant.
        return [
            KeywordHit(
                chunk_id=chunk_id,
                document_id=document_id,
                kind=DocumentKind(kind),
                score=-float(rank),
                text=raw_text,
                page_number=page_number,
                file_name=file_name,
            )
            for chunk_id, document_id, kind, page_number, raw_text, file_name, rank in rows
        ]

    def count_documents(self) -> int:
        with self._lock:
            row = self._connection.execute(
                "SELECT COUNT(DISTINCT document_id) FROM keyword_chunks"
            ).fetchone()

        return int(row[0])

    def close(self) -> None:
        with self._lock:
            self._connection.close()


def build_match_expression(query: str) -> str | None:
    """Convert free text into a safe FTS5 ``OR`` query.

    Every term is quoted, so user input can never inject FTS5 operators.
    A CJK run becomes a phrase of single characters, which matches the
    character-spaced form stored in the index.
    """

    clauses: list[str] = []

    for term in iter_search_terms(query):
        if contains_cjk(term):
            phrase = " ".join(space_cjk_characters(term).split())
        else:
            phrase = term

        clause = '"' + phrase.replace('"', '""') + '"'

        if clause not in clauses:
            clauses.append(clause)

        if len(clauses) >= _MAX_QUERY_TERMS:
            break

    if not clauses:
        return None

    return " OR ".join(clauses)
