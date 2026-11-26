from __future__ import annotations

import time
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from app.models import DocumentKind
from app.services.embeddings.base import Embedder
from app.services.embeddings.errors import EmbeddingConfigurationError
from app.services.embeddings.types import EmbeddingSpace
from app.services.retrieval.ranking import (
    ChannelCandidate,
    ChannelScore,
    HybridRanker,
    SearchMode,
)
from app.services.storage.keyword_index import KeywordIndex
from app.services.storage.manifest import IndexManifest, ManifestRecord
from app.services.storage.vector_store import VectorStore
from app.services.text_units import keyword_coverage

MAX_QUERY_LENGTH = 1000
MAX_TOP_K = 100


class InvalidQueryError(ValueError):
    """Raised when a search request is malformed."""


@dataclass(frozen=True, slots=True)
class SearchQuery:
    text: str
    top_k: int = 10
    modes: frozenset[SearchMode] = frozenset(SearchMode)
    extensions: frozenset[str] | None = None
    path_prefix: Path | None = None
    modified_after: datetime | None = None
    modified_before: datetime | None = None


@dataclass(frozen=True, slots=True)
class SearchResult:
    document_id: str
    kind: DocumentKind
    file_name: str
    extension: str
    paths: tuple[Path, ...]
    score: float
    channel_scores: Mapping[SearchMode, ChannelScore]
    snippet: str | None
    page_number: int | None
    chunk_id: str | None


@dataclass(frozen=True, slots=True)
class SearchResponse:
    query: str
    results: list[SearchResult]
    took_ms: float
    warnings: list[str] = field(default_factory=list)


class RetrievalService:
    """Runs hybrid local search over the keyword and vector indexes."""

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        keyword_index: KeywordIndex,
        manifest: IndexManifest,
        ranker: HybridRanker | None = None,
        candidate_multiplier: int = 5,
        timer: Callable[[], float] = time.perf_counter,
    ) -> None:
        if candidate_multiplier <= 0:
            raise ValueError("candidate_multiplier must be greater than zero")

        self._embedder = embedder
        self._vector_store = vector_store
        self._keyword_index = keyword_index
        self._manifest = manifest
        self._ranker = ranker or HybridRanker()
        self._candidate_multiplier = candidate_multiplier
        self._timer = timer

    def search(self, query: SearchQuery) -> SearchResponse:
        started = self._timer()
        text = self._validate(query)
        extensions = _normalize_extensions(query.extensions)
        pool_size = query.top_k * self._candidate_multiplier
        warnings: list[str] = []

        channel_hits: dict[SearchMode, list[ChannelCandidate]] = {}

        for mode in sorted(query.modes, key=lambda item: item.value):
            try:
                channel_hits[mode] = self._run_channel(
                    mode,
                    text,
                    pool_size,
                    extensions,
                )
            except EmbeddingConfigurationError as exc:
                # A missing model disables its channel rather than search.
                channel_hits[mode] = []
                warnings.append(f"{mode.value} search unavailable: {exc}")

        ranked = self._ranker.rank(channel_hits, query.modes)
        records = self._manifest.records_for_documents(
            item.document_id for item in ranked
        )

        results: list[SearchResult] = []

        for item in ranked:
            matching = [
                record
                for record in records.get(item.document_id, [])
                if _matches_filters(record, query)
            ]

            if not matching:
                continue

            first = matching[0]
            best = item.best_candidate

            results.append(
                SearchResult(
                    document_id=item.document_id,
                    kind=item.kind,
                    file_name=first.file_name,
                    extension=first.extension,
                    paths=tuple(record.path for record in matching),
                    score=item.score,
                    channel_scores=item.channel_scores,
                    snippet=best.snippet or None,
                    page_number=best.page_number,
                    chunk_id=best.chunk_id,
                )
            )

            if len(results) == query.top_k:
                break

        return SearchResponse(
            query=text,
            results=results,
            took_ms=(self._timer() - started) * 1000.0,
            warnings=warnings,
        )

    def _run_channel(
        self,
        mode: SearchMode,
        text: str,
        pool_size: int,
        extensions: frozenset[str] | None,
    ) -> list[ChannelCandidate]:
        if mode is SearchMode.KEYWORD:
            return [
                ChannelCandidate(
                    document_id=hit.document_id,
                    kind=hit.kind,
                    raw_score=hit.score,
                    chunk_id=hit.chunk_id,
                    snippet=hit.text or None,
                    page_number=hit.page_number,
                    absolute=keyword_coverage(text, hit.text, hit.file_name),
                )
                for hit in self._keyword_index.search(text, pool_size, extensions)
            ]

        space = (
            EmbeddingSpace.TEXT_SEMANTIC
            if mode is SearchMode.TEXT_SEMANTIC
            else EmbeddingSpace.MULTIMODAL
        )
        kind = (
            DocumentKind.TEXT
            if mode is SearchMode.TEXT_SEMANTIC
            else DocumentKind.IMAGE
        )

        [query_vector] = self._embedder.embed_texts([text], space=space)
        hits = self._vector_store.query(query_vector, pool_size, extensions)

        return [
            ChannelCandidate(
                document_id=hit.document_id,
                kind=kind,
                raw_score=hit.similarity,
                chunk_id=hit.record_id,
                snippet=hit.text,
                page_number=_optional_int(hit.metadata.get("page_number")),
            )
            for hit in hits
        ]

    @staticmethod
    def _validate(query: SearchQuery) -> str:
        text = query.text.strip()

        if not text:
            raise InvalidQueryError("Search query cannot be empty")

        if len(text) > MAX_QUERY_LENGTH:
            raise InvalidQueryError(
                f"Search query cannot exceed {MAX_QUERY_LENGTH} characters"
            )

        if not 1 <= query.top_k <= MAX_TOP_K:
            raise InvalidQueryError(f"top_k must be between 1 and {MAX_TOP_K}")

        if not query.modes:
            raise InvalidQueryError("At least one search mode is required")

        if (
            query.modified_after is not None
            and query.modified_before is not None
            and _as_utc(query.modified_after) > _as_utc(query.modified_before)
        ):
            raise InvalidQueryError(
                "modified_after must not be later than modified_before"
            )

        return text


def _normalize_extensions(
    extensions: Collection[str] | None,
) -> frozenset[str] | None:
    if not extensions:
        return None

    return frozenset(
        value.lower() if value.startswith(".") else f".{value.lower()}"
        for value in extensions
    )


def _matches_filters(record: ManifestRecord, query: SearchQuery) -> bool:
    if query.path_prefix is not None:
        prefix = query.path_prefix.expanduser().resolve()

        try:
            record.path.relative_to(prefix)
        except ValueError:
            return False

    if query.modified_after is not None and record.modified_at < _as_utc(
        query.modified_after
    ):
        return False

    if query.modified_before is not None and record.modified_at > _as_utc(
        query.modified_before
    ):
        return False

    return True


def _as_utc(value: datetime) -> datetime:
    """Treat naive datetimes as UTC so they compare with stored values."""

    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)

    return value.astimezone(timezone.utc)


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None

    if isinstance(value, int | float):
        return int(value)

    return None
