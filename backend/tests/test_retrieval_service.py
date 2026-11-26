import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from app.models import DocumentKind
from app.services.chunking.text_chunker import TextChunker
from app.services.embeddings.service import EmbeddingService
from app.services.embeddings.types import EmbeddingSpace
from app.services.indexing.indexing_service import IndexingService
from app.services.retrieval.ranking import SearchMode
from app.services.retrieval.retrieval_service import (
    InvalidQueryError,
    RetrievalService,
    SearchQuery,
)
from app.services.storage.keyword_index import KeywordIndex
from app.services.storage.manifest import IndexManifest

from support import (
    FAKE_TEXT_MODEL,
    HashingTextBackend,
    InMemoryVectorStore,
    make_embedding_service,
    write_png,
)


@pytest.fixture
def library(tmp_path: Path) -> Path:
    root = tmp_path / "library"
    (root / "work").mkdir(parents=True)
    (root / "work" / "budget.txt").write_text(
        "The project budget covers model conversion and accessibility testing.",
        encoding="utf-8",
    )
    (root / "recipes.txt").write_text(
        "Slow cooked tomato soup with basil and garlic.",
        encoding="utf-8",
    )
    (root / "copy_of_recipes.txt").write_text(
        "Slow cooked tomato soup with basil and garlic.",
        encoding="utf-8",
    )
    write_png(root / "red_car.png")
    write_png(root / "mountain_lake.png", color=(10, 120, 200))
    return root


def build(library: Path, embedder=None):
    embedder = embedder or make_embedding_service()
    vectors = InMemoryVectorStore()
    keywords = KeywordIndex(":memory:")
    manifest = IndexManifest(":memory:")
    IndexingService(
        embedder=embedder,
        vector_store=vectors,
        keyword_index=keywords,
        manifest=manifest,
        chunker=TextChunker(max_units=20, overlap_units=5),
    ).index_directory(library)
    retrieval = RetrievalService(
        embedder=embedder,
        vector_store=vectors,
        keyword_index=keywords,
        manifest=manifest,
    )
    return retrieval, manifest


def test_hybrid_search_returns_explained_text_result(library: Path) -> None:
    retrieval, _ = build(library)

    response = retrieval.search(SearchQuery(text="project budget", top_k=3))

    top = response.results[0]
    assert top.file_name == "budget.txt"
    assert top.kind is DocumentKind.TEXT
    assert top.snippet is not None and "budget" in top.snippet
    assert set(top.channel_scores) == {SearchMode.KEYWORD, SearchMode.TEXT_SEMANTIC}
    assert 0.0 < top.score <= 1.0
    assert response.warnings == []


def test_text_to_image_search_finds_image(library: Path) -> None:
    retrieval, _ = build(library)

    response = retrieval.search(
        SearchQuery(text="a red car", modes=frozenset({SearchMode.IMAGE_SEMANTIC}))
    )

    assert response.results[0].file_name == "red_car.png"
    assert response.results[0].kind is DocumentKind.IMAGE
    assert response.results[0].snippet is None


def test_duplicate_files_are_one_result_with_all_paths(library: Path) -> None:
    retrieval, _ = build(library)

    response = retrieval.search(SearchQuery(text="tomato soup", top_k=1))

    [result] = response.results
    assert sorted(path.name for path in result.paths) == [
        "copy_of_recipes.txt",
        "recipes.txt",
    ]


def test_extension_filter(library: Path) -> None:
    retrieval, _ = build(library)

    response = retrieval.search(
        SearchQuery(text="red car budget", extensions=frozenset({"PNG"}))
    )

    assert {result.extension for result in response.results} == {".png"}


def test_path_prefix_filter(library: Path) -> None:
    retrieval, _ = build(library)

    response = retrieval.search(
        SearchQuery(
            text="budget soup",
            modes=frozenset({SearchMode.KEYWORD}),
            path_prefix=library / "work",
        )
    )

    assert [result.file_name for result in response.results] == ["budget.txt"]


def test_modified_date_filter(library: Path) -> None:
    old_file = library / "recipes.txt"
    old_time = (datetime.now(timezone.utc) - timedelta(days=400)).timestamp()
    os.utime(old_file, (old_time, old_time))
    os.utime(library / "copy_of_recipes.txt", (old_time, old_time))
    retrieval, _ = build(library)

    cutoff = datetime.now(timezone.utc) - timedelta(days=30)
    recent = retrieval.search(
        SearchQuery(text="tomato soup", modified_after=cutoff.replace(tzinfo=None))
    )
    older = retrieval.search(SearchQuery(text="tomato soup", modified_before=cutoff))

    assert "recipes.txt" not in {result.file_name for result in recent.results}
    assert older.results[0].file_name in {"recipes.txt", "copy_of_recipes.txt"}


def test_missing_model_disables_only_its_channel(library: Path) -> None:
    text_only = EmbeddingService(
        [HashingTextBackend(EmbeddingSpace.TEXT_SEMANTIC, FAKE_TEXT_MODEL)]
    )
    full_retrieval, _ = build(library)
    retrieval = RetrievalService(
        embedder=text_only,
        vector_store=full_retrieval._vector_store,
        keyword_index=full_retrieval._keyword_index,
        manifest=full_retrieval._manifest,
    )

    response = retrieval.search(SearchQuery(text="budget"))

    assert response.results[0].file_name == "budget.txt"
    assert len(response.warnings) == 1
    assert response.warnings[0].startswith("image_semantic search unavailable")


def test_top_k_limits_results(library: Path) -> None:
    retrieval, _ = build(library)

    response = retrieval.search(SearchQuery(text="soup budget car lake", top_k=2))

    assert len(response.results) == 2
    assert response.took_ms >= 0


@pytest.mark.parametrize(
    "query",
    [
        SearchQuery(text="   "),
        SearchQuery(text="x" * 1001),
        SearchQuery(text="ok", top_k=0),
        SearchQuery(text="ok", top_k=101),
        SearchQuery(text="ok", modes=frozenset()),
        SearchQuery(
            text="ok",
            modified_after=datetime(2026, 2, 1, tzinfo=timezone.utc),
            modified_before=datetime(2026, 1, 1, tzinfo=timezone.utc),
        ),
    ],
)
def test_rejects_invalid_queries(library: Path, query: SearchQuery) -> None:
    retrieval, _ = build(library)

    with pytest.raises(InvalidQueryError):
        retrieval.search(query)


def test_rejects_invalid_candidate_multiplier() -> None:
    with pytest.raises(ValueError):
        RetrievalService(
            embedder=make_embedding_service(),
            vector_store=InMemoryVectorStore(),
            keyword_index=KeywordIndex(":memory:"),
            manifest=IndexManifest(":memory:"),
            candidate_multiplier=0,
        )
