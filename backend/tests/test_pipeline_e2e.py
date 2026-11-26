"""End-to-end pipeline test with real Chroma, SQLite and parsers.

Only the LiteRT models are replaced by deterministic hashing backends, so
this runs in CI without model files. ``test_real_models_e2e.py`` repeats
the flow with the real BERT and MobileCLIP models when they are present.
"""

from pathlib import Path

import pytest

pytest.importorskip("chromadb")

from app.config import AppSettings  # noqa: E402
from app.container import build_container  # noqa: E402
from app.services.embeddings.types import EmbeddingSpace  # noqa: E402
from app.services.retrieval.ranking import RankingConfig, SearchMode  # noqa: E402
from app.services.retrieval.retrieval_service import SearchQuery  # noqa: E402

from support import FAKE_DIMENSION, make_backends, write_docx, write_png  # noqa: E402


def make_settings(runtime: Path) -> AppSettings:
    return AppSettings(
        runtime_dir=runtime,
        models_dir=runtime / "models",
        ranking=RankingConfig(),
        chunk_max_units=30,
        chunk_overlap_units=5,
    )


def open_container(runtime: Path):
    return build_container(
        make_settings(runtime),
        backends=make_backends(),
        embedding_dimension=FAKE_DIMENSION,
    )


def test_ingest_parse_embed_store_search(tmp_path: Path) -> None:
    library = tmp_path / "library"
    library.mkdir()
    (library / "privacy.txt").write_text(
        "Local files never leave the device. Privacy is enforced offline.",
        encoding="utf-8",
    )
    write_docx(
        library / "roadmap.docx", ["Week five builds the accessible Flutter interface"]
    )
    write_png(library / "sunset_beach.png")

    container = open_container(tmp_path / "runtime")

    try:
        report = container.indexing_service.index_directory(library)

        assert len(report.indexed) == 3
        assert report.errors == []
        assert container.vector_store.count(EmbeddingSpace.MULTIMODAL) == 1

        privacy = container.retrieval_service.search(
            SearchQuery(text="privacy offline device")
        )
        assert privacy.results[0].file_name == "privacy.txt"

        roadmap = container.retrieval_service.search(
            SearchQuery(
                text="accessible flutter interface", extensions=frozenset({"docx"})
            )
        )
        assert [result.file_name for result in roadmap.results] == ["roadmap.docx"]

        beach = container.retrieval_service.search(
            SearchQuery(
                text="beach sunset", modes=frozenset({SearchMode.IMAGE_SEMANTIC})
            )
        )
        assert beach.results[0].file_name == "sunset_beach.png"
    finally:
        container.close()


def test_index_survives_restart_and_stays_incremental(tmp_path: Path) -> None:
    library = tmp_path / "library"
    library.mkdir()
    (library / "notes.txt").write_text("persistent hybrid index", encoding="utf-8")
    runtime = tmp_path / "runtime"

    first = open_container(runtime)
    first.indexing_service.index_directory(library)
    first.close()

    second = open_container(runtime)

    try:
        report = second.indexing_service.index_directory(library)
        assert report.indexed == []
        assert len(report.unchanged) == 1

        response = second.retrieval_service.search(SearchQuery(text="hybrid index"))
        assert response.results[0].file_name == "notes.txt"
    finally:
        second.close()
