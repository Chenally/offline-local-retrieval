"""Smoke test of the full pipeline with the real LiteRT models.

Skipped automatically when the model files are not provisioned in
``models/`` (or ``OLR_MODELS_DIR``). Run it explicitly with:

    PYTHONPATH=backend python -m pytest backend/tests/test_real_models_e2e.py -v

This test checks that real models flow through indexing and search and
produce sensible, clearly separable rankings. Fine-grained ranking quality
is measured by the NQ and COCO benchmarks, not here.
"""

import math
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("chromadb")
pytest.importorskip("ai_edge_litert")

from app.config import AppSettings  # noqa: E402
from app.container import build_container  # noqa: E402
from app.models import DocumentKind  # noqa: E402
from app.services.retrieval.ranking import SearchMode  # noqa: E402
from app.services.retrieval.retrieval_service import SearchQuery  # noqa: E402

SETTINGS = AppSettings.from_env()
SAMPLE_DATA = Path(__file__).resolve().parents[2] / "sample_data"

REQUIRED = [
    SETTINGS.text_model_path,
    SETTINGS.text_vocab_path,
    SETTINGS.mobileclip_text_model_path,
    SETTINGS.mobileclip_image_model_path,
    SETTINGS.mobileclip_tokenizer_path,
]

pytestmark = pytest.mark.skipif(
    not all(path.is_file() for path in REQUIRED),
    reason="LiteRT model files are not provisioned",
)

# A document on an unrelated topic gives semantic search a clear contrast
# with the two sample files, which both describe offline document search.
RECIPE_TEXT = (
    "Tomato soup recipe. Slowly cook ripe tomatoes with garlic, onion and "
    "fresh basil, then blend until smooth and season with salt and pepper."
)

TEXT_SEMANTIC = frozenset({SearchMode.TEXT_SEMANTIC})
IMAGE_SEMANTIC = frozenset({SearchMode.IMAGE_SEMANTIC})


@pytest.fixture
def container(tmp_path: Path):
    library = tmp_path / "library"
    shutil.copytree(SAMPLE_DATA, library)
    (library / "txt" / "recipe.txt").write_text(RECIPE_TEXT, encoding="utf-8")

    value = build_container(replace(SETTINGS, runtime_dir=tmp_path / "runtime"))
    value.library = library  # type: ignore[attr-defined]
    yield value
    value.close()


def names(response) -> list[str]:
    return [result.file_name for result in response.results]


def test_models_load_and_index_every_sample(container) -> None:
    assert container.component_status["text_semantic"] == "ready"
    assert container.component_status["mobileclip"] == "ready"

    report = container.indexing_service.index_directory(container.library)

    assert report.errors == []
    assert {path.name for path in report.indexed} == {
        "sample.txt",
        "sample.pdf",
        "sample.jpg",
        "recipe.txt",
    }


def test_text_semantic_search_separates_unrelated_topics(container) -> None:
    container.indexing_service.index_directory(container.library)

    cooking = container.retrieval_service.search(
        SearchQuery(text="how do I make tomato soup", modes=TEXT_SEMANTIC)
    )
    search = container.retrieval_service.search(
        SearchQuery(
            text="searching private documents offline without the cloud",
            modes=TEXT_SEMANTIC,
        )
    )

    assert names(cooking)[0] == "recipe.txt"
    assert names(search)[0] in {"sample.txt", "sample.pdf"}
    assert names(search)[-1] == "recipe.txt"

    for result in cooking.results + search.results:
        raw = result.channel_scores[SearchMode.TEXT_SEMANTIC].raw
        assert math.isfinite(raw) and -1.0 <= raw <= 1.0


def test_hybrid_search_uses_keywords_for_exact_terms(container) -> None:
    container.indexing_service.index_directory(container.library)

    keyword_only = container.retrieval_service.search(
        SearchQuery(text="PDFium", modes=frozenset({SearchMode.KEYWORD}))
    )
    hybrid = container.retrieval_service.search(SearchQuery(text="PDFium"))

    assert names(keyword_only) == ["sample.pdf"]
    assert keyword_only.results[0].page_number == 1

    pdf = next(result for result in hybrid.results if result.file_name == "sample.pdf")
    assert SearchMode.KEYWORD in pdf.channel_scores
    assert SearchMode.TEXT_SEMANTIC in pdf.channel_scores
    assert names(hybrid).index("sample.pdf") < names(hybrid).index("recipe.txt")


def test_text_to_image_search_returns_the_image(container) -> None:
    container.indexing_service.index_directory(container.library)

    response = container.retrieval_service.search(
        SearchQuery(text="a photo", modes=IMAGE_SEMANTIC)
    )

    [result] = response.results
    assert result.file_name == "sample.jpg"
    assert result.kind is DocumentKind.IMAGE
    assert math.isfinite(result.channel_scores[SearchMode.IMAGE_SEMANTIC].raw)
