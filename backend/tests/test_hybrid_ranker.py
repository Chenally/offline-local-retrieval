from pathlib import Path

import pytest
from app.services.embeddings.types import (
    EmbeddingModality,
    EmbeddingSpace,
)
from app.services.retrieval.errors import (
    InvalidRetrievalInputError,
    RetrievalConfigurationError,
)
from app.services.retrieval.hybrid_ranker import (
    HybridRanker,
)
from app.services.retrieval.types import (
    RetrievalCandidates,
    RetrievalMatch,
)


def _match(
    record_id: str,
    *,
    content: str,
    similarity: float,
    modality: EmbeddingModality,
    space: EmbeddingSpace,
) -> RetrievalMatch:
    return RetrievalMatch(
        record_id=record_id,
        content=content,
        source_path=Path(
            f"/local/{record_id}"
        ),
        source_sha256=f"sha-{record_id}",
        content_id=f"content-{record_id}",
        modality=modality,
        space=space,
        model_id=(
            "bert"
            if space
            is EmbeddingSpace.TEXT_SEMANTIC
            else "mobileclip"
        ),
        distance=1.0 - similarity,
        similarity=similarity,
    )


def test_ranks_text_and_images_with_explainable_scores(
) -> None:
    candidates = RetrievalCandidates(
        text_semantic=(
            _match(
                "text:one",
                content="Local retrieval guide",
                similarity=0.8,
                modality=EmbeddingModality.TEXT,
                space=(
                    EmbeddingSpace.TEXT_SEMANTIC
                ),
            ),
            _match(
                "text:two",
                content="Unrelated content",
                similarity=0.9,
                modality=EmbeddingModality.TEXT,
                space=(
                    EmbeddingSpace.TEXT_SEMANTIC
                ),
            ),
        ),
        multimodal_text=(
            _match(
                "text:one",
                content="Local retrieval guide",
                similarity=0.6,
                modality=EmbeddingModality.TEXT,
                space=EmbeddingSpace.MULTIMODAL,
            ),
        ),
        multimodal_images=(
            _match(
                "image:one",
                content="/local/image.png",
                similarity=0.7,
                modality=EmbeddingModality.IMAGE,
                space=EmbeddingSpace.MULTIMODAL,
            ),
        ),
    )

    results = HybridRanker().rank(
        "local retrieval",
        candidates,
    )

    assert [
        result.record_id
        for result in results
    ] == [
        "text:one",
        "image:one",
        "text:two",
    ]

    first = results[0]

    assert first.keyword_score == 1.0
    assert first.text_semantic_score == 0.8
    assert first.multimodal_score == 0.6
    assert first.final_score == pytest.approx(
        0.78
    )

    image = results[1]

    assert image.keyword_score == 0.0
    assert image.text_semantic_score == 0.0
    assert image.multimodal_score == 0.7
    assert image.final_score == 0.7

    text_without_multimodal = results[2]

    assert (
        text_without_multimodal.multimodal_score
        == 0.0
    )
    assert (
        text_without_multimodal.final_score
        == pytest.approx(0.45)
    )


def test_keeps_highest_duplicate_similarity() -> None:
    candidates = RetrievalCandidates(
        text_semantic=(
            _match(
                "text:duplicate",
                content="local",
                similarity=0.2,
                modality=EmbeddingModality.TEXT,
                space=(
                    EmbeddingSpace.TEXT_SEMANTIC
                ),
            ),
            _match(
                "text:duplicate",
                content="local",
                similarity=0.8,
                modality=EmbeddingModality.TEXT,
                space=(
                    EmbeddingSpace.TEXT_SEMANTIC
                ),
            ),
        ),
        multimodal_text=(),
        multimodal_images=(),
    )

    results = HybridRanker().rank(
        "local",
        candidates,
    )

    assert len(results) == 1
    assert (
        results[0].text_semantic_score
        == 0.8
    )
    assert results[0].final_score == (
        pytest.approx(0.6)
    )


def test_uses_record_id_as_stable_tie_breaker(
) -> None:
    candidates = RetrievalCandidates(
        text_semantic=(),
        multimodal_text=(),
        multimodal_images=(
            _match(
                "image:b",
                content="/local/b.png",
                similarity=0.5,
                modality=EmbeddingModality.IMAGE,
                space=EmbeddingSpace.MULTIMODAL,
            ),
            _match(
                "image:a",
                content="/local/a.png",
                similarity=0.5,
                modality=EmbeddingModality.IMAGE,
                space=EmbeddingSpace.MULTIMODAL,
            ),
        ),
    )

    results = HybridRanker().rank(
        "image",
        candidates,
    )

    assert [
        result.record_id
        for result in results
    ] == [
        "image:a",
        "image:b",
    ]


def test_empty_candidates_return_empty_results(
) -> None:
    candidates = RetrievalCandidates(
        text_semantic=(),
        multimodal_text=(),
        multimodal_images=(),
    )

    assert (
        HybridRanker().rank(
            "local retrieval",
            candidates,
        )
        == []
    )


@pytest.mark.parametrize(
    "query",
    [
        "",
        "   ",
    ],
)
def test_rejects_blank_query(
    query: str,
) -> None:
    candidates = RetrievalCandidates(
        text_semantic=(),
        multimodal_text=(),
        multimodal_images=(),
    )

    with pytest.raises(
        InvalidRetrievalInputError
    ):
        HybridRanker().rank(
            query,
            candidates,
        )


@pytest.mark.parametrize(
    "limit",
    [
        0,
        -1,
    ],
)
def test_rejects_invalid_limit(
    limit: int,
) -> None:
    candidates = RetrievalCandidates(
        text_semantic=(),
        multimodal_text=(),
        multimodal_images=(),
    )

    with pytest.raises(
        InvalidRetrievalInputError
    ):
        HybridRanker().rank(
            "local retrieval",
            candidates,
            limit=limit,
        )


@pytest.mark.parametrize(
    (
        "keyword_weight",
        "text_semantic_weight",
        "multimodal_weight",
    ),
    [
        (-0.1, 0.8, 0.3),
        (0.2, 0.5, 0.2),
        (float("nan"), 0.5, 0.5),
    ],
)
def test_rejects_invalid_weights(
    keyword_weight: float,
    text_semantic_weight: float,
    multimodal_weight: float,
) -> None:
    with pytest.raises(
        RetrievalConfigurationError
    ):
        HybridRanker(
            keyword_weight=keyword_weight,
            text_semantic_weight=(
                text_semantic_weight
            ),
            multimodal_weight=(
                multimodal_weight
            ),
        )
