import pytest
from app.models import DocumentKind
from app.services.retrieval.ranking import (
    ChannelCandidate,
    FusionMethod,
    HybridRanker,
    RankingConfig,
    SearchMode,
)

ALL_MODES = frozenset(SearchMode)


def text_hit(
    document_id: str, score: float, snippet: str | None = "text"
) -> ChannelCandidate:
    return ChannelCandidate(
        document_id=document_id,
        kind=DocumentKind.TEXT,
        raw_score=score,
        chunk_id=f"{document_id}:chunk",
        snippet=snippet,
    )


def image_hit(document_id: str, score: float) -> ChannelCandidate:
    return ChannelCandidate(
        document_id=document_id,
        kind=DocumentKind.IMAGE,
        raw_score=score,
        chunk_id=f"{document_id}:image",
    )


def test_collapses_chunks_to_best_chunk_per_document() -> None:
    ranked = HybridRanker().rank(
        {
            SearchMode.TEXT_SEMANTIC: [
                text_hit("a", 0.2, "weak"),
                text_hit("a", 0.9, "strong"),
                text_hit("b", 0.5),
            ]
        },
        {SearchMode.TEXT_SEMANTIC},
    )

    assert [item.document_id for item in ranked] == ["a", "b"]
    assert ranked[0].best_candidate.snippet == "strong"
    assert ranked[0].score == pytest.approx(1.0)
    assert ranked[1].score == pytest.approx(0.0)


def test_weighted_fusion_explains_every_channel() -> None:
    ranked = HybridRanker().rank(
        {
            SearchMode.KEYWORD: [text_hit("a", 8.0), text_hit("b", 2.0)],
            SearchMode.TEXT_SEMANTIC: [text_hit("b", 0.9), text_hit("a", 0.7)],
        },
        {SearchMode.KEYWORD, SearchMode.TEXT_SEMANTIC},
    )

    by_id = {item.document_id: item for item in ranked}
    a = by_id["a"]
    b = by_id["b"]

    # a: keyword 1.0 * 0.3, semantic 0.0 * 0.7 -> 0.3
    # b: keyword 0.0 * 0.3, semantic 1.0 * 0.7 -> 0.7
    assert a.score == pytest.approx(0.3)
    assert b.score == pytest.approx(0.7)
    assert [item.document_id for item in ranked] == ["b", "a"]
    assert a.channel_scores[SearchMode.KEYWORD].raw == 8.0
    assert a.channel_scores[SearchMode.KEYWORD].contribution == pytest.approx(0.3)
    assert b.channel_scores[SearchMode.TEXT_SEMANTIC].rank == 1


def test_images_and_text_share_one_scale() -> None:
    ranked = HybridRanker().rank(
        {
            SearchMode.TEXT_SEMANTIC: [text_hit("doc", 0.8)],
            SearchMode.IMAGE_SEMANTIC: [image_hit("img", 0.3)],
            SearchMode.KEYWORD: [],
        },
        ALL_MODES,
    )

    scores = {item.document_id: item.score for item in ranked}

    # Each document is the best in its only applicable semantic channel and
    # gets 0.7 / (0.3 + 0.7) because the keyword channel found nothing.
    assert scores["doc"] == pytest.approx(0.7)
    assert scores["img"] == pytest.approx(0.7)


def test_rrf_uses_ranks_not_raw_scores() -> None:
    ranker = HybridRanker(RankingConfig(method=FusionMethod.RRF, rrf_k=1))

    ranked = ranker.rank(
        {SearchMode.KEYWORD: [text_hit("a", 100.0), text_hit("b", 99.0)]},
        {SearchMode.KEYWORD},
    )

    assert ranked[0].score == pytest.approx(1.0)
    assert ranked[1].score == pytest.approx((1 / 3) / (1 / 2))


def test_min_raw_score_filters_weak_hits() -> None:
    ranker = HybridRanker(
        RankingConfig(min_raw_scores={SearchMode.IMAGE_SEMANTIC: 0.2})
    )

    ranked = ranker.rank(
        {SearchMode.IMAGE_SEMANTIC: [image_hit("good", 0.3), image_hit("bad", 0.1)]},
        {SearchMode.IMAGE_SEMANTIC},
    )

    assert [item.document_id for item in ranked] == ["good"]


def test_representative_prefers_candidate_with_snippet() -> None:
    ranked = HybridRanker().rank(
        {
            SearchMode.TEXT_SEMANTIC: [text_hit("a", 0.9, snippet=None)],
            SearchMode.KEYWORD: [text_hit("a", 1.0, snippet="keyword snippet")],
        },
        {SearchMode.TEXT_SEMANTIC, SearchMode.KEYWORD},
    )

    assert ranked[0].best_candidate.snippet == "keyword snippet"


def test_unrequested_channels_are_ignored() -> None:
    ranked = HybridRanker().rank(
        {SearchMode.KEYWORD: [text_hit("a", 1.0)]},
        {SearchMode.TEXT_SEMANTIC},
    )

    assert ranked == []


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        RankingConfig(weights={SearchMode.KEYWORD: -1.0})

    with pytest.raises(ValueError):
        RankingConfig(rrf_k=0)
