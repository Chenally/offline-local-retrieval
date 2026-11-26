from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum

from app.models import DocumentKind


class SearchMode(str, Enum):
    """Independent retrieval channels combined by the hybrid ranker."""

    KEYWORD = "keyword"
    TEXT_SEMANTIC = "text_semantic"
    IMAGE_SEMANTIC = "image_semantic"


class FusionMethod(str, Enum):
    WEIGHTED = "weighted"
    RRF = "rrf"


# Which document kinds each channel is able to score.
_APPLICABLE_KINDS: dict[SearchMode, frozenset[DocumentKind]] = {
    SearchMode.KEYWORD: frozenset({DocumentKind.TEXT, DocumentKind.IMAGE}),
    SearchMode.TEXT_SEMANTIC: frozenset({DocumentKind.TEXT}),
    SearchMode.IMAGE_SEMANTIC: frozenset({DocumentKind.IMAGE}),
}

DEFAULT_WEIGHTS: Mapping[SearchMode, float] = {
    SearchMode.KEYWORD: 0.3,
    SearchMode.TEXT_SEMANTIC: 0.7,
    SearchMode.IMAGE_SEMANTIC: 0.7,
}


@dataclass(frozen=True, slots=True)
class RankingConfig:
    weights: Mapping[SearchMode, float] = field(
        default_factory=lambda: dict(DEFAULT_WEIGHTS)
    )
    method: FusionMethod = FusionMethod.WEIGHTED
    rrf_k: int = 60
    # Raw-score floors per channel; hits below the floor are discarded.
    min_raw_scores: Mapping[SearchMode, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if any(weight < 0 for weight in self.weights.values()):
            raise ValueError("Ranking weights must be non-negative")

        if self.rrf_k <= 0:
            raise ValueError("rrf_k must be greater than zero")


@dataclass(frozen=True, slots=True)
class ChannelCandidate:
    """A chunk-level hit returned by one retrieval channel."""

    document_id: str
    kind: DocumentKind
    raw_score: float
    chunk_id: str | None = None
    snippet: str | None = None
    page_number: int | None = None


@dataclass(frozen=True, slots=True)
class ChannelScore:
    """How one channel contributed to a document's final score."""

    raw: float
    normalized: float
    weight: float
    contribution: float
    rank: int


@dataclass(frozen=True, slots=True)
class RankedDocument:
    document_id: str
    kind: DocumentKind
    score: float
    channel_scores: Mapping[SearchMode, ChannelScore]
    best_candidate: ChannelCandidate


class HybridRanker:
    """Explainable fusion of keyword and semantic retrieval channels.

    1. Chunk hits are collapsed to documents by keeping each document's
       best chunk per channel.
    2. Scores are normalized per channel: min-max scaling for the weighted
       method, ``1 / (k + rank)`` for reciprocal rank fusion.
    3. The final score is the weighted sum of normalized scores divided by
       the total weight of the requested channels that *can* score that
       document kind. Dividing by applicable weight keeps images and text
       documents on the same 0..1 scale even though text-semantic search
       cannot score an image and image-semantic search cannot score text.

    Every result keeps its per-channel raw score, normalized score, weight
    and contribution, so the user interface can explain the ranking.
    """

    def __init__(self, config: RankingConfig | None = None) -> None:
        self._config = config or RankingConfig()

    @property
    def config(self) -> RankingConfig:
        return self._config

    def rank(
        self,
        channel_hits: Mapping[SearchMode, Sequence[ChannelCandidate]],
        requested_modes: Collection[SearchMode],
    ) -> list[RankedDocument]:
        per_channel: dict[
            SearchMode, dict[str, tuple[int, ChannelCandidate, float]]
        ] = {}

        for mode in requested_modes:
            best = self._best_per_document(mode, channel_hits.get(mode, ()))
            per_channel[mode] = self._normalize(best)

        documents: dict[str, dict[SearchMode, ChannelScore]] = {}
        candidates: dict[str, list[tuple[float, ChannelCandidate]]] = {}

        for mode, entries in per_channel.items():
            weight = self._config.weights.get(mode, 0.0)

            for document_id, (rank, candidate, normalized) in entries.items():
                contribution = weight * normalized
                documents.setdefault(document_id, {})[mode] = ChannelScore(
                    raw=candidate.raw_score,
                    normalized=normalized,
                    weight=weight,
                    contribution=contribution,
                    rank=rank,
                )
                candidates.setdefault(document_id, []).append((contribution, candidate))

        ranked: list[RankedDocument] = []

        for document_id, scores in documents.items():
            document_candidates = candidates[document_id]
            kind = document_candidates[0][1].kind
            applicable_weight = sum(
                self._config.weights.get(mode, 0.0)
                for mode in requested_modes
                if kind in _APPLICABLE_KINDS[mode]
            )
            total = sum(score.contribution for score in scores.values())
            final = total / applicable_weight if applicable_weight > 0 else 0.0

            ranked.append(
                RankedDocument(
                    document_id=document_id,
                    kind=kind,
                    score=final,
                    channel_scores=scores,
                    best_candidate=self._representative(document_candidates),
                )
            )

        ranked.sort(key=lambda item: (-item.score, item.document_id))
        return ranked

    def _best_per_document(
        self,
        mode: SearchMode,
        hits: Sequence[ChannelCandidate],
    ) -> list[ChannelCandidate]:
        floor = self._config.min_raw_scores.get(mode)
        best: dict[str, ChannelCandidate] = {}

        for hit in hits:
            if floor is not None and hit.raw_score < floor:
                continue

            current = best.get(hit.document_id)

            if current is None or hit.raw_score > current.raw_score:
                best[hit.document_id] = hit

        return sorted(
            best.values(),
            key=lambda hit: (-hit.raw_score, hit.document_id),
        )

    def _normalize(
        self,
        ordered: list[ChannelCandidate],
    ) -> dict[str, tuple[int, ChannelCandidate, float]]:
        if not ordered:
            return {}

        if self._config.method is FusionMethod.RRF:
            # Scale so the top-ranked document receives 1.0.
            top = 1.0 / (self._config.rrf_k + 1)
            return {
                hit.document_id: (
                    rank,
                    hit,
                    (1.0 / (self._config.rrf_k + rank)) / top,
                )
                for rank, hit in enumerate(ordered, start=1)
            }

        highest = ordered[0].raw_score
        lowest = ordered[-1].raw_score
        spread = highest - lowest

        return {
            hit.document_id: (
                rank,
                hit,
                1.0 if spread == 0 else (hit.raw_score - lowest) / spread,
            )
            for rank, hit in enumerate(ordered, start=1)
        }

    @staticmethod
    def _representative(
        scored: list[tuple[float, ChannelCandidate]],
    ) -> ChannelCandidate:
        """Prefer the strongest candidate that has a text snippet."""

        ordered = sorted(scored, key=lambda item: -item[0])

        for _, candidate in ordered:
            if candidate.snippet:
                return candidate

        return ordered[0][1]
