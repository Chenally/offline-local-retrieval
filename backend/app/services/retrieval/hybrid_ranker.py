from __future__ import annotations

import math
import re

from app.services.embeddings.types import (
    EmbeddingModality,
)
from app.services.retrieval.errors import (
    InvalidRetrievalInputError,
    RetrievalConfigurationError,
)
from app.services.retrieval.types import (
    RankedResult,
    RetrievalCandidates,
    RetrievalMatch,
)

_TOKEN_PATTERN = re.compile(r"\w+")


class HybridRanker:
    """Combines keyword and vector evidence into stable scores."""

    def __init__(
        self,
        *,
        keyword_weight: float = 0.20,
        text_semantic_weight: float = 0.50,
        multimodal_weight: float = 0.30,
    ) -> None:
        weights = (
            keyword_weight,
            text_semantic_weight,
            multimodal_weight,
        )

        if (
            any(
                not math.isfinite(weight)
                or weight < 0.0
                for weight in weights
            )
            or not math.isclose(
                sum(weights),
                1.0,
                abs_tol=1e-9,
            )
        ):
            raise RetrievalConfigurationError(
                "Hybrid ranking weights must be "
                "finite, non-negative, and sum to one"
            )

        self._keyword_weight = keyword_weight
        self._text_semantic_weight = (
            text_semantic_weight
        )
        self._multimodal_weight = (
            multimodal_weight
        )

    def rank(
        self,
        query: str,
        candidates: RetrievalCandidates,
        *,
        limit: int = 10,
    ) -> list[RankedResult]:
        if (
            not isinstance(query, str)
            or not query.strip()
        ):
            raise InvalidRetrievalInputError(
                "Ranking query must not be blank"
            )

        if limit <= 0:
            raise InvalidRetrievalInputError(
                "Ranking limit must be greater than zero"
            )

        query_tokens = self._tokens(query)

        ranked = self._rank_text_candidates(
            query_tokens,
            candidates,
        )
        ranked.extend(
            self._rank_image_candidates(
                candidates
            )
        )

        ranked.sort(
            key=lambda result: (
                -result.final_score,
                result.record_id,
            )
        )

        return ranked[:limit]

    def _rank_text_candidates(
        self,
        query_tokens: set[str],
        candidates: RetrievalCandidates,
    ) -> list[RankedResult]:
        semantic_matches = self._best_matches(
            candidates.text_semantic
        )
        multimodal_matches = self._best_matches(
            candidates.multimodal_text
        )

        record_ids = (
            set(semantic_matches)
            | set(multimodal_matches)
        )

        results: list[RankedResult] = []

        for record_id in record_ids:
            semantic = semantic_matches.get(
                record_id
            )
            multimodal = multimodal_matches.get(
                record_id
            )
            source = semantic or multimodal

            if source is None:
                continue

            keyword_score = self._keyword_score(
                query_tokens,
                source.content,
            )
            semantic_score = (
                self._similarity_score(
                    semantic.similarity
                )
                if semantic is not None
                else 0.0
            )
            multimodal_score = (
                self._similarity_score(
                    multimodal.similarity
                )
                if multimodal is not None
                else 0.0
            )

            final_score = (
                self._keyword_weight
                * keyword_score
                + self._text_semantic_weight
                * semantic_score
                + self._multimodal_weight
                * multimodal_score
            )

            results.append(
                RankedResult(
                    record_id=record_id,
                    content=source.content,
                    source_path=source.source_path,
                    source_sha256=(
                        source.source_sha256
                    ),
                    modality=EmbeddingModality.TEXT,
                    keyword_score=keyword_score,
                    text_semantic_score=(
                        semantic_score
                    ),
                    multimodal_score=(
                        multimodal_score
                    ),
                    final_score=final_score,
                )
            )

        return results

    def _rank_image_candidates(
        self,
        candidates: RetrievalCandidates,
    ) -> list[RankedResult]:
        image_matches = self._best_matches(
            candidates.multimodal_images
        )

        results: list[RankedResult] = []

        for record_id, match in image_matches.items():
            multimodal_score = (
                self._similarity_score(
                    match.similarity
                )
            )

            results.append(
                RankedResult(
                    record_id=record_id,
                    content=match.content,
                    source_path=match.source_path,
                    source_sha256=(
                        match.source_sha256
                    ),
                    modality=EmbeddingModality.IMAGE,
                    keyword_score=0.0,
                    text_semantic_score=0.0,
                    multimodal_score=(
                        multimodal_score
                    ),
                    final_score=multimodal_score,
                )
            )

        return results

    @staticmethod
    def _best_matches(
        matches: tuple[RetrievalMatch, ...],
    ) -> dict[str, RetrievalMatch]:
        best: dict[str, RetrievalMatch] = {}

        for match in matches:
            current = best.get(
                match.record_id
            )

            if (
                current is None
                or match.similarity
                > current.similarity
            ):
                best[match.record_id] = match

        return best

    @classmethod
    def _keyword_score(
        cls,
        query_tokens: set[str],
        content: str,
    ) -> float:
        if not query_tokens:
            return 0.0

        content_tokens = cls._tokens(content)

        return (
            len(
                query_tokens
                & content_tokens
            )
            / len(query_tokens)
        )

    @staticmethod
    def _tokens(value: str) -> set[str]:
        return {
            token.casefold()
            for token in _TOKEN_PATTERN.findall(
                value
            )
        }

    @staticmethod
    def _similarity_score(
        similarity: float,
    ) -> float:
        return max(
            0.0,
            min(1.0, similarity),
        )
