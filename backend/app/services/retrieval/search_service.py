from __future__ import annotations

from app.services.retrieval.hybrid_ranker import (
    HybridRanker,
)
from app.services.retrieval.retrieval_service import (
    RetrievalService,
)
from app.services.retrieval.types import (
    RankedResult,
)


class SearchService:
    """Runs candidate retrieval followed by hybrid ranking."""

    def __init__(
        self,
        retrieval_service: RetrievalService,
        ranker: HybridRanker | None = None,
    ) -> None:
        self._retrieval_service = retrieval_service
        self._ranker = (
            ranker
            if ranker is not None
            else HybridRanker()
        )

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
    ) -> list[RankedResult]:
        candidates = (
            self._retrieval_service.retrieve(
                query,
                top_k=limit,
            )
        )

        return self._ranker.rank(
            query,
            candidates,
            limit=limit,
        )
