"""Ranking-quality metrics for the retrieval benchmark."""

from __future__ import annotations

import math
from collections.abc import Collection, Sequence
from dataclasses import dataclass


def recall_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float:
    """Fraction of relevant items that appear in the top ``k`` results."""

    if k <= 0:
        raise ValueError("k must be greater than zero")

    if not relevant:
        raise ValueError("relevant must not be empty")

    hits = len(set(ranked[:k]) & set(relevant))
    return hits / len(set(relevant))


def hit_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float:
    """1.0 when at least one relevant item is in the top ``k`` results."""

    if k <= 0:
        raise ValueError("k must be greater than zero")

    return 1.0 if set(ranked[:k]) & set(relevant) else 0.0


def reciprocal_rank(ranked: Sequence[str], relevant: Collection[str]) -> float:
    relevant_set = set(relevant)

    for position, item in enumerate(ranked, start=1):
        if item in relevant_set:
            return 1.0 / position

    return 0.0


def percentile(values: Sequence[float], fraction: float) -> float:
    """Nearest-rank percentile, e.g. ``fraction=0.95`` for p95."""

    if not values:
        raise ValueError("values must not be empty")

    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")

    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


@dataclass(frozen=True, slots=True)
class BenchmarkSummary:
    query_count: int
    recall_at: dict[int, float]
    hit_at: dict[int, float]
    mrr: float
    latency_p50_ms: float
    latency_p95_ms: float


def summarize(
    rankings: Sequence[Sequence[str]],
    relevant_sets: Sequence[Collection[str]],
    latencies_ms: Sequence[float],
    ks: Sequence[int] = (1, 5, 10),
) -> BenchmarkSummary:
    if not rankings:
        raise ValueError("At least one query is required")

    if not len(rankings) == len(relevant_sets) == len(latencies_ms):
        raise ValueError("rankings, relevant_sets and latencies_ms must align")

    count = len(rankings)

    return BenchmarkSummary(
        query_count=count,
        recall_at={
            k: sum(
                recall_at_k(ranked, relevant, k)
                for ranked, relevant in zip(rankings, relevant_sets, strict=True)
            )
            / count
            for k in ks
        },
        hit_at={
            k: sum(
                hit_at_k(ranked, relevant, k)
                for ranked, relevant in zip(rankings, relevant_sets, strict=True)
            )
            / count
            for k in ks
        },
        mrr=sum(
            reciprocal_rank(ranked, relevant)
            for ranked, relevant in zip(rankings, relevant_sets, strict=True)
        )
        / count,
        latency_p50_ms=percentile(latencies_ms, 0.5),
        latency_p95_ms=percentile(latencies_ms, 0.95),
    )
