import pytest
from app.services.evaluation.metrics import (
    hit_at_k,
    percentile,
    reciprocal_rank,
    recall_at_k,
    summarize,
)


def test_recall_hit_and_reciprocal_rank() -> None:
    ranked = ["x", "a", "y", "b"]

    assert recall_at_k(ranked, {"a", "b"}, 2) == 0.5
    assert recall_at_k(ranked, {"a", "b"}, 4) == 1.0
    assert hit_at_k(ranked, {"b"}, 3) == 0.0
    assert hit_at_k(ranked, {"b"}, 4) == 1.0
    assert reciprocal_rank(ranked, {"a", "b"}) == 0.5
    assert reciprocal_rank(ranked, {"z"}) == 0.0


def test_percentile_nearest_rank() -> None:
    values = [float(value) for value in range(1, 101)]

    assert percentile(values, 0.5) == 50.0
    assert percentile(values, 0.95) == 95.0
    assert percentile([7.0], 0.95) == 7.0


def test_summarize() -> None:
    summary = summarize(
        rankings=[["a", "x"], ["x", "b"]],
        relevant_sets=[{"a"}, {"b"}],
        latencies_ms=[10.0, 30.0],
        ks=(1, 2),
    )

    assert summary.query_count == 2
    assert summary.recall_at == {1: 0.5, 2: 1.0}
    assert summary.hit_at == {1: 0.5, 2: 1.0}
    assert summary.mrr == pytest.approx(0.75)
    assert summary.latency_p50_ms == 10.0
    assert summary.latency_p95_ms == 30.0


@pytest.mark.parametrize(
    "call",
    [
        lambda: recall_at_k(["a"], {"a"}, 0),
        lambda: recall_at_k(["a"], set(), 1),
        lambda: hit_at_k(["a"], {"a"}, 0),
        lambda: percentile([], 0.5),
        lambda: percentile([1.0], 0.0),
        lambda: summarize([], [], []),
        lambda: summarize([["a"]], [], [1.0]),
    ],
)
def test_invalid_inputs(call) -> None:
    with pytest.raises(ValueError):
        call()
