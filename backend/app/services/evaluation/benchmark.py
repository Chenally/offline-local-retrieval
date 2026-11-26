"""Run a prepared retrieval benchmark against the real local pipeline."""

from __future__ import annotations

import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from app.services.evaluation.datasets import load_dataset_info, load_queries
from app.services.evaluation.metrics import BenchmarkSummary, summarize
from app.services.indexing.indexing_service import IndexingService
from app.services.retrieval.ranking import SearchMode
from app.services.retrieval.retrieval_service import RetrievalService, SearchQuery


@dataclass(frozen=True, slots=True)
class ModeSetResult:
    modes: tuple[str, ...]
    summary: BenchmarkSummary


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    dataset: str
    corpus_files: int
    indexed_files: int
    index_errors: int
    indexing_seconds: float
    files_per_second: float
    peak_rss_mb: float | None
    top_k: int
    results: list[ModeSetResult]

    def to_dict(self) -> dict:
        data = asdict(self)

        for result in data["results"]:
            summary = result["summary"]
            summary["recall_at"] = {str(k): v for k, v in summary["recall_at"].items()}
            summary["hit_at"] = {str(k): v for k, v in summary["hit_at"].items()}

        return data

    def to_markdown(self) -> str:
        ks = sorted(self.results[0].summary.recall_at) if self.results else []
        header = (
            "| Modes | "
            + " | ".join(f"Recall@{k}" for k in ks)
            + " | MRR | p50 ms | p95 ms |"
        )
        divider = "|" + " --- |" * (len(ks) + 4)
        rows = [
            f"| {' + '.join(item.modes)} | "
            + " | ".join(f"{item.summary.recall_at[k]:.3f}" for k in ks)
            + f" | {item.summary.mrr:.3f}"
            + f" | {item.summary.latency_p50_ms:.1f}"
            + f" | {item.summary.latency_p95_ms:.1f} |"
            for item in self.results
        ]
        rss = f"{self.peak_rss_mb:.0f} MB" if self.peak_rss_mb is not None else "n/a"

        return "\n".join(
            [
                f"## Retrieval benchmark: {self.dataset}",
                "",
                f"- Corpus files: {self.corpus_files} "
                f"(indexed {self.indexed_files}, errors {self.index_errors})",
                f"- Indexing time: {self.indexing_seconds:.1f} s "
                f"({self.files_per_second:.2f} files/s)",
                f"- Peak resident memory: {rss}",
                f"- Queries per mode set: "
                f"{self.results[0].summary.query_count if self.results else 0}",
                "",
                header,
                divider,
                *rows,
                "",
            ]
        )


def run_benchmark(
    dataset_dir: Path,
    indexing_service: IndexingService,
    retrieval_service: RetrievalService,
    mode_sets: Sequence[Sequence[str]] | None = None,
    top_k: int = 10,
) -> BenchmarkReport:
    dataset_dir = dataset_dir.resolve()
    corpus_dir = dataset_dir / "corpus"
    info = load_dataset_info(dataset_dir)
    queries = load_queries(dataset_dir)

    if not queries:
        raise ValueError(f"No queries found in {dataset_dir}")

    started = time.perf_counter()
    report = indexing_service.index_directory(corpus_dir)
    indexing_seconds = time.perf_counter() - started
    processed = len(report.indexed) + len(report.duplicates) + len(report.unchanged)

    chosen_mode_sets = (
        mode_sets or info.get("mode_sets") or [["keyword", "text_semantic"]]
    )
    results: list[ModeSetResult] = []
    ks = sorted({1, 5, top_k})

    for raw_modes in chosen_mode_sets:
        modes = frozenset(SearchMode(mode) for mode in raw_modes)
        rankings: list[list[str]] = []
        relevant_sets: list[set[str]] = []
        latencies: list[float] = []

        for query in queries:
            response = retrieval_service.search(
                SearchQuery(text=query.query, top_k=top_k, modes=modes)
            )
            ranked: list[str] = []

            for result in response.results:
                for path in result.paths:
                    ranked.append(path.relative_to(corpus_dir).as_posix())

            rankings.append(ranked)
            relevant_sets.append(set(query.relevant))
            latencies.append(response.took_ms)

        results.append(
            ModeSetResult(
                modes=tuple(sorted(mode.value for mode in modes)),
                summary=summarize(rankings, relevant_sets, latencies, ks=ks),
            )
        )

    return BenchmarkReport(
        dataset=str(info.get("name", dataset_dir.name)),
        corpus_files=report.scanned,
        indexed_files=processed,
        index_errors=len(report.errors),
        indexing_seconds=indexing_seconds,
        files_per_second=processed / indexing_seconds if indexing_seconds > 0 else 0.0,
        peak_rss_mb=_peak_rss_mb(),
        top_k=top_k,
        results=results,
    )


def _peak_rss_mb() -> float | None:
    try:
        import resource
    except ImportError:  # Windows
        return None

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is bytes on macOS and kilobytes on Linux.
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024
