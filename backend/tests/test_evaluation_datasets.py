import gzip
import json
from pathlib import Path

from app.services.chunking.text_chunker import TextChunker
from app.services.evaluation.benchmark import run_benchmark
from app.services.evaluation.datasets import (
    load_dataset_info,
    load_queries,
    prepare_coco_subset,
    prepare_nq_subset,
)
from app.services.indexing.indexing_service import IndexingService
from app.services.retrieval.retrieval_service import RetrievalService
from app.services.storage.keyword_index import KeywordIndex
from app.services.storage.manifest import IndexManifest

from support import InMemoryVectorStore, make_embedding_service, write_png


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def make_beir(root: Path) -> Path:
    write_jsonl(
        root / "corpus.jsonl",
        [
            {
                "_id": "doc1",
                "title": "Mount Everest",
                "text": "Everest is the highest mountain.",
            },
            {
                "_id": "doc2",
                "title": "Nile",
                "text": "The Nile is a long river in Africa.",
            },
            {"_id": "doc/3", "title": "", "text": "Distractor about cooking pasta."},
            {
                "_id": "doc4",
                "title": "Other",
                "text": "Another distractor about weather.",
            },
        ],
    )
    write_jsonl(
        root / "queries.jsonl",
        [
            {"_id": "q1", "text": "what is the highest mountain"},
            {"_id": "q2", "text": "longest river in africa"},
            {"_id": "q3", "text": "query without judgments"},
        ],
    )
    (root / "qrels").mkdir()
    (root / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq1\tdoc1\t1\nq2\tdoc2\t1\nq2\tdoc4\t0\n",
        encoding="utf-8",
    )
    return root


def test_prepare_nq_subset(tmp_path: Path) -> None:
    beir = make_beir(tmp_path / "beir")
    out = tmp_path / "nq"

    queries, corpus = prepare_nq_subset(beir, out, num_queries=5, num_distractors=1)

    assert queries == 2
    assert corpus == 3
    files = sorted(path.name for path in (out / "corpus").iterdir())
    assert "doc1.txt" in files and "doc2.txt" in files
    assert (
        (out / "corpus" / "doc1.txt")
        .read_text(encoding="utf-8")
        .startswith("Mount Everest")
    )
    loaded = {query.query_id: query for query in load_queries(out)}
    assert loaded["nq-q1"].relevant == ("doc1.txt",)
    assert load_dataset_info(out)["mode_sets"][0] == ["keyword"]


def test_prepare_coco_subset(tmp_path: Path) -> None:
    images = tmp_path / "val2017"
    write_png(images / "000000000001.jpg")
    write_png(images / "000000000002.jpg")
    annotations = tmp_path / "captions.json"
    annotations.write_text(
        json.dumps(
            {
                "images": [
                    {"id": 1, "file_name": "000000000001.jpg"},
                    {"id": 2, "file_name": "000000000002.jpg"},
                    {"id": 3, "file_name": "missing.jpg"},
                ],
                "annotations": [
                    {"image_id": 1, "caption": "A dog on a beach. "},
                    {"image_id": 2, "caption": "Two people riding bikes."},
                    {"image_id": 3, "caption": "Not on disk."},
                ],
            }
        ),
        encoding="utf-8",
    )

    count = prepare_coco_subset(annotations, images, tmp_path / "coco", num_images=10)

    assert count == 2
    queries = load_queries(tmp_path / "coco")
    assert {query.query for query in queries} == {
        "A dog on a beach.",
        "Two people riding bikes.",
    }
    assert load_dataset_info(tmp_path / "missing") == {"name": "missing"}


def test_run_benchmark_reports_metrics(tmp_path: Path) -> None:
    out = tmp_path / "nq"
    prepare_nq_subset(make_beir(tmp_path / "beir"), out, num_distractors=2)

    embedder = make_embedding_service()
    store = InMemoryVectorStore()
    keywords = KeywordIndex(":memory:")
    manifest = IndexManifest(":memory:")
    indexing = IndexingService(
        embedder, store, keywords, manifest, chunker=TextChunker(40, 5)
    )
    retrieval = RetrievalService(embedder, store, keywords, manifest)

    resets: list[int] = []
    report = run_benchmark(
        out,
        indexing,
        retrieval,
        top_k=3,
        reset_query_cache=lambda: resets.append(1),
    )

    assert len(resets) == 3

    assert report.indexed_files == 4
    assert [item.modes for item in report.results] == [
        ("keyword",),
        ("text_semantic",),
        ("keyword", "text_semantic"),
    ]
    hybrid = report.results[-1].summary
    assert hybrid.query_count == 2
    assert hybrid.recall_at[3] == 1.0
    assert "Recall@1" in report.to_markdown()
    assert report.to_dict()["results"][0]["summary"]["recall_at"]["3"] >= 0


def test_prepare_nq_subset_reads_gzip_files(tmp_path: Path) -> None:
    beir = make_beir(tmp_path / "beir")

    for name in ["corpus.jsonl", "queries.jsonl"]:
        plain = beir / name
        with gzip.open(beir / f"{name}.gz", "wb") as compressed:
            compressed.write(plain.read_bytes())
        plain.unlink()

    queries, corpus = prepare_nq_subset(beir, tmp_path / "nq", num_distractors=1)

    assert (queries, corpus) == (2, 3)
