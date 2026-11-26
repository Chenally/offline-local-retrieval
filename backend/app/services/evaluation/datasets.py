"""Build small retrieval benchmarks from public datasets.

Every prepared dataset has the same layout so one benchmark runner can
evaluate text and image retrieval:

    <out>/
        corpus/            files to index (TXT for NQ, JPG for COCO)
        queries.jsonl      {"query_id", "query", "relevant": [relative paths]}
        dataset.json       {"name", "source", "mode_sets", ...}
"""

from __future__ import annotations

import csv
import json
import random
import re
import shutil
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class BenchmarkQuery:
    query_id: str
    query: str
    relevant: tuple[str, ...]


def load_queries(dataset_dir: Path) -> list[BenchmarkQuery]:
    queries: list[BenchmarkQuery] = []

    with (dataset_dir / "queries.jsonl").open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            item = json.loads(line)
            queries.append(
                BenchmarkQuery(
                    query_id=str(item["query_id"]),
                    query=str(item["query"]),
                    relevant=tuple(item["relevant"]),
                )
            )

    return queries


def load_dataset_info(dataset_dir: Path) -> dict:
    path = dataset_dir / "dataset.json"

    if not path.is_file():
        return {"name": dataset_dir.name}

    return json.loads(path.read_text(encoding="utf-8"))


def prepare_coco_subset(
    annotations_path: Path,
    images_dir: Path,
    output_dir: Path,
    num_images: int = 500,
    seed: int = 42,
) -> int:
    """Sample COCO images and use one caption per image as its query.

    ``annotations_path`` is ``captions_val2017.json`` and ``images_dir`` is
    the extracted ``val2017`` folder from https://cocodataset.org.
    Images are copied, not linked, because the scanner skips symlinks.
    """

    data = json.loads(annotations_path.read_text(encoding="utf-8"))
    captions: dict[int, list[str]] = {}

    for annotation in data["annotations"]:
        captions.setdefault(annotation["image_id"], []).append(
            annotation["caption"].strip()
        )

    images = [
        image
        for image in data["images"]
        if image["id"] in captions and (images_dir / image["file_name"]).is_file()
    ]
    images.sort(key=lambda image: image["id"])
    rng = random.Random(seed)
    selected = rng.sample(images, min(num_images, len(images)))

    corpus_dir = _reset_output(output_dir)

    with (output_dir / "queries.jsonl").open("w", encoding="utf-8") as queries:
        for image in selected:
            file_name = image["file_name"]
            shutil.copy2(images_dir / file_name, corpus_dir / file_name)
            caption = sorted(captions[image["id"]])[0]
            queries.write(
                json.dumps(
                    {
                        "query_id": f"coco-{image['id']}",
                        "query": caption,
                        "relevant": [file_name],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    _write_info(
        output_dir,
        name="coco-val2017-subset",
        source="COCO 2017 validation captions",
        mode_sets=[["image_semantic"], ["keyword", "image_semantic"]],
        corpus_size=len(selected),
        query_count=len(selected),
        seed=seed,
    )
    return len(selected)


def prepare_nq_subset(
    beir_dir: Path,
    output_dir: Path,
    num_queries: int = 300,
    num_distractors: int = 2000,
    split: str = "test",
    seed: int = 42,
) -> tuple[int, int]:
    """Sample Natural Questions queries plus distractor passages.

    ``beir_dir`` is the BEIR packaging of NQ (``corpus.jsonl``,
    ``queries.jsonl`` and ``qrels/<split>.tsv``). The full corpus has
    millions of passages, so it is streamed once and only the relevant
    passages plus a reservoir sample of distractors are written.
    """

    qrels: dict[str, set[str]] = {}

    with (beir_dir / "qrels" / f"{split}.tsv").open(encoding="utf-8") as file:
        reader = csv.reader(file, delimiter="\t")
        next(reader, None)  # header: query-id, corpus-id, score

        for row in reader:
            if len(row) >= 3 and int(row[2]) > 0:
                qrels.setdefault(row[0], set()).add(row[1])

    query_text: dict[str, str] = {}

    for item in _iter_jsonl(beir_dir / "queries.jsonl"):
        if item["_id"] in qrels:
            query_text[item["_id"]] = item["text"]

    rng = random.Random(seed)
    query_ids = sorted(query_id for query_id in qrels if query_id in query_text)
    selected = rng.sample(query_ids, min(num_queries, len(query_ids)))
    relevant_ids = set().union(*(qrels[query_id] for query_id in selected))

    corpus_dir = _reset_output(output_dir)
    reservoir: list[dict] = []
    seen_distractors = 0
    written: set[str] = set()

    for item in _iter_jsonl(beir_dir / "corpus.jsonl"):
        document_id = item["_id"]

        if document_id in relevant_ids:
            _write_passage(corpus_dir, item)
            written.add(document_id)
            continue

        seen_distractors += 1

        if len(reservoir) < num_distractors:
            reservoir.append(item)
        else:
            slot = rng.randrange(seen_distractors)

            if slot < num_distractors:
                reservoir[slot] = item

    for item in reservoir:
        _write_passage(corpus_dir, item)

    with (output_dir / "queries.jsonl").open("w", encoding="utf-8") as queries:
        for query_id in selected:
            relevant = sorted(
                _passage_file_name(document_id)
                for document_id in qrels[query_id]
                if document_id in written
            )

            if not relevant:
                continue

            queries.write(
                json.dumps(
                    {
                        "query_id": f"nq-{query_id}",
                        "query": query_text[query_id],
                        "relevant": relevant,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    corpus_size = len(written) + len(reservoir)
    _write_info(
        output_dir,
        name="nq-beir-subset",
        source=f"Natural Questions (BEIR, {split} split)",
        mode_sets=[["keyword"], ["text_semantic"], ["keyword", "text_semantic"]],
        corpus_size=corpus_size,
        query_count=len(selected),
        seed=seed,
    )
    return len(selected), corpus_size


def _passage_file_name(document_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", document_id) + ".txt"


def _write_passage(corpus_dir: Path, item: dict) -> None:
    title = str(item.get("title") or "").strip()
    text = str(item.get("text") or "").strip()
    body = f"{title}\n\n{text}" if title else text
    (corpus_dir / _passage_file_name(item["_id"])).write_text(body, encoding="utf-8")


def _iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


def _reset_output(output_dir: Path) -> Path:
    corpus_dir = output_dir / "corpus"

    if corpus_dir.exists():
        shutil.rmtree(corpus_dir)

    corpus_dir.mkdir(parents=True)
    return corpus_dir


def _write_info(output_dir: Path, **info: object) -> None:
    (output_dir / "dataset.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
