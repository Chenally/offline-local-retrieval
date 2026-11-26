"""Deterministic test doubles shared by Week 4 tests.

The hashing backends produce bag-of-words vectors, so semantic search in
tests behaves predictably (shared words => higher cosine similarity)
without loading LiteRT models.
"""

from __future__ import annotations

import hashlib
from collections.abc import Collection, Sequence
from pathlib import Path

import numpy as np
from app.services.embeddings.base import EmbeddingInput
from app.services.embeddings.service import EmbeddingService
from app.services.embeddings.types import (
    EmbeddingModality,
    EmbeddingSpace,
    EmbeddingVector,
)
from app.services.storage.vector_store import VectorHit, VectorRecord
from app.services.text_units import iter_search_terms

FAKE_DIMENSION = 64
FAKE_TEXT_MODEL = "fake-bert"
FAKE_CLIP_MODEL = "fake-clip"


def bag_of_words(text: str, dimension: int = FAKE_DIMENSION) -> np.ndarray:
    vector = np.zeros(dimension, dtype=np.float32)
    vector[0] = 1e-3  # avoid zero vectors for inputs without words

    for term in iter_search_terms(text.lower()):
        digest = hashlib.sha1(term.encode("utf-8")).digest()
        vector[1 + digest[0] % (dimension - 1)] += 1.0

    return vector


class HashingTextBackend:
    modality = EmbeddingModality.TEXT

    def __init__(self, space: EmbeddingSpace, model_id: str) -> None:
        self.space = space
        self.model_id = model_id
        self.calls: list[int] = []

    def embed_batch(self, inputs: list[EmbeddingInput]) -> np.ndarray:
        self.calls.append(len(inputs))
        return np.stack([bag_of_words(str(value)) for value in inputs])


class HashingImageBackend:
    """Embeds an image from the words in its file name."""

    modality = EmbeddingModality.IMAGE
    space = EmbeddingSpace.MULTIMODAL

    def __init__(self, model_id: str = FAKE_CLIP_MODEL) -> None:
        self.model_id = model_id
        self.calls: list[int] = []

    def embed_batch(self, inputs: list[EmbeddingInput]) -> np.ndarray:
        self.calls.append(len(inputs))
        return np.stack(
            [
                bag_of_words(Path(value).stem.replace("_", " ").replace("-", " "))
                for value in inputs
            ]
        )


def make_backends() -> list:
    return [
        HashingTextBackend(EmbeddingSpace.TEXT_SEMANTIC, FAKE_TEXT_MODEL),
        HashingTextBackend(EmbeddingSpace.MULTIMODAL, FAKE_CLIP_MODEL),
        HashingImageBackend(FAKE_CLIP_MODEL),
    ]


def make_embedding_service() -> EmbeddingService:
    return EmbeddingService(make_backends())


class InMemoryVectorStore:
    """Brute-force VectorStore used where Chroma is not under test."""

    def __init__(self) -> None:
        self.records: dict[EmbeddingSpace, dict[str, VectorRecord]] = {}
        self.fail_on_upsert: bool = False

    def upsert(self, records: Sequence[VectorRecord]) -> None:
        if self.fail_on_upsert:
            from app.services.storage.errors import StorageError

            raise StorageError("simulated vector store failure")

        for record in records:
            space = record.embedding.space
            self.records.setdefault(space, {})[record.record_id] = record

    def delete_document(self, document_id: str) -> None:
        for space_records in self.records.values():
            for record_id in [
                key
                for key, record in space_records.items()
                if record.document_id == document_id
            ]:
                del space_records[record_id]

    def query(
        self,
        embedding: EmbeddingVector,
        n_results: int,
        extensions: Collection[str] | None = None,
    ) -> list[VectorHit]:
        query = np.asarray(embedding.values, dtype=np.float32)
        hits: list[VectorHit] = []

        for record in self.records.get(embedding.space, {}).values():
            if extensions and record.metadata.get("extension") not in extensions:
                continue

            hits.append(
                VectorHit(
                    record_id=record.record_id,
                    document_id=record.document_id,
                    similarity=float(
                        np.asarray(record.embedding.values, dtype=np.float32) @ query
                    ),
                    metadata={
                        key: value
                        for key, value in record.metadata.items()
                        if value is not None
                    },
                    text=record.text,
                )
            )

        hits.sort(key=lambda hit: (-hit.similarity, hit.record_id))
        return hits[:n_results]

    def count(self, space: EmbeddingSpace) -> int:
        return len(self.records.get(space, {}))

    def document_ids(self, space: EmbeddingSpace) -> set[str]:
        return {record.document_id for record in self.records.get(space, {}).values()}


def write_png(path: Path, color: tuple[int, int, int] = (200, 30, 30)) -> Path:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color).save(path)
    return path


def write_docx(path: Path, paragraphs: Sequence[str]) -> Path:
    from docx import Document

    path.parent.mkdir(parents=True, exist_ok=True)
    document = Document()

    for paragraph in paragraphs:
        document.add_paragraph(paragraph)

    document.save(str(path))
    return path
