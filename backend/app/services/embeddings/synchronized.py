from __future__ import annotations

import threading
from pathlib import Path

from app.services.embeddings.service import EmbeddingService
from app.services.embeddings.types import EmbeddingSpace, EmbeddingVector


class SynchronizedEmbeddingService:
    """Serializes access to an ``EmbeddingService``.

    LiteRT interpreters are not safe to invoke from several threads at
    once. The API serves search requests while a background worker indexes
    files, so both paths share this wrapper instead of the raw service.
    """

    def __init__(self, service: EmbeddingService) -> None:
        self._service = service
        self._lock = threading.Lock()

    def embed_texts(
        self,
        texts: list[str],
        space: EmbeddingSpace = EmbeddingSpace.TEXT_SEMANTIC,
    ) -> list[EmbeddingVector]:
        with self._lock:
            return self._service.embed_texts(texts, space=space)

    def embed_images(
        self,
        image_paths: list[str | Path],
    ) -> list[EmbeddingVector]:
        with self._lock:
            return self._service.embed_images(image_paths)

    def clear_cache(self) -> None:
        with self._lock:
            self._service.clear_cache()
