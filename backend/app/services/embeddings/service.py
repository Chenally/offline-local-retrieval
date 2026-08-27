from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from app.services.embeddings.base import (
    EmbeddingBackend,
    EmbeddingInput,
)
from app.services.embeddings.cache import EmbeddingCache
from app.services.embeddings.errors import (
    EmbeddingBackendError,
    EmbeddingConfigurationError,
    EmbeddingError,
    InvalidEmbeddingInputError,
    InvalidEmbeddingVectorError,
)
from app.services.embeddings.normalization import l2_normalize
from app.services.embeddings.types import (
    EmbeddingModality,
    EmbeddingSpace,
    EmbeddingVector,
)


class EmbeddingService:
    """Provides normalized, cached, batched text and image embeddings."""

    def __init__(
        self,
        backends: Iterable[EmbeddingBackend],
        cache: EmbeddingCache | None = None,
        batch_size: int = 16,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        self._batch_size = batch_size
        self._cache = cache or EmbeddingCache()
        self._backends: dict[
            tuple[EmbeddingSpace, EmbeddingModality],
            EmbeddingBackend,
        ] = {}

        for backend in backends:
            key = (backend.space, backend.modality)

            if not backend.model_id.strip():
                raise EmbeddingConfigurationError(
                    "Embedding backend model_id cannot be empty"
                )

            if key in self._backends:
                raise EmbeddingConfigurationError(
                    "Only one backend can be registered for "
                    f"{backend.space.value}/{backend.modality.value}"
                )

            self._backends[key] = backend

    def embed_texts(
        self,
        texts: list[str],
        space: EmbeddingSpace = EmbeddingSpace.TEXT_SEMANTIC,
    ) -> list[EmbeddingVector]:
        return self._embed(
            inputs=texts,
            modality=EmbeddingModality.TEXT,
            space=space,
        )

    def embed_images(
        self,
        image_paths: list[str | Path],
    ) -> list[EmbeddingVector]:
        paths = [
            Path(path).expanduser().resolve()
            for path in image_paths
        ]

        return self._embed(
            inputs=paths,
            modality=EmbeddingModality.IMAGE,
            space=EmbeddingSpace.MULTIMODAL,
        )

    def _embed(
        self,
        inputs: list[EmbeddingInput],
        modality: EmbeddingModality,
        space: EmbeddingSpace,
    ) -> list[EmbeddingVector]:
        backend = self._get_backend(space, modality)

        if not inputs:
            return []

        prepared = [
            self._prepare_input(value, modality)
            for value in inputs
        ]

        results: list[EmbeddingVector | None] = [
            None
        ] * len(prepared)

        pending: dict[
            str,
            tuple[str, EmbeddingInput, list[int]],
        ] = {}

        for index, (value, content_id) in enumerate(prepared):
            cache_key = self._cache_key(
                backend.model_id,
                modality,
                space,
                content_id,
            )

            cached_values = self._cache.get(cache_key)

            if cached_values is not None:
                results[index] = self._result(
                    content_id=content_id,
                    values=cached_values,
                    backend=backend,
                )
                continue

            if cache_key in pending:
                pending[cache_key][2].append(index)
            else:
                pending[cache_key] = (
                    content_id,
                    value,
                    [index],
                )

        pending_items = list(pending.items())

        for start in range(
            0,
            len(pending_items),
            self._batch_size,
        ):
            batch = pending_items[
                start : start + self._batch_size
            ]

            batch_inputs = [
                item[1][1]
                for item in batch
            ]

            try:
                raw_vectors = np.asarray(
                    backend.embed_batch(batch_inputs),
                    dtype=np.float32,
                )
            except EmbeddingError:
                raise
            except Exception as exc:
                raise EmbeddingBackendError(
                    f"Embedding backend "
                    f"{backend.model_id} failed: {exc}"
                ) from exc

            if (
                raw_vectors.ndim != 2
                or raw_vectors.shape[0] != len(batch)
            ):
                raise InvalidEmbeddingVectorError(
                    f"Backend {backend.model_id} returned "
                    f"shape {raw_vectors.shape}; expected "
                    f"one vector per input"
                )

            for row, (
                cache_key,
                pending_value,
            ) in enumerate(batch):
                content_id, _, result_indices = pending_value

                normalized = l2_normalize(
                    raw_vectors[row]
                )

                self._cache.put(
                    cache_key,
                    normalized,
                )

                for index in result_indices:
                    results[index] = self._result(
                        content_id=content_id,
                        values=normalized,
                        backend=backend,
                    )

        if any(result is None for result in results):
            raise InvalidEmbeddingVectorError(
                "Embedding backend did not produce "
                "every requested result"
            )

        return [
            result
            for result in results
            if result is not None
        ]

    def _get_backend(
        self,
        space: EmbeddingSpace,
        modality: EmbeddingModality,
    ) -> EmbeddingBackend:
        backend = self._backends.get(
            (space, modality)
        )

        if backend is None:
            raise EmbeddingConfigurationError(
                "No embedding backend is registered for "
                f"{space.value}/{modality.value}"
            )

        return backend

    @staticmethod
    def _prepare_input(
        value: EmbeddingInput,
        modality: EmbeddingModality,
    ) -> tuple[EmbeddingInput, str]:
        if modality is EmbeddingModality.TEXT:
            if (
                not isinstance(value, str)
                or not value.strip()
            ):
                raise InvalidEmbeddingInputError(
                    "Text embedding input must be "
                    "a non-empty string"
                )

            content = value.encode("utf-8")

            return (
                value,
                hashlib.sha256(content).hexdigest(),
            )

        if (
            not isinstance(value, Path)
            or not value.is_file()
        ):
            raise InvalidEmbeddingInputError(
                "Image embedding input is not "
                f"a readable file: {value}"
            )

        try:
            content = value.read_bytes()
        except OSError as exc:
            raise InvalidEmbeddingInputError(
                f"Could not read image input "
                f"{value}: {exc}"
            ) from exc

        return (
            value,
            hashlib.sha256(content).hexdigest(),
        )

    @staticmethod
    def _cache_key(
        model_id: str,
        modality: EmbeddingModality,
        space: EmbeddingSpace,
        content_id: str,
    ) -> str:
        return (
            f"{model_id}:"
            f"{modality.value}:"
            f"{space.value}:"
            f"{content_id}"
        )

    @staticmethod
    def _result(
        content_id: str,
        values: tuple[float, ...],
        backend: EmbeddingBackend,
    ) -> EmbeddingVector:
        return EmbeddingVector(
            content_id=content_id,
            values=values,
            modality=backend.modality,
            space=backend.space,
            model_id=backend.model_id,
        )