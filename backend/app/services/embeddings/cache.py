from __future__ import annotations

from collections import OrderedDict
from threading import RLock


class EmbeddingCache:
    """Thread-safe in-memory LRU cache for normalized vectors."""

    def __init__(self, max_entries: int = 1024) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be greater than zero")

        self._max_entries = max_entries
        self._values: OrderedDict[str, tuple[float, ...]] = OrderedDict()
        self._lock = RLock()

    def get(self, key: str) -> tuple[float, ...] | None:
        with self._lock:
            value = self._values.get(key)

            if value is None:
                return None

            self._values.move_to_end(key)
            return value

    def put(self, key: str, value: tuple[float, ...]) -> None:
        with self._lock:
            self._values[key] = value
            self._values.move_to_end(key)

            while len(self._values) > self._max_entries:
                self._values.popitem(last=False)