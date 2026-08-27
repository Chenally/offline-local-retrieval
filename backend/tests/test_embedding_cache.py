import pytest
from app.services.embeddings.cache import EmbeddingCache


def test_cache_evicts_least_recently_used_value() -> None:
    cache = EmbeddingCache(max_entries=2)

    assert cache.get("missing") is None

    cache.put("a", (1.0,))
    cache.put("b", (2.0,))
    assert cache.get("a") == (1.0,)

    cache.put("c", (3.0,))

    assert cache.get("b") is None
    assert cache.get("a") == (1.0,)
    assert cache.get("c") == (3.0,)


def test_cache_rejects_invalid_capacity() -> None:
    with pytest.raises(ValueError, match="max_entries"):
        EmbeddingCache(max_entries=0)