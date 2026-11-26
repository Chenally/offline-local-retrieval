from app.services.embeddings.service import EmbeddingService
from app.services.embeddings.synchronized import SynchronizedEmbeddingService
from app.services.embeddings.types import EmbeddingSpace

from support import make_backends, make_embedding_service, write_png


def test_wrapper_delegates_and_clears_cache() -> None:
    backends = make_backends()

    service = SynchronizedEmbeddingService(EmbeddingService(backends))
    text_backend = backends[0]

    service.embed_texts(["hello world"], space=EmbeddingSpace.TEXT_SEMANTIC)
    service.embed_texts(["hello world"], space=EmbeddingSpace.TEXT_SEMANTIC)
    assert text_backend.calls == [1]

    service.clear_cache()
    service.embed_texts(["hello world"], space=EmbeddingSpace.TEXT_SEMANTIC)
    assert text_backend.calls == [1, 1]


def test_wrapper_embeds_images(tmp_path) -> None:
    image = write_png(tmp_path / "red_car.png")
    service = SynchronizedEmbeddingService(make_embedding_service())

    [vector] = service.embed_images([image])

    assert vector.space is EmbeddingSpace.MULTIMODAL
