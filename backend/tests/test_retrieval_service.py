from pathlib import Path

import pytest
from app.services.embeddings.errors import (
    EmbeddingBackendError,
)
from app.services.embeddings.types import (
    EmbeddingModality,
    EmbeddingSpace,
    EmbeddingVector,
)
from app.services.retrieval.errors import (
    InvalidRetrievalInputError,
    SearchError,
)
from app.services.retrieval.retrieval_service import (
    RetrievalService,
)
from app.services.retrieval.types import (
    RetrievalMatch,
)


class FakeEmbeddingService:
    def __init__(
        self,
        *,
        fail: bool = False,
        empty_batch: bool = False,
        wrong_contract: bool = False,
    ) -> None:
        self.calls: list[
            tuple[list[str], EmbeddingSpace]
        ] = []
        self.fail = fail
        self.empty_batch = empty_batch
        self.wrong_contract = wrong_contract

    def embed_texts(
        self,
        texts: list[str],
        space: EmbeddingSpace = (
            EmbeddingSpace.TEXT_SEMANTIC
        ),
    ) -> list[EmbeddingVector]:
        self.calls.append(
            (list(texts), space)
        )

        if self.fail:
            raise EmbeddingBackendError(
                "embedding failed"
            )

        if self.empty_batch:
            return []

        vector_space = space

        if (
            self.wrong_contract
            and space is EmbeddingSpace.TEXT_SEMANTIC
        ):
            vector_space = EmbeddingSpace.MULTIMODAL

        return [
            EmbeddingVector(
                content_id="query-content",
                values=(1.0, 0.0),
                modality=EmbeddingModality.TEXT,
                space=vector_space,
                model_id=(
                    "bert"
                    if space
                    is EmbeddingSpace.TEXT_SEMANTIC
                    else "mobileclip"
                ),
            )
        ]


class FakeVectorStore:
    def __init__(
        self,
        results: dict[
            tuple[
                EmbeddingSpace,
                EmbeddingModality,
            ],
            list[RetrievalMatch],
        ]
        | None = None,
    ) -> None:
        self.results = results or {}
        self.query_calls: list[
            tuple[
                EmbeddingVector,
                int,
                EmbeddingModality | None,
            ]
        ] = []

    def query(
        self,
        query: EmbeddingVector,
        *,
        top_k: int,
        modality: EmbeddingModality | None = None,
    ) -> list[RetrievalMatch]:
        self.query_calls.append(
            (
                query,
                top_k,
                modality,
            )
        )

        if modality is None:
            return []

        return list(
            self.results.get(
                (
                    query.space,
                    modality,
                ),
                [],
            )
        )


def _match(
    record_id: str,
    *,
    modality: EmbeddingModality,
    space: EmbeddingSpace,
) -> RetrievalMatch:
    return RetrievalMatch(
        record_id=record_id,
        content="local content",
        source_path=Path(
            f"/local/{record_id}"
        ),
        source_sha256=f"sha-{record_id}",
        content_id=f"content-{record_id}",
        modality=modality,
        space=space,
        model_id=(
            "bert"
            if space is EmbeddingSpace.TEXT_SEMANTIC
            else "mobileclip"
        ),
        distance=0.1,
        similarity=0.9,
    )


def test_routes_query_across_three_channels() -> None:
    semantic_match = _match(
        "text:semantic",
        modality=EmbeddingModality.TEXT,
        space=EmbeddingSpace.TEXT_SEMANTIC,
    )
    multimodal_text_match = _match(
        "text:multimodal",
        modality=EmbeddingModality.TEXT,
        space=EmbeddingSpace.MULTIMODAL,
    )
    image_match = _match(
        "image:multimodal",
        modality=EmbeddingModality.IMAGE,
        space=EmbeddingSpace.MULTIMODAL,
    )

    embeddings = FakeEmbeddingService()
    store = FakeVectorStore(
        {
            (
                EmbeddingSpace.TEXT_SEMANTIC,
                EmbeddingModality.TEXT,
            ): [semantic_match],
            (
                EmbeddingSpace.MULTIMODAL,
                EmbeddingModality.TEXT,
            ): [multimodal_text_match],
            (
                EmbeddingSpace.MULTIMODAL,
                EmbeddingModality.IMAGE,
            ): [image_match],
        }
    )
    service = RetrievalService(
        embeddings,
        store,
    )

    candidates = service.retrieve(
        "local retrieval",
        top_k=5,
    )

    assert embeddings.calls == [
        (
            ["local retrieval"],
            EmbeddingSpace.TEXT_SEMANTIC,
        ),
        (
            ["local retrieval"],
            EmbeddingSpace.MULTIMODAL,
        ),
    ]
    assert candidates.text_semantic == (
        semantic_match,
    )
    assert candidates.multimodal_text == (
        multimodal_text_match,
    )
    assert candidates.multimodal_images == (
        image_match,
    )
    assert [
        (
            vector.space,
            top_k,
            modality,
        )
        for (
            vector,
            top_k,
            modality,
        ) in store.query_calls
    ] == [
        (
            EmbeddingSpace.TEXT_SEMANTIC,
            5,
            EmbeddingModality.TEXT,
        ),
        (
            EmbeddingSpace.MULTIMODAL,
            5,
            EmbeddingModality.TEXT,
        ),
        (
            EmbeddingSpace.MULTIMODAL,
            5,
            EmbeddingModality.IMAGE,
        ),
    ]


@pytest.mark.parametrize(
    "query",
    [
        "",
        "   ",
    ],
)
def test_rejects_blank_query(
    query: str,
) -> None:
    embeddings = FakeEmbeddingService()
    store = FakeVectorStore()
    service = RetrievalService(
        embeddings,
        store,
    )

    with pytest.raises(
        InvalidRetrievalInputError
    ):
        service.retrieve(query)

    assert embeddings.calls == []
    assert store.query_calls == []


@pytest.mark.parametrize(
    "top_k",
    [
        0,
        -1,
    ],
)
def test_rejects_invalid_top_k(
    top_k: int,
) -> None:
    embeddings = FakeEmbeddingService()
    store = FakeVectorStore()
    service = RetrievalService(
        embeddings,
        store,
    )

    with pytest.raises(
        InvalidRetrievalInputError
    ):
        service.retrieve(
            "local retrieval",
            top_k=top_k,
        )

    assert embeddings.calls == []
    assert store.query_calls == []


def test_rejects_missing_query_embedding() -> None:
    store = FakeVectorStore()
    service = RetrievalService(
        FakeEmbeddingService(
            empty_batch=True
        ),
        store,
    )

    with pytest.raises(SearchError):
        service.retrieve("local retrieval")

    assert store.query_calls == []


def test_normalizes_embedding_failure() -> None:
    store = FakeVectorStore()
    service = RetrievalService(
        FakeEmbeddingService(fail=True),
        store,
    )

    with pytest.raises(SearchError):
        service.retrieve("local retrieval")

    assert store.query_calls == []


def test_rejects_wrong_embedding_contract() -> None:
    store = FakeVectorStore()
    service = RetrievalService(
        FakeEmbeddingService(
            wrong_contract=True
        ),
        store,
    )

    with pytest.raises(SearchError):
        service.retrieve("local retrieval")

    assert store.query_calls == []
