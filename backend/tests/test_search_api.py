import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("chromadb")

from app.config import AppSettings  # noqa: E402
from app.container import build_container  # noqa: E402
from app.main import create_app  # noqa: E402
from app.services.retrieval.ranking import RankingConfig  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from support import (  # noqa: E402
    FAKE_DIMENSION,
    make_backends,
    write_png,
)


@pytest.fixture
def client(tmp_path: Path):
    settings = AppSettings(
        runtime_dir=tmp_path / "runtime",
        models_dir=tmp_path / "models",
        ranking=RankingConfig(),
        chunk_max_units=20,
        chunk_overlap_units=5,
    )

    def factory():
        return build_container(
            settings,
            backends=make_backends(),
            embedding_dimension=FAKE_DIMENSION,
        )

    with TestClient(create_app(factory)) as test_client:
        yield test_client


def wait_for_job(client: TestClient, job_id: str) -> dict:
    for _ in range(500):
        job = client.get(f"/index/jobs/{job_id}").json()

        if job["state"] in {"succeeded", "failed"}:
            return job

        time.sleep(0.02)

    raise AssertionError("index job did not finish")


def index_and_wait(client: TestClient, directory: Path) -> dict:
    response = client.post("/index", json={"directory": str(directory)})
    assert response.status_code == 202
    return wait_for_job(client, response.json()["job_id"])


def test_liveness_still_works_without_models() -> None:
    def broken_factory():
        raise AssertionError("liveness must not build services")

    with TestClient(create_app(broken_factory)) as test_client:
        assert test_client.get("/health/live").json() == {"status": "ok"}


def test_readiness_returns_503_when_models_are_missing(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("OLR_RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("OLR_MODELS_DIR", str(tmp_path / "no-models"))

    with TestClient(create_app()) as test_client:
        response = test_client.get("/health/ready")

    assert response.status_code == 503


def test_index_status_and_search_flow(client: TestClient, tmp_path: Path) -> None:
    library = tmp_path / "library"
    library.mkdir()
    (library / "budget.txt").write_text(
        "The accessibility budget is approved.",
        encoding="utf-8",
    )
    write_png(library / "red_car.png")

    job = index_and_wait(client, library)

    assert job["state"] == "succeeded"
    assert job["report"]["indexed"] == 2
    assert job["report"]["errors"] == []

    status = client.get("/index/status").json()
    assert status["file_count"] == 2
    assert status["image_file_count"] == 1
    assert status["roots"] == [str(library.resolve())]
    assert status["active_job"] is None

    response = client.post(
        "/search", json={"query": "accessibility budget", "top_k": 5}
    )
    assert response.status_code == 200
    body = response.json()
    top = body["results"][0]
    assert top["file_name"] == "budget.txt"
    assert top["paths"] == [str((library / "budget.txt").resolve())]
    assert set(top["channel_scores"]) == {"keyword", "text_semantic"}
    assert top["channel_scores"]["keyword"]["weight"] == 0.3

    image = client.post(
        "/search",
        json={"query": "red car", "modes": ["image_semantic"], "extensions": ["png"]},
    ).json()
    assert image["results"][0]["kind"] == "image"

    ready = client.get("/health/ready").json()
    assert ready["status"] == "ready"


def test_remove_folder(client: TestClient, tmp_path: Path) -> None:
    library = tmp_path / "library"
    library.mkdir()
    (library / "a.txt").write_text("removable", encoding="utf-8")
    index_and_wait(client, library)

    response = client.post("/index/remove", json={"directory": str(library)})
    job = wait_for_job(client, response.json()["job_id"])

    assert job["state"] == "succeeded"
    assert job["removed_count"] == 1
    assert client.get("/index/status").json()["file_count"] == 0


def test_request_validation(client: TestClient, tmp_path: Path) -> None:
    assert (
        client.post("/index", json={"directory": str(tmp_path / "nope")}).status_code
        == 400
    )
    assert client.post("/search", json={"query": ""}).status_code == 422
    assert client.post("/search", json={"query": "x", "top_k": 0}).status_code == 422
    assert (
        client.post("/search", json={"query": "x", "modes": ["bogus"]}).status_code
        == 422
    )
    assert client.post("/search", json={"query": "   "}).status_code == 400
    assert client.get("/index/jobs/unknown").status_code == 404
