from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest
from app.models import DocumentKind
from app.services.storage.manifest import IndexManifest, ManifestRecord

NOW = datetime(2026, 11, 26, tzinfo=timezone.utc)


def make_record(
    path: str, document_id: str = "doc", root: str = "/data"
) -> ManifestRecord:
    return ManifestRecord(
        path=Path(path),
        root=Path(root),
        document_id=document_id,
        kind=DocumentKind.TEXT,
        extension=".txt",
        file_name=Path(path).name,
        size_bytes=10,
        modified_at=datetime(2026, 1, 1, 12, 30, 0, 123456, tzinfo=timezone.utc),
        record_count=3,
        indexed_at=NOW,
    )


@pytest.fixture
def manifest() -> IndexManifest:
    value = IndexManifest(":memory:")
    yield value
    value.close()


def test_upsert_get_and_update(manifest: IndexManifest) -> None:
    record = make_record("/data/a.txt")
    manifest.upsert(record)

    assert manifest.get(Path("/data/a.txt")) == record

    updated = replace(record, document_id="new", record_count=5)
    manifest.upsert(updated)

    assert manifest.get(Path("/data/a.txt")) == updated
    assert manifest.get(Path("/data/missing.txt")) is None


def test_records_under_uses_path_boundaries(manifest: IndexManifest) -> None:
    for path in ["/data/a.txt", "/data/sub/b.txt", "/data2/c.txt", "/dat/d.txt"]:
        manifest.upsert(make_record(path))

    paths = [record.path for record in manifest.records_under(Path("/data"))]

    assert paths == [Path("/data/a.txt"), Path("/data/sub/b.txt")]


def test_records_for_documents_groups_paths(manifest: IndexManifest) -> None:
    manifest.upsert(make_record("/data/a.txt", "same"))
    manifest.upsert(make_record("/data/copy.txt", "same"))
    manifest.upsert(make_record("/data/b.txt", "other"))

    grouped = manifest.records_for_documents(["same", "unknown"])

    assert [record.path.name for record in grouped["same"]] == ["a.txt", "copy.txt"]
    assert grouped["unknown"] == []
    assert manifest.find_document("other") is not None
    assert manifest.find_document("unknown") is None


def test_delete_roots_and_summary(manifest: IndexManifest) -> None:
    manifest.add_root(Path("/data"), NOW)
    manifest.add_root(Path("/data"), NOW)
    manifest.upsert(make_record("/data/a.txt", "x"))
    manifest.upsert(replace(make_record("/data/p.png", "y"), kind=DocumentKind.IMAGE))

    summary = manifest.summary()

    assert summary.roots == (Path("/data"),)
    assert summary.file_count == 2
    assert summary.document_count == 2
    assert summary.text_file_count == 1
    assert summary.image_file_count == 1

    manifest.delete(Path("/data/a.txt"))
    manifest.remove_root(Path("/data"))

    assert manifest.get(Path("/data/a.txt")) is None
    assert manifest.roots() == []


def test_empty_summary(manifest: IndexManifest) -> None:
    summary = manifest.summary()

    assert summary.file_count == 0
    assert summary.image_file_count == 0


def test_persists_to_disk(tmp_path: Path) -> None:
    path = tmp_path / "index" / "manifest.sqlite3"
    first = IndexManifest(path)
    first.upsert(make_record("/data/a.txt"))
    first.close()

    second = IndexManifest(path)

    assert second.get(Path("/data/a.txt")) is not None
    second.close()
