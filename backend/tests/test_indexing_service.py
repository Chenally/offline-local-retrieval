import os
from datetime import datetime, timezone
from pathlib import Path

import pytest
from app.models import DocumentKind
from app.services.chunking.text_chunker import TextChunker
from app.services.embeddings.types import EmbeddingSpace
from app.services.indexing.indexing_service import IndexingService, IndexProgress
from app.services.metadata_extractor import MetadataExtractor
from app.services.storage.keyword_index import KeywordIndex
from app.services.storage.manifest import IndexManifest

from support import (
    InMemoryVectorStore,
    make_embedding_service,
    write_docx,
    write_png,
)


class Harness:
    def __init__(self) -> None:
        self.vectors = InMemoryVectorStore()
        self.keywords = KeywordIndex(":memory:")
        self.manifest = IndexManifest(":memory:")
        self.service = IndexingService(
            embedder=make_embedding_service(),
            vector_store=self.vectors,
            keyword_index=self.keywords,
            manifest=self.manifest,
            chunker=TextChunker(max_units=8, overlap_units=2),
            clock=lambda: datetime(2026, 11, 26, tzinfo=timezone.utc),
        )

    def close(self) -> None:
        self.keywords.close()
        self.manifest.close()


@pytest.fixture
def harness() -> Harness:
    value = Harness()
    yield value
    value.close()


def bump_mtime(path: Path) -> None:
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))


def test_indexes_text_docx_and_images(tmp_path: Path, harness: Harness) -> None:
    (tmp_path / "notes.txt").write_text(
        "offline search keeps private files on the device " * 3,
        encoding="utf-8",
    )
    write_docx(tmp_path / "plan.docx", ["Week four builds hybrid ranking"])
    write_png(tmp_path / "red_car.png")
    (tmp_path / "sheet.xlsx").write_bytes(b"ignored")

    progress: list[IndexProgress] = []
    report = harness.service.index_directory(tmp_path, progress=progress.append)

    assert report.scanned == 4
    assert report.skipped_unsupported == 1
    assert sorted(path.name for path in report.indexed) == [
        "notes.txt",
        "plan.docx",
        "red_car.png",
    ]
    assert report.errors == []
    assert harness.vectors.count(EmbeddingSpace.TEXT_SEMANTIC) > 2
    assert harness.vectors.count(EmbeddingSpace.MULTIMODAL) == 1
    assert report.vector_count == (
        harness.vectors.count(EmbeddingSpace.TEXT_SEMANTIC)
        + harness.vectors.count(EmbeddingSpace.MULTIMODAL)
    )
    assert progress[-1] == IndexProgress(3, 3, None)
    assert harness.manifest.summary().image_file_count == 1
    assert harness.keywords.search("hybrid", limit=5)[0].kind is DocumentKind.TEXT


def test_second_run_skips_unchanged_files(tmp_path: Path, harness: Harness) -> None:
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    harness.service.index_directory(tmp_path)

    report = harness.service.index_directory(tmp_path)

    assert report.indexed == []
    assert [path.name for path in report.unchanged] == ["a.txt"]


def test_touched_file_with_same_bytes_is_not_reembedded(
    tmp_path: Path,
    harness: Harness,
) -> None:
    path = tmp_path / "a.txt"
    path.write_text("alpha", encoding="utf-8")
    harness.service.index_directory(tmp_path)
    bump_mtime(path)

    report = harness.service.index_directory(tmp_path)

    assert [item.name for item in report.unchanged] == ["a.txt"]
    record = harness.manifest.get(path.resolve())
    assert record is not None
    assert record.modified_at == datetime.fromtimestamp(
        path.stat().st_mtime, tz=timezone.utc
    )


def test_modified_file_replaces_old_vectors(tmp_path: Path, harness: Harness) -> None:
    path = tmp_path / "a.txt"
    path.write_text("old words", encoding="utf-8")
    harness.service.index_directory(tmp_path)
    old_ids = harness.vectors.document_ids(EmbeddingSpace.TEXT_SEMANTIC)

    path.write_text("completely new words here", encoding="utf-8")
    bump_mtime(path)
    report = harness.service.index_directory(tmp_path)

    new_ids = harness.vectors.document_ids(EmbeddingSpace.TEXT_SEMANTIC)
    assert [item.name for item in report.indexed] == ["a.txt"]
    assert old_ids.isdisjoint(new_ids)
    assert harness.keywords.search("old", limit=5) == []


def test_deleted_file_is_purged(tmp_path: Path, harness: Harness) -> None:
    path = tmp_path / "a.txt"
    path.write_text("temporary", encoding="utf-8")
    harness.service.index_directory(tmp_path)

    path.unlink()
    report = harness.service.index_directory(tmp_path)

    assert [item.name for item in report.removed] == ["a.txt"]
    assert harness.vectors.count(EmbeddingSpace.TEXT_SEMANTIC) == 0
    assert harness.keywords.search("temporary", limit=5) == []


def test_duplicates_share_vectors_until_last_copy_is_removed(
    tmp_path: Path,
    harness: Harness,
) -> None:
    (tmp_path / "a.txt").write_text("shared content", encoding="utf-8")
    (tmp_path / "b.txt").write_text("shared content", encoding="utf-8")

    report = harness.service.index_directory(tmp_path)

    assert len(report.indexed) == 1
    assert len(report.duplicates) == 1
    vector_count = harness.vectors.count(EmbeddingSpace.TEXT_SEMANTIC)

    (tmp_path / "a.txt").unlink()
    harness.service.index_directory(tmp_path)
    assert harness.vectors.count(EmbeddingSpace.TEXT_SEMANTIC) == vector_count

    (tmp_path / "b.txt").unlink()
    harness.service.index_directory(tmp_path)
    assert harness.vectors.count(EmbeddingSpace.TEXT_SEMANTIC) == 0


def test_file_errors_do_not_stop_the_batch(tmp_path: Path, harness: Harness) -> None:
    (tmp_path / "good.txt").write_text("fine", encoding="utf-8")
    (tmp_path / "broken.pdf").write_bytes(b"not a pdf")
    (tmp_path / "broken.png").write_bytes(b"not a png")
    (tmp_path / "latin1.txt").write_bytes("caf\xe9".encode("latin-1"))

    report = harness.service.index_directory(tmp_path)

    assert [path.name for path in report.indexed] == ["good.txt"]
    assert sorted(error.path.name for error in report.errors) == [
        "broken.pdf",
        "broken.png",
        "latin1.txt",
    ]
    assert {error.error_type for error in report.errors} == {"CorruptedFileError"}


def test_failed_write_leaves_file_unindexed_for_retry(
    tmp_path: Path,
    harness: Harness,
) -> None:
    (tmp_path / "a.txt").write_text("retry me", encoding="utf-8")
    harness.vectors.fail_on_upsert = True

    report = harness.service.index_directory(tmp_path)

    assert report.errors[0].error_type == "StorageError"
    assert harness.manifest.get((tmp_path / "a.txt").resolve()) is None
    assert harness.keywords.search("retry", limit=5) == []

    harness.vectors.fail_on_upsert = False
    retry = harness.service.index_directory(tmp_path)

    assert [path.name for path in retry.indexed] == ["a.txt"]


def test_text_without_words_is_findable_by_name(
    tmp_path: Path, harness: Harness
) -> None:
    (tmp_path / "quarterly_scan.txt").write_text("   ", encoding="utf-8")

    report = harness.service.index_directory(tmp_path)

    assert report.vector_count == 0
    assert harness.keywords.search("quarterly", limit=5)[0].chunk_id.endswith(":name")


def test_missing_directory_is_reported_and_index_kept(
    tmp_path: Path,
    harness: Harness,
) -> None:
    (tmp_path / "a.txt").write_text("keep", encoding="utf-8")
    harness.service.index_directory(tmp_path)

    report = harness.service.index_directory(tmp_path / "missing")

    assert report.errors[0].error_type == "InvalidDirectoryError"
    assert harness.manifest.summary().file_count == 1


def test_remove_directory_respects_other_roots(
    tmp_path: Path, harness: Harness
) -> None:
    outer = tmp_path / "outer"
    inner = outer / "inner"
    inner.mkdir(parents=True)
    (outer / "top.txt").write_text("top level", encoding="utf-8")
    (inner / "nested.txt").write_text("nested level", encoding="utf-8")
    harness.service.index_directory(outer)
    harness.service.index_directory(inner)

    removed = harness.service.remove_directory(inner)

    assert removed == []
    assert harness.manifest.summary().file_count == 2

    removed = harness.service.remove_directory(outer)

    assert sorted(path.name for path in removed) == ["nested.txt", "top.txt"]
    assert harness.vectors.count(EmbeddingSpace.TEXT_SEMANTIC) == 0
    assert harness.manifest.roots() == []


def test_permission_error_is_recorded(
    tmp_path: Path,
    harness: Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blocked = tmp_path / "blocked.txt"
    blocked.write_text("secret", encoding="utf-8")
    (tmp_path / "ok.txt").write_text("fine", encoding="utf-8")
    original_hash = MetadataExtractor._calculate_sha256

    def guarded_hash(extractor: MetadataExtractor, path: Path) -> str:
        if path == blocked.resolve():
            raise PermissionError
        return original_hash(extractor, path)

    monkeypatch.setattr(MetadataExtractor, "_calculate_sha256", guarded_hash)
    report = harness.service.index_directory(tmp_path)

    assert [path.name for path in report.indexed] == ["ok.txt"]
    assert report.errors[0].error_type == "FilePermissionError"


def test_unexpected_parser_failure_is_recorded(
    tmp_path: Path,
    harness: Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "a.txt").write_text("boom", encoding="utf-8")
    (tmp_path / "b.txt").write_text("fine", encoding="utf-8")
    from app.services.parsers.txt_parser import TxtParser

    original_parse = TxtParser.parse

    def flaky_parse(parser, path, metadata):
        if path.name == "a.txt":
            raise RuntimeError("parser bug")
        return original_parse(parser, path, metadata)

    monkeypatch.setattr(TxtParser, "parse", flaky_parse)
    report = harness.service.index_directory(tmp_path)

    assert [path.name for path in report.indexed] == ["b.txt"]
    assert report.errors[0].error_type == "RuntimeError"


def test_file_vanishing_during_indexing_is_a_read_error(
    tmp_path: Path,
    harness: Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "gone.txt").write_text("soon deleted", encoding="utf-8")
    original_get = harness.manifest.get

    def delete_then_get(path: Path):
        path.unlink(missing_ok=True)
        return original_get(path)

    monkeypatch.setattr(harness.manifest, "get", delete_then_get)
    report = harness.service.index_directory(tmp_path)

    assert report.errors[0].error_type == "FileReadError"
