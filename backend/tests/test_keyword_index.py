from pathlib import Path

import pytest
from app.models import DocumentKind
from app.services.storage.keyword_index import (
    KeywordEntry,
    KeywordIndex,
    build_match_expression,
)


@pytest.fixture
def index() -> KeywordIndex:
    keyword_index = KeywordIndex(":memory:")
    yield keyword_index
    keyword_index.close()


def add_text(
    index: KeywordIndex,
    document_id: str,
    text: str,
    file_name: str = "notes.txt",
    extension: str = ".txt",
) -> None:
    index.add_document(
        document_id=document_id,
        kind=DocumentKind.TEXT,
        extension=extension,
        file_name=file_name,
        entries=[KeywordEntry(chunk_id=f"{document_id}:0", text=text, page_number=1)],
    )


def test_bm25_ranks_more_relevant_chunk_first(index: KeywordIndex) -> None:
    add_text(index, "a", "budget budget budget for the offline project")
    add_text(
        index, "b", "a short note that mentions the budget once among many other words"
    )
    add_text(index, "c", "nothing relevant here")

    hits = index.search("budget", limit=10)

    assert [hit.document_id for hit in hits] == ["a", "b"]
    assert hits[0].score > hits[1].score > 0
    assert hits[0].text.startswith("budget")
    assert hits[0].page_number == 1


def test_matches_file_names_for_images(index: KeywordIndex) -> None:
    index.add_document(
        document_id="img",
        kind=DocumentKind.IMAGE,
        extension=".png",
        file_name="red_sports-car.png",
        entries=[KeywordEntry(chunk_id="img:image", text="")],
    )

    [hit] = index.search("sports car", limit=5)

    assert hit.document_id == "img"
    assert hit.kind is DocumentKind.IMAGE


def test_chinese_phrase_search(index: KeywordIndex) -> None:
    add_text(index, "zh", "这是一份离线检索系统的设计文档")
    add_text(index, "other", "检测系统")

    hits = index.search("离线检索", limit=5)

    assert [hit.document_id for hit in hits] == ["zh"]


def test_extension_filter(index: KeywordIndex) -> None:
    add_text(index, "txt", "shared keyword", extension=".txt")
    add_text(index, "pdf", "shared keyword", file_name="a.pdf", extension=".pdf")

    hits = index.search("keyword", limit=5, extensions={".pdf"})

    assert [hit.document_id for hit in hits] == ["pdf"]


def test_re_adding_document_replaces_rows_and_delete_removes(
    index: KeywordIndex,
) -> None:
    add_text(index, "a", "old content")
    add_text(index, "a", "new content")

    assert index.search("old", limit=5) == []
    assert len(index.search("new", limit=5)) == 1

    index.delete_document("a")

    assert index.search("new", limit=5) == []
    assert index.count_documents() == 0


def test_operators_in_user_input_are_treated_as_text(index: KeywordIndex) -> None:
    add_text(index, "a", "near and or not")

    # Unquoted, these would be FTS5 syntax errors.
    assert index.search('NEAR( "unbalanced * AND', limit=5)
    assert index.search("***", limit=5) == []


def test_build_match_expression() -> None:
    assert build_match_expression("Hello, 世界!") == '"Hello" OR "世 界"'
    assert build_match_expression("hello hello") == '"hello"'
    assert build_match_expression("  ...  ") is None


def test_persists_to_disk(tmp_path: Path) -> None:
    path = tmp_path / "index" / "keyword.sqlite3"
    first = KeywordIndex(path)
    add_text(first, "a", "persistent words")
    first.close()

    second = KeywordIndex(path)

    assert len(second.search("persistent", limit=5)) == 1
    second.close()


def test_rejects_invalid_limit(index: KeywordIndex) -> None:
    with pytest.raises(ValueError):
        index.search("x", limit=0)
