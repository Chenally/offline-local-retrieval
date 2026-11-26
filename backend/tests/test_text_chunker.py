from datetime import datetime, timezone
from pathlib import Path

import pytest
from app.models import FileMetadata, ParsedDocument
from app.services.chunking.text_chunker import TextChunker


def make_document(
    text: str = "",
    page_texts: tuple[str, ...] = (),
) -> ParsedDocument:
    metadata = FileMetadata(
        path=Path("/tmp/doc.txt"),
        file_name="doc.txt",
        extension=".txt",
        mime_type="text/plain",
        size_bytes=len(text),
        modified_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        sha256="abc",
    )
    return ParsedDocument(
        document_id="abc",
        text=text,
        metadata=metadata,
        page_texts=page_texts,
    )


def test_short_text_produces_one_chunk_with_offsets() -> None:
    document = make_document("  alpha beta gamma  ")

    [chunk] = TextChunker(max_units=10, overlap_units=2).chunk(document)

    assert chunk.chunk_id == "abc:0"
    assert chunk.text == "alpha beta gamma"
    assert document.text[chunk.char_start : chunk.char_end] == chunk.text
    assert chunk.page_number is None


def test_long_text_uses_overlapping_windows() -> None:
    words = [f"w{index}" for index in range(10)]
    document = make_document(" ".join(words))

    chunks = TextChunker(max_units=4, overlap_units=1).chunk(document)

    assert [chunk.text.split() for chunk in chunks] == [
        ["w0", "w1", "w2", "w3"],
        ["w3", "w4", "w5", "w6"],
        ["w6", "w7", "w8", "w9"],
    ]
    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2]


def test_last_window_is_not_duplicated_when_text_fits_exactly() -> None:
    document = make_document("a b c d")

    chunks = TextChunker(max_units=4, overlap_units=2).chunk(document)

    assert len(chunks) == 1


def test_pages_are_chunked_separately() -> None:
    document = make_document(
        text="page one\n\npage two",
        page_texts=("page one", "", "page two"),
    )

    chunks = TextChunker(max_units=10, overlap_units=0).chunk(document)

    assert [(chunk.page_number, chunk.text) for chunk in chunks] == [
        (1, "page one"),
        (3, "page two"),
    ]
    assert [chunk.chunk_id for chunk in chunks] == ["abc:0", "abc:1"]


def test_cjk_text_is_split_per_character() -> None:
    document = make_document("离线检索系统")

    chunks = TextChunker(max_units=3, overlap_units=0).chunk(document)

    assert [chunk.text for chunk in chunks] == ["离线检", "索系统"]


def test_empty_text_produces_no_chunks() -> None:
    assert TextChunker().chunk(make_document("   \n ")) == []


@pytest.mark.parametrize(
    ("max_units", "overlap_units"),
    [(0, 0), (4, 4), (4, -1)],
)
def test_rejects_invalid_configuration(max_units: int, overlap_units: int) -> None:
    with pytest.raises(ValueError):
        TextChunker(max_units=max_units, overlap_units=overlap_units)
