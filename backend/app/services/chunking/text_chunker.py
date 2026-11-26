from __future__ import annotations

from dataclasses import dataclass

from app.models import ParsedDocument
from app.services.text_units import iter_unit_spans

# The BERT embedder accepts 128 WordPiece tokens including [CLS] and [SEP].
# English words average roughly 1.3 WordPiece tokens, so 80 units leave a
# safety margin before the backend has to truncate a chunk.
DEFAULT_MAX_UNITS = 80
DEFAULT_OVERLAP_UNITS = 20


@dataclass(frozen=True, slots=True)
class TextChunk:
    """A retrievable passage of a parsed document."""

    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    page_number: int | None
    char_start: int
    char_end: int


class TextChunker:
    """Splits document text into overlapping, page-aware windows.

    Chunks never cross a PDF page boundary, so every search hit can be
    traced back to one page. Character offsets are relative to the page
    text for paged documents and to the full text otherwise.
    """

    def __init__(
        self,
        max_units: int = DEFAULT_MAX_UNITS,
        overlap_units: int = DEFAULT_OVERLAP_UNITS,
    ) -> None:
        if max_units <= 0:
            raise ValueError("max_units must be greater than zero")

        if not 0 <= overlap_units < max_units:
            raise ValueError(
                "overlap_units must be non-negative and smaller than max_units"
            )

        self._max_units = max_units
        self._step = max_units - overlap_units

    def chunk(self, document: ParsedDocument) -> list[TextChunk]:
        sections: list[tuple[int | None, str]]

        if document.page_texts:
            sections = [
                (page_index + 1, page_text)
                for page_index, page_text in enumerate(document.page_texts)
            ]
        else:
            sections = [(None, document.text)]

        chunks: list[TextChunk] = []

        for page_number, text in sections:
            spans = list(iter_unit_spans(text))
            start = 0

            while start < len(spans):
                window = spans[start : start + self._max_units]
                char_start = window[0][0]
                char_end = window[-1][1]
                chunk_index = len(chunks)

                chunks.append(
                    TextChunk(
                        chunk_id=f"{document.document_id}:{chunk_index}",
                        document_id=document.document_id,
                        chunk_index=chunk_index,
                        text=text[char_start:char_end],
                        page_number=page_number,
                        char_start=char_start,
                        char_end=char_end,
                    )
                )

                if start + self._max_units >= len(spans):
                    break

                start += self._step

        return chunks
