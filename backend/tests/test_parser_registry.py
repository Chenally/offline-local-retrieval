from pathlib import Path

import pytest

from app.errors import UnsupportedFileTypeError
from app.services.parsers.pdf_parser import PdfParser
from app.services.parsers.registry import ParserRegistry
from app.services.parsers.txt_parser import TxtParser


def test_returns_parser_for_supported_extension() -> None:
    registry = ParserRegistry()

    assert isinstance(
        registry.get_parser(Path("sample.TXT")),
        TxtParser,
    )

    assert isinstance(
        registry.get_parser(Path("sample.PDF")),
        PdfParser,
    )


def test_rejects_unsupported_extension() -> None:
    with pytest.raises(UnsupportedFileTypeError):
        ParserRegistry().get_parser(
            Path("sample.docx")
        )


def test_rejects_duplicate_parser_registration() -> None:
    with pytest.raises(
        ValueError,
        match="already registered",
    ):
        ParserRegistry(
            (
                TxtParser(),
                TxtParser(),
            )
        )
        