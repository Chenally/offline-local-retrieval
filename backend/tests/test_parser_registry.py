from pathlib import Path

import pytest

from app.errors import UnsupportedFileTypeError
from app.services.parsers.docx_parser import DocxParser
from app.services.parsers.image_parser import ImageParser
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


@pytest.mark.parametrize(
    ("file_name", "parser_type"),
    [
        ("report.DOCX", DocxParser),
        ("photo.jpg", ImageParser),
        ("photo.JPEG", ImageParser),
        ("screen.png", ImageParser),
    ],
)
def test_returns_week4_parsers(file_name: str, parser_type: type) -> None:
    registry = ParserRegistry()

    assert isinstance(registry.get_parser(Path(file_name)), parser_type)
    assert registry.supports(Path(file_name))


def test_supports_is_false_for_unknown_extension() -> None:
    assert not ParserRegistry().supports(Path("sheet.xlsx"))


def test_rejects_unsupported_extension() -> None:
    with pytest.raises(UnsupportedFileTypeError):
        ParserRegistry().get_parser(
            Path("sample.xlsx")
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
        