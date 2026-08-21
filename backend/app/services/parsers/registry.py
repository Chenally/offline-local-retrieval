from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from app.errors import UnsupportedFileTypeError
from app.services.parsers.base import FileParser
from app.services.parsers.pdf_parser import PdfParser
from app.services.parsers.txt_parser import TxtParser


class ParserRegistry:
    """Selects the parser registered for a file extension."""

    def __init__(
        self,
        parsers: Iterable[FileParser] | None = None,
    ) -> None:
        parser_instances = parsers or (
            TxtParser(),
            PdfParser(),
        )

        self._parsers: dict[str, FileParser] = {}

        for parser in parser_instances:
            for extension in parser.supported_extensions:
                normalized_extension = extension.lower()

                if normalized_extension in self._parsers:
                    raise ValueError(
                        f"A parser is already registered for "
                        f"{normalized_extension}"
                    )

                self._parsers[normalized_extension] = parser

    def get_parser(self, path: Path) -> FileParser:
        parser = self._parsers.get(path.suffix.lower())

        if parser is None:
            raise UnsupportedFileTypeError(path)

        return parser