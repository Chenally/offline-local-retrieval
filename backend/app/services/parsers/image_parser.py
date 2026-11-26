from __future__ import annotations

from pathlib import Path

from PIL import Image, UnidentifiedImageError

from app.errors import (
    CorruptedFileError,
    FilePermissionError,
)
from app.models import DocumentKind, FileMetadata, ParsedDocument


class ImageParser:
    """Validates a local JPG or PNG image without extracting text.

    Images are embedded later by the MobileCLIP image encoder. The parser
    only guarantees that the file is a decodable image, so a damaged file
    is reported during ingestion instead of failing inside inference.
    """

    supported_extensions = frozenset({".jpg", ".jpeg", ".png"})

    def parse(
        self,
        path: Path,
        metadata: FileMetadata,
    ) -> ParsedDocument:
        try:
            with Image.open(path) as image:
                image.verify()
        except PermissionError as exc:
            raise FilePermissionError(path) from exc
        except (
            UnidentifiedImageError,
            SyntaxError,
            ValueError,
        ) as exc:
            raise CorruptedFileError(
                path,
                f"Invalid image data: {exc}",
            ) from exc
        except OSError as exc:
            # Pillow raises OSError for truncated image data.
            raise CorruptedFileError(path, str(exc)) from exc

        return ParsedDocument(
            document_id=metadata.sha256,
            text="",
            metadata=metadata,
            kind=DocumentKind.IMAGE,
        )
