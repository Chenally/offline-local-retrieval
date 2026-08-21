from pathlib import Path


class FileIngestionError(Exception):
    """Base class for expected ingestion failures."""


class InvalidDirectoryError(FileIngestionError):
    def __init__(self, path: Path, message: str) -> None:
        super().__init__(f"{message}: {path}")


class UnsupportedFileTypeError(FileIngestionError):
    def __init__(self, path: Path) -> None:
        extension = path.suffix.lower() or "<no extension>"
        super().__init__(f"Unsupported file type {extension}: {path}")


class CorruptedFileError(FileIngestionError):
    def __init__(self, path: Path, message: str) -> None:
        super().__init__(f"Could not parse {path}: {message}")


class FileReadError(FileIngestionError):
    def __init__(self, path: Path, message: str) -> None:
        super().__init__(f"Could not read {path}: {message}")


class FilePermissionError(FileIngestionError):
    def __init__(self, path: Path) -> None:
        super().__init__(f"Permission denied: {path}")