from __future__ import annotations

from pathlib import Path

from app.errors import FilePermissionError, InvalidDirectoryError


class FileScanner:
    """Recursively discovers regular, non-hidden local files."""

    def scan(self, directory: str | Path) -> list[Path]:
        root = Path(directory).expanduser().resolve()

        if not root.exists():
            raise InvalidDirectoryError(root, "Directory does not exist")

        if not root.is_dir():
            raise InvalidDirectoryError(root, "Path is not a directory")

        try:
            files = [
                path.resolve()
                for path in root.rglob("*")
                if path.is_file()
                and not path.is_symlink()
                and not self._is_hidden(path, root)
            ]
        except PermissionError as exc:
            raise FilePermissionError(root) from exc

        return sorted(files, key=lambda path: str(path).casefold())

    @staticmethod
    def _is_hidden(path: Path, root: Path) -> bool:
        relative_path = path.relative_to(root)
        return any(part.startswith(".") for part in relative_path.parts)
        