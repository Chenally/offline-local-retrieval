class StorageError(Exception):
    """Base class for expected vector, keyword and manifest failures."""


class IncompatibleVectorError(StorageError):
    """Raised when a vector does not match its collection's model contract."""
