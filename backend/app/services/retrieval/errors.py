class RetrievalError(Exception):
    """Base class for expected indexing and retrieval failures."""


class InvalidRetrievalInputError(RetrievalError):
    """Raised when indexing or query input is invalid."""


class RetrievalConfigurationError(RetrievalError):
    """Raised when persistent retrieval is configured incorrectly."""


class VectorStoreError(RetrievalError):
    """Raised when persistent vector storage fails."""

class IndexingError(RetrievalError):
    """Raised when local content cannot be converted into index records."""

class SearchError(RetrievalError):
    """Raised when retrieval candidates cannot be collected."""