class EmbeddingError(Exception):
    """Base class for expected embedding failures."""


class InvalidEmbeddingInputError(EmbeddingError):
    """Raised when an embedding request contains invalid input."""


class InvalidEmbeddingVectorError(EmbeddingError):
    """Raised when a backend returns an invalid embedding vector."""


class EmbeddingBackendError(EmbeddingError):
    """Raised when a model backend fails during inference."""


class EmbeddingConfigurationError(EmbeddingError):
    """Raised when the embedding service is configured incorrectly."""