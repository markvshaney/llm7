# llm7/ollama/exceptions.py
class OllamaError(Exception):
    """Base exception class for Ollama-related errors."""

    pass


class ModelNotFoundError(OllamaError):
    """Raised when specified model is not available."""

    pass


class ConnectionError(OllamaError):
    """Raised when cannot connect to Ollama service."""

    pass


class ResponseError(OllamaError):
    """Raised when receiving invalid response from Ollama."""

    pass


class ConfigurationError(OllamaError):
    """Raised when there's an issue with model configuration."""

    pass


class TokenLimitError(OllamaError):
    """Raised when input exceeds model's token limit."""

    pass
