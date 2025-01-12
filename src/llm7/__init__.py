# src/llm7/__init__.py
from .ollama.interface import OllamaInterface
from .ollama.exceptions import ModelNotFoundError

# Import model_manager last to avoid circular import
from .ollama.ollama_model_manager import model_manager

__all__ = ['OllamaInterface', 'model_manager', 'ModelNotFoundError']