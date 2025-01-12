# llm7/ollama/ollama_model_manager.py

from typing import Any, Dict, Optional

from config.settings import settings  # Direct import from root instead of relative import


class OllamaModelManager:
    """Manager class for handling different Ollama models and their configurations."""

    def __init__(self):
        # Load default configurations
        self._default_config = settings.get("llm.defaults", {})
        self._model_configs = settings.get("llm.models", {})
        self._options = settings.get("llm.options", {})

        # Load current model from config
        self._current_model = settings.get("llm.default_model", "mistral")

    def _get_model_config(self, model_name: str) -> Dict[str, Any]:
        """
        Get complete configuration for a model, combining defaults with model-specific settings.
        """
        # Start with default configuration
        config = self._default_config.copy()

        # Update with model-specific configuration if it exists
        if model_name in self._model_configs:
            config.update(self._model_configs[model_name])

        return config

    @property
    def available_models(self) -> list:
        """Get list of available models."""
        return list(self._model_configs.keys())

    @property
    def current_model(self) -> str:
        """Get current model name."""
        return self._current_model

    @property
    def current_config(self) -> Dict[str, Any]:
        """Get current model's configuration."""
        return self._get_model_config(self._current_model)

    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get configuration for specific model."""
        if model_name not in self._model_configs:
            raise ValueError(
                f"Model {model_name} not found. Available models: {self.available_models}"
            )
        return self._get_model_config(model_name)

    def set_current_model(self, model_name: str) -> None:
        """Set current model."""
        if model_name not in self._model_configs:
            raise ValueError(
                f"Model {model_name} not found. Available models: {self.available_models}"
            )
        self._current_model = model_name
        settings.set("llm.default_model", model_name)

    def add_model_config(self, model_name: str, config: Dict[str, Any]) -> None:
        """
        Add new model configuration. Only needs to specify parameters that differ from defaults.
        """
        self._model_configs[model_name] = config.copy()
        # Update settings
        settings.set(f"llm.models.{model_name}", config)

    def update_model_config(self, model_name: str, updates: Dict[str, Any]) -> None:
        """Update existing model configuration."""
        if model_name not in self._model_configs:
            raise ValueError(
                f"Model {model_name} not found. Available models: {self.available_models}"
            )
        current_config = self._model_configs.get(model_name, {})
        current_config.update(updates)
        self._model_configs[model_name] = current_config
        # Update settings
        settings.set(f"llm.models.{model_name}", current_config)

    @property
    def options(self) -> Dict[str, Any]:
        """Get global Ollama options."""
        return self._options.copy()

    def set_option(self, key: str, value: Any) -> None:
        """Set a global Ollama option."""
        self._options[key] = value
        settings.set(f"llm.options.{key}", value)


# Create singleton instance
model_manager = OllamaModelManager()

"""
Ollama Model Manager - Manages different models and their configurations.

Example Usage:
--------------
from llm7.ollama.interface import ollama_interface
from llm7.ollama.model_manager import model_manager

# Basic generation
response = await ollama_interface.generate("Explain how to make a cup of coffee.")

# Streaming chat
async for chunk in ollama_interface.chat("Tell me a short story about a robot.", stream=True):
    print(chunk, end='', flush=True)

# Multi-turn conversation
response1 = await ollama_interface.chat("What are the three laws of robotics?")
response2 = await ollama_interface.chat("Who created these laws?")

# View conversation history
for message in ollama_interface.conversation_history:
    print(f"{message['role'].title()}: {message['content']}\n")

# Switch between models
response1 = await ollama_interface.generate("Write a function to calculate fibonacci numbers.")
ollama_interface.switch_model('codellama')
response2 = await ollama_interface.generate("Write a function to calculate fibonacci numbers.")

# Use custom parameters
response = await ollama_interface.generate(
    "Brainstorm creative names for a tech startup.",
    temperature=0.9,  # More creative
    max_tokens=100    # Shorter response
)

# Error handling
try:
    ollama_interface.switch_model('non-existent-model')
except Exception as e:
    print("Caught error:", str(e))

# Add a new model configuration
model_manager.add_model_config('new-model', {
    'temperature': 0.6,
    'system_prompt': "You are a specialized AI assistant."
})

# Update existing model configuration
model_manager.update_model_config('mistral', {
    'temperature': 0.65
})

# Switch to a different model
model_manager.set_current_model('codellama')

# Get current model's configuration
config = model_manager.current_config

# List available models
models = model_manager.available_models

# Change global options
model_manager.set_option('stream_response', True)
"""