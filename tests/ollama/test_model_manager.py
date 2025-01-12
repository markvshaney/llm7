import pytest
from unittest.mock import AsyncMock, Mock
import asyncio
from llm7.ollama.exceptions import ModelNotFoundError
from llm7.ollama.ollama_model_manager import model_manager
from llm7.ollama.interface import OllamaInterface

# Test configurations
VALID_MODEL_CONFIG = {
    "temperature": 0.8,
    "max_tokens": 2000,
    "system_prompt": "Test prompt"
}

INVALID_CONFIGS = [
    {"temperature": 2.0},  # Temperature too high
    {"temperature": -1.0},  # Temperature too low
    {"max_tokens": 0},     # Invalid token count
    {"max_tokens": -100},  # Negative token count
]

@pytest.fixture
async def ollama():
    """Create an OllamaInterface instance for testing."""
    interface = OllamaInterface()
    yield interface
    # Cleanup after each test
    interface.reset_conversation()

@pytest.fixture
def mock_interface():
    """Create a mock interface for testing."""
    interface = Mock(spec=OllamaInterface)
    interface.generate = AsyncMock()
    interface.reset_conversation = Mock()
    return interface

class TestModelManager:
    """Test suite for OllamaModelManager."""

    def test_available_models(self):
        """Test getting available models."""
        models = model_manager.available_models
        assert isinstance(models, list)
        assert len(models) > 0
        assert all(isinstance(model, str) for model in models)

    def test_model_config_access(self):
        """Test model configuration access."""
        model = model_manager.current_model
        config = model_manager.current_config

        assert isinstance(config, dict)
        required_keys = {"temperature", "max_tokens"}
        assert all(key in config for key in required_keys)
        assert isinstance(config["temperature"], (int, float))
        assert isinstance(config["max_tokens"], int)

    def test_add_model_config(self):
        """Test adding new model configuration."""
        test_model = "test-model"

        # Test adding valid config
        model_manager.add_model_config(test_model, VALID_MODEL_CONFIG)
        assert test_model in model_manager.available_models

        retrieved_config = model_manager.get_model_config(test_model)
        assert retrieved_config["temperature"] == VALID_MODEL_CONFIG["temperature"]
        assert retrieved_config["max_tokens"] == VALID_MODEL_CONFIG["max_tokens"]

        # Clean up
        if test_model in model_manager.available_models:
            model_manager._configs.pop(test_model, None)

    def test_update_model_config(self):
        """Test updating model configuration."""
        model = model_manager.current_model
        original_config = model_manager.current_config.copy()

        try:
            # Test valid update
            new_temp = 0.7
            model_manager.update_model_config(model, {"temperature": new_temp})
            assert model_manager.current_config["temperature"] == new_temp

        finally:
            # Reset to original config
            model_manager.update_model_config(model, original_config)

    @pytest.mark.asyncio
    async def test_streaming_response(self, mock_interface):
        """Test streaming response generation."""
        test_chunks = ["Response", " ", "Chunk", "!"]

        async def mock_stream():
            for chunk in test_chunks:
                yield chunk

        mock_interface.generate = AsyncMock(return_value=mock_stream())
        model_manager._interface = mock_interface

        chunks = []
        async for chunk in await mock_interface.generate("Test prompt", stream=True):
            chunks.append(chunk)

        assert chunks == test_chunks
        assert all(isinstance(chunk, str) for chunk in chunks)

    @pytest.mark.asyncio
    async def test_model_switching(self, mock_interface):
        """Test model switching functionality."""
        current_model = model_manager.current_model
        try:
            # Find an alternate model
            alternate_model = next(
                m for m in model_manager.available_models
                if m != current_model
            )

            # Switch model
            model_manager.switch_model(alternate_model)
            assert model_manager.current_model == alternate_model

            # Test invalid model switch
            with pytest.raises(ModelNotFoundError):
                model_manager.switch_model("non-existent-model")

        finally:
            # Reset to original model
            model_manager.switch_model(current_model)

    @pytest.mark.parametrize("invalid_config", INVALID_CONFIGS)
    def test_model_config_validation(self, invalid_config):
        """Test model configuration validation with various invalid configs."""
        with pytest.raises(ValueError):
            model_manager.update_model_config(
                model_manager.current_model,
                invalid_config
            )

        with pytest.raises(ValueError):
            model_manager.add_model_config(
                "test-invalid-model",
                invalid_config
            )

    def test_get_nonexistent_model_config(self):
        """Test getting configuration for non-existent model."""
        with pytest.raises(ModelNotFoundError):
            model_manager.get_model_config("non-existent-model")