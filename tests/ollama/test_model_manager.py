# tests/ollama/test_interface.py
import asyncio

import pytest

from llm7.ollama.exceptions import ModelNotFoundError
from llm7.ollama.interface import OllamaInterface
from llm7.ollama.model_manager import model_manager


@pytest.fixture
def ollama():
    """Create a fresh OllamaInterface instance for each test."""
    interface = OllamaInterface()
    yield interface
    # Cleanup after each test
    interface.reset_conversation()


@pytest.mark.asyncio
async def test_basic_generation(ollama):
    """Test basic text generation."""
    response = await ollama.generate("Test prompt")
    assert isinstance(response, str)
    assert len(response) > 0


@pytest.mark.asyncio
async def test_streaming_response(ollama):
    """Test streaming response generation."""
    chunks = []
    async for chunk in ollama.generate("Test prompt", stream=True):
        chunks.append(chunk)

    assert len(chunks) > 0
    assert all(isinstance(chunk, str) for chunk in chunks)


@pytest.mark.asyncio
async def test_chat_conversation(ollama):
    """Test multi-turn conversation."""
    response1 = await ollama.chat("Hello")
    assert isinstance(response1, str)

    response2 = await ollama.chat("How are you?")
    assert isinstance(response2, str)

    history = ollama.conversation_history
    assert (
        len(history) == 4
    )  # 2 user messages + 2 assistant responses


def test_model_switching(ollama):
    """Test model switching functionality."""
    initial_model = ollama.current_model

    # Switch to a different model
    new_model = [
        m
        for m in model_manager.available_models
        if m != initial_model
    ][0]
    ollama.switch_model(new_model)
    assert ollama.current_model == new_model

    # Test switching to invalid model
    with pytest.raises(ModelNotFoundError):
        ollama.switch_model("non-existent-model")


@pytest.mark.asyncio
async def test_context_injection(ollama):
    """Test context injection."""
    context = "Respond in a poetic style"
    ollama.inject_context(context)

    history = ollama.conversation_history
    assert len(history) == 1
    assert history[0]["role"] == "system"
    assert history[0]["content"] == context


@pytest.mark.asyncio
async def test_parameter_override(ollama):
    """Test parameter override functionality."""
    response = await ollama.generate(
        "Test prompt", temperature=0.1, max_tokens=50
    )
    assert isinstance(response, str)
    assert len(response) > 0


def test_conversation_reset(ollama):
    """Test conversation reset functionality."""
    asyncio.run(ollama.chat("Test message"))
    assert len(ollama.conversation_history) > 0

    ollama.reset_conversation()
    assert len(ollama.conversation_history) == 0


# tests/ollama/test_model_manager.py
import pytest

from llm7.ollama.exceptions import ModelNotFoundError
from llm7.ollama.model_manager import model_manager


def test_available_models():
    """Test getting available models."""
    models = model_manager.available_models
    assert isinstance(models, list)
    assert len(models) > 0


def test_model_config():
    """Test model configuration access."""
    model = model_manager.current_model
    config = model_manager.current_config

    assert isinstance(config, dict)
    assert "temperature" in config
    assert "max_tokens" in config


def test_add_model_config():
    """Test adding new model configuration."""
    new_config = {
        "temperature": 0.8,
        "max_tokens": 2000,
        "system_prompt": "Test prompt",
    }

    model_manager.add_model_config("test-model", new_config)
    assert "test-model" in model_manager.available_models

    retrieved_config = model_manager.get_model_config("test-model")
    assert (
        retrieved_config["temperature"] == new_config["temperature"]
    )


def test_update_model_config():
    """Test updating model configuration."""
    model = model_manager.current_model
    original_temp = model_manager.current_config["temperature"]

    model_manager.update_model_config(model, {"temperature": 0.9})
    assert model_manager.current_config["temperature"] == 0.9

    # Reset to original
    model_manager.update_model_config(
        model, {"temperature": original_temp}
    )
