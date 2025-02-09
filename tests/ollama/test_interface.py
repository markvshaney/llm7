import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from llm7.ollama.exceptions import ModelNotFoundError
from llm7.ollama.interface import OllamaInterface
from llm7.ollama.ollama_model_manager import model_manager


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
    async def mock_stream():
        test_chunks = ["Hello", " ", "World", "!"]
        for chunk in test_chunks:
            yield chunk

    # Mock the generate method to return our mock stream
    ollama.generate = AsyncMock(return_value=mock_stream())

    async for chunk in await ollama.generate("Test prompt", stream=True):
        chunks.append(chunk)

    assert len(chunks) > 0
    assert all(isinstance(chunk, str) for chunk in chunks)
    assert chunks == ["Hello", " ", "World", "!"]


@pytest.mark.asyncio
async def test_chat_conversation(ollama):
    """Test multi-turn conversation."""
    response1 = await ollama.chat("Hello")
    assert isinstance(response1, str)

    response2 = await ollama.chat("How are you?")
    assert isinstance(response2, str)

    history = ollama.conversation_history
    assert len(history) == 4  # 2 user messages + 2 assistant responses


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


@pytest.mark.asyncio
async def test_streaming_error_handling(ollama):
    """Test error handling in streaming response."""
    async def error_stream():
        yield "Starting"
        raise ValueError("Test error")

    ollama.generate = AsyncMock(return_value=error_stream())

    chunks = []
    with pytest.raises(ValueError, match="Test error"):
        async for chunk in await ollama.generate("Test prompt", stream=True):
            chunks.append(chunk)

    assert chunks == ["Starting"]  # Should get first chunk before error