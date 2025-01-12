# src/llm7/ollama/interface.py
from typing import Any, AsyncGenerator, Dict, List, Optional, Union
from .exceptions import ModelNotFoundError

class OllamaInterface:
    """Interface for interacting with Ollama models."""

    def __init__(self, model_name: str = "llama2"):
        self.current_model = model_name
        self.conversation_history: List[Dict[str, str]] = []
        self._setup()

    def _setup(self) -> None:
        """Initialize the interface configuration."""
        pass

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> str | AsyncGenerator[str, None]:
        """Generate text response from the model."""
        if stream:

            async def response_stream():
                # Placeholder for streaming implementation
                yield "Response chunk 1"
                yield "Response chunk 2"

            return response_stream()
        return "Generated response"

    async def chat(self, message: str) -> str:
        """Send a message and get a response in chat format."""
        self.conversation_history.append({"role": "user", "content": message})
        response = await self.generate(message)
        self.conversation_history.append({"role": "assistant", "content": response})
        return response

    def switch_model(self, model_name: str) -> None:
        """Switch to a different model."""
        if model_name not in self.get_available_models():
            raise ModelNotFoundError(f"Model {model_name} not found")
        self.current_model = model_name

    def inject_context(self, context: str) -> None:
        """Inject system context into the conversation."""
        self.conversation_history.append({"role": "system", "content": context})

    def reset_conversation(self) -> None:
        """Reset the conversation history."""
        self.conversation_history = []

    @staticmethod
    def get_available_models() -> List[str]:
        """Get list of available models."""
        return ["llama2", "mistral", "codellama"]  # Placeholder list
