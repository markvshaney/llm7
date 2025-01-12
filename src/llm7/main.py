import asyncio
from typing import Dict, List

from llm7.ollama.ollama_model_manager import model_manager
from llm7.ollama.ollama_config import OllamaLLM

# Main module for LLM3 Assistant that interacts with LLM models, manages conversations, and processes user input.

class OllamaAssistant:
    """Handles interaction with the LLM and manages conversation state."""

    def __init__(self) -> None:
        """Initialize the assistant with an LLM instance and an empty conversation history."""
        self.llm = OllamaLLM()
        self._conversation_history: List[Dict[str, str]] = []

    def show_models(self) -> None:
        """Display available models and the current model."""
        print("\nAvailable models:")
        for model in model_manager.available_models:
            prefix = (
                "*" if model == model_manager.current_model else " "
            )
            print(f"{prefix} {model}")

    def switch_model(self, model_name: str) -> None:
        """Switch to a different model."""
        try:
            model_manager.set_current_model(model_name)
            print(f"\nSwitched to model: {model_name}")
            self._conversation_history = []  # Reset conversation
        except ValueError as e:
            print(f"\nError: {str(e)}")

    def show_commands(self) -> None:
        """Display the list of available commands."""
        print("\nAvailable commands:")
        print("- 'quit': Exit the program")
        print("- 'reset': Reset conversation history")
        print("- 'history': Show conversation history")
        print("- 'models': Show available models")
        print("- 'switch <model>': Switch to a different model")
        print("- 'help': Show this help message")

    async def process_input(self, user_input: str) -> str:
        """Process the user's input and return the assistant's response."""
        if user_input.lower().startswith("switch "):
            model_name = user_input.split(" ")[1]
            self.switch_model(model_name)
            return ""

        # Store user message
        self._conversation_history.append(
            {"role": "user", "content": user_input}
        )

        # Generate response
        response = self.llm.generate(user_input)

        if "error" in response:
            return f"Error: {response['error']}"

        # Store assistant response
        assistant_response = response.get(
            "response", "No response generated"
        )
        self._conversation_history.append(
            {"role": "assistant", "content": assistant_response}
        )

        return assistant_response

def initialize_assistant(model_name: str) -> OllamaAssistant:
    """Initialize the assistant with a specific model."""
    assistant = OllamaAssistant()
    assistant.switch_model(model_name)
    return assistant

async def main() -> None:
    """Run the Ollama Assistant and handle user interaction."""
    assistant = OllamaAssistant()

    print("Ollama Assistant")
    print("-------------")
    print("Type 'help' for available commands")
    assistant.show_models()

    while True:
        user_input = input(
            "\nEnter your question (or command): "
        ).strip()

        if user_input.lower() == "quit":
            print("\nGoodbye!")
            break

        elif user_input.lower() == "help":
            assistant.show_commands()
            continue

        elif user_input.lower() == "reset":
            assistant._conversation_history = []
            print("\nConversation history reset.")
            continue

        elif user_input.lower() == "models":
            assistant.show_models()
            continue

        elif user_input.lower() == "history":
            history = assistant._conversation_history
            if not history:
                print("\nNo conversation history yet.")
            else:
                print("\nConversation history:")
                for msg in history:
                    role = msg["role"].capitalize()
                    print(f"\n{role}: {msg['content']}")
            continue

        # Process user input
        response = await assistant.process_input(user_input)
        if response:
            print("\nResponse:", response)


if __name__ == "__main__":
    asyncio.run(main())
