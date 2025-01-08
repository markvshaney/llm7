# examples/ollama_examples.py
import asyncio
from llm7.ollama.interface import ollama_interface
from llm7.ollama.model_manager import model_manager

async def basic_generation():
    """Example of basic text generation."""
    response = await ollama_interface.generate(
        "Explain how to make a cup of coffee."
    )
    print("Basic Generation Response:", response)

async def streaming_chat():
    """Example of streaming chat conversation."""
    print("\nStreaming Chat Response:")
    async for chunk in ollama_interface.chat(
        "Tell me a short story about a robot.",
        stream=True
    ):
        print(chunk, end='', flush=True)
    print("\n")

async def multi_turn_conversation():
    """Example of multi-turn conversation."""
    print("\nMulti-turn Conversation:")
    
    # First message
    response1 = await ollama_interface.chat(
        "What are the three laws of robotics?"
    )
    print("Assistant:", response1)
    
    # Follow-up question
    response2 = await ollama_interface.chat(
        "Who created these laws?"
    )
    print("Assistant:", response2)
    
    # Show conversation history
    print("\nConversation History:")
    for message in ollama_interface.conversation_history:
        print(f"{message['role'].title()}: {message['content']}\n")

async def model_switching():
    """Example of switching between models."""
    print("\nModel Switching Example:")
    
    # Use default model
    response1 = await ollama_interface.generate(
        "Write a function to calculate fibonacci numbers."
    )
    print("Default Model Response:", response1)
    
    # Switch to codellama
    ollama_interface.switch_model('codellama')
    response2 = await ollama_interface.generate(
        "Write a function to calculate fibonacci numbers."
    )
    print("CodeLlama Response:", response2)

async def custom_parameters():
    """Example of using custom parameters."""
    response = await ollama_interface.generate(
        "Brainstorm creative names for a tech startup.",
        temperature=0.9,  # More creative
        max_tokens=100    # Shorter response
    )
    print("\nCustom Parameters Response:", response)

async def error_handling():
    """Example of error handling."""
    try:
        # Try to use non-existent model
        ollama_interface.switch_model('non-existent-model')
    except Exception as e:
        print("\nCaught error:", str(e))
    
    # Switch back to valid model
    ollama_interface.switch_model('mistral')

async def main():
    """Run all examples."""
    print("Running Ollama Interface Examples...")
    
    await basic_generation()
    await streaming_chat()
    await multi_turn_conversation()
    await model_switching()
    await custom_parameters()
    await error_handling()

if __name__ == "__main__":
    asyncio.run(main())