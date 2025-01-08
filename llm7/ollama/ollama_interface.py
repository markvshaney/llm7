# llm7/ollama/interface.py
from typing import Dict, Any, Optional, AsyncGenerator, Union
import ollama
from .model_manager import model_manager

class OllamaInterface:
    """Interface for interacting with Ollama models."""
    
    def __init__(self):
        self._current_conversation = []
        self._system_prompt = None
        self._load_config()
    
    def _load_config(self) -> None:
        """Load current configuration settings."""
        config = model_manager.current_config
        self._system_prompt = config.get('system_prompt')
        
    def _prepare_messages(self, new_message: Union[str, Dict[str, str]]) -> list:
        """Prepare message list for Ollama API."""
        messages = []
        
        # Add system prompt if set
        if self._system_prompt:
            messages.append({
                "role": "system",
                "content": self._system_prompt
            })
        
        # Add conversation history
        messages.extend(self._current_conversation)
        
        # Add new message
        if isinstance(new_message, str):
            messages.append({
                "role": "user",
                "content": new_message
            })
        else:
            messages.append(new_message)
            
        return messages
    
    def _prepare_options(self, **kwargs) -> Dict[str, Any]:
        """Prepare options for Ollama API call."""
        # Start with current model config
        options = model_manager.current_config.copy()
        
        # Update with global options
        options.update(model_manager.options)
        
        # Override with any provided kwargs
        options.update(kwargs)
        
        return options
    
    async def generate(self, prompt: str, stream: bool = None, **kwargs) -> Union[str, AsyncGenerator[str, None]]:
        """
        Generate response using current model.
        
        Args:
            prompt: Input prompt
            stream: Override streaming setting from config
            **kwargs: Optional parameters to override configuration
            
        Returns:
            Generated response text or stream of response chunks
        """
        options = self._prepare_options(**kwargs)
        messages = self._prepare_messages(prompt)
        
        # Override streaming if specified
        if stream is not None:
            options['stream'] = stream
        
        if options.get('stream', False):
            async for chunk in ollama.chat(
                model=model_manager.current_model,
                messages=messages,
                stream=True,
                **options
            ):
                yield chunk['message']['content']
        else:
            response = await ollama.chat(
                model=model_manager.current_model,
                messages=messages,
                **options
            )
            return response['message']['content']
    
    async def chat(self, message: str, stream: bool = None, **kwargs) -> Union[str, AsyncGenerator[str, None]]:
        """
        Chat with memory of conversation history.
        
        Args:
            message: User message
            stream: Override streaming setting from config
            **kwargs: Optional parameters to override configuration
            
        Returns:
            Assistant's response or stream of response chunks
        """
        # Add user message to conversation
        self._current_conversation.append({
            "role": "user",
            "content": message
        })
        
        options = self._prepare_options(**kwargs)
        messages = self._prepare_messages([])  # Empty list since message is already in conversation
        
        # Override streaming if specified
        if stream is not None:
            options['stream'] = stream
            
        if options.get('stream', False):
            response_content = []
            async for chunk in ollama.chat(
                model=model_manager.current_model,
                messages=messages,
                stream=True,
                **options
            ):
                content = chunk['message']['content']
                response_content.append(content)
                yield content
                
            # Add complete response to conversation history
            self._current_conversation.append({
                "role": "assistant",
                "content": ''.join(response_content)
            })
        else:
            response = await ollama.chat(
                model=model_manager.current_model,
                messages=messages,
                **options
            )
            
            # Add response to conversation history
            self._current_conversation.append({
                "role": "assistant",
                "content": response['message']['content']
            })
            
            return response['message']['content']
    
    def reset_conversation(self) -> None:
        """Reset the conversation history."""
        self._current_conversation = []
        self._load_config()  # Reload config in case it changed
    
    def switch_model(self, model_name: str) -> None:
        """
        Switch to a different model.
        Resets conversation history and loads new model config.
        """
        model_manager.set_current_model(model_name)
        self.reset_conversation()
    
    @property
    def conversation_history(self) -> list:
        """Get the current conversation history."""
        return self._current_conversation.copy()
    
    def inject_context(self, context: str, role: str = "system") -> None:
        """
        Inject context into the conversation.
        Useful for providing additional context or instructions.
        """
        self._current_conversation.append({
            "role": role,
            "content": context
        })

# Create default instance
ollama_interface = OllamaInterface()
