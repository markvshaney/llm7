# llm7/ollama/logger.py
import logging
import time
from functools import wraps
from pathlib import Path
from config.settings import settings

# Set up logger
logger = logging.getLogger('ollama')
logger.setLevel(logging.INFO)

# Create logs directory if it doesn't exist
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)

# Create handlers
file_handler = logging.FileHandler('logs/ollama.log')
console_handler = logging.StreamHandler()

# Create formatters and add it to handlers
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
file_formatter = logging.Formatter(log_format)
console_formatter = logging.Formatter(log_format)

file_handler.setFormatter(file_formatter)
console_handler.setFormatter(console_formatter)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

def log_interaction(func):
    """Decorator to log model interactions."""
    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        start_time = time.time()
        method_name = func.__name__
        model_name = self._current_model if hasattr(self, '_current_model') else 'unknown'
        
        try:
            logger.info(f"Starting {method_name} with model {model_name}")
            result = await func(self, *args, **kwargs)
            
            execution_time = time.time() - start_time
            logger.info(
                f"Completed {method_name} with model {model_name} "
                f"in {execution_time:.2f} seconds"
            )
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(
                f"Error in {method_name} with model {model_name} "
                f"after {execution_time:.2f} seconds: {str(e)}"
            )
            raise
    
    return wrapper

def log_model_switch(func):
    """Decorator to log model switching."""
    @wraps(func)
    def wrapper(self, model_name, *args, **kwargs):
        logger.info(f"Switching model from {self._current_model} to {model_name}")
        try:
            result = func(self, model_name, *args, **kwargs)
            logger.info(f"Successfully switched to model {model_name}")
            return result
        except Exception as e:
            logger.error(f"Failed to switch to model {model_name}: {str(e)}")
            raise
    
    return wrapper