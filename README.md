# LLM7

A framework for interacting with Large Language Models (LLMs) with integrated memory management.

## Features

- Memory management for LLM conversations
- Ollama integration for local LLM hosting
- Configurable storage backends
- Extensible architecture for multiple LLM providers

## Installation

```bash
# Clone the repository
git clone [your-repo-url]
cd llm7

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"
```

## Project Structure

```
llm7/
├── llm7/              # Main package
│   ├── memory/       # Memory management
│   ├── ollama/       # Ollama integration
│   └── utils/        # Utilities
├── config/           # Configuration files
├── logs/             # Log files
├── storage/          # Data storage
├── tests/            # Test files
└── utils/            # Project utilities
```

## Usage

```python
from llm7.ollama import OllamaInterface
from llm7.memory import MemoryManager

# Initialize components
llm = OllamaInterface()
memory = MemoryManager()

# Use the system
response = llm.generate("Your prompt here")
```

## Development

```bash
# Run tests
pytest

# Format code
black llm7
isort llm7
```

## License

[Your chosen license]

## Contributing

[Your contribution guidelines]
