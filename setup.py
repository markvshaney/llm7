from setuptools import setup, find_packages

setup(
    name="llm7",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "requests>=2.25.1",
        "chromadb>=0.4.0",
        "python-dotenv>=0.19.0",
        "pyyaml>=6.0.1",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.3.0",
            "isort>=5.10.1",
            "flake8>=4.0.1",
        ],
    },
    python_requires=">=3.8",
    author="Your Name",
    description="LLM interaction framework with memory management",
    keywords="llm, ai, memory, ollama",
)