"""Memory management module for document storage and retrieval using ChromaDB.

This module provides classes and utilities for managing document storage,
chunking, and semantic search capabilities using ChromaDB as the backend.
"""

import re
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Literal, Optional

import chromadb
import tiktoken
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from langchain.text_splitter import (
    CharacterTextSplitter,
    Language,
    MarkdownTextSplitter,
    PythonCodeTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)


@dataclass
class Document:
    """Represents a document with content and metadata."""

    content: str
    metadata: Dict[str, Any]
    id: Optional[str] = None
    content_type: str = "text"  # text, code, markdown, etc.


@dataclass
class ChunkingConfig:
    """Configuration for document chunking strategies and parameters."""

    strategy: Literal[
        "token", "recursive", "character", "code", "markdown", "semantic"
    ] = "token"
    chunk_size: int = 500
    chunk_overlap: int = 50
    separators: List[str] = field(default_factory=lambda: ["\n\n", "\n", ".", " "])
    model_name: str = "cl100k_base"
    language: Optional[Language] = None  # For code chunking
    chunk_validators: List[Callable[[str], bool]] = field(default_factory=list)
    post_processors: List[Callable[[str], str]] = field(default_factory=list)


class ChunkValidator(ABC):
    """Abstract base class for chunk validators."""

    @abstractmethod
    def validate(self, chunk: str) -> bool:
        """Validate a text chunk according to specific criteria."""
        pass


class MinLengthValidator(ChunkValidator):
    """Validator ensuring chunks meet minimum length requirements."""

    def __init__(self, min_length: int) -> None:
        """Initialize the validator with minimum length threshold.

        Args:
            min_length: Minimum required length for chunks.
        """
        self.min_length = min_length

    def validate(self, chunk: str) -> bool:
        """Check if chunk meets minimum length requirement."""
        return len(chunk.strip()) >= self.min_length


class CodeBlockValidator(ChunkValidator):
    """Validator ensuring code blocks maintain structural integrity."""

    def validate(self, chunk: str) -> bool:
        """Check if code block brackets are properly balanced."""
        opening_count = chunk.count("{")
        closing_count = chunk.count("}")
        return opening_count == closing_count


class SemanticChunker:
    """Chunks text based on semantic boundaries using regex patterns."""

    def __init__(self) -> None:
        """Initialize semantic chunking patterns."""
        self.patterns = {
            "paragraph": r"\n\n+",
            "sentence": r"(?<=[.!?])\s+",
            "section": r"(?m)^#{1,6}\s+",  # Markdown headers
            "code_block": r"```[\s\S]*?```",  # Markdown code blocks
        }

    def chunk_text(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Split text into semantic chunks while preserving structure.

        Args:
            text: Input text to be chunked.
            chunk_size: Maximum size of each chunk.
            chunk_overlap: Number of overlapping characters between chunks.

        Returns:
            List of text chunks.
        """
        # First, protect special blocks (like code blocks)
        protected_blocks: Dict[str, str] = {}
        for i, match in enumerate(re.finditer(self.patterns["code_block"], text)):
            placeholder = f"__CODE_BLOCK_{i}__"
            protected_blocks[placeholder] = match.group(0)
            text = text.replace(match.group(0), placeholder)

        # Split into semantic units (paragraphs, then sentences if needed)
        chunks: List[str] = []
        paragraphs = re.split(self.patterns["paragraph"], text)

        current_chunk: List[str] = []
        current_length = 0

        for para in paragraphs:
            if current_length + len(para) > chunk_size:
                sentences = re.split(self.patterns["sentence"], para)
                for sentence in sentences:
                    if current_length + len(sentence) > chunk_size:
                        if current_chunk:
                            chunks.append(" ".join(current_chunk))
                        current_chunk = [sentence]
                        current_length = len(sentence)
                    else:
                        current_chunk.append(sentence)
                        current_length += len(sentence)
            else:
                current_chunk.append(para)
                current_length += len(para)

            if current_length >= chunk_size:
                chunks.append(" ".join(current_chunk))
                overlap_tokens = current_chunk[-1:] if chunk_overlap > 0 else []
                current_chunk = overlap_tokens
                current_length = sum(len(t) for t in current_chunk)

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        # Restore protected blocks
        for i, chunk in enumerate(chunks):
            for placeholder, original in protected_blocks.items():
                chunks[i] = chunks[i].replace(placeholder, original)

        return chunks


class MemoryManager:
    """Manages document storage, retrieval, and search using ChromaDB."""

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunking_config: Optional[ChunkingConfig] = None,
    ) -> None:
        """Initialize the Memory Manager.

        Args:
            collection_name: Name of the ChromaDB collection.
            embedding_model: Name of the embedding model to use.
            chunking_config: Configuration for text chunking.
        """
        self.chunking_config = chunking_config or ChunkingConfig()

        # Initialize ChromaDB client
        self.client = chromadb.Client(
            Settings(chroma_db_impl="duckdb+parquet", persist_directory="./chroma_db")
        )

        # Set up embedding function
        self.embedding_function = (
            embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=embedding_model
            )
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name, embedding_function=self.embedding_function
        )

        # Initialize chunking components
        self._initialize_chunking_strategy()
        self.semantic_chunker = SemanticChunker()

    def _initialize_chunking_strategy(self) -> None:
        """Initialize the text splitter based on configuration."""
        common_args = {
            "chunk_size": self.chunking_config.chunk_size,
            "chunk_overlap": self.chunking_config.chunk_overlap,
        }

        if self.chunking_config.strategy == "token":
            self.tokenizer = tiktoken.get_encoding(self.chunking_config.model_name)
            self.text_splitter = TokenTextSplitter(
                encoding_name=self.chunking_config.model_name, **common_args
            )
        elif self.chunking_config.strategy == "recursive":
            self.text_splitter = RecursiveCharacterTextSplitter(
                separators=self.chunking_config.separators, **common_args
            )
        elif self.chunking_config.strategy == "character":
            self.text_splitter = CharacterTextSplitter(
                separator=self.chunking_config.separators[0], **common_args
            )
        elif self.chunking_config.strategy == "code":
            if not self.chunking_config.language:
                raise ValueError("Language must be specified for code chunking")
            self.text_splitter = PythonCodeTextSplitter(**common_args)
        elif self.chunking_config.strategy == "markdown":
            self.text_splitter = MarkdownTextSplitter(**common_args)
        elif self.chunking_config.strategy == "semantic":
            # Semantic chunking is handled separately
            pass
        else:
            raise ValueError(
                f"Unsupported chunking strategy: {self.chunking_config.strategy}"
            )

    def _validate_chunk(self, chunk: str) -> bool:
        """Run all configured validators on a chunk."""
        return all(
            validator(chunk) for validator in self.chunking_config.chunk_validators
        )

    def _process_chunk(self, chunk: str) -> str:
        """Apply all configured post-processors to a chunk."""
        for processor in self.chunking_config.post_processors:
            chunk = processor(chunk)
        return chunk

    def _chunk_text(self, text: str, content_type: str = "text") -> List[str]:
        """Split text into chunks using the configured strategy.

        Args:
            text: Input text to be chunked.
            content_type: Type of content being chunked.

        Returns:
            List of text chunks.
        """
        if self.chunking_config.strategy == "semantic":
            chunks = self.semantic_chunker.chunk_text(
                text,
                self.chunking_config.chunk_size,
                self.chunking_config.chunk_overlap,
            )
        elif self.chunking_config.strategy == "token":
            tokens = self.tokenizer.encode(text)
            chunks = []

            i = 0
            while i < len(tokens):
                chunk_end = min(i + self.chunking_config.chunk_size, len(tokens))
                chunk = tokens[i:chunk_end]
                chunk_text = self.tokenizer.decode(chunk)
                chunks.append(chunk_text)
                i += (
                    self.chunking_config.chunk_size - self.chunking_config.chunk_overlap
                )
        else:
            chunks = self.text_splitter.split_text(text)

        # Validate and process chunks
        processed_chunks = []
        for chunk in chunks:
            if self._validate_chunk(chunk):
                processed_chunk = self._process_chunk(chunk)
                processed_chunks.append(processed_chunk)

        return processed_chunks

    def add_document(self, document: Document) -> List[str]:
        """Add a document to the memory store.

        Args:
            document: Document object containing content and metadata.

        Returns:
            List of chunk IDs for the added document.
        """
        if document.id is None:
            document.id = str(uuid.uuid4())

        chunks = self._chunk_text(document.content, document.content_type)
        chunk_ids = []

        for i, chunk in enumerate(chunks):
            chunk_id = f"{document.id}_chunk_{i}"
            chunk_metadata = {
                **document.metadata,
                "document_id": document.id,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "chunking_strategy": self.chunking_config.strategy,
                "content_type": document.content_type,
            }

            self.collection.add(
                documents=[chunk], metadatas=[chunk_metadata], ids=[chunk_id]
            )
            chunk_ids.append(chunk_id)

        return chunk_ids

    def search(
        self,
        query: str,
        n_results: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search for relevant document chunks.

        Args:
            query: Search query string.
            n_results: Number of results to return.
            metadata_filter: Optional filter for metadata fields.

        Returns:
            List of matching chunks with metadata and distance scores.
        """
        where = metadata_filter if metadata_filter else {}

        results = self.collection.query(
            query_texts=[query], n_results=n_results, where=where
        )

        formatted_results = []
        for i in range(len(results["ids"][0])):
            formatted_results.append(
                {
                    "id": results["ids"][0][i],
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                }
            )

        return formatted_results

    def get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """Retrieve all chunks for a specific document.

        Args:
            document_id: ID of the document to retrieve chunks for.

        Returns:
            List of document chunks with metadata, sorted by chunk index.
        """
        results = self.collection.get(where={"document_id": document_id})

        chunks = []
        for i in range(len(results["ids"])):
            chunks.append(
                {
                    "id": results["ids"][i],
                    "content": results["documents"][i],
                    "metadata": results["metadatas"][i],
                }
            )

        return sorted(chunks, key=lambda x: x["metadata"]["chunk_index"])
