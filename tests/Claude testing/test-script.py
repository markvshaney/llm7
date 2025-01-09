import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional

import chromadb
import tiktoken
from chromadb.config import Settings
from chromadb.utils import embedding_functions


@dataclass
class Document:
    content: str
    metadata: Dict
    id: str = None


class MemoryManager:
    def __init__(
        self,
        collection_name: str = "documents",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize ChromaDB client
        self.client = chromadb.Client(
            Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=".\\chroma_db",
            )
        )

        # Set up embedding function
        self.embedding_function = (
            embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=embedding_model
            )
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
        )

        # Initialize tokenizer for chunking
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def _chunk_text(self, text: str) -> List[str]:
        tokens = self.tokenizer.encode(text)
        chunks = []

        i = 0
        while i < len(tokens):
            # Get chunk of tokens
            chunk_end = min(i + self.chunk_size, len(tokens))
            chunk = tokens[i:chunk_end]

            # Convert back to text
            chunk_text = self.tokenizer.decode(chunk)
            chunks.append(chunk_text)

            # Move to next chunk, considering overlap
            i += self.chunk_size - self.chunk_overlap

        return chunks

    def add_document(self, document: Document) -> List[str]:
        # Generate document ID if not provided
        if document.id is None:
            document.id = str(uuid.uuid4())

        # Chunk the document
        chunks = self._chunk_text(document.content)
        chunk_ids = []

        # Store each chunk
        for i, chunk in enumerate(chunks):
            chunk_id = f"{document.id}_chunk_{i}"
            chunk_metadata = {
                **document.metadata,
                "document_id": document.id,
                "chunk_index": i,
                "total_chunks": len(chunks),
            }

            self.collection.add(
                documents=[chunk],
                metadatas=[chunk_metadata],
                ids=[chunk_id],
            )
            chunk_ids.append(chunk_id)

        return chunk_ids

    def search(
        self,
        query: str,
        n_results: int = 5,
        metadata_filter: Optional[Dict] = None,
    ) -> List[Dict]:
        where = metadata_filter if metadata_filter else {}

        results = self.collection.query(
            query_texts=[query], n_results=n_results, where=where
        )

        # Format results
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

    def get_document_chunks(self, document_id: str) -> List[Dict]:
        results = self.collection.get(
            where={"document_id": document_id}
        )

        # Format and sort chunks by index
        chunks = []
        for i in range(len(results["ids"])):
            chunks.append(
                {
                    "id": results["ids"][i],
                    "content": results["documents"][i],
                    "metadata": results["metadatas"][i],
                }
            )

        return sorted(
            chunks, key=lambda x: x["metadata"]["chunk_index"]
        )


def main():
    # Initialize Memory Manager
    print("Initializing Memory Manager...")
    memory_manager = MemoryManager(
        collection_name="test_collection",
        chunk_size=200,  # Smaller chunk size for testing
        chunk_overlap=20,
    )

    # Create a test document
    test_content = """
    Python is a high-level, interpreted programming language known for its simplicity and readability.
    It supports multiple programming paradigms, including procedural, object-oriented, and functional programming.
    Python's design philosophy emphasizes code readability with its notable use of significant whitespace.

    Key features of Python include:
    - Dynamic typing and dynamic binding
    - Automatic memory management
    - Support for multiple programming paradigms
    - Extensive standard library

    Python is widely used in:
    1. Web development (Django, Flask)
    2. Data science and machine learning (NumPy, Pandas, TensorFlow)
    3. Artificial intelligence and neural networks
    4. Scientific computing
    5. Automation and scripting
    """

    test_document = Document(
        content=test_content,
        metadata={
            "title": "Python Programming Overview",
            "category": "programming",
            "language": "Python",
            "type": "documentation",
        },
    )

    # Add document to memory store
    print("\nAdding test document...")
    chunk_ids = memory_manager.add_document(test_document)
    print(f"Created {len(chunk_ids)} chunks")

    # Test search functionality
    print("\nTesting search functionality...")

    # Test case 1: General search
    print("\nSearch query: 'programming paradigms'")
    results = memory_manager.search("programming paradigms")
    for i, result in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"Content: {result['content']}")
        print(f"Distance: {result['distance']:.4f}")

    # Test case 2: Search with metadata filter
    print("\nSearch with metadata filter (category: programming)")
    filtered_results = memory_manager.search(
        "machine learning",
        metadata_filter={"category": "programming"},
    )
    for i, result in enumerate(filtered_results, 1):
        print(f"\nFiltered Result {i}:")
        print(f"Content: {result['content']}")
        print(f"Distance: {result['distance']:.4f}")

    # Test retrieving all chunks for the document
    print("\nRetrieving all document chunks...")
    all_chunks = memory_manager.get_document_chunks(test_document.id)
    print(f"Retrieved {len(all_chunks)} chunks")
    for i, chunk in enumerate(all_chunks, 1):
        print(f"\nChunk {i}:")
        print(f"Content: {chunk['content']}")
        print(f"Chunk Index: {chunk['metadata']['chunk_index']}")


if __name__ == "__main__":
    main()
