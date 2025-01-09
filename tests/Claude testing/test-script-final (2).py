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
        """Initialize the Memory Manager with ChromaDB."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize ChromaDB client with new configuration
        self.client = chromadb.PersistentClient(path=".\\chroma_db")

        # Set up embedding function
        self.embedding_function = (
            embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=embedding_model
            )
        )

        # Delete collection if it exists
        try:
            self.client.delete_collection(collection_name)
        except:
            pass

        # Create fresh collection
        self.collection = self.client.create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
        )

        # Initialize tokenizer for chunking
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks based on token count."""
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
        """Add a document to the memory store."""
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
        """Search for relevant document chunks."""
        # Construct where clause for metadata filtering
        where = None
        if metadata_filter:
            if len(metadata_filter) == 1:
                # Single filter can use direct equality
                where = metadata_filter
            else:
                # Multiple filters need to use $and with individual conditions
                where = {
                    "$and": [
                        {key: {"$eq": value}}
                        for key, value in metadata_filter.items()
                    ]
                }

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
        """Retrieve all chunks for a specific document."""
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

    # Create test documents
    test_documents = [
        Document(
            content="""
            Python is a high-level programming language known for its simplicity and readability.
            It supports multiple programming paradigms, including procedural, object-oriented, and
            functional programming. Python emphasizes clean syntax and powerful libraries.

            Common uses include web development, data science, AI, and automation tasks.
            """,
            metadata={
                "title": "Python Programming Overview",
                "category": "programming",
                "language": "Python",
                "type": "documentation",
            },
        ),
        Document(
            content="""
            Machine learning is a subset of artificial intelligence that focuses on developing
            systems that can learn from and make decisions based on data. Deep learning,
            a subset of machine learning, uses neural networks with multiple layers to analyze
            complex patterns in data.

            Popular machine learning frameworks include TensorFlow, PyTorch, and scikit-learn.
            These tools provide extensive capabilities for both research and production applications.
            """,
            metadata={
                "title": "Machine Learning Introduction",
                "category": "ai",
                "field": "data science",
                "type": "educational",
            },
        ),
        Document(
            content="""
            Docker containers provide a lightweight, consistent environment for applications.
            Each container includes all dependencies needed to run the application, ensuring
            it works the same way across different systems and environments.

            Key benefits include isolation, portability, and efficient resource usage. Docker
            enables microservices architecture and simplifies deployment processes.
            """,
            metadata={
                "title": "Docker Container Basics",
                "category": "devops",
                "technology": "containers",
                "type": "tutorial",
            },
        ),
    ]

    # Add documents to memory store
    print("\nAdding test documents...")
    for i, doc in enumerate(test_documents, 1):
        chunk_ids = memory_manager.add_document(doc)
        print(f"Document {i}: Created {len(chunk_ids)} chunks")

    # Test different search scenarios
    search_tests = [
        {
            "name": "Semantic similarity search",
            "query": "artificial intelligence frameworks",
            "filter": None,
            "description": "Testing semantic understanding of related concepts",
        },
        {
            "name": "Technical concept search",
            "query": "environment setup and dependencies",
            "filter": None,
            "description": "Finding technical content across different domains",
        },
        {
            "name": "Educational content search",
            "query": "learning and tutorials",
            "filter": {"type": "educational"},
            "description": "Filtering by content type",
        },
        {
            "name": "Domain-specific search",
            "query": "software development tools",
            "filter": {"category": "programming"},
            "description": "Filtering by technical domain",
        },
        {
            "name": "Multi-field filter search",
            "query": "deployment",
            "filter": {"category": "devops", "type": "tutorial"},
            "description": "Testing multiple metadata filters",
        },
        {
            "name": "Natural language query",
            "query": "how do containers help with deployment",
            "filter": None,
            "description": "Testing natural language understanding",
        },
    ]

    for test in search_tests:
        print("\n" + "=" * 80)
        print(f"Test: {test['name']}")
        print(f"Description: {test['description']}")
        print(f"Query: '{test['query']}'")
        if test["filter"]:
            print(f"Metadata Filter: {test['filter']}")
        print("-" * 40)

        results = memory_manager.search(
            test["query"], metadata_filter=test["filter"]
        )

        if not results:
            print("\nNo results found matching the criteria.")
        else:
            for i, result in enumerate(results, 1):
                print(f"\nResult {i}:")
                print(f"Title: {result['metadata']['title']}")
                print(
                    f"Category: {result['metadata'].get('category', 'N/A')}"
                )
                print(
                    f"Type: {result['metadata'].get('type', 'N/A')}"
                )
                print(
                    f"Relevance Score: {1 - result['distance']:.2%}"
                )
                print("\nContent:")
                print(result["content"].strip())
                print(f"\nDistance: {result['distance']:.4f}")


if __name__ == "__main__":
    main()
