from typing import List, Dict, Optional
import chromadb
from chromadb.api import Collection
from chromadb.utils import embedding_functions
from dataclasses import dataclass, field
import uuid

@dataclass
class DocumentMetadata:
    """Metadata for a document"""
    source: str
    doc_type: str
    timestamp: str
    tags: List[str] = field(default_factory=list)
    custom_metadata: Dict = field(default_factory=dict)

@dataclass
class Document:
    """Document class with content and metadata"""
    content: str
    metadata: DocumentMetadata
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

class MemoryManager:
    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        collection_name: str = "documents",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """Initialize Memory Manager with ChromaDB configuration.
        
        Args:
            persist_dir: Directory for ChromaDB persistence
            collection_name: Name for the ChromaDB collection
            embedding_model: Name of the embedding model to use
        """
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_dir)
        
        # Set up embedding function
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        
        # Delete collection if it exists (for clean start)
        try:
            self.client.delete_collection(collection_name)
        except ValueError:
            pass
            
        # Create collection with proper configuration
        self.collection = self.client.create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}  # Specify distance metric
        )
        
    def get_collection(self) -> Collection:
        """Get the current ChromaDB collection.
        
        Returns:
            The current ChromaDB collection
        """
        return self.collection
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the current collection.
        
        Returns:
            Dictionary containing collection statistics
        """
        count = self.collection.count()
        return {
            "total_documents": count,
            "collection_name": self.collection.name,
            "metadata": self.collection.metadata
        }