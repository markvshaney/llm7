from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import tiktoken
import uuid
from dataclasses import dataclass

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
        chunk_overlap: int = 50
    ):
        """Initialize the Memory Manager with ChromaDB.
        
        Args:
            collection_name: Name of the ChromaDB collection
            embedding_model: Name of the embedding model to use
            chunk_size: Maximum number of tokens per chunk
            chunk_overlap: Number of overlapping tokens between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize ChromaDB client
        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=".\\chroma_db"
        ))
        
        # Set up embedding function
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )
        
        # Initialize tokenizer for chunking
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks based on token count.
        
        Args:
            text: Text to be chunked
            
        Returns:
            List of text chunks
        """
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
            i += (self.chunk_size - self.chunk_overlap)
            
        return chunks

    def add_document(self, document: Document) -> List[str]:
        """Add a document to the memory store.
        
        Args:
            document: Document object containing content and metadata
            
        Returns:
            List of chunk IDs created
        """
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
                "total_chunks": len(chunks)
            }
            
            self.collection.add(
                documents=[chunk],
                metadatas=[chunk_metadata],
                ids=[chunk_id]
            )
            chunk_ids.append(chunk_id)
            
        return chunk_ids

    def search(
        self,
        query: str,
        n_results: int = 5,
        metadata_filter: Optional[Dict] = None
    ) -> List[Dict]:
        """Search for relevant document chunks.
        
        Args:
            query: Search query
            n_results: Number of results to return
            metadata_filter: Optional filter for metadata fields
            
        Returns:
            List of matching chunks with their metadata
        """
        where = metadata_filter if metadata_filter else {}
        
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                'id': results['ids'][0][i],
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i]
            })
            
        return formatted_results

    def get_document_chunks(self, document_id: str) -> List[Dict]:
        """Retrieve all chunks for a specific document.
        
        Args:
            document_id: ID of the document
            
        Returns:
            List of chunks with their metadata
        """
        results = self.collection.get(
            where={"document_id": document_id}
        )
        
        # Format and sort chunks by index
        chunks = []
        for i in range(len(results['ids'])):
            chunks.append({
                'id': results['ids'][i],
                'content': results['documents'][i],
                'metadata': results['metadatas'][i]
            })
            
        return sorted(chunks, key=lambda x: x['metadata']['chunk_index'])
