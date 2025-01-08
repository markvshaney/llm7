from typing import List, Dict, Optional, Union
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import tiktoken
import uuid
from dataclasses import dataclass
import time
from functools import wraps

@dataclass
class Document:
    content: str
    metadata: Dict
    id: str = None

def retry_operation(max_attempts: int = 3, delay: float = 1.0):
    """Decorator for retrying operations that might fail temporarily"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < max_attempts - 1:
                        time.sleep(delay * (attempt + 1))  # Exponential backoff
            raise last_error
        return wrapper
    return decorator

class MemoryManager:
    def __init__(
        self,
        collection_name: str = "documents",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        """Initialize the Memory Manager with ChromaDB."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize ChromaDB client
        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory="./chroma_db"
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
        """Split text into chunks based on token count."""
        tokens = self.tokenizer.encode(text)
        chunks = []
        
        i = 0
        while i < len(tokens):
            chunk_end = min(i + self.chunk_size, len(tokens))
            chunk = tokens[i:chunk_end]
            chunk_text = self.tokenizer.decode(chunk)
            chunks.append(chunk_text)
            i += (self.chunk_size - self.chunk_overlap)
            
        return chunks

    @retry_operation()
    def add_document(self, document: Document) -> List[str]:
        """Add a document to the memory store."""
        if document.id is None:
            document.id = str(uuid.uuid4())
            
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

    @retry_operation()
    def add_documents(self, documents: List[Document]) -> Dict[str, List[str]]:
        """Add multiple documents in batch."""
        results = {}
        all_chunks = []
        all_metadatas = []
        all_ids = []
        
        for document in documents:
            if document.id is None:
                document.id = str(uuid.uuid4())
                
            chunks = self._chunk_text(document.content)
            document_chunk_ids = []
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{document.id}_chunk_{i}"
                chunk_metadata = {
                    **document.metadata,
                    "document_id": document.id,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
                
                all_chunks.append(chunk)
                all_metadatas.append(chunk_metadata)
                all_ids.append(chunk_id)
                document_chunk_ids.append(chunk_id)
            
            results[document.id] = document_chunk_ids
        
        # Batch add all chunks
        self.collection.add(
            documents=all_chunks,
            metadatas=all_metadatas,
            ids=all_ids
        )
        
        return results

    @retry_operation()
    def update_document(self, document: Document) -> List[str]:
        """Update an existing document."""
        if document.id is None:
            raise ValueError("Document ID is required for updates")
            
        # Delete existing chunks
        self.delete_document(document.id)
        
        # Add updated document
        return self.add_document(document)

    @retry_operation()
    def delete_document(self, document_id: str):
        """Delete a document and all its chunks."""
        # Get all chunk IDs for the document
        chunks = self.get_document_chunks(document_id)
        chunk_ids = [chunk['id'] for chunk in chunks]
        
        # Delete chunks
        if chunk_ids:
            self.collection.delete(ids=chunk_ids)

    @retry_operation()
    def search(
        self,
        query: str,
        n_results: int = 5,
        metadata_filter: Optional[Dict] = None,
        return_documents: bool = False
    ) -> Union[List[Dict], Dict[str, List[Dict]]]:
        """
        Search for relevant document chunks.
        
        Args:
            query: Search query
            n_results: Number of results to return
            metadata_filter: Optional filter for metadata fields
            return_documents: If True, groups results by document
            
        Returns:
            If return_documents is False: List of matching chunks
            If return_documents is True: Dict of document_id -> list of chunks
        """
        where = metadata_filter if metadata_filter else {}
        
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where
        )
        
        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                'id': results['ids'][0][i],
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i]
            })
        
        if not return_documents:
            return formatted_results
            
        # Group results by document
        grouped_results = {}
        for result in formatted_results:
            doc_id = result['metadata']['document_id']
            if doc_id not in grouped_results:
                grouped_results[doc_id] = []
            grouped_results[doc_id].append(result)
            
        return grouped_results

    @retry_operation()
    def get_document_chunks(self, document_id: str) -> List[Dict]:
        """Retrieve all chunks for a specific document."""
        results = self.collection.get(
            where={"document_id": document_id}
        )
        
        chunks = []
        for i in range(len(results['ids'])):
            chunks.append({
                'id': results['ids'][i],
                'content': results['documents'][i],
                'metadata': results['metadatas'][i]
            })
            
        return sorted(chunks, key=lambda x: x['metadata']['chunk_index'])