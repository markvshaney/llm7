from memory_manager import MemoryManager, Document

def main():
    # Initialize Memory Manager
    print("Initializing Memory Manager...")
    memory_manager = MemoryManager(
        collection_name="test_collection",
        chunk_size=200,  # Smaller chunk size for testing
        chunk_overlap=20
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
            "type": "documentation"
        }
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
        metadata_filter={"category": "programming"}
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
