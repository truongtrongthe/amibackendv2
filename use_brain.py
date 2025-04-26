import numpy as np
import os
import asyncio
import time
import random
from active_brain import ActiveBrain
from brain_singleton import init_brain, get_brain, reset_brain, set_graph_version, get_current_graph_version, load_brain_vectors, is_brain_loaded
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Pinecone index and graph version to test
PINECONE_INDEX = "9well"
GRAPH_VERSION_ID = "bd5b8bc1-d0cb-4e3e-9cd7-68a8563366fc"

# Sample test queries
test_queries = [
    "What is the capital of France?",
    "How does machine learning work?",
    "Explain neural networks"
]

# Initialize global brain instance
init_brain(
    dim=1536, 
    namespace="", 
    graph_version_ids=[GRAPH_VERSION_ID],
    pinecone_index_name=PINECONE_INDEX
)

# Get reference to the global brain instance
brain = get_brain()

async def load_brain():
    """
    Legacy function that uses the brain_singleton to load vectors.
    This maintains compatibility with existing code.
    """
    # Clean up existing files if they exist
    if os.path.exists("faiss_index.bin"):
        os.remove("faiss_index.bin")
    if os.path.exists("metadata.pkl"):
        os.remove("metadata.pkl")
    
    # Check for Pinecone API key
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        print("ERROR: PINECONE_API_KEY environment variable is required")
        return False
    
    # Use the singleton's loading method
    return await load_brain_vectors(get_current_graph_version())

async def flick_out(input_text: str = "", graph_version_id: str = "") -> dict:
    """Return vectors from brain without printing results."""
    global brain
    
    # If a new graph version is requested, update the configuration
    if graph_version_id and graph_version_id != get_current_graph_version():
        # Set the new graph version which will reset the brain if needed
        version_changed = set_graph_version(graph_version_id)
        if version_changed:
            # Get the updated brain reference
            brain = get_brain()
    
    # Ensure brain is loaded before querying
    if not is_brain_loaded():
        success = await load_brain_vectors()
        if not success:
            print("Failed to load brain. Cannot proceed with query.")
            return {"error": "Failed to load brain"}

    results = await brain.get_similar_vectors_by_text(input_text, top_k=3)
    
    # Convert results to a serializable format
    formatted_results = []
    for vector_id, vector, metadata, similarity in results:
        formatted_results.append({
            "vector_id": vector_id,
            "similarity": similarity,
            "metadata": metadata,
            "vector_preview": vector[:10].tolist()  # Show first 10 values
        })
    
    return {
        "query": input_text,
        "graph_version_id": get_current_graph_version(),
        "results": formatted_results
    }

async def query_queries(input_text:str=""):
    """Query the brain with the provided input text or run batch queries."""
    # Ensure brain is loaded before querying
    if not is_brain_loaded():
        success = await load_brain_vectors()
        if not success:
            print("Failed to load brain. Cannot proceed with query.")
            return
    
    # Use the provided input text or default test queries
    use_batch = not input_text
    
    if input_text:
        results = await brain.get_similar_vectors_by_text(input_text, top_k=3)
        # Show all results for individual queries too
        for result_idx, (vector_id, vector, metadata, similarity) in enumerate(results):
            print(f"\nResult {result_idx + 1}: {vector_id} (Score: {similarity:.6f})")
                
            # Show vector details
            print(f"Vector dimension: {len(vector)}")
            print(f"First 5 values: {vector[:5]}")
                
            # Print raw content if available
            if 'raw' in metadata:
                raw_content = metadata['raw']
                # Truncate if too long
                if len(raw_content) > 200:
                    raw_content = raw_content[:200] + "..."
                print(f"Raw content: {raw_content}")
                    
            # Print other useful metadata fields if available
            useful_fields = ['chunk_id', 'source', 'categories_primary']
            for field in useful_fields:
                if field in metadata:
                    print(f"{field}: {metadata[field]}")
                
            if result_idx < len(results) - 1:  # Don't print separator after last result
                print("-" * 50)  # Separator between results
    
    # Run batch query if no input text provided
    if use_batch:
        print("\n--- BATCH QUERY PERFORMANCE ---")
        
        start_time = time.time()
        batch_results = await brain.batch_similarity_search(test_queries, top_k=3)
        batch_time = time.time() - start_time
        
        print(f"Total batch time for {len(test_queries)} queries: {batch_time:.6f} seconds")
        print(f"Average time per query: {batch_time/len(test_queries):.6f} seconds")
        
        # Print a summary of the results
        print("\n--- BATCH RESULTS SUMMARY ---")
        for query_index, query in enumerate(test_queries):
            results = batch_results[query]
            print(f"\nQuery {query_index + 1}: \"{query}\"")
            print(f"Found {len(results)} similar vectors")
            
            # Show all results instead of just the top one
            for result_idx, (vector_id, vector, metadata, similarity) in enumerate(results):
                print(f"\nResult {result_idx + 1}: {vector_id} (Score: {similarity:.6f})")
                
                # Show vector snippets
                print(f"Vector dimension: {len(vector)}")
                print(f"First 5 values: {vector[:5]}")
                
                # Print raw content if available
                if 'raw' in metadata:
                    raw_content = metadata['raw']
                    # Truncate if too long
                    if len(raw_content) > 200:
                        raw_content = raw_content[:200] + "..."
                    print(f"Raw content: {raw_content}")
                    
                # Print other useful metadata fields if available
                useful_fields = ['chunk_id', 'source', 'categories_primary']
                for field in useful_fields:
                    if field in metadata:
                        print(f"{field}: {metadata[field]}")
                
                print("-" * 50)  # Separator between results

if __name__ == "__main__":
    # Initial load of brain 
    asyncio.run(load_brain())
    
    # Example of multiple queries without reloading brain
    query_text = "What is the capital of France?"
    asyncio.run(query_queries(query_text))
    
    # Another query without reloading brain
    query_text2 = "Tell me about neural networks"
    asyncio.run(query_queries(query_text2))
    
    # Run the batch test queries
    asyncio.run(query_queries())
    