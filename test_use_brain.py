import numpy as np
import os
import asyncio
import time
import random
import json
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

def json_serialize(obj):
    """
    Recursively convert an object and its nested contents to be JSON serializable.
    Handles NumPy types, lists, dictionaries, and other common types.
    
    Args:
        obj: Any Python object to make JSON serializable
        
    Returns:
        JSON serializable version of the object
    """
    if obj is None:
        return None
        
    # Handle NumPy types
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.int32, np.int64, np.uint32, np.uint64)):
        return int(obj)
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.bool_)):
        return bool(obj)
        
    # Handle iterables
    if isinstance(obj, list) or isinstance(obj, tuple):
        return [json_serialize(item) for item in obj]
        
    # Handle dictionaries
    if isinstance(obj, dict):
        return {str(key): json_serialize(value) for key, value in obj.items()}
        
    # Special handling for custom objects with __dict__
    if hasattr(obj, '__dict__'):
        return {key: json_serialize(value) for key, value in obj.__dict__.items() 
                if not key.startswith('_')}
                
    # Try to make it JSON serializable, or convert to string as last resort
    try:
        json.dumps(obj)
        return obj
    except (TypeError, OverflowError):
        return str(obj)

async def load_brain():
    """
    Legacy function that uses the brain_singleton to load vectors.
    This maintains compatibility with existing code.
    """
    # Clean up existing files if they exist
    if os.path.exists("faiss_index.bin"):
        try:
            os.remove("faiss_index.bin")
            print("Removed existing faiss_index.bin file")
        except Exception as e:
            print(f"ERROR: Could not remove faiss_index.bin: {e}")
    
    if os.path.exists("metadata.pkl"):
        try:
            os.remove("metadata.pkl")
            print("Removed existing metadata.pkl file")
        except Exception as e:
            print(f"ERROR: Could not remove metadata.pkl: {e}")
    
    # Check for Pinecone API key
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        print("ERROR: PINECONE_API_KEY environment variable is required")
        return False
    
    # Use the singleton's loading method with force_delete=True
    return await load_brain_vectors(get_current_graph_version(), force_delete=True)

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
    
    # Ensure brain is loaded before querying with extra verification
    if not is_brain_loaded() or not hasattr(brain, 'faiss_index') or (hasattr(brain, 'faiss_index') and brain.faiss_index.ntotal == 0):
        print(f"Brain vectors not loaded or index empty. Loading vectors for graph version: {get_current_graph_version()}")
        success = await load_brain_vectors(get_current_graph_version(), force_delete=True)
        if not success:
            print("Failed to load brain. Cannot proceed with query.")
            return {"error": "Failed to load brain"}
        
        # After loading, verify that vectors were actually loaded
        if hasattr(brain, 'faiss_index'):
            vector_count = brain.faiss_index.ntotal
            print(f"Verified brain loaded with {vector_count} vectors")
            if vector_count == 0:
                print("WARNING: Brain loaded but no vectors found in FAISS index")
                return {"error": "Brain loaded but index is empty", "graph_version_id": get_current_graph_version()}
        else:
            print("WARNING: FAISS index not initialized after loading")
            return {"error": "FAISS index not initialized", "graph_version_id": get_current_graph_version()}

    # Additional verification before query
    if hasattr(brain, 'faiss_index'):
        vector_count = brain.faiss_index.ntotal
        print(f"FAISS index contains {vector_count} vectors before querying")
        if vector_count == 0:
            print("WARNING: FAISS index is empty, attempting reload")
            # Force a reload
            reset_brain()
            brain = get_brain()
            success = await load_brain_vectors(get_current_graph_version(), force_delete=True)
            if not success or brain.faiss_index.ntotal == 0:
                return {"error": "FAISS index remains empty after reload", "graph_version_id": get_current_graph_version()}
    
    # Now perform the query
    try:
        results = await brain.get_similar_vectors_by_text(input_text, top_k=3)
        
        # Convert results to a serializable format
        formatted_results = []
        for vector_id, vector, metadata, similarity in results:
            formatted_results.append({
                "vector_id": vector_id,
                "similarity": similarity,
                "metadata": metadata,
                "vector_preview": vector[:10] if hasattr(vector, '__getitem__') else []
            })
        
        response = {
            "query": input_text,
            "graph_version_id": get_current_graph_version(),
            "vector_count": brain.faiss_index.ntotal if hasattr(brain, 'faiss_index') else 0,
            "results": formatted_results
        }
        
        # Make sure everything is JSON serializable
        return json_serialize(response)
    except Exception as e:
        import traceback
        print(f"Error in flick_out: {e}")
        print(traceback.format_exc())
        return {
            "error": f"Error processing query: {str(e)}",
            "query": input_text,
            "graph_version_id": get_current_graph_version()
        }

async def query_queries(input_text:str=""):
    """Query the brain with the provided input text or run batch queries."""
    global brain
    
    # Ensure brain is loaded before querying with extra verification
    if not is_brain_loaded() or not hasattr(brain, 'faiss_index') or (hasattr(brain, 'faiss_index') and brain.faiss_index.ntotal == 0):
        print(f"Brain vectors not loaded or index empty. Loading vectors for graph version: {get_current_graph_version()}")
        success = await load_brain_vectors(get_current_graph_version(), force_delete=True)
        if not success:
            print("Failed to load brain. Cannot proceed with query.")
            return
        
        # After loading, verify that vectors were actually loaded
        if hasattr(brain, 'faiss_index'):
            vector_count = brain.faiss_index.ntotal
            print(f"Verified brain loaded with {vector_count} vectors")
            if vector_count == 0:
                print("WARNING: Brain loaded but no vectors found in FAISS index")
                return
        else:
            print("WARNING: FAISS index not initialized after loading")
            return
    
    # Additional verification before query
    if hasattr(brain, 'faiss_index'):
        vector_count = brain.faiss_index.ntotal
        print(f"FAISS index contains {vector_count} vectors before querying")
        if vector_count == 0:
            print("WARNING: FAISS index is empty, attempting reload")
            # Force a reload
            reset_brain()
            brain = get_brain()
            success = await load_brain_vectors(get_current_graph_version(), force_delete=True)
            if not success or brain.faiss_index.ntotal == 0:
                print("ERROR: FAISS index remains empty after reload")
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
    query_text = "Xử lý từ chối"
    asyncio.run(query_queries(query_text))
    
    
    