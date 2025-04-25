import numpy as np
import os
import asyncio
import time
import random
from active_brain import ActiveBrain
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Pinecone index and graph version to test
PINECONE_INDEX = "9well"
GRAPH_VERSION_ID = "bd5b8bc1-d0cb-4e3e-9cd7-68a8563366fc"

# Test queries
TEST_QUERIES = [
    "Tôi bị xuất tinh sớm và muốn được tư vấn",
    "Tôi muốn tìm hiểu về các phương pháp điều trị vô sinh",
    "Tôi đang gặp rối loạn cương dương và cần tìm kiếm giải pháp",
    "Tôi muốn biết chi phí khám sức khỏe sinh sản tại bệnh viện này"
]

async def test_active_brain():
    """Simplified test focusing only on loading vectors and querying."""
    print("\n=== Testing ActiveBrain Loading and Querying ===")
    
    # Clean up existing files
    if os.path.exists("faiss_index.bin"):
        os.remove("faiss_index.bin")
    if os.path.exists("metadata.pkl"):
        os.remove("metadata.pkl")
    
    # Check for Pinecone API key
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        print("ERROR: PINECONE_API_KEY environment variable is required")
        return
    
    # Initialize ActiveBrain with Pinecone index and graph version ID
    print(f"\nInitializing ActiveBrain with graph version: {GRAPH_VERSION_ID}")
    brain = ActiveBrain(
        dim=1536, 
        namespace="", 
        graph_version_ids=[GRAPH_VERSION_ID],
        pinecone_index_name=PINECONE_INDEX
    )

    # Test loading vectors from graph version
    print("\n=== LOADING VECTORS FROM GRAPH VERSION ===")
    try:
        print(f"Loading vectors from graph version: {GRAPH_VERSION_ID}")
        await brain.load_all_vectors_from_graph_version(GRAPH_VERSION_ID)
        
        # Print FAISS index details after loading
        vector_count = brain.faiss_index.ntotal
        print(f"\n=== FAISS INDEX AFTER LOADING ===")
        print(f"Loaded {vector_count} vectors into FAISS")
        print(f"Vector IDs in memory: {len(brain.vector_ids)}")
        print(f"Metadata entries: {len(brain.metadata)}")
        
        if vector_count == 0:
            print("No vectors loaded. Skipping remaining tests.")
            return
            
        # Print sample of loaded data
        if brain.vector_ids:
            print("\n=== SAMPLE OF LOADED VECTORS ===")
            for i, vector_id in enumerate(list(brain.vector_ids)[:3]):
                print(f"Vector {i+1} ID: {vector_id}")
                if vector_id in brain.metadata:
                    print(f"Metadata keys: {list(brain.metadata[vector_id].keys())}")
            
    except Exception as e:
        print(f"ERROR loading vectors: {e}")
        import traceback
        print(traceback.format_exc())
        return
    
    # Test similarity search with performance measurements
    print("\n=== PERFORMANCE TESTING WITH MULTIPLE QUERIES ===")
    
    # Define test queries
    test_queries = [
        "Tôi bị xuất tinh sớm và muốn được tư vấn",
        "Tôi đang gặp rối loạn cương dương và cần tìm kiếm giải pháp",
        "Tôi muốn tìm hiểu về các phương pháp điều trị vô sinh",
        "Tôi muốn biết chi phí khám sức khỏe sinh sản tại bệnh viện này",
        "Làm thế nào để cải thiện chất lượng tinh trùng"
    ]
    
    # First, test individual queries for comparison
    print("\n--- INDIVIDUAL QUERY PERFORMANCE ---")
    individual_query_times = []
    
    for query_index, test_query in enumerate(test_queries[:2]):  # Test only first 2 queries individually
        print(f"\nQuery {query_index + 1}: \"{test_query}\"")
        
        # Time the embedding + query process
        start_time = time.time()
        results = await brain.get_similar_vectors_by_text(test_query, top_k=3)
        query_time = time.time() - start_time
        individual_query_times.append(query_time)
        
        print(f"Query time: {query_time:.6f} seconds")
        print(f"Found {len(results)} similar vectors")
        
        # Show the top result
        if results:
            vector_id, vector, metadata, similarity = results[0]
            print(f"\nTop Result: {vector_id} (Score: {similarity:.6f})")
            print(f"Vector dimension: {len(vector)}")
            print(f"First 10 values: {vector[:10]}")
            
            # Print raw content if available
            if 'raw' in metadata:
                raw_content = metadata['raw']
                # Truncate if too long
                if len(raw_content) > 300:
                    raw_content = raw_content[:300] + "..."
                print(f"Raw content: {raw_content}")
    
    # Now test batch query performance
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
        
        if results:
            vector_id, vector, metadata, similarity = results[0]
            print(f"Top Result: {vector_id} (Score: {similarity:.6f})")
    
    # Print performance comparison
    if individual_query_times:
        avg_individual_time = sum(individual_query_times) / len(individual_query_times)
        print("\n=== PERFORMANCE COMPARISON ===")
        print(f"Average individual query time: {avg_individual_time:.6f} seconds")
        print(f"Average batch query time: {batch_time/len(test_queries):.6f} seconds")
        
        if avg_individual_time > 0:
            speedup = avg_individual_time / (batch_time/len(test_queries))
            print(f"Speedup factor: {speedup:.2f}x")
    
    # Memory usage
    print("\n=== MEMORY USAGE ===")
    mem_usage = brain.get_memory_usage()
    print(f"Current memory usage: {mem_usage:.2f} MB")
    
    print("\nTest completed")

if __name__ == "__main__":
    asyncio.run(test_active_brain()) 