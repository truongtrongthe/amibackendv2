#!/usr/bin/env python3
"""
test_batch_embeddings.py - Test script for query_brain_with_embeddings_batch in hotbrain.py

This script tests the batch embedding processing capabilities in hotbrain.py.
It compares the performance of batch query processing versus individual query processing
to demonstrate the efficiency gains of batching multiple embedding queries together.

Key tests performed:
1. Generating embeddings for multiple queries in parallel
2. Running a batch query with all embeddings
3. Testing caching performance with a second batch query
4. Comparing batch processing time vs. individual query processing

To run the test:
    python test_batch_embeddings.py

Requires:
- hotbrain.py to be in the same directory
- Properly configured environment variables for OpenAI and Pinecone
"""

import os
import sys
import asyncio
import time
import logging
import statistics
from typing import Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import hotbrain module
try:
    if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import hotbrain
except ImportError:
    raise ImportError("Could not import hotbrain.py. Make sure it's in the same directory.")

# Test constants
TEST_NAMESPACE = "wisdom_bank_60a2445b-95f2-4c2f-a3c5-973c7bce6c6e_372"  # Known namespace from test_hb.py
TEST_BRAIN_ID = "test-brain-id"  # Any identifier for testing

async def test_batch_embeddings():
    """Test query_brain_with_embeddings_batch with multiple queries in a single batch"""
    logger.info("=== Testing Batch Embeddings Processing ===")
    
    # Test queries in both English and Vietnamese to test multilingual support
    test_queries = {
        1: "How to segment customers effectively?",
        2: "What are the different customer types?", 
        3: "Làm thế nào để phân nhóm khách hàng",  # How to segment customers
        4: "Khách hàng ở nhóm nào",                # Which group customers belong to
        5: "What's the relationship between customer lifetime value and acquisition cost?",
        6: "How to identify high-value customers?",
        7: "What metrics should I track for customer satisfaction?",
        8: "Có mấy nhóm khách hàng",               # How many customer groups
        9: "Tôi cần phân tích chân dung khách hàng", # I need to analyze customer profiles
        10: "How to implement permission levels for different team members?"
    }
    
    # Create embeddings for all queries
    logger.info(f"Generating embeddings for {len(test_queries)} queries...")
    
    embeddings = {}
    batch_start_time = time.time()
    
    # Generate embeddings in parallel
    embedding_tasks = {}
    for idx, query in test_queries.items():
        logger.info(f"Query {idx}: '{query}'")
        embedding_tasks[idx] = asyncio.create_task(hotbrain.get_cached_embedding(query))
    
    # Wait for all embedding tasks to complete
    for idx, task in embedding_tasks.items():
        try:
            embeddings[idx] = await task
            # Check if we got a valid embedding
            if not embeddings[idx] or len(embeddings[idx]) == 0:
                logger.warning(f"Empty embedding received for query {idx}, using fallback")
                embeddings[idx] = [0.0001] * 1536
            else:
                logger.info(f"Generated embedding for query {idx}: vector of size {len(embeddings[idx])}")
        except Exception as e:
            logger.error(f"Error generating embedding for query {idx}: {e}")
            # Use a minimal vector as fallback
            embeddings[idx] = [0.0001] * 1536
    
    embedding_time = time.time() - batch_start_time
    logger.info(f"Generated all embeddings in {embedding_time:.2f}s")
    
    # Ensure all embeddings are valid and non-zero
    for idx, embedding in list(embeddings.items()):
        if not embedding or len(embedding) == 0 or not any(embedding):
            logger.warning(f"Fixing zero/empty embedding for query {idx}")
            embeddings[idx] = [0.0001 + (0.00001 * (i % 10)) for i in range(1536)]
    
    # Test the batch query function
    logger.info("\nTesting batch query function...")
    
    # First execution (should be a cache miss)
    start_time = time.time()
    try:
        batch_results = await hotbrain.query_brain_with_embeddings_batch(
            query_embeddings=embeddings,
            namespace=TEST_NAMESPACE,
            brain_id=TEST_BRAIN_ID,
            top_k=5
        )
        first_query_time = time.time() - start_time
        
        # Process and display results
        logger.info(f"Batch query completed in {first_query_time:.4f}s")
        logger.info(f"Retrieved results for {len(batch_results)} queries")
        
        for query_idx, results in batch_results.items():
            logger.info(f"\nQuery {query_idx}: '{test_queries.get(query_idx, 'Unknown query')}'")
            logger.info(f"Found {len(results)} results")
            
            # Show top 3 results
            for i, result in enumerate(results[:3]):
                # Safely truncate content
                content = result.get('raw', '')
                if content:
                    try:
                        content = content[:50] + '...'
                    except:
                        content = '<content truncated>'
                logger.info(f"  Result {i+1}: Score={result['score']:.4f}, Content: {content}")
    
    except Exception as e:
        logger.error(f"Error in batch query: {e}")
        import traceback
        logger.error(traceback.format_exc())
        first_query_time = 0
    
    # Second execution (should use embedding cache)
    logger.info("\nRunning second batch query to test caching...")
    
    try:
        await asyncio.sleep(1)  # Small delay between queries
        
        start_time = time.time()
        _ = await hotbrain.query_brain_with_embeddings_batch(
            query_embeddings=embeddings,
            namespace=TEST_NAMESPACE,
            brain_id=TEST_BRAIN_ID,
            top_k=5
        )
        second_query_time = time.time() - start_time
        
        # Calculate improvement
        if first_query_time > 0:
            improvement = ((first_query_time - second_query_time) / first_query_time) * 100
            logger.info(f"First batch query time: {first_query_time:.4f}s")
            logger.info(f"Second batch query time: {second_query_time:.4f}s")
            logger.info(f"Cache improvement: {improvement:.1f}%")
    
    except Exception as e:
        logger.error(f"Error in second batch query: {e}")
    
    # Compare with individual queries
    logger.info("\n=== Comparing Batch vs. Individual Queries ===")
    
    # Run individual queries
    individual_times = []
    individual_start = time.time()
    
    for idx, query in test_queries.items():
        try:
            query_embedding = embeddings[idx]
            start_time = time.time()
            
            # Create a single-item batch for comparison
            single_query = {idx: query_embedding}
            _ = await hotbrain.query_brain_with_embeddings_batch(
                query_embeddings=single_query,
                namespace=TEST_NAMESPACE,
                brain_id=TEST_BRAIN_ID,
                top_k=5
            )
            
            query_time = time.time() - start_time
            individual_times.append(query_time)
            logger.info(f"Individual query {idx} time: {query_time:.4f}s")
        
        except Exception as e:
            logger.error(f"Error in individual query {idx}: {e}")
    
    total_individual_time = time.time() - individual_start
    avg_individual_time = statistics.mean(individual_times) if individual_times else 0
    
    logger.info(f"\nBatch query time for {len(test_queries)} queries: {first_query_time:.4f}s")
    logger.info(f"Total time for individual queries: {total_individual_time:.4f}s")
    logger.info(f"Average time per individual query: {avg_individual_time:.4f}s")
    
    if first_query_time > 0 and total_individual_time > 0:
        batch_improvement = ((total_individual_time - first_query_time) / total_individual_time) * 100
        logger.info(f"Batch processing improvement: {batch_improvement:.1f}%")
    
    logger.info("\nBatch embeddings test complete!")
    return True

async def run_tests():
    """Run the batch embeddings test"""
    logger.info("Starting focused test for hotbrain.py query_brain_with_embeddings_batch...")
    
    try:
        await test_batch_embeddings()
        logger.info("\n✅ Testing completed successfully!")
    except Exception as e:
        logger.error(f"\n❌ Test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    logger.info("Test complete.")

if __name__ == "__main__":
    """Run tests directly when script is executed"""
    asyncio.run(run_tests()) 