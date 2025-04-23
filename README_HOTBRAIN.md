# HotBrain Module

## Overview
The `hotbrain.py` module provides optimized vector database operations with advanced caching capabilities for improved performance and reduced API costs. This module was created to refactor and enhance the embedding and vector query operations from the database.py module.

## Key Features

1. **Enhanced Embedding Cache**
   - Persistent cache with TTL (Time-To-Live)
   - LRU-like eviction strategy for optimal memory usage
   - MD5 hashing for longer text to optimize cache key storage
   - Cache statistics monitoring
   - Disk-based persistence for cache durability between restarts

2. **Optimized Query Operations**
   - Batch processing of multiple queries at once
   - Parallel execution with controlled concurrency
   - Smart filtering of relevant vector indexes
   - Comprehensive error handling with retries

3. **Performance Optimizations**
   - Rate limiting to avoid API throttling
   - Semaphore-based concurrency control
   - Single embedding generation for repeated queries
   - Smart batching for OpenAI API optimization (20 embeddings per request)

## Main Functions

### Embedding Management
- `get_cached_embedding(text)`: Get or generate an embedding with caching
- `batch_get_embeddings(texts)`: Generate embeddings for multiple texts efficiently

### Knowledge Querying
- `query_graph_knowledge(version_id, query, top_k)`: Query across all brains in a graph
- `query_brain_with_embeddings_batch(query_embeddings, namespace, brain_id, top_k)`: Process multiple queries in a single brain

### Brain Structure Management
- `get_version_brain_banks(version_id)`: Get all brain bank information with caching

## Usage Example

```python
from hotbrain import query_graph_knowledge, batch_get_embeddings

# Example: Batch process multiple text pieces for embeddings
texts = ["What is machine learning?", "How does AI work?", "Tell me about neural networks"]
embedding_dict = await batch_get_embeddings(texts)

# Example: Query knowledge across a graph
results = await query_graph_knowledge(
    version_id="a1b2c3d4-e5f6-7890-abcd-1234567890ab",
    query="How do transformers work?",
    top_k=5
)
```

## Performance Benefits

The enhanced caching system can provide significant performance improvements:

1. **API Cost Reduction**: By caching embeddings, the module reduces the number of API calls to OpenAI, potentially saving thousands of dollars per month for high-volume applications.

2. **Latency Improvement**: Cache hits avoid the network round-trip to the embedding API, reducing response times by 300-500ms per query.

3. **Batch Efficiency**: Processing embeddings in batches of 20 reduces API calls by up to 95% compared to individual calls.

## Future Enhancements

- Redis-based distributed cache for multi-server deployments
- Adaptive TTL based on query frequency
- Pre-warming cache with common queries
- Compression for embedding storage 