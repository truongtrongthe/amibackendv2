"""
hotbrain.py - Functions for handling batched semantic queries with Pinecone-based persistent storage
"""

import os
import asyncio
import time
import json
import logging
from typing import Dict, List, Optional, Any, Union
import tenacity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache hit statistics for monitoring
_cache_stats = {
    "embedding_exact_hits": 0,
    "embedding_similar_hits": 0, 
    "embedding_misses": 0,
    "brain_hits": 0,
    "brain_misses": 0,
    "node_stores": 0,
    "node_updates": 0,
    "node_hits": 0,
    "node_misses": 0
}

# Helper function to safely extract vectors from Pinecone response
def extract_vectors_from_response(response):
    """Handle different Pinecone API response formats"""
    try:
        # New Pinecone API (where response is a FetchResponse object)
        if hasattr(response, 'vectors'):
            return response.vectors
        # Legacy Pinecone API (where response is a dict with 'vectors' key)
        elif hasattr(response, 'get'):
            return response.get("vectors", {})
        # Direct access if neither method works
        elif isinstance(response, dict) and "vectors" in response:
            return response["vectors"]
        else:
            # Last resort if no vectors attribute is found
            logger.warning(f"Unexpected Pinecone response format: {type(response)}")
            return {}
    except Exception as e:
        logger.error(f"Error extracting vectors from Pinecone response: {e}")
        return {}

# Cache TTL settings
_node_cache_ttl = 86400  # 24 hours for node embeddings
_brain_cache_ttl = 600   # 10 minutes for brain structure info

# Ensure we're consistently using the same EMBEDDINGS object from utilities
try:
    from utilities import EMBEDDINGS, logger
    logger.info("Successfully imported EMBEDDINGS from utilities module")
except ImportError:
    logger.error("Failed to import EMBEDDINGS from utilities - required for embedding generation")
    raise

# Initialize Supabase client
from supabase import create_client, Client
spb_url = os.getenv("SUPABASE_URL")
spb_key = os.getenv("SUPABASE_KEY")

if spb_url and spb_key:
    supabase = create_client(spb_url, spb_key)
    logger.info("Supabase client initialized")
else:
    logger.error("Supabase credentials not found - required for brain structure retrieval")
    raise ValueError("Missing Supabase credentials")

# Initialize Pinecone client
from pinecone import Pinecone

# Initialize with environment variables
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
ami_index_name = os.getenv("PRESET", "ami-index")
ent_index_name = os.getenv("ENT", "ent-index")

ami_index = pc.Index(ami_index_name)
ent_index = pc.Index(ent_index_name)
# Store the names directly as attributes for easier access
ami_index.display_name = ami_index_name
ent_index.display_name = ent_index_name
logger.info(f"Pinecone indexes initialized: {ami_index_name}, {ent_index_name}")

def index_name(index) -> str:
    """Get the display name of an index"""
    try:
        # First try our custom attribute
        if hasattr(index, 'display_name'):
            return index.display_name
        # Then try standard attributes
        elif hasattr(index, 'name'):
            return f"{index.name}"
        elif hasattr(index, '_name'):  # Check for protected name attribute
            return f"{index._name}"
        elif hasattr(index, 'describe_index_stats'):  # Try to check if it's a valid index object
            return "valid-index-no-name"
        return "unknown-index"
    except Exception:
        return "unknown-index"

async def get_cached_embedding(text: str, similarity_threshold: float = 0.70) -> List[float]:
    """
    Get embedding for text from Pinecone if similar exists, otherwise generate new one.
    
    Args:
        text: The text to generate embedding for
        similarity_threshold: Threshold for considering embeddings similar
        
    Returns:
        List of floats representing the embedding vector
    """
    global _cache_stats
    
    # Generate a deterministic ID for this text
    import hashlib
    text_hash = hashlib.md5(text.encode()).hexdigest()
    
    # Check if embedding exists in Pinecone cache namespace
    try:
        # First check exact match by ID
        cache_key = f"embedding_{text_hash}"
        
        results = await asyncio.to_thread(
            ami_index.fetch,
            ids=[cache_key],
            namespace="embeddings_cache"
        )
        
        # Extract vectors safely using the helper function
        vectors = extract_vectors_from_response(results)
        
        if cache_key in vectors:
            _cache_stats["embedding_exact_hits"] += 1
            # Handle both dict-based and object-based vector formats
            if isinstance(vectors[cache_key], dict) and "values" in vectors[cache_key]:
                return vectors[cache_key]["values"]
            elif hasattr(vectors[cache_key], "values"):
                return vectors[cache_key].values
            else:
                logger.warning(f"Unexpected vector format in cache hit: {type(vectors[cache_key])}")
                # Fall through to generate a new embedding
        
        # No exact match, check for semantic similarity
        if len(text) > 5:  # Only for non-trivial text
            # Generate embedding for input text
            embedding = await EMBEDDINGS.aembed_query(text)
            
            # Query for similar cached embeddings
            results = await asyncio.to_thread(
                ami_index.query,
                vector=embedding,
                top_k=1,
                score_threshold=similarity_threshold,
                include_metadata=True,
                namespace="embeddings_cache"
            )
            
            # Handle both old and new Pinecone API response formats
            matches = []
            if hasattr(results, 'matches'):
                matches = results.matches
            elif hasattr(results, 'get'):
                matches = results.get("matches", [])
            
            if matches and len(matches) > 0:
                # Extract score from the match (handle both dict and object formats)
                match_score = matches[0].get("score", 0) if isinstance(matches[0], dict) else getattr(matches[0], "score", 0)
                
                if match_score >= similarity_threshold:
                    _cache_stats["embedding_similar_hits"] += 1
                    logger.info(f"[EMBEDDING_CACHE] Similar cache hit ({match_score:.4f}) for: '{text[:30]}...'")
                    
                    # Extract values from the match (handle both dict and object formats)
                    if isinstance(matches[0], dict) and "values" in matches[0]:
                        return matches[0]["values"]
                    elif hasattr(matches[0], "values"):
                        return matches[0].values
                    else:
                        logger.warning(f"Cannot extract values from match: {type(matches[0])}")
            
            # No similar embedding found, store the newly generated one
            vector = {
                "id": cache_key,
                "values": embedding,
                "metadata": {
                    "original_text": text[:1000],  # Truncate very long text
                    "timestamp": time.time()
                }
            }
            
            await asyncio.to_thread(ami_index.upsert, vectors=[vector], namespace="embeddings_cache")
            _cache_stats["embedding_misses"] += 1
            
            return embedding
    except Exception as e:
        logger.error(f"Error checking embedding cache: {e}")
    
    # Fallback: generate new embedding
    try:
        embedding = await EMBEDDINGS.aembed_query(text)
        
        # Try to store it for future use
        try:
            vector = {
                "id": cache_key,
                "values": embedding,
                "metadata": {
                    "original_text": text[:1000],  # Truncate very long text
                    "timestamp": time.time()
                }
            }
            await asyncio.to_thread(ami_index.upsert, vectors=[vector], namespace="embeddings_cache")
        except Exception as store_err:
            logger.error(f"Error storing embedding in cache: {store_err}")
        
        _cache_stats["embedding_misses"] += 1
        return embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        # Return a zero vector as fallback
        return [0.0] * 1536

def calculate_cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    if len(vec1) != len(vec2):
        return 0.0
    
    # Use numpy for better performance
    import numpy as np
    vec1_np = np.array(vec1)
    vec2_np = np.array(vec2)
    similarity = np.dot(vec1_np, vec2_np) / (np.linalg.norm(vec1_np) * np.linalg.norm(vec2_np))
    return float(similarity)

async def get_version_brain_banks(version_id: str) -> List[Dict[str, str]]:
    """
    Get the bank names for all brains in a version with Pinecone-based caching
    
    Args:
        version_id: UUID of the graph version
    
    Returns:
        List of dicts containing brain_id and bank_name
    """
    global _cache_stats
    
    # Check if we have a cached entry in Pinecone
    cache_key = f"brain_banks_{version_id}"
    
    try:
        # Try to fetch from Pinecone cache
        results = await asyncio.to_thread(
            ami_index.fetch,
            ids=[cache_key],
            namespace="structure_cache"
        )
        
        # Extract vectors safely using the helper function
        vectors = extract_vectors_from_response(results)
        
        if cache_key in vectors:
            # Extract metadata - handling both dict and object formats
            if isinstance(vectors[cache_key], dict) and "metadata" in vectors[cache_key]:
                metadata = vectors[cache_key]["metadata"]
            elif hasattr(vectors[cache_key], "metadata"):
                metadata = vectors[cache_key].metadata
            else:
                logger.warning(f"Cannot extract metadata from cache entry: {type(vectors[cache_key])}")
                metadata = {}
                
            # Deserialize the JSON string to retrieve the original data structure
            brain_banks_json = metadata.get("brain_banks", "[]") if isinstance(metadata, dict) else getattr(metadata, "brain_banks", "[]")
            cached_data = json.loads(brain_banks_json)
            timestamp = metadata.get("timestamp", 0) if isinstance(metadata, dict) else getattr(metadata, "timestamp", 0)
            
            # Check if the cache entry is still valid
            if time.time() - timestamp < _brain_cache_ttl:
                _cache_stats["brain_hits"] += 1
                logger.info(f"Using cached brain structure for version {version_id}, {len(cached_data)} brains")
                return cached_data
    except Exception as e:
        logger.error(f"Error checking brain structure cache: {e}")
    
    # Cache miss or expired, query the database
    _cache_stats["brain_misses"] += 1
    try:
        # First get the brain IDs from the version
        version_response = supabase.table("brain_graph_version")\
            .select("brain_ids", "status")\
            .eq("id", version_id)\
            .execute()
        
        if not version_response.data:
            logger.error(f"Version {version_id} not found")
            return []
            
        version_data = version_response.data[0]
        if version_data["status"] != "published":
            logger.warning(f"Version {version_id} is not published")
            return []
            
        brain_ids = version_data["brain_ids"]
        if not brain_ids:
            logger.warning(f"No brain IDs found for version {version_id}")
            return []
            
        # Get bank names for all brains
        brain_response = supabase.table("brain")\
            .select("id", "brain_id", "bank_name")\
            .in_("id", brain_ids)\
            .execute()
            
        if not brain_response.data:
            logger.warning(f"No brain data found for brain IDs: {brain_ids}")
            return []
        
        brain_banks = [
            {
                "id": brain["id"],
                "brain_id": brain["brain_id"],
                "bank_name": brain["bank_name"]
            }
            for brain in brain_response.data
        ]
        
        # Store in Pinecone cache - skip if errors occur but continue with the data
        try:
            # We're just using Pinecone as a key-value store for this data
            # The vector itself doesn't matter as we're fetching by ID
            placeholder_vector = [0.0001] * 1536  # Small non-zero values to avoid Pinecone error
            
            # Hash the brain_banks data to use as vector elements
            # This ensures vector has some meaningful variation instead of all the same values
            import hashlib
            data_hash = hashlib.md5(str(brain_banks).encode()).hexdigest()
            seed = int(data_hash, 16) % 10000  # Convert hash to int for seeding
            
            # Generate a sparse vector with some variation
            random_vector = [0.0] * 1536
            import random
            random.seed(seed)
            for i in range(50):  # Just put 50 non-zero values
                idx = random.randint(0, 1535)
                random_vector[idx] = random.uniform(0.1, 1.0)
            
            # Serialize complex data structures to JSON strings for Pinecone metadata
            vector = {
                "id": cache_key,
                "values": random_vector,
                "metadata": {
                    "brain_banks": json.dumps(brain_banks),  # Serialize to JSON string
                    "timestamp": time.time()
                }
            }
            
            await asyncio.to_thread(ami_index.upsert, vectors=[vector], namespace="structure_cache")
        except Exception as cache_err:
            logger.error(f"Error caching brain structure: {cache_err}")
            # Just log error, don't prevent returning the data
        
        logger.info(f"Retrieved and cached brain structure for version {version_id}, {len(brain_banks)} brains")
        return brain_banks
    except Exception as e:
        logger.error(f"Error getting brain banks: {e}")
        return []

async def _query_brain_with_embedding(embedding, namespace: str, top_k: int = 10, brain_id: str = None) -> List[Dict]:
    """
    Query a brain using a pre-computed embedding to avoid redundant embedding generation
    
    Args:
        embedding: Pre-computed embedding vector
        namespace: Brain namespace/bank name
        top_k: Maximum number of results to return
        brain_id: ID of the brain being queried
        
    Returns:
        List of knowledge entries from the specified brain
    """
    try:
        knowledge = []
        
        # Ensure embedding has non-zero values
        if not any(embedding):
            logger.warning(f"Zero embedding detected in _query_brain_with_embedding for namespace {namespace}, using minimal values")
            # Generate a minimal non-zero embedding to avoid Pinecone errors - use a stable but varied pattern
            embedding = []
            for i in range(1536):
                # Create a small pattern of non-zero values that varies slightly to avoid all identical values
                embedding.append(0.0001 + (0.00001 * (i % 10)))
        
        # Query both indexes with the same embedding
        for index in [ami_index, ent_index]:
            try:
                # Verify that the index is properly defined
                if not hasattr(index, 'query'):
                    logger.error(f"Invalid index object when querying namespace {namespace}")
                    continue
                    
                # Verify the vector dimension is correct
                if len(embedding) != 1536:
                    logger.error(f"Invalid embedding dimension {len(embedding)}, expected 1536")
                    embedding = [0.0001] * 1536  # Create fallback embedding
                
                # Use to_thread for async execution since Pinecone client is synchronous
                filter_dict = {"categories_special": {"$in": ["", "description", "document", "procedural"]}}
                
                # Add explicit logging to trace the embedding
                logger.info(f"Querying {index_name(index)} with vector of dimension {len(embedding)} and non-zero count {sum(1 for x in embedding if x != 0)}")
                
                results = await asyncio.to_thread(
                    index.query,
                    vector=embedding,
                    top_k=top_k,
                    include_metadata=True,
                    namespace=namespace,
                    filter=filter_dict
                )
                
                matches = results.get("matches", [])
                # Only log match count if non-zero
                if matches:
                    logger.info(f"Found {len(matches)} matches in {index_name(index)} for namespace {namespace}")
                
                for match in matches:
                    try:
                        # Extract metadata safely with fallbacks
                        metadata = match.get("metadata", {})
                        result = {
                            "id": match.get("id", ""),
                            "bank_name": namespace,
                            "brain_id": brain_id,
                            "raw": metadata.get("raw", ""),
                            "categories": {
                                "primary": metadata.get("categories_primary", "unknown"),
                                "special": metadata.get("categories_special", "")
                            },
                            "confidence": metadata.get("confidence", 0.0),
                            "score": match.get("score", 0.0)
                        }
                        knowledge.append(result)
                    except Exception as e:
                        logger.error(f"Error processing match: {e}")
                        continue
            except Exception as e:
                logger.error(f"Knowledge query failed in {index_name(index)} for namespace {namespace}: {e}")
        
        # Skip sorting if only 1 or 0 results
        if len(knowledge) <= 1:
            return knowledge
            
        # Sort results by score
        knowledge.sort(key=lambda x: x["score"], reverse=True)
        
        # Limit to top_k
        if len(knowledge) > top_k:
            knowledge = knowledge[:top_k]
            
        return knowledge
    except Exception as e:
        logger.error(f"Error querying brain with embedding for namespace {namespace}: {e}")
        return []

async def query_graph_knowledge(version_id: str, query: str, top_k: int = 10) -> List[Dict]:
    """
    Query knowledge across all brains in a graph version
    
    Args:
        version_id: UUID of the graph version
        query: The search query
        top_k: Maximum number of results to return per brain
    
    Returns:
        List of knowledge entries from all brains, sorted by relevance
    """
    try:
        # Generate a cache key for this query
        import hashlib
        query_hash = hashlib.md5(f"{version_id}_{query}_{top_k}".encode()).hexdigest()
        cache_key = f"query_result_{query_hash}"
        
        # Check if result is already in cache
        try:
            results = await asyncio.to_thread(
                ami_index.fetch,
                ids=[cache_key],
                namespace="query_cache"
            )
            
            vectors = results.get("vectors", {})
            if cache_key in vectors:
                metadata = vectors[cache_key]["metadata"]
                # Safely deserialize the JSON string to retrieve the original results
                results_json = metadata.get("results", "[]")
                # Ensure we're deserializing a string, not a list
                if isinstance(results_json, str):
                    cached_results = json.loads(results_json)
                else:
                    # If for some reason we already have a Python object, use it directly
                    cached_results = results_json
                timestamp = metadata.get("timestamp", 0)
                
                # Use cached results if less than 60 seconds old
                if time.time() - timestamp < 60:
                    logger.info(f"Using cached query results for '{query}'")
                    return cached_results
        except Exception as cache_err:
            logger.error(f"Error checking query cache: {cache_err}")
        
        # Get brain banks and embedding in parallel
        brain_banks_task = asyncio.create_task(get_version_brain_banks(version_id))
        embedding_task = asyncio.create_task(get_cached_embedding(query))
        
        # Wait for both tasks to complete
        brain_banks, embedding = await asyncio.gather(brain_banks_task, embedding_task)
        
        if not brain_banks:
            logger.warning(f"No brain banks found for version {version_id}")
            return []
        
        # Ensure embedding has non-zero values
        if not any(embedding):
            logger.warning(f"Zero embedding detected for query '{query}', using minimal values")
            # Generate a minimal non-zero embedding to avoid Pinecone errors
            embedding = [0.0001] * len(embedding)
        
        # Use a shared timeout for all brain queries
        timeout = 5.0  # 5 second timeout
        
        # Query all brains in parallel directly with the embedding
        all_results = []
        tasks = []
        
        for brain in brain_banks:
            # Create tasks for querying each brain's namespace
            ns = brain["bank_name"]
            tasks.append(
                asyncio.create_task(
                    _query_brain_with_embedding(embedding, ns, top_k, brain["id"])
                )
            )
        
        # Run all tasks in parallel with timeout
        try:
            # Wait for all tasks to complete within timeout
            brain_results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=timeout)
            
            # Flatten results
            for results in brain_results:
                all_results.extend(results)
        except asyncio.TimeoutError:
            # If timeout occurs, use whatever results we have
            logger.warning(f"Timeout occurred while querying brains for '{query}', using partial results")
            for task in tasks:
                if task.done() and not task.exception():
                    all_results.extend(task.result())
        
        # Sort by score and take top_k
        sorted_results = sorted(
            all_results,
            key=lambda x: x["score"],
            reverse=True
        )[:top_k]
        
        # Cache results for future use - handle errors gracefully
        try:
            # Store results in Pinecone with random sparse vector to avoid all-zeros error
            import random
            random.seed(int(query_hash[:8], 16))  # Use query hash as seed
            sparse_vector = [0.0] * 1536
            for i in range(50):  # Put 50 non-zero values at random positions
                idx = random.randint(0, 1535)
                sparse_vector[idx] = random.uniform(0.1, 1.0)
            
            # Serialize complex data structures to JSON strings
            vector = {
                "id": cache_key,
                "values": sparse_vector,
                "metadata": {
                    "results": json.dumps(sorted_results),  # Serialize to JSON string
                    "timestamp": time.time(),
                    "query": query
                }
            }
            
            await asyncio.to_thread(ami_index.upsert, vectors=[vector], namespace="query_cache")
        except Exception as cache_err:
            logger.error(f"Error caching query results: {cache_err}")
            # Just log error, don't prevent returning results
        
        logger.info(f"Found {len(sorted_results)} results across {len(brain_banks)} brains for version {version_id}")
        return sorted_results
        
    except Exception as e:
        logger.error(f"Error in graph knowledge query: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return []

async def query_brain_with_embeddings_batch(
    query_embeddings: Dict[int, List[float]], 
    namespace: str, 
    brain_id: str, 
    top_k: int = 5, 
    metadata_filter: Optional[Dict] = None
) -> Dict[int, List[Dict]]:
    """
    Batch process multiple query embeddings for a single brain namespace with optimized parallelism.

    Args:
        query_embeddings: Dictionary mapping query indices to their embeddings.
        namespace: Brain namespace/bank name.
        brain_id: ID of the brain being queried.
        top_k: Maximum number of results to return per query.
        metadata_filter: Optional Pinecone metadata filter.

    Returns:
        Dictionary mapping query indices to their results.
    """
    try:
        start_time = time.time()
        results_by_query = {}
        indexes = [ami_index, ent_index]  # Use global ami_index, ent_index

        # Skip irrelevant indexes based on brain_id or namespace
        relevant_indexes = []
        for index in indexes:
            index_name_str = index_name(index)
            # Check if namespace is relevant to index
            if index_name_str in namespace or brain_id in index_name_str:
                relevant_indexes.append(index)
        if not relevant_indexes:
            relevant_indexes = indexes  # Fallback to all indexes

        @tenacity.retry(
            stop=tenacity.stop_after_attempt(2),
            wait=tenacity.wait_exponential(multiplier=0.5, min=0.5, max=2)
        )
        async def query_index(query_idx, embedding, index):
            """Internal function to query an index with retries"""
            # Get index name first for logging
            index_name_str = index_name(index)
            
            # Validate index before continuing
            if not hasattr(index, 'query') or not callable(getattr(index, 'query', None)):
                logger.error(f"Invalid index object for query {query_idx}, missing query method")
                return query_idx, []
        
            try:
                # Ensure embedding has at least one non-zero value
                if not any(embedding) or all(v == 0 for v in embedding):
                    logger.warning(f"Zero embedding detected for query {query_idx}, generating minimal non-zero values")
                    # Create a non-zero vector that will be accepted by Pinecone
                    minimal_value = 0.0001
                    embedding = [minimal_value] * len(embedding)
                    # Ensure first few values are different to avoid uniform vectors
                    if len(embedding) > 0:
                        embedding[0] = minimal_value * 1.1
                    if len(embedding) > 1:
                        embedding[1] = minimal_value * 1.2
                    if len(embedding) > 2:
                        embedding[2] = minimal_value * 0.9
                
                # Validate vector dimension - Pinecone requires consistent dimensions
                if not embedding or len(embedding) not in [768, 1536]:
                    logger.warning(f"Invalid embedding dimension {len(embedding) if embedding else 0}, creating fallback vector of 1536 dimensions")
                    embedding = [0.0001 * (i % 10 + 1) for i in range(1536)]  # Create varied fallback

                # Extra logging to diagnose the issue
                logger.info(f"Querying Pinecone index '{index_name_str}' (namespace: {namespace}) with vector dimension {len(embedding)}")
                
                # Use to_thread for async execution since Pinecone client is synchronous 
                try:
                    results = await asyncio.to_thread(
                        index.query,
                        vector=embedding,
                        top_k=top_k,
                        include_metadata=True,
                        namespace=namespace,
                        filter=metadata_filter or {
                            "categories_special": {"$in": ["", "description", "document", "procedural"]}
                        }
                    )
                except Exception as query_error:
                    logger.error(f"Error in Pinecone query for '{index_name_str}': {str(query_error)}")
                    # Extra retry with a more robust vector if the first attempt failed
                    logger.info(f"Retrying with a different vector for index '{index_name_str}'")
                    embedding = [0.001 * ((i % 20) + 1) for i in range(1536)]
                    results = await asyncio.to_thread(
                        index.query,
                        vector=embedding,
                        top_k=top_k,
                        include_metadata=True, 
                        namespace=namespace,
                        filter=metadata_filter or {
                            "categories_special": {"$in": ["", "description", "document", "procedural"]}
                        }
                    )
                
                matches = results.get("matches", [])
                if matches:
                    logger.info(f"Found {len(matches)} matches in index '{index_name_str}' for query {query_idx}")
                
                query_results = []
                for match in matches:
                    # Extract metadata safely with fallbacks
                    metadata = match.get("metadata", {})
                    
                    # Create a sanitized metadata object for Pinecone compatibility
                    sanitized_metadata = {}
                    for key, value in metadata.items():
                        if isinstance(value, (str, int, float, bool)) or (isinstance(value, list) and all(isinstance(x, str) for x in value)):
                            sanitized_metadata[key] = value
                        else:
                            # For complex data types, we'll keep them as is in the result
                            # as this is not being stored in Pinecone but returned to the caller
                            sanitized_metadata[key] = value
                    
                    result = {
                        "id": match.get("id", ""),
                        "bank_name": namespace,
                        "brain_id": brain_id,
                        "raw": metadata.get("raw", ""),
                        "categories": {
                            "primary": metadata.get("categories_primary", "unknown"),
                            "special": metadata.get("categories_special", "")
                        },
                        "confidence": metadata.get("confidence", 0.0),
                        "score": match.get("score", 0.0),
                        "metadata": sanitized_metadata  # Include processed metadata
                    }
                    query_results.append(result)
                return query_idx, query_results
            except Exception as e:
                logger.error(f"Error querying {index_name_str} for query {query_idx}: {e}")
                return query_idx, []

        # Process all queries and indexes in parallel
        tasks = []
        for query_idx, embedding in query_embeddings.items():
            for index in relevant_indexes:
                tasks.append(query_index(query_idx, embedding, index))

        # Wait for all tasks to complete with timeout
        try:
            timeout = 5.0  # 5 second timeout
            all_results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=timeout)
        except asyncio.TimeoutError:
            # If timeout occurs, use whatever results we have
            logger.warning(f"Timeout occurred during batch query, using partial results")
            all_results = []
            for task in tasks:
                if task.done() and not task.exception():
                    all_results.append(task.result())

        for query_idx, query_results in all_results:
            if query_idx not in results_by_query:
                results_by_query[query_idx] = []
            results_by_query[query_idx].extend(query_results)
            
            # Skip sorting if only 1 or 0 results
            if len(query_results) <= 1:
                continue
                
            # Sort results by score and limit to top_k
            results_by_query[query_idx] = sorted(
                results_by_query[query_idx],
                key=lambda x: x["score"],
                reverse=True
            )[:top_k]

        total_results = sum(len(results) for results in results_by_query.values())
        logger.info(f"Batch query complete for {len(query_embeddings)} embeddings in {time.time() - start_time:.2f}s: {total_results} results")
        return results_by_query
    except Exception as e:
        logger.error(f"Error in batch query processing for namespace {namespace}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {}

# Add the provided node functions for persistent memory

async def store_called_node(node_id: str, node_content: str, metadata: Dict[str, Any], namespace: str = "called_nodes") -> bool:
    """Store a node in Pinecone with vibe_score for relevance."""
    try:
        embedding = await get_cached_embedding(node_content)
        # Check for similar nodes to adjust vibe_score
        similar_nodes = await recall_similar_nodes(node_content, namespace, top_k=1, similarity_threshold=0.95)
        vibe_score = 0.5 if similar_nodes else 1.0  # Lower for duplicates
        
        # Process metadata to ensure it only contains primitive types
        processed_metadata = {
            "raw": node_content,
            "timestamp": time.time(),
            "vibe_score": vibe_score
        }
        
        # Add other metadata, serializing complex objects if needed
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)) or (isinstance(value, list) and all(isinstance(x, str) for x in value)):
                processed_metadata[key] = value
            else:
                # Serialize complex objects to JSON
                processed_metadata[key] = json.dumps(value)
                
        vector = {
            "id": node_id,
            "values": embedding,
            "metadata": processed_metadata
        }
        
        await asyncio.to_thread(ami_index.upsert, vectors=[vector], namespace=namespace)
        _cache_stats["node_stores"] += 1
        logger.info(f"Stored node {node_id} in {namespace} with vibe_score {vibe_score}")
        return True
    except Exception as e:
        logger.error(f"Error storing node {node_id}: {e}")
        return False

async def update_called_node(node_id: str, node_content: str, metadata: Dict[str, Any], namespace: str = "called_nodes") -> bool:
    """Update a node, boosting its vibe_score."""
    try:
        embedding = await get_cached_embedding(node_content)
        # Fetch existing node to get current vibe_score
        results = await asyncio.to_thread(
            ami_index.fetch,
            ids=[node_id],
            namespace=namespace
        )
        
        # Get current vibe_score or use default
        current_vibe_score = 1.0  # Default value
        
        # Extract vectors safely using the helper function
        vectors = extract_vectors_from_response(results)
        
        if node_id in vectors:
            # Extract metadata - handling both dict and object formats
            if isinstance(vectors[node_id], dict) and "metadata" in vectors[node_id]:
                node_metadata = vectors[node_id]["metadata"]
                if isinstance(node_metadata, dict):
                    current_vibe_score = node_metadata.get("vibe_score", 1.0)
                else:
                    current_vibe_score = getattr(node_metadata, "vibe_score", 1.0)
            elif hasattr(vectors[node_id], "metadata"):
                node_metadata = vectors[node_id].metadata
                current_vibe_score = getattr(node_metadata, "vibe_score", 1.0)
            else:
                logger.warning(f"Node {node_id} found but cannot extract metadata, using default vibe_score")
        else:
            logger.warning(f"Node {node_id} not found in fetch call, using default vibe_score")
            
        vibe_score = min(current_vibe_score + 0.2, 2.0)  # Boost, cap at 2.0
        
        # Ensure embedding has non-zero values
        if not any(embedding):
            logger.warning(f"Invalid zero embedding for update of node '{node_id}'")
            embedding = [0.0001] * len(embedding)  # Use small non-zero values
        
        # Process metadata to ensure it only contains primitive types
        processed_metadata = {
            "raw": node_content,
            "timestamp": time.time(),
            "vibe_score": vibe_score
        }
        
        # Add other metadata, serializing complex objects if needed
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)) or (isinstance(value, list) and all(isinstance(x, str) for x in value)):
                processed_metadata[key] = value
            else:
                # Serialize complex objects to JSON
                processed_metadata[key] = json.dumps(value)
            
        vector = {
            "id": node_id,
            "values": embedding,
            "metadata": processed_metadata
        }
        
        await asyncio.to_thread(ami_index.upsert, vectors=[vector], namespace=namespace)
        _cache_stats["node_updates"] += 1
        logger.info(f"Updated node {node_id} in {namespace} with vibe_score {vibe_score}")
        return True
    except Exception as e:
        logger.error(f"Error updating node {node_id}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

async def recall_similar_nodes(input_text: str, namespace: str = "called_nodes", top_k: int = 5, similarity_threshold: float = 0.7) -> List[Dict]:
    """Recall nodes, prioritizing high vibe_score."""
    try:
        embedding = await get_cached_embedding(input_text)
        
        # Ensure we have a valid embedding with non-zero values
        if not any(embedding):
            logger.warning(f"Invalid zero embedding generated for input '{input_text[:30]}...'")
            # Use a small non-zero value to avoid Pinecone errors
            embedding = [0.0001] * len(embedding)
        
        results = await asyncio.to_thread(
            ami_index.query,
            vector=embedding,
            top_k=top_k,
            include_metadata=True,
            namespace=namespace
        )
        
        nodes = []
        
        # Handle both old and new Pinecone API response formats
        matches = []
        if hasattr(results, 'matches'):
            matches = results.matches
        elif hasattr(results, 'get'):
            matches = results.get("matches", [])
        
        for m in matches:
            # Extract score (handle both dict and object formats)
            score = m.get("score", 0) if isinstance(m, dict) else getattr(m, "score", 0)
            
            if score >= similarity_threshold:
                # Extract id (handle both dict and object formats)
                node_id = m.get("id", "") if isinstance(m, dict) else getattr(m, "id", "")
                
                # Extract metadata (handle both dict and object formats)
                if isinstance(m, dict) and "metadata" in m:
                    metadata = m["metadata"]
                elif hasattr(m, "metadata"):
                    metadata = m.metadata
                    # Convert to dict if it's not already
                    if not isinstance(metadata, dict):
                        metadata = {k: getattr(metadata, k) for k in dir(metadata) 
                                    if not k.startswith('_') and not callable(getattr(metadata, k))}
                else:
                    logger.warning(f"Cannot extract metadata from match: {type(m)}")
                    continue
                
                processed_metadata = {}
                
                # Copy primitive values directly
                for key, value in metadata.items():
                    if key != "raw" and key != "vibe_score" and key != "timestamp":
                        # Try to parse JSON for potential serialized objects
                        try:
                            if isinstance(value, str) and (value.startswith('{') or value.startswith('[')):
                                processed_metadata[key] = json.loads(value)
                            else:
                                processed_metadata[key] = value
                        except (json.JSONDecodeError, TypeError):
                            # If it's not valid JSON, keep the original value
                            processed_metadata[key] = value
                    else:
                        processed_metadata[key] = value
                
                # Get raw content
                raw_content = metadata.get("raw", "") if isinstance(metadata, dict) else getattr(metadata, "raw", "")
                
                # Get vibe_score with fallback
                vibe_score = metadata.get("vibe_score", 1.0) if isinstance(metadata, dict) else getattr(metadata, "vibe_score", 1.0)
                
                node = {
                    "id": node_id,
                    "raw": raw_content,
                    "metadata": processed_metadata,
                    "score": score,
                    "vibe_score": vibe_score
                }
                nodes.append(node)
                
        # Sort by vibe_score (primary) and score (secondary)
        nodes.sort(key=lambda x: (x["vibe_score"], x["score"]), reverse=True)
        _cache_stats["node_hits"] += len(nodes)
        _cache_stats["node_misses"] += 1 if not nodes else 0
        logger.info(f"Recalled {len(nodes)} nodes for '{input_text[:30]}...'")
        return nodes
    except Exception as e:
        logger.error(f"Error recalling nodes: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return []

async def recall_similar_nodes_batch(
    inputs: Dict[int, str], namespace: str = "called_nodes", top_k: int = 5, similarity_threshold: float = 0.7
) -> Dict[int, List[Dict]]:
    """Batch recall nodes with vibe_score prioritization."""
    try:
        embedding_tasks = {idx: asyncio.create_task(get_cached_embedding(text)) for idx, text in inputs.items()}
        embeddings = {idx: await task for idx, task in embedding_tasks.items()}
        results = await query_brain_with_embeddings_batch(
            query_embeddings=embeddings,
            namespace=namespace,
            brain_id="nodes",
            top_k=top_k
        )
        
        filtered_results = {}
        for idx, nodes in results.items():
            processed_nodes = []
            for node in nodes:
                if node["score"] >= similarity_threshold:
                    # Process metadata to handle serialized JSON fields
                    metadata = node["metadata"]
                    processed_metadata = {}
                    
                    # Copy primitive values directly
                    for key, value in metadata.items():
                        if key != "raw" and key != "vibe_score" and key != "timestamp":
                            # Try to parse JSON for potential serialized objects
                            try:
                                if isinstance(value, str) and (value.startswith('{') or value.startswith('[')):
                                    processed_metadata[key] = json.loads(value)
                                else:
                                    processed_metadata[key] = value
                            except (json.JSONDecodeError, TypeError):
                                # If it's not valid JSON, keep the original value
                                processed_metadata[key] = value
                        else:
                            processed_metadata[key] = value
                    
                    # Update the node with processed metadata
                    node["metadata"] = processed_metadata
                    processed_nodes.append(node)
            
            # Sort the nodes by vibe_score and score
            sorted_nodes = sorted(
                processed_nodes,
                key=lambda x: (x["metadata"].get("vibe_score", 1.0), x["score"]), 
                reverse=True
            )
            filtered_results[idx] = sorted_nodes
            
        total_hits = sum(len(nodes) for nodes in filtered_results.values())
        _cache_stats["node_hits"] += total_hits
        _cache_stats["node_misses"] += sum(1 for nodes in filtered_results.values() if not nodes)
        logger.info(f"Batch recalled {total_hits} nodes for {len(inputs)} inputs")
        return filtered_results
    except Exception as e:
        logger.error(f"Error in batch node recall: {e}")
        return {}

async def cleanup_old_nodes(namespace: str = "called_nodes", max_age_seconds: float = 604800, min_vibe_score: float = 0.1, node_ids: Optional[List[str]] = None) -> None:
    """Delete low-vibe or old nodes from Pinecone."""
    try:
        if node_ids:
            await asyncio.to_thread(ami_index.delete, ids=node_ids, namespace=namespace)
            logger.info(f"Deleted {len(node_ids)} specific nodes from {namespace}")
            return
        
        cutoff = time.time() - max_age_seconds
        
        # Use a valid filter format that Pinecone can process
        # Pinecone filter doesn't support $or directly for basic types
        # So we'll make separate queries for each condition and combine the results
        
        # First query: for old nodes
        try:
            time_results = await asyncio.to_thread(
                ami_index.query,
                vector=[0.0001] * 1536,  # Use non-zero minimal vector
                top_k=1000,
                namespace=namespace,
                filter={"timestamp": {"$lt": cutoff}}
            )
            time_matches = time_results.get("matches", [])
        except Exception as e:
            logger.error(f"Error querying old nodes: {e}")
            time_matches = []
            
        # Second query: for low vibe score nodes
        try:
            vibe_results = await asyncio.to_thread(
                ami_index.query,
                vector=[0.0001] * 1536,  # Use non-zero minimal vector
                top_k=1000,
                namespace=namespace,
                filter={"vibe_score": {"$lt": min_vibe_score}}
            )
            vibe_matches = vibe_results.get("matches", [])
        except Exception as e:
            logger.error(f"Error querying low vibe nodes: {e}")
            vibe_matches = []
        
        # Combine the IDs from both queries
        time_ids = [m["id"] for m in time_matches]
        vibe_ids = [m["id"] for m in vibe_matches]
        all_ids = list(set(time_ids + vibe_ids))  # Remove duplicates
        
        if all_ids:
            # Delete nodes in batches to avoid overwhelming the API
            batch_size = 100
            for i in range(0, len(all_ids), batch_size):
                batch_ids = all_ids[i:i+batch_size]
                await asyncio.to_thread(ami_index.delete, ids=batch_ids, namespace=namespace)
            
            logger.info(f"Deleted {len(all_ids)} low-vibe/old nodes from {namespace}")
    except Exception as e:
        logger.error(f"Error cleaning up nodes: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    """
    When run directly, just import the module. Tests are now in test_hb.py
    """
    print("hotbrain.py module loaded. Run test_hb.py for testing.")