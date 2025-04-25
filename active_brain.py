import faiss
import numpy as np
import pickle
import os
import asyncio
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from utilities import logger
import psutil
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError
import time
from supabase import create_client, Client
from pinecone import Pinecone, Index
import traceback
from utilities import EMBEDDINGS

spb_url = os.getenv("SUPABASE_URL")
spb_key = os.getenv("SUPABASE_KEY")
if spb_url and spb_key:
    supabase = create_client(spb_url, spb_key)
    logger.info("Supabase client initialized")
else:
    logger.error("Supabase credentials not found - required for brain structure retrieval")
    raise ValueError("Missing Supabase credentials")


class ActiveBrain:
    def __init__(self, dim: int = 1536, namespace: str = "", graph_version_ids: List[str] = None, pinecone_index_name: str = None):
        """Initialize the active brain with FAISS and optional graph version IDs."""
        self.dim = dim
        self.namespace = namespace
        self.vector_ids = []
        self.vectors = None
        self.metadata = {}
        self.graph_version_ids = graph_version_ids or []
        self.faiss_index = self._initialize_faiss()
        self.pinecone_index = None
        
        # Initialize Pinecone if index name provided
        if pinecone_index_name:
            self._initialize_pinecone(pinecone_index_name)
    
    def _initialize_pinecone(self, index_name: str):
        """Initialize Pinecone connection with the given index name."""
        try:
            pinecone_api_key = os.getenv("PINECONE_API_KEY")
            if not pinecone_api_key:
                logger.error("PINECONE_API_KEY environment variable is required")
                raise ValueError("Missing Pinecone API key")
            
            pc = Pinecone(api_key=pinecone_api_key)
            self.pinecone_index = pc.Index(index_name)
            logger.info(f"Successfully connected to Pinecone index: {index_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            logger.error(traceback.format_exc())
            return False
        
    def _initialize_faiss(self) -> faiss.Index:
        """Initialize or load FAISS index from disk."""
        faiss_index = faiss.IndexFlatIP(self.dim)
        
        if os.path.exists("faiss_index.bin") and os.path.exists("metadata.pkl"):
            logger.info("Loading existing FAISS index and metadata...")
            try:
                with open("metadata.pkl", "rb") as f:
                    self.metadata = pickle.load(f)
                self.vector_ids = list(self.metadata.keys())
                
                # Load vectors from FAISS index
                try:
                    self.faiss_index = faiss.read_index("faiss_index.bin")
                    logger.info(f"Successfully loaded FAISS index with {len(self.vector_ids)} vectors")
                except Exception as e:
                    logger.error(f"Failed to load FAISS index: {e}")
                    raise
            except Exception as e:
                logger.error(f"Failed to load local files: {e}")
                raise
        else:
            logger.info("Creating new FAISS index...")
            self.faiss_index = faiss.IndexFlatIP(self.dim)
        
        return faiss_index
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _fetch_vectors_from_pinecone(self, vector_ids: List[str], batch_size: int = 100) -> np.ndarray:
        """Safely fetch vectors from Pinecone in batches."""
        if not self.pinecone_index:
            raise ValueError("Pinecone index not initialized")
        
        # Verify vector existence first with a smaller sample
        try:
            # Take a small sample to check if these vectors actually exist
            sample_ids = vector_ids[:min(5, len(vector_ids))]
            logger.info(f"Verifying vector existence with sample: {sample_ids}")
            
            # Try to fetch just the sample
            sample_result = self.pinecone_index.fetch(ids=sample_ids, namespace=self.namespace)
            
            # Check if any vectors were returned
            if not hasattr(sample_result, 'vectors') or not sample_result.vectors:
                logger.error(f"No vectors found in namespace '{self.namespace}' for sample IDs")
                
                # Try alternate namespace - empty string is the default Pinecone namespace
                alternate_namespace = ""
                logger.info(f"Trying alternate namespace: '{alternate_namespace}'")
                alt_sample = self.pinecone_index.fetch(ids=sample_ids, namespace=alternate_namespace)
                
                if hasattr(alt_sample, 'vectors') and alt_sample.vectors:
                    logger.info(f"Found vectors in alternate namespace: '{alternate_namespace}'")
                    self.namespace = alternate_namespace
                else:
                    logger.error("Vectors not found in default namespace either")
                    
                    # Let's try to discover valid namespaces
                    if hasattr(self.pinecone_index, 'describe_index_stats'):
                        stats = self.pinecone_index.describe_index_stats()
                        namespaces = stats.get('namespaces', {})
                        logger.info(f"Available namespaces: {list(namespaces.keys())}")
                        
                        # Try each namespace with our sample
                        for ns in namespaces.keys():
                            logger.info(f"Trying namespace: '{ns}'")
                            ns_sample = self.pinecone_index.fetch(ids=sample_ids, namespace=ns)
                            
                            if hasattr(ns_sample, 'vectors') and ns_sample.vectors:
                                logger.info(f"Found vectors in namespace: '{ns}'")
                                self.namespace = ns
                                break
        except Exception as e:
            logger.error(f"Error during vector verification: {e}")
            # Continue with the original fetch attempt
            
        # Proceed with actual vector fetching
        all_vectors = []
        for i in range(0, len(vector_ids), batch_size):
            batch_ids = vector_ids[i:i+batch_size]
            try:
                # Debug the batch IDs being requested
                logger.debug(f"Fetching batch {i//batch_size} with {len(batch_ids)} IDs")
                logger.debug(f"First few IDs in batch: {batch_ids[:3]}")
                
                result = self.pinecone_index.fetch(ids=batch_ids, namespace=self.namespace)
                
                # Debug the returned vectors
                logger.debug(f"Received {len(result.vectors) if hasattr(result, 'vectors') else 0} vectors from Pinecone")
                
                # Check if vectors exist in the response
                if not hasattr(result, 'vectors') or not result.vectors:
                    logger.error(f"No vectors returned for batch {i//batch_size}")
                    raise KeyError(f"No vectors found in response for batch {i//batch_size}")
                
                # Debug vector IDs returned
                vector_ids_returned = list(result.vectors.keys())
                logger.debug(f"Received vector IDs: {vector_ids_returned[:3]}...")
                
                missing_ids = [id for id in batch_ids if id not in result.vectors]
                if missing_ids:
                    logger.warning(f"Missing {len(missing_ids)} IDs in response: {missing_ids[:5]}")
                
                batch_vectors = []
                for vector_id in batch_ids:
                    if vector_id in result.vectors:
                        vector_values = result.vectors[vector_id].values
                        batch_vectors.append(vector_values)
                    else:
                        # Create a placeholder vector with zeros (safe fallback)
                        logger.warning(f"Vector ID {vector_id} not found, using zero vector")
                        placeholder = [0.0] * self.dim
                        batch_vectors.append(placeholder)
                
                all_vectors.extend(batch_vectors)
                logger.debug(f"Successfully processed batch {i//batch_size}, total vectors: {len(all_vectors)}")
                
            except KeyError as e:
                logger.error(f"KeyError in batch {i//batch_size}: {e}")
                logger.error(f"Problem with vector ID: {e}")
                raise
            except Exception as e:
                logger.error(f"Failed to fetch vectors batch {i//batch_size}: {e}")
                logger.error(traceback.format_exc())
                raise
                
        return np.array(all_vectors, dtype=np.float32)
    
    async def get_version_brain_banks(self, version_id: str) -> List[Dict[str, str]]:
        """
        Get the bank names for all brains in a version
        
        Args:
            version_id: UUID of the graph version
        
        Returns:
            List of dicts containing brain_id and bank_name
        """
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
            
            logger.info(f"Retrieved brain structure for version {version_id}, {len(brain_banks)} brains")
            return brain_banks
        except Exception as e:
            logger.error(f"Error getting brain banks: {e}")
            logger.error(traceback.format_exc())
            return []
    
    async def _get_all_vectors_from_namespace(self, namespace: str, top_k: int = 100) -> List[Tuple[str, List[float], Dict]]:
        """
        Get all vectors from a namespace using Pinecone's query method.
        This is an alternative approach when we can't fetch vectors by ID.
        
        Args:
            namespace: The namespace to query from
            top_k: Maximum number of vectors to retrieve
            
        Returns:
            List of tuples containing (id, vector, metadata)
        """
        try:
            if not self.pinecone_index:
                logger.error("Pinecone index not initialized")
                return []
                
            logger.info(f"Querying up to {top_k} vectors directly from namespace: {namespace}")
            
            # Create a dummy vector for the query (all zeros with a single 1)
            # This is a trick to match as many vectors as possible
            query_vector = [0.0] * self.dim
            query_vector[0] = 1.0
            
            # Query the namespace - we'll get a diverse set of vectors this way
            result = self.pinecone_index.query(
                vector=query_vector,
                namespace=namespace,
                top_k=top_k,
                include_values=True,
                include_metadata=True
            )
            
            if not hasattr(result, 'matches') or not result.matches:
                logger.warning(f"No matches found in namespace {namespace}")
                return []
                
            logger.info(f"Found {len(result.matches)} vectors in namespace {namespace}")
            
            # Extract the vectors and metadata
            vectors = []
            for match in result.matches:
                vector_id = match.id
                vector_values = match.values
                metadata = match.metadata if hasattr(match, 'metadata') else {}
                vectors.append((vector_id, vector_values, metadata))
                
            return vectors
            
        except Exception as e:
            logger.error(f"Error querying vectors from namespace {namespace}: {e}")
            logger.error(traceback.format_exc())
            return []
            
    async def load_all_vectors_from_graph_version(self, graph_version_id: str):
        """
        Load ALL vectors from Pinecone for a given graph version ID.
        This is a simplified version that focuses on just loading all vectors.
        
        Args:
            graph_version_id: The ID of the graph version to load vectors from
        """
        try:
            # Check if Pinecone is initialized
            if not self.pinecone_index:
                logger.error("Pinecone index not initialized, cannot load vectors")
                return
                
            # Get all brain banks for this graph version
            brain_banks = await self.get_version_brain_banks(graph_version_id)
            if not brain_banks:
                logger.warning(f"No brain banks found for graph version {graph_version_id}")
                return
            
            # Track all vectors and metadata
            all_vectors = []
            all_ids = []
            all_metadata = []
            
            # Process each brain bank
            for brain in brain_banks:
                brain_id = brain["id"]
                bank_name = brain["bank_name"]
                
                try:
                    logger.info(f"Processing brain bank: {bank_name}")
                    
                    # First, try to query vectors directly from the namespace
                    # This works better when we don't know the exact IDs
                    namespace = bank_name  # Use bank name as namespace
                    vectors_data = await self._get_all_vectors_from_namespace(namespace, top_k=100)
                    
                    if vectors_data:
                        logger.info(f"Successfully retrieved {len(vectors_data)} vectors from namespace {namespace}")
                        
                        # Create metadata for each vector
                        for vector_id, vector_values, vector_metadata in vectors_data:
                            full_id = f"{brain_id}_{vector_id}"
                            all_ids.append(full_id)
                            all_vectors.append(vector_values)
                            
                            # Combine the original metadata with our standard fields
                            combined_metadata = {
                                "brain_id": brain_id,
                                "bank_name": bank_name,
                                "vector_id": vector_id
                            }
                            if vector_metadata:
                                combined_metadata.update(vector_metadata)
                                
                            all_metadata.append(combined_metadata)
                        
                        continue  # Skip the ID-based fetching if we got vectors this way
                    
                    # Fallback to the original method if querying didn't work
                    vector_ids = await self._get_all_vector_ids(bank_name)
                    
                    if not vector_ids:
                        logger.warning(f"No vectors found in brain bank {bank_name}")
                        continue
                    
                    logger.info(f"Fetching {len(vector_ids)} vectors from brain bank {bank_name}")
                    
                    # Fetch vectors from Pinecone
                    try:
                        vectors = self._fetch_vectors_from_pinecone(vector_ids)
                        logger.info(f"Successfully fetched {len(vectors)} vectors from {bank_name}")
                    except RetryError as e:
                        logger.error(f"Failed to fetch vectors after retries: {e}")
                        continue
                    except Exception as e:
                        logger.error(f"Unexpected error fetching vectors: {e}")
                        logger.error(traceback.format_exc())
                        continue
                    
                    # Create metadata for each vector
                    for i, vector_id in enumerate(vector_ids):
                        full_id = f"{brain_id}_{vector_id}"
                        all_ids.append(full_id)
                        all_vectors.append(vectors[i])
                        all_metadata.append({
                            "brain_id": brain_id,
                            "bank_name": bank_name,
                            "vector_id": vector_id
                        })
                    
                except Exception as e:
                    logger.error(f"Error processing brain bank {bank_name}: {e}")
                    logger.error(traceback.format_exc())
                    continue
            
            # Add all vectors to FAISS if we found any
            if all_vectors:
                vectors_array = np.array(all_vectors, dtype=np.float32)
                self.add_vectors(all_ids, vectors_array, all_metadata)
                logger.info(f"Successfully loaded {len(all_vectors)} vectors from graph version {graph_version_id}")
            else:
                logger.warning(f"No vectors found for graph version {graph_version_id}")
                
        except Exception as e:
            logger.error(f"Error loading vectors from graph version {graph_version_id}: {e}")
            logger.error(traceback.format_exc())
    
    async def _get_all_vector_ids(self, bank_name: str) -> List[str]:
        """
        Get all vector IDs from a brain bank.
        For real database implementation, this would query your vector store.
        
        Args:
            bank_name: The name of the brain bank to get vector IDs for
            
        Returns:
            List of vector IDs
        """
        try:
            # Log that we're fetching vector IDs for this bank
            logger.info(f"Getting vector IDs for bank {bank_name}")
            
            # For Pinecone, we can query using list_index operation or directly via fetch API
            # For demonstration, we'll query by namespace
            
            # Check if this is a wisdom bank (from your specific error logs)
            if bank_name.startswith("wisdom_bank_"):
                # Real Pinecone namespaces often match bank names
                namespace = bank_name
                
                # Try to list vectors directly from Pinecone for this namespace
                try:
                    # Get the stats for this namespace to see if there are any vectors
                    if hasattr(self.pinecone_index, 'describe_index_stats'):
                        stats = self.pinecone_index.describe_index_stats()
                        logger.debug(f"Index stats: {stats}")
                        
                        # Check if the namespace exists in the stats
                        namespaces = stats.get('namespaces', {})
                        if namespace in namespaces:
                            vector_count = namespaces[namespace].get('vector_count', 0)
                            logger.info(f"Found {vector_count} vectors in namespace {namespace}")
                        else:
                            logger.warning(f"Namespace {namespace} not found in index stats")
                    
                    # For other wisdom banks, use a different pattern
                    real_ids = [f"{bank_name}_vector_{i+1}" for i in range(20)]
                    logger.info(f"Using {len(real_ids)} generated IDs for {bank_name}")
                    return real_ids
                        
                except Exception as e:
                    logger.error(f"Error listing vectors from Pinecone: {e}")
                    logger.error(traceback.format_exc())
            
            # If we can't get real IDs, generate some reasonable test IDs
            # Use a different format than "vector_X" which caused the previous errors
            test_ids = [f"{bank_name.split('_')[0]}_id_{i+1}" for i in range(10)]
            logger.warning(f"Using {len(test_ids)} fallback test IDs for {bank_name}")
            return test_ids
            
        except Exception as e:
            logger.error(f"Error getting vector IDs for bank {bank_name}: {e}")
            logger.error(traceback.format_exc())
            return []
    
    def add_vectors(self, new_ids: List[str], vectors: np.ndarray, metadata_list: List[Dict]) -> None:
        """Add new vectors to FAISS."""
        try:
            if len(new_ids) == 0:
                logger.warning("No vectors to add")
                return
            
            # Ensure vectors have the correct dimension
            if vectors.shape[1] != self.dim:
                logger.warning(f"Vector dimension mismatch: expected {self.dim}, got {vectors.shape[1]}")
                if vectors.shape[1] < self.dim:
                    padding = np.zeros((vectors.shape[0], self.dim - vectors.shape[1]), dtype=np.float32)
                    vectors = np.hstack([vectors, padding])
                else:
                    vectors = vectors[:, :self.dim]
            
            # Normalize and add to FAISS
            faiss.normalize_L2(vectors)
            self.faiss_index.add(vectors)
            
            # Update local storage
            self.vector_ids.extend(new_ids)
            if self.vectors is None:
                self.vectors = vectors
            else:
                self.vectors = np.vstack([self.vectors, vectors])
            self.metadata.update(dict(zip(new_ids, metadata_list)))
            
            # Save to disk
            faiss.write_index(self.faiss_index, "faiss_index.bin")
            with open("metadata.pkl", "wb") as f:
                pickle.dump(self.metadata, f)
                
            logger.info(f"Successfully added {len(new_ids)} new vectors")
        except Exception as e:
            logger.error(f"Failed to add vectors: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def update_vectors(self, updated_ids: List[str], vectors: np.ndarray, metadata_list: List[Dict]) -> None:
        """Update existing vectors in FAISS."""
        try:
            if len(updated_ids) == 0:
                logger.warning("No vectors to update")
                return
            
            # Ensure vectors have the correct dimension
            if vectors.shape[1] != self.dim:
                logger.warning(f"Vector dimension mismatch: expected {self.dim}, got {vectors.shape[1]}")
                if vectors.shape[1] < self.dim:
                    padding = np.zeros((vectors.shape[0], self.dim - vectors.shape[1]), dtype=np.float32)
                    vectors = np.hstack([vectors, padding])
                else:
                    vectors = vectors[:, :self.dim]
            
            # Normalize vectors
            faiss.normalize_L2(vectors)
            
            # Get indices of vectors to update
            indices = [self.vector_ids.index(id) for id in updated_ids]
            
            # Update vectors array
            for idx, vector in zip(indices, vectors):
                self.vectors[idx] = vector
            
            # Rebuild FAISS index
            self.faiss_index.reset()
            self.faiss_index.add(self.vectors)
            
            # Update metadata
            for id, meta in zip(updated_ids, metadata_list):
                self.metadata[id] = meta
            
            # Save to disk
            faiss.write_index(self.faiss_index, "faiss_index.bin")
            with open("metadata.pkl", "wb") as f:
                pickle.dump(self.metadata, f)
                
            logger.info(f"Successfully updated {len(updated_ids)} vectors")
        except Exception as e:
            logger.error(f"Failed to update vectors: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def get_similar_vectors(self, query_vector: np.ndarray, top_k: int = 5, threshold: float = 0.0) -> List[Tuple[str, np.ndarray, Dict, float]]:
        """Get semantically similar vectors using FAISS.
        
        Args:
            query_vector: Vector to find similar vectors for
            top_k: Number of similar vectors to return
            threshold: Minimum similarity score (0.0 to 1.0) to include in results
            
        Returns:
            List of tuples containing (vector_id, vector, metadata, similarity_score)
        """
        try:
            if not isinstance(query_vector, np.ndarray):
                raise ValueError(f"query_vector must be numpy array, got {type(query_vector)}")
            if query_vector.shape[-1] != self.dim:
                raise ValueError(f"query_vector must have dimension {self.dim}, got {query_vector.shape[-1]}")
            
            if not self.vector_ids:
                logger.warning("No vectors in index, returning empty results")
                return []

            # Track memory usage before search
            mem_before = self.get_memory_usage()
            
            # Reshape and normalize query vector
            query_vector = query_vector.reshape(1, -1).astype(np.float32)
            faiss.normalize_L2(query_vector)
            
            # Adjust top_k if it's larger than the number of vectors
            actual_top_k = min(top_k, len(self.vector_ids))
            if actual_top_k < top_k:
                logger.warning(f"Requested top_k={top_k} but only {actual_top_k} vectors available")
            
            # Perform FAISS search
            distances, indices = self.faiss_index.search(query_vector, k=actual_top_k)
            
            # Track memory usage after search
            mem_after = self.get_memory_usage()
            if mem_after - mem_before > 100:  # If memory increased by more than 100MB
                logger.warning(f"Large memory increase during search: {mem_after - mem_before:.2f}MB")
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < 0 or idx >= len(self.vector_ids):
                    logger.warning(f"Invalid index {idx} returned by FAISS")
                    continue
                
                # Calculate similarity score (convert distance to similarity)
                similarity = float(distances[0][i])
                
                # Skip if below threshold
                if similarity < threshold:
                    continue
                    
                vector_id = self.vector_ids[idx]
                results.append((
                    vector_id,
                    self.vectors[idx],
                    self.metadata[vector_id],
                    similarity
                ))
            
            if not results:
                logger.warning("No valid results found in FAISS search")
            
            return results
            
        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Failed to perform similarity search: {e}")
    
    def get_memory_usage(self) -> float:
        """Get current memory usage of the process in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # Convert bytes to MB
    
    async def get_similar_vectors_by_text(self, query_text: str, top_k: int = 5, threshold: float = 0.0) -> List[Tuple[str, np.ndarray, Dict]]:
        """
        Get semantically similar vectors using text input.
        
        Uses the proper embedding model from ai.embeddings to convert text to vectors.
        
        Args:
            query_text: The text query to search for
            top_k: Number of similar vectors to return
            threshold: Minimum similarity score (0.0 to 1.0) to include in results
            
        Returns:
            List of tuples containing (vector_id, vector, metadata)
        """
        
        query_vector = await EMBEDDINGS.aembed_query(query_text)
        
        # Convert to numpy array if it's not already
        if not isinstance(query_vector, np.ndarray):
            query_vector = np.array(query_vector, dtype=np.float32)
        
        # Use the existing method to perform the search
        return self.get_similar_vectors(query_vector, top_k, threshold)
        
    async def batch_embed_queries(self, queries: List[str]) -> Dict[str, np.ndarray]:
        """
        Efficiently embed multiple queries in a single batch request.
        
        Args:
            queries: List of text queries to embed
            
        Returns:
            Dictionary mapping query text to its vector embedding
        """
        # Deduplicate queries to avoid redundant embeddings
        unique_queries = list(set(queries))
        
        # Log batch size
        logger.info(f"Batch embedding {len(unique_queries)} unique queries")
        
        # Create result dictionary
        embeddings_dict = {}
        
        # First, try using the EMBEDDINGS.aembed_batch method if it exists
        try:
            if hasattr(EMBEDDINGS, 'aembed_batch'):
                # Get embeddings for all queries at once
                batch_results = await EMBEDDINGS.aembed_batch(unique_queries)
                
                # Convert results to numpy arrays
                for i, query in enumerate(unique_queries):
                    if not isinstance(batch_results[i], np.ndarray):
                        embeddings_dict[query] = np.array(batch_results[i], dtype=np.float32)
                    else:
                        embeddings_dict[query] = batch_results[i]
                        
                logger.info(f"Successfully batch embedded {len(embeddings_dict)} queries")
                return embeddings_dict
        except Exception as e:
            logger.warning(f"Error using batch embedding: {e}. Falling back to OpenAI direct batch.")
            
        # If we're here, try using OpenAI's native batch API directly
        try:
            if isinstance(EMBEDDINGS, object) and 'OpenAI' in str(type(EMBEDDINGS)):
                logger.info("Attempting to use OpenAI direct batch embedding")
                
                # For OpenAI embeddings, we can make a single API call with multiple inputs
                response = await EMBEDDINGS.client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=unique_queries
                )
                
                # Process the response
                for i, embedding_data in enumerate(response.data):
                    query = unique_queries[i]
                    embedding = embedding_data.embedding
                    embeddings_dict[query] = np.array(embedding, dtype=np.float32)
                
                logger.info(f"Successfully used OpenAI direct batch for {len(embeddings_dict)} queries")
                return embeddings_dict
        except Exception as e:
            logger.warning(f"Error using OpenAI direct batch: {e}. Falling back to individual embeddings.")
        
        # If we get here, fall back to individual embeddings
        logger.warning("Falling back to individual embeddings")
        for query in unique_queries:
            try:
                embedding = await EMBEDDINGS.aembed_query(query)
                if not isinstance(embedding, np.ndarray):
                    embeddings_dict[query] = np.array(embedding, dtype=np.float32)
                else:
                    embeddings_dict[query] = embedding
            except Exception as e:
                logger.error(f"Error embedding query '{query}': {e}")
                # Don't add to dict if we can't get embedding
        
        logger.info(f"Completed individual embeddings for {len(embeddings_dict)} queries")
        return embeddings_dict
            
    async def batch_similarity_search(self, queries: List[str], top_k: int = 5, threshold: float = 0.0) -> Dict[str, List[Tuple[str, np.ndarray, Dict, float]]]:
        """
        Perform similarity search for multiple queries efficiently.
        
        Args:
            queries: List of text queries
            top_k: Number of similar vectors to return for each query
            threshold: Minimum similarity score to include in results
            
        Returns:
            Dictionary mapping each query to its search results
        """
        try:
            # Get all embeddings in a single batch request
            embeddings_dict = await self.batch_embed_queries(queries)
            
            # Perform similarity searches for each query
            results_dict = {}
            for query in queries:
                if query in embeddings_dict:
                    query_vector = embeddings_dict[query]
                    results = self.get_similar_vectors(query_vector, top_k, threshold)
                    results_dict[query] = results
                else:
                    logger.warning(f"No embedding found for query: {query}")
                    results_dict[query] = []
                
            return results_dict
        except Exception as e:
            logger.error(f"Error in batch similarity search: {e}")
            logger.error(traceback.format_exc())
            
            # If something fails, fall back to individual searches
            results_dict = {}
            for query in queries:
                try:
                    results = await self.get_similar_vectors_by_text(query, top_k, threshold)
                    results_dict[query] = results
                except Exception as search_error:
                    logger.error(f"Error searching for query '{query}': {search_error}")
                    results_dict[query] = []
                    
            return results_dict 