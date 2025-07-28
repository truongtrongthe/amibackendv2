import faiss
from pinecone import Pinecone
import numpy as np
import pickle
import os
from tenacity import retry, stop_after_attempt, wait_exponential
import psutil
from typing import List, Dict, Tuple, Any, Optional
from datetime import datetime
import uuid
from utilities import logger
import time

class VectorManager:
    def __init__(self, dim: int = 1536, namespace: str = ""):
        """Initialize the vector manager with Pinecone and FAISS."""
        self.dim = dim
        self.namespace = namespace
        self.vector_ids = []
        self.vectors = None
        self.metadata = {}
        
        # Initialize Pinecone
        pinecone_api_key = os.environ.get("PINECONE_API_KEY")
        if not pinecone_api_key:
            raise ValueError("PINECONE_API_KEY environment variable not set")
            
        try:
            pc = Pinecone(api_key=pinecone_api_key)
            self.pinecone_index = pc.Index("dev")
            logger.info(f"Successfully initialized Pinecone index: dev")
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone index: {e}")
            raise
        
        # Initialize or load FAISS index
        self.faiss_index = self._initialize_faiss()
        
    def _initialize_faiss(self) -> faiss.Index:
        """Initialize or load FAISS index from disk."""
        faiss_index = faiss.IndexFlatIP(self.dim)
        
        if os.path.exists("faiss_index.bin") and os.path.exists("metadata.pkl"):
            logger.info("Loading existing FAISS index and metadata...")
            try:
                with open("metadata.pkl", "rb") as f:
                    self.metadata = pickle.load(f)
                self.vector_ids = list(self.metadata.keys())
                
                # Try to fetch vectors from Pinecone
                try:
                    self.vectors = self._fetch_vectors_from_pinecone(self.vector_ids)
                    faiss_index = faiss.read_index("faiss_index.bin")
                except Exception as e:
                    logger.warning(f"Failed to load vectors from Pinecone: {e}")
                    logger.info("Will attempt to reload all vectors from Pinecone...")
                    # Remove local files as they're out of sync
                    if os.path.exists("faiss_index.bin"):
                        os.remove("faiss_index.bin")
                    if os.path.exists("metadata.pkl"):
                        os.remove("metadata.pkl")
                    # Reset state and try to preload
                    self.vector_ids = []
                    self.vectors = np.array([], dtype=np.float32).reshape(0, self.dim)
                    self.metadata = {}
                    self._preload_vectors(faiss_index)
            except Exception as e:
                logger.error(f"Failed to load local files: {e}")
                logger.info("Will attempt to reload all vectors from Pinecone...")
                self._preload_vectors(faiss_index)
        else:
            logger.info("Creating new FAISS index and loading vectors from Pinecone...")
            self._preload_vectors(faiss_index)
        
        if len(self.vector_ids) == 0:
            logger.warning("No vectors loaded. This might indicate an issue with Pinecone connection or empty index.")
        else:
            logger.info(f"FAISS index initialized with {len(self.vector_ids)} vectors")
        
        return faiss_index
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _fetch_vectors_from_pinecone(self, vector_ids: List[str], batch_size: int = 100) -> np.ndarray:
        """Safely fetch vectors from Pinecone in batches."""
        all_vectors = []
        for i in range(0, len(vector_ids), batch_size):
            batch_ids = vector_ids[i:i+batch_size]
            try:
                result = self.pinecone_index.fetch(ids=batch_ids, namespace=self.namespace)
                batch_vectors = [result.vectors[id].values for id in batch_ids]
                all_vectors.extend(batch_vectors)
            except Exception as e:
                logger.error(f"Failed to fetch vectors batch {i//batch_size}: {e}")
                raise
        return np.array(all_vectors, dtype=np.float32)
    
    def _preload_vectors(self, faiss_index: faiss.Index, batch_size: int = 500) -> None:
        """Preload vectors from Pinecone into FAISS."""
        logger.info("Preloading vectors from Pinecone...")
        try:
            # First get total vector count from stats
            stats = self.pinecone_index.describe_index_stats()
            total_vectors = stats.total_vector_count
            logger.info(f"Found {total_vectors} total vectors in Pinecone")
            
            if total_vectors == 0:
                logger.warning("No vectors found in Pinecone index")
                self.vector_ids = []
                self.vectors = np.array([], dtype=np.float32).reshape(0, self.dim)
                self.metadata = {}
                return
            
            # Get all vector IDs in one go with a large top_k
            logger.info("Retrieving all vector IDs...")
            query_response = self.pinecone_index.query(
                vector=[0.0] * self.dim,
                top_k=total_vectors,
                include_metadata=True,
                namespace=self.namespace
            )
            
            if not query_response.matches:
                logger.warning("No vectors found in Pinecone index")
                self.vector_ids = []
                self.vectors = np.array([], dtype=np.float32).reshape(0, self.dim)
                self.metadata = {}
                return
            
            vector_ids = [match.id for match in query_response.matches]
            logger.info(f"Retrieved {len(vector_ids)} vector IDs")
            
            # Now fetch the actual vectors in larger batches
            logger.info(f"Fetching {len(vector_ids)} vectors in batches of {batch_size}...")
            all_vectors = []
            all_metadata = {}
            successful_ids = set()
            
            # Process all chunks of vectors
            for i in range(0, len(vector_ids), batch_size):
                batch_ids = vector_ids[i:i+batch_size]
                try:
                    result = self.pinecone_index.fetch(ids=batch_ids, namespace=self.namespace)
                    for id in batch_ids:
                        if id in result.vectors:
                            all_vectors.append(result.vectors[id].values)
                            all_metadata[id] = result.vectors[id].metadata
                            successful_ids.add(id)
                    logger.info(f"Fetched batch {i//batch_size + 1}/{(len(vector_ids) + batch_size - 1)//batch_size}")
                except Exception as e:
                    logger.error(f"Failed to fetch batch {i//batch_size + 1}: {e}")
                    # Try fetching individual vectors from failed batch
                    for id in batch_ids:
                        try:
                            result = self.pinecone_index.fetch(ids=[id], namespace=self.namespace)
                            if id in result.vectors:
                                all_vectors.append(result.vectors[id].values)
                                all_metadata[id] = result.vectors[id].metadata
                                successful_ids.add(id)
                                logger.info(f"Successfully fetched individual vector {id}")
                        except Exception as e:
                            logger.error(f"Failed to fetch individual vector {id}: {e}")
            
            if not all_vectors:
                logger.warning("No vectors successfully fetched from Pinecone")
                self.vector_ids = []
                self.vectors = np.array([], dtype=np.float32).reshape(0, self.dim)
                self.metadata = {}
                return
            
            # Only keep successfully fetched vectors and their metadata
            self.vector_ids = list(successful_ids)
            self.vectors = np.array(all_vectors, dtype=np.float32)
            self.metadata = all_metadata
            
            logger.info(f"Successfully fetched {len(self.vector_ids)}/{len(vector_ids)} vectors")
            
            # Ensure vectors have the correct dimension
            if self.vectors.shape[1] != self.dim:
                logger.warning(f"Vector dimension mismatch: expected {self.dim}, got {self.vectors.shape[1]}")
                # Reshape vectors to match expected dimension
                if self.vectors.shape[1] < self.dim:
                    # Pad with zeros if dimension is too small
                    padding = np.zeros((self.vectors.shape[0], self.dim - self.vectors.shape[1]), dtype=np.float32)
                    self.vectors = np.hstack([self.vectors, padding])
                else:
                    # Truncate if dimension is too large
                    self.vectors = self.vectors[:, :self.dim]
            
            # Normalize and add to FAISS
            logger.info("Normalizing vectors and adding to FAISS...")
            faiss.normalize_L2(self.vectors)
            faiss_index.reset()
            faiss_index.add(self.vectors)
            
            # Save to disk
            logger.info("Saving index and metadata to disk...")
            faiss.write_index(faiss_index, "faiss_index.bin")
            with open("metadata.pkl", "wb") as f:
                pickle.dump(self.metadata, f)
                
            logger.info(f"Successfully preloaded {len(self.vector_ids)} vectors")
        except Exception as e:
            logger.error(f"Failed to preload vectors: {e}")
            raise
    
    def add_new_vectors(self, new_ids: List[str], vectors: List[List[float]], metadata_list: List[Dict]) -> None:
        """Add new vectors to both Pinecone and FAISS."""
        try:
            # Prepare vectors for Pinecone in larger batches
            batch_size = 1000  # Increased batch size
            pinecone_vectors = []
            
            for i in range(0, len(new_ids), batch_size):
                batch_ids = new_ids[i:i+batch_size]
                batch_vectors = vectors[i:i+batch_size]
                batch_metadata = metadata_list[i:i+batch_size]
                
                # Prepare batch
                batch = []
                for id, vector, metadata in zip(batch_ids, batch_vectors, batch_metadata):
                    metadata.update({
                        "created_at": datetime.now().isoformat(),
                        "confidence": 0.9,
                        "source": "user"
                    })
                    batch.append((id, vector, metadata))
                
                # Upsert batch to Pinecone
                self.pinecone_index.upsert(vectors=batch, namespace=self.namespace)
                logger.info(f"Upserted batch {i//batch_size + 1}/{(len(new_ids) + batch_size - 1)//batch_size}")
            
            # Add all vectors to FAISS at once
            new_vectors = np.array(vectors, dtype=np.float32)
            faiss.normalize_L2(new_vectors)
            self.faiss_index.add(new_vectors)
            
            # Update local storage
            self.vector_ids.extend(new_ids)
            if self.vectors is None:
                self.vectors = new_vectors
            else:
                self.vectors = np.vstack([self.vectors, new_vectors])
            self.metadata.update(dict(zip(new_ids, metadata_list)))
            
            # Save to disk
            faiss.write_index(self.faiss_index, "faiss_index.bin")
            with open("metadata.pkl", "wb") as f:
                pickle.dump(self.metadata, f)
                
            logger.info(f"Successfully added {len(new_ids)} new vectors")
        except Exception as e:
            logger.error(f"Failed to add new vectors: {e}")
            raise
    
    def update_vectors(self, updated_ids: List[str], vectors: List[List[float]], metadata_list: List[Dict]) -> None:
        """Update existing vectors in both Pinecone and FAISS."""
        try:
            # Update Pinecone
            pinecone_vectors = []
            for id, vector, metadata in zip(updated_ids, vectors, metadata_list):
                metadata["updated_at"] = datetime.now().isoformat()
                pinecone_vectors.append((id, vector, metadata))
            self.pinecone_index.upsert(vectors=pinecone_vectors, namespace=self.namespace)
            
            # Update FAISS
            updated_vectors = np.array(vectors, dtype=np.float32)
            faiss.normalize_L2(updated_vectors)
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
            raise
    
    def get_similar_vectors(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[str, np.ndarray, Dict]]:
        """Get semantically similar vectors using FAISS.
        
        Args:
            query_vector: Input vector to find similar vectors for
            top_k: Maximum number of similar vectors to return
            
        Returns:
            List of tuples containing (vector_id, vector, metadata)
            
        Raises:
            ValueError: If query_vector has invalid shape or type
            RuntimeError: If FAISS search fails
        """
        # Input validation
        if not isinstance(query_vector, np.ndarray):
            raise ValueError(f"query_vector must be numpy array, got {type(query_vector)}")
        if query_vector.shape[-1] != self.dim:
            raise ValueError(f"query_vector must have dimension {self.dim}, got {query_vector.shape[-1]}")
        
        if not self.vector_ids:
            logger.warning("No vectors in index, returning empty results")
            return []

        # Track memory usage before search
        mem_before = self.get_memory_usage()
        
        try:
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
            for idx in indices[0]:
                if idx < 0 or idx >= len(self.vector_ids):
                    logger.warning(f"Invalid index {idx} returned by FAISS")
                    continue
                vector_id = self.vector_ids[idx]
                results.append((
                    vector_id,
                    self.vectors[idx],
                    self.metadata[vector_id]
                ))
            
            if not results:
                logger.warning("No valid results found in FAISS search")
            
            return results
            
        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            raise RuntimeError(f"Failed to perform similarity search: {e}")
    
    def get_memory_usage(self) -> float:
        """Get current memory usage of the process in MB.
        
        Returns:
            Memory usage in MB
        """
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # Convert bytes to MB

    def cleanup(self):
        """Clean up all vectors from both FAISS and Pinecone."""
        try:
            self.pinecone_index.delete(delete_all=True, namespace=self.namespace)
            logger.info("Successfully cleaned up Pinecone vectors")
        except Exception as e:
            if "Namespace not found" in str(e):
                logger.info(f"No data to clean up in namespace '{self.namespace}'")
            else:
                logger.error(f"Failed to clean up: {e}")
        
        self.faiss_index = None
        self.faiss_index = self._initialize_faiss()



def generate_psychology_data(num_vectors: int = 3000) -> List[Tuple[str, List[float], Dict]]:
    """Generate random vectors with psychology-related metadata."""
    psychology_concepts = [
        "Cognitive Behavioral Therapy", "Mindfulness", "Depression", "Anxiety",
        "Personality Development", "Mental Health", "Emotional Intelligence",
        "Psychological Resilience", "Stress Management", "Self-awareness",
        "Behavioral Psychology", "Clinical Psychology", "Social Psychology",
        "Developmental Psychology", "Psychoanalysis", "Positive Psychology",
        "Trauma and Recovery", "Psychological Assessment", "Mental Wellness",
        "Psychological Disorders", "Therapeutic Approaches", "Psychology Research",
        "Human Behavior", "Psychological Theories", "Cognitive Development",
        "Emotional Regulation", "Psychological Well-being", "Mental Processes",
        "Behavioral Patterns", "Psychological Interventions"
    ]
    
    psychology_texts = [
        "Understanding human behavior through cognitive processes",
        "Exploring emotional intelligence in daily interactions",
        "Managing stress through mindfulness techniques",
        "Developing resilience in challenging situations",
        "Applying behavioral psychology principles",
        "Analyzing personality development patterns",
        "Implementing therapeutic interventions",
        "Studying mental health and well-being",
        "Investigating psychological disorders",
        "Researching human development stages",
        "Examining social psychology dynamics",
        "Practicing mindfulness meditation",
        "Understanding trauma recovery processes",
        "Developing coping mechanisms",
        "Exploring consciousness and awareness",
        "Analyzing decision-making processes",
        "Studying memory and learning",
        "Investigating motivation factors",
        "Understanding emotional responses",
        "Examining psychological assessments"
    ]
    
    vectors = []
    for i in range(num_vectors):
        vector_id = f"psych_vec_{i}"
        # Generate random vector
        vector = np.random.rand(1536).astype(np.float32)
        # Normalize the vector
        vector = vector / np.linalg.norm(vector)
        
        # Create metadata with psychology content
        concept = np.random.choice(psychology_concepts)
        text = np.random.choice(psychology_texts)
        metadata = {
            "text": f"{concept}: {text}",
            "categories_primary": "psychology",
            "categories_special": "knowledge",
            "user_id": "test_user",
            "concept": concept
        }
        
        vectors.append((vector_id, vector.tolist(), metadata))
    
    return vectors

def generate_psychology_queries() -> List[Tuple[str, np.ndarray]]:
    """Generate test queries related to psychology."""
    queries = [
        "What is cognitive behavioral therapy?",
        "How does mindfulness affect mental health?",
        "Explain depression and its symptoms",
        "What are anxiety management techniques?",
        "Describe personality development stages",
        "How to improve emotional intelligence?",
        "What is psychological resilience?",
        "Effective stress management strategies",
        "Understanding self-awareness",
        "Principles of behavioral psychology",
        "Applications of clinical psychology",
        "Social psychology in everyday life",
        "Stages of cognitive development",
        "Basic concepts of psychoanalysis",
        "Benefits of positive psychology",
        "Trauma recovery methods",
        "Types of psychological assessments",
        "Maintaining mental wellness",
        "Common psychological disorders",
        "Modern therapeutic approaches",
        "Latest psychology research trends",
        "Understanding human behavior patterns",
        "Major psychological theories",
        "Emotional regulation techniques",
        "Factors in psychological well-being",
        "Mental processes and consciousness"
    ]
    
    query_vectors = []
    for query in queries:
        # Generate a random vector for the query (in practice, this would be generated by an embedding model)
        vector = np.random.rand(1536).astype(np.float32)
        vector = vector / np.linalg.norm(vector)
        query_vectors.append((query, vector))
    
    return query_vectors

def add_vectors_to_pinecone(num_vectors: int = 1500, batch_size: int = 200) -> Tuple[float, float]:
    """Add vectors to Pinecone and return generation and upsert times.
    
    Args:
        num_vectors: Number of vectors to generate and add
        batch_size: Size of batches for upserting to Pinecone
        
    Returns:
        Tuple of (generation_time, upsert_time) in seconds
    """
    print("\n=== Adding vectors to Pinecone ===")
    
    # Initialize VectorManager
    print("\nInitializing VectorManager...")
    manager = VectorManager(namespace="test")
    
    # Generate vectors
    print(f"\nGenerating {num_vectors} psychology test vectors...")
    start_time = time.time()
    test_vectors = generate_psychology_data(num_vectors)
    generation_time = time.time() - start_time
    print(f"Vector generation time: {generation_time:.2f} seconds")
    
    # Add vectors to Pinecone
    print("\nUpserting vectors to Pinecone...")
    start_time = time.time()
    for i in range(0, len(test_vectors), batch_size):
        batch = test_vectors[i:i+batch_size]
        ids = [v[0] for v in batch]
        vectors = [v[1] for v in batch]
        metadata = [v[2] for v in batch]
        manager.add_new_vectors(ids, vectors, metadata)
        print(f"Upserted batch {i//batch_size + 1}/{len(test_vectors)//batch_size + 1}")
    upsert_time = time.time() - start_time
    print(f"Pinecone upsert time: {upsert_time:.2f} seconds")
    
    return generation_time, upsert_time

def test_faiss_retrieval() -> Tuple[float, List[float], float]:
    """Test FAISS retrieval performance.
    
    Returns:
        Tuple of (loading_time, query_times, memory_usage)
        - loading_time: Time to load vectors from Pinecone to FAISS
        - query_times: List of query execution times
        - memory_usage: Current memory usage in MB
    """
    print("\n=== Testing FAISS Retrieval ===")
    
    # Clean up any existing files first
    if os.path.exists("faiss_index.bin"):
        os.remove("faiss_index.bin")
    if os.path.exists("metadata.pkl"):
        os.remove("metadata.pkl")
    
    # Test loading from Pinecone to FAISS
    print("\nTesting Pinecone to FAISS loading time...")
    start_time = time.time()
    manager = VectorManager(namespace="test")
    loading_time = time.time() - start_time
    print(f"Loading time from Pinecone to FAISS: {loading_time:.2f} seconds")
    
    # Run queries
    print("\nTesting query performance...")
    queries = generate_psychology_queries()
    query_times = []
    
    print("\nRunning 26 test queries...")
    for query_text, query_vector in queries:
        start_time = time.time()
        results = manager.get_similar_vectors(query_vector, top_k=5)
        query_time = time.time() - start_time
        query_times.append(query_time)
        
        print(f"\nQuery: {query_text}")
        print(f"Query time: {query_time:.4f} seconds")
        print("Top results:")
        for vector_id, _, metadata in results:
            print(f"- {metadata['concept']}: {metadata['text'][:100]}...")
    
    # Calculate memory usage
    memory_usage = manager.get_memory_usage()
    
    # Clean up
    #print("\nCleaning up test data...")
    #manager.cleanup()
    #print("Test completed and cleaned up successfully!")
    
    return loading_time, query_times, memory_usage

def run_performance_test():
    """Run complete performance test with both vector addition and retrieval."""
    print("\n=== Starting Performance Test ===")
    
    # Part 1: Add vectors to Pinecone
    generation_time, upsert_time = add_vectors_to_pinecone()
    
    # Part 2: Test FAISS retrieval
    #loading_time, query_times, memory_usage = test_faiss_retrieval()
    
    # Print summary
    #avg_query_time = sum(query_times) / len(query_times)
    print(f"\nPerformance Summary:")
    print(f"- Vector generation time: {generation_time:.2f} seconds")
    print(f"- Pinecone upsert time: {upsert_time:.2f} seconds")
    #print(f"- Loading time from Pinecone to FAISS: {loading_time:.2f} seconds")
    #print(f"- Average query time: {avg_query_time:.4f} seconds")
    #print(f"- Min query time: {min(query_times):.4f} seconds")
    #print(f"- Max query time: {max(query_times):.4f} seconds")
    #print(f"- Current memory usage: {memory_usage:.2f} MB")

if __name__ == "__main__":
    run_performance_test()
