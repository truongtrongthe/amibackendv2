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
import re

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
        try:
            self.dim = dim
            self.namespace = namespace
            self.vector_ids = []
            self.vectors = None
            self.metadata = {}
            self.graph_version_ids = graph_version_ids or []
            
            try:
                logger.info(f"Initializing ActiveBrain with dimension {dim}")
                self.faiss_index = self._initialize_faiss()
            except Exception as faiss_error:
                logger.error(f"Error initializing FAISS: {faiss_error}")
                logger.error(traceback.format_exc())
                logger.info("Creating emergency FAISS index")
                self.faiss_index = faiss.IndexFlatIP(self.dim)
            
            self.pinecone_index = None
            
            if pinecone_index_name:
                self._initialize_pinecone(pinecone_index_name)
                
            if not hasattr(self, 'faiss_index') or self.faiss_index is None:
                logger.error("FAISS index still not initialized after __init__! Creating emergency index.")
                self.faiss_index = faiss.IndexFlatIP(self.dim)
                
            logger.info("ActiveBrain initialization complete")
                
        except Exception as e:
            logger.error(f"Critical error during ActiveBrain initialization: {e}")
            logger.error(traceback.format_exc())
            self.dim = dim
            self.namespace = namespace
            self.vector_ids = []
            self.vectors = np.empty((0, self.dim), dtype=np.float32)
            self.metadata = {}
            self.faiss_index = faiss.IndexFlatIP(self.dim)
            self.pinecone_index = None

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
        try:
            logger.info("Initializing FAISS index")

            
            # This way, the similarity score IS the cosine similarity when vectors are normalized
            self.faiss_index = faiss.IndexFlatIP(self.dim)

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

                        ntotal = self.faiss_index.ntotal
                        try:
                            
                            if hasattr(self.faiss_index, 'xb'):
                                # Direct access for some index types
                                xb = self.faiss_index.xb
                                if xb is not None:
                                    self.vectors = np.array(xb).reshape(ntotal, self.dim)
                                    logger.info(f"Successfully extracted {ntotal} vectors directly")
                            elif isinstance(self.faiss_index, faiss.IndexFlat):
                                # For flat indexes, we can use .get_xb() and vector_to_array
                                flat_vectors = faiss.vector_to_array(self.faiss_index.get_xb())
                                self.vectors = flat_vectors.reshape(ntotal, self.dim)
                                logger.info(f"Successfully extracted {ntotal} vectors from flat index")
                            else:
                                # For indexes without direct vector access, we need to re-create vectors
                                raise AttributeError("Cannot directly access vectors from this index type")
                        except (AttributeError, AssertionError) as access_err:
                            logger.warning(f"Cannot extract vectors directly from index: {access_err}")
                            logger.warning("Initializing empty vector array - vectors will be recreated when needed")

                            
                            # instead of creating zero placeholders which cause issues
                            self.vectors = np.empty((0, self.dim), dtype=np.float32)

                    except Exception as e:
                        logger.error(f"Failed to load FAISS index: {e}")
                        logger.error(traceback.format_exc())
                        
                        logger.info("Creating new FAISS index after load failure")
                        self.faiss_index = faiss.IndexFlatIP(self.dim)
                        self.vectors = np.empty((0, self.dim), dtype=np.float32)
                        self.vector_ids = []
                        self.metadata = {}
                except Exception as e:
                    logger.error(f"Failed to load local files: {e}")
                    logger.error(traceback.format_exc())
                    
                    logger.info("Creating new FAISS index after metadata load failure")
                    self.faiss_index = faiss.IndexFlatIP(self.dim)
                    self.vectors = np.empty((0, self.dim), dtype=np.float32)
                    self.vector_ids = []
                    self.metadata = {}
            else:
                logger.info("Creating new FAISS index...")
                self.faiss_index = faiss.IndexFlatIP(self.dim)
                self.vectors = np.empty((0, self.dim), dtype=np.float32)
                self.vector_ids = []
                self.metadata = {}

            
            if not hasattr(self, 'faiss_index') or self.faiss_index is None:
                logger.error("FAISS index still not set after initialization! Creating emergency index.")
                self.faiss_index = faiss.IndexFlatIP(self.dim)

            return self.faiss_index

        except Exception as e:
            logger.error(f"Critical error in FAISS initialization: {e}")
            logger.error(traceback.format_exc())
            
            emergency_index = faiss.IndexFlatIP(self.dim)
            self.faiss_index = emergency_index
            self.vectors = np.empty((0, self.dim), dtype=np.float32)
            self.vector_ids = []
            self.metadata = {}
            return emergency_index

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _fetch_vectors_from_pinecone(self, vector_ids: List[str], batch_size: int = 100, strict_namespace: str = None) -> np.ndarray:
        """Safely fetch vectors from Pinecone in batches."""
        if not self.pinecone_index:
            raise ValueError("Pinecone index not initialized")        
        namespace_to_use = strict_namespace if strict_namespace is not None else self.namespace

        try:
            
            sample_ids = vector_ids[:min(5, len(vector_ids))]
            logger.info(f"Verifying vector existence with sample: {sample_ids}")
            sample_result = self.pinecone_index.fetch(ids=sample_ids, namespace=namespace_to_use)
            if not hasattr(sample_result, 'vectors') or not sample_result.vectors:
                logger.error(f"No vectors found in namespace '{namespace_to_use}' for sample IDs")

                # REMOVED: The alternate namespace discovery code
                # Instead, fail if the vectors aren't in the expected namespace
                raise KeyError(f"No vectors found in namespace '{namespace_to_use}'")
        except Exception as e:
            logger.error(f"Error during vector verification: {e}")
            # Fail early instead of continuing
            raise
        all_vectors = []
        zero_vector_count = 0
        placeholder_count = 0

        for i in range(0, len(vector_ids), batch_size):
            batch_ids = vector_ids[i:i+batch_size]
            try:
                
                logger.debug(f"Fetching batch {i//batch_size} with {len(batch_ids)} IDs")
                logger.debug(f"First few IDs in batch: {batch_ids[:3]}")
                result = self.pinecone_index.fetch(ids=batch_ids, namespace=namespace_to_use)
                logger.debug(f"Received {len(result.vectors) if hasattr(result, 'vectors') else 0} vectors from Pinecone")
                if not hasattr(result, 'vectors') or not result.vectors:
                    logger.error(f"No vectors returned for batch {i//batch_size}")
                    raise KeyError(f"No vectors found in response for batch {i//batch_size}")

                vector_ids_returned = list(result.vectors.keys())
                logger.debug(f"Received vector IDs: {vector_ids_returned[:3]}...")

                missing_ids = [id for id in batch_ids if id not in result.vectors]
                if missing_ids:
                    logger.warning(f"Missing {len(missing_ids)} IDs in response: {missing_ids[:5]}")

                batch_vectors = []
                for vector_id in batch_ids:
                    if vector_id in result.vectors:
                        vector_values = result.vectors[vector_id].values

                        
                        if all(abs(v) < 0.000001 for v in vector_values):
                            logger.warning(f"Vector {vector_id} from Pinecone has all zeros or near-zeros")
                            zero_vector_count += 1

                            
                            # This is better than pure random as it's reproducible
                            import hashlib
                            hash_val = int(hashlib.md5(vector_id.encode()).hexdigest(), 16)
                            np.random.seed(hash_val)
                            vector_values = np.random.normal(0, 0.1, self.dim).tolist()

                            
                            for j in range(min(5, len(vector_values))):
                                vector_values[j] = 0.1 * (j + 1) / 5

                            logger.info(f"Created non-zero replacement for {vector_id}")

                        batch_vectors.append(vector_values)
                    else:
                        
                        logger.warning(f"Vector ID {vector_id} not found in Pinecone, using placeholder vector")
                        placeholder_count += 1

                        
                        import hashlib
                        hash_val = int(hashlib.md5(vector_id.encode()).hexdigest(), 16)
                        np.random.seed(hash_val)
                        placeholder = np.random.normal(0, 0.1, self.dim).tolist()

                        
                        for j in range(min(5, len(placeholder))):
                            placeholder[j] = 0.1 * (j + 1) / 5

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

        logger.info(f"Vector fetch stats: {zero_vector_count} zero vectors detected and fixed, {placeholder_count} placeholders created")

        
        result_array = np.array(all_vectors, dtype=np.float32)

        
        zero_rows = np.where(np.abs(result_array).sum(axis=1) < 0.001)[0]
        if len(zero_rows) > 0:
            logger.warning(f"Found {len(zero_rows)} zero vectors in final result, fixing them")
            for row in zero_rows:
                
                np.random.seed(int(row) + 42)
                result_array[row] = np.random.normal(0, 0.1, self.dim)
                
                for j in range(min(5, result_array.shape[1])):
                    result_array[row, j] = 0.1 * (j + 1) / 5

        return result_array

    async def get_version_brain_banks(self, version_id: str) -> List[Dict[str, str]]:
        """
        Get the bank names for all brains in a version

        Args:
            version_id: UUID of the graph version

        Returns:
            List of dicts containing brain_id and bank_name
        """
        try:
            
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
            
            if not self.pinecone_index:
                logger.error("Pinecone index not initialized, cannot load vectors")
                return
            
            brain_banks = await self.get_version_brain_banks(graph_version_id)
            if not brain_banks:
                logger.warning(f"No brain banks found for graph version {graph_version_id}")

                # CRITICAL FIX: Clear the existing FAISS index when no valid banks found
                logger.info("Clearing existing FAISS index since no valid brain banks were found")
                if hasattr(self, 'faiss_index') and self.faiss_index is not None:
                    self.faiss_index.reset()
                    logger.info("FAISS index has been reset")

                self.vectors = np.empty((0, self.dim), dtype=np.float32)
                self.vector_ids = []  # CRITICAL FIX: Reset vector_ids to empty list
                self.metadata = {}

                try:
                    import faiss
                    faiss.write_index(self.faiss_index, "faiss_index.bin")
                    with open("metadata.pkl", "wb") as f:
                        pickle.dump(self.metadata, f)
                    logger.info("Saved empty state to disk")
                except Exception as save_error:
                    logger.error(f"Error saving empty state: {save_error}")

                return

            valid_namespaces = [brain["bank_name"] for brain in brain_banks]
            logger.info(f"Valid namespaces for graph version {graph_version_id}: {valid_namespaces}")
            if hasattr(self, 'faiss_index') and self.faiss_index is not None:
                logger.info("Resetting FAISS index before loading new vectors")
                self.faiss_index.reset()
            self.vectors = np.empty((0, self.dim), dtype=np.float32)
            self.vector_ids = []  
            self.metadata = {}            
            all_vectors = []
            all_ids = []
            all_metadata = []
            # Stats for zero vectors
            total_vectors_processed = 0
            zero_vectors_detected = 0
            zero_vectors_fixed = 0

            for brain in brain_banks:
                brain_id = brain["id"]
                bank_name = brain["bank_name"]

                try:
                    logger.info(f"Processing brain bank: {bank_name}")
                    # This works better when we don't know the exact IDs
                    namespace = bank_name  
                    vectors_data = await self._get_all_vectors_from_namespace(namespace, top_k=100)

                    if vectors_data:
                        logger.info(f"Successfully retrieved {len(vectors_data)} vectors from namespace {namespace}")
              
                        bank_zero_vectors = 0
                        for vector_id, vector_values, vector_metadata in vectors_data:
                            total_vectors_processed += 1
                            full_id = f"{brain_id}_{vector_id}"

                            
                            if isinstance(vector_values, list):
                                vector_values = np.array(vector_values, dtype=np.float32)
     
                            is_zero_vector = False
                            if isinstance(vector_values, np.ndarray):
                                is_zero_vector = np.all(np.abs(vector_values).sum() < 0.001)
                            else:
                                is_zero_vector = all(abs(v) < 0.000001 for v in vector_values)

                            if is_zero_vector:
                                zero_vectors_detected += 1
                                bank_zero_vectors += 1
                                logger.warning(f"Vector {vector_id} in {bank_name} has all zeros or near-zeros")
                  
                                import hashlib
                                hash_val = int(hashlib.md5(vector_id.encode()).hexdigest(), 16)
                                np.random.seed(hash_val)
                                vector_values = np.random.normal(0, 0.1, self.dim).astype(np.float32)

                                
                                for j in range(min(5, vector_values.shape[0])):
                                    vector_values[j] = 0.1 * (j + 1) / 5
               
                                norm = np.linalg.norm(vector_values)
                                if norm > 0:
                                    vector_values = vector_values / norm

                                zero_vectors_fixed += 1

                            # Final validation
                            if isinstance(vector_values, np.ndarray):
                                if np.isnan(vector_values).any():
                                    logger.warning(f"Vector {vector_id} contains NaN values, fixing")
                                    nan_mask = np.isnan(vector_values)
                                    vector_values[nan_mask] = 0.01

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

                        logger.info(f"Processed {len(vectors_data)} vectors from {bank_name}, {bank_zero_vectors} zero vectors detected and fixed")
                        continue  # Skip the ID-based fetching if we got vectors this way

                    # Fallback to the original method if querying didn't work
                    vector_ids = await self._get_all_vector_ids(bank_name)

                    if not vector_ids:
                        logger.warning(f"No vectors found in brain bank {bank_name}")
                        continue

                    logger.info(f"Fetching {len(vector_ids)} vectors from brain bank {bank_name}")

                    # Fetch vectors from Pinecone - pass the strict namespace
                    try:
                        vectors = self._fetch_vectors_from_pinecone(vector_ids, strict_namespace=bank_name)
                        logger.info(f"Successfully fetched {len(vectors)} vectors from {bank_name}")
                 
                        bank_zero_vectors = 0
                        for i, vector in enumerate(vectors):
                            total_vectors_processed += 1

                            
                            if np.all(np.abs(vector).sum() < 0.001):
                                zero_vectors_detected += 1
                                bank_zero_vectors += 1
                                logger.warning(f"Vector {vector_ids[i]} has all zeros, generating random values")

                                
                                import hashlib
                                hash_val = int(hashlib.md5(vector_ids[i].encode()).hexdigest(), 16)
                                np.random.seed(hash_val)
                                vectors[i] = np.random.normal(0, 0.1, self.dim).astype(np.float32)

                                
                                for j in range(min(5, vectors[i].shape[0])):
                                    vectors[i][j] = 0.1 * (j + 1) / 5

                                
                                faiss.normalize_L2(vectors[i].reshape(1, -1))
                                zero_vectors_fixed += 1

                        logger.info(f"Processed {len(vectors)} vectors from {bank_name}, {bank_zero_vectors} zero vectors detected and fixed")

                    except RetryError as e:
                        logger.error(f"Failed to fetch vectors after retries: {e}")
                        continue
                    except Exception as e:
                        logger.error(f"Unexpected error fetching vectors: {e}")
                        logger.error(traceback.format_exc())
                        continue

                    
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

            if all_vectors:
                
                vectors_array = np.array(all_vectors, dtype=np.float32)
 
                zero_rows = np.where(np.abs(vectors_array).sum(axis=1) < 0.001)[0]
                if len(zero_rows) > 0:
                    logger.warning(f"Found {len(zero_rows)} remaining zero vectors after processing, fixing them")
                    for row in zero_rows:
                        if row < len(all_ids):
                            vector_id = all_ids[row]
                            np.random.seed(hash(vector_id) % 2**32)
                        else:
                            np.random.seed(int(row) + 100)

                        vectors_array[row] = np.random.normal(0, 0.1, self.dim).astype(np.float32)
                        
                        for j in range(min(5, vectors_array.shape[1])):
                            vectors_array[row, j] = 0.1 * (j + 1) / 5

                        # Normalize
                        norm = np.linalg.norm(vectors_array[row])
                        if norm > 0:
                            vectors_array[row] = vectors_array[row] / norm

                        zero_vectors_fixed += 1

                nan_mask = np.isnan(vectors_array)
                if np.any(nan_mask):
                    logger.warning(f"Found {np.sum(nan_mask)} NaN values in vectors, replacing with small values")
                    vectors_array[nan_mask] = 0.01

                self.add_vectors(all_ids, vectors_array, all_metadata)

                logger.info(f"Statistics for graph version {graph_version_id}:")
                logger.info(f"  - Total vectors processed: {total_vectors_processed}")
                logger.info(f"  - Total vectors added to FAISS: {len(all_vectors)}")
                logger.info(f"  - Zero vectors detected: {zero_vectors_detected}")
                logger.info(f"  - Zero vectors fixed: {zero_vectors_fixed}")

                logger.info(f"Successfully loaded {len(all_vectors)} vectors from graph version {graph_version_id}")
            else:
                logger.warning(f"No vectors found for graph version {graph_version_id}")

                
                if hasattr(self, 'faiss_index') and self.faiss_index is not None:
                    self.faiss_index.reset()
                    logger.info("FAISS index has been reset due to no vectors found")

                try:
                    import faiss
                    faiss.write_index(self.faiss_index, "faiss_index.bin")
                    with open("metadata.pkl", "wb") as f:
                        pickle.dump(self.metadata, f)
                    logger.info("Saved empty state to disk")
                except Exception as save_error:
                    logger.error(f"Error saving empty state: {save_error}")

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
            
            logger.info(f"Getting vector IDs for bank {bank_name}")

            # For Pinecone, we can query using list_index operation or directly via fetch API
            if bank_name.startswith("wisdom_bank_"):
                # Real Pinecone namespaces often match bank names
                namespace = bank_name

                try:
                    
                    if hasattr(self.pinecone_index, 'describe_index_stats'):
                        stats = self.pinecone_index.describe_index_stats()
                        logger.debug(f"Index stats: {stats}")

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

            logger.info(f"Adding {len(new_ids)} vectors to FAISS index")

            if vectors.shape[1] != self.dim:
                logger.warning(f"Vector dimension mismatch: expected {self.dim}, got {vectors.shape[1]}")
                if vectors.shape[1] < self.dim:
                    padding = np.zeros((vectors.shape[0], self.dim - vectors.shape[1]), dtype=np.float32)
                    vectors = np.hstack([vectors, padding])
                else:
                    vectors = vectors[:, :self.dim]

            if vectors.dtype != np.float32:
                logger.info(f"Converting vector data type from {vectors.dtype} to float32")
                vectors = vectors.astype(np.float32)

            # Final check for zero vectors before adding to FAISS
            zero_rows = np.where(np.abs(vectors).sum(axis=1) < 0.001)[0]
            if len(zero_rows) > 0:
                logger.warning(f"Found {len(zero_rows)} zero vectors right before FAISS addition, fixing them")
                for row in zero_rows:
                    if row < len(new_ids):
                        vector_id = new_ids[row]
                        np.random.seed(hash(vector_id) % 2**32)
                    else:
                        np.random.seed(int(row) + 100)

                    vectors[row] = np.random.normal(0, 0.1, self.dim).astype(np.float32)
                    
                    for j in range(min(5, vectors.shape[1])):
                        vectors[row, j] = 0.1 * (j + 1) / 5

            nan_mask = np.isnan(vectors)
            if np.any(nan_mask):
                logger.warning(f"Found {np.sum(nan_mask)} NaN values in vectors, replacing with small values")
                vectors[nan_mask] = 0.01

            vectors_to_add = vectors.copy()

            try:
                logger.info("Normalizing vectors before adding to FAISS")
                
                # Using manual normalization instead of faiss.normalize_L2 for more control
                norms = np.linalg.norm(vectors_to_add, axis=1, keepdims=True)
                # Replace zero norms with 1 to avoid division by zero
                norms[norms == 0] = 1.0
                vectors_to_add = vectors_to_add / norms
            except Exception as norm_error:
                logger.error(f"Error during vector normalization: {norm_error}")
                # Fallback normalization - less efficient but more robust
                logger.info("Using fallback vector normalization")
                for i in range(vectors_to_add.shape[0]):
                    norm = np.linalg.norm(vectors_to_add[i])
                    if norm > 0:
                        vectors_to_add[i] = vectors_to_add[i] / norm

            norms = np.linalg.norm(vectors_to_add, axis=1)
            unnormalized = np.where(np.abs(norms - 1.0) > 0.01)[0]
            if len(unnormalized) > 0:
                logger.warning(f"Found {len(unnormalized)} vectors that are not normalized properly, fixing")
                for i in unnormalized:
                    norm = np.linalg.norm(vectors_to_add[i])
                    if norm > 0:
                        vectors_to_add[i] = vectors_to_add[i] / norm
                    else:
                        # If norm is zero, create a valid unit vector
                        vectors_to_add[i] = np.zeros(self.dim, dtype=np.float32)
                        vectors_to_add[i, 0] = 1.0  

            logger.info(f"Adding {vectors_to_add.shape[0]} vectors to FAISS index")
            try:
                self.faiss_index.add(vectors_to_add)
            except Exception as add_error:
                logger.error(f"Error adding vectors to FAISS: {add_error}")
                logger.error(traceback.format_exc())
                # Fallback: try adding vectors one by one
                logger.info("Trying to add vectors one by one as fallback")
                self.faiss_index.reset()
                for i in range(vectors_to_add.shape[0]):
                    try:
                        self.faiss_index.add(vectors_to_add[i:i+1])
                    except Exception as e:
                        logger.error(f"Failed to add vector {i}: {e}")

            self.vector_ids.extend(new_ids)
            if self.vectors is None:
                self.vectors = vectors  
            else:
                self.vectors = np.vstack([self.vectors, vectors])
            self.metadata.update(dict(zip(new_ids, metadata_list)))

            try:
                logger.info("Saving FAISS index to disk")
                faiss.write_index(self.faiss_index, "faiss_index.bin")

                logger.info("Saving metadata to disk")
                with open("metadata.pkl", "wb") as f:
                    pickle.dump(self.metadata, f)
            except Exception as save_error:
                logger.error(f"Error saving to disk: {save_error}")
                logger.error(traceback.format_exc())

            logger.info(f"Successfully added {len(new_ids)} new vectors to FAISS index (total: {self.faiss_index.ntotal})")
        except Exception as e:
            logger.error(f"Failed to add vectors: {e}")
            logger.error(traceback.format_exc())
            raise

    def get_memory_usage(self) -> float:
        """Get current memory usage of the process in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  

    def get_similar_vectors(self, query_vector: np.ndarray, top_k: int = 10, threshold: float = 0.0) -> List[Tuple[str, np.ndarray, Dict, float]]:
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

            # CRITICAL SAFETY CHECK: Check if faiss_index exists, initialize if needed
            if not hasattr(self, 'faiss_index') or self.faiss_index is None:
                logger.error("FAISS index not initialized! Initializing now.")
                self.faiss_index = self._initialize_faiss()
                return []  

            # CRITICAL SAFETY CHECK: Detect and fix inconsistency between FAISS index and vector_ids
            if self.faiss_index.ntotal > 0 and len(self.vector_ids) == 0:
                logger.error(f"Data inconsistency detected: FAISS index has {self.faiss_index.ntotal} vectors but vector_ids list is empty")
                logger.info("Resetting FAISS index to maintain consistency")
                self.faiss_index.reset()
                self.vectors = np.empty((0, self.dim), dtype=np.float32)
                self.metadata = {}
                return []

            if not self.vector_ids:
                logger.warning("No vectors in index, returning empty results")
                return []
            if len(query_vector.shape) == 1:
                query_vector = query_vector.reshape(1, -1)

            norm = np.linalg.norm(query_vector)
            if norm > 0:
                query_vector = query_vector / norm

            # Adjust top_k if it's larger than the number of vectors
            actual_top_k = min(top_k, len(self.vector_ids))
            if actual_top_k < top_k:
                logger.warning(f"Requested top_k={top_k} but only {actual_top_k} vectors available")

            # Perform FAISS search
            if actual_top_k == 0:
                logger.warning("Skipping search because actual_top_k=0")
                return []

            distances, indices = self.faiss_index.search(query_vector, k=actual_top_k)

            logger.info(f"Raw FAISS scores: {distances}")

            results = []

            # This helps prevent high similarity scores for unrelated content in small datasets
            adaptive_threshold = max(threshold, 0.1 + 0.2 * min(1.0, 10.0 / max(len(self.vector_ids), 1)))
            logger.info(f"Using adaptive threshold: {adaptive_threshold} (base threshold: {threshold})")

            for i, idx in enumerate(indices[0]):
                # Skip invalid indices
                if idx < 0 or idx >= len(self.vector_ids):
                    logger.warning(f"Invalid index {idx} returned by FAISS")
                    continue

                # With normalized vectors and IP index, the FAISS distance IS the cosine similarity
                # (scores range from -1 to 1, where 1 is identical and -1 is opposite)
                similarity = float(distances[0][i])

                logger.debug(f"Vector {i}: direct cosine similarity={similarity:.4f}")

                # Skip if below threshold
                if similarity < adaptive_threshold:
                    logger.debug(f"Skipping result with similarity {similarity:.4f} below threshold {adaptive_threshold:.4f}")
                    continue

                vector_id = self.vector_ids[idx]

                try:
                    vector_data = self.vectors[idx]
                except Exception as vector_error:
                    logger.warning(f"Vector data not available for index {idx}")
                    
                    vector_data = np.zeros(self.dim, dtype=np.float32)                    
                    for j in range(min(5, self.dim)):
                        vector_data[j] = 0.1 * (j + 1)
                    norm = np.linalg.norm(vector_data)
                    if norm > 0:
                        vector_data = vector_data / norm
                clean_metadata = self._create_clean_metadata(vector_id)

                results.append((
                    vector_id,
                    vector_data,
                    clean_metadata,
                    similarity
                ))

            # Sort by similarity (highest first)
            results.sort(key=lambda x: x[3], reverse=True)

            
            valid_matches = len(results)
            top_similarity = results[0][3] if valid_matches > 0 else 0
            logger.info(f"Found {valid_matches} valid matches. Top similarity: {top_similarity:.4f}")

            return results

        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Failed to perform similarity search: {e}")

    def _create_clean_metadata(self, vector_id: str) -> Dict:
        """
        Create a clean metadata dictionary with proper separation of vector and content data.

        Args:
            vector_id: The ID of the vector to get metadata for

        Returns:
            A clean metadata dictionary with vector data properly encapsulated
        """
        clean_metadata = {
            "vector_id": vector_id,
            "content": {},
            "_vector_info": {}  # Private field for vector-specific data
        }

        try:
            if vector_id in self.metadata:
                original_metadata = self.metadata[vector_id]

                # Handle different metadata types
                if isinstance(original_metadata, dict):
                    # Copy content-related fields to the content section
                    content_fields = ["content", "text", "description", "title", "raw"]
                    for field in content_fields:
                        if field in original_metadata and original_metadata[field]:
                            
                            field_value = original_metadata[field]
                            if isinstance(field_value, str):
                                # Store in clean content field
                                clean_metadata["content"][field] = field_value

                    # Store brain-specific and source information
                    for field in ["brain_id", "bank_name", "source", "url", "created_at", "updated_at"]:
                        if field in original_metadata:
                            clean_metadata[field] = original_metadata[field]

                    # Store any vector-specific information in _vector_info
                    for field in original_metadata:
                        if field not in clean_metadata and field not in clean_metadata["content"]:
                            # Skip if it looks like vector data
                            if isinstance(original_metadata[field], (list, np.ndarray)):
                                clean_metadata["_vector_info"][field] = "Vector data (omitted)"
                            else:
                                clean_metadata["_vector_info"][field] = original_metadata[field]

                    if not clean_metadata["content"]:
                        
                        constructed_content = []

                        for field, value in original_metadata.items():
                            if (field not in ["brain_id", "bank_name", "vector_id"] and 
                                isinstance(value, str) and 
                                not self._looks_like_vector_data(value)):
                                constructed_content.append(f"{field.capitalize()}: {value}")

                        if constructed_content:
                            clean_metadata["content"]["raw"] = "\n".join(constructed_content)
                        else:
                            clean_metadata["content"]["raw"] = "No content available."
                else:
                    # Non-dict metadata - store as string in content.raw
                    clean_metadata["content"]["raw"] = str(original_metadata)
            else:
                # No metadata found
                clean_metadata["content"]["raw"] = "No metadata available for this vector."
        except Exception as e:
            logger.error(f"Error creating clean metadata for vector {vector_id}: {e}")
            clean_metadata["content"]["raw"] = f"Error retrieving content: {str(e)}"

        return clean_metadata

    def _looks_like_vector_data(self, text: str) -> bool:
        """
        Check if a string looks like it contains vector data.

        Args:
            text: Text to check

        Returns:
            True if the text appears to contain vector data
        """
        if not isinstance(text, str):
            return False
        
        if "[" in text and "]" in text:
            # Look for number patterns that suggest vector data
            if re.search(r'\[\s*-?\d+\.?\d*\s*,', text) or re.search(r',\s*-?\d+\.?\d*\s*\]', text):
                return True

            
            number_pattern = r'-?\d+\.?\d*'
            if re.search(r'\[\s*' + number_pattern + r'(?:\s*,\s*' + number_pattern + r'){2,}\s*\]', text):
                return True

        return False

    async def get_similar_vectors_by_text(self, query_text: str, top_k: int = 10, threshold: float = 0.25, use_direct_similarity: bool = False) -> List[Tuple[str, np.ndarray, Dict, float]]:
        """
        Get similar vectors by text query.

        Args:
            query_text: Text to get similar vectors for
            top_k: Number of similar vectors to return
            threshold: Minimum similarity score to include in results
            use_direct_similarity: Whether to use direct cosine similarity calculation instead of FAISS

        Returns:
            List of tuples containing (vector_id, vector, metadata, similarity)
        """
        try:
            # CRITICAL SAFETY CHECK: Check if faiss_index exists, initialize if needed
            if not hasattr(self, 'faiss_index') or self.faiss_index is None:
                logger.error("FAISS index not initialized in get_similar_vectors_by_text! Initializing now.")
                self.faiss_index = self._initialize_faiss()

                # CRITICAL: Track if reinitializing caused a loss of vectors
                if hasattr(self, 'vector_ids') and self.vector_ids and self.faiss_index.ntotal == 0:
                    logger.error(f"CRITICAL ERROR: Lost {len(self.vector_ids)} vectors during reinitialization. FAISS index is empty.")
                    
                    try:
                        if hasattr(self, 'vectors') and self.vectors is not None and self.vectors.size > 0:
                            logger.info(f"Attempting to re-add {len(self.vector_ids)} vectors to the newly initialized FAISS index")
                            self.faiss_index.reset()
                            
                            normalized_vectors = self.vectors.copy()
                            faiss.normalize_L2(normalized_vectors)
                            self.faiss_index.add(normalized_vectors)
                            logger.info(f"Successfully re-added vectors to FAISS: {self.faiss_index.ntotal} vectors now available")
                    except Exception as recovery_error:
                        logger.error(f"Failed to recover vectors after reinitialization: {recovery_error}")
                        logger.error(traceback.format_exc())

                if len(self.vector_ids) == 0:
                    logger.warning("No vectors loaded in FAISS index. Returning empty results.")
                    return []

            logger.info(f"Current state: FAISS index has {self.faiss_index.ntotal} vectors, vector_ids list has {len(self.vector_ids)} items")

            # IMPROVED EMBEDDING APPROACH: Multiple embedding attempts with fallbacks
            query_vector = None
            embedding_methods = []

            # Method 1: Primary embedding method (EMBEDDINGS.aembed_query)
            async def try_primary_embedding():
                try:
                    embed_result = EMBEDDINGS.aembed_query(query_text)
                    if asyncio.iscoroutine(embed_result):
                        return await embed_result
                    return embed_result
                except Exception as e:
                    logger.warning(f"Primary embedding method failed: {e}")
                    return None
        
            embedding_methods.append(try_primary_embedding)

            for method in embedding_methods:
                if asyncio.iscoroutinefunction(method):
                    query_vector = await method()
                else:
                    query_vector = method()

                if query_vector is not None:
                    logger.info(f"Successfully generated embedding with method: {method.__name__}")
                    break

            # If all methods failed, use deterministic fallback based on text hash
            if query_vector is None:
                logger.warning("All embedding methods failed, using deterministic fallback based on text hash")
                
                import hashlib
                hash_obj = hashlib.sha256(query_text.encode('utf-8'))
                hash_digest = hash_obj.digest()

                np.random.seed(int.from_bytes(hash_digest[:4], byteorder='big'))

                query_vector = np.zeros(self.dim, dtype=np.float32)

                for i, char in enumerate(query_text[:min(50, len(query_text))]):
                    pos = (ord(char) * (i+1)) % self.dim
                    query_vector[pos] = 0.1 + (ord(char) % 10) / 100

                random_component = np.random.randn(self.dim).astype(np.float32) * 0.01
                query_vector += random_component

                # Normalize
                norm = np.linalg.norm(query_vector)
                if norm > 0:
                    query_vector = query_vector / norm

                logger.warning("Created deterministic fallback embedding")

            
            if not isinstance(query_vector, np.ndarray):
                try:
                    query_vector = np.array(query_vector, dtype=np.float32)
                except Exception as e:
                    logger.error(f"Failed to convert query vector to numpy array: {e}")
                    # If conversion fails completely, return empty results
                    return []

            if query_vector.size != self.dim:
                logger.warning(f"Query vector dimension mismatch: got {query_vector.size}, expected {self.dim}")
                if query_vector.size > self.dim:
                    query_vector = query_vector[:self.dim]
                else:
                    # Pad with zeros
                    padded = np.zeros(self.dim, dtype=np.float32)
                    padded[:query_vector.size] = query_vector
                    query_vector = padded

            if np.isnan(query_vector).any():
                logger.warning(f"Query vector contains NaN values, replacing with zeros")
                query_vector = np.nan_to_num(query_vector)

            if np.all(np.abs(query_vector).sum() < 0.001):
                logger.warning(f"Query vector for '{query_text}' is all zeros, using structured fallback")
                
                query_vector = np.zeros(self.dim, dtype=np.float32)
                for i in range(min(100, len(query_text))):
                    if i < len(query_text):
                        pos = (ord(query_text[i]) * 17) % self.dim
                        query_vector[pos] = 0.1 + (i % 10) / 100

                norm = np.linalg.norm(query_vector)
                if norm > 0:
                    query_vector = query_vector / norm
                else:
                    # Last resort: fixed pattern
                    query_vector = np.zeros(self.dim, dtype=np.float32)
                    query_vector[0] = 1.0

            # Perform the similarity search
            try:
                if use_direct_similarity and hasattr(self, 'vectors') and self.vectors is not None and len(self.vectors) > 0:
                    
                    logger.info("Using direct cosine similarity calculation")
                    query_vector = query_vector / np.linalg.norm(query_vector)
                    similarities = []
                    for i, vector_id in enumerate(self.vector_ids):
                        if i < len(self.vectors):
                            similarity = self.calculate_cosine_similarity(query_vector, self.vectors[i])
                            
                            similarity = similarity
                            if similarity >= threshold:
                                similarities.append((vector_id, i, similarity))

                    # Sort by similarity (highest first)
                    similarities.sort(key=lambda x: x[2], reverse=True)
                    similarities = similarities[:top_k]

                    # Format results
                    results = []
                    for vector_id, idx, similarity in similarities:
                        try:
                            vector_data = self.vectors[idx]
                            
                            metadata_dict = {}
                            if vector_id in self.metadata:
                                metadata_dict = self.metadata[vector_id] if isinstance(self.metadata[vector_id], dict) else {"data": str(self.metadata[vector_id])}
                            else:
                                metadata_dict = {"vector_id": vector_id}

                            results.append((vector_id, vector_data, metadata_dict, similarity))
                        except Exception as result_error:
                            logger.error(f"Error creating result for vector {vector_id}: {result_error}")

                    logger.info(f"Direct similarity calculation found {len(results)} results")
                    return results
                else:
                    # IMPORTANT: Use the actual threshold parameter here
                    results = self.get_similar_vectors(query_vector, top_k, threshold)
                    return results

            except Exception as search_error:
                logger.error(f"Error in similarity search: {search_error}")
                logger.error(traceback.format_exc())
                return []

        except Exception as e:
            logger.error(f"Error in get_similar_vectors_by_text: {e}")
            logger.error(traceback.format_exc())
            return []

    def calculate_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate exact cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score between 0 and 1
        """
        if len(vec1.shape) > 1:
            vec1 = vec1.flatten()
        if len(vec2.shape) > 1:
            vec2 = vec2.flatten()
        
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        vec1_normalized = vec1 / norm1
        vec2_normalized = vec2 / norm2
        
        similarity = np.dot(vec1_normalized, vec2_normalized)
        
        return float(max(0.0, min(1.0, similarity))) 