from active_brain import ActiveBrain
import asyncio
import os
import numpy as np

# Global brain instance - initially None
_brain_instance = None
# Flag to track if brain has been loaded with vectors
_brain_loaded = False

# Default configuration values
_default_config = {
    "dim": 1536,
    "namespace": "",
    "graph_version_ids": [],
    "pinecone_index_name": "9well"  # Default index name
}

# Current configuration
_current_config = _default_config.copy()

def init_brain(dim=1536, namespace="", graph_version_ids=None, pinecone_index_name=None):
    """Initialize the global brain instance with the provided parameters."""
    global _brain_instance, _current_config
    
    # Update current configuration
    _current_config["dim"] = dim
    _current_config["namespace"] = namespace
    _current_config["graph_version_ids"] = graph_version_ids or []
    if pinecone_index_name:
        _current_config["pinecone_index_name"] = pinecone_index_name
    
    if _brain_instance is None:
        _brain_instance = ActiveBrain(
            dim=dim,
            namespace=namespace,
            graph_version_ids=graph_version_ids,
            pinecone_index_name=pinecone_index_name
        )
    return _brain_instance

def get_brain():
    """Get the global brain instance. Initialize with defaults if not already initialized."""
    global _brain_instance
    if _brain_instance is None:
        _brain_instance = ActiveBrain(
            dim=_current_config["dim"], 
            namespace=_current_config["namespace"],
            graph_version_ids=_current_config["graph_version_ids"],
            pinecone_index_name=_current_config["pinecone_index_name"]
        )
    return _brain_instance

def reset_brain():
    """Reset the global brain instance to None."""
    global _brain_instance, _brain_loaded
    _brain_instance = None
    _brain_loaded = False

def set_graph_version(graph_version_id):
    """
    Set a new graph version ID for the brain.
    
    Args:
        graph_version_id: The new graph version ID to use
        
    Returns:
        bool: True if a new brain instance was created, False if the graph version was already set
    """
    global _brain_instance, _current_config, _brain_loaded
    
    # Check if we're already using this graph version
    if _current_config["graph_version_ids"] and _current_config["graph_version_ids"][0] == graph_version_id:
        return False
    
    # Update the configuration with the new graph version ID
    _current_config["graph_version_ids"] = [graph_version_id]
    
    # Reset the brain so it will be recreated with the new graph version
    reset_brain()
    
    return True

def get_current_graph_version():
    """Get the current graph version ID that's being used."""
    if _current_config["graph_version_ids"]:
        return _current_config["graph_version_ids"][0]
    return None

async def load_brain_vectors(graph_version_id=None):
    """
    Load vectors into the brain from the specified graph version.
    If no graph_version_id is provided, use the current one.
    
    Args:
        graph_version_id: Optional graph version ID to use (will set if different)
        
    Returns:
        bool: True if vectors were loaded successfully, False otherwise
    """
    global _brain_instance, _brain_loaded
    import time
    import numpy as np
    
    start_time = time.time()
    
    print(f"Starting brain vector loading process [time: {start_time}]")
    
    # Set the graph version if provided and different from current
    version_changed = False
    if graph_version_id:
        current = get_current_graph_version()
        if current != graph_version_id:
            print(f"Setting graph version from {current} to {graph_version_id}")
            version_changed = set_graph_version(graph_version_id)
            print(f"Version changed: {version_changed}")
    
    # If already loaded and version didn't change, just return success
    if _brain_loaded and not version_changed:
        print("Brain already loaded with current version, skipping reload")
        return True
    
    # Clean up existing files if they exist - always remove when loading
    # to ensure we have a clean state
    if os.path.exists("faiss_index.bin"):
        try:
            os.remove("faiss_index.bin")
            print("Removed existing faiss_index.bin file")
        except Exception as e:
            print(f"Warning: Could not remove faiss_index.bin: {e}")
            
    if os.path.exists("metadata.pkl"):
        try:
            os.remove("metadata.pkl")
            print("Removed existing metadata.pkl file")
        except Exception as e:
            print(f"Warning: Could not remove metadata.pkl: {e}")
    
    # Ensure brain instance exists
    brain = get_brain()
    if brain is None:
        print("Failed to initialize brain instance")
        return False
    
    try:
        # Get the current graph version to load
        current_version = get_current_graph_version()
        if not current_version:
            print("No graph version set, cannot load vectors")
            return False
            
        print(f"Loading vectors for graph version: {current_version}")
        
        # Load vectors with timeout handling
        try:
            import asyncio
            # Set a reasonable timeout for loading (5 minutes)
            load_task = asyncio.create_task(brain.load_all_vectors_from_graph_version(current_version))
            await asyncio.wait_for(load_task, timeout=300)  # 5 minute timeout
        except asyncio.TimeoutError:
            print("Vector loading timed out after 5 minutes")
            return False
        except Exception as e:
            import traceback
            print(f"ERROR during vector loading: {e}")
            print(traceback.format_exc())
            return False
        
        # Check if vectors were loaded
        vector_count = brain.faiss_index.ntotal if hasattr(brain, 'faiss_index') else 0
        if vector_count == 0:
            print("No vectors loaded into FAISS index")
            return False
        
        # Validate the loaded vectors
        try:
            if hasattr(brain, 'vectors') and brain.vectors is not None:
                # Check for zero vectors
                zero_vectors = 0
                fixed_vectors = 0
                
                if len(brain.vectors) > 0:
                    # Find vectors with all zeros or near-zeros
                    zero_mask = np.abs(brain.vectors).sum(axis=1) < 0.001
                    zero_vectors = np.sum(zero_mask)
                    
                    # Fix zero vectors if any are found
                    if zero_vectors > 0:
                        print(f"WARNING: {zero_vectors} vectors with all zeros detected, fixing them...")
                        
                        # For each zero vector, create a deterministic random replacement
                        for i in np.where(zero_mask)[0]:
                            if i < len(brain.vector_ids):
                                vector_id = brain.vector_ids[i]
                                # Create deterministic vector based on ID
                                import hashlib
                                hash_val = int(hashlib.md5(vector_id.encode()).hexdigest(), 16)
                                np.random.seed(hash_val)
                            else:
                                # Fallback for vectors without IDs
                                np.random.seed(i + 42)
                                
                            # Generate random vector with normal distribution
                            brain.vectors[i] = np.random.normal(0, 0.1, brain.dim).astype(np.float32)
                            
                            # Ensure some definite non-zero values
                            for j in range(min(5, brain.dim)):
                                brain.vectors[i][j] = 0.1 * (j + 1) / 5
                                
                            # Normalize the vector
                            norm = np.linalg.norm(brain.vectors[i])
                            if norm > 0:
                                brain.vectors[i] = brain.vectors[i] / norm
                                
                            fixed_vectors += 1
                        
                        # Rebuild FAISS index with fixed vectors
                        brain.faiss_index.reset()
                        import faiss
                        vectors_to_add = brain.vectors.copy()
                        faiss.normalize_L2(vectors_to_add)
                        brain.faiss_index.add(vectors_to_add)
                        print(f"Rebuilt FAISS index with {fixed_vectors} fixed vectors")
                        
                        # Save the updated index to disk
                        faiss.write_index(brain.faiss_index, "faiss_index.bin")
                        print("Saved updated FAISS index to disk")
                    
                    # Check for NaN values
                    nan_vectors = np.isnan(brain.vectors).any(axis=1).sum()
                    if nan_vectors > 0:
                        print(f"WARNING: {nan_vectors} vectors with NaN values detected in loaded vectors")
                
                print(f"Vector validation: {len(brain.vectors)} total vectors, {zero_vectors} zero vectors detected, {fixed_vectors} vectors fixed")
            else:
                print("WARNING: No vector data available for validation")
        except Exception as validation_error:
            print(f"Error during vector validation: {validation_error}")
            import traceback
            print(traceback.format_exc())
            # Continue despite validation errors
        
        # Perform a quick test query to validate the index
        try:
            print("Performing test query on loaded vectors...")
            # Generate a simple test vector
            test_vector = np.random.randn(brain.dim).astype(np.float32)
            # Normalize the vector
            import faiss
            faiss.normalize_L2(test_vector.reshape(1, -1))
            
            # Run a test query with zero threshold to ensure we get results
            results = brain.get_similar_vectors(test_vector, top_k=3, threshold=0.0)
            print(f"Test query returned {len(results)} results")
            
            if len(results) > 0:
                # Show the top similarity score
                print(f"Top similarity score from test query: {results[0][3]:.4f}")
                
                # Print some information about the matched vector
                top_match = results[0]
                vector_id, vector, metadata, similarity = top_match
                print(f"Top match vector ID: {vector_id}")
                print(f"Top match metadata: {metadata}")
            
            if not results and vector_count > 0:
                print("WARNING: Test query returned no results despite vectors being loaded")
                # This is a warning but not necessarily a failure
            
        except Exception as test_error:
            print(f"Test query failed: {test_error}")
            import traceback
            print(traceback.format_exc())
            # Continue despite test errors
        
        end_time = time.time()
        loading_time = end_time - start_time
        
        print(f"Successfully loaded {vector_count} vectors for graph version {current_version} in {loading_time:.2f} seconds")
        
        # Set loaded flag
        _brain_loaded = True
        return True
    except Exception as e:
        import traceback
        print(f"ERROR loading vectors: {e}")
        print(traceback.format_exc())
        return False

def is_brain_loaded():
    """Check if the brain has been loaded with vectors."""
    global _brain_loaded
    return _brain_loaded 