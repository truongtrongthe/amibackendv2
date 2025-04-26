from active_brain import ActiveBrain
import asyncio
import os
import numpy as np
import time
import fcntl
import pickle
import faiss

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

# Paths for index file and metadata
FAISS_INDEX_PATH = "faiss_index.bin"
METADATA_PATH = "metadata.pkl"
INDEX_LOCK_PATH = "faiss_index.lock"

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
        
        # Try to load from disk if available
        try_load_from_disk()
    
    # Ensure metadata is always a dictionary
    if hasattr(_brain_instance, 'metadata'):
        if not isinstance(_brain_instance.metadata, dict):
            print("Converting metadata from list to dictionary")
            # Convert list to dictionary using vector_ids as keys if available
            if hasattr(_brain_instance, 'vector_ids') and len(_brain_instance.vector_ids) == len(_brain_instance.metadata):
                _brain_instance.metadata = dict(zip(_brain_instance.vector_ids, _brain_instance.metadata))
            else:
                # Fallback: create dictionary with numeric keys
                _brain_instance.metadata = {str(i): meta for i, meta in enumerate(_brain_instance.metadata)}
    else:
        _brain_instance.metadata = {}
        
    return _brain_instance

def reset_brain():
    """Reset the global brain instance to None and release any resources."""
    global _brain_instance, _brain_loaded
    
    # Log the reset operation
    print(f"Resetting brain instance (PID: {os.getpid()})")
    
    # If the brain instance exists, try to clean up any resources
    if _brain_instance is not None:
        try:
            # If there's a FAISS index, make sure we don't leave it in a bad state
            if hasattr(_brain_instance, 'faiss_index') and _brain_instance.faiss_index is not None:
                try:
                    print(f"Existing FAISS index has {_brain_instance.faiss_index.ntotal} vectors before reset")
                except Exception as e:
                    print(f"Error accessing FAISS index properties: {e}")
            
            # Explicitly release large data structures
            if hasattr(_brain_instance, 'vectors') and _brain_instance.vectors is not None:
                del _brain_instance.vectors
            
            if hasattr(_brain_instance, 'metadata') and _brain_instance.metadata is not None:
                del _brain_instance.metadata
            
            if hasattr(_brain_instance, 'vector_ids') and _brain_instance.vector_ids is not None:
                del _brain_instance.vector_ids
            
            # Try to explicitly release the FAISS index
            if hasattr(_brain_instance, 'faiss_index') and _brain_instance.faiss_index is not None:
                try:
                    del _brain_instance.faiss_index
                except Exception as e:
                    print(f"Error releasing FAISS index: {e}")
        except Exception as e:
            print(f"Error during brain cleanup: {e}")
            import traceback
            print(traceback.format_exc())
    
    # Finally set the instance to None
    _brain_instance = None
    _brain_loaded = False
    print("Brain instance reset complete")

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
    if _current_config["graph_version_ids"] and len(_current_config["graph_version_ids"]) > 0:
        return _current_config["graph_version_ids"][0]
    return None

def acquire_lock(lock_path, timeout=30):
    """
    Acquire a file lock with timeout.
    
    Args:
        lock_path: Path to the lock file
        timeout: Maximum time to wait for lock in seconds
        
    Returns:
        file handle, or None if lock acquisition failed
    """
    start_time = time.time()
    lock_file = None
    
    try:
        # Make sure the lock file's directory exists
        lock_dir = os.path.dirname(lock_path)
        if lock_dir and not os.path.exists(lock_dir):
            try:
                os.makedirs(lock_dir, exist_ok=True)
                print(f"Created directory for lock: {lock_dir}")
            except Exception as mkdir_error:
                print(f"Error creating lock directory: {mkdir_error}")
        
        # Create the lock file if it doesn't exist
        lock_file = open(lock_path, 'w+')
        
        # Set non-blocking flag for easier debugging
        old_flags = fcntl.fcntl(lock_file.fileno(), fcntl.F_GETFL)
        fcntl.fcntl(lock_file.fileno(), fcntl.F_SETFL, old_flags | os.O_NONBLOCK)
        
        attempts = 0
        while True:
            attempts += 1
            try:
                # Try to acquire the lock (non-blocking)
                fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                print(f"Successfully acquired lock on {lock_path} after {attempts} attempts")
                
                # Write PID to lockfile for debugging
                lock_file.seek(0)
                lock_file.write(f"{os.getpid()}")
                lock_file.flush()
                
                return lock_file
            except IOError as e:
                # Check if we've exceeded the timeout
                if time.time() - start_time > timeout:
                    print(f"Timeout waiting for lock on {lock_path} after {attempts} attempts")
                    # Try to read the PID of the process holding the lock
                    try:
                        with open(lock_path, 'r') as existing_lock:
                            pid = existing_lock.read().strip()
                            print(f"Lock appears to be held by process {pid}")
                    except Exception:
                        pass
                    
                    if lock_file:
                        lock_file.close()
                    return None
                
                # Wait a bit before trying again
                print(f"Waiting for lock on {lock_path}... (attempt {attempts})")
                time.sleep(0.5)
    except Exception as e:
        print(f"Error acquiring lock: {e}")
        import traceback
        print(traceback.format_exc())
        if lock_file:
            lock_file.close()
        return None

def release_lock(lock_file):
    """
    Release a file lock.
    
    Args:
        lock_file: File handle to release
    """
    if lock_file:
        try:
            # Clear the lockfile contents before releasing
            lock_file.seek(0)
            lock_file.truncate()
            lock_file.write("released")
            lock_file.flush()
            
            # Release the lock
            fcntl.flock(lock_file, fcntl.LOCK_UN)
            lock_file.close()
            print("Released lock")
        except Exception as e:
            print(f"Error releasing lock: {e}")
            import traceback
            print(traceback.format_exc())
            # Try to force close if normal release failed
            try:
                lock_file.close()
            except:
                pass

def try_load_from_disk():
    """
    Try to load brain vectors from disk if available.
    
    Returns:
        bool: True if vectors were loaded successfully, False otherwise
    """
    global _brain_instance, _brain_loaded
    
    if not _brain_instance:
        return False
    
    # Check if files exist
    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(METADATA_PATH):
        return False
    
    try:
        # Acquire lock for reading
        lock_file = acquire_lock(INDEX_LOCK_PATH)
        if not lock_file:
            print("Could not acquire lock for reading index files")
            return False
        
        # Load FAISS index
        try:
            print(f"Loading FAISS index from {FAISS_INDEX_PATH}")
            _brain_instance.faiss_index = faiss.read_index(FAISS_INDEX_PATH)
            
            # Load metadata
            print(f"Loading metadata from {METADATA_PATH}")
            with open(METADATA_PATH, 'rb') as f:
                metadata = pickle.load(f)
                _brain_instance.vector_ids = metadata.get('vector_ids', [])
                _brain_instance.vectors = metadata.get('vectors', None)
                _brain_instance.metadata = metadata.get('metadata', {})
            
            print(f"Successfully loaded {_brain_instance.faiss_index.ntotal} vectors from disk")
            _brain_loaded = True
            return True
        except Exception as e:
            print(f"Error loading index from disk: {e}")
            import traceback
            print(traceback.format_exc())
            return False
        finally:
            # Release lock
            release_lock(lock_file)
    except Exception as e:
        print(f"Unexpected error in try_load_from_disk: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def save_to_disk():
    """
    Save brain vectors to disk.
    
    Returns:
        bool: True if vectors were saved successfully, False otherwise
    """
    global _brain_instance
    
    if not _brain_instance or not hasattr(_brain_instance, 'faiss_index'):
        print("No brain instance or FAISS index to save")
        return False
    
    try:
        # Acquire lock for writing
        lock_file = acquire_lock(INDEX_LOCK_PATH)
        if not lock_file:
            print("Could not acquire lock for saving index files")
            return False
        
        try:
            # Save FAISS index
            print(f"Saving FAISS index to {FAISS_INDEX_PATH}")
            faiss.write_index(_brain_instance.faiss_index, FAISS_INDEX_PATH)
            
            # Handle metadata conversion if it's a list instead of a dictionary
            brain_metadata = getattr(_brain_instance, 'metadata', {})
            if isinstance(brain_metadata, list):
                print("Converting metadata from list to dictionary before saving")
                # Convert list to dictionary using vector_ids as keys if available
                vector_ids = getattr(_brain_instance, 'vector_ids', [])
                if len(vector_ids) == len(brain_metadata):
                    brain_metadata = dict(zip(vector_ids, brain_metadata))
                else:
                    # Fallback: create dictionary with numeric keys
                    brain_metadata = {str(i): meta for i, meta in enumerate(brain_metadata)}
            
            # Save metadata
            print(f"Saving metadata to {METADATA_PATH}")
            metadata = {
                'vector_ids': getattr(_brain_instance, 'vector_ids', []),
                'vectors': getattr(_brain_instance, 'vectors', None),
                'metadata': brain_metadata
            }
            with open(METADATA_PATH, 'wb') as f:
                pickle.dump(metadata, f)
            
            print(f"Successfully saved {_brain_instance.faiss_index.ntotal} vectors to disk")
            return True
        except Exception as e:
            print(f"Error saving index to disk: {e}")
            import traceback
            print(traceback.format_exc())
            return False
        finally:
            # Release lock
            release_lock(lock_file)
    except Exception as e:
        print(f"Unexpected error in save_to_disk: {e}")
        import traceback
        print(traceback.format_exc())
        return False

async def load_brain_vectors(graph_version_id=None, force_delete=True):
    """
    Load vectors into the brain from the specified graph version.
    If no graph_version_id is provided, use the current one.
    
    Args:
        graph_version_id: Optional graph version ID to use (will set if different)
        force_delete: Force deletion of existing index files even if brain is already loaded
        
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
    
    # If already loaded and version didn't change, we might still need to clean files
    if _brain_loaded and not version_changed and not force_delete:
        print("Brain already loaded with current version, skipping reload")
        return True
    
    # Clean up existing files if they exist - always remove when loading
    # to ensure we have a clean state
    delete_success = True
    
    # Acquire lock for deleting files
    lock_file = acquire_lock(INDEX_LOCK_PATH)
    if not lock_file:
        print("Could not acquire lock for deleting index files")
        return False
        
    try:
        # Check for FAISS index file
        if os.path.exists(FAISS_INDEX_PATH):
            # First check if we have write permissions to the file
            if not os.access(FAISS_INDEX_PATH, os.W_OK):
                print(f"WARNING: No write permissions to {FAISS_INDEX_PATH}")
                try:
                    # Try to make the file writable
                    os.chmod(FAISS_INDEX_PATH, 0o644)
                    print(f"Changed permissions for {FAISS_INDEX_PATH}")
                except Exception as perm_error:
                    print(f"ERROR: Could not change permissions: {perm_error}")
            
            # Now try to remove the file, with retries
            for attempt in range(3):
                try:
                    print(f"Attempting to remove faiss_index.bin file (attempt {attempt+1})")
                    os.remove(FAISS_INDEX_PATH)
                    print("Successfully removed existing faiss_index.bin file")
                    break
                except Exception as e:
                    print(f"ERROR: Could not remove faiss_index.bin (attempt {attempt+1}): {e}")
                    import traceback
                    print(traceback.format_exc())
                    
                    if attempt < 2:  # Don't sleep on last attempt
                        print("Waiting before retry...")
                        time.sleep(1)  # Wait a bit before retrying
                    else:
                        delete_success = False
                        
        # Check for metadata file
        if os.path.exists(METADATA_PATH):
            # First check if we have write permissions to the file
            if not os.access(METADATA_PATH, os.W_OK):
                print(f"WARNING: No write permissions to {METADATA_PATH}")
                try:
                    # Try to make the file writable
                    os.chmod(METADATA_PATH, 0o644)
                    print(f"Changed permissions for {METADATA_PATH}")
                except Exception as perm_error:
                    print(f"ERROR: Could not change permissions: {perm_error}")
            
            # Now try to remove the file, with retries
            for attempt in range(3):
                try:
                    print(f"Attempting to remove metadata.pkl file (attempt {attempt+1})")
                    os.remove(METADATA_PATH)
                    print("Successfully removed existing metadata.pkl file")
                    break
                except Exception as e:
                    print(f"ERROR: Could not remove metadata.pkl (attempt {attempt+1}): {e}")
                    import traceback
                    print(traceback.format_exc())
                    
                    if attempt < 2:  # Don't sleep on last attempt
                        print("Waiting before retry...")
                        time.sleep(1)  # Wait a bit before retrying
                    else:
                        delete_success = False
    finally:
        # Release lock
        release_lock(lock_file)
    
    if not delete_success:
        print("WARNING: Failed to delete one or more existing index files. This may cause stale data to be used.")
    
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
                        save_to_disk()
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
        
        # Save to disk so other workers can access
        save_success = save_to_disk()
        if not save_success:
            print("WARNING: Failed to save vectors to disk. Other workers may not see these vectors.")
        
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
    global _brain_loaded, _brain_instance
    
    # If we have a flag indicating it's loaded, first check that
    if _brain_loaded:
        return True
        
    # Otherwise, check if we have a brain instance with a loaded index
    if _brain_instance and hasattr(_brain_instance, 'faiss_index'):
        # Check if the FAISS index has vectors
        if _brain_instance.faiss_index.ntotal > 0:
            _brain_loaded = True
            return True
            
    # Try loading from disk as a last resort
    if try_load_from_disk():
        return True
        
    return False 