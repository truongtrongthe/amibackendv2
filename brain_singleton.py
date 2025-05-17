from active_brain import ActiveBrain
import asyncio
import os
import numpy as np
import time
import fcntl
import pickle
import faiss
import json
import threading
from typing import Optional, Union, Any
from utilities import logger
import traceback

# Global brain instance - initially None
_brain_instance = None
# Flag to track if brain has been loaded with vectors
_brain_loaded = False
_brain_graph_version = None
# Use threading.RLock() instead of asyncio.Lock() for proper thread safety
_brain_lock = threading.RLock()
_brain_async_lock = asyncio.Lock()  # Keep for async functions
_last_reset_time = 0
_reset_in_progress = False
_pending_reset = False
_version_check_lock = threading.RLock()  # New lock for version checks

# Minimum time between resets (seconds) to prevent rapid resets
MIN_RESET_INTERVAL = 30

# Default configuration values
_default_config = {
    "dim": 1536,
    "namespace": "",
    "graph_version_ids": [],
    "pinecone_index_name": "9well",  # Default index name
    "_skip_disk_load": False
}

# Current configuration
_current_config = _default_config.copy()

# Paths for index file and metadata
FAISS_INDEX_PATH = "faiss_index.bin"
METADATA_PATH = "metadata.pkl"
INDEX_LOCK_PATH = "faiss_index.lock"

def init_brain(dim=1536, namespace="", graph_version_ids=None, pinecone_index_name=None):
    """Initialize the global brain instance with the provided parameters."""
    global _brain_instance, _current_config, _brain_graph_version
    
    with _brain_lock:
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
                
            # Initialize graph version based on config if provided
            if graph_version_ids and len(graph_version_ids) > 0 and _brain_graph_version is None:
                _brain_graph_version = graph_version_ids[0]
                logger.info(f"Setting initial brain graph version to {_brain_graph_version}")
                
        return _brain_instance

def get_brain_sync(graph_version_id: Optional[str] = None) -> Any:
    """
    Synchronous version of get_brain for compatibility with non-async code
    
    Args:
        graph_version_id: Optional graph version ID to load
        
    Returns:
        ActiveBrain instance
    """
    global _brain_instance, _brain_graph_version, _pending_reset
    
    try:
        # Import inside function to avoid circular imports
        from active_brain import ActiveBrain
        
        with _brain_lock:
            # Create brain if it doesn't exist yet
            if _brain_instance is None:
                logger.info("Creating new brain instance (sync)")
                _brain_instance = ActiveBrain(pinecone_index_name="9well")
                _brain_graph_version = None
            
            # Check if we need to reset and load a specific graph version
            with _version_check_lock:
                if graph_version_id and _brain_graph_version != graph_version_id:
                    logger.info(f"Brain version mismatch: current={_brain_graph_version}, requested={graph_version_id}")
                    
                    if not _reset_in_progress and not _pending_reset:
                        logger.warning("Starting synchronous brain reset due to version mismatch")
                        # We have to perform a synchronous reset since we're in a sync context
                        reset_brain()
                        # Set the graph version so it's correct for subsequent checks
                        _brain_graph_version = graph_version_id
                    else:
                        logger.warning("Reset in progress, sync brain access will return current version")
            
            return _brain_instance
    except Exception as e:
        logger.error(f"Error getting brain (sync): {e}")
        logger.error(traceback.format_exc())
        return None

# Backwards compatibility alias - still needed for code that uses the original signature
def get_brain() -> Any:
    """
    Backwards compatible function to get the brain
    """
    return get_brain_sync()

async def get_brain(graph_version_id: Optional[str] = None) -> Any:
    """
    Get or create brain singleton instance (async version)
    
    If graph_version_id is provided, will verify that the brain is loaded with the correct graph version.
    If not, will reset the brain and load the specified graph version.
    
    Args:
        graph_version_id: Optional graph version ID to load
        
    Returns:
        ActiveBrain instance
    """
    global _brain_instance, _brain_graph_version, _pending_reset, _reset_in_progress
    
    try:
        # Import inside function to avoid circular imports
        from active_brain import ActiveBrain
        
        # Use atomic version check with lock
        with _brain_lock:
            # Create brain if it doesn't exist yet
            if _brain_instance is None:
                logger.info("Creating new brain instance")
                _brain_instance = ActiveBrain(pinecone_index_name="9well")
                _brain_graph_version = None
            
            # Check if we need to reset and load a specific graph version
            with _version_check_lock:
                current_version = _brain_graph_version  # Cache to avoid race conditions
                
                if graph_version_id and current_version != graph_version_id:
                    logger.info(f"Brain version mismatch: current={current_version}, requested={graph_version_id}")
                
                    # Only schedule a reset if one is not already in progress or pending
                    if not _reset_in_progress and not _pending_reset:
                        _pending_reset = True
                        logger.info(f"Scheduling reset for version {graph_version_id}")
                        
                        # Perform reset in background to not block the request
                        asyncio.create_task(reset_brain_and_load_version(graph_version_id))
                    else:
                        logger.info(f"Reset already pending/in progress, will load version {graph_version_id} when complete")
            
            return _brain_instance
    except Exception as e:
        logger.error(f"Error getting brain: {e}")
        logger.error(traceback.format_exc())
        return None

async def reset_brain_and_load_version(graph_version_id: str):
    """
    Reset brain and load a specific graph version
    
    Args:
        graph_version_id: Graph version ID to load
    """
    global _brain_instance, _brain_graph_version, _brain_async_lock, _last_reset_time, _reset_in_progress, _pending_reset
    
    # Set flags to prevent concurrent resets
    with _version_check_lock:
        # Prevent multiple resets from running concurrently
        if _reset_in_progress:
            logger.info(f"Reset already in progress, skipping redundant reset for version {graph_version_id}")
            return
        
        _reset_in_progress = True
        _pending_reset = False

    try:
        # Check if we've reset recently to avoid thrashing
        current_time = time.time()
        if current_time - _last_reset_time < MIN_RESET_INTERVAL:
            wait_time = MIN_RESET_INTERVAL - (current_time - _last_reset_time)
            logger.info(f"Reset attempted too soon, waiting {wait_time:.2f} seconds")
            await asyncio.sleep(wait_time)
        
        # Acquire lock to prevent concurrent modification - use both threading and asyncio locks
        with _brain_lock:
            async with _brain_async_lock:
                logger.info(f"Resetting brain instance (PID: {os.getpid()})")
                
                # Log existing state before reset
                vector_count = 0
                if _brain_instance and hasattr(_brain_instance, 'faiss_index'):
                    vector_count = _brain_instance.faiss_index.ntotal
                    logger.info(f"Existing FAISS index has {vector_count} vectors before reset")
                
                try:
                    # Set graph version BEFORE loading vectors
                    with _version_check_lock:
                        _brain_graph_version = graph_version_id
                        logger.info(f"Set graph version to {graph_version_id}")
                        
                        # Load vectors for this graph version
                        if _brain_instance:
                            logger.info(f"Starting brain vector loading process [time: {time.time()}]")
                            
                            # Check if the load_all_vectors_from_graph_version method returns a coroutine
                            load_result = _brain_instance.load_all_vectors_from_graph_version(graph_version_id)
                            
                            # Await the result if it's a coroutine
                            if asyncio.iscoroutine(load_result):
                                await load_result
                            
                            # Update last reset time
                            _last_reset_time = time.time()
                            
                            # Verify vectors were loaded
                            if hasattr(_brain_instance, 'faiss_index'):
                                new_vector_count = _brain_instance.faiss_index.ntotal
                                logger.info(f"Brain reset complete. FAISS index now has {new_vector_count} vectors")
                                
                                # If vectors were lost, log a warning
                                if vector_count > 0 and new_vector_count == 0:
                                    logger.error(f"CRITICAL: Lost all vectors during reset! Previous: {vector_count}, Current: {new_vector_count}")
                            
                            logger.info("Brain instance reset complete")
                        else:
                            logger.error("Cannot load vectors - brain instance is None")
                            
                except Exception as load_error:
                    logger.error(f"Error loading vectors: {load_error}")
                    logger.error(traceback.format_exc())
    except Exception as e:
        logger.error(f"Error during brain reset: {e}")
        logger.error(traceback.format_exc())
    finally:
        # Clear reset-in-progress flag
        with _version_check_lock:
            _reset_in_progress = False

def reset_brain():
    """Synchronous function to reset the brain singleton"""
    global _brain_instance, _brain_graph_version, _last_reset_time, _pending_reset, _reset_in_progress
    
    try:
        # Use lock to ensure thread safety
        with _brain_lock:
            with _version_check_lock:
                # Set flags to prevent concurrent resets
                if _reset_in_progress:
                    logger.info(f"Reset already in progress, skipping redundant sync reset")
                    return
                _reset_in_progress = True
                _pending_reset = False
            
        logger.info(f"Resetting brain instance (PID: {os.getpid()})")
        
        # Log existing state before reset
        vector_count = 0
        if _brain_instance and hasattr(_brain_instance, 'faiss_index'):
            vector_count = _brain_instance.faiss_index.ntotal
            logger.info(f"Existing FAISS index has {vector_count} vectors before reset")
        
        # Create a new brain instance
        from active_brain import ActiveBrain
        _brain_instance = ActiveBrain(pinecone_index_name="9well")
        
        # Update last reset time
        _last_reset_time = time.time()
        
        logger.info("Brain instance reset complete")
    except Exception as e:
        logger.error(f"Error resetting brain: {e}")
        logger.error(traceback.format_exc())
    finally:
        # Clear reset-in-progress flag
        with _version_check_lock:
            _reset_in_progress = False

def get_current_graph_version() -> Optional[str]:
    """Get the current graph version ID"""
    global _brain_graph_version
    
    # Use lock to ensure thread safety during read
    with _version_check_lock:
        return _brain_graph_version

def set_graph_version(graph_version_id):
    """
    Set a new graph version ID for the brain.
    
    Args:
        graph_version_id: The new graph version ID to use
        
    Returns:
        bool: True if a new brain instance was created, False if the graph version was already set
    """
    global _brain_instance, _current_config, _brain_loaded, _brain_graph_version
    
    # Use locks to ensure thread safety
    with _brain_lock:
        with _version_check_lock:
            # Check if we're already using this graph version
            if _brain_graph_version == graph_version_id:
                return False
            
            # Update the configuration with the new graph version ID
            _current_config["graph_version_ids"] = [graph_version_id]
            
            # Reset the brain so it will be recreated with the new graph version
            reset_brain()
            
            # Explicitly set the graph version after reset
            _brain_graph_version = graph_version_id
            logger.info(f"Graph version explicitly set to {graph_version_id}")
    
    return True

def is_brain_loaded():
    """Check if the brain has been loaded with vectors."""
    global _brain_loaded, _brain_instance
    
    # Use lock to ensure thread safety
    with _brain_lock:
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

async def activate_brain_with_version(graph_version_id):
    """
    Activate the brain with a specific graph version ID.
    This performs a full reset and reload of vectors for the specified graph version.
    
    Args:
        graph_version_id: The graph version ID to activate
        
    Returns:
        dict: A dictionary with activation results including success status and stats
    """
    global _brain_instance, _brain_loaded, _brain_graph_version
    
    # Set the graph version ID with thread safety
    with _version_check_lock:
        # Reset flags first
        _reset_in_progress = False
        _pending_reset = False
        
        # Force version to None to ensure reset triggers properly
        _brain_graph_version = None
    
    # Always perform a full reset first
    reset_brain()
    
    # Then load the vectors
    success = await load_brain_vectors(graph_version_id, force_delete=True)
    
    if success:
        # Get updated brain instance
        import asyncio
        brain_coroutine = get_brain(graph_version_id)
        brain = await brain_coroutine if asyncio.iscoroutine(brain_coroutine) else brain_coroutine
        
        # Return stats for verification
        return {
            "success": True,
            "graph_version_id": get_current_graph_version(),
            "loaded": is_brain_loaded(),
            "vector_count": brain.faiss_index.ntotal if (brain and hasattr(brain, 'faiss_index')) else 0
        }
    else:
        return {
            "success": False,
            "error": "Failed to load brain vectors"
        }

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
    
    # Check if we should skip disk loading (e.g., during active vector loading)
    if hasattr(_current_config, "_skip_disk_load") and _current_config["_skip_disk_load"]:
        print("Skipping load from disk due to active vector loading")
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
    global _brain_instance, _brain_loaded, _current_config
    import time
    import numpy as np
    import asyncio
    
    start_time = time.time()
    
    print(f"Starting brain vector loading process [time: {start_time}]")
    
    # Add a flag to skip loading from disk during active vector loading
    _current_config["_skip_disk_load"] = True
    
    # Set the graph version if provided and different from current
    version_changed = False
    if graph_version_id:
        current = get_current_graph_version()
        if current != graph_version_id:
            print(f"Setting graph version from {current} to {graph_version_id}")
            version_changed = set_graph_version(graph_version_id)
            print(f"Version changed: {version_changed}")
    
    # Always ensure we have a clean state before loading vectors
    # This prevents old vectors from being used if the load fails
    if _brain_instance is not None:
        print("Clearing the brain state before loading new vectors")
        # Explicitly release large data structures if they exist
        if hasattr(_brain_instance, 'vectors') and _brain_instance.vectors is not None:
            print(f"Clearing existing vectors array with shape: {_brain_instance.vectors.shape if hasattr(_brain_instance.vectors, 'shape') else 'unknown'}")
            _brain_instance.vectors = None
        
        if hasattr(_brain_instance, 'metadata') and _brain_instance.metadata is not None:
            print(f"Clearing existing metadata (size: {len(_brain_instance.metadata) if isinstance(_brain_instance.metadata, dict) else 'unknown'})")
            _brain_instance.metadata = {}
        
        if hasattr(_brain_instance, 'vector_ids') and _brain_instance.vector_ids is not None:
            print(f"Clearing existing vector IDs (count: {len(_brain_instance.vector_ids)})")
            _brain_instance.vector_ids = []
        
        # Reset the FAISS index if it exists
        if hasattr(_brain_instance, 'faiss_index') and _brain_instance.faiss_index is not None:
            try:
                print(f"Resetting existing FAISS index with {_brain_instance.faiss_index.ntotal} vectors")
                _brain_instance.faiss_index.reset()
                print("FAISS index has been reset")
            except Exception as reset_error:
                print(f"Error resetting FAISS index: {reset_error}")
                # If we can't reset, try to recreate it
                try:
                    import faiss
                    print("Creating new FAISS index")
                    _brain_instance.faiss_index = faiss.IndexFlatIP(_brain_instance.dim)
                except Exception as create_error:
                    print(f"Error creating new FAISS index: {create_error}")
    
    # If already loaded and version didn't change, we might still need to clean files
    if _brain_loaded and not version_changed and not force_delete:
        print("Brain already loaded with current version, skipping reload")
        # Remove disk load skip flag
        _current_config["_skip_disk_load"] = False
        return True
    
    # Clean up existing files if they exist - always remove when loading
    # to ensure we have a clean state
    delete_success = True
    
    # Acquire lock for deleting files
    lock_file = acquire_lock(INDEX_LOCK_PATH)
    if not lock_file:
        print("Could not acquire lock for deleting index files")
        # Remove disk load skip flag
        _current_config["_skip_disk_load"] = False
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
    
    # Set _brain_loaded to False to avoid using cached data
    _brain_loaded = False
    
    # Ensure brain instance exists - CRITICAL FIX: Await the coroutine from get_brain()
    brain_coroutine = get_brain()
    brain = await brain_coroutine if asyncio.iscoroutine(brain_coroutine) else brain_coroutine
    
    if brain is None:
        print("Failed to initialize brain instance")
        # Remove disk load skip flag
        _current_config["_skip_disk_load"] = False
        return False
    
    try:
        # Get the current graph version to load
        current_version = get_current_graph_version()
        if not current_version:
            print("No graph version set, cannot load vectors")
            # Remove disk load skip flag
            _current_config["_skip_disk_load"] = False
            return False
            
        print(f"Loading vectors for graph version: {current_version}")
        
        # Load vectors with timeout handling
        try:
            import asyncio
            # Set a reasonable timeout for loading (5 minutes)
            # CRITICAL FIX: Ensure we're calling the method on the brain instance, not on a coroutine
            if asyncio.iscoroutine(brain.load_all_vectors_from_graph_version):
                # This shouldn't happen if get_brain() was properly awaited above
                print("WARNING: brain.load_all_vectors_from_graph_version is a coroutine - this indicates an error")
                load_task = asyncio.create_task(brain.load_all_vectors_from_graph_version(current_version))
            else:
                # Normal case - the method returns a coroutine which we then create a task from
                load_task = asyncio.create_task(brain.load_all_vectors_from_graph_version(current_version))
                
            await asyncio.wait_for(load_task, timeout=300)  # 5 minute timeout
        except asyncio.TimeoutError:
            print("Vector loading timed out after 5 minutes")
            # Remove disk load skip flag
            _current_config["_skip_disk_load"] = False
            return False
        except Exception as e:
            import traceback
            print(f"ERROR during vector loading: {e}")
            print(traceback.format_exc())
            # Remove disk load skip flag
            _current_config["_skip_disk_load"] = False
            return False
        
        # Check if vectors were loaded
        vector_count = brain.faiss_index.ntotal if hasattr(brain, 'faiss_index') else 0
        if vector_count == 0:
            print("No vectors loaded into FAISS index")
            
            # Set _brain_loaded to False as we have no vectors
            _brain_loaded = False
            
            # Save the empty state to disk
            try:
                import faiss
                faiss.write_index(brain.faiss_index, FAISS_INDEX_PATH)
                with open(METADATA_PATH, 'wb') as f:
                    pickle.dump({}, f)
                print("Saved empty state to disk")
            except Exception as save_error:
                print(f"Error saving empty state: {save_error}")
            
            # Remove disk load skip flag
            _current_config["_skip_disk_load"] = False    
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
            results = brain.get_similar_vectors(test_vector, top_k=10, threshold=0.0)
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
        
        # Remove disk load skip flag
        _current_config["_skip_disk_load"] = False
        return True
    except Exception as e:
        import traceback
        print(f"ERROR loading vectors: {e}")
        print(traceback.format_exc())
        # Remove disk load skip flag
        _current_config["_skip_disk_load"] = False
        return False

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

def json_serialize_for_brain(obj):
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
        return [json_serialize_for_brain(item) for item in obj]
        
    # Handle dictionaries
    if isinstance(obj, dict):
        return {str(key): json_serialize_for_brain(value) for key, value in obj.items()}
        
    # Special handling for custom objects with __dict__
    if hasattr(obj, '__dict__'):
        return {key: json_serialize_for_brain(value) for key, value in obj.__dict__.items() 
                if not key.startswith('_')}
                
    # Try to make it JSON serializable, or convert to string as last resort
    try:
        json.dumps(obj)
        return obj
    except (TypeError, OverflowError):
        return str(obj) 

async def flick_out(input_text: str = "", graph_version_id: str = "") -> dict:
    """
    Return vectors from brain without printing results.
    
    Args:
        input_text: Text to search for similar vectors
        graph_version_id: Optional graph version to use
        
    Returns:
        A dictionary with query results, ready for JSON serialization
    """
    # Thread-safe check for graph version
    with _version_check_lock:
        current_version = _brain_graph_version
        version_change_needed = graph_version_id and graph_version_id != current_version
    
    # If a new graph version is requested, update the configuration
    if version_change_needed:
        # Set the new graph version which will reset the brain if needed
        set_graph_version(graph_version_id)
    
    # Get the brain instance
    brain_coroutine = get_brain(graph_version_id)
    
    # Properly handle coroutines - check and await if needed
    import asyncio
    brain = await brain_coroutine if asyncio.iscoroutine(brain_coroutine) else brain_coroutine
    
    # Ensure brain is loaded before querying
    with _brain_lock:
        brain_loaded = is_brain_loaded()
        has_index = brain and hasattr(brain, 'faiss_index')
        vectors_loaded = has_index and brain.faiss_index.ntotal > 0
    
    if not brain_loaded or not has_index or not vectors_loaded:
        # Try to load the brain
        with _version_check_lock:
            current_version = _brain_graph_version
        
        success = await load_brain_vectors(current_version, force_delete=True)
        if not success:
            return {
                "error": "Failed to load brain",
                "query": input_text,
                "graph_version_id": get_current_graph_version()
            }
    
    # Now perform the query
    try:
        # Check if the method returns a coroutine
        result_or_coroutine = brain.get_similar_vectors_by_text(input_text, top_k=10)
        
        # Handle coroutine if necessary
        results = await result_or_coroutine if asyncio.iscoroutine(result_or_coroutine) else result_or_coroutine
        
        # Convert results to a serializable format
        formatted_results = []
        for vector_id, vector, metadata, similarity in results:
            formatted_results.append({
                "vector_id": vector_id,
                "similarity": similarity,
                "metadata": metadata,
                "vector_preview": vector[:10] if hasattr(vector, '__getitem__') else []
            })
        
        with _version_check_lock:
            current_version = _brain_graph_version
        
        with _brain_lock:
            vector_count = brain.faiss_index.ntotal if hasattr(brain, 'faiss_index') else 0
        
        response = {
            "query": input_text,
            "graph_version_id": current_version,
            "vector_count": vector_count,
            "results": formatted_results
        }
        
        # Make sure everything is JSON serializable
        return json_serialize_for_brain(response)
    except Exception as e:
        import traceback
        print(f"Error in flick_out: {e}")
        print(traceback.format_exc())
        
        with _version_check_lock:
            current_version = _brain_graph_version
            
        return {
            "error": f"Error processing query: {str(e)}",
            "query": input_text,
            "graph_version_id": current_version
        } 