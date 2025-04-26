from active_brain import ActiveBrain
import asyncio
import os

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
    
    # Set the graph version if provided and different from current
    version_changed = False
    if graph_version_id:
        current = get_current_graph_version()
        if current != graph_version_id:
            version_changed = set_graph_version(graph_version_id)
    
    # If already loaded and version didn't change, just return success
    if _brain_loaded and not version_changed:
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
    
    try:
        # Get the current graph version to load
        current_version = get_current_graph_version()
        if not current_version:
            return False
            
        # Load vectors
        await brain.load_all_vectors_from_graph_version(current_version)
        
        # Check if vectors were loaded
        vector_count = brain.faiss_index.ntotal
        if vector_count == 0:
            print("No vectors loaded.")
            return False
            
        print(f"Successfully loaded {vector_count} vectors for graph version {current_version}")
        
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