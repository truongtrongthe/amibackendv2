from active_brain import ActiveBrain

# Global brain instance - initially None
_brain_instance = None

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
    global _brain_instance
    _brain_instance = None

def set_graph_version(graph_version_id):
    """
    Set a new graph version ID for the brain.
    
    Args:
        graph_version_id: The new graph version ID to use
        
    Returns:
        bool: True if a new brain instance was created, False if the graph version was already set
    """
    global _brain_instance, _current_config
    
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