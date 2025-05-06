"""Cache module for profile knowledge"""
from typing import List, Dict, Any
from utilities import logger
import time

# Global cache for profile knowledge
_profile_knowledge_cache = {"entries": [], "timestamp": 0}

def store_profile_knowledge(knowledge_entries: List[Dict]) -> None:
    """
    Store profile knowledge entries in the cache.
    
    Args:
        knowledge_entries: List of knowledge entry dictionaries
    """
    global _profile_knowledge_cache
    
    _profile_knowledge_cache["entries"] = knowledge_entries
    _profile_knowledge_cache["timestamp"] = time.time()
    
    logger.info(f"Stored {len(knowledge_entries)} profile knowledge entries in cache")

def get_profile_knowledge() -> List[Dict]:
    """
    Retrieve cached profile knowledge entries if not expired.
    
    Returns:
        List of knowledge entry dictionaries or empty list if expired
    """
    global _profile_knowledge_cache
    
    # Check for cache expiration (5 minutes)
    cache_age = time.time() - _profile_knowledge_cache["timestamp"]
    if cache_age > 300:  # 5 minutes
        logger.info(f"Profile knowledge cache expired ({cache_age:.1f} seconds old)")
        return []
    
    entries = _profile_knowledge_cache["entries"]
    if entries:
        logger.info(f"Retrieved {len(entries)} profile knowledge entries from cache")
        return entries
    else:
        logger.info("No profile knowledge entries found in cache")
        return []

