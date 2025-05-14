from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from utilities import logger
from brain_singleton import get_brain_sync, get_brain
import json
import re
import time
import asyncio
import traceback
from tool_helpers import ensure_brain_loaded

# Get access to brain for the module scope - will be replaced when using it inside async functions
brain = get_brain_sync()


async def fetch_knowledge_with_similarity(query: str, graph_version_id: str = "", state: Optional[Dict] = None) -> Dict:
    """
    A specialized version of fetch_knowledge that includes similarity scores in the output.
    This is specifically for tool_learning.py to avoid changing behavior for other tools.
    
    Args:
        query: The query to search for
        graph_version_id: The graph version ID to search in
        state: Optional state dictionary
        
    Returns:
        String with knowledge entries and their similarity scores
    """
    try:
        # Skip knowledge retrieval if no graph_version_id provided
        if not graph_version_id:
            logger.info(f"No graph version ID provided")
            return ""
        if await ensure_brain_loaded(graph_version_id):
            # Get brain with the correct graph version - this is important for async operations
            brain = await get_brain(graph_version_id)
            if not brain:
                logger.error("Failed to get brain instance")
                return ""
        
            knowledge_entries = []
                
            logger.info("Fetching knowledge on user analysis techniques")
            
            try:
                # Use a lower threshold to allow more results
                results = await brain.get_similar_vectors_by_text(query, top_k=5, threshold=0.0)
                
                # Check if we got any results
                if not results:
                    logger.info(f"No vectors found for query: {query}")
                
                for vector_id, vector, metadata, similarity in results:
                    # Use a much lower similarity threshold for knowledge retrieval
                    if similarity < 0.35:
                        logger.info(f"Low similarity {similarity} for vector {vector_id}, but keeping it to provide context")
                            
                    if any(entry.get("id") == vector_id for entry in knowledge_entries):
                        logger.info(f"Skipping duplicate knowledge entry: {vector_id}")
                        continue
                    
                    # Handle the case where metadata is a numpy.ndarray instead of a dictionary
                    if isinstance(metadata, dict):
                        raw_text = metadata.get("raw", "")
                    else:
                        logger.warning(f"Metadata for vector {vector_id} is not a dictionary but {type(metadata)}. Converting to empty string.")
                        raw_text = ""
                    
                    # If raw_text is empty but we have some kind of metadata, use it anyway
                    if not raw_text and metadata:
                        if isinstance(metadata, dict):
                            # Try to extract any useful content from metadata
                            for key in ["content", "text", "data"]:
                                if key in metadata and metadata[key]:
                                    raw_text = str(metadata[key])
                                    break
                            
                            # If still no content, try to serialize the entire metadata
                            if not raw_text:
                                try:
                                    raw_text = json.dumps(metadata, ensure_ascii=False)
                                except:
                                    raw_text = str(metadata)
                        else:
                            # Try to convert non-dict metadata to string
                            raw_text = str(metadata)
                            
                    knowledge_entries.append({
                                "id": vector_id,
                                "query": query,
                                "raw": raw_text,
                                "similarity": float(similarity)
                            })
            except Exception as e:
                    logger.warning(f"Error retrieving knowledge: {query}, {str(e)}")
                    logger.warning(traceback.format_exc())
             
        # Process knowledge entries
        if knowledge_entries:
            sorted_entries = sorted(knowledge_entries, key=lambda x: (-x.get("priority", 0), -x.get("similarity", 0)))
            selected_entries = sorted_entries[:5]  # Take top 5 entries
            
            # Calculate the top similarity score for logging
            top_similarity = max([entry.get("similarity", 0.0) for entry in selected_entries], default=0.0)
            logger.info(f"Found {len(selected_entries)} valid matches. Top similarity: {top_similarity:.4f}")
            
            # Use the specialized knowledge formatter that includes similarity scores
            knowledge_context = prepare_knowledge_with_similarity(
                selected_entries,
                query,
                top_similarity
            )
        else:
            knowledge_context = ""
            logger.info(f"No knowledge entries found for user profiling")
        
        return knowledge_context
        
    except Exception as e:
        logger.error(f"Error in knowledge retrieval: {str(e)}\n{traceback.format_exc()}")
        return {
            "status": "error",
            "message": str(e)
        }

def prepare_knowledge_with_similarity(knowledge_entries: List[Dict[str, Any]], query: str, top_similarity: float = 0.0) -> str:
    """
    Specialized version of prepare_knowledge that includes similarity scores in the output.
    This is specifically for tool_learning.py to avoid changing behavior for other tools.
    
    Args:
        knowledge_entries: List of knowledge entries to format
        query: Original query
        top_similarity: The highest similarity score found
        
    Returns:
        Formatted string with knowledge entries and their similarity scores
    """
    if not knowledge_entries:
        return f"No relevant knowledge found. Query: {query}"
    
    # Include the top similarity score at the beginning of the output
    formatted_output = [f"Found {len(knowledge_entries)} valid matches. Top similarity: {top_similarity:.4f}"]
    
    # Format each knowledge entry
    for i, entry in enumerate(knowledge_entries, 1):
        title = entry.get("title", "Untitled")
        raw_text = entry.get("raw", "")
        similarity = entry.get("similarity", 0.0)
        
        # Clean up title if it's a vector representation
        if isinstance(title, str) and title.startswith("[") and "]" in title:
            if "..." in title or any(x in title for x in ["-0.", "0.", "1.", "2."]):
                title = "Untitled"
        
        # Format the entry with similarity score
        entry_text = f"KNOWLEDGE ENTRY {i}:\nTitle: {title}\nSimilarity: {similarity:.4f}\n\n{raw_text}\n----\n"
        formatted_output.append(entry_text)
    
    return "\n".join(formatted_output)
    