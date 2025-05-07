from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from utilities import logger
from brain_singleton import get_brain, set_graph_version, is_brain_loaded, load_brain_vectors, get_current_graph_version
import json
import re
import time
import asyncio
import traceback
from tool_helpers import ensure_brain_loaded

# Access brain singleton
brain = get_brain()


async def fetch_knowledge(query: str, graph_version_id: str = "", state: Optional[Dict] = None) -> Dict:
    """
    Build a knowledge context  
        
    Returns:
        Dictionary with the user portrait as a string and metadata
    """
    try:
         
        # Skip knowledge retrieval if no graph_version_id provided
        if not graph_version_id:
            logger.info(f"No graph version ID provided")
            return ""
        if await ensure_brain_loaded(graph_version_id):
        
            knowledge_entries = []
                
            logger.info("Fetching knowledge on user analysis techniques")
            
            try:
                results = await brain.get_similar_vectors_by_text(query, top_k=5)
                for vector_id, _, metadata, similarity in results:
                    if similarity < 0.4:
                        continue
                            
                    if any(entry.get("id") == vector_id for entry in knowledge_entries):
                        logger.info(f"Skipping duplicate knowledge entry: {vector_id}")
                        continue
                            
                    raw_text = metadata.get("raw", "")
                        
                    knowledge_entries.append({
                                "id": vector_id,
                                "query": query,
                                "raw": raw_text,
                                "similarity": float(similarity)
                            })
            except Exception as e:
                    logger.warning(f"Error retrieving knowledge: {query}, {str(e)}")
             
        # Process knowledge entries
        if knowledge_entries:
            sorted_entries = sorted(knowledge_entries, key=lambda x: (-x.get("priority", 0), -x.get("similarity", 0)))
            selected_entries = sorted_entries[:5]  # Take top 3 entries
            logger.info(f"Selected knowledge entries for profiling: {len(selected_entries)}")
            
            knowledge_context = prepare_knowledge(
                selected_entries,
                query,
                is_profiling=True
            )
            logger.info(f"Knowledge For Profiling: {knowledge_context}")
        else:
            knowledge_context = ""
            logger.info(f"No knowledge entries found for user profiling")
        
        return knowledge_context
        
    except Exception as e:
        logger.error(f"Error in user portrait creation: {str(e)}\n{traceback.format_exc()}")
        return {
            "status": "error",
            "message": str(e)
        }


def prepare_knowledge(knowledge_entries: List[Dict[str, Any]], query: str, is_profiling: bool = False) -> str:
    """Prepare knowledge entries for LLM synthesis."""
    if not knowledge_entries:
        return "No relevant knowledge found."
    
    # Lower threshold for profiling queries
    similarity_threshold = 0.2 if is_profiling else 0.4
    
    # Process and rank entries
    ranked_entries = []
    for entry in knowledge_entries:
        try:
            # Get raw text with fallback
            raw_text = entry.get("raw", "")
            if not raw_text:
                # Try to construct raw text from other fields
                title = entry.get("title", "")
                description = entry.get("description", "")
                content = entry.get("content", "")
                raw_text = f"{title}\n{description}\n{content}".strip()
            
            # Calculate relevance score
            similarity = entry.get("similarity", 0.0)
            query_match = query.lower() in raw_text.lower()
            relevance_score = similarity * 0.7 + (1.0 if query_match else 0.0) * 0.3
            
            if relevance_score >= similarity_threshold:
                ranked_entries.append((entry, relevance_score, raw_text))
        except Exception as e:
            logger.error(f"Error processing knowledge entry: {str(e)}")
            continue
    
    # Sort by relevance
    ranked_entries.sort(key=lambda x: x[1], reverse=True)
    
    # Select top entries
    top_entries = ranked_entries[:5]  # Increased from 3 to 5 for profiling
    
    if not top_entries:
        return "No sufficiently relevant knowledge found."
    
    # Format output
    formatted_output = []
    for i, (entry, score, raw_text) in enumerate(top_entries, 1):
        title = entry.get("title", "Untitled")
        formatted_output.append(f"KNOWLEDGE ENTRY {i}:\nTitle: {title}\n\n{raw_text}\n----\n")
    
    return "\n".join(formatted_output)
