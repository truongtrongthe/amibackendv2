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
                    # Use a much lower similarity threshold (even 0.1 would be reasonable for knowledge retrieval)
                    # since we want to prioritize having some context over having none
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
            
            knowledge_context = prepare_knowledge(
                selected_entries,
                query,
                is_profiling=True
            )
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
            
            # Check if raw text appears to be vector data or arrays of IDs
            if raw_text and isinstance(raw_text, str):
                # Filter out content that looks like vector data
                if (raw_text.startswith("[") and "]" in raw_text and 
                    ("..." in raw_text or 
                     any(x in raw_text for x in ["-0.", "0.", "1.", "2."]) or
                     "_tfl-user_" in raw_text)):  # Also filter arrays of IDs
                    
                    logger.warning(f"Raw content appears to be vector data or ID array, ignoring it: {raw_text[:50]}...")
                    raw_text = ""
            
            if not raw_text:
                # Try to construct raw text from other fields
                title = entry.get("title", "")
                description = entry.get("description", "")
                content = entry.get("content", "")
                
                # Check if these fields also contain vector data
                fields_to_check = {"title": title, "description": description, "content": content}
                for field_name, field_value in fields_to_check.items():
                    if isinstance(field_value, str) and field_value.startswith("[") and "]" in field_value:
                        if ("..." in field_value or 
                            any(x in field_value for x in ["-0.", "0.", "1.", "2."]) or
                            "_tfl-user_" in field_value):
                            
                            logger.warning(f"Field {field_name} contains vector data, clearing it: {field_value[:50]}...")
                            if field_name == "title":
                                title = "Untitled"
                            elif field_name == "description":
                                description = ""
                            elif field_name == "content":
                                content = ""
                
                # Try other fields if main ones contain vector data
                if not (title.strip() or description.strip() or content.strip()):
                    # Look through all fields for usable text content
                    for key, value in entry.items():
                        if key not in ["raw", "title", "description", "content", "id", "similarity", "query"] and isinstance(value, str):
                            if not (value.startswith("[") and "]" in value and 
                                   ("..." in value or any(x in value for x in ["-0.", "0.", "1.", "2."]))):
                                if key == "data":
                                    content = value
                                else:
                                    content += f"{key.capitalize()}: {value}\n"
                
                raw_text = f"{title}\n{description}\n{content}".strip()
                
                # If still no content, provide a fallback
                if not raw_text:
                    raw_text = f"No useful content available for query: {query}"
            
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
        # Clean up title if it's a vector representation
        if isinstance(title, str) and title.startswith("[") and "]" in title:
            if "..." in title or any(x in title for x in ["-0.", "0.", "1.", "2."]):
                title = "Untitled"
                
        formatted_output.append(f"KNOWLEDGE ENTRY {i}:\nTitle: {title}\n\n{raw_text}\n----\n")
    
    return "\n".join(formatted_output)

def emit_analysis_event(thread_id: str, data: Dict[str, Any]) -> bool:
    """
    Emit an analysis event to all clients in a thread room
    
    Args:
        thread_id: The thread ID to send the event to
        data: The analysis event data to send
        
    Returns:
        bool: True if message was delivered to active sessions, False otherwise
    """
    try:
        from socketio_manager import emit_analysis_event as socket_emit
        return socket_emit(thread_id, data)
    except ImportError:
        logger.error("socketio_manager not available for analysis event emission")
        return False
    except Exception as e:
        logger.error(f"Error emitting analysis event: {str(e)}")
        return False

def emit_knowledge_event(thread_id: str, data: Dict[str, Any]) -> bool:
    """
    Emit a knowledge event to all clients in a thread room
    
    Args:
        thread_id: The thread ID to send the event to
        data: The knowledge event data to send
        
    Returns:
        bool: True if message was delivered to active sessions, False otherwise
    """
    try:
        from socketio_manager import emit_knowledge_event as socket_emit
        return socket_emit(thread_id, data)
    except ImportError:
        logger.error("socketio_manager not available for knowledge event emission")
        return False
    except Exception as e:
        logger.error(f"Error emitting knowledge event: {str(e)}")
        return False

def emit_next_action_event(thread_id: str, data: Dict[str, Any]) -> bool:
    """
    Emit a next_action event to all clients in a thread room
    
    Args:
        thread_id: The thread ID to send the event to
        data: The next_action event data to send
        
    Returns:
        bool: True if message was delivered to active sessions, False otherwise
    """
    try:
        from socketio_manager import emit_next_action_event as socket_emit
        return socket_emit(thread_id, data)
    except ImportError:
        logger.error("socketio_manager not available for next action event emission")
        return False
    except Exception as e:
        logger.error(f"Error emitting next action event: {str(e)}")
        return False

async def save_knowledge(query: str, knowledge: str, graph_version_id: str = "") -> bool:
    """
    Save knowledge to the knowledge graph
    
    Args:
        query: The query that was used to retrieve the knowledge
        knowledge: The knowledge to save
        graph_version_id: The graph version ID to save the knowledge to
        
    Returns:
        bool: True if knowledge was saved, False otherwise
    """
    try:
        from database import save_training_with_chunk
        return save_training_with_chunk(query, knowledge, graph_version_id)
    except Exception as e:
        logger.error(f"Error saving knowledge: {str(e)}")
        return False

