from typing import List, Dict, Any, Optional
from utilities import logger
import json
import asyncio

async def emit_analysis_event(thread_id: str, data: Dict[str, Any]) -> bool:
    """
    Emit an analysis event - fallback version that works with both Flask and FastAPI
    Returns True if the event was delivered, False otherwise
    """
    try:
        # Add thread_id to data if not present
        if "thread_id" not in data:
            data["thread_id"] = thread_id
            
        # Try to import from main (FastAPI version)
        from main import emit_analysis_event as async_emit
        # Directly await the async function since we're now async
        return await async_emit(thread_id, data)
    except ImportError:
        try:
            # Fallback to socketio_manager (Flask version) if available
            from socketio_manager import emit_analysis_event as socket_emit
            return socket_emit(thread_id, data)
        except ImportError:
            # Last resort fallback - just log
            logger.error(f"Could not emit analysis event - no emit function found")
            logger.info(f"Would have emitted to {thread_id}: {str(data)[:100]}...")
            return False
    except Exception as e:
        logger.error(f"Error in emit_analysis_event: {str(e)}")
        return False

async def emit_knowledge_event(thread_id: str, data: Dict[str, Any]) -> bool:
    """
    Emit a knowledge event - fallback version that works with both Flask and FastAPI
    Returns True if the event was delivered, False otherwise
    """
    try:
        # Add thread_id to data if not present
        if "thread_id" not in data:
            data["thread_id"] = thread_id
            
        # Try to import from main (FastAPI version)
        from main import emit_knowledge_event as async_emit
        # Directly await the async function since we're now async
        return await async_emit(thread_id, data)
    except ImportError:
        try:
            # Fallback to socketio_manager (Flask version) if available
            from socketio_manager import emit_knowledge_event as socket_emit
            return socket_emit(thread_id, data)
        except ImportError:
            # Last resort fallback - just log
            logger.error(f"Could not emit knowledge event - no emit function found")
            logger.info(f"Would have emitted to {thread_id}: {str(data)[:100]}...")
            return False
    except Exception as e:
        logger.error(f"Error in emit_knowledge_event: {str(e)}")
        return False

async def emit_next_action_event(thread_id: str, data: Dict[str, Any]) -> bool:
    """
    Emit a next_action event - fallback version that works with both Flask and FastAPI
    Returns True if the event was delivered, False otherwise
    """
    try:
        # Add thread_id to data if not present
        if "thread_id" not in data:
            data["thread_id"] = thread_id
            
        # Try to import from main (FastAPI version)
        from main import emit_next_action_event as async_emit
        # Directly await the async function since we're now async
        return await async_emit(thread_id, data)
    except ImportError:
        try:
            # Fallback to socketio_manager (Flask version) if available
            from socketio_manager import emit_next_action_event as socket_emit
            return socket_emit(thread_id, data)
        except ImportError:
            # Last resort fallback - just log
            logger.error(f"Could not emit next_action event - no emit function found")
            logger.info(f"Would have emitted to {thread_id}: {str(data)[:100]}...")
            return False
    except Exception as e:
        logger.error(f"Error in emit_next_action_event: {str(e)}")
        return False

