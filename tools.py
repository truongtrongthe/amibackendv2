from typing import List, Dict, Any, Optional
from utilities import logger
import json
import asyncio

def emit_analysis_event(thread_id: str, data: Dict[str, Any]) -> bool:
    """
    Emit an analysis event - fallback version that works with both Flask and FastAPI
    Returns True if the event was delivered, False otherwise
    """
    try:
        # Try to import from main (FastAPI version)
        from main import emit_analysis_event as async_emit
        # Get or create an event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Run the async function
        if loop.is_running():
            # Create a future to get the result
            fut = asyncio.run_coroutine_threadsafe(async_emit(thread_id, data), loop)
            return fut.result(timeout=5)  # 5 second timeout
        else:
            return loop.run_until_complete(async_emit(thread_id, data))
    except ImportError:
        try:
            # Fallback to socketio_manager (Flask version) if available
            from socketio_manager import emit_analysis_event as socket_emit
            return socket_emit(thread_id, data)
        except ImportError:
            # Last resort fallback - just log
            import logging
            logging = logging.getLogger(__name__)
            logging.error(f"Could not emit analysis event - no emit function found")
            logging.info(f"Would have emitted to {thread_id}: {str(data)[:100]}...")
            return False

def emit_knowledge_event(thread_id: str, data: Dict[str, Any]) -> bool:
    """
    Emit a knowledge event - fallback version that works with both Flask and FastAPI
    Returns True if the event was delivered, False otherwise
    """
    try:
        # Try to import from main (FastAPI version)
        from main import emit_knowledge_event as async_emit
        # Get or create an event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Run the async function
        if loop.is_running():
            # Create a future to get the result
            fut = asyncio.run_coroutine_threadsafe(async_emit(thread_id, data), loop)
            return fut.result(timeout=5)  # 5 second timeout
        else:
            return loop.run_until_complete(async_emit(thread_id, data))
    except ImportError:
        try:
            # Fallback to socketio_manager (Flask version) if available
            from socketio_manager import emit_knowledge_event as socket_emit
            return socket_emit(thread_id, data)
        except ImportError:
            # Last resort fallback - just log
            import logging
            logging = logging.getLogger(__name__)
            logging.error(f"Could not emit knowledge event - no emit function found")
            logging.info(f"Would have emitted to {thread_id}: {str(data)[:100]}...")
            return False

def emit_next_action_event(thread_id: str, data: Dict[str, Any]) -> bool:
    """
    Emit a next_action event - fallback version that works with both Flask and FastAPI
    Returns True if the event was delivered, False otherwise
    """
    try:
        # Try to import from main (FastAPI version)
        from main import emit_next_action_event as async_emit
        # Get or create an event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Run the async function
        if loop.is_running():
            # Create a future to get the result
            fut = asyncio.run_coroutine_threadsafe(async_emit(thread_id, data), loop)
            return fut.result(timeout=5)  # 5 second timeout
        else:
            return loop.run_until_complete(async_emit(thread_id, data))
    except ImportError:
        try:
            # Fallback to socketio_manager (Flask version) if available
            from socketio_manager import emit_next_action_event as socket_emit
            return socket_emit(thread_id, data)
        except ImportError:
            # Last resort fallback - just log
            import logging
            logging = logging.getLogger(__name__)
            logging.error(f"Could not emit next_action event - no emit function found")
            logging.info(f"Would have emitted to {thread_id}: {str(data)[:100]}...")
            return False

