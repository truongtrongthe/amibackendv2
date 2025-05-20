#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import threading
import queue
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def run_async_in_thread(async_func, *args, **kwargs):
    """
    Run an async function in a separate thread with its own event loop,
    or in the current event loop if one is already running.
    Returns the result of the async function.
    Handles exceptions by propagating them back to the caller.
    
    Args:
        async_func: The async function to run
        *args, **kwargs: Arguments to pass to the async function
    
    Returns:
        The result of the async function
    """
    # First, check if we're already in an event loop
    try:
        current_loop = asyncio.get_event_loop()
        if current_loop.is_running():
            logger.info("Event loop already running, using nest_asyncio to run coroutine")
            # If we're already in a running loop, use it directly with nest_asyncio
            return current_loop.run_until_complete(async_func(*args, **kwargs))
    except RuntimeError:
        # No event loop in this thread, we'll create one in a new thread
        logger.info("No event loop in current thread, creating new thread with event loop")
        # No 'pass' statement here to ensure the code continues to the thread creation
    
    # If we get here, we need to create a new thread with its own event loop
    result_queue = queue.Queue()
    
    def thread_target():
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Run the async function and get the result
            result = loop.run_until_complete(async_func(*args, **kwargs))
            # Put the result in the queue
            result_queue.put(result)
        except Exception as e:
            # If there's an exception, put it in the queue
            logger.error(f"Error in async thread: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            result_queue.put(e)
        finally:
            # Always close the loop
            loop.close()
    
    # Start the thread and wait for it to finish
    thread = threading.Thread(target=thread_target)
    thread.start()
    thread.join()
    
    # Get the result (or exception) from the queue
    result = result_queue.get()
    
    # If it's an exception, raise it
    if isinstance(result, Exception):
        raise result
    
    return result 