

# Apply eventlet monkey patch before any other imports
import eventlet
eventlet.monkey_patch()

# Add at the top with other imports
import nest_asyncio
nest_asyncio.apply()  # Apply patch to allow nested event loops

# Then other imports
from flask import Flask, Response, request, jsonify, stream_with_context
import json
from uuid import UUID, uuid4
from datetime import datetime
import time

from flask_cors import CORS
import asyncio
from typing import List, Optional, Dict, Any  # Added List and Optional imports
from collections import deque
import threading
import queue
# Keep track of recent webhook requests to detect duplicates
recent_requests = deque(maxlen=1000)

from supabase import create_client, Client
import os
from flask_socketio import SocketIO, emit, join_room, leave_room
from utilities import logger
import traceback


# Import SocketIO functionality from socketio_manager.py
from socketio_manager import init_socketio
from socketio_manager import (
    socketio, ws_sessions, session_lock, 
    undelivered_messages, message_lock,
)

# Add a dictionary to store locks for each thread_id
thread_locks = {}
thread_locks_lock = threading.RLock()
# Add a lock maintenance mechanism
thread_lock_last_used = {}
thread_lock_cleanup_interval = 300  # 5 minutes

# Add a background thread to clean up unused locks
def start_thread_lock_cleanup():
    """Start a background thread to periodically clean up unused thread locks"""
    def cleanup_worker():
        while True:
            try:
                to_delete = []
                now = time.time()
                
                with thread_locks_lock:
                    # Find locks that haven't been used in the cleanup interval
                    for tid, last_used in thread_lock_last_used.items():
                        if now - last_used > thread_lock_cleanup_interval:
                            # Only delete if not currently owned
                            if tid in thread_locks and not thread_locks[tid]._is_owned():
                                to_delete.append(tid)
                    
                    # Delete the identified locks
                    for tid in to_delete:
                        del thread_locks[tid]
                        del thread_lock_last_used[tid]
                        logger.info(f"Cleaned up unused thread lock for thread_id {tid}")
                
                # Log periodic stats about locks
                with thread_locks_lock:
                    logger.info(f"Thread locks stats: {len(thread_locks)} active locks, {len(to_delete)} cleaned up")
            
            except Exception as e:
                logger.error(f"Error in thread lock cleanup: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
            
            # Sleep for interval
            time.sleep(60)  # Check every minute
    
    # Start the cleanup thread
    thread = threading.Thread(target=cleanup_worker, daemon=True)
    thread.start()
    logger.info("Started thread lock cleanup worker")

# Start the cleanup thread when module loads
start_thread_lock_cleanup()

# Thread-safe function to run async code in separate thread
# Import run_async_in_thread from async_utils
from async_utils import run_async_in_thread

# Load inbox mapping for Chatwoot
INBOX_MAPPING = {}
try:
    with open('inbox_mapping.json', 'r') as f:
        mapping_data = json.load(f)
        # Create a lookup dictionary by inbox_id
        for inbox in mapping_data.get('inboxes', []):
            INBOX_MAPPING[inbox['inbox_id']] = {
                'organization_id': inbox['organization_id'],
                'facebook_page_id': inbox['facebook_page_id'],
                'page_name': inbox['page_name']
            }
    logger.info(f"Loaded {len(INBOX_MAPPING)} inbox mappings")
except Exception as e:
    logger.error(f"Failed to load inbox_mapping.json: {e}")
    logger.warning("Chatwoot webhook will operate without inbox validation")

spb_url = os.getenv("SUPABASE_URL", "https://example.supabase.co")
spb_key = os.getenv("SUPABASE_KEY", "your-supabase-key")

# Add proper error handling for Supabase initialization
try:
    supabase: Client = create_client(spb_url, spb_key)
    logger.info("Supabase client initialized successfully in main.py")
except Exception as e:
    logger.error(f"Failed to initialize Supabase client in main.py: {e}")
    
app = Flask(__name__)

# Add numpy and json imports (needed elsewhere)
import numpy as np
import json

app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'ami_secret_key')

# Simple CORS configuration that was working before
CORS(app, resources={r"/*": {"origins": "*"}})  # Enable CORS for all routes, all origins

# Initialize SocketIO
if socketio is None:
    socketio = init_socketio(app)  # Only initialize if not already done
else:
    print("SocketIO already initialized - using existing instance")


loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

# Add a utility class for processing async requests in threads
class RequestProcessor:
    """
    Utility class to handle async request processing in isolated threads.
    This provides a cleaner way to manage event loop isolation and thread safety.
    """
    
    @staticmethod
    def process_in_thread(processor_func, *args, **kwargs):
        """
        Process a request in a dedicated thread with complete event loop isolation.
        
        Args:
            processor_func: The async function to run in the isolated thread
            *args, **kwargs: Arguments to pass to the processor function
            
        Returns:
            Queue: A queue that will receive response items
        """
        # Create response queue for communication
        response_queue = queue.Queue()
        
        # Define the thread worker
        def thread_worker():
            try:
                # Create a new event loop for this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                # Create a fully isolated execution environment
                def isolated_run():
                    async def async_processor():
                        try:
                            # Run the provided processor function
                            async_gen = processor_func(*args, **kwargs)
                            
                            # Process all items from the generator
                            async for item in async_gen:
                                response_queue.put(item)
                                
                        except Exception as e:
                            logger.error(f"Error in async processing: {str(e)}")
                            logger.error(traceback.format_exc())
                            response_queue.put({"error": f"Processing error: {str(e)}"})
                        finally:
                            # Signal end of stream
                            response_queue.put(None)
                    
                    try:
                        # Run the async processor in this thread's event loop
                        return loop.run_until_complete(async_processor())
                    except Exception as e:
                        logger.error(f"Error in event loop execution: {str(e)}")
                        logger.error(traceback.format_exc())
                        response_queue.put({"error": f"Loop error: {str(e)}"})
                        response_queue.put(None)
                    finally:
                        # Clean up the event loop
                        try:
                            # Cancel pending tasks
                            pending = asyncio.all_tasks(loop)
                            for task in pending:
                                task.cancel()
                            
                            # Wait for tasks to be cancelled
                            if pending:
                                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                            
                            # Close the loop
                            loop.close()
                        except Exception as close_err:
                            logger.error(f"Error closing event loop: {str(close_err)}")
                
                # Run the isolated function
                isolated_run()
                
            except Exception as e:
                logger.error(f"Error in thread execution: {str(e)}")
                logger.error(traceback.format_exc())
                response_queue.put({"error": f"Thread error: {str(e)}"})
                response_queue.put(None)
        
        # Start the worker thread
        threading.Thread(target=thread_worker, daemon=True).start()
        
        # Return the queue for the caller to consume
        return response_queue
    
    @staticmethod
    def stream_response(response_queue, start_time, thread_id=None):
        """
        Create a generator function for streaming responses from a queue.
        
        Args:
            response_queue: Queue containing response items
            start_time: The time when the request started
            thread_id: Optional thread ID for logging
            
        Returns:
            function: A generator function that yields response items
        """
        def generate_response():
            try:
                # Yield items as they become available from the worker thread
                while True:
                    try:
                        # Get next item with timeout to allow for checking if client disconnected
                        item = response_queue.get(timeout=1.0)
                        if item is None:  # Signal for end of stream
                            break
                        
                        # Pass through the string data directly if it's already formatted
                        if isinstance(item, str) and item.startswith("data:"):
                            yield item
                        else:
                            # Yield the item as SSE format
                            yield f"data: {json.dumps(item)}\n\n"
                        
                    except queue.Empty:
                        # No data available yet, check if client is still connected
                        # Use a safer check that works with eventlet
                        if request.environ.get('werkzeug.server.shutdown'):
                            logger.info("Client disconnected, stopping stream")
                            break
                        
                        # Check if the connection is closed (safely)
                        try:
                            # See if connection is still alive without causing exceptions
                            if not request.environ.get('wsgi.input'):
                                logger.info("Client disconnected (no input), stopping stream")
                                break
                        except Exception:
                            # Any exception here likely means the connection is broken
                            logger.info("Client disconnected (error checking), stopping stream")
                            break
                            
                        continue  # Continue waiting for data
                        
                end_time = datetime.now()
                elapsed = (end_time - start_time).total_seconds()
                thread_info = f"for thread {thread_id} " if thread_id else ""
                logger.info(f"[SESSION_TRACE] === END request {thread_info}- total time: {elapsed:.2f}s ===")
                
            except Exception as e:
                error_msg = f"Error in response generator: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                yield f"data: {json.dumps({'error': error_msg})}\n\n"
        
        return generate_response

def handle_options():
    """Common OPTIONS handler for all endpoints."""
    response = Response()
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    response.headers['Access-Control-Max-Age'] = '86400'
    return response, 200

def create_stream_response(gen):
    """
    Common response creator for streaming endpoints.
    Works with both regular generators and async generators.
    """
    # For async generators, we need a special handler that preserves Flask context
    if hasattr(gen, '__aiter__'):
        from flask import copy_current_request_context, current_app
        import asyncio
        
        # Get the current app and request context outside the wrapper
        app = current_app._get_current_object()
        
        # Define a wrapper that will consume the async generator while preserving Flask context
        @copy_current_request_context
        def async_generator_handler():
            # Create a new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # This function will run in the current thread and consume the async generator
            def run_async_generator():
                async def consume_async_generator():
                    try:
                        async for item in gen:
                            yield item
                    except Exception as e:
                        logger.error(f"Error consuming async generator: {str(e)}")
                        import traceback
                        logger.error(traceback.format_exc())
                
                # Create a list to store all generated items
                all_items = []
                
                # Run the async generator to completion and collect all items
                try:
                    coro = consume_async_generator().__aiter__().__anext__()
                    while True:
                        try:
                            item = loop.run_until_complete(coro)
                            all_items.append(item)
                            coro = consume_async_generator().__aiter__().__anext__()
                        except StopAsyncIteration:
                            break
                except Exception as e:
                    logger.error(f"Error in async generator execution: {str(e)}")
                    import traceback
                    logger.error(traceback.format_exc())
                finally:
                    loop.close()
                
                # Return all collected items
                return all_items
            
            # Collect all items first
            try:
                items = run_async_generator()
                
                # Now yield them one by one in the Flask context
                for item in items:
                    yield item
            except Exception as e:
                logger.error(f"Error in async generator handler: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
        
        # Use the context-preserving wrapper
        wrapped_gen = async_generator_handler()
    else:
        # Regular generator can be used directly
        wrapped_gen = gen
    
    # Use Flask's stream_with_context to ensure request context is maintained
    from flask import stream_with_context
    
    # Create the response
    response = Response(stream_with_context(wrapped_gen), mimetype='text/event-stream')
    response.headers['X-Accel-Buffering'] = 'no'
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

# WebSocket events and handlers
@socketio.on('connect')
def handle_connect():
    """Handle new WebSocket connections"""
    session_id = request.sid
    logger.info(f"[SESSION_TRACE] New WebSocket connection: {session_id}, Transport: {request.environ.get('socketio.transport')}")
    # Log more detailed connection information
    #logger.info(f"[SESSION_TRACE] Connection details - Headers: {dict(request.headers)}, Remote addr: {request.remote_addr}")
    emit('connected', {'status': 'connected', 'session_id': session_id})

@socketio.on('register_session')
def handle_register(data):
    """Register a client session with conversation details"""
    session_id = request.sid
    thread_id = data.get('thread_id')
    user_id = data.get('user_id', 'thefusionlab')
    
    if not thread_id:
        thread_id = f"thread_{uuid4()}"
        
    transport_type = request.environ.get('socketio.transport', 'unknown')
    logger.info(f"[SESSION_TRACE] Registering session {session_id} with transport {transport_type}")
    
    # Register this session
    with session_lock:
        ws_sessions[session_id] = {
            'thread_id': thread_id,
            'user_id': user_id,
            'status': 'ready',
            'transport': transport_type,
            'connected_at': datetime.now().isoformat(),
            'last_activity': datetime.now().isoformat()
        }
        # Log all active sessions for the thread
        thread_sessions = [sid for sid, data in ws_sessions.items() if data.get('thread_id') == thread_id]
        logger.info(f"[SESSION_TRACE] Thread {thread_id} now has {len(thread_sessions)} active sessions: {thread_sessions}")
    
    # Join a room based on thread_id for targeted messages
    join_room(thread_id)
    logger.info(f"[SESSION_TRACE] Session {session_id} joined room {thread_id}, user {user_id}")
    
    # Let client know registration was successful
    emit('session_registered', {
        'status': 'ready',
        'thread_id': thread_id,
        'session_id': session_id
    })

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    session_id = request.sid
    # Clean up session data
    with session_lock:
        if session_id in ws_sessions:
            thread_id = ws_sessions[session_id].get('thread_id')
            if thread_id:
                leave_room(thread_id)
            del ws_sessions[session_id]
            #logger.info(f"[SESSION_TRACE] Session {session_id} disconnected and removed from ws_sessions")

# Add a new ping handler to track session activity
@socketio.on('ping')
def handle_ping(data=None):
    """Handle client ping to keep session alive and track activity"""
    session_id = request.sid
    #logger.debug(f"[SESSION_TRACE] Ping received from session {session_id}")
    # Return pong with server timestamp
    return {'pong': datetime.now().isoformat(), 'session_id': session_id}

# Function to emit analysis events to specific thread
def emit_analysis_event(thread_id: str, data: Dict[str, Any]):
    """Emit an analysis event to all clients in a thread room"""
    
    #logger.debug(f"[SESSION_TRACE] Event data preview: {str(data)[:100]}...")
    socketio.emit('analysis_update', data, room=thread_id)

# Function to emit next_action events to specific thread
def emit_next_action_event(thread_id: str, data: Dict[str, Any]):
    """Emit a next_action event to all clients in a thread room"""
    
    #logger.debug(f"[SESSION_TRACE] Next Action event data preview: {str(data)[:100]}...")
    socketio.emit('next_action', data, room=thread_id)


@app.route('/havefun', methods=['POST', 'OPTIONS'])
def havefun():
    """
    Handle havefun requests with complete event loop isolation.
    Each request runs in its own thread with a dedicated event loop using the RequestProcessor.
    """
    if request.method == 'OPTIONS':
        return handle_options()
    
    start_time = datetime.now()
    logger.info(f"[SESSION_TRACE] === BEGIN havefun request at {start_time.isoformat()} ===")
    
    # Parse request data
    data = request.get_json() or {}
    user_input = data.get("user_input", "")
    user_id = data.get("user_id", "thefusionlab")
    thread_id = data.get("thread_id", "chat_thread")
    graph_version_id = data.get("graph_version_id", "")
    use_websocket = data.get("use_websocket", False)
    
    # Define the processor function that will run in the isolated thread
    async def process_convo_request():
        # Import here to avoid circular imports
        from ami import convo_stream
        
        # Set up thread-local lock if needed
        if thread_id:
            # Use the thread_locks mechanism defined in this file
            with thread_locks_lock:
                if thread_id not in thread_locks:
                    thread_locks[thread_id] = threading.RLock()
                thread_lock = thread_locks[thread_id]
                thread_lock_last_used[thread_id] = time.time()
            
            # Try to acquire the lock with timeout
            acquired = thread_lock.acquire(timeout=60)  # 60-second timeout
            
            if not acquired:
                logger.error(f"Could not acquire lock for thread {thread_id} after 60 seconds")
                yield {"error": "Server busy. Please try again."}
                return
            
            logger.info(f"Acquired lock for thread {thread_id}")
        
        try:
            # Set up the conversation stream - it's an async generator
            stream = convo_stream(
                user_input=user_input,
                thread_id=thread_id,
                user_id=user_id,
                graph_version_id=graph_version_id,
                use_websocket=use_websocket,
                thread_id_for_analysis=thread_id
            )
            
            # Process each item
            async for item in stream:
                # Yield each item to be processed by the RequestProcessor
                yield item
                
        finally:
            # Always release the lock when done
            if thread_id:
                thread_lock.release()
                logger.info(f"Released lock for thread {thread_id}")
    
    # Process the request in an isolated thread using the utility class
    response_queue = RequestProcessor.process_in_thread(process_convo_request)
    
    # Create a streaming response with our generator
    response = Response(
        stream_with_context(RequestProcessor.stream_response(response_queue, start_time, thread_id)()),
        mimetype='text/event-stream'
    )
    response.headers['X-Accel-Buffering'] = 'no'
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

@app.route('/conversation/learning', methods=['POST', 'OPTIONS'])
def conversation_learning():
    """
    Handle conversation requests using the learning-based conversation system.
    This endpoint uses tool_learning.py for knowledge similarity checks and active learning.
    """
    if request.method == 'OPTIONS':
        return handle_options()
    
    start_time = datetime.now()
    logger.info(f"[SESSION_TRACE] === BEGIN LEARNING CONVERSATION request at {start_time.isoformat()} ===")
    
    data = request.get_json() or {}
    user_input = data.get("user_input", "")
    user_id = data.get("user_id", "learner")
    thread_id = data.get("thread_id", "learning_thread")
    graph_version_id = data.get("graph_version_id", "")
    use_websocket = data.get("use_websocket", False)
    
    # Get or create a lock for this thread_id
    with thread_locks_lock:
        if thread_id not in thread_locks:
            thread_locks[thread_id] = threading.RLock()
        thread_lock = thread_locks[thread_id]
        thread_lock_last_used[thread_id] = time.time()
    
    # Update session info if using WebSockets
    if use_websocket:
        active_session_exists = False
        
        with session_lock:
            thread_sessions = [sid for sid, data in ws_sessions.items() if data.get('thread_id') == thread_id]
            active_session_exists = len(thread_sessions) > 0            
            if active_session_exists:
                for sid in thread_sessions:
                    ws_sessions[sid]['last_activity'] = datetime.now().isoformat()
                    ws_sessions[sid]['api_request_time'] = datetime.now().isoformat()
                    ws_sessions[sid]['has_pending_request'] = True
    
    # Define the processor function that will run in the isolated thread
    async def process_learning_request():
        # Import directly within the async function to ensure proper event loop isolation
        from ami import convo_stream_learning
        
        # Set up thread-local lock
        if thread_id:
            acquired = thread_lock.acquire(timeout=60)  # 60-second timeout
            
            if not acquired:
                logger.error(f"Could not acquire lock for thread {thread_id} after 60 seconds")
                yield {"error": "Server busy. Please try again."}
                return
            
            logger.info(f"Acquired lock for thread {thread_id}")
        
        try:
            # Prepare stream parameters
            stream_params = {
                "user_input": user_input,
                "user_id": user_id,
                "thread_id": thread_id,
                "graph_version_id": graph_version_id,
                "mode": "learning"
            }
            
            if use_websocket:
                stream_params["use_websocket"] = True
                stream_params["thread_id_for_analysis"] = thread_id
            
            # Get the stream
            stream = convo_stream_learning(**stream_params)
            
            # Process each item
            async for item in stream:
                yield item
                
            # Update thread lock last used time
            with thread_locks_lock:
                thread_lock_last_used[thread_id] = time.time()
            
            # If using WebSockets, update session to indicate request is done
            if use_websocket:
                with session_lock:
                    thread_sessions = [sid for sid, data in ws_sessions.items() if data.get('thread_id') == thread_id]
                    for sid in thread_sessions:
                        if 'has_pending_request' in ws_sessions[sid]:
                            ws_sessions[sid]['has_pending_request'] = False
                
        finally:
            # Always release the lock when done
            if thread_id:
                thread_lock.release()
                logger.info(f"Released lock for thread {thread_id}")
    
    # Process the request in an isolated thread
    response_queue = RequestProcessor.process_in_thread(process_learning_request)
    
    # Create a streaming response with our generator
    response = Response(
        stream_with_context(RequestProcessor.stream_response(response_queue, start_time, thread_id)()),
        mimetype='text/event-stream'
    )
    response.headers['X-Accel-Buffering'] = 'no'
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

# Run the server when executed directly
if __name__ == '__main__':
    
    # Get port from environment or use default
    port = int(os.environ.get('PORT', 5001))
    host = os.environ.get('HOST', '0.0.0.0')
    
    print(f"Starting SocketIO server on {host}:{port}")
    socketio.run(app, host=host, port=port, debug=True)
else:
    
    print("Running in production mode - initializing brain for WSGI workers")
    