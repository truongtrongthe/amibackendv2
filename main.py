#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Apply eventlet monkey patch before any other imports
import eventlet
eventlet.monkey_patch()

# Add at the top with other imports
import nest_asyncio
nest_asyncio.apply()  # Apply patch to allow nested event loops

# SentenceTransformer has to be imported before FAISS
from training_prep import process_document,refine_document
from training_prep_new import understand_document,save_document_insights, understand_cluster
#FAISS importing here
from brain_singleton import init_brain, set_graph_version, load_brain_vectors, is_brain_loaded, get_current_graph_version, get_brain, flick_out, activate_brain_with_version

# Then other imports
from flask import Flask, Response, request, jsonify
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
from threading import Lock
from utilities import logger
from enrich_profile import ProfileEnricher


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

# Store active WebSocket sessions - already imported from socketio_manager
# ws_sessions = {}  
# session_lock = Lock()

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

# Initialize the brain with default configuration at startup
DEFAULT_PINECONE_INDEX = "9well"
#DEFAULT_GRAPH_VERSION = "54cb6723-4752-431b-9395-64936c40ccb9"
DEFAULT_GRAPH_VERSION = "bd5b8bc1-d0cb-4e3e-9cd7-68a8563366fc"
#DEFAULT_GRAPH_VERSION = "ab126ea1-a526-4c52-8a2d-9b6392e44bd8"

# Initialize the brain singleton
init_brain(
    dim=1536, 
    namespace="", 
    graph_version_ids=[DEFAULT_GRAPH_VERSION],
    pinecone_index_name=DEFAULT_PINECONE_INDEX
)

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
    Handle havefun requests using a thread-based approach to avoid Flask/Eventlet event loop conflicts.
    """
    if request.method == 'OPTIONS':
        return handle_options()
    
    start_time = datetime.now()
    logger.info(f"[SESSION_TRACE] === BEGIN havefun request at {start_time.isoformat()} ===")
    
    data = request.get_json() or {}
    user_input = data.get("user_input", "")
    user_id = data.get("user_id", "thefusionlab")
    thread_id = data.get("thread_id", "chat_thread")
    graph_version_id = data.get("graph_version_id", "")
    use_websocket = data.get("use_websocket", False)  # New flag to enable WebSocket

    # Get or create a lock for this thread_id
    thread_lock = None
    with thread_locks_lock:
        if thread_id not in thread_locks:
            thread_locks[thread_id] = threading.RLock()
        thread_lock = thread_locks[thread_id]
        thread_lock_last_used[thread_id] = time.time()
    
    # Update session info if using WebSockets - do this outside the thread lock
    # to minimize lock contention
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
    
    # Prepare stream parameters outside the lock
    stream_params = {
        "user_input": user_input,
        "user_id": user_id,
        "thread_id": thread_id,
        "graph_version_id": graph_version_id,
        "mode": "mc"
    }
    
    if use_websocket:
        stream_params["use_websocket"] = True
        stream_params["thread_id_for_analysis"] = thread_id

    # Function to run the stream processing with proper locking
    async def process_stream_with_lock():
        outputs = []
        # Acquire the thread-specific lock only for the processing part
        with thread_lock:
            try:
                # Import directly here to ensure fresh imports
                from ami import convo_stream
                
                # Define the async process function
                async def process_stream():
                    """Process the stream and return all items."""
                    try:
                        # Get the stream
                        stream = convo_stream(**stream_params)
                        
                        # Process all the output
                        async for item in stream:
                            outputs.append(item)
                    except Exception as e:
                        error_msg = f"Error processing stream: {str(e)}"
                        logger.error(error_msg)
                        import traceback
                        logger.error(traceback.format_exc())
                        outputs.append(f"data: {json.dumps({'error': error_msg})}\n\n")
                    
                    return True
                
                # Run the async function
                await process_stream()
                
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
                
            except Exception as e:
                error_msg = f"Error in stream processing: {str(e)}"
                logger.error(error_msg)
                import traceback
                logger.error(traceback.format_exc())
                outputs.append(f"data: {json.dumps({'error': error_msg})}\n\n")
        
        return outputs
    
    # Create a synchronous response streaming solution that runs the async processing
    def generate_response():
        """Generate streaming response items synchronously."""
        try:
            # Run the async processing with proper locking
            outputs = run_async_in_thread(process_stream_with_lock)
            
            # Yield each output
            for item in outputs:
                yield item
            
            # Log completion
            end_time = datetime.now()
            elapsed = (end_time - start_time).total_seconds()
            logger.info(f"[SESSION_TRACE] === END havefun request - total time: {elapsed:.2f}s ===")
            
        except Exception as outer_e:
            # Handle any errors in the outer function
            error_msg = f"Error in response generator: {str(outer_e)}"
            logger.error(error_msg)
            import traceback
            logger.error(traceback.format_exc())
            yield f"data: {json.dumps({'error': error_msg})}\n\n"
    
    # Create a streaming response with our generator
    from flask import stream_with_context
    response = Response(stream_with_context(generate_response()), mimetype='text/event-stream')
    response.headers['X-Accel-Buffering'] = 'no'
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

@app.route('/activate-brain', methods=['POST', 'OPTIONS'])
def activate_brain():
    """
    Activate the brain with a specific graph version ID.
    This endpoint will load vectors for the specified graph version.
    """
    if request.method == 'OPTIONS':
        return handle_options()
    
    start_time = datetime.now()
    logger.info(f"[SESSION_TRACE] === BEGIN ACTIVATE BRAIN request at {start_time.isoformat()} ===")
    
    data = request.get_json() or {}
    graph_version_id = data.get("graph_version_id", "")
    
    if not graph_version_id:
        return jsonify({"error": "graph_version_id is required"}), 400
    
    try:
        # Call the centralized activate_brain_with_version function
        result = run_async_in_thread(activate_brain_with_version, graph_version_id)
        
        # Log completion
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()
        logger.info(f"[SESSION_TRACE] === END ACTIVATE BRAIN request - total time: {elapsed:.2f}s ===")
        
        if result["success"]:
            return jsonify({
                "message": "Brain activated successfully", 
                "graph_version_id": result["graph_version_id"],
                "loaded": result["loaded"],
                "elapsed_seconds": elapsed,
                "worker_id": "",
                "vector_count": result["vector_count"]
            }), 200
        else:
            return jsonify({"error": result["error"]}), 500
            
    except Exception as e:
        # Handle any errors
        error_msg = f"Error in activate_brain: {str(e)}"
        logger.error(error_msg)
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({"error": error_msg}), 500

@app.route('/examine', methods=['POST', 'OPTIONS'])
def examine_a_brain():
    """
    Handle examine an active brain requests using a thread-based approach to avoid Flask/Eventlet event loop conflicts.
    """
    if request.method == 'OPTIONS':
        return handle_options()
    
    start_time = datetime.now()
    logger.info(f"[SESSION_TRACE] === BEGIN EXAMINE request at {start_time.isoformat()} ===")
    
    data = request.get_json() or {}
    user_input = data.get("user_input", "")
    
    if not user_input:
        return jsonify({"error": "user_input is required"}), 400
        
    try:
        # Use the flick_out function now directly from brain_singleton
        brain_results = run_async_in_thread(flick_out, user_input)
        
        # Log completion
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()
        logger.info(f"[SESSION_TRACE] === END EXAMINE request - total time: {elapsed:.2f}s ===")
        return Response(json.dumps(brain_results), mimetype='application/json'), 200
    except Exception as e:
        # Handle any errors
        error_msg = f"Error in examine_a_brain: {str(e)}"
        logger.error(error_msg)
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({"error": error_msg}), 500

@app.route('/process-document', methods=['POST', 'OPTIONS'])
def process_document_endpoint():
    if request.method == 'OPTIONS':
        return handle_options()

    if 'user_id' not in request.form or 'bank_name' not in request.form or 'reformatted_text' not in request.form:
        return jsonify({"error": "Missing user_id, bank_name, or reformatted_text"}), 400

    user_id = request.form['user_id']
    bank_name = request.form['bank_name']
    reformatted_text = request.form['reformatted_text']
    knowledge_elements = request.form['knowledge_elements'] # Get knowledge_elements from frontend
    mode = request.form.get('mode', 'default')  # Optional mode parameter

    # Debug logging
    logger.info(f"Process document request for bank_name={bank_name}")
    logger.info(f"Knowledge elements provided to MAIN: {len(knowledge_elements)} characters")
    if knowledge_elements and len(knowledge_elements) > 0:
        logger.info(f"First 100 chars of knowledge elements hit at MAIN: {knowledge_elements[:100]}...")
        logger.info(f"Knowledge elements contain 'KEY POINT': {'KEY POINT' in knowledge_elements}")
    logger.debug(f"First 100 chars of reformatted_text: {reformatted_text[:100]}...")
    if not reformatted_text.strip():
        return jsonify({"error": "Empty reformatted_text provided"}), 400

    # Use the thread-based approach instead of the global event loop
    try:
        # Run the async process_document function in a separate thread
        # We only pass the processed text and knowledge elements - no file to avoid reprocessing
        success = run_async_in_thread(
            process_document, 
            text=reformatted_text, 
            user_id=user_id, 
            mode=mode, 
            bank=bank_name,
            knowledge_elements=knowledge_elements  # Pass knowledge_elements to process_document
        )
        
        if success:
            return jsonify({
                "message": "Document processed successfully"
            }), 200
        else:
            return jsonify({
                "error": "Failed to process document"
            }), 500
    except Exception as e:
        logger.error(f"Error in process_document: {str(e)}")
        return jsonify({
            "error": f"Error processing document: {str(e)}"
        }), 500

@app.route('/understand-document', methods=['POST', 'OPTIONS'])
def understand_document_endpoint():
    if request.method == 'OPTIONS':
        return handle_options()

    if 'file' not in request.files:
        return jsonify({"success": False, "error": "Missing file"}), 400

    file = request.files['file']

    if not file.filename:
        return jsonify({"success": False, "error": "No file selected"}), 400
        
    # Determine file type from extension
    file_extension = file.filename.split('.')[-1].lower()
    if file_extension not in ['docx', 'pdf']:
        return jsonify({"success": False, "error": f"Unsupported file type: {file_extension}. Only DOCX and PDF files are supported."}), 400

    # Use the thread-based approach instead of the global event loop
    try:
        # Read file content into BytesIO
        file_content = file.read()
        if not file_content:
            logger.warning(f"Empty file content for {file.filename}")
            return jsonify({"success": False, "error": "Empty file content."}), 400
            
        # Create BytesIO object from file content
        from io import BytesIO
        file_bytes = BytesIO(file_content)
        
        # Log file info for debugging
        logger.info(f"Processing file '{file.filename}' ({len(file_content)} bytes) as {file_extension}")
        
        # Run the async understand_document function in a separate thread
        result = run_async_in_thread(
            understand_document, 
            input_source=file_bytes,  # Pass BytesIO object
            file_type=file_extension  # Pass detected file type
        )
        
        # Validate result structure
        if not isinstance(result, dict):
            logger.error(f"Invalid result type from understand_document: {type(result)}")
            return jsonify({
                "success": False,
                "error": "Document processing returned invalid data structure",
                "error_type": "ProcessingError"
            }), 500
        
        # Check for success
        if result.get("success", False):
            # Validate presence of document_insights
            if "document_insights" not in result:
                logger.warning("Missing document_insights in successful result")
                result["document_insights"] = {}
                
            # Log success statistics
            insights = result["document_insights"]
            logger.info(f"Document processed successfully: {insights.get('metadata', {}).get('sentence_count', 0)} sentences, "
                        f"{insights.get('metadata', {}).get('cluster_count', 0)} clusters")
            
            # Return the document insights with proper structure
            return jsonify(result), 200
        else:
            error_msg = result.get("error", "Failed to process document")
            error_type = result.get("error_type", "UnknownError")
            logger.error(f"Document processing failed: {error_type}: {error_msg}")
            return jsonify({
                "success": False,
                "error": error_msg,
                "error_type": error_type
            }), 500
    except Exception as e:
        logger.error(f"Error in understand_document: {type(e).__name__}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": f"Error processing document: {str(e)}",
            "error_type": type(e).__name__
        }), 500

@app.route('/save-document-insights', methods=['POST', 'OPTIONS'])
def save_document_insights_endpoint():
    if request.method == 'OPTIONS':
        return handle_options()

    # Check for data in JSON format or form data
    if request.is_json:
        # Process JSON request
        if 'document_insight' not in request.json:
            return jsonify({"success": False, "error": "Missing document_insight in request body"}), 400
            
        if 'user_id' not in request.json:
            return jsonify({"success": False, "error": "Missing user_id in request body"}), 400
            
        if 'bank_name' not in request.json:
            return jsonify({"success": False, "error": "Missing bank_name in request body"}), 400
            
        document_insight = request.json['document_insight']
        user_id = request.json['user_id']
        bank_name = request.json['bank_name']
        mode = request.json.get('mode', 'default')
    else:
        # Process form data
        if 'document_insight' not in request.form:
            return jsonify({"success": False, "error": "Missing document_insight in form data"}), 400
            
        if 'user_id' not in request.form:
            return jsonify({"success": False, "error": "Missing user_id in form data"}), 400
            
        if 'bank_name' not in request.form:
            return jsonify({"success": False, "error": "Missing bank_name in form data"}), 400
            
        document_insight = request.form['document_insight']
        user_id = request.form['user_id']
        bank_name = request.form['bank_name']
        mode = request.form.get('mode', 'default')

    # Ensure document_insight is a JSON string
    if isinstance(document_insight, dict):
        document_insight = json.dumps(document_insight)
    elif not isinstance(document_insight, str):
        return jsonify({
            "success": False, 
            "error": f"Invalid document_insight format. Expected JSON object or string, got {type(document_insight).__name__}"
        }), 400

    # Log request details
    logger.info(f"Saving document insights to bank '{bank_name}' for user '{user_id}'")
    
    # Use the thread-based approach to call the async function
    try:
        # Run the async save_document_insights function in a separate thread
        success = run_async_in_thread(
            save_document_insights, 
            document_insight=document_insight, 
            user_id=user_id, 
            mode=mode, 
            bank=bank_name
        )

        if success:
            logger.info(f"Document insights saved successfully to bank '{bank_name}'")
            return jsonify({
                "success": True,
                "message": "Document insights saved successfully"
            }), 200
        else:
            logger.error(f"Failed to save document insights to bank '{bank_name}'")
            return jsonify({
                "success": False, 
                "error": "Failed to save document insights"
            }), 500
    except Exception as e:
        logger.error(f"Error in save_document_insights endpoint: {type(e).__name__}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({
            "success": False, 
            "error": f"Error saving document insights: {str(e)}",
            "error_type": type(e).__name__
        }), 500

@app.route('/')
def home():
    return "Hello, It's me Ami!"

@app.route('/ping', methods=['POST'])
def ping():
    return "Pong"

@app.route('/understand-cluster', methods=['POST', 'OPTIONS'])
def understand_cluster_endpoint():
    if request.method == 'OPTIONS':
        return handle_options()

    # Check for data in JSON format
    if not request.is_json:
        return jsonify({"success": False, "error": "Request must be in JSON format"}), 400

    # Validate required fields
    if 'sentences' not in request.json:
        return jsonify({"success": False, "error": "Missing 'sentences' field in request body"}), 400
        
    sentences = request.json['sentences']
    
    # Validate sentences is a list
    if not isinstance(sentences, list):
        return jsonify({"success": False, "error": "The 'sentences' field must be a list of strings"}), 400
        
    # Validate sentences are not empty
    if not sentences:
        return jsonify({"success": False, "error": "The 'sentences' list cannot be empty"}), 400
        
    # Validate all sentences are strings
    if not all(isinstance(s, str) for s in sentences):
        return jsonify({"success": False, "error": "All items in 'sentences' must be strings"}), 400

    # Log information
    logger.info(f"Processing {len(sentences)} sentences in understand-cluster endpoint")
    
    # Use the thread-based approach to call the async function
    try:
        # Run the async understand_cluster function in a separate thread
        result = run_async_in_thread(understand_cluster, sentences)
        
        # Validate result structure
        if not isinstance(result, dict):
            logger.error(f"Invalid result type from understand_cluster: {type(result)}")
            return jsonify({
                "success": False,
                "error": "Cluster processing returned invalid data structure",
                "error_type": "ProcessingError"
            }), 500
        
        # Check for success
        if result.get("success", False):
            # Log success statistics
            insights = result["document_insights"]
            logger.info(f"Cluster processed successfully: {insights.get('metadata', {}).get('sentence_count', 0)} sentences")
            
            # Return the document insights with proper structure
            return jsonify(result), 200
        else:
            error_msg = result.get("error", "Failed to process sentences")
            error_type = result.get("error_type", "UnknownError")
            logger.error(f"Cluster processing failed: {error_type}: {error_msg}")
            return jsonify({
                "success": False,
                "error": error_msg,
                "error_type": error_type
            }), 500
    except Exception as e:
        logger.error(f"Error in understand_cluster: {type(e).__name__}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": f"Error processing sentences: {str(e)}",
            "error_type": type(e).__name__
        }), 500

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
    thread_lock = None
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
    
    # Function to run the stream processing with proper locking
    async def process_stream_with_lock():
        outputs = []
        # Acquire the thread-specific lock only for the processing part
        with thread_lock:
            try:
                # Import directly here to ensure fresh imports
                from ami import convo_stream_learning
                
                # Define the async process function
                async def process_stream():
                    """Process the stream and return all items."""
                    try:
                        # Get the stream using our learning-based stream function
                        stream = convo_stream_learning(**stream_params)
                        
                        # Process all the output
                        async for item in stream:
                            outputs.append(item)
                    except Exception as e:
                        error_msg = f"Error processing learning stream: {str(e)}"
                        logger.error(error_msg)
                        import traceback
                        logger.error(traceback.format_exc())
                        outputs.append(f"data: {json.dumps({'error': error_msg})}\n\n")
                    
                    return True
                
                # Run the async function
                await process_stream()
                
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
                
            except Exception as e:
                error_msg = f"Error in learning stream processing: {str(e)}"
                logger.error(error_msg)
                import traceback
                logger.error(traceback.format_exc())
                outputs.append(f"data: {json.dumps({'error': error_msg})}\n\n")
        
        return outputs
    
    # Create a synchronous response streaming solution that runs the async processing
    def generate_response():
        """Generate streaming response items synchronously."""
        try:
            # Run the async processing with proper locking
            outputs = run_async_in_thread(process_stream_with_lock)
            
            # Yield each output
            for item in outputs:
                yield item
            
            # Log completion
            end_time = datetime.now()
            elapsed = (end_time - start_time).total_seconds()
            logger.info(f"[SESSION_TRACE] === END LEARNING CONVERSATION request - total time: {elapsed:.2f}s ===")
            
        except Exception as outer_e:
            # Handle any errors in the outer function
            error_msg = f"Error in learning response generator: {str(outer_e)}"
            logger.error(error_msg)
            import traceback
            logger.error(traceback.format_exc())
            yield f"data: {json.dumps({'error': error_msg})}\n\n"
    
    # Create a streaming response with our generator
    from flask import stream_with_context
    response = Response(stream_with_context(generate_response()), mimetype='text/event-stream')
    response.headers['X-Accel-Buffering'] = 'no'
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

# Import the init_app function from routes.py
from routes import init_app

# Initialize the app with the blueprint and configurations from routes.py
init_app(app)

# Run the server when executed directly
if __name__ == '__main__':
    # Start the brain loading in a background thread
    def load_brain_task():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            logger.info("Starting brain vector loading...")
            # Always force deletion of existing files
            result = loop.run_until_complete(load_brain_vectors(force_delete=True))
            if result:
                logger.info("Successfully loaded brain vectors")
            else:
                logger.warning("Failed to load brain vectors")
        except Exception as e:
            logger.error(f"Error loading brain vectors: {e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            loop.close()
    
    # Launch brain loading thread
    logger.info("Starting brain initialization thread...")
    brain_thread = threading.Thread(target=load_brain_task)
    brain_thread.daemon = True
    brain_thread.start()
    
    # Get port from environment or use default
    port = int(os.environ.get('PORT', 5001))
    host = os.environ.get('HOST', '0.0.0.0')
    
    print(f"Starting SocketIO server on {host}:{port}")
    socketio.run(app, host=host, port=port, debug=True)
else:
    
    print("Running in production mode - initializing brain for WSGI workers")
    