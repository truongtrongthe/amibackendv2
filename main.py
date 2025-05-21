#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Apply eventlet monkey patch before any other imports
import eventlet
eventlet.monkey_patch()

# Add at the top with other imports
import nest_asyncio
nest_asyncio.apply()  # Apply patch to allow nested event loops

import asyncio
import json
import time
import os
import traceback
from typing import Dict, List, Optional, Any, AsyncGenerator
from datetime import datetime
from uuid import uuid4, UUID
from collections import deque

import socketio
import uvicorn
from fastapi import FastAPI, BackgroundTasks, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

# Import router from fastapi_routes
from fastapi_routes import router as api_router
from supabase import create_client, Client

from utilities import logger
# Initialize FastAPI app
app = FastAPI(title="AMI Backend")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
    max_age=86400
)

# Helper function for OPTIONS requests
def handle_options():
    """Common OPTIONS handler for all endpoints."""
    return JSONResponse(
        content={},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
            "Access-Control-Max-Age": "86400"
        }
    )

# Keep track of recent webhook requests to detect duplicates
recent_requests = deque(maxlen=1000)

# Register the API router
app.include_router(api_router)

# Initialize socketio manager (import only)
import socketio_manager_async

# SocketIO setup
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')
socket_app = socketio.ASGIApp(sio, app)

# Load numpy and json imports (needed elsewhere)
import numpy as np
import json

# Initialize the app
app.config = {}
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'ami_secret_key')

# Supabase initialization
spb_url = os.getenv("SUPABASE_URL", "https://example.supabase.co")
spb_key = os.getenv("SUPABASE_KEY", "your-supabase-key")

# Add proper error handling for Supabase initialization
try:
    supabase: Client = create_client(spb_url, spb_key)
    logger.info("Supabase client initialized successfully in main.py")
except Exception as e:
    logger.error(f"Failed to initialize Supabase client in main.py: {e}")

# Lock manager for conversation thread synchronization
class LockManager:
    def __init__(self):
        self.locks: Dict[str, asyncio.Lock] = {}
        self.locks_lock = asyncio.Lock()
        self.last_used: Dict[str, float] = {}
        self.cleanup_task = None
        self.cleanup_interval = 300  # 5 minutes
        
    async def get_lock(self, thread_id: str) -> asyncio.Lock:
        """Get or create a lock for a specific thread_id"""
        async with self.locks_lock:
            if thread_id not in self.locks:
                self.locks[thread_id] = asyncio.Lock()
            self.last_used[thread_id] = time.time()
            
            # Start cleanup task if it's not running
            if not self.cleanup_task or self.cleanup_task.done():
                self.cleanup_task = asyncio.create_task(self._cleanup_locks_periodically())
                
            return self.locks[thread_id]
    
    async def _cleanup_locks_periodically(self):
        """Periodically clean up unused locks"""
        while True:
            await asyncio.sleep(60)  # Check every minute
            
            to_delete = []
            now = time.time()
            
            async with self.locks_lock:
                # Find locks that haven't been used for a while and aren't locked
                for tid, last_used in self.last_used.items():
                    if now - last_used > self.cleanup_interval:
                        if tid in self.locks and not self.locks[tid].locked():
                            to_delete.append(tid)
                
                # Delete the identified locks
                for tid in to_delete:
                    del self.locks[tid]
                    del self.last_used[tid]
                    logger.info(f"Cleaned up unused thread lock for thread_id {tid}")
                
                # Log periodic stats about locks
                logger.info(f"Thread locks stats: {len(self.locks)} active locks, {len(to_delete)} cleaned up")

# Initialize lock manager
lock_manager = LockManager()

# SocketIO session management
ws_sessions = {}
session_lock = asyncio.Lock()
undelivered_messages = {}
message_lock = asyncio.Lock()

# Share session data with socketio_manager_async by providing direct references
# This avoids circular imports between modules
socketio_manager_async.session_lock = session_lock
socketio_manager_async.message_lock = message_lock
socketio_manager_async.undelivered_messages = undelivered_messages

# Initialize socketio with shared session storage
socketio_manager_async.setup_socketio(sio)

# Helper function for OPTIONS requests
def handle_options():
    """Common OPTIONS handler for all endpoints."""
    return JSONResponse(
        content={},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
            "Access-Control-Max-Age": "86400"
        }
    )

@sio.on('connect')
async def handle_connect(sid, environ):
    """Handle client connection"""
    # Log the connection for debugging session issues
    logger.info(f"Client connected: {sid}")
    
    # Set the main sessions reference after we have at least one session
    # This ensures we're not passing an empty dictionary during initialization
    socketio_manager_async.set_main_sessions(ws_sessions)
    
    # Extra logging to debug sessions
    if 'main_ws_sessions' in dir(socketio_manager_async):
        logger.info(f"Connect: Socket module session count: {len(socketio_manager_async.main_ws_sessions) if socketio_manager_async.main_ws_sessions else 0}")
    logger.info(f"Connect: Main app session count: {len(ws_sessions)}")

@sio.on('register_session')
async def handle_register(sid, data):
    thread_id = data.get('thread_id')
    user_id = data.get('user_id', 'anonymous')
    
    if not thread_id:
        await sio.emit('error', {'message': 'No thread_id provided'}, room=sid)
        return
        
    async with session_lock:
        ws_sessions[sid] = {
            'thread_id': thread_id,
            'user_id': user_id,
            'last_activity': datetime.now().isoformat(),
            'connected_at': datetime.now().isoformat()
        }
        
        # Re-share the sessions dict with the socketio manager module each time to ensure reference is current
        socketio_manager_async.set_main_sessions(ws_sessions)
        logger.info(f"After registration: Main app session count: {len(ws_sessions)}")
        
    await sio.enter_room(sid, thread_id)
    await sio.emit('session_registered', {
        'status': 'ready',
        'thread_id': thread_id,
        'session_id': sid
    }, room=sid)
    
    logger.info(f"Session {sid} registered to thread {thread_id} for user {user_id}")
    
    # Send any undelivered messages
    async with message_lock:
        if thread_id in undelivered_messages:
            for msg in undelivered_messages[thread_id]:
                await sio.emit(msg['event'], msg['data'], room=sid)
            logger.info(f"Sent {len(undelivered_messages[thread_id])} undelivered messages to {sid}")
            undelivered_messages[thread_id] = []

@sio.on('disconnect')
async def handle_disconnect(sid):
    async with session_lock:
        if sid in ws_sessions:
            thread_id = ws_sessions[sid].get('thread_id')
            del ws_sessions[sid]
            logger.info(f"Client disconnected: {sid} from thread {thread_id}")
        else:
            logger.info(f"Client disconnected: {sid} (no session data)")

@sio.on('ping')
async def handle_ping(sid, data=None):
    thread_id = None
    async with session_lock:
        if sid in ws_sessions:
            thread_id = ws_sessions[sid].get('thread_id')
            ws_sessions[sid]['last_activity'] = datetime.now().isoformat()
    
    if thread_id:
        await sio.emit('pong', {'thread_id': thread_id, 'timestamp': datetime.now().isoformat()}, room=sid)

# WebSocket emit functions
async def emit_analysis_event(thread_id: str, data: Dict[str, Any]) -> bool:
    """Emit an analysis event to all clients in a thread room"""
    if "thread_id" not in data:
        data["thread_id"] = thread_id
    
    # Use socketio_manager_async version
    return await socketio_manager_async.emit_analysis_event(thread_id, data)

async def emit_knowledge_event(thread_id: str, data: Dict[str, Any]) -> bool:
    """Emit a knowledge event to all clients in a thread room"""
    if "thread_id" not in data:
        data["thread_id"] = thread_id
    
    # Use socketio_manager_async version
    return await socketio_manager_async.emit_knowledge_event(thread_id, data)

async def emit_next_action_event(thread_id: str, data: Dict[str, Any]) -> bool:
    """Emit a next_action event to all clients in a thread room"""
    if "thread_id" not in data:
        data["thread_id"] = thread_id
    
    # Use socketio_manager_async version
    return await socketio_manager_async.emit_next_action_event(thread_id, data)

# Request models
class HaveFunRequest(BaseModel):
    user_input: str
    user_id: str = "thefusionlab"
    thread_id: str = "chat_thread"
    graph_version_id: str = ""
    use_websocket: bool = False
    socket_id: Optional[str] = None

class ConversationLearningRequest(BaseModel):
    user_input: str
    user_id: str = "learner"
    thread_id: str = "learning_thread"
    graph_version_id: str = ""
    use_websocket: bool = False

class ProcessDocumentRequest(BaseModel):
    user_id: str
    bank_name: str
    reformatted_text: str
    knowledge_elements: str
    mode: str = "default"

class UnderstandDocumentRequest(BaseModel):
    user_id: str
    bank_name: str
    reformatted_text: str
    mode: str = "default"

class SaveDocumentInsightsRequest(BaseModel):
    user_id: str
    bank_name: str
    text: str
    insights: str
    mode: str = "default"

# Main havefun endpoint
@app.post('/havefun')
async def havefun(request: HaveFunRequest, background_tasks: BackgroundTasks):
    """
    Handle havefun requests with asyncio lock isolation.
    Each request for a unique thread_id can run in parallel.
    """
    start_time = datetime.now()
    request_id = str(uuid4())[:8]  # Generate a short request ID for tracing
    
    logger.info(f"[REQUEST:{request_id}] === BEGIN havefun request at {start_time.isoformat()} ===")
    logger.info(f"[REQUEST:{request_id}] Request params: user_id={request.user_id}, thread_id={request.thread_id}, use_websocket={request.use_websocket}")
    logger.info(f"[REQUEST:{request_id}] User input: '{request.user_input[:50]}...' (truncated)")
    
    # Get a lock for this thread_id
    thread_id = request.thread_id
    logger.info(f"[REQUEST:{request_id}] Getting lock for thread {thread_id}")
    thread_lock = await lock_manager.get_lock(thread_id)
    logger.info(f"[REQUEST:{request_id}] Got lock manager lock for thread {thread_id}")
    
    # Log WebSocket sessions before processing
    if request.use_websocket:
        async with session_lock:
            thread_sessions = [sid for sid, data in ws_sessions.items() if data.get('thread_id') == thread_id]
            logger.info(f"[REQUEST:{request_id}] Before processing: {len(thread_sessions)} WebSocket sessions for thread {thread_id}: {thread_sessions}")
    
    # Always use StreamingResponse, regardless of WebSocket flag
    # The WebSocket flag is only used to determine whether to emit events via WebSocket
    # in addition to the HTTP stream
    logger.info(f"[REQUEST:{request_id}] Creating StreamingResponse with generate_sse_stream")
    response = StreamingResponse(
        generate_sse_stream(request, thread_lock, start_time, request_id),
        media_type="text/event-stream",
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache",
            "Access-Control-Allow-Origin": "*"
        }
    )
    logger.info(f"[REQUEST:{request_id}] Returning StreamingResponse to client")
    return response

@app.options('/havefun')
async def havefun_options():
    return handle_options()

# Generate SSE stream for all requests
async def generate_sse_stream(request: HaveFunRequest, thread_lock: asyncio.Lock, start_time: datetime, request_id: str):
    """Generate an SSE stream with the conversation response"""
    thread_id = request.thread_id
    
    # Update WebSocket sessions if request.use_websocket is true
    if request.use_websocket:
        # Check if there are active WebSocket sessions for this thread
        active_session_exists = False
        
        async with session_lock:
            # If socket_id is provided, prioritize that specific session
            if request.socket_id and request.socket_id in ws_sessions:
                logger.info(f"[REQUEST:{request_id}] Using provided socket_id: {request.socket_id} for thread {thread_id}")
                sid = request.socket_id
                # Update session data for this socket_id or register it if needed
                if ws_sessions[sid].get('thread_id') != thread_id:
                    logger.info(f"[REQUEST:{request_id}] Updating socket {sid} with new thread_id: {thread_id}")
                    ws_sessions[sid]['thread_id'] = thread_id
                    # Make sure the socket is in the correct room
                    await sio.enter_room(sid, thread_id)
                    # Send registration confirmation
                    await sio.emit('session_registered', {
                        'status': 'ready',
                        'thread_id': thread_id,
                        'session_id': sid
                    }, room=sid)
                
                ws_sessions[sid]['last_activity'] = datetime.now().isoformat()
                ws_sessions[sid]['api_request_time'] = datetime.now().isoformat()
                ws_sessions[sid]['has_pending_request'] = True
                active_session_exists = True
                logger.info(f"[REQUEST:{request_id}] Updated session {sid} for thread {thread_id}")
            else:
                # Otherwise use all sessions for this thread
                thread_sessions = [sid for sid, data in ws_sessions.items() if data.get('thread_id') == thread_id]
                active_session_exists = len(thread_sessions) > 0            
                
                if active_session_exists:
                    logger.info(f"[REQUEST:{request_id}] Found {len(thread_sessions)} WebSocket sessions for thread {thread_id}: {thread_sessions}")
                    for sid in thread_sessions:
                        ws_sessions[sid]['last_activity'] = datetime.now().isoformat()
                        ws_sessions[sid]['api_request_time'] = datetime.now().isoformat()
                        ws_sessions[sid]['has_pending_request'] = True
                        logger.info(f"[REQUEST:{request_id}] Updated session {sid} for thread {thread_id}")
                else:
                    logger.warning(f"[REQUEST:{request_id}] No active WebSocket sessions found for thread {thread_id} before processing")
                
        # If using WebSockets, immediately return a processing status and continue in background
        if request.use_websocket:
            # Send initial response indicating processing via WebSocket
            process_id = str(uuid4())
            processing_message = {
                "status": "processing",
                "message": "AI is working on your request...",
                "thread_id": thread_id,
                "process_id": process_id
            }
            logger.info(f"[REQUEST:{request_id}] Sending WebSocket processing message for thread {thread_id}")
            
            # Yield the processing message as the HTTP response
            logger.info(f"[REQUEST:{request_id}] Yielding initial processing message")
            yield f"data: {json.dumps(processing_message)}\n\n"
    
    try:
        # Try to acquire the lock with timeout
        async with asyncio.timeout(60):  # 60-second timeout
            async with thread_lock:
                logger.info(f"[REQUEST:{request_id}] Acquired lock for thread {thread_id}")
                
                # Check active sessions again after acquiring lock
                if request.use_websocket:
                    async with session_lock:
                        thread_sessions = [sid for sid, data in ws_sessions.items() if data.get('thread_id') == thread_id]
                        logger.info(f"[REQUEST:{request_id}] After lock: {len(thread_sessions)} WebSocket sessions for thread {thread_id}")
        
                # Import here to avoid circular imports
                from ami import convo_stream
                
                # Log request to convo_stream
                logger.info(f"[REQUEST:{request_id}] Calling convo_stream for thread {thread_id}, use_websocket={request.use_websocket}")
                
                # Track response count
                response_count = 0
                
                # Process the conversation and yield results
                async for result in convo_stream(
                    user_input=request.user_input,
                    thread_id=thread_id,
                    user_id=request.user_id,
                    graph_version_id=request.graph_version_id,
                    use_websocket=request.use_websocket,
                    thread_id_for_analysis=thread_id
                ):
                    response_count += 1
                    # Format the response as SSE
                    if isinstance(result, str) and result.startswith("data: "):
                        # Already formatted for SSE
                        logger.info(f"[REQUEST:{request_id}] #{response_count} Yielding string SSE response for thread {thread_id}")
                        yield result + "\n"
                    elif isinstance(result, dict):
                        # Format JSON for SSE
                        logger.info(f"[REQUEST:{request_id}] #{response_count} Yielding dict response for thread {thread_id}: {str(result)[:100]}...")
                        yield f"data: {json.dumps(result)}\n\n"
                    else:
                        # For string responses without SSE format
                        logger.info(f"[REQUEST:{request_id}] #{response_count} Yielding message response for thread {thread_id}: {str(result)[:100]}...")
                        yield f"data: {json.dumps({'message': result})}\n\n"
                
                logger.info(f"[REQUEST:{request_id}] Completed yielding {response_count} responses for thread {thread_id}")
                
                # Update WebSocket sessions to indicate request is complete if using WebSocket
                if request.use_websocket:
                    async with session_lock:
                        thread_sessions = [sid for sid, data in ws_sessions.items() if data.get('thread_id') == thread_id]
                        logger.info(f"[REQUEST:{request_id}] After processing: {len(thread_sessions)} WebSocket sessions for thread {thread_id}")
                        for sid in thread_sessions:
                            if 'has_pending_request' in ws_sessions[sid]:
                                ws_sessions[sid]['has_pending_request'] = False
                                logger.info(f"[REQUEST:{request_id}] Updated session {sid} - set has_pending_request=False")
                
                logger.info(f"[REQUEST:{request_id}] Released lock for thread {thread_id}")
    except asyncio.TimeoutError:
        logger.error(f"[REQUEST:{request_id}] Could not acquire lock for thread {thread_id} after 60 seconds")
        logger.info(f"[REQUEST:{request_id}] Yielding timeout error for thread {thread_id}")
        yield f"data: {json.dumps({'error': 'Server busy. Please try again.'})}\n\n"
    except Exception as e:
        logger.error(f"[REQUEST:{request_id}] Error generating SSE stream: {str(e)}")
        logger.error(traceback.format_exc())
        logger.info(f"[REQUEST:{request_id}] Yielding error for thread {thread_id}: {str(e)}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
    finally:
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()
        logger.info(f"[REQUEST:{request_id}] === END SSE request for thread {thread_id} - total time: {elapsed:.2f}s ===")

# Conversation Learning endpoint
@app.post('/conversation/learning')
async def conversation_learning(request: ConversationLearningRequest, background_tasks: BackgroundTasks):
    """
    Handle conversation requests using the learning-based conversation system.
    This endpoint uses tool_learning.py for knowledge similarity checks and active learning.
    """
    start_time = datetime.now()
    request_id = str(uuid4())[:8]  # Generate a short request ID for tracing
    
    logger.info(f"[REQUEST:{request_id}] === BEGIN LEARNING CONVERSATION request at {start_time.isoformat()} ===")
    logger.info(f"[REQUEST:{request_id}] Request params: user_id={request.user_id}, thread_id={request.thread_id}, use_websocket={request.use_websocket}")
    
    # Get a lock for this thread_id
    thread_id = request.thread_id
    logger.info(f"[REQUEST:{request_id}] Getting lock for thread {thread_id}")
    thread_lock = await lock_manager.get_lock(thread_id)
    logger.info(f"[REQUEST:{request_id}] Got lock manager lock for thread {thread_id}")
    
    # Always use StreamingResponse, regardless of WebSocket flag
    # The WebSocket flag is only used to determine whether to emit events via WebSocket
    # in addition to the HTTP stream
    logger.info(f"[REQUEST:{request_id}] Creating StreamingResponse with generate_learning_sse_stream")
    response = StreamingResponse(
        generate_learning_sse_stream(request, thread_lock, start_time, request_id),
        media_type="text/event-stream",
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache",
            "Access-Control-Allow-Origin": "*"
        }
    )
    logger.info(f"[REQUEST:{request_id}] Returning StreamingResponse to client")
    return response

@app.options('/conversation/learning')
async def conversation_learning_options():
    return handle_options()

# Generate SSE stream for learning requests
async def generate_learning_sse_stream(request: ConversationLearningRequest, thread_lock: asyncio.Lock, start_time: datetime, request_id: str):
    """Generate an SSE stream with the learning conversation response"""
    thread_id = request.thread_id
    
    # Update WebSocket sessions if request.use_websocket is true
    if request.use_websocket:
        # Check if there are active WebSocket sessions for this thread
        active_session_exists = False
        
        async with session_lock:
            thread_sessions = [sid for sid, data in ws_sessions.items() if data.get('thread_id') == thread_id]
            active_session_exists = len(thread_sessions) > 0            
            if active_session_exists:
                logger.info(f"[REQUEST:{request_id}] Found {len(thread_sessions)} WebSocket sessions for thread {thread_id}: {thread_sessions}")
                for sid in thread_sessions:
                    ws_sessions[sid]['last_activity'] = datetime.now().isoformat()
                    ws_sessions[sid]['api_request_time'] = datetime.now().isoformat()
                    ws_sessions[sid]['has_pending_request'] = True
                    logger.info(f"[REQUEST:{request_id}] Updated session {sid} for thread {thread_id}")
            else:
                logger.warning(f"[REQUEST:{request_id}] No active WebSocket sessions found for thread {thread_id} before processing")
                
        # If using WebSockets, immediately return a processing status and continue in background
        if request.use_websocket:
            # Send initial response indicating processing via WebSocket
            process_id = str(uuid4())
            processing_message = {
                "status": "processing",
                "message": "Request is being processed and results will be sent via WebSocket",
                "thread_id": thread_id,
                "process_id": process_id
            }
            logger.info(f"[REQUEST:{request_id}] Sending WebSocket processing message for thread {thread_id}")
            
            # Yield the processing message as the HTTP response
            logger.info(f"[REQUEST:{request_id}] Yielding initial processing message")
            yield f"data: {json.dumps(processing_message)}\n\n"
    
    try:
        # Try to acquire the lock with timeout
        async with asyncio.timeout(60):  # 60-second timeout
            async with thread_lock:
                logger.info(f"[REQUEST:{request_id}] Acquired lock for thread {thread_id}")
                
                # Check active sessions again after acquiring lock
                if request.use_websocket:
                    async with session_lock:
                        thread_sessions = [sid for sid, data in ws_sessions.items() if data.get('thread_id') == thread_id]
                        logger.info(f"[REQUEST:{request_id}] After lock: {len(thread_sessions)} WebSocket sessions for thread {thread_id}")
        
                # Import here to avoid circular imports
                from ami import convo_stream_learning
                
                # Log request to convo_stream_learning
                logger.info(f"[REQUEST:{request_id}] Calling convo_stream_learning for thread {thread_id}, use_websocket={request.use_websocket}")
                
                # Track response count
                response_count = 0
                
                # Process the conversation and yield results
                async for result in convo_stream_learning(
                    user_input=request.user_input,
                    thread_id=thread_id,
                    user_id=request.user_id,
                    graph_version_id=request.graph_version_id,
                    use_websocket=request.use_websocket,
                    thread_id_for_analysis=thread_id
                ):
                    response_count += 1
                    # Format the response as SSE
                    if isinstance(result, str) and result.startswith("data: "):
                        # Already formatted for SSE
                        logger.info(f"[REQUEST:{request_id}] #{response_count} Yielding string SSE response for thread {thread_id}")
                        yield result + "\n"
                    elif isinstance(result, dict):
                        # Format JSON for SSE
                        logger.info(f"[REQUEST:{request_id}] #{response_count} Yielding dict response for thread {thread_id}: {str(result)[:100]}...")
                        yield f"data: {json.dumps(result)}\n\n"
                    else:
                        # For string responses without SSE format
                        logger.info(f"[REQUEST:{request_id}] #{response_count} Yielding message response for thread {thread_id}: {str(result)[:100]}...")
                        yield f"data: {json.dumps({'message': result})}\n\n"
                
                logger.info(f"[REQUEST:{request_id}] Completed yielding {response_count} responses for thread {thread_id}")
                
                # Update WebSocket sessions to indicate request is complete if using WebSocket
                if request.use_websocket:
                    async with session_lock:
                        thread_sessions = [sid for sid, data in ws_sessions.items() if data.get('thread_id') == thread_id]
                        logger.info(f"[REQUEST:{request_id}] After processing: {len(thread_sessions)} WebSocket sessions for thread {thread_id}")
                        for sid in thread_sessions:
                            if 'has_pending_request' in ws_sessions[sid]:
                                ws_sessions[sid]['has_pending_request'] = False
                                logger.info(f"[REQUEST:{request_id}] Updated session {sid} - set has_pending_request=False")
                
                logger.info(f"[REQUEST:{request_id}] Released lock for thread {thread_id}")
    except asyncio.TimeoutError:
        logger.error(f"[REQUEST:{request_id}] Could not acquire lock for thread {thread_id} after 60 seconds")
        logger.info(f"[REQUEST:{request_id}] Yielding timeout error for thread {thread_id}")
        yield f"data: {json.dumps({'error': 'Server busy. Please try again.'})}\n\n"
    except Exception as e:
        logger.error(f"[REQUEST:{request_id}] Error generating learning SSE stream: {str(e)}")
        logger.error(traceback.format_exc())
        logger.info(f"[REQUEST:{request_id}] Yielding error for thread {thread_id}: {str(e)}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
    finally:
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()
        logger.info(f"[REQUEST:{request_id}] === END learning SSE request for thread {thread_id} - total time: {elapsed:.2f}s ===")



# Run the application
if __name__ == "__main__":
    uvicorn.run(socket_app, host="0.0.0.0", port=5001) 