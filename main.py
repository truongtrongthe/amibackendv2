#FASTAPI app

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
from braingraph_routes import router as braingraph_router
from contact_apis import router as contact_router
from waitlist import router as waitlist_router
from login import router as login_router
from aibrain import router as brain_router
from supabase import create_client, Client
from ava import AVA

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

# Register the API routers
app.include_router(api_router)
app.include_router(braingraph_router)
app.include_router(contact_router)
app.include_router(waitlist_router)
app.include_router(login_router)
app.include_router(brain_router)

# Initialize socketio manager (import only)
import socketio_manager_async

# SocketIO setup
import socketio
sio_server = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')
socket_app = socketio.ASGIApp(sio_server, app)

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

# AVA instance management - create a shared instance
ava_instance = None
ava_lock = asyncio.Lock()

async def get_ava_instance() -> AVA:
    """Get or create a shared AVA instance"""
    global ava_instance
    async with ava_lock:
        if ava_instance is None:
            ava_instance = AVA()
            await ava_instance.initialize()
            logger.info("Initialized shared AVA instance")
        return ava_instance

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
socketio_manager_async.setup_socketio(sio_server)

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

@sio_server.on('connect')
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

@sio_server.on('register_session')
async def handle_register(sid, data):
    thread_id = data.get('thread_id')
    user_id = data.get('user_id', 'anonymous')
    
    if not thread_id:
        await sio_server.emit('error', {'message': 'No thread_id provided'}, room=sid)
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
        
    await sio_server.enter_room(sid, thread_id)
    await sio_server.emit('session_registered', {
        'status': 'ready',
        'thread_id': thread_id,
        'session_id': sid
    }, room=sid)
    
    logger.info(f"Session {sid} registered to thread {thread_id} for user {user_id}")
    
    # Send any undelivered messages
    async with message_lock:
        if thread_id in undelivered_messages:
            for msg in undelivered_messages[thread_id]:
                await sio_server.emit(msg['event'], msg['data'], room=sid)
            logger.info(f"Sent {len(undelivered_messages[thread_id])} undelivered messages to {sid}")
            undelivered_messages[thread_id] = []

@sio_server.on('disconnect')
async def handle_disconnect(sid):
    async with session_lock:
        if sid in ws_sessions:
            thread_id = ws_sessions[sid].get('thread_id')
            del ws_sessions[sid]
            logger.info(f"Client disconnected: {sid} from thread {thread_id}")
        else:
            logger.info(f"Client disconnected: {sid} (no session data)")

@sio_server.on('ping')
async def handle_ping(sid, data=None):
    thread_id = None
    async with session_lock:
        if sid in ws_sessions:
            thread_id = ws_sessions[sid].get('thread_id')
            ws_sessions[sid]['last_activity'] = datetime.now().isoformat()
    
    if thread_id:
        await sio_server.emit('pong', {'thread_id': thread_id, 'timestamp': datetime.now().isoformat()}, room=sid)

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

async def emit_learning_intent_event(thread_id: str, data: Dict[str, Any]) -> bool:
    """Emit a learning intent event to all clients in a thread room"""
    if "thread_id" not in data:
        data["thread_id"] = thread_id
    
    # Use socketio_manager_async version
    return await socketio_manager_async.emit_learning_intent_event(thread_id, data)

async def emit_learning_knowledge_event(thread_id: str, data: Dict[str, Any]) -> bool:
    """Emit a learning knowledge event to all clients in a thread room"""
    if "thread_id" not in data:
        data["thread_id"] = thread_id
    
    # Use socketio_manager_async version
    return await socketio_manager_async.emit_learning_knowledge_event(thread_id, data)

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
    org_id: str = "unknown"  # Add org_id field with default value

class ConversationPilotRequest(BaseModel):
    user_input: str
    user_id: str = "pilot_user"
    thread_id: str = "pilot_thread"
    graph_version_id: str = ""
    use_websocket: bool = False
    org_id: str = "unknown"  # Add org_id field with default value

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

class QueryKnowledgeRequest(BaseModel):
    vector_id: str
    bank_name: str = "conversation"

class ConversationGradingRequest(BaseModel):
    graph_version_id: str = ""

class ToolExecuteRequest(BaseModel):
    tool_name: str
    parameters: Dict[str, Any]
    org_id: str = "unknown"  # Add org_id field with default value

class UpdateBrainGraphRequest(BaseModel):
    graph_id: str
    name: Optional[str] = None
    description: Optional[str] = None

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
                    await sio_server.enter_room(sid, thread_id)
                    # Send registration confirmation
                    await sio_server.emit('session_registered', {
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
        async with asyncio.timeout(180):  # 180-second timeout (increased from 60 seconds)
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
                        #logger.info(f"[REQUEST:{request_id}] #{response_count} Yielding string SSE response for thread {thread_id}")
                        yield result + "\n"
                    elif isinstance(result, dict):
                        # Format JSON for SSE
                        #logger.info(f"[REQUEST:{request_id}] #{response_count} Yielding dict response for thread {thread_id}: {str(result)[:100]}...")
                        yield f"data: {json.dumps(result)}\n\n"
                    else:
                        # For string responses without SSE format
                        #logger.info(f"[REQUEST:{request_id}] #{response_count} Yielding message response for thread {thread_id}: {str(result)[:100]}...")
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
        logger.error(f"[REQUEST:{request_id}] Could not acquire lock for thread {thread_id} after 180 seconds")
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

# Conversation Pilot endpoint - similar to learning but without knowledge saving
@app.post('/conversation/pilot')
async def conversation_pilot(request: ConversationPilotRequest, background_tasks: BackgroundTasks):
    """
    Handle conversation requests using a pilot conversation system.
    This endpoint is similar to /conversation/learning but does NOT save any knowledge.
    It only communicates with the user without any knowledge management.
    """
    start_time = datetime.now()
    request_id = str(uuid4())[:8]  # Generate a short request ID for tracing
    
    logger.info(f"[REQUEST:{request_id}] === BEGIN PILOT CONVERSATION request at {start_time.isoformat()} ===")
    logger.info(f"[REQUEST:{request_id}] Request params: user_id={request.user_id}, thread_id={request.thread_id}, use_websocket={request.use_websocket}")
    
    # Get a lock for this thread_id
    thread_id = request.thread_id
    logger.info(f"[REQUEST:{request_id}] Getting lock for thread {thread_id}")
    thread_lock = await lock_manager.get_lock(thread_id)
    logger.info(f"[REQUEST:{request_id}] Got lock manager lock for thread {thread_id}")
    
    # Always use StreamingResponse, regardless of WebSocket flag
    # The WebSocket flag is only used to determine whether to emit events via WebSocket
    # in addition to the HTTP stream
    logger.info(f"[REQUEST:{request_id}] Creating StreamingResponse with generate_pilot_sse_stream")
    response = StreamingResponse(
        generate_pilot_sse_stream(request, thread_lock, start_time, request_id),
        media_type="text/event-stream",
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache",
            "Access-Control-Allow-Origin": "*"
        }
    )
    logger.info(f"[REQUEST:{request_id}] Returning StreamingResponse to client")
    return response

@app.options('/conversation/pilot')
async def conversation_pilot_options():
    return handle_options()

# COT Processor Grading endpoint
@app.post('/conversation/grading')
async def conversation_grading(request: ConversationGradingRequest):
    """
    Validate COTProcessor knowledge loading and return comprehensive knowledge bases.
    Returns the three main knowledge components: Profiling, Communication, and Business Objectives.
    """
    start_time = datetime.now()
    request_id = str(uuid4())[:8]
    
    logger.info(f"[REQUEST:{request_id}] === BEGIN conversation/grading request at {start_time.isoformat()} ===")
    logger.info(f"[REQUEST:{request_id}] graph_version_id={request.graph_version_id}")
    
    try:
        # Import the grading module
        from grading import COTProcessorGrader
        
        # Initialize and run grading
        grader = COTProcessorGrader()
        
        # Initialize COTProcessor with the provided graph_version_id
        graph_version_id = request.graph_version_id if request.graph_version_id else None
        logger.info(f"[REQUEST:{request_id}] Initializing COTProcessor with graph_version_id: {graph_version_id}")
        
        success = await grader.initialize_cot_processor(graph_version_id)
        
        if not success:
            logger.error(f"[REQUEST:{request_id}] Failed to initialize COTProcessor")
            return {
                "status": "error",
                "error": "Failed to initialize COTProcessor",
                "request_id": request_id,
                "elapsed_time": (datetime.now() - start_time).total_seconds()
            }
        
        logger.info(f"[REQUEST:{request_id}] COTProcessor initialized successfully")
        
        # Get the three comprehensive knowledge bases
        profiling_skills = grader.get_comprehensive_profiling_skills()
        communication_skills = grader.get_comprehensive_communication_skills()
        business_objectives = grader.get_comprehensive_business_objectives()
        
        # Validate that all knowledge bases were loaded successfully
        validation_results = grader.validate_knowledge_loading()
        
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()
        
        # Prepare the response
        response_data = {
            "status": "success",
            "request_id": request_id,
            "elapsed_time": elapsed,
            "graph_version_id": grader.cot_processor.graph_version_id if grader.cot_processor else None,
            "validation": {
                "overall_status": validation_results.get("overall_status", "unknown"),
                "basic_knowledge_valid": len([v for v in validation_results.get("basic_knowledge", {}).values() if v.get("is_valid", False)]),
                "comprehensive_knowledge_valid": len([v for v in validation_results.get("comprehensive_knowledge", {}).values() if v.get("is_valid", False)])
            },
            "knowledge_bases": {
                "profiling_instinct": {
                    "knowledge_context": profiling_skills.get("knowledge_context", ""),
                    "metadata": profiling_skills.get("metadata", {}),
                    "content_length": len(profiling_skills.get("knowledge_context", "")),
                    "is_valid": "error" not in profiling_skills
                },
                "communication_instinct": {
                    "knowledge_context": communication_skills.get("knowledge_context", ""),
                    "metadata": communication_skills.get("metadata", {}),
                    "content_length": len(communication_skills.get("knowledge_context", "")),
                    "is_valid": "error" not in communication_skills
                },
                "business_objectives_instinct": {
                    "knowledge_context": business_objectives.get("knowledge_context", ""),
                    "metadata": business_objectives.get("metadata", {}),
                    "content_length": len(business_objectives.get("knowledge_context", "")),
                    "is_valid": "error" not in business_objectives
                }
            }
        }
        
        logger.info(f"[REQUEST:{request_id}] Successfully generated grading response:")
        logger.info(f"[REQUEST:{request_id}] - Profiling Instinct: {response_data['knowledge_bases']['profiling_instinct']['content_length']} chars")
        logger.info(f"[REQUEST:{request_id}] - Communication Instinct: {response_data['knowledge_bases']['communication_instinct']['content_length']} chars")
        logger.info(f"[REQUEST:{request_id}] - Business Objectives Instinct: {response_data['knowledge_bases']['business_objectives_instinct']['content_length']} chars")
        logger.info(f"[REQUEST:{request_id}] - Overall validation status: {response_data['validation']['overall_status']}")
        
        return response_data
        
    except Exception as e:
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()
        
        logger.error(f"[REQUEST:{request_id}] Error in conversation/grading endpoint: {str(e)}")
        import traceback
        logger.error(f"[REQUEST:{request_id}] Traceback: {traceback.format_exc()}")
        
        return {
            "status": "error",
            "error": str(e),
            "request_id": request_id,
            "elapsed_time": elapsed
        }
    finally:
        logger.info(f"[REQUEST:{request_id}] === END conversation/grading request - total time: {(datetime.now() - start_time).total_seconds():.2f}s ===")

@app.options('/conversation/grading')
async def conversation_grading_options():
    return handle_options()

# Query Knowledge endpoint
@app.post('/query-knowledge')
async def query_knowledge_endpoint(request: QueryKnowledgeRequest):
    """
    Fetch a specific vector by its ID from Pinecone knowledge base.
    """
    start_time = datetime.now()
    request_id = str(uuid4())[:8]
    
    logger.info(f"[REQUEST:{request_id}] === BEGIN query-knowledge request at {start_time.isoformat()} ===")
    logger.info(f"[REQUEST:{request_id}] Fetching vector_id={request.vector_id}, bank_name={request.bank_name}")
    
    try:
        # Import the fetch_vector function
        from pccontroller import fetch_vector
        
        # Fetch the vector
        result = await fetch_vector(
            vector_id=request.vector_id,
            bank_name=request.bank_name
        )
        
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()
        
        if result.get("success"):
            logger.info(f"[REQUEST:{request_id}] Successfully fetched vector {request.vector_id} - time: {elapsed:.2f}s")
            return {
                "status": "success",
                "data": result,
                "request_id": request_id,
                "elapsed_time": elapsed
            }
        else:
            logger.warning(f"[REQUEST:{request_id}] Failed to fetch vector {request.vector_id}: {result.get('error')} - time: {elapsed:.2f}s")
            return {
                "status": "error",
                "error": result.get("error", "Unknown error"),
                "vector_id": request.vector_id,
                "request_id": request_id,
                "elapsed_time": elapsed
            }
            
    except Exception as e:
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()
        
        logger.error(f"[REQUEST:{request_id}] Error in query-knowledge endpoint: {str(e)} - time: {elapsed:.2f}s")
        import traceback
        logger.error(f"[REQUEST:{request_id}] Traceback: {traceback.format_exc()}")
        
        return {
            "status": "error",
            "error": str(e),
            "vector_id": request.vector_id,
            "request_id": request_id,
            "elapsed_time": elapsed
        }

@app.options('/query-knowledge')
async def query_knowledge_options():
    return handle_options()

# Tools Execute endpoint
@app.post('/api/tools/execute')
async def execute_tool_endpoint(request: ToolExecuteRequest):
    """
    Execute a tool via AVA instance. Used for handling update decisions and other tool calls.
    """
    start_time = datetime.now()
    request_id = str(uuid4())[:8]
    
    logger.info(f"[REQUEST:{request_id}] === BEGIN /api/tools/execute request at {start_time.isoformat()} ===")
    logger.info(f"[REQUEST:{request_id}] Tool: {request.tool_name}, Parameters: {request.parameters}")
    
    try:
        # Get AVA instance
        ava = await get_ava_instance()
        
        # Add org_id to parameters if not present
        if "org_id" not in request.parameters:
            request.parameters["org_id"] = request.org_id
        
        # Execute the tool
        result = await ava.execute_tool(request.tool_name, request.parameters)
        
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()
        
        logger.info(f"[REQUEST:{request_id}] Tool execution completed - time: {elapsed:.2f}s")
        logger.info(f"[REQUEST:{request_id}] Result status: {result.get('status', 'unknown')}")
        
        # Return the result with additional metadata
        response = {
            **result,
            "request_id": request_id,
            "elapsed_time": elapsed,
            "tool_name": request.tool_name
        }
        
        return JSONResponse(content=response)
            
    except Exception as e:
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()
        
        logger.error(f"[REQUEST:{request_id}] Error in /api/tools/execute endpoint: {str(e)} - time: {elapsed:.2f}s")
        import traceback
        logger.error(f"[REQUEST:{request_id}] Traceback: {traceback.format_exc()}")
        
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(e),
                "tool_name": request.tool_name,
                "request_id": request_id,
                "elapsed_time": elapsed
            }
        )
    finally:
        logger.info(f"[REQUEST:{request_id}] === END /api/tools/execute request - total time: {(datetime.now() - start_time).total_seconds():.2f}s ===")

@app.options('/api/tools/execute')
async def execute_tool_options():
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
        async with asyncio.timeout(180):  # 180-second timeout (increased from 60 seconds)
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
                logger.info(f"[REQUEST:{request_id}] Org ID: {request.org_id}")
                # Track response count
                response_count = 0
                
                # Process the conversation and yield results
                async for result in convo_stream_learning(
                    user_input=request.user_input,
                    thread_id=thread_id,
                    user_id=request.user_id,
                    graph_version_id=request.graph_version_id,
                    use_websocket=request.use_websocket,
                    thread_id_for_analysis=thread_id,
                    org_id=request.org_id  # Pass org_id to convo_stream_learning
                ):
                    response_count += 1
                    # Format the response as SSE
                    if isinstance(result, str) and result.startswith("data: "):
                        # Already formatted for SSE
                        #logger.info(f"[REQUEST:{request_id}] #{response_count} Yielding string SSE response for thread {thread_id}")
                        yield result + "\n"
                    elif isinstance(result, dict):
                        # Format JSON for SSE
                        #logger.info(f"[REQUEST:{request_id}] #{response_count} Yielding dict response for thread {thread_id}: {str(result)[:100]}...")
                        yield f"data: {json.dumps(result)}\n\n"
                    else:
                        # For string responses without SSE format
                        #logger.info(f"[REQUEST:{request_id}] #{response_count} Yielding message response for thread {thread_id}: {str(result)[:100]}...")
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
        logger.error(f"[REQUEST:{request_id}] Could not acquire lock for thread {thread_id} after 180 seconds")
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

# Generate SSE stream for pilot requests
async def generate_pilot_sse_stream(request: ConversationPilotRequest, thread_lock: asyncio.Lock, start_time: datetime, request_id: str):
    """Generate an SSE stream with the pilot conversation response - NO knowledge saving"""
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
        async with asyncio.timeout(180):  # 180-second timeout (increased from 60 seconds)
            async with thread_lock:
                logger.info(f"[REQUEST:{request_id}] Acquired lock for thread {thread_id}")
                
                # Check active sessions again after acquiring lock
                if request.use_websocket:
                    async with session_lock:
                        thread_sessions = [sid for sid, data in ws_sessions.items() if data.get('thread_id') == thread_id]
                        logger.info(f"[REQUEST:{request_id}] After lock: {len(thread_sessions)} WebSocket sessions for thread {thread_id}")
        
                # Import here to avoid circular imports
                from ami import convo_stream_pilot
                
                # Log request to convo_stream_pilot
                logger.info(f"[REQUEST:{request_id}] Calling convo_stream_pilot for thread {thread_id}, use_websocket={request.use_websocket}")
                logger.info(f"[REQUEST:{request_id}] Org ID: {request.org_id}")
                # Track response count
                response_count = 0
                
                # Process the conversation and yield results
                async for result in convo_stream_pilot(
                    user_input=request.user_input,
                    thread_id=thread_id,
                    user_id=request.user_id,
                    graph_version_id=request.graph_version_id,
                    use_websocket=request.use_websocket,
                    thread_id_for_analysis=thread_id,
                    org_id=request.org_id  # Pass org_id to convo_stream_pilot
                ):
                    response_count += 1
                    # Format the response as SSE
                    if isinstance(result, str) and result.startswith("data: "):
                        # Already formatted for SSE
                        #logger.info(f"[REQUEST:{request_id}] #{response_count} Yielding string SSE response for thread {thread_id}")
                        yield result + "\n"
                    elif isinstance(result, dict):
                        # Format JSON for SSE
                        #logger.info(f"[REQUEST:{request_id}] #{response_count} Yielding dict response for thread {thread_id}: {str(result)[:100]}...")
                        yield f"data: {json.dumps(result)}\n\n"
                    else:
                        # For string responses without SSE format
                        #logger.info(f"[REQUEST:{request_id}] #{response_count} Yielding message response for thread {thread_id}: {str(result)[:100]}...")
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
        logger.error(f"[REQUEST:{request_id}] Could not acquire lock for thread {thread_id} after 180 seconds")
        logger.info(f"[REQUEST:{request_id}] Yielding timeout error for thread {thread_id}")
        yield f"data: {json.dumps({'error': 'Server busy. Please try again.'})}\n\n"
    except Exception as e:
        logger.error(f"[REQUEST:{request_id}] Error generating pilot SSE stream: {str(e)}")
        logger.error(traceback.format_exc())
        logger.info(f"[REQUEST:{request_id}] Yielding error for thread {thread_id}: {str(e)}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
    finally:
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()
        logger.info(f"[REQUEST:{request_id}] === END pilot SSE request for thread {thread_id} - total time: {elapsed:.2f}s ===")

# Make sure we explicitly define the OPTIONS endpoint for chatwoot webhook
@app.options('/webhook/chatwoot')
async def chatwoot_webhook_options():
    return handle_options()

@app.post('/update-brain-graph')
async def update_brain_graph_endpoint(request: UpdateBrainGraphRequest):
    """Update the name and/or description of a brain graph"""
    from braingraph import update_brain_graph
    try:
        updated_graph = update_brain_graph(
            request.graph_id,
            name=request.name,
            description=request.description
        )
        return {
            "message": "Brain graph updated successfully",
            "brain_graph": {
                "id": updated_graph.id,
                "org_id": updated_graph.org_id,
                "name": updated_graph.name,
                "description": updated_graph.description,
                "created_date": updated_graph.created_date.isoformat()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.options('/update-brain-graph')
async def update_brain_graph_options():
    return handle_options()

# Run the application
if __name__ == "__main__":
    uvicorn.run(socket_app, host="0.0.0.0", port=5001) 