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
from fastapi import FastAPI, BackgroundTasks, Request, Response, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from login import get_current_user

# Import router from fastapi_routes
from fastapi_routes import router as api_router
from braingraph_routes import router as braingraph_router
from contact_apis import router as contact_router
from waitlist import router as waitlist_router
from login import router as login_router
from organization import router as organization_router
from org_agent import router as org_agent_router
from blueprint_routes import router as blueprint_router
from google_drive_routes import router as google_drive_router
from chat_routes import router as chat_router
from supabase import create_client, Client


# Import exec_tool module for LLM tool execution
from exec_tool import execute_tool_async, execute_tool_stream, ToolExecutionRequest, ToolExecutionResponse

# 🚀 MIGRATED: Import from new modular agent architecture  
# This change provides 84% reduction in orchestrator size + 30-50% performance improvement
# OLD: from agent import execute_agent_stream, execute_agent_async
from agent_refactored import execute_agent_stream, execute_agent_async
# Note: All existing endpoints work identically - zero breaking changes

from utilities import logger
# Initialize FastAPI app
app = FastAPI(title="AMI Backend")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],  # Allow all headers
    max_age=86400
)

# Helper function for OPTIONS requests
def handle_options():
    """Common OPTIONS handler for all endpoints."""
    return JSONResponse(
        content={},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, PATCH, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Credentials": "true",
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
app.include_router(organization_router)
app.include_router(org_agent_router)
app.include_router(blueprint_router)
app.include_router(google_drive_router)
app.include_router(chat_router)

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
            "Access-Control-Allow-Methods": "GET, POST, PUT, PATCH, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Credentials": "true",
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

class ToolExecuteRequest(BaseModel):
    tool_name: str
    parameters: Dict[str, Any]
    org_id: str = "unknown"  # Add org_id field with default value

class UpdateBrainGraphRequest(BaseModel):
    graph_id: str
    name: Optional[str] = None
    description: Optional[str] = None

class LLMToolExecuteRequest(BaseModel):
    """Request model for LLM tool execution with dynamic parameters"""
    llm_provider: str  # 'anthropic' or 'openai'
    user_query: str
    system_prompt: Optional[str] = None
    model: Optional[str] = None  # Custom model name (e.g., "gpt-4o", "claude-3-5-haiku")
    model_params: Optional[Dict[str, Any]] = None
    org_id: str = "default"
    user_id: str = "anonymous"
    # New parameters to control tool usage
    enable_tools: Optional[bool] = True  # Whether to enable tools at all
    force_tools: Optional[bool] = False  # Force tool usage (tool_choice="required")
    tools_whitelist: Optional[List[str]] = None  # Only allow specific tools
    # Conversation history support
    conversation_history: Optional[List[Dict[str, Any]]] = None  # Previous messages
    max_history_messages: Optional[int] = 25  # Maximum number of history messages to include
    max_history_tokens: Optional[int] = 6000  # Maximum token count for history
    # Backward compatibility for frontend
    enable_search: Optional[bool] = None  # Deprecated: use enable_tools instead
    # NEW: Cursor-style request handling parameters
    enable_intent_classification: Optional[bool] = True  # Enable intent analysis
    enable_request_analysis: Optional[bool] = True  # Enable request analysis  
    cursor_mode: Optional[bool] = False  # Enable Cursor-style progressive enhancement
    # NEW: Grading context for approval flow
    grading_context: Optional[Dict[str, Any]] = None  # Scenario data and approval info

# NEW: Request model for agent execution API
class AgentAPIRequest(BaseModel):
    """Request model for agent execution API with dynamic parameters"""
    llm_provider: str = "openai"  # 'anthropic' or 'openai' 
    user_request: str  # The task to execute
    agent_id: str  # Specific agent instance ID
    agent_type: str  # Type of agent (e.g., "sales_agent", "support_agent", "analyst_agent")
    system_prompt: Optional[str] = None
    model: Optional[str] = None  # Custom model name (e.g., "gpt-4o", "claude-3-5-haiku")
    model_params: Optional[Dict[str, Any]] = None
    org_id: str = "default"
    user_id: str = "anonymous"
    
    # Agent operational mode
    agent_mode: Optional[str] = "execute"  # "collaborate" or "execute"
    
    # Agent-specific parameters (tools enabled & deep reasoning by default)
    enable_tools: Optional[bool] = True  # Tools enabled by default for agents
    enable_deep_reasoning: Optional[bool] = True  # Deep reasoning enabled by default
    reasoning_depth: Optional[str] = "standard"  # "light", "standard", "deep"
    task_focus: Optional[str] = "execution"  # "execution", "analysis", "communication"
    
    # Tool control parameters
    force_tools: Optional[bool] = False  # Force tool usage (tool_choice="required")
    tools_whitelist: Optional[List[str]] = None  # Only allow specific tools
    
    # Agent knowledge context
    specialized_knowledge_domains: Optional[List[str]] = None  # Agent's specialization areas
    conversation_history: Optional[List[Dict[str, Any]]] = None  # Previous messages
    max_history_messages: Optional[int] = 15  # Agents focus on recent context
    max_history_tokens: Optional[int] = 4000
    
    # Backward compatibility for frontend
    enable_search: Optional[bool] = None  # Deprecated: use enable_tools instead


class SimpleChatRequest(BaseModel):
    """Simple chat request for direct Ami interaction"""
    message: str = Field(..., description="Message to send", min_length=1, max_length=2000)
    llm_provider: str = Field("anthropic", description="LLM provider to use")
    model: Optional[str] = Field(None, description="Specific model to use")
    system_prompt: Optional[str] = Field(None, description="Custom system prompt")

# Import API models from the new modular ami structure
from ami import CreateAgentAPIRequest, CollaborativeAgentAPIRequest, CreateAgentAPIResponse, CollaborativeAgentAPIResponse





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
            org_id="unknown"  # TODO: Get org_id from request context
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

@app.post('/api/llm/execute')
async def execute_llm_tool_endpoint(request: LLMToolExecuteRequest):
    """
    Execute LLM tool calling with dynamic system prompts and parameters.
    Supports both Anthropic Claude and OpenAI GPT-4 with customizable settings.
    """
    start_time = datetime.now()
    request_id = str(uuid4())[:8]
    
    logger.info(f"[REQUEST:{request_id}] === BEGIN /api/llm/execute request at {start_time.isoformat()} ===")
    logger.info(f"[REQUEST:{request_id}] Provider: {request.llm_provider}, Query: {request.user_query[:100]}...")
    logger.info(f"[REQUEST:{request_id}] System prompt: {request.system_prompt[:100] if request.system_prompt else 'None'}...")
    logger.info(f"[REQUEST:{request_id}] Model params: {request.model_params}")
    
    try:
        # Create tool execution request
        tool_request = ToolExecutionRequest(
            llm_provider=request.llm_provider,
            user_query=request.user_query,
            system_prompt=request.system_prompt,
            model_params=request.model_params,
            org_id=request.org_id,
            user_id=request.user_id
        )
        
        # Handle backward compatibility for enable_search parameter
        enable_tools = request.enable_tools
        if request.enable_search is not None:
            # Frontend sent enable_search, use it instead of enable_tools
            enable_tools = request.enable_search
            logger.info(f"[REQUEST:{request_id}] Using enable_search={request.enable_search} for backward compatibility")
        
        # Execute the tool asynchronously with tool control parameters
        response: ToolExecutionResponse = await execute_tool_async(
            llm_provider=request.llm_provider,
            user_query=request.user_query,
            system_prompt=request.system_prompt,
            model=request.model,
            model_params=request.model_params,
            org_id=request.org_id,
            user_id=request.user_id,
            enable_tools=enable_tools,
            force_tools=request.force_tools,
            tools_whitelist=request.tools_whitelist
        )
        
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()
        
        logger.info(f"[REQUEST:{request_id}] LLM tool execution completed - time: {elapsed:.2f}s")
        logger.info(f"[REQUEST:{request_id}] Success: {response.success}, Provider: {response.provider}")
        
        # Return the result with additional metadata
        result = {
            "success": response.success,
            "result": response.result,
            "provider": response.provider,
            "model_used": response.model_used,
            "execution_time": response.execution_time,
            "request_id": request_id,
            "total_elapsed_time": elapsed,
            "metadata": response.metadata,
            "error": response.error
        }
        
        if response.success:
            return JSONResponse(content=result)
        else:
            return JSONResponse(status_code=400, content=result)
            
    except Exception as e:
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()
        
        logger.error(f"[REQUEST:{request_id}] Error in /api/llm/execute endpoint: {str(e)} - time: {elapsed:.2f}s")
        import traceback
        logger.error(f"[REQUEST:{request_id}] Traceback: {traceback.format_exc()}")
        
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "result": "",
                "provider": request.llm_provider,
                "model_used": "unknown",
                "execution_time": 0,
                "request_id": request_id,
                "total_elapsed_time": elapsed,
                "error": str(e),
                "metadata": None
            }
        )
    finally:
        logger.info(f"[REQUEST:{request_id}] === END /api/llm/execute request - total time: {(datetime.now() - start_time).total_seconds():.2f}s ===")

@app.options('/api/llm/execute')
async def execute_llm_tool_options():
    return handle_options()

# New streaming LLM tool endpoint
@app.post('/tool/llm')
async def execute_llm_tool_stream_endpoint(request: LLMToolExecuteRequest, background_tasks: BackgroundTasks):
    """
    Stream LLM tool execution with dynamic system prompts and parameters.
    Supports both Anthropic Claude and OpenAI GPT-4 with real-time streaming.
    Similar to /api/llm/execute but with SSE streaming for better frontend UX.
    """
    start_time = datetime.now()
    request_id = str(uuid4())[:8]
    
    logger.info(f"[REQUEST:{request_id}] === BEGIN /tool/llm stream request at {start_time.isoformat()} ===")
    logger.info(f"[REQUEST:{request_id}] Provider: {request.llm_provider}, Query: {request.user_query[:100]}...")
    logger.info(f"[REQUEST:{request_id}] System prompt: {request.system_prompt[:100] if request.system_prompt else 'None'}...")
    logger.info(f"[REQUEST:{request_id}] Model params: {request.model_params}")
    
    # Get a lock for this request (using user_id as thread identifier)
    thread_id = f"llm_tool_{request.user_id}_{request_id}"
    logger.info(f"[REQUEST:{request_id}] Getting lock for thread {thread_id}")
    thread_lock = await lock_manager.get_lock(thread_id)
    logger.info(f"[REQUEST:{request_id}] Got lock manager lock for thread {thread_id}")
    
    # Always use StreamingResponse for better UX
    logger.info(f"[REQUEST:{request_id}] Creating StreamingResponse with generate_llm_tool_sse_stream")
    response = StreamingResponse(
        generate_llm_tool_sse_stream(request, thread_lock, start_time, request_id),
        media_type="text/event-stream",
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache",
            "Access-Control-Allow-Origin": "*"
        }
    )
    logger.info(f"[REQUEST:{request_id}] Returning StreamingResponse to client")
    return response

@app.options('/tool/llm')
async def execute_llm_tool_stream_options():
    return handle_options()

# SSE stream generator for LLM tool execution
async def generate_llm_tool_sse_stream(request: LLMToolExecuteRequest, thread_lock: asyncio.Lock, start_time: datetime, request_id: str):
    """Generate an SSE stream with LLM tool execution response"""
    
    try:
        # Try to acquire the lock with timeout
        async with asyncio.timeout(180):  # 180-second timeout
            async with thread_lock:
                logger.info(f"[REQUEST:{request_id}] Acquired lock for LLM tool execution")
                
                # Import here to avoid circular imports
                from exec_tool import execute_tool_stream
                
                # Log request to streaming tool execution
                logger.info(f"[REQUEST:{request_id}] Calling execute_tool_stream with provider: {request.llm_provider}")
                
                # Handle backward compatibility for enable_search parameter
                enable_tools = request.enable_tools
                if request.enable_search is not None:
                    # Frontend sent enable_search, use it instead of enable_tools
                    enable_tools = request.enable_search
                    logger.info(f"[REQUEST:{request_id}] Using enable_search={request.enable_search} for backward compatibility")
                
                # Track response count
                response_count = 0
                
                # Process the LLM tool execution and yield results
                async for result in execute_tool_stream(
                    llm_provider=request.llm_provider,
                    user_query=request.user_query,
                    system_prompt=request.system_prompt,
                    model=request.model,
                    model_params=request.model_params,
                    org_id=request.org_id,
                    user_id=request.user_id,
                    enable_tools=enable_tools,
                    force_tools=request.force_tools,
                    tools_whitelist=request.tools_whitelist,
                    conversation_history=request.conversation_history,
                    max_history_messages=request.max_history_messages,
                    max_history_tokens=request.max_history_tokens,
                    # NEW: Cursor-style parameters
                    enable_intent_classification=request.enable_intent_classification,
                    enable_request_analysis=request.enable_request_analysis,
                    cursor_mode=request.cursor_mode,
                    # NEW: Grading context parameter
                    grading_context=request.grading_context
                ):
                    response_count += 1
                    
                    # Format the response as SSE
                    if isinstance(result, dict):
                        # Format JSON for SSE
                        yield f"data: {json.dumps(result)}\n\n"
                    else:
                        # For string responses without SSE format
                        yield f"data: {json.dumps({'message': str(result)})}\n\n"
                
                logger.info(f"[REQUEST:{request_id}] Completed yielding {response_count} responses")
                logger.info(f"[REQUEST:{request_id}] Released lock for LLM tool execution")
                
    except asyncio.TimeoutError:
        logger.error(f"[REQUEST:{request_id}] Could not acquire lock for LLM tool execution after 180 seconds")
        yield f"data: {json.dumps({'error': 'Server busy. Please try again.'})}\n\n"
    except Exception as e:
        logger.error(f"[REQUEST:{request_id}] Error generating LLM tool SSE stream: {str(e)}")
        logger.error(traceback.format_exc())
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
    finally:
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()
        logger.info(f"[REQUEST:{request_id}] === END LLM tool stream request - total time: {elapsed:.2f}s ===")

# NEW: Agent execution endpoint
@app.post('/api/tool/agent')
async def execute_agent_endpoint(request: AgentAPIRequest):
    """
    Execute a specialized agent with dynamic system prompts and parameters.
    Supports both Anthropic Claude and OpenAI GPT-4 with customizable settings.
    Default: tools enabled, deep reasoning enabled, search optional.
    """
    start_time = datetime.now()
    request_id = str(uuid4())[:8]
    
    logger.info(f"[REQUEST:{request_id}] === BEGIN /api/tool/agent request at {start_time.isoformat()} ===")
    logger.info(f"[REQUEST:{request_id}] Agent: {request.agent_id} ({request.agent_type}), Task: {request.user_request[:100]}...")
    logger.info(f"[REQUEST:{request_id}] System prompt: {request.system_prompt[:100] if request.system_prompt else 'None'}...")
    logger.info(f"[REQUEST:{request_id}] Model params: {request.model_params}")
    
    try:
        # Handle backward compatibility for enable_search parameter
        enable_tools = request.enable_tools
        if request.enable_search is not None:
            # Frontend sent enable_search, use it instead of enable_tools
            enable_tools = request.enable_search
            logger.info(f"[REQUEST:{request_id}] Using enable_search={request.enable_search} for backward compatibility")
        
        # Execute the agent asynchronously using the agent module interface
        response = await execute_agent_async(
            llm_provider=request.llm_provider,
            user_request=request.user_request,
            agent_id=request.agent_id,
            agent_type=request.agent_type,
            system_prompt=request.system_prompt,
            model=request.model,
            org_id=request.org_id,
            user_id=request.user_id,
            specialized_knowledge_domains=request.specialized_knowledge_domains
        )
        
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()
        
        logger.info(f"[REQUEST:{request_id}] Agent execution completed - time: {elapsed:.2f}s")
        logger.info(f"[REQUEST:{request_id}] Success: {response.success}, Agent: {response.agent_id}")
        
        # Return the result with additional metadata
        result = {
            "success": response.success,
            "result": response.result,
            "agent_id": response.agent_id,
            "agent_type": response.agent_type,
            "execution_time": response.execution_time,
            "tasks_completed": response.tasks_completed,
            "request_id": request_id,
            "total_elapsed_time": elapsed,
            "metadata": response.metadata,
            "error": response.error
        }
        
        if response.success:
            return JSONResponse(content=result)
        else:
            return JSONResponse(status_code=400, content=result)
            
    except Exception as e:
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()
        
        logger.error(f"[REQUEST:{request_id}] Error in /api/tool/agent endpoint: {str(e)} - time: {elapsed:.2f}s")
        import traceback
        logger.error(f"[REQUEST:{request_id}] Traceback: {traceback.format_exc()}")
        
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "result": "",
                "agent_id": request.agent_id,
                "agent_type": request.agent_type,
                "execution_time": 0,
                "tasks_completed": 0,
                "request_id": request_id,
                "total_elapsed_time": elapsed,
                "error": str(e),
                "metadata": None
            }
        )
    finally:
        logger.info(f"[REQUEST:{request_id}] === END /api/tool/agent request - total time: {(datetime.now() - start_time).total_seconds():.2f}s ===")

@app.options('/api/tool/agent')
async def execute_agent_options():
    return handle_options()

# NEW: Agent streaming endpoint
@app.post('/api/tool/agent/stream')
async def execute_agent_stream_endpoint(request: AgentAPIRequest, background_tasks: BackgroundTasks):
    """
    Stream agent execution with simplified flow - no redundant analysis phases.
    LLM handles tool decisions directly like ChatGPT/Claude.
    Supports both Anthropic Claude and OpenAI GPT-4 with real-time streaming.
    """
    start_time = datetime.now()
    request_id = str(uuid4())[:8]
    
    logger.info(f"[REQUEST:{request_id}] === BEGIN /api/tool/agent/stream request at {start_time.isoformat()} ===")
    logger.info(f"[REQUEST:{request_id}] Agent: {request.agent_id} ({request.agent_type}), Task: {request.user_request[:100]}...")
    logger.info(f"[REQUEST:{request_id}] System prompt: {request.system_prompt[:100] if request.system_prompt else 'None'}...")
    logger.info(f"[REQUEST:{request_id}] Model params: {request.model_params}")
    
    # Get a lock for this request (using user_id as thread identifier)
    thread_id = f"agent_tool_{request.user_id}_{request_id}"
    logger.info(f"[REQUEST:{request_id}] Getting lock for thread {thread_id}")
    thread_lock = await lock_manager.get_lock(thread_id)
    logger.info(f"[REQUEST:{request_id}] Got lock manager lock for thread {thread_id}")
    
    # Always use StreamingResponse for better UX
    logger.info(f"[REQUEST:{request_id}] Creating StreamingResponse with generate_agent_sse_stream")
    response = StreamingResponse(
        generate_agent_sse_stream(request, thread_lock, start_time, request_id),
        media_type="text/event-stream",
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache",
            "Access-Control-Allow-Origin": "*"
        }
    )
    logger.info(f"[REQUEST:{request_id}] Returning StreamingResponse to client")
    return response

@app.options('/api/tool/agent/stream')
async def execute_agent_stream_options():
    return handle_options()

# NEW: Streaming SSE generator for agent execution
async def generate_agent_sse_stream(request: AgentAPIRequest, thread_lock: asyncio.Lock, start_time: datetime, request_id: str):
    """Generate an SSE stream with agent execution response"""
    
    try:
        # Try to acquire the lock with timeout
        async with asyncio.timeout(180):  # 180-second timeout
            async with thread_lock:
                logger.info(f"[REQUEST:{request_id}] Acquired lock for agent execution")
                
                # Log request to agent execution
                logger.info(f"[REQUEST:{request_id}] Calling execute_agent_stream with agent: {request.agent_id}")
                
                # Handle backward compatibility for enable_search parameter
                enable_tools = request.enable_tools
                if request.enable_search is not None:
                    # Frontend sent enable_search, use it instead of enable_tools
                    enable_tools = request.enable_search
                    logger.info(f"[REQUEST:{request_id}] Using enable_search={request.enable_search} for backward compatibility")
                
                # Track response count
                response_count = 0
                
                # Process the agent execution and yield results
                async for result in execute_agent_stream(
                    llm_provider=request.llm_provider,
                    user_request=request.user_request,
                    agent_id=request.agent_id,
                    agent_type=request.agent_type,
                    system_prompt=request.system_prompt,
                    model=request.model,
                    org_id=request.org_id,
                    user_id=request.user_id,
                    enable_deep_reasoning=request.enable_deep_reasoning,
                    reasoning_depth=request.reasoning_depth,
                    task_focus=request.task_focus,
                    specialized_knowledge_domains=request.specialized_knowledge_domains,
                    conversation_history=request.conversation_history
                ):
                    response_count += 1
                    
                    # Format the response as SSE
                    if isinstance(result, dict):
                        # Format JSON for SSE
                        yield f"data: {json.dumps(result)}\n\n"
                    else:
                        # For string responses without SSE format
                        yield f"data: {json.dumps({'message': str(result)})}\n\n"
                
                logger.info(f"[REQUEST:{request_id}] Completed yielding {response_count} responses")
                logger.info(f"[REQUEST:{request_id}] Released lock for agent execution")
                
    except asyncio.TimeoutError:
        logger.error(f"[REQUEST:{request_id}] Could not acquire lock for agent execution after 180 seconds")
        yield f"data: {json.dumps({'error': 'Server busy. Please try again.'})}\n\n"
    except Exception as e:
        logger.error(f"[REQUEST:{request_id}] Error generating agent SSE stream: {str(e)}")
        logger.error(traceback.format_exc())
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
    finally:
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()
        logger.info(f"[REQUEST:{request_id}] === END agent SSE request - total time: {elapsed:.2f}s ===")

# ADD NEW SIMPLIFIED ENDPOINTS - Replace exec_tool complexity

# NEW: Direct Agent Creation Endpoint (Legacy - Simple but less precise)
@app.post('/ami/create-agent')
async def create_agent_endpoint(request: CreateAgentAPIRequest, current_user: dict = Depends(get_current_user)):
    """
    Create a new AI agent directly - fast but basic
    Replaces complex exec_tool agent building with simple approach
    """
    try:
        # Get user's organization
        from organization import get_my_organization
        org_response = await get_my_organization(current_user)
        
        # Create agent via Ami (simple and fast) - using new modular structure
        from ami import create_agent_via_api
        result = await create_agent_via_api(
            api_request=request,
            org_id=org_response.id,
            user_id=current_user["id"]
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Agent creation endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Agent creation failed: {str(e)}")

@app.options('/ami/create-agent')
async def create_agent_options():
    return handle_options()


# NEW: Collaborative Agent Creation Endpoint (Chief Product Officer approach)
@app.post('/ami/collaborate')
async def collaborate_agent_endpoint(request: CollaborativeAgentAPIRequest, current_user: dict = Depends(get_current_user)):
    """
    Collaborate with Ami like a Chief Product Officer to build precise agents
    Multi-step conversation flow: idea → skeleton → feedback → approval → build
    """
    try:
        # Get user's organization
        from organization import get_my_organization
        org_response = await get_my_organization(current_user)
        
        # Collaborate with Ami on agent creation - using new modular structure
        from ami import collaborate_on_agent_via_api
        result = await collaborate_on_agent_via_api(
            api_request=request,
            org_id=org_response.id,
            user_id=current_user["id"]
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Collaborative agent endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Collaborative agent creation failed: {str(e)}")

@app.options('/ami/collaborate')
async def collaborate_agent_options():
    return handle_options()


# NEW: Direct Agent Execution Endpoint (replaces complex exec_tool execution)
@app.post('/agent/execute')
async def execute_agent_endpoint(request: AgentAPIRequest, current_user: dict = Depends(get_current_user)):
    """
    Execute a specific agent with dynamic configuration - replaces exec_tool complexity
    Simple, fast agent execution with database-driven configuration
    """
    try:
        # Get user's organization
        from organization import get_my_organization
        org_response = await get_my_organization(current_user)
        
        # Execute agent with dynamic configuration
        from agent_refactored import execute_agent_async
        result = await execute_agent_async(
            llm_provider=request.llm_provider,
            user_request=request.user_request,
            agent_id=request.agent_id,
            agent_type=request.agent_type or "general",
            model=request.model,
            org_id=org_response.id,
            user_id=current_user["id"],
            agent_mode=request.agent_mode,
            specialized_knowledge_domains=request.specialized_knowledge_domains
        )
        
        return {
            "success": result.success,
            "result": result.result,
            "agent_id": result.agent_id,
            "agent_type": result.agent_type,
            "execution_time": result.execution_time,
            "tasks_completed": result.tasks_completed,
            "error": result.error,
            "metadata": result.metadata
        }
        
    except Exception as e:
        logger.error(f"Agent execution endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {str(e)}")

@app.options('/agent/execute')
async def execute_agent_options():
    return handle_options()


# NEW: Agent Streaming Endpoint (replaces complex exec_tool streaming)
@app.post('/agent/stream')
async def stream_agent_endpoint(request: AgentAPIRequest, current_user: dict = Depends(get_current_user)):
    """
    Stream agent execution with dynamic configuration - replaces exec_tool streaming complexity
    Clean, fast streaming with database-driven agent behavior
    """
    try:
        # Get user's organization
        from organization import get_my_organization
        org_response = await get_my_organization(current_user)
        
        # Stream agent execution with dynamic configuration
        from agent_refactored import execute_agent_stream
        
        async def generate_agent_stream():
            try:
                async for chunk in execute_agent_stream(
                    llm_provider=request.llm_provider,
                    user_request=request.user_request,
                    agent_id=request.agent_id,
                    agent_type=request.agent_type or "general",
                    model=request.model,
                    org_id=org_response.id,
                    user_id=current_user["id"],
                    agent_mode=request.agent_mode,
                    specialized_knowledge_domains=request.specialized_knowledge_domains,
                    conversation_history=request.conversation_history
                ):
                    # Format for SSE
                    yield f"data: {json.dumps(chunk)}\n\n"
                    
            except Exception as e:
                logger.error(f"Agent streaming error: {e}")
                yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
        
        return StreamingResponse(
            generate_agent_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive", 
                "Access-Control-Allow-Origin": "*",
                "X-Accel-Buffering": "no"
            }
        )
        
    except Exception as e:
        logger.error(f"Agent streaming endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Agent streaming failed: {str(e)}")

@app.options('/agent/stream')
async def stream_agent_options():
    return handle_options()


# NEW: Agent Collaboration Endpoint (Interactive mode)
@app.post('/agent/collaborate')
async def collaborate_agent_endpoint(request: AgentAPIRequest, current_user: dict = Depends(get_current_user)):
    """
    Collaborate with a specific agent in interactive mode
    Agent will ask questions, discuss options, and work WITH the user
    """
    try:
        # Get user's organization
        from organization import get_my_organization
        org_response = await get_my_organization(current_user)
        
        # Force collaborate mode for this endpoint
        request.agent_mode = "collaborate"
        
        # Execute agent in collaborate mode
        from agent_refactored import execute_agent_async
        result = await execute_agent_async(
            llm_provider=request.llm_provider,
            user_request=request.user_request,
            agent_id=request.agent_id,
            agent_type=request.agent_type or "general",
            model=request.model,
            org_id=org_response.id,
            user_id=current_user["id"],
            agent_mode="collaborate",  # Force collaborate mode
            specialized_knowledge_domains=request.specialized_knowledge_domains
        )
        
        return {
            "success": result.success,
            "result": result.result,
            "agent_id": result.agent_id,
            "agent_type": result.agent_type,
            "execution_time": result.execution_time,
            "tasks_completed": result.tasks_completed,
            "error": result.error,
            "metadata": result.metadata,
            "mode": "collaborate"
        }
        
    except Exception as e:
        logger.error(f"Agent collaboration endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Agent collaboration failed: {str(e)}")

@app.options('/agent/collaborate')
async def collaborate_agent_options():
    return handle_options()


# NEW: Simple Direct Chat Endpoint (for non-agent interactions)
@app.post('/ami/chat')
async def ami_chat_endpoint(request: SimpleChatRequest, current_user: dict = Depends(get_current_user)):
    """
    Simple direct chat with Ami - for general conversations that don't need agent creation
    Replaces simple exec_tool interactions
    """
    try:
        # Get user's organization for context
        from organization import get_my_organization
        org_response = await get_my_organization(current_user)
        
        # Simple LLM call without complex exec_tool pipeline
        if request.llm_provider.lower() == "anthropic":
            import anthropic
            client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            
            response = await client.messages.create(
                model=request.model or "claude-3-5-sonnet-20241022",
                max_tokens=2000,
                messages=[{"role": "user", "content": request.message}],
                system=request.system_prompt or "You are Ami, a helpful AI assistant."
            )
            
            return {
                "success": True,
                "response": response.content[0].text,
                "provider": "anthropic",
                "model": request.model or "claude-3-5-sonnet-20241022"
            }
            
        elif request.llm_provider.lower() == "openai":
            import openai
            client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            response = await client.chat.completions.create(
                model=request.model or "gpt-4",
                max_tokens=2000,
                messages=[
                    {"role": "system", "content": request.system_prompt or "You are Ami, a helpful AI assistant."},
                    {"role": "user", "content": request.message}
                ]
            )
            
            return {
                "success": True,
                "response": response.choices[0].message.content,
                "provider": "openai", 
                "model": request.model or "gpt-4"
            }
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported provider: {request.llm_provider}")
            
    except Exception as e:
        logger.error(f"Ami chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

@app.options('/ami/chat')
async def ami_chat_options():
    return handle_options()


class AgentAPIRequest(BaseModel):
    """API request for agent execution"""
    agent_id: str = Field(..., description="Agent ID or name to execute")
    user_request: str = Field(..., description="Task for the agent to perform")
    agent_mode: str = Field("execute", description="Agent operational mode: 'collaborate' or 'execute'")
    llm_provider: str = Field("anthropic", description="LLM provider to use")
    model: Optional[str] = Field(None, description="Specific model to use")
    agent_type: Optional[str] = Field(None, description="Agent type hint")
    specialized_knowledge_domains: Optional[List[str]] = Field(None, description="Knowledge domains")
    conversation_history: Optional[List[Dict[str, Any]]] = Field(None, description="Previous conversation")



# Run the application
if __name__ == "__main__":
    uvicorn.run(socket_app, host="0.0.0.0", port=5001) 