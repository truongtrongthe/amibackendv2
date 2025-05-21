"""
Socket.IO management module for AMI backend (FastAPI version).
This module contains all WebSocket functionality for FastAPI.
"""

import logging
import asyncio
from datetime import datetime
import time
from typing import Dict, Any, List, Set, Optional
import json
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('socket')

# Suppress Socket.IO event emission logs
logging.getLogger('socketio').setLevel(logging.WARNING)
logging.getLogger('engineio').setLevel(logging.WARNING)

# Create a new SocketIO instance - will be initialized in setup_socketio
sio = None

# Session storage for WebSocket connections
ws_sessions = {}
session_lock = asyncio.Lock()

# Storage for undelivered messages
undelivered_messages = {}
message_lock = asyncio.Lock()

# Global counter for tracking session operations
session_modification_counter = 0

# Socket.IO instance
init_count = 0

async def log_session_state(operation, session_id=None, thread_id=None):
    """Log the current state of the sessions dictionary with a stack trace"""
    global session_modification_counter
    session_modification_counter += 1
    
    # Create a snapshot of the current sessions
    async with session_lock:
        session_count = len(ws_sessions)
        session_data = {}
        for sid, data in list(ws_sessions.items()):
            session_data[sid] = {
                'thread_id': data.get('thread_id', 'none'),
                'transport': data.get('transport', 'unknown'),
                'last_activity': data.get('last_activity', 'unknown')
            }
    
    # Log the operation and state
    logger.debug(f"[SESSION_TRACE] [{session_modification_counter}] {operation} - {session_count} sessions active")
    if session_id:
        logger.debug(f"[SESSION_TRACE] [{session_modification_counter}] Session ID: {session_id}")
    if thread_id:
        logger.debug(f"[SESSION_TRACE] [{session_modification_counter}] Thread ID: {thread_id}")
    
    # Get the call stack
    stack = traceback.format_stack()
    relevant_stack = stack[:-2]
    logger.debug(f"[SESSION_TRACE] [{session_modification_counter}] Stack trace:\n{''.join(relevant_stack)}")

def setup_socketio(app_sio):
    """Register the socketio instance from main.py"""
    global sio, init_count
    init_count += 1
    
    if sio is not None:
        logger.warning(f"[SESSION_TRACE] SocketIO already initialized! Init count: {init_count}")
        return sio
        
    sio = app_sio
    logger.info(f"[SESSION_TRACE] Initializing SocketIO for the first time. Init count: {init_count}")
    
    # Register event handlers
    register_handlers()
    
    # Start session cleanup task - deferred to when event loop is running
    logger.info("Session cleanup will start when the event loop is running")
    
    # Create a non-async function to start the cleanup task when the event loop is running
    @sio.on('connect')
    async def _start_cleanup_on_first_connect(sid, environ):
        # Start the cleanup task when the first connection is made (event loop will be running)
        logger.info("Starting cleanup task on first connection")
        asyncio.create_task(start_session_cleanup())
    
    return sio

def register_handlers():
    """Register all Socket.IO event handlers with the FastAPI socketio instance"""
    if not sio:
        logger.error("Cannot register handlers - sio instance not initialized")
        return
    
    @sio.on('connect')
    async def handle_connect(sid, environ):
        """Handle new WebSocket connections"""
        logger.info(f"Client connected: {sid}")
        
        # Get client details from environ
        transport_type = "websocket"
        remote_addr = environ.get('REMOTE_ADDR', 'unknown') if environ else 'unknown'
        
        # CRITICAL FIX: Check if this session ID already exists
        async with session_lock:
            is_reconnect = sid in ws_sessions
            if is_reconnect:
                old_data = ws_sessions[sid]
                old_thread_id = old_data.get('thread_id')
                logger.info(f"RECONNECTION detected for session {sid} with thread {old_thread_id}")
        
        # Log connection info
        logger.info(f"New connection: {sid} | Remote IP: {remote_addr} | Transport: {transport_type}")
        
        # Store connection time for tracking
        await sio.emit('connected', {
            'status': 'connected', 
            'session_id': sid,
            'transport': transport_type,
            'server_time': datetime.now().isoformat()
        }, room=sid)
        
        # CRITICAL FIX: Add pre-registration that preserves thread_id from previous session
        async with session_lock:
            if is_reconnect:
                # If it's a reconnection, preserve the thread_id and other important data
                old_thread_id = ws_sessions[sid].get('thread_id')
                old_user_id = ws_sessions[sid].get('user_id')
                
                # Update the session with new transport but keep the thread_id
                ws_sessions[sid] = {
                    'thread_id': old_thread_id,
                    'user_id': old_user_id,
                    'status': 'connected',
                    'connected_at': datetime.now().isoformat(),
                    'remote_addr': remote_addr,
                    'transport': transport_type,
                    'reconnected': True
                }
                
                # Rejoin the room for the thread
                if old_thread_id:
                    try:
                        await sio.enter_room(sid, old_thread_id)
                        logger.info(f"Automatically rejoined room {old_thread_id} for reconnected session {sid}")
                    except Exception as e:
                        logger.error(f"Error rejoining room {old_thread_id}: {str(e)}")
                        
                logger.info(f"Updated reconnected session {sid} with thread {old_thread_id}")
            else:
                # If it's a new connection, create a new pre-registered session
                ws_sessions[sid] = {
                    'pre_registered': True,
                    'transport': transport_type,
                    'connected_at': datetime.now().isoformat(),
                    'remote_addr': remote_addr,
                    'status': 'connected'
                }
                logger.info(f"Pre-registered new session {sid} with transport {transport_type}")

    @sio.on('register_session')
    async def handle_register(sid, data):
        """Handle client registration with a thread_id"""
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
            
        await sio.enter_room(sid, thread_id)
        await sio.emit('registered', {
            'thread_id': thread_id,
            'status': 'success'
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
        """Handle client disconnection"""
        async with session_lock:
            if sid in ws_sessions:
                thread_id = ws_sessions[sid].get('thread_id')
                del ws_sessions[sid]
                logger.info(f"Client disconnected: {sid} from thread {thread_id}")
            else:
                logger.info(f"Client disconnected: {sid} (no session data)")

    @sio.on('ping')
    async def handle_ping(sid, data=None):
        """Handle client ping to keep session alive"""
        thread_id = None
        async with session_lock:
            if sid in ws_sessions:
                thread_id = ws_sessions[sid].get('thread_id')
                ws_sessions[sid]['last_activity'] = datetime.now().isoformat()
        
        if thread_id:
            await sio.emit('pong', {'thread_id': thread_id, 'timestamp': datetime.now().isoformat()}, room=sid)

    @sio.on('request_missed_messages')
    async def handle_missed_messages_request(sid, data):
        """Handle client request for missed messages"""
        thread_id = data.get('thread_id')
        if not thread_id:
            await sio.emit('error', {'message': 'No thread_id provided for missed messages request'}, room=sid)
            return
            
        async with session_lock:
            if sid in ws_sessions:
                ws_sessions[sid]['last_activity'] = datetime.now().isoformat()
        
        # Send any undelivered messages
        async with message_lock:
            if thread_id in undelivered_messages and undelivered_messages[thread_id]:
                for msg in undelivered_messages[thread_id]:
                    await sio.emit(msg['event'], msg['data'], room=sid)
                logger.info(f"Sent {len(undelivered_messages[thread_id])} missed messages to {sid} for thread {thread_id}")
                undelivered_messages[thread_id] = []
            else:
                await sio.emit('no_missed_messages', {'thread_id': thread_id}, room=sid)

async def emit_analysis_event(thread_id: str, data: Dict[str, Any]) -> bool:
    """
    Emit an analysis event to all clients in a thread room
    
    Returns:
        bool: True if message was delivered to active sessions, False otherwise
    """
    if not sio:
        logger.error("Cannot emit - sio instance not initialized")
        return False
        
    # Check if there are any active sessions in this thread room
    active_sessions_count = 0
    active_session_ids = []
    
    async with session_lock:
        for session_id, session_data in ws_sessions.items():
            stored_thread_id = session_data.get('thread_id')
            
            if stored_thread_id == thread_id:
                active_sessions_count += 1
                active_session_ids.append(session_id)
                # Update last activity timestamp to mark session as active
                session_data['last_activity'] = datetime.now().isoformat()
    
    if active_sessions_count > 0:
        # First try to emit to the room
        try:
            await sio.emit('analysis', data, room=thread_id)
        except Exception as e:
            logger.error(f"Error emitting to room {thread_id}: {str(e)}")
        
        # Also send directly to each session as a backup
        success = False
        for session_id in active_session_ids:
            try:
                await sio.emit('analysis', data, room=session_id)
                success = True
            except Exception as e:
                logger.error(f"Failed direct delivery to session {session_id}: {str(e)}")
        
        # Store message in case not all deliveries were successful
        if not success:
            async with message_lock:
                if thread_id not in undelivered_messages:
                    undelivered_messages[thread_id] = []
                undelivered_messages[thread_id].append({'event': 'analysis', 'data': data})
        
        return True
    else:
        logger.warning(f"No active sessions found for thread {thread_id}, analysis event not delivered")
        
        # Store undelivered message for later retrieval
        async with message_lock:
            if thread_id not in undelivered_messages:
                undelivered_messages[thread_id] = []
            undelivered_messages[thread_id].append({'event': 'analysis', 'data': data})
        
        return False

async def emit_knowledge_event(thread_id: str, data: Dict[str, Any]) -> bool:
    """
    Emit a knowledge event to all clients in a thread room
    
    Returns:
        bool: True if message was delivered to active sessions, False otherwise
    """
    if not sio:
        logger.error("Cannot emit - sio instance not initialized")
        return False
        
    # Check if there are any active sessions in this thread room
    active_sessions_count = 0
    active_session_ids = []
    
    async with session_lock:
        for session_id, session_data in ws_sessions.items():
            stored_thread_id = session_data.get('thread_id')
            
            # Match sessions with the target thread_id
            if stored_thread_id == thread_id:
                active_sessions_count += 1
                active_session_ids.append(session_id)
                # Update last activity timestamp to mark session as active
                session_data['last_activity'] = datetime.now().isoformat()
    
    if active_sessions_count > 0:
        logger.info(f"Emitting knowledge event to thread {thread_id} ({active_sessions_count} active sessions)")
        
        # First try to emit to the room
        try:
            await sio.emit('knowledge', data, room=thread_id)
        except Exception as e:
            logger.error(f"Error emitting knowledge event to room {thread_id}: {str(e)}")
        
        # Also send directly to each session as a backup
        success = False
        for session_id in active_session_ids:
            try:
                await sio.emit('knowledge', data, room=session_id)
                success = True
            except Exception as e:
                logger.error(f"Failed direct knowledge delivery to session {session_id}: {str(e)}")
        
        # Store message in case not all deliveries were successful
        if not success:
            async with message_lock:
                if thread_id not in undelivered_messages:
                    undelivered_messages[thread_id] = []
                undelivered_messages[thread_id].append({'event': 'knowledge', 'data': data})
        
        return True
    else:
        logger.warning(f"No active sessions found for thread {thread_id}, knowledge event not delivered")
        
        # Store undelivered message for later retrieval
        async with message_lock:
            if thread_id not in undelivered_messages:
                undelivered_messages[thread_id] = []
            undelivered_messages[thread_id].append({'event': 'knowledge', 'data': data})
        
        return False

async def emit_next_action_event(thread_id: str, data: Dict[str, Any]) -> bool:
    """
    Emit a next_action event to all clients in a thread room
    
    Returns:
        bool: True if message was delivered to active sessions, False otherwise
    """
    if not sio:
        logger.error("Cannot emit - sio instance not initialized")
        return False
        
    # Check if there are any active sessions in this thread room
    active_sessions_count = 0
    active_session_ids = []
    
    async with session_lock:
        for session_id, session_data in ws_sessions.items():
            stored_thread_id = session_data.get('thread_id')
            
            # Match sessions with the target thread_id
            if stored_thread_id == thread_id:
                active_sessions_count += 1
                active_session_ids.append(session_id)
                # Update last activity timestamp to mark session as active
                session_data['last_activity'] = datetime.now().isoformat()
    
    if active_sessions_count > 0:
        # First try to emit to the room
        try:
            await sio.emit('next_actions', data, room=thread_id)
        except Exception as e:
            logger.error(f"Error emitting next_action event to room {thread_id}: {str(e)}")
        
        # Also send directly to each session as a backup
        success = False
        for session_id in active_session_ids:
            try:
                await sio.emit('next_actions', data, room=session_id)
                success = True
            except Exception as e:
                logger.error(f"Failed direct next_action delivery to session {session_id}: {str(e)}")
        
        # Store message in case not all deliveries were successful
        if not success:
            async with message_lock:
                if thread_id not in undelivered_messages:
                    undelivered_messages[thread_id] = []
                undelivered_messages[thread_id].append({'event': 'next_actions', 'data': data})
        
        return True
    else:
        # Store undelivered message for later retrieval
        async with message_lock:
            if thread_id not in undelivered_messages:
                undelivered_messages[thread_id] = []
            undelivered_messages[thread_id].append({'event': 'next_actions', 'data': data})
        
        return False

async def start_session_cleanup():
    """Start a background task to clean up stale sessions"""
    while True:
        await asyncio.sleep(60)  # Check every 60 seconds
        try:
            now = datetime.now()
            
            # Track sessions that need attention
            stale_sessions = []
            warning_sessions = []  # Sessions approaching timeout but not yet stale
            
            async with session_lock:
                for session_id, session_data in list(ws_sessions.items()):
                    # Check if session has a last_activity
                    last_activity = session_data.get('last_activity', session_data.get('connected_at'))
                    if not last_activity:
                        continue
                        
                    try:
                        if isinstance(last_activity, str):
                            last_activity_time = datetime.fromisoformat(last_activity)
                            inactive_time = (now - last_activity_time).total_seconds()
                            
                            # If session is approaching timeout (>15 min), send a ping attempt
                            if inactive_time > 900 and inactive_time <= 1800:  # Between 15-30 minutes
                                warning_sessions.append((session_id, session_data, inactive_time))
                                logger.warning(f"[SESSION_TRACE] Session {session_id} inactive for {inactive_time:.1f} seconds, sending ping attempt")
                            
                            # If session has been inactive for over 30 minutes, mark it for cleanup
                            elif inactive_time > 1800:  # 30 minutes
                                stale_sessions.append(session_id)
                                logger.warning(f"[SESSION_TRACE] Session {session_id} inactive for {inactive_time:.1f} seconds, marked for cleanup")
                    except Exception as e:
                        logger.error(f"[SESSION_TRACE] Error checking session activity: {str(e)}")
            
            # Try to refresh warning sessions first
            for session_id, session_data, inactive_time in warning_sessions:
                try:
                    thread_id = session_data.get('thread_id')
                    if thread_id:
                        logger.info(f"[SESSION_TRACE] Attempting to refresh session {session_id} for thread {thread_id}")
                        try:
                            # Send a direct ping message to the session
                            await sio.emit('system_ping', {
                                'timestamp': now.isoformat(),
                                'message': 'Session activity check'
                            }, room=session_id)
                            
                            # Update the last ping time as a server-side activity marker
                            async with session_lock:
                                if session_id in ws_sessions:
                                    ws_sessions[session_id]['server_ping'] = now.isoformat()
                                    logger.info(f"[SESSION_TRACE] Sent refresh ping to session {session_id}")
                        except Exception as e:
                            logger.error(f"[SESSION_TRACE] Failed to send refresh ping to session {session_id}: {str(e)}")
                except Exception as e:
                    logger.error(f"[SESSION_TRACE] Error refreshing session {session_id}: {str(e)}")
            
            # Clean up stale sessions
            for session_id in stale_sessions:
                try:
                    async with session_lock:
                        if session_id in ws_sessions:
                            thread_id = ws_sessions[session_id].get('thread_id')
                            if thread_id:
                                await sio.leave_room(session_id, thread_id)
                            del ws_sessions[session_id]
                            logger.info(f"[SESSION_TRACE] Cleaned up stale session {session_id}")
                except Exception as e:
                    logger.error(f"[SESSION_TRACE] Error cleaning up session {session_id}: {str(e)}")
                    
        except Exception as e:
            logger.error(f"[SESSION_TRACE] Error in session cleanup task: {str(e)}") 