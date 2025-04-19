"""
Socket.IO management module for AMI backend.
This module contains all WebSocket functionality, separated from the main HTTP endpoints.
"""

import logging
import threading
from datetime import datetime
import time
from typing import Dict, Any, List, Set, Optional
from flask import request
from flask_socketio import emit, join_room, leave_room, SocketIO
import json
import traceback  # Add this import

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('socket')

# Suppress Socket.IO event emission logs
logging.getLogger('socketio').setLevel(logging.WARNING)
logging.getLogger('engineio').setLevel(logging.WARNING)

# Create a new SocketIO instance - will be initialized in init_socketio
socketio = None

# Session storage for WebSocket connections
ws_sessions = {}
session_lock = threading.RLock()

# Storage for undelivered messages
# Structure:
# {
#   thread_id: {
#     "analysis": [events...],
#     "knowledge": [events...],
#     "other": [events...]
#   }
# }
undelivered_messages = {}
message_lock = threading.RLock()

# Global counter for tracking session operations
session_modification_counter = 0

# Socket.IO instance
socketio = None
init_count = 0

def log_session_state(operation, session_id=None, thread_id=None):
    """Log the current state of the sessions dictionary with a stack trace"""
    global session_modification_counter
    session_modification_counter += 1
    
    # Create a snapshot of the current sessions
    with session_lock:
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
    
    # Log a few session entries if available
    if session_data:
        sample = list(session_data.items())[:3]
        for sid, data in sample:
            logger.debug(f"[SESSION_TRACE] [{session_modification_counter}] Sample - ID: {sid}, Thread: {data['thread_id']}, Last: {data['last_activity']}")
    
    # Get the call stack
    stack = traceback.format_stack()
    # Skip the last two entries (this function and its caller)
    relevant_stack = stack[:-2]
    logger.debug(f"[SESSION_TRACE] [{session_modification_counter}] Stack trace:\n{''.join(relevant_stack)}")

def init_socketio(app):
    """Initialize the Socket.IO instance with the Flask app"""
    global socketio, init_count
    init_count += 1
    
    if socketio is not None:
        logger.warning(f"[SESSION_TRACE] SocketIO already initialized! Init count: {init_count}")
        logger.warning(f"[SESSION_TRACE] This could cause session data loss if reinitializing a new instance")
        # Log stack trace to see where this is being called from
        stack = traceback.format_stack()
        logger.warning(f"[SESSION_TRACE] init_socketio stack trace:\n{''.join(stack[:-1])}")
        # Return existing instance to prevent overwriting
        return socketio
    
    logger.info(f"[SESSION_TRACE] Initializing SocketIO for the first time. Init count: {init_count}")
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet', logger=False, engineio_logger=False)
    
    # Register event handlers
    register_handlers(socketio)
    
    # Start session cleanup thread
    start_session_cleanup()
    
    return socketio

def register_handlers(socketio_instance):
    """Register all Socket.IO event handlers"""
    
    @socketio_instance.on('connect')
    def handle_connect(auth=None):
        """Handle new WebSocket connections"""
        session_id = request.sid
        log_session_state(f"New connection from {session_id}", session_id)
        
        logger.info(f"[SESSION_TRACE] New WebSocket connection: {session_id}, Transport: {request.environ.get('socketio.transport')}")
        # Log more detailed connection information
        logger.info(f"[SESSION_TRACE] Connection details - Headers: {dict(request.headers)}, Remote addr: {request.remote_addr}")
        
        # Get client details
        client_details = {
            'remote_addr': request.remote_addr,
            'user_agent': request.headers.get('User-Agent', 'Unknown'),
            'query_string': request.query_string.decode('utf-8'),
            'origin': request.headers.get('Origin', 'Unknown'),
            'transport': request.args.get('transport', 'Unknown')
        }
        
        # Accept all transport types but log them for debugging
        transport_type = client_details['transport']
        
        # CRITICAL FIX: Check if this session ID already exists
        with session_lock:
            is_reconnect = session_id in ws_sessions
            if is_reconnect:
                old_data = ws_sessions[session_id]
                old_thread_id = old_data.get('thread_id')
                logger.info(f"RECONNECTION detected for session {session_id} with thread {old_thread_id}")
        
        # Log detailed connection info
        logger.info(f"New connection: {session_id} | Remote IP: {request.remote_addr} | "
                    f"Transport: {transport_type} | Origin: {client_details['origin']} | "
                    f"Same IP connections: {len([s for s in ws_sessions.values() if s.get('remote_addr') == request.remote_addr])} | "
                    f"Total active connections: {len(ws_sessions)}")
        
        # Store connection time for tracking
        emit('connected', {
            'status': 'connected', 
            'session_id': session_id,
            'transport': transport_type,
            'server_time': datetime.now().isoformat()
        })
        
        # CRITICAL FIX: Add pre-registration that preserves thread_id from previous session
        with session_lock:
            if is_reconnect:
                # If it's a reconnection, preserve the thread_id and other important data
                old_thread_id = ws_sessions[session_id].get('thread_id')
                old_user_id = ws_sessions[session_id].get('user_id')
                
                # Update the session with new transport but keep the thread_id
                ws_sessions[session_id] = {
                    'thread_id': old_thread_id,
                    'user_id': old_user_id,
                    'status': 'connected',
                    'connected_at': datetime.now().isoformat(),
                    'remote_addr': request.remote_addr,
                    'transport': transport_type,
                    'reconnected': True
                }
                
                # Rejoin the room for the thread
                if old_thread_id:
                    try:
                        join_room(old_thread_id)
                        logger.info(f"Automatically rejoined room {old_thread_id} for reconnected session {session_id}")
                    except Exception as e:
                        logger.error(f"Error rejoining room {old_thread_id}: {str(e)}")
                        
                logger.info(f"Updated reconnected session {session_id} with thread {old_thread_id}")
            else:
                # If it's a new connection, create a new pre-registered session
                ws_sessions[session_id] = {
                    'pre_registered': True,
                    'transport': transport_type,
                    'connected_at': datetime.now().isoformat(),
                    'remote_addr': request.remote_addr,
                    'status': 'connected'
                }
                logger.info(f"Pre-registered new session {session_id} with transport {transport_type}")

    @socketio_instance.on('ping')
    def handle_ping(data=None):
        """Handle client ping to keep session alive"""
        session_id = request.sid
        thread_id = None
        transport = "unknown"
        
        with session_lock:
            if session_id in ws_sessions:
                # Update last ping time
                ws_sessions[session_id]['last_ping'] = datetime.now().isoformat()
                ws_sessions[session_id]['last_activity'] = datetime.now().isoformat()
                
                # Get thread_id and transport for logging
                thread_id = ws_sessions[session_id].get('thread_id')
                transport = ws_sessions[session_id].get('transport', 'unknown')
                
                # Update client_info if provided
                if isinstance(data, dict) and data.get('client_info'):
                    ws_sessions[session_id]['client_info'] = data.get('client_info')
                
                logger.debug(f"[SESSION_TRACE] Received ping from session {session_id} in thread {thread_id}, transport: {transport}")
        
        # Send pong response with session details
        return {
            'pong': True, 
            'timestamp': datetime.now().isoformat(),
            'session_id': session_id,
            'thread_id': thread_id,
            'transport': transport
        }

    @socketio_instance.on('system_ping_response')
    def handle_system_ping_response(data=None):
        """Handle response to system-initiated ping"""
        session_id = request.sid
        with session_lock:
            if session_id in ws_sessions:
                # Update last activity time
                ws_sessions[session_id]['last_activity'] = datetime.now().isoformat()
                ws_sessions[session_id]['system_ping_response'] = datetime.now().isoformat()
                thread_id = ws_sessions[session_id].get('thread_id')
                logger.info(f"[SESSION_TRACE] Received system ping response from session {session_id} for thread {thread_id}")
        
        return {'status': 'acknowledged', 'timestamp': datetime.now().isoformat()}

    @socketio_instance.on('register_session')
    def handle_register(data):
        """Register a client session with conversation details"""
        session_id = request.sid
        thread_id = data.get('thread_id')
        user_id = data.get('user_id', 'thefusionlab')
        
        log_session_state(f"Registering session {session_id}", session_id, thread_id)
        
        if not thread_id:
            from uuid import uuid4
            thread_id = f"thread_{uuid4()}"
            logger.info(f"Generated new thread_id: {thread_id} for session {session_id}")
        
        # Get transport type (accept any transport)
        transport = request.args.get('transport', 'Unknown')
        
        # Check if this session previously exists (reconnection or pre-registration)
        with session_lock:
            prev_data = ws_sessions.get(session_id, {})
            is_reconnect = bool(prev_data)
            was_pre_registered = prev_data.get('pre_registered', False)
            old_thread_id = prev_data.get('thread_id')
            
            # If switching threads, leave the old room first
            if old_thread_id and old_thread_id != thread_id:
                try:
                    leave_room(old_thread_id)
                    logger.info(f"Left old room {old_thread_id} for session {session_id}")
                except Exception as e:
                    logger.error(f"Error leaving old room {old_thread_id}: {str(e)}")
        
        # Check if we have missed messages for this thread
        has_missed_messages = False
        missed_messages_count = 0
        with message_lock:
            has_missed_messages = thread_id in undelivered_messages
            if has_missed_messages:
                missed_messages_count = len(undelivered_messages[thread_id])
            
        # Join the room before registering the session
        try:
            join_room(thread_id)
            logger.info(f"Joined room {thread_id} for session {session_id}")
            is_in_room = True
            
            # Verify room membership (for debug only)
            logger.info(f"Session {session_id} joined room {thread_id}")
        except Exception as e:
            logger.error(f"Error joining room {thread_id} for session {session_id}: {str(e)}")
            is_in_room = False
        
        # Register this session with thread_id
        with session_lock:
            ws_sessions[session_id] = {
                'thread_id': thread_id,
                'user_id': user_id,
                'status': 'ready',
                'connected_at': prev_data.get('connected_at', datetime.now().isoformat()),
                'remote_addr': request.remote_addr,
                'transport': transport,
                'is_in_socketio': True,
                'is_in_room': is_in_room
            }
            
            # Count active sessions for this thread
            thread_sessions = sum(1 for s in ws_sessions.values() if s.get('thread_id') == thread_id)
        
        # Detailed log about registration
        if was_pre_registered:
            logger.info(f"Upgrading pre-registered session {session_id} for thread {thread_id}, user {user_id} | Transport: {transport}")
        elif is_reconnect:
            logger.info(f"Re-registered session {session_id} for thread {thread_id}, user {user_id} | Transport: {transport} | {thread_sessions} active sessions for this thread")
        else:
            logger.info(f"Registered NEW session {session_id} for thread {thread_id}, user {user_id} | Transport: {transport} | {thread_sessions} active sessions for this thread")
        
        # Let client know registration was successful
        status_msg = {
            'status': 'ready',
            'thread_id': thread_id,
            'session_id': session_id,
            'transport': transport,
            'is_in_room': is_in_room
        }
        
        # Add missed messages info if available
        if has_missed_messages:
            status_msg['missed_messages'] = {
                'available': True, 
                'count': missed_messages_count
            }
            logger.info(f"Notifying session {session_id} about {missed_messages_count} missed messages for thread {thread_id}")
        
        emit('session_registered', status_msg)
        
        # Verify the session was properly registered by checking our internal tracking
        with session_lock:
            is_registered = session_id in ws_sessions
            thread_from_session = ws_sessions.get(session_id, {}).get('thread_id')
            logger.info(f"Verification: Session {session_id} registered: {is_registered}, thread: {thread_from_session}")
            if not is_registered or thread_from_session != thread_id:
                logger.error(f"CRITICAL: Session {session_id} registration problem. Registered: {is_registered}, Expected thread: {thread_id}, Actual: {thread_from_session}")
        
        # If client reconnected and there are missed messages, send them automatically
        if has_missed_messages:
            logger.info(f"Auto-delivering {missed_messages_count} missed messages to session {session_id} for thread {thread_id}")
            with message_lock:
                missed_messages = undelivered_messages.get(thread_id, [])
                # Clear the queue after sending
                if thread_id in undelivered_messages:
                    del undelivered_messages[thread_id]
            
            # Send all missed messages
            for msg in missed_messages:
                try:
                    # Try both room and direct delivery
                    emit('analysis_update', msg)
                    logger.info(f"Sent missed message directly to client {session_id}")
                except Exception as e:
                    logger.error(f"Failed to send missed message: {str(e)}")
            
            # Send confirmation that all missed messages were delivered
            emit('missed_messages_delivered', {
                'count': missed_messages_count,
                'thread_id': thread_id
            })
        
        # After registration
        print(f"Current active sessions AFTER registration: {list(ws_sessions.keys())}")
        print(f"Total active sessions: {len(ws_sessions)}")
        print(f"========= REGISTER SESSION COMPLETED =========")

    @socketio_instance.on('disconnect')
    def handle_disconnect(auth=None):
        """Handle client disconnection"""
        session_id = request.sid
        
        log_session_state(f"Disconnecting session {session_id}", session_id)
        
        # Clean up session data
        thread_id = None
        thread_sessions_remaining = 0
        
        with session_lock:
            if session_id in ws_sessions:
                thread_id = ws_sessions[session_id].get('thread_id')
                user_id = ws_sessions[session_id].get('user_id')
                connected_at = ws_sessions[session_id].get('connected_at', 'unknown')
                
                # Calculate session duration if we have a connection timestamp
                duration_str = "unknown duration"
                if connected_at != 'unknown':
                    try:
                        connected_time = datetime.fromisoformat(connected_at)
                        duration_secs = (datetime.now() - connected_time).total_seconds()
                        duration_str = f"{duration_secs:.1f} seconds"
                    except (ValueError, TypeError):
                        pass
                
                if thread_id:
                    leave_room(thread_id)
                    # Count remaining sessions in this thread
                    thread_sessions_remaining = sum(1 for s in ws_sessions.values() 
                                                  if s.get('thread_id') == thread_id and s is not ws_sessions[session_id])
                
                del ws_sessions[session_id]
                logger.info(f"Session {session_id} disconnected and removed | User: {user_id} | Thread: {thread_id} | Duration: {duration_str} | Remaining sessions for thread: {thread_sessions_remaining} | Total active sessions: {len(ws_sessions)}")
            else:
                logger.warning(f"Disconnect received for unknown session: {session_id}")

    @socketio_instance.on('request_missed_messages')
    def handle_missed_messages_request(data):
        """Handle client request for missed messages after reconnection"""
        session_id = request.sid
        thread_id = data.get('thread_id')
        
        if not thread_id:
            logger.warning(f"Missing thread_id in missed messages request from session {session_id}")
            return
        
        logger.info(f"Session {session_id} requested missed messages for thread {thread_id}")
        
        # Get missed messages for this thread
        analysis_messages = []
        knowledge_messages = []
        next_action_messages = []
        other_messages = []
        
        with message_lock:
            if thread_id in undelivered_messages:
                thread_messages = undelivered_messages[thread_id]
                
                # Get messages by type
                if isinstance(thread_messages, dict):
                    # New structured format
                    analysis_messages = thread_messages.get('analysis', [])
                    knowledge_messages = thread_messages.get('knowledge', [])
                    next_action_messages = thread_messages.get('next_action', [])
                    other_messages = thread_messages.get('other', [])
                else:
                    # Legacy format (list) - treat all as analysis
                    analysis_messages = thread_messages
                
                # Clear the queue after retrieving
                del undelivered_messages[thread_id]
        
        # Keep track of message types for logging
        analysis_count = len(analysis_messages)
        knowledge_count = len(knowledge_messages)
        next_action_count = len(next_action_messages)
        other_count = len(other_messages)
        total_count = analysis_count + knowledge_count + next_action_count + other_count
        
        if total_count > 0:
            logger.info(f"Sending {total_count} missed messages to session {session_id} for thread {thread_id}")
            
            # Send all analysis messages
            for msg in analysis_messages:
                emit('analysis_update', msg)
            
            # Send all knowledge messages
            for msg in knowledge_messages:
                emit('knowledge', msg)
            
            # Send all next_action messages
            for msg in next_action_messages:
                emit('next_action', msg)
            
            # Send other message types
            for msg in other_messages:
                if isinstance(msg, dict) and 'type' in msg:
                    # Use the type field if available
                    event_type = msg['type']
                    if event_type == 'analysis':
                        emit('analysis_update', msg)
                    elif event_type == 'knowledge':
                        emit('knowledge', msg)
                    elif event_type == 'next_action':
                        emit('next_action', msg)
                    else:
                        # Try using the type as the event name
                        emit(event_type, msg)
                else:
                    # Default to analysis_update
                    emit('analysis_update', msg)
            
            logger.info(f"Sent {analysis_count} analysis events, {knowledge_count} knowledge events, " 
                      f"{next_action_count} next_action events, and {other_count} other events to session {session_id}")
        else:
            logger.info(f"No missed messages found for thread {thread_id}")
            emit('missed_messages_status', {'status': 'none', 'thread_id': thread_id})

    @socketio_instance.on('force_room_join')
    def force_room_join(data):
        """Force a session to join a room"""
        session_id = request.sid
        thread_id = data.get('thread_id')
        
        if not thread_id:
            emit('error', {'message': 'Missing thread_id parameter'})
            return
        
        # Join the room
        try:
            join_room(thread_id)
            logger.info(f"FORCE JOINED room {thread_id} for session {session_id}")
            
            # Update session data
            with session_lock:
                if session_id in ws_sessions:
                    ws_sessions[session_id]['thread_id'] = thread_id
                    ws_sessions[session_id]['is_in_room'] = True
            
            emit('room_joined', {
                'thread_id': thread_id,
                'session_id': session_id,
                'success': True
            })
        except Exception as e:
            logger.error(f"Error force joining room {thread_id}: {str(e)}")
            emit('error', {'message': f'Failed to join room: {str(e)}'})

    @socketio_instance.on('debug_socket_status')
    def handle_debug_status():
        """Handle debug requests from clients"""
        session_id = request.sid
        
        thread_id = None
        with session_lock:
            if session_id in ws_sessions:
                thread_id = ws_sessions[session_id].get('thread_id')
        
        # Count sessions per thread
        thread_counts = {}
        with session_lock:
            for s_data in ws_sessions.values():
                t_id = s_data.get('thread_id')
                if t_id:
                    thread_counts[t_id] = thread_counts.get(t_id, 0) + 1
        
        emit('debug_status', {
            'session_id': session_id,
            'thread_id': thread_id,
            'session_count': len(ws_sessions),
            'thread_counts': thread_counts,
            'is_registered': bool(thread_id),
            'timestamp': datetime.now().isoformat()
        })


def emit_analysis_event(thread_id: str, data: Dict[str, Any]):
    """
    Emit an analysis event to all clients in a thread room
    
    Returns:
        bool: True if message was delivered to active sessions, False otherwise
    """
    # Check if there are any active sessions in this thread room
    active_sessions_count = 0
    active_session_ids = []
    session_transports = []
    
    with session_lock:
        for session_id, session_data in ws_sessions.items():
            stored_thread_id = session_data.get('thread_id')
            
            if stored_thread_id == thread_id:
                active_sessions_count += 1
                active_session_ids.append(session_id)
                transport = session_data.get('transport', 'unknown')
                session_transports.append(transport)
                # Update last activity timestamp to mark session as active
                session_data['last_activity'] = datetime.now().isoformat()
    
    if active_sessions_count > 0:
        #logger.info(f"Emitting analysis event to thread {thread_id} ({active_sessions_count} active sessions)")
        
        # First try to emit to the room
        try:
            socketio.emit('analysis_update', data, room=thread_id)
        except Exception as e:
            logger.error(f"Error emitting to room {thread_id}: {str(e)}")
        
        # Also send directly to each session as a backup
        success = False
        for session_id in active_session_ids:
            try:
                socketio.emit('analysis_update', data, room=session_id)
                success = True
            except Exception as e:
                logger.error(f"Failed direct delivery to session {session_id}: {str(e)}")
        
        # Store message in case not all deliveries were successful
        if not success:
            with message_lock:
                if thread_id not in undelivered_messages:
                    undelivered_messages[thread_id] = {}
                if 'analysis' not in undelivered_messages[thread_id]:
                    undelivered_messages[thread_id]['analysis'] = []
                undelivered_messages[thread_id]['analysis'] = (undelivered_messages[thread_id]['analysis'] + [data])[-50:]
        
        return True
    else:
        logger.warning(f"No active sessions found for thread {thread_id}, analysis event not delivered")
        
        # Store undelivered message for later retrieval
        with message_lock:
            if thread_id not in undelivered_messages:
                undelivered_messages[thread_id] = {}
            if 'analysis' not in undelivered_messages[thread_id]:
                undelivered_messages[thread_id]['analysis'] = []
            # Only keep the last 50 messages per thread
            undelivered_messages[thread_id]['analysis'] = (undelivered_messages[thread_id]['analysis'] + [data])[-50:]
        
        return False


def emit_knowledge_event(thread_id: str, data: Dict[str, Any]):
    """
    Emit a knowledge event to all clients in a thread room
    
    Args:
        thread_id: The thread ID to send the event to
        data: The knowledge event data to send
        
    Returns:
        bool: True if message was delivered to active sessions, False otherwise
    """
    # Check if there are any active sessions in this thread room
    active_sessions_count = 0
    active_session_ids = []
    
    with session_lock:
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
            socketio.emit('knowledge', data, room=thread_id)
        except Exception as e:
            logger.error(f"Error emitting knowledge event to room {thread_id}: {str(e)}")
        
        # Also send directly to each session as a backup
        success = False
        for session_id in active_session_ids:
            try:
                socketio.emit('knowledge', data, room=session_id)
                success = True
            except Exception as e:
                logger.error(f"Failed direct knowledge delivery to session {session_id}: {str(e)}")
        
        # Store message in case not all deliveries were successful
        if not success:
            with message_lock:
                if thread_id not in undelivered_messages:
                    undelivered_messages[thread_id] = {}
                if 'knowledge' not in undelivered_messages[thread_id]:
                    undelivered_messages[thread_id]['knowledge'] = []
                # Only keep the last 50 messages per thread
                undelivered_messages[thread_id]['knowledge'] = (undelivered_messages[thread_id]['knowledge'] + [data])[-50:]
        
        return True
    else:
        logger.warning(f"No active sessions found for thread {thread_id}, knowledge event not delivered")
        
        # Store undelivered message for later retrieval
        with message_lock:
            if thread_id not in undelivered_messages:
                undelivered_messages[thread_id] = {}
            if 'knowledge' not in undelivered_messages[thread_id]:
                undelivered_messages[thread_id]['knowledge'] = []
            # Only keep the last 50 messages per thread
            undelivered_messages[thread_id]['knowledge'] = (undelivered_messages[thread_id]['knowledge'] + [data])[-50:]
        
        return False


def emit_next_action_event(thread_id: str, data: Dict[str, Any]):
    """
    Emit a next_action event to all clients in a thread room
    
    Args:
        thread_id: The thread ID to send the event to
        data: The next_action event data to send
        
    Returns:
        bool: True if message was delivered to active sessions, False otherwise
    """
    # Check if there are any active sessions in this thread room
    active_sessions_count = 0
    active_session_ids = []
    
    with session_lock:
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
            socketio.emit('next_action', data, room=thread_id)
        except Exception as e:
            logger.error(f"Error emitting next_action event to room {thread_id}: {str(e)}")
        
        # Also send directly to each session as a backup
        success = False
        for session_id in active_session_ids:
            try:
                socketio.emit('next_action', data, room=session_id)
                success = True
            except Exception as e:
                logger.error(f"Failed direct next_action delivery to session {session_id}: {str(e)}")
        
        # Store message in case not all deliveries were successful
        if not success:
            with message_lock:
                if thread_id not in undelivered_messages:
                    undelivered_messages[thread_id] = {}
                if 'next_action' not in undelivered_messages[thread_id]:
                    undelivered_messages[thread_id]['next_action'] = []
                # Only keep the last 50 messages per thread
                undelivered_messages[thread_id]['next_action'] = (undelivered_messages[thread_id]['next_action'] + [data])[-50:]
        
        return True
    else:
        # Store undelivered message for later retrieval
        with message_lock:
            if thread_id not in undelivered_messages:
                undelivered_messages[thread_id] = {}
            if 'next_action' not in undelivered_messages[thread_id]:
                undelivered_messages[thread_id]['next_action'] = []
            # Only keep the last 50 messages per thread
            undelivered_messages[thread_id]['next_action'] = (undelivered_messages[thread_id]['next_action'] + [data])[-50:]
        
        return False


def start_session_cleanup():
    """Start a background thread to clean up stale sessions"""
    def cleanup_worker():
        while True:
            time.sleep(60)  # Check every 60 seconds
            try:
                now = datetime.now()
                #logger.info(f"[SESSION_TRACE] Running session cleanup check at {now.isoformat()}")
                
                # Log current session count before cleanup
                with session_lock:
                    #logger.info(f"[SESSION_TRACE] Current active sessions before cleanup: {len(ws_sessions)}")
                    if ws_sessions:
                        # Log the first few sessions
                        sessions_sample = list(ws_sessions.items())[:3]
                        for sid, session_data in sessions_sample:
                            thread_id = session_data.get('thread_id', 'none')
                            last_activity = session_data.get('last_activity', 'unknown')
                            #logger.info(f"[SESSION_TRACE] Session sample - ID: {sid}, Thread: {thread_id}, Last activity: {last_activity}")
                
                # Track sessions that need attention
                stale_sessions = []
                warning_sessions = []  # Sessions approaching timeout but not yet stale
                
                with session_lock:
                    for session_id, session_data in list(ws_sessions.items()):
                        # Check if session has a last_ping
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
                                elif inactive_time > 1800:  # 30 minutes (was 5 minutes)
                                    stale_sessions.append(session_id)
                                    logger.warning(f"[SESSION_TRACE] Session {session_id} inactive for {inactive_time:.1f} seconds, marked for cleanup")
                        except Exception as e:
                            logger.error(f"[SESSION_TRACE] Error checking session activity: {str(e)}")
                
                # Try to refresh warning sessions first
                for session_id, session_data, inactive_time in warning_sessions:
                    try:
                        thread_id = session_data.get('thread_id')
                        if thread_id:
                            # Send a system ping to the session to see if it's still alive
                            logger.info(f"[SESSION_TRACE] Attempting to refresh session {session_id} for thread {thread_id}")
                            try:
                                # Send a direct ping message to the session
                                socketio.emit('system_ping', {
                                    'timestamp': now.isoformat(),
                                    'message': 'Session activity check'
                                }, room=session_id)
                                
                                # Update the last ping time as a server-side activity marker
                                with session_lock:
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
                        with session_lock:
                            if session_id in ws_sessions:
                                thread_id = ws_sessions[session_id].get('thread_id')
                                if thread_id:
                                    leave_room(thread_id, session_id)
                                del ws_sessions[session_id]
                                logger.info(f"[SESSION_TRACE] Cleaned up stale session {session_id}")
                    except Exception as e:
                        logger.error(f"[SESSION_TRACE] Error cleaning up session {session_id}: {str(e)}")
                        
            except Exception as e:
                logger.error(f"[SESSION_TRACE] Error in session cleanup worker: {str(e)}")
    
    # Start the cleanup thread
    cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
    cleanup_thread.start()
    logger.info("Started session cleanup background thread")
