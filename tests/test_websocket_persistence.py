#!/usr/bin/env python
"""
WebSocket Session Persistence Test

This script tests whether WebSocket sessions are properly maintained during API requests.
It establishes a WebSocket connection, registers a session, then makes an API request
and verifies the session remains active throughout the process.
"""

import asyncio
import sys
import json
import time
import uuid
from datetime import datetime
import socketio  # pip install python-socketio
import requests  # pip install requests
import aiohttp   # pip install aiohttp

SERVER_URL = 'http://localhost:5001'  # Update if your server is on a different port
THREAD_ID = f"thread_test_{uuid.uuid4()}"
TEST_MESSAGE = "This is a test message"
TIMEOUT = 60  # seconds to wait for completion

# Socket.IO client for async operations
sio = socketio.AsyncClient(logger=True, engineio_logger=True)

# Global state
connected = False
registered = False
messages_received = []
undelivered_queried = False

@sio.event
async def connect():
    global connected
    print(f"[{datetime.now().isoformat()}] Connected to server")
    connected = True

@sio.event
async def disconnect():
    global connected
    print(f"[{datetime.now().isoformat()}] Disconnected from server")
    connected = False

@sio.event
async def connect_error(data):
    print(f"[{datetime.now().isoformat()}] Connection error: {data}")

@sio.event
async def session_registered(data):
    global registered
    print(f"[{datetime.now().isoformat()}] Session registered: {data}")
    registered = True

@sio.event
async def analysis_update(data):
    print(f"[{datetime.now().isoformat()}] Received analysis update: {data.get('content', '')[:50]}...")
    messages_received.append(data)

@sio.event
async def missed_messages(data):
    print(f"[{datetime.now().isoformat()}] Received missed messages: {len(data.get('messages', []))} messages")
    for msg in data.get('messages', []):
        messages_received.append(msg)

async def register_session(thread_id):
    """Register a session with the server"""
    print(f"[{datetime.now().isoformat()}] Registering session with thread_id: {thread_id}")
    await sio.emit('register_session', {'thread_id': thread_id})
    # Wait for confirmation
    timeout = 5
    start = time.time()
    while not registered and time.time() - start < timeout:
        await asyncio.sleep(0.1)
    
    if not registered:
        print(f"[{datetime.now().isoformat()}] Failed to register session within {timeout} seconds")
        return False
    return True

async def send_ping():
    """Send a ping to keep the connection alive"""
    print(f"[{datetime.now().isoformat()}] Sending ping")
    response = await sio.call('ping', {'timestamp': datetime.now().isoformat()})
    print(f"[{datetime.now().isoformat()}] Received pong: {response}")

async def request_missed_messages(thread_id):
    """Request any missed messages for a thread"""
    global undelivered_queried
    print(f"[{datetime.now().isoformat()}] Requesting missed messages for thread: {thread_id}")
    await sio.emit('request_missed_messages', {'thread_id': thread_id})
    undelivered_queried = True

async def make_api_request(thread_id, message):
    """Make an API request to the /havefun endpoint"""
    print(f"[{datetime.now().isoformat()}] Making API request to /havefun")
    data = {
        'user_input': message,
        'user_id': 'test_user',
        'thread_id': thread_id,
        'use_websocket': True
    }
    
    # Make synchronous request to not block the event loop
    response = await asyncio.to_thread(
        requests.post,
        f"{SERVER_URL}/havefun",
        json=data,
        headers={'Content-Type': 'application/json'},
        stream=True  # Important for SSE streaming
    )
    
    if response.status_code != 200:
        print(f"[{datetime.now().isoformat()}] API request failed: {response.status_code} {response.text}")
        return False
    
    # Process streaming response
    print(f"[{datetime.now().isoformat()}] Processing response stream")
    for line in response.iter_lines():
        if line:
            try:
                text = line.decode('utf-8')
                # Skip event stream format markers
                if text.startswith('data: '):
                    text = text[6:]
                if text.strip():
                    print(f"[{datetime.now().isoformat()}] Received chunk: {text[:50]}...")
            except Exception as e:
                print(f"[{datetime.now().isoformat()}] Error processing response chunk: {e}")
    
    print(f"[{datetime.now().isoformat()}] API request complete")
    return True

async def debug_sessions(thread_id=None):
    """Query the debug endpoints to see session status"""
    try:
        # First check all sessions
        response = await asyncio.to_thread(
            requests.get,
            f"{SERVER_URL}/debug/all-sessions"
        )
        if response.status_code == 200:
            data = response.json()
            print(f"[{datetime.now().isoformat()}] All sessions: {json.dumps(data, indent=2)}")
        else:
            print(f"[{datetime.now().isoformat()}] Failed to get all sessions: {response.status_code}")
        
        # If thread_id specified, check that thread's sessions
        if thread_id:
            response = await asyncio.to_thread(
                requests.get,
                f"{SERVER_URL}/debug/thread-sessions?thread_id={thread_id}"
            )
            if response.status_code == 200:
                data = response.json()
                print(f"[{datetime.now().isoformat()}] Thread sessions: {json.dumps(data, indent=2)}")
            else:
                print(f"[{datetime.now().isoformat()}] Failed to get thread sessions: {response.status_code}")
    except Exception as e:
        print(f"[{datetime.now().isoformat()}] Error querying debug endpoints: {e}")

async def test_websocket_persistence():
    """Main test function"""
    try:
        print(f"[{datetime.now().isoformat()}] Starting WebSocket persistence test")
        print(f"[{datetime.now().isoformat()}] Server URL: {SERVER_URL}")
        print(f"[{datetime.now().isoformat()}] Thread ID: {THREAD_ID}")
        
        # Step 1: Connect to server
        print(f"[{datetime.now().isoformat()}] Connecting to server...")
        await sio.connect(SERVER_URL, transports=['websocket', 'polling'])
        
        if not connected:
            print(f"[{datetime.now().isoformat()}] Failed to connect to server")
            return False
        
        # Step 2: Register session
        if not await register_session(THREAD_ID):
            print(f"[{datetime.now().isoformat()}] Failed to register session")
            return False
        
        # Step 3: Check server-side session state
        print(f"[{datetime.now().isoformat()}] Checking initial server-side session state")
        await debug_sessions(THREAD_ID)
        
        # Step 4: Send a ping to keep the connection alive
        await send_ping()
        
        # Step 5: Make API request
        print(f"[{datetime.now().isoformat()}] Making API request")
        api_success = await make_api_request(THREAD_ID, TEST_MESSAGE)
        if not api_success:
            print(f"[{datetime.now().isoformat()}] API request failed")
        
        # Step 6: Send another ping to verify connection is still alive
        print(f"[{datetime.now().isoformat()}] Sending ping after API request")
        await send_ping()
        
        # Step 7: Check server-side session state again
        print(f"[{datetime.now().isoformat()}] Checking server-side session state after API request")
        await debug_sessions(THREAD_ID)
        
        # Step 8: Request any missed messages
        await request_missed_messages(THREAD_ID)
        
        # Step 9: Wait for analysis updates to arrive
        print(f"[{datetime.now().isoformat()}] Waiting for analysis updates...")
        start = time.time()
        while time.time() - start < TIMEOUT and len(messages_received) == 0:
            print(f"[{datetime.now().isoformat()}] Waiting... (elapsed: {time.time() - start:.1f}s)")
            await send_ping()  # Keep the connection alive
            await asyncio.sleep(5)
            
            # Try querying missed messages periodically
            if time.time() - start > 15 and not undelivered_queried:
                await request_missed_messages(THREAD_ID)
        
        # Step 10: Summarize results
        if len(messages_received) > 0:
            print(f"[{datetime.now().isoformat()}] Success! Received {len(messages_received)} analysis updates")
            return True
        else:
            print(f"[{datetime.now().isoformat()}] Test failed: No analysis updates received")
            # One last check of server state
            await debug_sessions(THREAD_ID)
            return False
    
    except Exception as e:
        print(f"[{datetime.now().isoformat()}] Error in test: {e}")
        return False
    finally:
        # Clean up
        if connected:
            print(f"[{datetime.now().isoformat()}] Disconnecting from server")
            await sio.disconnect()

if __name__ == "__main__":
    print(f"[{datetime.now().isoformat()}] WebSocket Session Persistence Test")
    
    # Run the test
    success = asyncio.run(test_websocket_persistence())
    
    # Exit with appropriate code
    sys.exit(0 if success else 1) 