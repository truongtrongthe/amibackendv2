#!/usr/bin/env python3
"""
Simple test server that simulates the /havefun endpoint with both sequential and concurrent processing modes.
Uses a fully isolated approach to event loop handling to prevent any conflicts.
"""

from flask import Flask, Response, request, jsonify, stream_with_context
import json
import time
import uuid
import threading
import argparse
import asyncio
import queue
from datetime import datetime

app = Flask(__name__)

# Simulate thread locks for sequential mode
thread_locks = {}
thread_locks_lock = threading.RLock()

def run_async_task(async_func, *args, **kwargs):
    """
    Run an async function in a completely isolated thread with its own event loop.
    Returns the result of the async function.
    
    Args:
        async_func: The async function to run
        *args, **kwargs: Arguments to pass to the async function
    
    Returns:
        The result of the async function
    """
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
            print(f"Error in async thread: {str(e)}")
            import traceback
            print(traceback.format_exc())
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

# Example async function to process a single chunk
async def process_chunk_async(chunk_type, content, request_id, delay=0.5):
    """Process a single chunk of the response asynchronously"""
    await asyncio.sleep(delay)
    
    if chunk_type == "analysis":
        return f"data: {json.dumps({'type': 'analysis', 'content': content, 'request_id': request_id})}\n\n"
    elif chunk_type == "message":
        return f"data: {json.dumps({'message': content, 'request_id': request_id})}\n\n"
    else:
        return f"data: {json.dumps({'type': chunk_type, 'content': content, 'request_id': request_id})}\n\n"

# This is a synchronous function that uses run_async_task for each async operation
def process_request_sync(user_input, request_id, concurrent_mode=True):
    """
    Process a request synchronously, using isolated async calls for each step.
    This approach eliminates event loop conflicts by ensuring each async operation
    happens in a completely isolated context.
    """
    results = []
    
    # First analysis event - processed in an isolated async context
    analysis_chunk = run_async_task(
        process_chunk_async, 
        "analysis", 
        f"Processing request: {user_input}", 
        request_id,
        delay=0.1
    )
    results.append(analysis_chunk)
    
    # Simulate processing time
    time.sleep(2)
    
    # Second analysis event - processed in another isolated async context
    analysis_chunk = run_async_task(
        process_chunk_async, 
        "analysis", 
        f"Analysis complete for: {user_input}", 
        request_id,
        delay=0.1
    )
    results.append(analysis_chunk)
    
    # Response lines - each processed in its own isolated context
    message_chunks = [
        f"Hello! I received your message: {user_input}",
        f"This is request {request_id}",
        f"Processing in {'concurrent' if concurrent_mode else 'sequential'} mode",
        "Thank you for your patience!"
    ]
    
    for message in message_chunks:
        chunk = run_async_task(
            process_chunk_async, 
            "message", 
            message, 
            request_id,
            delay=0.1
        )
        results.append(chunk)
        time.sleep(0.5)  # Simulate streaming delay
    
    return results

@app.route('/havefun', methods=['POST'])
def havefun():
    """
    Handle test requests, simulating either sequential or concurrent processing.
    Uses a fully synchronous approach with isolated async tasks to avoid any event loop conflicts.
    """
    start_time = datetime.now()
    print(f"=== BEGIN havefun request at {start_time.isoformat()} ===")
    
    data = request.get_json() or {}
    user_input = data.get("user_input", "")
    thread_id = data.get("thread_id", f"default_thread_{uuid.uuid4()}")
    concurrent_mode = data.get("concurrent_mode", True)  # Default to concurrent mode
    
    # Generate a request ID
    request_id = str(uuid.uuid4())
    
    # In sequential mode, we use a thread lock to simulate the old behavior
    thread_lock = None
    if not concurrent_mode:
        with thread_locks_lock:
            if thread_id not in thread_locks:
                thread_locks[thread_id] = threading.RLock()
            thread_lock = thread_locks[thread_id]
    
    # Create a synchronous response streaming solution
    def generate_response():
        """Generate streaming response items synchronously."""
        try:
            # If in sequential mode, acquire the lock
            if not concurrent_mode and thread_lock:
                thread_lock.acquire()
                print(f"Request {request_id}: Acquired lock for thread {thread_id}")
            
            try:
                # Process the request in a fully synchronous way, with each async operation isolated
                results = process_request_sync(user_input, request_id, concurrent_mode)
                
                # Yield each result
                for item in results:
                    yield item
                
            finally:
                # Always release the lock in sequential mode
                if not concurrent_mode and thread_lock:
                    thread_lock.release()
                    print(f"Request {request_id}: Released lock for thread {thread_id}")
            
            # Log completion
            end_time = datetime.now()
            elapsed = (end_time - start_time).total_seconds()
            print(f"=== END havefun request {request_id} - total time: {elapsed:.2f}s ===")
            
        except Exception as e:
            # Handle any errors
            error_msg = f"Error in response generator: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'error': error_msg})}\n\n"
    
    # Create a streaming response with our generator
    from flask import stream_with_context
    response = Response(stream_with_context(generate_response()), mimetype='text/event-stream')
    response.headers['X-Accel-Buffering'] = 'no'
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

@app.route('/')
def home():
    return """
    <html>
    <head><title>Test Server for Concurrency Demo</title></head>
    <body>
        <h1>Test Server for Concurrency Demo</h1>
        <p>Use the test_havefun_concurrency.py script to send requests to this server.</p>
        <p>Example: <code>python test_havefun_concurrency.py --url http://localhost:5005/havefun --requests 3</code></p>
    </body>
    </html>
    """

def main():
    parser = argparse.ArgumentParser(description="Run a test server that simulates sequential or concurrent processing")
    parser.add_argument("--port", type=int, default=5005, help="Port to run the server on")
    parser.add_argument("--host", default="0.0.0.0", help="Host to run the server on")
    
    args = parser.parse_args()
    
    print(f"Starting test server on {args.host}:{args.port}")
    print("Use test_havefun_concurrency.py to send requests to this server")
    print("To test sequential mode: add 'concurrent_mode': false to the request data")
    
    app.run(host=args.host, port=args.port, debug=True, threaded=True)

if __name__ == "__main__":
    main() 