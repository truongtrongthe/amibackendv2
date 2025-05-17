#!/usr/bin/env python3
"""
Test script to validate the concurrent processing capabilities of the /havefun endpoint.
This script simulates multiple concurrent requests to the same thread_id.
"""

import requests
import json
import time
import threading
import uuid
import argparse
from datetime import datetime
import sys

def make_request(endpoint, thread_id, message, index, delay=0, concurrent_mode=True):
    """
    Make a request to the specified endpoint and handle the streaming response.
    
    Args:
        endpoint: The API endpoint to call
        thread_id: The thread_id to use for the conversation
        message: The message to send
        index: The request index number for logging
        delay: Optional delay before starting the request (seconds)
        concurrent_mode: Whether to use concurrent processing mode
    """
    # Apply delay if specified
    if delay > 0:
        print(f"Request {index}: Waiting {delay}s before starting...")
        time.sleep(delay)
    
    start_time = time.time()
    request_data = {
        "user_input": f"{message} (Request {index})",
        "user_id": "test_user",
        "thread_id": thread_id,
        "use_websocket": False,
        "concurrent_mode": concurrent_mode
    }
    
    print(f"Request {index}: Starting at {datetime.now().strftime('%H:%M:%S.%f')} - Thread: {thread_id}")
    print(f"Request {index}: Mode: {'Concurrent' if concurrent_mode else 'Sequential'}")
    
    # Make the request with streaming response handling
    try:
        response = requests.post(
            endpoint,
            json=request_data,
            headers={"Content-Type": "application/json"},
            stream=True  # Enable streaming response
        )
        
        if response.status_code != 200:
            print(f"Request {index}: Error - Status code {response.status_code}")
            print(response.text)
            return
        
        # Process the streaming response
        line_count = 0
        first_chunk_time = None
        last_chunk_time = None
        time_to_first = 0
        
        for line in response.iter_lines():
            if not line:
                continue
                
            # Decode the line
            decoded_line = line.decode('utf-8')
            if not decoded_line.startswith('data: '):
                continue
                
            # Parse the JSON data
            try:
                json_str = decoded_line[6:]  # Remove 'data: ' prefix
                data = json.loads(json_str)
                
                # Record time of first chunk
                if first_chunk_time is None:
                    first_chunk_time = time.time()
                    time_to_first = first_chunk_time - start_time
                    print(f"Request {index}: First chunk received after {time_to_first:.2f}s")
                
                # Update last chunk time
                last_chunk_time = time.time()
                
                # Count the line
                line_count += 1
                
                # Print info about the chunk
                if "message" in data:
                    print(f"Request {index}: Message chunk: {data['message'][:50]}...")
                elif "type" in data and data["type"] == "analysis":
                    print(f"Request {index}: Analysis event: {data.get('content', '')[:50]}...")
                
            except json.JSONDecodeError:
                print(f"Request {index}: Error parsing JSON: {decoded_line}")
        
        # Calculate times
        total_time = time.time() - start_time
        if first_chunk_time is not None and last_chunk_time is not None:
            streaming_time = last_chunk_time - first_chunk_time
        else:
            streaming_time = 0
        
        print(f"Request {index}: Completed after {total_time:.2f}s - Received {line_count} chunks")
        print(f"Request {index}: Time to first chunk: {time_to_first:.2f}s, Streaming time: {streaming_time:.2f}s")
        
    except requests.RequestException as e:
        print(f"Request {index}: Connection error: {str(e)}")
    except Exception as e:
        print(f"Request {index}: Unexpected error: {str(e)}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test concurrent requests to /havefun")
    parser.add_argument("--url", default="http://localhost:5005/havefun", 
                      help="URL of the /havefun endpoint")
    parser.add_argument("--requests", type=int, default=3, 
                      help="Number of concurrent requests to make")
    parser.add_argument("--thread-id", default=None,
                      help="Thread ID to use (will be generated if not provided)")
    parser.add_argument("--message", default="Tell me about yourself", 
                      help="Message to send in each request")
    parser.add_argument("--stagger", type=float, default=0.5,
                      help="Stagger time between requests in seconds")
    parser.add_argument("--sequential", action="store_true",
                      help="Use sequential mode instead of concurrent mode")
    
    args = parser.parse_args()
    
    # Generate thread ID if not provided
    thread_id = args.thread_id or f"test_{uuid.uuid4()}"
    
    print(f"Starting {args.requests} {'sequential' if args.sequential else 'concurrent'} requests to {args.url}")
    print(f"Using thread ID: {thread_id}")
    print(f"Message template: '{args.message}'")
    print(f"Stagger time: {args.stagger}s")
    print("-" * 50)
    
    # Start threads for concurrent requests
    threads = []
    for i in range(args.requests):
        delay = i * args.stagger  # Stagger requests slightly for better visibility
        thread = threading.Thread(
            target=make_request,
            args=(args.url, thread_id, args.message, i+1, delay, not args.sequential)
        )
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    print("-" * 50)
    print(f"All {args.requests} requests completed!")

if __name__ == "__main__":
    main() 