#!/usr/bin/env python3
"""
Test script to verify that concurrent requests to the /havefun endpoint with the same thread_id
work correctly after implementing the event loop isolation fix.
"""

import requests
import json
import time
import threading
import uuid
from datetime import datetime
import argparse
import sys

def make_request(endpoint, thread_id, message, index, delay=0):
    """
    Make a request to the specified endpoint and handle the streaming response.
    
    Args:
        endpoint: The API endpoint to call
        thread_id: The thread_id to use for the conversation
        message: The message to send
        index: The request index number for logging
        delay: Optional delay before starting the request (seconds)
    """
    # Apply delay if specified
    if delay > 0:
        print(f"Request {index} waiting {delay}s before starting...")
        time.sleep(delay)
    
    start_time = datetime.now()
    print(f"Request {index} starting at {start_time.isoformat()}")
    
    # Prepare request data
    payload = {
        "user_input": message,
        "thread_id": thread_id,
        "user_id": f"test_user_{index}",
        "use_websocket": False
    }
    
    try:
        # Make the request with streaming response
        response = requests.post(
            endpoint,
            json=payload,
            headers={"Content-Type": "application/json"},
            stream=True
        )
        
        if not response.ok:
            print(f"Request {index} failed with status {response.status_code}")
            if response.text:
                print(f"Error: {response.text}")
            return
        
        # Process the streaming response
        print(f"Request {index} receiving response...")
        event_count = 0
        first_response_time = None
        error_received = False
        
        for line in response.iter_lines():
            if not line:
                continue
                
            # Decode the line
            decoded = line.decode('utf-8')
            if not decoded.startswith('data: '):
                continue
                
            # Parse the JSON data
            try:
                data = json.loads(decoded[6:])  # Remove 'data: ' prefix
                
                # Record the time of first response
                if event_count == 0:
                    first_response_time = datetime.now()
                    time_to_first = (first_response_time - start_time).total_seconds()
                    print(f"Request {index} got first response in {time_to_first:.2f}s")
                
                # Check for error
                if 'error' in data:
                    print(f"Request {index} received error: {data['error']}")
                    error_received = True
                    break
                
                # Print message if available
                if 'message' in data:
                    if event_count < 2:  # Only show the first few messages to avoid flooding
                        print(f"Request {index} received: {data['message'][:50]}...")
                
                event_count += 1
            except json.JSONDecodeError:
                print(f"Request {index} received invalid JSON: {decoded}")
        
        # Log completion
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()
        elapsed_from_first = (end_time - (first_response_time or start_time)).total_seconds()
        
        status = "ERROR" if error_received else "SUCCESS"
        print(f"Request {index} finished with {status} - received {event_count} events")
        print(f"Request {index} total time: {elapsed:.2f}s, processing time: {elapsed_from_first:.2f}s")
        
    except Exception as e:
        print(f"Request {index} failed with exception: {e}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test concurrent requests to the /havefun endpoint')
    parser.add_argument('--url', type=str, default='http://localhost:5000/havefun',
                      help='URL of the API endpoint')
    parser.add_argument('--requests', type=int, default=3,
                      help='Number of concurrent requests to make')
    parser.add_argument('--thread-id', type=str, default=f'test_thread_{int(time.time())}',
                      help='Thread ID to use for all requests')
    parser.add_argument('--message', type=str, default='Hello, this is a test message.',
                      help='Message to send in each request')
    parser.add_argument('--delay', type=int, default=0,
                      help='Delay between starting requests (seconds)')
    
    args = parser.parse_args()
    
    print(f"Starting {args.requests} concurrent requests to {args.url}")
    print(f"Using thread_id: {args.thread_id}")
    
    # Create and start threads for each request
    threads = []
    for i in range(args.requests):
        # Calculate delay for this request if needed
        delay = i * args.delay if args.delay > 0 else 0
        
        # Create a message specific to this request
        message = f"{args.message} (Request {i+1}/{args.requests})"
        
        # Start thread
        thread = threading.Thread(
            target=make_request,
            args=(args.url, args.thread_id, message, i+1, delay)
        )
        thread.daemon = True
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    print("All requests completed!")

if __name__ == "__main__":
    main() 