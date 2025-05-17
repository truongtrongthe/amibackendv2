#!/usr/bin/env python3
"""
Test script for verifying the fixed concurrency handling in Flask endpoints.
This script sends multiple concurrent requests to the same thread_id and verifies
that all requests are successful.
"""

import requests
import threading
import time
import uuid
import argparse
import json
from datetime import datetime

def make_request(url, thread_id, message, req_id, delay=0, verbose=True):
    """Make a request to the specified endpoint with the given parameters."""
    if delay > 0:
        time.sleep(delay)
    
    # Prepare request data
    payload = {
        "user_input": f"Test concurrent message (Request req-{req_id})",
        "thread_id": thread_id,
        "user_id": "test_user",
        "use_websocket": False
    }
    
    start_time = datetime.now()
    if verbose:
        print(f"[{start_time.strftime('%H:%M:%S.%f')}] Request {req_id}: START")
    
    try:
        # Make the request with streaming response
        response = requests.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            stream=True
        )
        
        if verbose:
            print(f"Request {req_id}: Status code {response.status_code}")
        
        if not response.ok:
            print(f"Request {req_id}: Request failed with status {response.status_code}")
            if response.text:
                print(f"Request {req_id}: Error: {response.text}")
            return False
        
        # Process the streaming response
        event_count = 0
        for line in response.iter_lines():
            if not line:
                continue
            
            try:
                # Decode the line and remove the "data: " prefix
                line = line.decode('utf-8')
                if not line.startswith('data: '):
                    continue
                
                data = json.loads(line[6:])  # Skip "data: " prefix
                
                if 'error' in data:
                    print(f"Request {req_id}: Error received: {data['error']}")
                    return False
                
                event_count += 1
                
                if verbose and event_count <= 3:
                    # Print first few events for visibility
                    print(f"Request {req_id}: Event {event_count}: {line[:100]}...")
                
            except Exception as e:
                print(f"Request {req_id}: Error parsing event: {str(e)}, Line: {line[:50]}...")
        
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()
        
        if verbose:
            print(f"[{end_time.strftime('%H:%M:%S.%f')}] Request {req_id}: COMPLETE - Received {event_count} events in {elapsed:.2f} seconds")
        
        return True
    
    except Exception as e:
        print(f"Request {req_id}: Exception occurred: {str(e)}")
        return False

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test fixed concurrency in Flask endpoints")
    parser.add_argument("--url", default="http://localhost:5000/havefun", 
                      help="URL of the endpoint to test")
    parser.add_argument("--requests", type=int, default=3, 
                      help="Number of concurrent requests to make")
    parser.add_argument("--thread-id", default="test_fixed_concurrency_thread",
                      help="Thread ID to use (all requests will use the same ID)")
    parser.add_argument("--delay", type=float, default=0.1,
                      help="Stagger time between requests in seconds")
    parser.add_argument("--sequential", action="store_true",
                      help="Run requests sequentially instead of concurrently")
    parser.add_argument("--verbose", action="store_true",
                      help="Print detailed output for each request")
    
    args = parser.parse_args()
    
    print(f"Starting {args.requests} {'sequential' if args.sequential else 'concurrent'} requests to {args.url}")
    print(f"Using thread ID: {args.thread_id}")
    print(f"Stagger delay: {args.delay}s")
    print("-" * 60)
    
    start_time = datetime.now()
    
    # Start threads for requests
    threads = []
    success_count = 0
    
    for i in range(args.requests):
        delay = i * args.delay  # Stagger requests slightly
        
        if args.sequential:
            # Run sequentially
            success = make_request(args.url, args.thread_id, "Test message", i+1, delay, args.verbose)
            if success:
                success_count += 1
        else:
            # Run concurrently
            thread = threading.Thread(
                target=lambda idx=i+1: make_request(args.url, args.thread_id, "Test message", idx, delay, args.verbose) and globals().update({'success_count': success_count + 1})
            )
            threads.append(thread)
            thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds()
    
    print("-" * 60)
    print(f"Test completed in {total_time:.2f} seconds")
    print(f"Success: {success_count}/{args.requests} requests")
    print(f"Result: {'SUCCESS' if success_count == args.requests else 'FAILURE'}")

if __name__ == "__main__":
    main() 