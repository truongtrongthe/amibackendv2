#!/usr/bin/env python3
"""
Test script to verify concurrent requests with the same thread_id are processed correctly
after fixing the event loop isolation issue in the Flask endpoints.
"""

import requests
import threading
import time
import argparse
import json
import uuid
from datetime import datetime

def make_request(url, thread_id, message, request_id):
    """
    Make a streaming request to the specified URL with the given parameters.
    
    Args:
        url: The endpoint URL to call
        thread_id: The thread_id to use for the request
        message: The message to send
        request_id: Unique identifier for this request for logging purposes
    
    Returns:
        Response content and timing information
    """
    print(f"[{request_id}] Starting request to {url} with thread_id {thread_id}")
    start_time = time.time()
    
    # Prepare the request data
    data = {
        "user_input": message,
        "thread_id": thread_id,
        "user_id": "test_user",
        "use_websocket": False
    }
    
    events = []
    complete = False
    error = None
    
    try:
        # Make the request with streaming response
        response = requests.post(
            url, 
            json=data,
            headers={"Content-Type": "application/json"},
            stream=True  # Enable streaming
        )
        
        # Print response status code and headers for debugging
        print(f"[{request_id}] Response status: {response.status_code}")
        print(f"[{request_id}] Response headers: {dict(response.headers)}")
        
        # Process the streaming response
        for line in response.iter_lines():
            if line:
                # Lines starting with "data: " contain the actual data
                if line.startswith(b"data: "):
                    try:
                        # Extract and parse the JSON data
                        json_str = line[6:].decode('utf-8')
                        json_data = json.loads(json_str)
                        events.append(json_data)
                        
                        # Print progress
                        if 'message' in json_data:
                            progress = json_data.get('message', '')[:30]
                            if len(progress) == 30:
                                progress += "..."
                            print(f"[{request_id}] Received: {progress}")
                        
                        # Check for 'complete' flag to identify when the response is done
                        if json_data.get('complete', False):
                            complete = True
                    except json.JSONDecodeError:
                        print(f"[{request_id}] Error parsing JSON from: {line}")
                    except Exception as e:
                        print(f"[{request_id}] Error processing response data: {str(e)}")
    except Exception as e:
        error = str(e)
        print(f"[{request_id}] Request failed: {error}")
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    result = {
        "request_id": request_id,
        "elapsed_time": elapsed,
        "events_count": len(events),
        "complete": complete,
        "error": error
    }
    
    print(f"[{request_id}] Request completed in {elapsed:.2f} seconds, received {len(events)} events")
    return result

def run_concurrent_test(url, thread_id, message, num_requests):
    """Run multiple concurrent requests with the same thread_id"""
    threads = []
    results = []
    
    start_time = time.time()
    print(f"Starting {num_requests} concurrent requests to thread_id {thread_id}")
    
    # Create and start threads for each request
    for i in range(num_requests):
        request_id = f"req-{i+1}"
        thread = threading.Thread(
            target=lambda r_id=request_id: results.append(
                make_request(url, thread_id, f"{message} (Request {r_id})", r_id)
            )
        )
        threads.append(thread)
        thread.start()
        # Small delay to ensure log messages don't get interleaved
        time.sleep(0.1)
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    end_time = time.time()
    total_elapsed = end_time - start_time
    
    # Calculate statistics
    successful = sum(1 for r in results if r["error"] is None)
    completed = sum(1 for r in results if r["complete"])
    
    # Sort by elapsed time
    results.sort(key=lambda x: x["elapsed_time"])
    
    fastest = results[0]["elapsed_time"] if results else 0
    slowest = results[-1]["elapsed_time"] if results else 0
    average = sum(r["elapsed_time"] for r in results) / len(results) if results else 0
    
    # Print results
    print("\n" + "="*70)
    print(f"CONCURRENT TEST RESULTS: {num_requests} requests to same thread_id")
    print("="*70)
    print(f"Total test time: {total_elapsed:.2f} seconds")
    print(f"Successful requests: {successful}/{num_requests}")
    print(f"Completed responses: {completed}/{num_requests}")
    print(f"Time statistics - Fastest: {fastest:.2f}s, Slowest: {slowest:.2f}s, Average: {average:.2f}s")
    
    for r in results:
        status = "ERROR" if r["error"] else ("COMPLETE" if r["complete"] else "INCOMPLETE")
        print(f"- {r['request_id']}: {r['elapsed_time']:.2f}s, {r['events_count']} events, {status}")
    
    # Determine overall success
    if successful == num_requests and completed == num_requests:
        print("\n✅ TEST PASSED: All requests completed successfully")
        return True
    else:
        print("\n❌ TEST FAILED: Some requests failed or did not complete")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test concurrent requests to the same thread_id")
    parser.add_argument("--url", default="http://localhost:5000/havefun", help="URL endpoint to test")
    parser.add_argument("--requests", type=int, default=3, help="Number of concurrent requests to make")
    parser.add_argument("--thread-id", default=None, help="Thread ID to use (default: auto-generated)")
    parser.add_argument("--message", default="Test concurrent message", help="Message to send")
    
    args = parser.parse_args()
    
    # Generate a unique thread ID if not specified
    thread_id = args.thread_id or f"test_thread_{uuid.uuid4()}"
    
    # Run the test
    success = run_concurrent_test(args.url, thread_id, args.message, args.requests)
    
    # Exit with appropriate code
    exit(0 if success else 1) 