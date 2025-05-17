#!/usr/bin/env python3
"""
Simple test script for the /havefun endpoint.
"""

import requests
import json
from datetime import datetime

def test_havefun():
    """Test a single request to the /havefun endpoint."""
    url = "http://localhost:5005/havefun"
    
    # Prepare request data
    payload = {
        "user_input": "This is a simple test message",
        "thread_id": "simple_test_thread",
        "user_id": "test_user",
        "use_websocket": False
    }
    
    print(f"Making request to {url}...")
    start_time = datetime.now()
    
    try:
        # Make the request with streaming response
        response = requests.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            stream=True
        )
        
        print(f"Response status: {response.status_code}")
        
        if not response.ok:
            print(f"Request failed with status {response.status_code}")
            if response.text:
                print(f"Error: {response.text}")
            return
        
        # Process the streaming response
        print("Receiving response...")
        event_count = 0
        
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
                
                # Check for error
                if 'error' in data:
                    print(f"Received error: {data['error']}")
                    continue
                
                # Print message if available
                if 'message' in data:
                    print(f"Received: {data['message'][:50]}...")
                
                event_count += 1
            except json.JSONDecodeError:
                print(f"Received invalid JSON: {decoded}")
        
        # Log completion
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()
        
        print(f"Request completed - received {event_count} events")
        print(f"Total time: {elapsed:.2f}s")
        
    except Exception as e:
        print(f"Request failed with exception: {e}")

if __name__ == "__main__":
    test_havefun() 