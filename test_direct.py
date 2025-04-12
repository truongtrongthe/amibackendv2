#!/usr/bin/env python3
"""
Direct HTTP request test for the havefun endpoint
"""

import requests
import json
import sys

# Test server URL
BASE_URL = "http://localhost:5001"

# Test message
TEST_MESSAGE = "Ligit?"

# Headers including Content-Type but no CORS (the server may handle it)
headers = {
    "Content-Type": "application/json"
}

# Payload
payload = {
    "user_input": TEST_MESSAGE,
    "user_id": "test_user",
    "thread_id": "test_thread",
    "graph_version_id": ""
}

print(f"Sending request to {BASE_URL}/havefun")
print(f"Payload: {json.dumps(payload, indent=2)}")
print(f"Headers: {headers}")

try:
    # Make the POST request
    response = requests.post(
        f"{BASE_URL}/havefun",
        json=payload,
        headers=headers,
        stream=True  # Important for SSE
    )
    
    print(f"\nPOST Response Status: {response.status_code}")
    print(f"POST Response Headers: {dict(response.headers)}")
    
    if response.status_code == 200:
        print("\nReceiving SSE stream:")
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data = line[6:].strip()
                    try:
                        parsed = json.loads(data)
                        if 'message' in parsed:
                            print(f"  > {parsed['message']}")
                    except json.JSONDecodeError:
                        print(f"Failed to parse: {data}")
    else:
        print(f"Error response: {response.text}")
        
except Exception as e:
    print(f"Error: {str(e)}")
    sys.exit(1) 