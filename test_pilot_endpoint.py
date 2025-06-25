#!/usr/bin/env python3
"""
Simple test script for the /conversation/pilot endpoint
"""

import json
import asyncio
import aiohttp

async def test_pilot_endpoint():
    """Test the pilot endpoint with a simple message"""
    
    url = "http://localhost:5001/conversation/pilot"
    
    test_data = {
        "user_input": "Hello! Can you help me understand what you can do?",
        "user_id": "test_user",
        "thread_id": "test_pilot_session",
        "use_websocket": False
    }
    
    print("Testing /conversation/pilot endpoint...")
    print(f"Sending: {test_data['user_input']}")
    print("-" * 50)
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=test_data,
                headers={"Content-Type": "application/json"}
            ) as response:
                
                print(f"Status: {response.status}")
                
                if response.status != 200:
                    text = await response.text()
                    print(f"Error: {text}")
                    return
                
                # Read SSE stream
                accumulated_response = ""
                async for line in response.content:
                    line_str = line.decode('utf-8').strip()
                    
                    if line_str.startswith('data: '):
                        json_str = line_str[6:]  # Remove 'data: ' prefix
                        try:
                            data = json.loads(json_str)
                            
                            if data.get('status') == 'streaming':
                                accumulated_response += data.get('content', '')
                                print(f"Streaming: {data.get('content', '')}", end='', flush=True)
                            
                            elif data.get('type') == 'response_complete':
                                print(f"\n\nComplete Response: {data.get('message', '')}")
                                print(f"\nMetadata:")
                                metadata = data.get('metadata', {})
                                print(f"  - Mode: {metadata.get('mode', 'unknown')}")
                                print(f"  - Pilot Mode: {metadata.get('pilot_mode', False)}")
                                print(f"  - Knowledge Saving: {metadata.get('knowledge_saving', True)}")
                                print(f"  - Teaching Intent Disabled: {metadata.get('teaching_intent_disabled', False)}")
                                print(f"  - Has Teaching Intent: {metadata.get('has_teaching_intent', False)}")
                                print(f"  - Should Save Knowledge: {metadata.get('should_save_knowledge', False)}")
                                
                            elif data.get('error'):
                                print(f"\nError: {data.get('error')}")
                                
                        except json.JSONDecodeError as e:
                            print(f"\nJSON decode error: {e}")
                            print(f"Raw data: {json_str}")
                
                print("\n" + "-" * 50)
                print("✅ Test completed successfully!")
                
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_pilot_endpoint())
 