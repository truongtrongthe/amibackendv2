#!/usr/bin/env python3
"""
Test script to verify ami.py with MCWithTools works correctly.
"""

import asyncio
import json
from ami import convo_stream

async def test_ami_tools():
    """Test the convo_stream function with MCWithTools integration"""
    print("Testing ami.py with MCWithTools...")
    
    # Test messages to process
    test_messages = [
        "Hello, how are you today?",
        "What can you help me with?",
        "Tell me about artificial intelligence."
    ]
    
    # Process each message
    thread_id = f"test_thread_{asyncio.get_event_loop().time()}"
    
    for i, message in enumerate(test_messages):
        print(f"\n\n--- Testing message {i+1}: {message} ---\n")
        
        # Stream the response
        response_chunks = []
        analysis_received = False
        
        async for chunk in convo_stream(
            user_input=message,
            thread_id=thread_id,
            user_id="test_user"
        ):
            # Parse the chunk
            if chunk.startswith("data: "):
                data = json.loads(chunk[6:])
                if "type" in data and data["type"] == "analysis":
                    # Analysis chunk
                    if data.get("complete", False):
                        print("Analysis complete!")
                        analysis_received = True
                    else:
                        print(f"Analysis progress: {data.get('content', '')[:50]}...")
                elif "message" in data:
                    # Response chunk
                    response_chunks.append(data["message"])
                    print(f"Response chunk: {data['message']}")
        
        # Print the full response
        full_response = "\n".join(response_chunks)
        print(f"\n--- Full response ---\n{full_response}\n")
        
        # Verify we received analysis
        if analysis_received:
            print("✅ Analysis was received correctly")
        else:
            print("❌ No analysis was received")

if __name__ == "__main__":
    # Set OpenAI API key if needed
    import os
    if "OPENAI_API_KEY" not in os.environ:
        import sys
        if len(sys.argv) > 1:
            os.environ["OPENAI_API_KEY"] = sys.argv[1]
        else:
            print("Please provide an OpenAI API key as the first argument or set OPENAI_API_KEY")
            sys.exit(1)
    
    # Run the test
    asyncio.run(test_ami_tools()) 