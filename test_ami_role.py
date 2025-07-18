#!/usr/bin/env python3
"""
Test script to verify that the Ami role is working correctly in the /tool/llm endpoint
"""

import asyncio
import json
import aiohttp
from typing import Dict, Any


async def test_ami_role():
    """Test the Ami role in the /tool/llm endpoint"""
    
    # Test configuration
    base_url = "http://localhost:8000"
    endpoint = f"{base_url}/tool/llm"
    
    # Test cases to verify Ami's role
    test_cases = [
        {
            "name": "Basic Ami Introduction",
            "query": "Hello, who are you?",
            "expected_keywords": ["Ami", "co-builder", "AI agent", "help", "build"]
        },
        {
            "name": "AI Development Question",
            "query": "How can I build an AI agent for customer service?",
            "expected_keywords": ["Ami", "design", "build", "optimize", "AI agent", "architecture"]
        },
        {
            "name": "Coding Assistance",
            "query": "Help me debug this Python code for my AI project",
            "expected_keywords": ["Ami", "debug", "coding", "assist", "AI development"]
        },
        {
            "name": "Technology Guidance",
            "query": "What are the best frameworks for building AI agents?",
            "expected_keywords": ["Ami", "frameworks", "AI", "technologies", "guidance"]
        }
    ]
    
    print("🧪 Testing Ami Role in /tool/llm Endpoint")
    print("=" * 60)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 Test {i}: {test_case['name']}")
        print(f"Query: {test_case['query']}")
        print("-" * 40)
        
        # Prepare request payload
        payload = {
            "llm_provider": "openai",
            "user_query": test_case['query'],
            "enable_tools": False,  # Disable tools for simpler testing
            "cursor_mode": False,   # Disable cursor mode for cleaner output
            "org_id": "test_org",
            "user_id": "test_user"
        }
        
        try:
            # Make the request
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    endpoint,
                    json=payload,
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "text/event-stream"
                    }
                ) as response:
                    
                    if response.status != 200:
                        print(f"❌ Error: HTTP {response.status}")
                        continue
                    
                    # Read the streaming response
                    full_response = ""
                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        if line.startswith('data: '):
                            try:
                                data = json.loads(line[6:])  # Remove 'data: ' prefix
                                if data.get('type') == 'response_chunk':
                                    content = data.get('content', '')
                                    full_response += content
                                    print(content, end='', flush=True)
                            except json.JSONDecodeError:
                                continue
                    
                    print("\n" + "-" * 40)
                    
                    # Check if Ami role is present
                    ami_keywords_found = []
                    for keyword in test_case['expected_keywords']:
                        if keyword.lower() in full_response.lower():
                            ami_keywords_found.append(keyword)
                    
                    if ami_keywords_found:
                        print(f"✅ Ami role detected! Found keywords: {', '.join(ami_keywords_found)}")
                    else:
                        print(f"❌ Ami role not detected. Expected keywords: {', '.join(test_case['expected_keywords'])}")
                        print(f"Response preview: {full_response[:200]}...")
                    
        except Exception as e:
            print(f"❌ Error during test: {str(e)}")
    
    print("\n" + "=" * 60)
    print("🎯 Ami Role Testing Complete!")


async def test_ami_with_tools():
    """Test Ami role with tools enabled"""
    
    print("\n🔧 Testing Ami Role with Tools Enabled")
    print("=" * 60)
    
    base_url = "http://localhost:8000"
    endpoint = f"{base_url}/tool/llm"
    
    # Test with tools enabled
    payload = {
        "llm_provider": "openai",
        "user_query": "What are the latest developments in AI agent frameworks?",
        "enable_tools": True,
        "cursor_mode": True,  # Enable cursor mode to see Ami's thinking
        "org_id": "test_org",
        "user_id": "test_user"
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                endpoint,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream"
                }
            ) as response:
                
                if response.status != 200:
                    print(f"❌ Error: HTTP {response.status}")
                    return
                
                print("🔄 Streaming response with Ami's thoughts and tools...")
                print("-" * 40)
                
                # Read the streaming response
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if line.startswith('data: '):
                        try:
                            data = json.loads(line[6:])  # Remove 'data: ' prefix
                            
                            # Display different types of events
                            event_type = data.get('type', 'unknown')
                            
                            if event_type == 'thinking':
                                print(f"💭 Ami's Thought: {data.get('content', '')}")
                            elif event_type == 'analysis_start':
                                print(f"🎯 {data.get('content', '')}")
                            elif event_type == 'analysis_complete':
                                print(f"📊 {data.get('content', '')}")
                            elif event_type == 'response_chunk':
                                content = data.get('content', '')
                                print(content, end='', flush=True)
                            elif event_type == 'complete':
                                print(f"\n✅ {data.get('content', '')}")
                                
                        except json.JSONDecodeError:
                            continue
                
                print("\n" + "-" * 40)
                print("✅ Ami with tools test completed!")
                
    except Exception as e:
        print(f"❌ Error during tools test: {str(e)}")


if __name__ == "__main__":
    print("🚀 Starting Ami Role Tests...")
    
    # Run the tests
    asyncio.run(test_ami_role())
    asyncio.run(test_ami_with_tools())
    
    print("\n🎉 All tests completed!") 