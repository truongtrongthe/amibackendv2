#!/usr/bin/env python3

"""
Test script for the new streaming LLM endpoint with tool control options
Tests different configurations for controlling when tools are used.
"""

import asyncio
import json
import time
import aiohttp
from typing import AsyncGenerator, Dict, Any


class StreamingLLMTester:
    """Test class for the streaming LLM endpoint"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.endpoint = f"{base_url}/tool/llm"
        
    async def test_streaming_response(self, test_name: str, request_data: Dict[str, Any]) -> None:
        """Test a single streaming request"""
        print(f"\n{'='*60}")
        print(f"🧪 TEST: {test_name}")
        print(f"{'='*60}")
        
        # Print request details
        print(f"📋 Request Details:")
        print(f"   Provider: {request_data.get('llm_provider', 'N/A')}")
        print(f"   Query: {request_data.get('user_query', 'N/A')}")
        print(f"   Enable Tools: {request_data.get('enable_tools', 'N/A')}")
        print(f"   Force Tools: {request_data.get('force_tools', 'N/A')}")
        print(f"   Tools Whitelist: {request_data.get('tools_whitelist', 'N/A')}")
        print(f"   System Prompt: {request_data.get('system_prompt', 'Default')[:100]}...")
        
        start_time = time.time()
        first_chunk_time = None
        total_chunks = 0
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.endpoint, json=request_data) as response:
                    if response.status != 200:
                        print(f"❌ ERROR: HTTP {response.status}")
                        text = await response.text()
                        print(f"Response: {text}")
                        return
                    
                    print(f"\n📡 Streaming Response:")
                    print("-" * 50)
                    
                    async for line in response.content:
                        if line:
                            line_str = line.decode('utf-8').strip()
                            if line_str.startswith('data: '):
                                data_str = line_str[6:]  # Remove 'data: ' prefix
                                if data_str == '[DONE]':
                                    break
                                try:
                                    data = json.loads(data_str)
                                    
                                    # Record first chunk time
                                    if first_chunk_time is None and data.get('type') == 'response_chunk':
                                        first_chunk_time = time.time() - start_time
                                    
                                    # Handle different event types
                                    if data.get('type') == 'status':
                                        print(f"🔄 Status: {data.get('content', 'Unknown')}")
                                    elif data.get('type') == 'response_chunk':
                                        content = data.get('content', '')
                                        print(content, end='', flush=True)
                                        total_chunks += 1
                                    elif data.get('type') == 'response_complete':
                                        print(f"\n✅ Response Complete")
                                    elif data.get('type') == 'complete':
                                        print(f"\n🎉 Execution Complete!")
                                        execution_time = data.get('execution_time', 0)
                                        print(f"⏱️  Total execution time: {execution_time:.2f}s")
                                        print(f"🚀 Time to first chunk: {first_chunk_time:.2f}s" if first_chunk_time else "")
                                        print(f"📊 Total chunks received: {total_chunks}")
                                        print(f"🔧 Provider: {data.get('provider', 'Unknown')}")
                                        print(f"🤖 Model: {data.get('model_used', 'Unknown')}")
                                        
                                        # Print metadata if available
                                        if 'metadata' in data:
                                            metadata = data['metadata']
                                            print(f"📋 Metadata:")
                                            print(f"   - Org ID: {metadata.get('org_id', 'N/A')}")
                                            print(f"   - User ID: {metadata.get('user_id', 'N/A')}")
                                            print(f"   - Tools Used: {metadata.get('tools_used', [])}")
                                            
                                    elif data.get('type') == 'error':
                                        print(f"\n❌ Error: {data.get('content', 'Unknown error')}")
                                        return
                                        
                                except json.JSONDecodeError:
                                    print(f"⚠️  Could not parse JSON: {data_str}")
                                    continue
                                    
        except Exception as e:
            print(f"❌ Request failed: {e}")
            
        print(f"\n📊 Summary:")
        print(f"   Total test time: {time.time() - start_time:.2f}s")
        print(f"   First chunk time: {first_chunk_time:.2f}s" if first_chunk_time else "   No chunks received")
        print(f"   Total chunks: {total_chunks}")

    async def run_all_tests(self):
        """Run all test scenarios"""
        print("🚀 Starting comprehensive streaming LLM tests...")
        print(f"🔗 Testing endpoint: {self.endpoint}")
        
        # Test 1: Default behavior (with tools enabled)
        await self.test_streaming_response(
            "Default with Tools Enabled",
            {
                "llm_provider": "openai",
                "user_query": "What is the capital of France?",
                "system_prompt": "You are a helpful assistant.",
                "model_params": {
                    "temperature": 0.7,
                    "max_tokens": 1000
                },
                "enable_tools": True,  # This is the default
                "force_tools": False,
                "tools_whitelist": None
            }
        )
        
        # Test 2: Tools disabled - should NOT use search
        await self.test_streaming_response(
            "Tools Disabled - No Search",
            {
                "llm_provider": "openai",
                "user_query": "What is the capital of France?",
                "system_prompt": "You are a helpful assistant.",
                "model_params": {
                    "temperature": 0.7,
                    "max_tokens": 1000
                },
                "enable_tools": False,  # Disable tools
                "force_tools": False,
                "tools_whitelist": None
            }
        )
        
        # Test 3: Force tools - should ALWAYS use search
        await self.test_streaming_response(
            "Force Tools - Always Search",
            {
                "llm_provider": "openai",
                "user_query": "What is 2 + 2?",  # Simple question that normally wouldn't need search
                "system_prompt": "You are a helpful assistant.",
                "model_params": {
                    "temperature": 0.7,
                    "max_tokens": 1000
                },
                "enable_tools": True,
                "force_tools": True,  # Force tool usage
                "tools_whitelist": None
            }
        )
        
        # Test 4: Anthropic with tools disabled
        await self.test_streaming_response(
            "Anthropic - Tools Disabled",
            {
                "llm_provider": "anthropic",
                "user_query": "What is the capital of Japan?",
                "system_prompt": "You are a helpful assistant.",
                "model_params": {
                    "temperature": 0.7,
                    "max_tokens": 1000
                },
                "enable_tools": False,  # Disable tools
                "force_tools": False,
                "tools_whitelist": None
            }
        )
        
        # Test 5: Custom system prompt that discourages searching
        await self.test_streaming_response(
            "Custom Prompt - Discourage Search",
            {
                "llm_provider": "openai",
                "user_query": "What is the largest planet in our solar system?",
                "system_prompt": "You are a knowledgeable assistant. Answer questions directly from your training data without searching for additional information.",
                "model_params": {
                    "temperature": 0.7,
                    "max_tokens": 1000
                },
                "enable_tools": True,  # Tools enabled but prompt discourages use
                "force_tools": False,
                "tools_whitelist": None
            }
        )
        
        # Test 6: Whitelist specific tools (only search allowed)
        await self.test_streaming_response(
            "Tools Whitelist - Search Only",
            {
                "llm_provider": "openai",
                "user_query": "What's the current weather in Tokyo?",
                "system_prompt": "You are a helpful assistant.",
                "model_params": {
                    "temperature": 0.7,
                    "max_tokens": 1000
                },
                "enable_tools": True,
                "force_tools": False,
                "tools_whitelist": ["search"]  # Only allow search tool
            }
        )
        
        print(f"\n🎉 All tests completed!")
        print(f"📊 Summary: 6 different tool control scenarios tested")


async def main():
    """Main test function"""
    tester = StreamingLLMTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    print("🧪 Streaming LLM Tool Control Test Suite")
    print("=" * 50)
    asyncio.run(main()) 