"""
Test script for the new /api/tool/agent endpoint
"""

import asyncio
import aiohttp
import json
from datetime import datetime

async def test_agent_endpoint():
    """Test the new agent endpoint with various configurations"""
    
    # Test configurations
    test_cases = [
        {
            "name": "Basic Sales Agent Test",
            "request": {
                "llm_provider": "openai",
                "user_request": "Analyze the potential for expanding our product line to include eco-friendly options",
                "agent_id": "sales_analyst_001", 
                "agent_type": "sales_agent",
                "specialized_knowledge_domains": ["sales", "market_analysis", "product_development"],
                "org_id": "5376f68e-ff74-43fe-a72c-c1be9c6ae652",
                "user_id": "user_GjswTZUz8Fhv2v2X0c8OyA"
            }
        },
        {
            "name": "Customer Support Agent Test",
            "request": {
                "llm_provider": "anthropic",
                "user_request": "Help me understand why a customer's order was delayed and what we can do to resolve it",
                "agent_id": "support_agent_002",
                "agent_type": "support_agent", 
                "specialized_knowledge_domains": ["customer_service", "logistics", "problem_resolution"],
                "reasoning_depth": "deep",
                "org_id": "5376f68e-ff74-43fe-a72c-c1be9c6ae652",
                "user_id": "user_GjswTZUz8Fhv2v2X0c8OyA"
            }
        },
        {
            "name": "Data Analyst Agent Test (Tools Disabled)",
            "request": {
                "llm_provider": "openai",
                "user_request": "Create a summary of key performance indicators for Q4 based on general business knowledge",
                "agent_id": "analyst_agent_003",
                "agent_type": "analyst_agent",
                "specialized_knowledge_domains": ["data_analysis", "business_intelligence", "reporting"],
                "enable_tools": False,  # Test with tools disabled
                "org_id": "5376f68e-ff74-43fe-a72c-c1be9c6ae652",
                "user_id": "user_GjswTZUz8Fhv2v2X0c8OyA"
            }
        }
    ]
    
    base_url = "http://localhost:8000"
    
    for test_case in test_cases:
        print(f"\n{'='*60}")
        print(f"🧪 Testing: {test_case['name']}")
        print(f"{'='*60}")
        
        start_time = datetime.now()
        
        try:
            async with aiohttp.ClientSession() as session:
                print(f"📤 Sending request to /api/tool/agent...")
                print(f"🤖 Agent: {test_case['request']['agent_id']} ({test_case['request']['agent_type']})")
                print(f"🎯 Task: {test_case['request']['user_request'][:80]}...")
                
                async with session.post(
                    f"{base_url}/api/tool/agent",
                    json=test_case['request'],
                    headers={"Content-Type": "application/json"}
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        elapsed = (datetime.now() - start_time).total_seconds()
                        
                        print(f"✅ SUCCESS (HTTP {response.status}) - {elapsed:.2f}s")
                        print(f"🤖 Agent: {result.get('agent_id')} ({result.get('agent_type')})")
                        print(f"⏱️ Execution Time: {result.get('execution_time', 0):.2f}s")
                        print(f"📋 Tasks Completed: {result.get('tasks_completed', 0)}")
                        print(f"📝 Result Preview: {result.get('result', '')[:200]}...")
                        
                        if result.get('metadata'):
                            print(f"🔧 Tools Used: {result['metadata'].get('tools_used', [])}")
                        
                    else:
                        error_data = await response.json()
                        elapsed = (datetime.now() - start_time).total_seconds()
                        
                        print(f"❌ FAILED (HTTP {response.status}) - {elapsed:.2f}s")
                        print(f"🚨 Error: {error_data.get('error', 'Unknown error')}")
                        
        except Exception as e:
            elapsed = (datetime.now() - start_time).total_seconds()
            print(f"💥 EXCEPTION - {elapsed:.2f}s")
            print(f"🚨 Error: {str(e)}")


async def test_agent_streaming_endpoint():
    """Test the new agent streaming endpoint"""
    
    print(f"\n{'='*60}")
    print(f"🧪 Testing: Agent Streaming Endpoint")
    print(f"{'='*60}")
    
    start_time = datetime.now()
    
    test_request = {
        "llm_provider": "openai",
        "user_request": "Analyze current market trends for renewable energy products and provide strategic recommendations",
        "agent_id": "strategy_agent_004",
        "agent_type": "strategy_agent",
        "specialized_knowledge_domains": ["market_research", "strategic_planning", "renewable_energy"],
        "reasoning_depth": "standard",
        "org_id": "5376f68e-ff74-43fe-a72c-c1be9c6ae652",
        "user_id": "user_GjswTZUz8Fhv2v2X0c8OyA"
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            print(f"📤 Sending streaming request to /api/tool/agent/stream...")
            print(f"🤖 Agent: {test_request['agent_id']} ({test_request['agent_type']})")
            print(f"🎯 Task: {test_request['user_request'][:80]}...")
            
            async with session.post(
                "http://localhost:8000/api/tool/agent/stream",
                json=test_request,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream"
                }
            ) as response:
                
                if response.status == 200:
                    print(f"✅ Connected to stream (HTTP {response.status})")
                    print(f"📡 Receiving streaming data...")
                    print(f"{'─'*60}")
                    
                    chunk_count = 0
                    async for data in response.content.iter_any():
                        chunk_count += 1
                        text = data.decode('utf-8')
                        
                        # Parse SSE data
                        for line in text.split('\n'):
                            if line.startswith('data: '):
                                try:
                                    event_data = json.loads(line[6:])  # Remove 'data: ' prefix
                                    
                                    event_type = event_data.get('type', 'unknown')
                                    content = event_data.get('content', '')
                                    
                                    if event_type == 'thinking':
                                        thought_type = event_data.get('thought_type', 'general')
                                        agent_id = event_data.get('agent_id', 'unknown')
                                        print(f"🧠 [{thought_type}] {content}")
                                        
                                    elif event_type == 'response_chunk':
                                        print(f"📝 {content}", end='', flush=True)
                                        
                                    elif event_type == 'response_complete':
                                        print(f"\n✨ Response completed")
                                        
                                    elif event_type == 'complete':
                                        elapsed = (datetime.now() - start_time).total_seconds()
                                        execution_time = event_data.get('execution_time', 0)
                                        print(f"\n🎉 Agent execution completed!")
                                        print(f"⏱️ Total Time: {elapsed:.2f}s | Execution Time: {execution_time:.2f}s")
                                        print(f"📊 Chunks Received: {chunk_count}")
                                        return
                                        
                                    elif event_type == 'error':
                                        print(f"\n❌ Error: {content}")
                                        return
                                        
                                except json.JSONDecodeError:
                                    pass  # Skip malformed JSON
                                    
                else:
                    print(f"❌ FAILED (HTTP {response.status})")
                    error_text = await response.text()
                    print(f"🚨 Error: {error_text}")
                    
    except Exception as e:
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"💥 EXCEPTION - {elapsed:.2f}s")
        print(f"🚨 Error: {str(e)}")


async def main():
    """Run all tests"""
    print("🚀 Starting Agent Endpoint Tests")
    print(f"⏰ Started at: {datetime.now().isoformat()}")
    
    # Test the synchronous endpoint
    await test_agent_endpoint()
    
    # Test the streaming endpoint
    await test_agent_streaming_endpoint()
    
    print(f"\n🏁 All tests completed at: {datetime.now().isoformat()}")


if __name__ == "__main__":
    asyncio.run(main()) 