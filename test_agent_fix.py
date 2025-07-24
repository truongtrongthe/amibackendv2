"""
Simple test to verify the agent endpoint fix
"""

import asyncio
import aiohttp
import json
from datetime import datetime

async def test_agent_basic():
    """Test basic agent functionality"""
    
    test_request = {
        "llm_provider": "openai",
        "user_request": "Hello! Can you help me analyze sales trends?",
        "agent_id": "test_agent_001",
        "agent_type": "sales_agent",
        "specialized_knowledge_domains": ["sales", "analytics"],
        "org_id": "5376f68e-ff74-43fe-a72c-c1be9c6ae652",
        "user_id": "user_GjswTZUz8Fhv2v2X0c8OyA"
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            print(f"🧪 Testing agent endpoint fix...")
            print(f"📤 Sending request to /api/tool/agent")
            
            async with session.post(
                "http://localhost:8000/api/tool/agent",
                json=test_request,
                headers={"Content-Type": "application/json"}
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    print(f"✅ SUCCESS! Agent endpoint is working")
                    print(f"🤖 Agent: {result.get('agent_id')}")
                    print(f"📝 Result preview: {result.get('result', '')[:100]}...")
                    return True
                else:
                    error_data = await response.json()
                    print(f"❌ FAILED (HTTP {response.status})")
                    print(f"🚨 Error: {error_data.get('error', 'Unknown error')}")
                    return False
                    
    except Exception as e:
        print(f"💥 EXCEPTION: {str(e)}")
        return False

async def test_agent_streaming():
    """Test agent streaming endpoint"""
    
    test_request = {
        "llm_provider": "openai",
        "user_request": "Xin chào! Em có thể giúp anh phân tích dữ liệu bán hàng không?",  # Vietnamese test
        "agent_id": "test_agent_002",
        "agent_type": "analyst_agent",
        "specialized_knowledge_domains": ["data_analysis", "vietnamese_language"],
        "org_id": "5376f68e-ff74-43fe-a72c-c1be9c6ae652",
        "user_id": "user_GjswTZUz8Fhv2v2X0c8OyA"
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            print(f"\n🧪 Testing agent streaming endpoint...")
            print(f"📤 Sending Vietnamese request to /api/tool/agent/stream")
            
            async with session.post(
                "http://localhost:8000/api/tool/agent/stream",
                json=test_request,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream"
                }
            ) as response:
                
                if response.status == 200:
                    print(f"✅ Connected to stream successfully")
                    print(f"📡 Receiving data...")
                    
                    chunk_count = 0
                    async for data in response.content.iter_any():
                        chunk_count += 1
                        text = data.decode('utf-8')
                        
                        for line in text.split('\n'):
                            if line.startswith('data: '):
                                try:
                                    event_data = json.loads(line[6:])
                                    event_type = event_data.get('type', 'unknown')
                                    
                                    if event_type == 'thinking':
                                        print(f"🧠 {event_data.get('content', '')}")
                                    elif event_type == 'response_chunk':
                                        print(f"📝 {event_data.get('content', '')}", end='', flush=True)
                                    elif event_type == 'complete':
                                        print(f"\n✅ Stream completed successfully!")
                                        print(f"📊 Total chunks: {chunk_count}")
                                        return True
                                    elif event_type == 'error':
                                        print(f"\n❌ Error in stream: {event_data.get('content', '')}")
                                        return False
                                        
                                except json.JSONDecodeError:
                                    pass
                                    
                        if chunk_count > 50:  # Prevent infinite loop
                            print(f"\n✅ Stream working (stopped after {chunk_count} chunks)")
                            return True
                            
                else:
                    error_text = await response.text()
                    print(f"❌ Stream failed (HTTP {response.status})")
                    print(f"🚨 Error: {error_text}")
                    return False
                    
    except Exception as e:
        print(f"💥 EXCEPTION: {str(e)}")
        return False

async def main():
    """Run tests"""
    print("🚀 Testing Agent Endpoint Fix")
    print(f"⏰ Started at: {datetime.now().isoformat()}")
    
    # Test basic endpoint
    basic_success = await test_agent_basic()
    
    # Test streaming endpoint  
    stream_success = await test_agent_streaming()
    
    if basic_success and stream_success:
        print(f"\n🎉 All tests passed! Agent endpoint is working correctly.")
    else:
        print(f"\n❌ Some tests failed. Check the error messages above.")
    
    print(f"🏁 Tests completed at: {datetime.now().isoformat()}")

if __name__ == "__main__":
    asyncio.run(main()) 