#!/usr/bin/env python3
"""
Test script to verify that the updated Ami co-builder/copilot behavior is working correctly
"""

import asyncio
import json
import aiohttp
from typing import Dict, Any


async def test_ami_copilot_behavior():
    """Test the updated Ami co-builder/copilot behavior"""
    
    # Test configuration
    base_url = "http://localhost:8000"
    endpoint = f"{base_url}/tool/llm"
    
    # Test cases to verify Ami's updated behavior
    test_cases = [
        {
            "name": "M&A Consultant Introduction",
            "query": "Anh làm tư vấn M&A doanh nghiệp",
            "expected_patterns": [
                "Ami",
                "co-builder", 
                "AI agent",
                "xây dựng",
                "tư vấn M&A",
                "2-3 use cases",
                "technical heavy lifting",
                "copilot"
            ]
        },
        {
            "name": "Healthcare Professional",
            "query": "I'm a doctor working in a hospital",
            "expected_patterns": [
                "Ami",
                "co-builder",
                "AI agent",
                "healthcare",
                "hospital",
                "use case",
                "technical",
                "copilot"
            ]
        },
        {
            "name": "E-commerce Business Owner",
            "query": "I run an online store selling electronics",
            "expected_patterns": [
                "Ami",
                "co-builder",
                "AI agent",
                "e-commerce",
                "online store",
                "use case",
                "technical",
                "build together"
            ]
        },
        {
            "name": "Generic Hello",
            "query": "Hello, who are you?",
            "expected_patterns": [
                "Ami",
                "co-builder",
                "AI agent",
                "help",
                "build",
                "copilot",
                "technical"
            ]
        }
    ]
    
    print("🧪 Testing Updated Ami Co-Builder/Copilot Behavior")
    print("=" * 70)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 Test {i}: {test_case['name']}")
        print(f"Query: {test_case['query']}")
        print("-" * 50)
        
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
                    
                    print("\n" + "-" * 50)
                    
                    # Check if Ami copilot behavior is present
                    patterns_found = []
                    for pattern in test_case['expected_patterns']:
                        if pattern.lower() in full_response.lower():
                            patterns_found.append(pattern)
                    
                    # Analysis
                    total_patterns = len(test_case['expected_patterns'])
                    found_patterns = len(patterns_found)
                    success_rate = (found_patterns / total_patterns) * 100
                    
                    print(f"📊 Analysis:")
                    print(f"   Found patterns: {found_patterns}/{total_patterns} ({success_rate:.1f}%)")
                    print(f"   Matched: {', '.join(patterns_found)}")
                    
                    if success_rate >= 60:  # At least 60% patterns found
                        print(f"✅ Ami copilot behavior detected!")
                    else:
                        print(f"❌ Ami copilot behavior needs improvement")
                        missing = [p for p in test_case['expected_patterns'] if p not in patterns_found]
                        print(f"   Missing: {', '.join(missing)}")
                    
                    # Check for specific copilot behaviors
                    copilot_behaviors = {
                        "Introduces as co-builder": any(word in full_response.lower() for word in ["co-builder", "cộng tác", "đồng hành"]),
                        "Suggests use cases": any(word in full_response.lower() for word in ["use case", "ứng dụng", "giải pháp"]),
                        "Mentions technical handling": any(word in full_response.lower() for word in ["technical", "kỹ thuật", "technical heavy lifting", "xử lý kỹ thuật"]),
                        "Asks to build together": any(word in full_response.lower() for word in ["build together", "xây dựng cùng", "cùng tạo", "together"])
                    }
                    
                    print(f"🎯 Copilot Behaviors:")
                    for behavior, present in copilot_behaviors.items():
                        status = "✅" if present else "❌"
                        print(f"   {status} {behavior}")
                    
        except Exception as e:
            print(f"❌ Error during test: {str(e)}")
    
    print("\n" + "=" * 70)
    print("🎯 Ami Co-Builder/Copilot Testing Complete!")


async def test_ami_follow_up_behavior():
    """Test Ami's follow-up behavior when user asks for more details"""
    
    print("\n🔄 Testing Ami Follow-up Behavior")
    print("=" * 70)
    
    base_url = "http://localhost:8000"
    endpoint = f"{base_url}/tool/llm"
    
    # Simulate a follow-up conversation
    follow_up_query = "Tell me more about the business valuation AI agent"
    
    payload = {
        "llm_provider": "openai",
        "user_query": follow_up_query,
        "enable_tools": False,
        "cursor_mode": False,
        "org_id": "test_org",
        "user_id": "test_user"
    }
    
    print(f"📋 Follow-up Query: {follow_up_query}")
    print("-" * 50)
    
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
                
                print("\n" + "-" * 50)
                
                # Check if Ami maintains copilot behavior in follow-ups
                follow_up_behaviors = {
                    "Maintains Ami identity": "ami" in full_response.lower(),
                    "Stays business-focused": any(word in full_response.lower() for word in ["business", "practical", "value", "solution"]),
                    "Avoids too much technical detail": not any(word in full_response.lower() for word in ["algorithm", "neural network", "machine learning model", "architecture"]),
                    "Offers to build together": any(word in full_response.lower() for word in ["build", "create", "develop", "implement", "together"])
                }
                
                print(f"🎯 Follow-up Behaviors:")
                for behavior, present in follow_up_behaviors.items():
                    status = "✅" if present else "❌"
                    print(f"   {status} {behavior}")
                
    except Exception as e:
        print(f"❌ Error during follow-up test: {str(e)}")


if __name__ == "__main__":
    print("🚀 Starting Ami Co-Builder/Copilot Tests...")
    
    # Run the tests
    asyncio.run(test_ami_copilot_behavior())
    asyncio.run(test_ami_follow_up_behavior())
    
    print("\n🎉 All copilot tests completed!")
    print("\n💡 Expected Ami Behavior:")
    print("   1. Introduces as co-builder that helps build AI agents")
    print("   2. Suggests 2-3 relevant AI agent use cases for user's domain")
    print("   3. Emphasizes handling technical complexity")
    print("   4. Asks which use case to build together")
    print("   5. Keeps responses business-focused and accessible") 