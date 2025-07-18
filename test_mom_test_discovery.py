#!/usr/bin/env python3
"""
Test script to verify Mom Test discovery system is working correctly
"""

import asyncio
import json
import aiohttp
from typing import Dict, Any


async def test_mom_test_discovery():
    """Test Mom Test-inspired discovery with different scenarios"""
    
    base_url = "http://localhost:8000"
    endpoint = f"{base_url}/tool/llm"
    
    # Test scenarios for Mom Test discovery
    test_scenarios = [
        {
            "name": "M&A Consultant (Vietnamese)",
            "query": "Anh làm tư vấn M&A doanh nghiệp",
            "expected_mom_test_behaviors": [
                "past behavior questions",
                "specific examples",
                "time-based questions",
                "concrete work examples",
                "no AI pitching"
            ]
        },
        {
            "name": "Generic Hello",
            "query": "Hello, who are you?",
            "expected_mom_test_behaviors": [
                "genuine curiosity",
                "work-focused questions",
                "past behavior inquiry",
                "conversational tone",
                "no immediate AI suggestions"
            ]
        },
        {
            "name": "Healthcare Worker",
            "query": "I work at a hospital",
            "expected_mom_test_behaviors": [
                "specific work questions",
                "time-based inquiry",
                "concrete examples",
                "healthcare context awareness",
                "genuine curiosity"
            ]
        },
        {
            "name": "Vague Interest",
            "query": "I'm interested in automation",
            "expected_mom_test_behaviors": [
                "past behavior questions",
                "specific examples",
                "work context inquiry",
                "no immediate solutions",
                "conversational discovery"
            ]
        }
    ]
    
    print("🧪 Testing Mom Test Discovery System")
    print("=" * 60)
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n📋 Test {i}: {scenario['name']}")
        print(f"Query: {scenario['query']}")
        print("-" * 40)
        
        # Test with human context enabled
        payload = {
            "llm_provider": "openai",
            "user_query": scenario['query'],
            "enable_tools": True,  # Enable tools to use human context
            "cursor_mode": False,  # Keep simple for testing
            "org_id": "test_org",  # This should trigger TechCorp context
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
                        continue
                    
                    # Collect the full response
                    full_response = ""
                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        if line.startswith('data: '):
                            try:
                                data = json.loads(line[6:])
                                if data.get('type') == 'response_chunk':
                                    content = data.get('content', '')
                                    full_response += content
                                    print(content, end='', flush=True)
                            except json.JSONDecodeError:
                                continue
                    
                    print("\n" + "-" * 40)
                    
                    # Analyze Mom Test behaviors
                    mom_test_analysis = analyze_mom_test_response(full_response, scenario)
                    
                    print(f"📊 Mom Test Analysis:")
                    for behavior, present in mom_test_analysis.items():
                        status = "✅" if present else "❌"
                        print(f"   {status} {behavior}")
                    
                    # Overall score
                    score = sum(mom_test_analysis.values()) / len(mom_test_analysis) * 100
                    print(f"🎯 Mom Test Score: {score:.1f}%")
                    
                    if score >= 70:
                        print("✅ Good Mom Test approach!")
                    else:
                        print("❌ Needs improvement in Mom Test approach")
                    
        except Exception as e:
            print(f"❌ Error during test: {str(e)}")
    
    print("\n" + "=" * 60)
    print("🎯 Mom Test Discovery Testing Complete!")


def analyze_mom_test_response(response: str, scenario: Dict) -> Dict[str, bool]:
    """Analyze if response follows Mom Test principles"""
    
    response_lower = response.lower()
    
    # Core Mom Test behaviors to check
    analysis = {
        "Introduces as Ami": "ami" in response_lower,
        "Asks about past behavior": any(phrase in response_lower for phrase in [
            "what did you", "yesterday", "last week", "last time", "spent time", 
            "what took", "how long", "when did you last"
        ]),
        "Seeks specific examples": any(phrase in response_lower for phrase in [
            "can you walk me through", "what happened", "specific", "example", 
            "tell me about", "how did you"
        ]),
        "Avoids AI pitching": not any(phrase in response_lower for phrase in [
            "ai agent", "automate", "build together", "i can help you build",
            "ai solution", "let me suggest"
        ]),
        "Shows genuine curiosity": any(phrase in response_lower for phrase in [
            "curious", "interested", "tell me", "what", "how", "why"
        ]),
        "Conversational tone": any(phrase in response_lower for phrase in [
            "i'm curious", "tell me", "what's", "how's", "sounds like"
        ]),
        "Focuses on work/problems": any(phrase in response_lower for phrase in [
            "work", "job", "task", "challenge", "problem", "process", "daily"
        ])
    }
    
    # Scenario-specific checks
    if "M&A" in scenario['name']:
        analysis["Context awareness"] = any(phrase in response_lower for phrase in [
            "m&a", "finance", "deal", "client", "analysis", "consulting"
        ])
    elif "Healthcare" in scenario['name']:
        analysis["Context awareness"] = any(phrase in response_lower for phrase in [
            "hospital", "patient", "medical", "healthcare", "shift", "documentation"
        ])
    else:
        analysis["Context awareness"] = True  # Default to true for generic cases
    
    return analysis


async def test_context_integration():
    """Test that human context is properly integrated"""
    
    print("\n🔍 Testing Context Integration")
    print("=" * 60)
    
    base_url = "http://localhost:8000"
    endpoint = f"{base_url}/tool/llm"
    
    # Test with known org_id that should have context
    payload = {
        "llm_provider": "openai",
        "user_query": "Hi there",
        "enable_tools": True,
        "org_id": "test_org",  # Should map to TechCorp finance context
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
                
                print("🔄 Testing context integration...")
                print("-" * 40)
                
                full_response = ""
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if line.startswith('data: '):
                        try:
                            data = json.loads(line[6:])
                            if data.get('type') == 'response_chunk':
                                content = data.get('content', '')
                                full_response += content
                                print(content, end='', flush=True)
                        except json.JSONDecodeError:
                            continue
                
                print("\n" + "-" * 40)
                
                # Check for context awareness
                context_indicators = {
                    "Mentions TechCorp": "techcorp" in full_response.lower(),
                    "Shows finance awareness": any(word in full_response.lower() for word in ["finance", "financial", "m&a"]),
                    "Uses contextual questions": any(phrase in full_response.lower() for phrase in [
                        "deal", "client", "analysis", "report", "valuation"
                    ])
                }
                
                print("🎯 Context Integration Check:")
                for indicator, present in context_indicators.items():
                    status = "✅" if present else "❌"
                    print(f"   {status} {indicator}")
                
    except Exception as e:
        print(f"❌ Error during context test: {str(e)}")


if __name__ == "__main__":
    print("🚀 Starting Mom Test Discovery Tests...")
    
    # Run the tests
    asyncio.run(test_mom_test_discovery())
    asyncio.run(test_context_integration())
    
    print("\n🎉 All Mom Test tests completed!")
    print("\n💡 Expected Mom Test Behaviors:")
    print("   1. Introduces as Ami, co-builder")
    print("   2. Asks about past behavior, not future hypotheticals")
    print("   3. Seeks specific, concrete examples")
    print("   4. Shows genuine curiosity about their work")
    print("   5. Avoids immediate AI pitching")
    print("   6. Uses conversational, natural tone")
    print("   7. Focuses on real work and problems") 