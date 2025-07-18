#!/usr/bin/env python3
"""
Test script to verify "Guide me" scenario with no context
"""

import asyncio
import json
import aiohttp
from human_context_tool import HumanContextTool


async def test_no_context_scenario():
    """Test the 'Guide me' scenario with no context"""
    
    print("🧪 Testing 'Guide me' Scenario with No Context")
    print("=" * 60)
    
    # Test 1: Direct human context tool test
    print("\n📋 Test 1: Direct Human Context Tool Test")
    print("-" * 40)
    
    # Create tool with no database connections (simulates no context)
    tool = HumanContextTool(db_connection=None, braingraph_service=None)
    
    # Test with unknown org_id and user_id
    try:
        # This should return minimal/empty context
        human_context = await tool.get_human_context("unknown_user", "unknown_org")
        
        print(f"Context Retrieved:")
        print(f"  Org: {human_context['org_profile']['name']} ({human_context['org_profile']['industry']})")
        print(f"  User: {human_context['user_profile']['role']}")
        print(f"  Challenges: {human_context['org_profile']['challenges']}")
        
        # Generate discovery strategy with minimal context
        strategy = await tool.generate_discovery_strategy(human_context, "openai")
        
        print(f"\n📝 Generated Strategy for No Context:")
        print(f"  Opener: {strategy['opener']}")
        print(f"  Questions:")
        for i, q in enumerate(strategy['discovery_questions'], 1):
            print(f"    {i}. {q}")
        print(f"  Approach: {strategy['conversational_approach']}")
        
        # Analyze if it's Mom Test compliant
        is_mom_test = analyze_mom_test_compliance(strategy)
        print(f"\n📊 Mom Test Analysis:")
        for check, result in is_mom_test.items():
            status = "✅" if result else "❌"
            print(f"  {status} {check}")
        
    except Exception as e:
        print(f"❌ Error in direct test: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Full endpoint test
    print(f"\n📋 Test 2: Full Endpoint Test")
    print("-" * 40)
    
    await test_guide_me_endpoint()
    
    print("\n" + "=" * 60)
    print("🎯 No Context Scenario Testing Complete!")


async def test_guide_me_endpoint():
    """Test the full endpoint with 'Guide me' query"""
    
    base_url = "http://localhost:8000"
    endpoint = f"{base_url}/tool/llm"
    
    payload = {
        "llm_provider": "openai",
        "user_query": "Guide me",
        "enable_tools": True,
        "org_id": "unknown_org",  # This should trigger minimal context
        "user_id": "unknown_user"
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
                
                print("🔄 Testing 'Guide me' with no context...")
                
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
                
                # Analyze the response
                response_analysis = analyze_guide_me_response(full_response)
                
                print(f"📊 Response Analysis:")
                for check, result in response_analysis.items():
                    status = "✅" if result else "❌"
                    print(f"  {status} {check}")
                
                # Check if it follows the expected flow
                expected_flow = check_expected_flow(full_response)
                print(f"\n🎯 Expected Flow Check:")
                print(f"  ✅ Introduces as Ami: {expected_flow['introduces_ami']}")
                print(f"  ✅ Asks Mom Test questions: {expected_flow['asks_mom_test']}")
                print(f"  ✅ Avoids immediate AI suggestions: {expected_flow['avoids_ai_suggestions']}")
                print(f"  ✅ Shows curiosity about work: {expected_flow['shows_curiosity']}")
                
    except Exception as e:
        print(f"❌ Error during endpoint test: {e}")


def analyze_mom_test_compliance(strategy: dict) -> dict:
    """Analyze if the strategy follows Mom Test principles"""
    
    opener = strategy.get('opener', '').lower()
    questions = ' '.join(strategy.get('discovery_questions', [])).lower()
    approach = strategy.get('conversational_approach', '').lower()
    
    all_content = f"{opener} {questions} {approach}"
    
    return {
        "Introduces as Ami": "ami" in opener,
        "Asks about past behavior": any(word in questions for word in ["what did", "yesterday", "last", "spent", "took"]),
        "Seeks specific examples": any(word in questions for word in ["walk me through", "tell me", "how", "what happened"]),
        "Avoids AI pitching": not any(word in all_content for word in ["ai agent", "automate", "build together", "solution"]),
        "Shows genuine curiosity": any(word in all_content for word in ["curious", "tell me", "what", "how"]),
        "Conversational tone": any(word in all_content for word in ["hi", "hello", "curious", "tell me"])
    }


def analyze_guide_me_response(response: str) -> dict:
    """Analyze the response to 'Guide me' query"""
    
    response_lower = response.lower()
    
    return {
        "Introduces as Ami": "ami" in response_lower,
        "Asks discovery questions": any(word in response_lower for word in ["what", "how", "tell me", "curious"]),
        "Focuses on past behavior": any(word in response_lower for word in ["yesterday", "last", "spent", "did you"]),
        "Avoids immediate solutions": not any(word in response_lower for word in ["ai agent", "build together", "automate"]),
        "Shows curiosity": any(word in response_lower for word in ["curious", "interested", "tell me", "what do you"]),
        "Conversational tone": any(word in response_lower for word in ["hi", "hello", "i'm", "what's", "how's"])
    }


def check_expected_flow(response: str) -> dict:
    """Check if response follows expected Mom Test flow"""
    
    response_lower = response.lower()
    
    return {
        "introduces_ami": "ami" in response_lower,
        "asks_mom_test": any(word in response_lower for word in ["what did", "yesterday", "last", "spent", "how long"]),
        "avoids_ai_suggestions": not any(word in response_lower for word in ["ai agent", "build", "automate", "solution"]),
        "shows_curiosity": any(word in response_lower for word in ["curious", "tell me", "what", "how", "interested"])
    }


async def test_empty_context_generation():
    """Test LLM generation with truly empty context"""
    
    print(f"\n📋 Test 3: Empty Context Generation")
    print("-" * 40)
    
    # Create completely empty context
    empty_context = {
        "org_profile": {
            "name": "Unknown",
            "industry": "unknown",
            "size": "unknown",
            "challenges": [],
            "focus_areas": [],
            "tech_maturity": "unknown"
        },
        "user_profile": {
            "name": "User",
            "role": "unknown",
            "department": "unknown",
            "interests": [],
            "skills": [],
            "previous_projects": []
        },
        "braingraph_insights": {
            "user_interests": [],
            "work_patterns": [],
            "org_knowledge": [],
            "recent_topics": []
        },
        "conversation_patterns": {
            "patterns": [],
            "topics": [],
            "clues": []
        }
    }
    
    tool = HumanContextTool()
    
    try:
        strategy = await tool.generate_discovery_strategy(empty_context, "openai")
        
        print(f"📝 Strategy with Empty Context:")
        print(f"  Opener: {strategy['opener']}")
        print(f"  Questions:")
        for i, q in enumerate(strategy['discovery_questions'], 1):
            print(f"    {i}. {q}")
        
        # This should generate generic but Mom Test compliant questions
        is_generic_mom_test = analyze_generic_mom_test(strategy)
        print(f"\n📊 Generic Mom Test Check:")
        for check, result in is_generic_mom_test.items():
            status = "✅" if result else "❌"
            print(f"  {status} {check}")
        
    except Exception as e:
        print(f"❌ Error in empty context test: {e}")


def analyze_generic_mom_test(strategy: dict) -> dict:
    """Analyze if generic strategy is still Mom Test compliant"""
    
    questions = ' '.join(strategy.get('discovery_questions', [])).lower()
    
    return {
        "Generic but specific": any(word in questions for word in ["yesterday", "last", "spent", "did you"]),
        "Work-focused": any(word in questions for word in ["work", "job", "task", "project"]),
        "Time-focused": any(word in questions for word in ["time", "long", "took", "spent"]),
        "Past behavior": any(word in questions for word in ["what did", "yesterday", "last", "spent"]),
        "No AI mention": not any(word in questions for word in ["ai", "automate", "agent", "build"])
    }


if __name__ == "__main__":
    print("🚀 Starting No Context Scenario Tests...")
    
    # Run the tests
    asyncio.run(test_no_context_scenario())
    asyncio.run(test_empty_context_generation())
    
    print("\n🎉 No context scenario tests completed!")
    print("\n💡 Expected Behavior for 'Guide me' with no context:")
    print("   1. Ami introduces herself as co-builder")
    print("   2. Asks generic Mom Test questions about work")
    print("   3. Shows curiosity about their daily tasks")
    print("   4. Avoids suggesting AI agents immediately")
    print("   5. Focuses on understanding their actual work first") 