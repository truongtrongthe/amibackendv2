#!/usr/bin/env python3
"""
Test script to demonstrate the grading tool integration with exec_tool
Shows the complete flow: grading request detection → capability analysis → scenario proposal → execution
"""

import asyncio
import json
from datetime import datetime
from exec_tool import ExecutiveTool, ToolExecutionRequest

async def test_grading_integration():
    """Test the complete grading integration"""
    
    print("🧪 GRADING TOOL INTEGRATION TEST")
    print("=" * 60)
    print("Testing comprehensive agent capability analysis and grading scenario generation")
    print()
    
    # Initialize the executive tool
    executive_tool = ExecutiveTool()
    
    # Test scenarios that should trigger grading flow
    grading_test_scenarios = [
        {
            "name": "Direct Grading Request",
            "query": "I want to try out my agent's capabilities",
            "expected_flow": ["grading detection", "capability analysis", "scenario proposal"]
        },
        {
            "name": "Testing Performance Request",
            "query": "Can you test my agent and show its best performance?",
            "expected_flow": ["grading detection", "comprehensive analysis", "optimal scenario"]
        },
        {
            "name": "Agent Evaluation Request", 
            "query": "Help me evaluate what my agent can do",
            "expected_flow": ["grading detection", "brain vector analysis", "demonstration proposal"]
        }
    ]
    
    for scenario in grading_test_scenarios:
        await test_grading_scenario(executive_tool, scenario)
    
    print("\n✅ All grading integration tests completed!")

async def test_grading_scenario(executive_tool: ExecutiveTool, scenario: dict):
    """Test a single grading scenario"""
    
    print(f"\n🧪 TESTING: {scenario['name']}")
    print("=" * 50)
    print(f"Query: '{scenario['query']}'")
    print(f"Expected Flow: {scenario['expected_flow']}")
    print("-" * 50)
    
    # Create grading request
    request = ToolExecutionRequest(
        llm_provider="openai",  # Use OpenAI for consistent testing
        user_query=scenario['query'],
        enable_tools=True,
        cursor_mode=True,
        org_id="test_grading_org",
        user_id="test_grading_user"
    )
    
    # Test grading detection
    is_grading = await executive_tool.detect_grading_request(scenario['query'])
    print(f"🔍 Grading detection: {'✅ DETECTED' if is_grading else '❌ NOT DETECTED'}")
    
    if not is_grading:
        print("⚠️ Query should have been detected as grading request")
        return
    
    # Track response components
    response_components = {
        "thinking_steps": [],
        "grading_analysis": [],
        "scenario_proposals": [],
        "agent_demonstrations": [],
        "error_messages": []
    }
    
    print("\n📡 STREAMING RESPONSE:")
    print("-" * 30)
    
    try:
        # Execute and capture all chunks
        async for chunk in executive_tool._handle_grading_request(request):
            await process_grading_chunk(chunk, response_components)
            
        # Analyze results
        print("\n📊 ANALYSIS RESULTS:")
        print("-" * 20)
        print(f"Thinking steps: {len(response_components['thinking_steps'])}")
        print(f"Grading analysis: {len(response_components['grading_analysis'])}")
        print(f"Scenario proposals: {len(response_components['scenario_proposals'])}")
        print(f"Agent demonstrations: {len(response_components['agent_demonstrations'])}")
        
        if response_components['error_messages']:
            print(f"❌ Errors: {len(response_components['error_messages'])}")
            for error in response_components['error_messages']:
                print(f"   • {error}")
        else:
            print("✅ No errors detected")
            
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()

async def process_grading_chunk(chunk: dict, components: dict):
    """Process and categorize a single response chunk"""
    
    chunk_type = chunk.get("type", "unknown")
    content = chunk.get("content", "")
    timestamp = chunk.get("timestamp", "")
    
    # Display chunk based on type
    if chunk_type == "thinking":
        thought_type = chunk.get("thought_type", "general")
        print(f"💭 [{thought_type}] {content}")
        components["thinking_steps"].append(chunk)
        
    elif chunk_type == "grading_scenario_proposal":
        print(f"🎯 [SCENARIO PROPOSAL] {content}")
        components["scenario_proposals"].append(chunk)
        
    elif chunk_type == "agent_demonstration":
        demo_step = chunk.get("demo_step", "unknown")
        print(f"🤖 [DEMO: {demo_step}] {content}")
        components["agent_demonstrations"].append(chunk)
        
    elif chunk_type == "grading_assessment":
        print(f"📊 [ASSESSMENT] {content}")
        components["grading_analysis"].append(chunk)
        
    elif chunk_type == "error":
        print(f"❌ [ERROR] {content}")
        components["error_messages"].append(content)
        
    else:
        print(f"📝 [{chunk_type.upper()}] {content}")

async def test_comprehensive_capability_analysis():
    """Test the comprehensive capability analysis separately"""
    
    print("\n🔬 TESTING COMPREHENSIVE CAPABILITY ANALYSIS")
    print("=" * 60)
    
    executive_tool = ExecutiveTool()
    
    request = ToolExecutionRequest(
        llm_provider="openai",
        user_query="test capabilities",
        org_id="test_org",
        user_id="test_user"
    )
    
    try:
        # Test comprehensive capability analysis
        capabilities = await executive_tool._read_comprehensive_agent_capabilities(request)
        
        if capabilities.get("success"):
            print(f"✅ Successfully analyzed {capabilities['total_vectors_analyzed']} vectors")
            print(f"   Domains covered: {capabilities['unique_domains_covered']}")
            print(f"   Capabilities found: {len(capabilities.get('capabilities', []))}")
            
            # Show sample capabilities
            sample_caps = capabilities.get('capabilities', [])[:3]
            if sample_caps:
                print("\n🎯 Sample Capabilities:")
                for cap in sample_caps:
                    print(f"   • {cap.skill} (confidence: {cap.confidence:.1%})")
        else:
            print(f"❌ Analysis failed: {capabilities.get('error')}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")

async def test_grading_tool_directly():
    """Test the grading tool directly"""
    
    print("\n🛠️ TESTING GRADING TOOL DIRECTLY")
    print("=" * 60)
    
    try:
        from grading_tool import GradingTool
        
        grading_tool = GradingTool()
        
        # Test capability analysis
        capabilities = await grading_tool.get_comprehensive_agent_capabilities(
            user_id="test_user",
            org_id="test_org",
            max_vectors=50
        )
        
        if capabilities.get("success"):
            print(f"✅ Direct grading tool test successful")
            print(f"   Vectors analyzed: {capabilities['total_vectors_analyzed']}")
            print(f"   Domains covered: {capabilities['unique_domains_covered']}")
        else:
            print(f"❌ Direct grading tool test failed: {capabilities.get('error')}")
            
    except Exception as e:
        print(f"❌ Direct grading tool test failed: {e}")

if __name__ == "__main__":
    """Run all grading integration tests"""
    
    print("🚀 Starting Grading Tool Integration Tests")
    print("=" * 80)
    
    asyncio.run(test_grading_integration())
    
    # Additional component tests
    asyncio.run(test_comprehensive_capability_analysis())
    asyncio.run(test_grading_tool_directly())
    
    print("\n🎉 All tests completed!")
    print("=" * 80) 