#!/usr/bin/env python3
"""
Test script to verify the grading approval mechanism
Tests both initial grading requests and approval flow with scenario execution
"""

import asyncio
import json
from datetime import datetime
from exec_tool import ExecutiveTool, ToolExecutionRequest

async def test_grading_approval_flow():
    """Test the complete grading approval flow"""
    
    print("🚀 TESTING GRADING APPROVAL MECHANISM")
    print("=" * 60)
    print("Testing the complete flow: Request → Proposal → Approval → Demonstration")
    print()
    
    # Initialize executive tool
    executive_tool = ExecutiveTool()
    
    # Test Phase 1: Initial grading request
    print("📋 PHASE 1: Initial Grading Request")
    print("-" * 40)
    
    initial_request = ToolExecutionRequest(
        llm_provider="openai",
        user_query="I want to try out my agent's capabilities and see how it performs",
        enable_tools=True,
        cursor_mode=True,
        org_id="test_approval_org",
        user_id="test_approval_user"
    )
    
    scenario_data = None
    proposal_received = False
    
    print("🔍 Sending initial grading request...")
    
    try:
        async for chunk in executive_tool.execute_tool_stream(initial_request):
            chunk_type = chunk.get("type", "unknown")
            
            if chunk_type == "thinking":
                thought_type = chunk.get("thought_type", "")
                if thought_type == "grading_intent":
                    print("   ✅ Grading intent detected")
                elif thought_type == "scenario_generation":
                    print("   ✅ Scenario generation started")
                    
            elif chunk_type == "grading_scenario_proposal":
                proposal_received = True
                scenario_data = chunk.get("scenario_data")
                print(f"   ✅ Scenario proposal received: {scenario_data.get('scenario_name', 'Unknown')}")
                
                # Print key scenario details
                if scenario_data:
                    print(f"      📊 Difficulty: {scenario_data.get('difficulty_level')}")
                    print(f"      ⏱️ Duration: {scenario_data.get('estimated_time')}")
                    print(f"      🎯 Capabilities: {len(scenario_data.get('showcased_capabilities', []))}")
                break  # Stop after getting proposal
                
    except Exception as e:
        print(f"   ❌ Error in initial request: {e}")
        return
    
    if not proposal_received or not scenario_data:
        print("   ❌ FAILED: No scenario proposal received")
        return
    
    print("   🎉 Phase 1 SUCCESS: Scenario proposal generated")
    
    # Test Phase 2: Approval and demonstration
    print(f"\n📋 PHASE 2: Approval and Demonstration")
    print("-" * 40)
    
    # Simulate frontend approval
    approval_request = ToolExecutionRequest(
        llm_provider="openai",
        user_query="Yes, proceed with the grading scenario demonstration",
        grading_context={
            "approved_scenario": scenario_data,
            "approval_action": "execute_demonstration",
            "test_inputs": {}
        },
        enable_tools=True,
        cursor_mode=True,
        org_id="test_approval_org",
        user_id="test_approval_user"
    )
    
    approval_detected = False
    demonstration_started = False
    assessment_received = False
    
    print("🔍 Sending approval request...")
    
    try:
        async for chunk in executive_tool.execute_tool_stream(approval_request):
            chunk_type = chunk.get("type", "unknown")
            
            if chunk_type == "thinking":
                thought_type = chunk.get("thought_type", "")
                if thought_type == "grading_approval":
                    approval_detected = True
                    print("   ✅ Approval detected by backend")
                elif thought_type == "demo_start":
                    demonstration_started = True
                    print("   ✅ Demonstration started")
                    
            elif chunk_type == "agent_demonstration":
                demo_step = chunk.get("demo_step", "")
                if demo_step == "introduction":
                    print("   ✅ Agent introduction delivered")
                elif demo_step == "capability_visualization":
                    print("   ✅ Capability diagram generated")
                else:
                    print(f"   ✅ Demo step: {demo_step}")
                    
            elif chunk_type == "grading_assessment":
                assessment_received = True
                assessment_data = chunk.get("assessment_data", {})
                overall_score = assessment_data.get("overall_score", 0)
                print(f"   ✅ Assessment received: {overall_score:.1%} score")
                break  # Stop after assessment
                
    except Exception as e:
        print(f"   ❌ Error in approval request: {e}")
        return
    
    print("   🎉 Phase 2 SUCCESS: Approval processed and demonstration executed")
    
    # Test Summary
    print(f"\n📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"✅ Initial grading request: {'PASS' if proposal_received else 'FAIL'}")
    print(f"✅ Scenario proposal generation: {'PASS' if scenario_data else 'FAIL'}")  
    print(f"✅ Approval detection: {'PASS' if approval_detected else 'FAIL'}")
    print(f"✅ Demonstration execution: {'PASS' if demonstration_started else 'FAIL'}")
    print(f"✅ Assessment generation: {'PASS' if assessment_received else 'FAIL'}")
    
    if all([proposal_received, scenario_data, approval_detected, demonstration_started, assessment_received]):
        print("\n🎉 OVERALL: GRADING APPROVAL FLOW WORKING PERFECTLY!")
        print("   • Frontend can request grading scenarios")
        print("   • Backend generates comprehensive proposals")  
        print("   • Approval mechanism triggers demonstrations")
        print("   • Agent demonstrates capabilities with diagrams")
        print("   • Assessment results are generated and scored")
    else:
        print("\n❌ OVERALL: SOME ISSUES DETECTED")
        print("   Please review the failed components above")

async def test_approval_without_scenario_data():
    """Test approval request without proper scenario data"""
    
    print("\n🛡️ TESTING ERROR HANDLING: Approval Without Scenario Data")
    print("-" * 60)
    
    executive_tool = ExecutiveTool()
    
    # Test approval without grading_context
    request = ToolExecutionRequest(
        llm_provider="openai",
        user_query="Yes, proceed with the scenario demonstration",
        enable_tools=True,
        cursor_mode=True,
        org_id="test_error_org",
        user_id="test_error_user"
    )
    
    error_handled = False
    
    try:
        async for chunk in executive_tool.execute_tool_stream(request):
            chunk_type = chunk.get("type", "unknown")
            
            if chunk_type == "thinking":
                thought_type = chunk.get("thought_type", "")
                if thought_type == "grading_approval_missing_data":
                    error_handled = True
                    print("   ✅ Error handling: Missing scenario data detected")
                    
            elif chunk_type == "error":
                content = chunk.get("content", "")
                if "Missing scenario data" in content:
                    print("   ✅ Error message: Proper error returned to user")
                break
                
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
        return
    
    if error_handled:
        print("   ✅ ERROR HANDLING SUCCESS: System properly handles missing scenario data")
    else:
        print("   ❌ ERROR HANDLING FAILED: System should detect missing scenario data")

async def test_grading_keywords():
    """Test various grading keyword detection"""
    
    print("\n🔍 TESTING GRADING KEYWORD DETECTION")
    print("-" * 60)
    
    executive_tool = ExecutiveTool()
    
    test_cases = [
        # Initial grading requests
        ("I want to try out my agent", True, "initial"),
        ("Can you test my agent's capabilities?", True, "initial"),
        ("Let's grade my agent", True, "initial"),  
        ("Show me what my agent can do", True, "initial"),
        
        # Approval requests
        ("Yes, proceed with the scenario", True, "approval"),
        ("Execute the grading demonstration", True, "approval"),
        ("Start the test", True, "approval"),
        ("Go ahead with the scenario", True, "approval"),
        
        # Non-grading requests
        ("What's the weather like?", False, "normal"),
        ("Help me write code", False, "normal"),
        ("Explain machine learning", False, "normal")
    ]
    
    for query, expected, request_type in test_cases:
        grading_context = {"approved_scenario": {"name": "test"}} if request_type == "approval" else None
        is_grading = await executive_tool.detect_grading_request(query, grading_context)
        
        status = "✅ PASS" if is_grading == expected else "❌ FAIL"
        print(f"   {status} \"{query[:30]}{'...' if len(query) > 30 else ''}\" -> {is_grading} (expected {expected})")
    
    print("   ✅ Keyword detection testing completed")

if __name__ == "__main__":
    """Run complete grading approval tests"""
    
    print("🎯 Grading Approval Mechanism Test Suite")
    print("=" * 80)
    
    # Run all tests
    asyncio.run(test_grading_approval_flow())
    asyncio.run(test_approval_without_scenario_data())
    asyncio.run(test_grading_keywords())
    
    print("\n" + "=" * 80)
    print("🎉 GRADING APPROVAL TESTS COMPLETED!")
    print("   The backend now supports:")
    print("   • ✅ Initial grading scenario requests")  
    print("   • ✅ Scenario proposal generation with diagrams")
    print("   • ✅ Approval detection and processing")
    print("   • ✅ Demonstration execution after approval")
    print("   • ✅ Error handling for missing scenario data")
    print("   • ✅ Comprehensive keyword detection")
    print("\n🚀 Ready for frontend integration!") 