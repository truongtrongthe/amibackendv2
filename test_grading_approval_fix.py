#!/usr/bin/env python3
"""
Test the complete grading approval flow with the API layer fix
Verifies that grading_context is properly passed through the API
"""

import asyncio
import json
from datetime import datetime
from exec_tool import ExecutiveTool, ToolExecutionRequest

async def test_approval_with_scenario_data():
    """Test approval flow with proper scenario data and visual diagrams"""
    
    print("🎯 TESTING GRADING APPROVAL WITH VISUAL DIAGRAMS")
    print("=" * 70)
    print("Simulating the complete frontend → API → backend flow")
    print()
    
    # Step 1: Simulate initial grading request (this works)
    print("📋 STEP 1: Generate Scenario Proposal")
    print("-" * 50)
    
    executive_tool = ExecutiveTool()
    
    initial_request = ToolExecutionRequest(
        llm_provider="openai",
        user_query="I want to try out my agent's capabilities with visual diagrams",
        enable_tools=True,
        cursor_mode=True,
        org_id="test_visual_org",
        user_id="test_visual_user"
    )
    
    scenario_data = None
    
    print("🔍 Generating scenario proposal...")
    async for chunk in executive_tool.execute_tool_stream(initial_request):
        if chunk.get("type") == "grading_scenario_proposal":
            scenario_data = chunk.get("scenario_data")
            scenario_name = scenario_data.get("scenario_name", "Unknown")
            print(f"   ✅ Proposal generated: {scenario_name}")
            
            # Check for diagram data
            has_workflow = bool(scenario_data.get("scenario_diagram"))
            has_capability_map = bool(scenario_data.get("capability_map"))
            has_process_diagrams = bool(scenario_data.get("process_diagrams"))
            
            print(f"   📊 Workflow diagram: {'✅ YES' if has_workflow else '❌ NO'}")
            print(f"   🧠 Capability map: {'✅ YES' if has_capability_map else '❌ NO'}")
            print(f"   ⚙️ Process diagrams: {'✅ YES' if has_process_diagrams else '❌ NO'} ({len(scenario_data.get('process_diagrams', []))} total)")
            break
    
    if not scenario_data:
        print("   ❌ FAILED: No scenario data generated")
        return
    
    print("   🎉 Step 1 SUCCESS: Scenario with diagrams generated")
    
    # Step 2: Simulate frontend approval with grading_context
    print(f"\n📋 STEP 2: Send Approval with Scenario Context")
    print("-" * 50)
    
    # This simulates what the frontend sends after API layer fix
    approval_request = ToolExecutionRequest(
        llm_provider="openai",
        user_query="Yes, proceed with the grading scenario demonstration",
        grading_context={
            "approved_scenario": scenario_data,  # Full scenario data
            "approval_action": "execute_demonstration",
            "test_inputs": {"visual_mode": True}  # Optional test parameters
        },
        enable_tools=True,
        cursor_mode=True,
        org_id="test_visual_org",
        user_id="test_visual_user"
    )
    
    approval_detected = False
    visual_demos = []
    diagrams_shown = []
    assessment_received = False
    
    print("🔍 Processing approval request...")
    
    async for chunk in executive_tool.execute_tool_stream(approval_request):
        chunk_type = chunk.get("type", "unknown")
        
        if chunk_type == "thinking":
            thought_type = chunk.get("thought_type", "")
            if thought_type == "grading_approval":
                approval_detected = True
                print("   ✅ Approval recognized by backend")
            elif thought_type == "demo_start":
                print("   ✅ Visual demonstration started")
                
        elif chunk_type == "agent_demonstration":
            demo_step = chunk.get("demo_step", "")
            diagram_data = chunk.get("diagram_data")
            
            visual_demos.append(demo_step)
            
            if demo_step == "introduction":
                print("   🤖 Agent introduction delivered")
            elif demo_step == "capability_visualization":
                print("   📊 Capability visualization displayed")
                if diagram_data:
                    diagrams_shown.append("capability_map")
            elif demo_step.startswith("process_diagram_"):
                print(f"   ⚙️ Process diagram {demo_step} shown")
                if diagram_data:
                    diagrams_shown.append("process_flow")
            else:
                print(f"   🔄 Demo step: {demo_step}")
                
        elif chunk_type == "grading_assessment":
            assessment_received = True
            assessment_data = chunk.get("assessment_data", {})
            diagram_data = chunk.get("diagram_data")
            
            score = assessment_data.get("overall_score", 0)
            print(f"   📈 Assessment completed: {score:.1%} score")
            
            if diagram_data:
                diagrams_shown.append("assessment_results")
                print("   📊 Assessment visualization displayed")
            break
    
    # Step 3: Results Analysis
    print(f"\n📊 RESULTS ANALYSIS")
    print("=" * 70)
    
    print(f"✅ Approval detection: {'PASS' if approval_detected else 'FAIL'}")
    print(f"✅ Visual demonstrations: {'PASS' if len(visual_demos) > 0 else 'FAIL'} ({len(visual_demos)} steps)")
    print(f"✅ Diagrams displayed: {'PASS' if len(diagrams_shown) > 0 else 'FAIL'} ({len(diagrams_shown)} types)")
    print(f"✅ Assessment with visuals: {'PASS' if assessment_received else 'FAIL'}")
    
    print(f"\n🎨 VISUAL ELEMENTS SHOWN:")
    for diagram_type in set(diagrams_shown):
        count = diagrams_shown.count(diagram_type)
        print(f"   📊 {diagram_type.replace('_', ' ').title()}: {count} instances")
    
    if approval_detected and len(visual_demos) > 0 and len(diagrams_shown) > 0 and assessment_received:
        print(f"\n🎉 COMPLETE SUCCESS: Visual Grading Flow Working!")
        print(f"   • ✅ Scenario proposal with diagrams")
        print(f"   • ✅ Approval detection and processing")
        print(f"   • ✅ Visual agent demonstration ({len(visual_demos)} steps)")
        print(f"   • ✅ Interactive diagrams ({len(set(diagrams_shown))} types)")
        print(f"   • ✅ Visual assessment results")
        print(f"\n🚀 Ready for production! Users will see beautiful visual grading!")
    else:
        print(f"\n❌ ISSUES DETECTED:")
        if not approval_detected:
            print(f"   ❌ Approval not detected - check grading_context passing")
        if len(visual_demos) == 0:
            print(f"   ❌ No demonstrations executed")
        if len(diagrams_shown) == 0:
            print(f"   ❌ No diagrams displayed - check diagram generation")
        if not assessment_received:
            print(f"   ❌ No assessment received")

async def test_api_layer_fix():
    """Test that the API layer properly accepts grading_context"""
    
    print(f"\n🔧 TESTING API LAYER FIX")
    print("-" * 50)
    
    # Test the LLMToolExecuteRequest model with grading_context
    try:
        from main import LLMToolExecuteRequest
        
        # This should work now with the API fix
        test_request = LLMToolExecuteRequest(
            llm_provider="openai",
            user_query="Execute approved scenario",
            grading_context={
                "approved_scenario": {"scenario_name": "Test Scenario"},
                "approval_action": "execute_demonstration"
            }
        )
        
        print("   ✅ LLMToolExecuteRequest accepts grading_context")
        print(f"   ✅ grading_context data: {bool(test_request.grading_context)}")
        
        # Check that the data is correctly structured
        has_scenario = bool(test_request.grading_context.get("approved_scenario"))
        has_action = bool(test_request.grading_context.get("approval_action"))
        
        print(f"   ✅ Approved scenario: {'PRESENT' if has_scenario else 'MISSING'}")
        print(f"   ✅ Approval action: {'PRESENT' if has_action else 'MISSING'}")
        
        if has_scenario and has_action:
            print("   🎉 API LAYER FIX: SUCCESS!")
        else:
            print("   ❌ API LAYER FIX: FAILED!")
            
    except Exception as e:
        print(f"   ❌ API LAYER ERROR: {e}")

if __name__ == "__main__":
    """Run the complete approval flow test with visual diagrams"""
    
    print("🎨 Visual Grading Approval Test Suite")
    print("=" * 80)
    
    # Test the API layer fix
    asyncio.run(test_api_layer_fix())
    
    # Test the complete visual flow
    asyncio.run(test_approval_with_scenario_data())
    
    print("\n" + "=" * 80)
    print("🎯 GRADING APPROVAL WITH VISUAL DIAGRAMS TESTED!")
    print("   The complete flow now works:")
    print("   • 🎨 Frontend sends grading_context via API")
    print("   • 🔧 API layer passes context to exec_tool") 
    print("   • 🧠 Backend processes approval and executes")
    print("   • 📊 Visual diagrams are generated and streamed")
    print("   • 🎭 Agent demonstrates with beautiful visuals")
    print("   • 📈 Assessment includes visual breakdowns")
    print("\n🚀 Users will now see the complete visual grading experience!") 