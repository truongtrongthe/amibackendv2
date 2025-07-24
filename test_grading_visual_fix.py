#!/usr/bin/env python3
"""
Test the visual diagram fix for grading scenarios
Verifies that the fallback scenario now generates diagrams
"""

import asyncio
import json
from datetime import datetime
from exec_tool import ExecutiveTool, ToolExecutionRequest

async def test_visual_diagram_fix():
    """Test that the fallback scenario now generates visual diagrams"""
    
    print("🎨 TESTING VISUAL DIAGRAM FIX")
    print("=" * 60)
    print("Testing that fallback scenario generates diagrams...")
    print()
    
    executive_tool = ExecutiveTool()
    
    # Test 1: Generate scenario proposal
    print("📋 STEP 1: Generate Scenario with Visual Diagrams")
    print("-" * 50)
    
    initial_request = ToolExecutionRequest(
        llm_provider="openai",
        user_query="I want to try out my agent's capabilities with beautiful visual diagrams",
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
            
            # Check for diagram data (this should now work!)
            has_workflow = bool(scenario_data.get("scenario_diagram"))
            has_capability_map = bool(scenario_data.get("capability_map"))
            has_process_diagrams = bool(scenario_data.get("process_diagrams"))
            
            print(f"   📊 Workflow diagram: {'✅ YES' if has_workflow else '❌ NO'}")
            print(f"   🧠 Capability map: {'✅ YES' if has_capability_map else '❌ NO'}")
            print(f"   ⚙️ Process diagrams: {'✅ YES' if has_process_diagrams else '❌ NO'} ({len(scenario_data.get('process_diagrams', []))} total)")
            
            if has_workflow and has_capability_map and has_process_diagrams:
                print(f"   🎉 SUCCESS: All diagrams generated!")
            else:
                print(f"   ❌ FAILED: Some diagrams missing")
            break
    
    if not scenario_data:
        print("   ❌ FAILED: No scenario data generated")
        return False
    
    # Test 2: Execute the scenario with diagrams
    print(f"\n📋 STEP 2: Execute Scenario with Visual Demonstration")
    print("-" * 50)
    
    approval_request = ToolExecutionRequest(
        llm_provider="openai",
        user_query="Yes, proceed with the visual grading scenario demonstration",
        grading_context={
            "approved_scenario": scenario_data,
            "approval_action": "execute_demonstration",
            "test_inputs": {"visual_mode": True}
        },
        enable_tools=True,
        cursor_mode=True,
        org_id="test_visual_org",
        user_id="test_visual_user"
    )
    
    visual_elements_shown = []
    demonstration_completed = False
    
    print("🔍 Processing approval and demonstration...")
    
    async for chunk in executive_tool.execute_tool_stream(approval_request):
        chunk_type = chunk.get("type", "unknown")
        
        if chunk_type == "thinking":
            thought_type = chunk.get("thought_type", "")
            if thought_type == "grading_approval":
                print("   ✅ Approval recognized")
            elif thought_type == "demo_start":
                print("   ✅ Visual demonstration started")
                
        elif chunk_type == "agent_demonstration":
            demo_step = chunk.get("demo_step", "")
            diagram_data = chunk.get("diagram_data")
            
            if demo_step == "introduction":
                print("   🤖 Agent introduction delivered")
            elif demo_step == "capability_visualization":
                print("   📊 Capability visualization displayed")
                if diagram_data:
                    visual_elements_shown.append("capability_map")
            elif demo_step.startswith("process_diagram_"):
                print(f"   ⚙️ Process diagram shown: {demo_step}")
                if diagram_data:
                    visual_elements_shown.append("process_flow")
            else:
                print(f"   🔄 Demo step: {demo_step}")
                
        elif chunk_type == "grading_assessment":
            demonstration_completed = True
            diagram_data = chunk.get("diagram_data")
            assessment_data = chunk.get("assessment_data", {})
            
            score = assessment_data.get("overall_score", 0)
            print(f"   📈 Assessment completed: {score:.1%} score")
            
            if diagram_data:
                visual_elements_shown.append("assessment_results")
                print("   📊 Assessment visualization displayed")
            break
    
    # Results
    print(f"\n📊 VISUAL DIAGRAM TEST RESULTS")
    print("=" * 60)
    
    diagrams_in_scenario = all([
        bool(scenario_data.get("scenario_diagram")),
        bool(scenario_data.get("capability_map")),
        bool(scenario_data.get("process_diagrams"))
    ])
    
    visual_demo_working = len(visual_elements_shown) > 0
    
    print(f"✅ Scenario diagram generation: {'PASS' if diagrams_in_scenario else 'FAIL'}")
    print(f"✅ Visual demonstration: {'PASS' if visual_demo_working else 'FAIL'}")
    print(f"✅ Complete flow: {'PASS' if demonstration_completed else 'FAIL'}")
    
    print(f"\n🎨 Visual Elements Rendered:")
    for element in set(visual_elements_shown):
        count = visual_elements_shown.count(element)
        print(f"   📊 {element.replace('_', ' ').title()}: {count} instances")
    
    success = diagrams_in_scenario and visual_demo_working and demonstration_completed
    
    if success:
        print(f"\n🎉 COMPLETE SUCCESS: Visual Grading System Working!")
        print(f"   • ✅ Fallback scenario generates diagrams")
        print(f"   • ✅ Visual demonstrations work properly")
        print(f"   • ✅ Assessment includes visual breakdowns")
        print(f"\n🚀 Users will now see beautiful visual grading experiences!")
    else:
        print(f"\n❌ ISSUES REMAIN:")
        if not diagrams_in_scenario:
            print(f"   ❌ Scenario diagrams still not generated")
        if not visual_demo_working:
            print(f"   ❌ Visual demonstration not working")
        if not demonstration_completed:
            print(f"   ❌ Demonstration flow incomplete")
    
    return success

if __name__ == "__main__":
    """Test the visual diagram fix"""
    
    print("🎨 Visual Grading Diagram Fix Test")
    print("=" * 80)
    
    success = asyncio.run(test_visual_diagram_fix())
    
    print("\n" + "=" * 80)
    if success:
        print("🎯 VISUAL DIAGRAM FIX: SUCCESS!")
        print("   The grading system now generates beautiful diagrams!")
        print("   • ✅ Fallback scenarios include visual elements")
        print("   • ✅ Approval flow triggers visual demonstrations")
        print("   • ✅ Assessment results show visual breakdowns")
        print("\n🚀 Ready for production with visual grading!")
    else:
        print("❌ VISUAL DIAGRAM FIX: ISSUES DETECTED")
        print("   Additional debugging needed for visual elements") 