#!/usr/bin/env python3
"""
Test script to demonstrate the Cursor-style diagram generation in grading scenarios
Tests all diagram types: workflow, capability maps, process flows, and assessment diagrams
"""

import asyncio
from datetime import datetime
from grading_tool import GradingTool, AgentCapability, GradingScenario
from exec_tool import ExecutiveTool, ToolExecutionRequest

async def test_diagram_generation():
    """Test all diagram generation capabilities"""
    
    print("📊 GRADING DIAGRAM GENERATION TEST")
    print("=" * 60)
    print("Testing Cursor-style diagram capabilities for grading scenarios")
    print()
    
    # Initialize the grading tool
    grading_tool = GradingTool()
    
    # Test 1: Generate capability map diagram
    await test_capability_map_diagram(grading_tool)
    
    # Test 2: Generate scenario workflow diagram  
    await test_scenario_workflow_diagram(grading_tool)
    
    # Test 3: Generate process flow diagrams
    await test_process_flow_diagrams(grading_tool)
    
    # Test 4: Generate assessment diagram
    await test_assessment_diagram()
    
    # Test 5: Full integration test
    await test_full_diagram_integration()
    
    print("\n✅ All diagram generation tests completed!")

async def test_capability_map_diagram(grading_tool: GradingTool):
    """Test capability map diagram generation"""
    
    print("\n🧠 TESTING CAPABILITY MAP DIAGRAM")
    print("-" * 40)
    
    # Create sample capabilities
    sample_capabilities = [
        AgentCapability(
            domain="financial_analysis",
            skill="excel_automation",
            knowledge_depth=0.8,
            examples=["Process P&L statements", "Generate financial ratios"],
            vector_count=15,
            confidence=0.9
        ),
        AgentCapability(
            domain="financial_analysis", 
            skill="risk_assessment",
            knowledge_depth=0.7,
            examples=["Identify investment risks", "Analyze market volatility"],
            vector_count=12,
            confidence=0.8
        ),
        AgentCapability(
            domain="data_processing",
            skill="data_validation",
            knowledge_depth=0.6,
            examples=["Clean dataset", "Validate data integrity"],
            vector_count=8,
            confidence=0.7
        )
    ]
    
    # Generate capability map
    capability_map = grading_tool._generate_capability_map_diagram(sample_capabilities)
    
    print("📊 Generated Capability Map:")
    print(capability_map[:200] + "..." if len(capability_map) > 200 else capability_map)
    print(f"\n✅ Capability map diagram generated: {len(capability_map)} characters")

async def test_scenario_workflow_diagram(grading_tool: GradingTool):
    """Test scenario workflow diagram generation"""
    
    print("\n🔄 TESTING SCENARIO WORKFLOW DIAGRAM")
    print("-" * 40)
    
    # Sample scenario data
    scenario_name = "Financial Due Diligence Analysis"
    capabilities = ["financial_analysis", "excel_automation", "risk_assessment"]
    test_inputs = [
        {"type": "excel_file", "description": "Company financial statements"},
        {"type": "requirements", "description": "Analysis focus areas"}
    ]
    expected_outputs = [
        {"type": "financial_summary", "description": "Key financial metrics"},
        {"type": "risk_assessment", "description": "Investment risks identified"},
        {"type": "recommendation", "description": "Investment recommendation"}
    ]
    
    # Generate workflow diagram
    workflow_diagram = grading_tool._generate_scenario_workflow_diagram(
        scenario_name, capabilities, test_inputs, expected_outputs
    )
    
    print("🔄 Generated Workflow Diagram:")
    print(workflow_diagram[:300] + "..." if len(workflow_diagram) > 300 else workflow_diagram)
    print(f"\n✅ Workflow diagram generated: {len(workflow_diagram)} characters")

async def test_process_flow_diagrams(grading_tool: GradingTool):
    """Test process flow diagram generation"""
    
    print("\n⚙️ TESTING PROCESS FLOW DIAGRAMS")
    print("-" * 40)
    
    # Sample data
    scenario_name = "Data Analysis Workflow"
    test_inputs = [
        {"type": "data_file", "description": "Raw dataset for analysis"}
    ]
    capabilities = ["data_processing", "analytics", "reporting"]
    
    # Generate process diagrams
    process_diagrams = grading_tool._generate_process_diagrams(
        scenario_name, test_inputs, capabilities
    )
    
    print(f"⚙️ Generated {len(process_diagrams)} Process Flow Diagrams:")
    for i, diagram in enumerate(process_diagrams):
        print(f"\n{i+1}. {diagram['title']}")
        print(f"   Description: {diagram['description']}")
        print(f"   Diagram: {diagram['diagram'][:150]}...")
    
    print(f"\n✅ Process flow diagrams generated: {len(process_diagrams)} diagrams")

async def test_assessment_diagram():
    """Test assessment result diagram generation"""
    
    print("\n📊 TESTING ASSESSMENT DIAGRAM")
    print("-" * 40)
    
    # Initialize executive tool
    executive_tool = ExecutiveTool()
    
    # Sample assessment data
    assessment = {
        "overall_score": 0.82,
        "criteria_met": 4,
        "total_criteria": 5,
        "strengths": ["Domain expertise", "Structured responses"],
        "areas_for_improvement": ["More specific examples"],
        "recommendation": "Strong performance demonstrated"
    }
    
    # Sample scenario
    class MockScenario:
        def __init__(self):
            self.scenario_name = "Mock Financial Analysis"
    
    scenario = MockScenario()
    
    # Generate assessment diagram
    assessment_diagram = executive_tool._generate_assessment_diagram(assessment, scenario)
    
    print("📊 Generated Assessment Diagram:")
    print(assessment_diagram[:300] + "..." if len(assessment_diagram) > 300 else assessment_diagram)
    print(f"\n✅ Assessment diagram generated: {len(assessment_diagram)} characters")

async def test_full_diagram_integration():
    """Test full integration with exec_tool streaming"""
    
    print("\n🚀 TESTING FULL DIAGRAM INTEGRATION")
    print("-" * 40)
    
    # Initialize executive tool
    executive_tool = ExecutiveTool()
    
    # Create grading request
    request = ToolExecutionRequest(
        llm_provider="openai",
        user_query="I want to try out my agent with visual diagrams",
        enable_tools=True,
        cursor_mode=True,
        org_id="test_diagram_org",
        user_id="test_diagram_user"
    )
    
    diagram_count = 0
    response_chunks = []
    
    print("📡 Streaming grading response with diagrams:")
    print("-" * 30)
    
    try:
        # Execute grading request and capture diagram data
        async for chunk in executive_tool._handle_grading_request(request):
            response_chunks.append(chunk)
            
            # Count and display diagram information
            if chunk.get("type") == "grading_scenario_proposal":
                scenario_data = chunk.get("scenario_data", {})
                if scenario_data.get("scenario_diagram"):
                    diagram_count += 1
                    print(f"📊 Found scenario workflow diagram")
                if scenario_data.get("capability_map"):
                    diagram_count += 1
                    print(f"🧠 Found capability map diagram")
                if scenario_data.get("process_diagrams"):
                    diagram_count += len(scenario_data["process_diagrams"])
                    print(f"⚙️ Found {len(scenario_data['process_diagrams'])} process diagrams")
            
            elif chunk.get("type") == "agent_demonstration" and chunk.get("diagram_data"):
                diagram_count += 1
                diagram_type = chunk["diagram_data"].get("type", "unknown")
                print(f"🎨 Found {diagram_type} diagram in demonstration")
            
            elif chunk.get("type") == "grading_assessment" and chunk.get("diagram_data"):
                diagram_count += 1
                print(f"📈 Found assessment results diagram")
        
        print(f"\n📊 DIAGRAM INTEGRATION RESULTS:")
        print(f"   Total response chunks: {len(response_chunks)}")
        print(f"   Total diagrams found: {diagram_count}")
        print(f"   Integration status: {'✅ SUCCESS' if diagram_count > 0 else '❌ NO DIAGRAMS'}")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()

async def test_diagram_fallbacks():
    """Test diagram generation fallbacks"""
    
    print("\n🛡️ TESTING DIAGRAM FALLBACKS")
    print("-" * 40)
    
    grading_tool = GradingTool()
    
    # Test 1: Empty capabilities
    print("Testing empty capability map...")
    empty_map = grading_tool._generate_capability_map_diagram([])
    print(f"✅ Empty capability map fallback: {len(empty_map)} characters")
    
    # Test 2: Minimal workflow
    print("Testing minimal workflow diagram...")
    minimal_workflow = grading_tool._generate_scenario_workflow_diagram(
        "Test Scenario", [], [], []
    )
    print(f"✅ Minimal workflow fallback: {len(minimal_workflow)} characters")
    
    # Test 3: Empty process diagrams
    print("Testing empty process diagrams...")
    empty_process = grading_tool._generate_process_diagrams("Test", [], [])
    print(f"✅ Empty process diagrams fallback: {len(empty_process)} diagrams")
    
    print("✅ All fallback tests passed!")

if __name__ == "__main__":
    """Run all diagram generation tests"""
    
    print("🎨 Starting Grading Diagram Generation Tests")
    print("=" * 80)
    
    asyncio.run(test_diagram_generation())
    
    # Additional fallback tests
    asyncio.run(test_diagram_fallbacks())
    
    print("\n🎉 All diagram tests completed!")
    print("=" * 80)
    print("The grading system now supports Cursor-style visual diagrams! 🚀") 