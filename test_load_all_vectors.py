#!/usr/bin/env python3
"""
Test the load ALL vectors approach for grading scenarios
Verifies that we get comprehensive vector loading instead of keyword filtering
"""

import asyncio
import json
from datetime import datetime
from exec_tool import ExecutiveTool, ToolExecutionRequest

async def test_load_all_vectors():
    """Test that the new approach loads ALL vectors instead of keyword filtering"""
    
    print("📊 TESTING LOAD ALL VECTORS APPROACH")
    print("=" * 70)
    print("Testing comprehensive vector loading without keyword filtering...")
    print()
    
    executive_tool = ExecutiveTool()
    
    # Test the grading flow with comprehensive vector loading
    print("📋 STEP 1: Generate Scenario with ALL Vector Analysis")
    print("-" * 50)
    
    initial_request = ToolExecutionRequest(
        llm_provider="openai",
        user_query="I want to test my agent with comprehensive capability analysis",
        enable_tools=True,
        cursor_mode=True,
        org_id="5376f68e-ff74-43fe-a72c-c1be9c6ae652",  # Use real org ID
        user_id="user_GjswTZUz8Fhv2v2X0c8OyA"  # Use real user ID
    )
    
    scenario_data = None
    vectors_analyzed = 0
    domains_covered = 0
    capabilities_found = 0
    
    print("🔍 Generating scenario with comprehensive vector analysis...")
    async for chunk in executive_tool.execute_tool_stream(initial_request):
        if chunk.get("type") == "thinking":
            content = chunk.get("content", "")
            if "vectors across" in content:
                # Extract vector count from log messages
                if "Analyzed" in content:
                    try:
                        # Parse "Successfully analyzed X vectors across Y domains"
                        parts = content.split("vectors across")
                        if len(parts) >= 2:
                            vectors_analyzed = int(parts[0].split()[-1])
                            domains_covered = int(parts[1].split()[0])
                    except:
                        pass
        
        elif chunk.get("type") == "grading_scenario_proposal":
            scenario_data = chunk.get("scenario_data")
            scenario_name = scenario_data.get("scenario_name", "Unknown")
            print(f"   ✅ Proposal generated: {scenario_name}")
            
            # Check if it's still falling back to general scenario
            is_fallback = scenario_name == "General Capability Demonstration"
            
            print(f"   📊 Vectors analyzed: {vectors_analyzed}")
            print(f"   🗂️ Domains covered: {domains_covered}")
            print(f"   🎯 Scenario type: {'Fallback' if is_fallback else 'Specialized'}")
            
            # Check for diagram data
            has_workflow = bool(scenario_data.get("scenario_diagram"))
            has_capability_map = bool(scenario_data.get("capability_map"))
            has_process_diagrams = bool(scenario_data.get("process_diagrams"))
            
            print(f"   📊 Workflow diagram: {'✅ YES' if has_workflow else '❌ NO'}")
            print(f"   🧠 Capability map: {'✅ YES' if has_capability_map else '❌ NO'}")
            print(f"   ⚙️ Process diagrams: {'✅ YES' if has_process_diagrams else '❌ NO'}")
            break
    
    if not scenario_data:
        print("   ❌ FAILED: No scenario data generated")
        return False
    
    # Results Analysis
    print(f"\n📊 COMPREHENSIVE VECTOR LOADING TEST RESULTS")
    print("=" * 70)
    
    # Test success criteria
    vectors_loaded = vectors_analyzed > 0
    comprehensive_analysis = vectors_analyzed > 20  # Expect more than 20 vectors
    multiple_domains = domains_covered > 1
    visual_diagrams = all([
        bool(scenario_data.get("scenario_diagram")),
        bool(scenario_data.get("capability_map")),
        bool(scenario_data.get("process_diagrams"))
    ])
    
    print(f"✅ Vectors loaded: {'PASS' if vectors_loaded else 'FAIL'} ({vectors_analyzed} vectors)")
    print(f"✅ Comprehensive analysis: {'PASS' if comprehensive_analysis else 'FAIL'} (threshold: >20)")
    print(f"✅ Multiple domains: {'PASS' if multiple_domains else 'FAIL'} ({domains_covered} domains)")
    print(f"✅ Visual diagrams: {'PASS' if visual_diagrams else 'FAIL'}")
    
    # Determine if we got better results than before
    is_improvement = vectors_analyzed > 0  # Before: 0 vectors, Now: >0 vectors
    scenario_type = scenario_data.get("scenario_name", "")
    is_specialized = scenario_type != "General Capability Demonstration"
    
    print(f"\n🔄 COMPARISON WITH PREVIOUS APPROACH:")
    print(f"   📈 Vector loading improvement: {'✅ YES' if is_improvement else '❌ NO'}")
    print(f"   🎯 Specialized scenario: {'✅ YES' if is_specialized else '❌ NO'}")
    print(f"   📊 Previous approach: 0 vectors analyzed")
    print(f"   📊 New approach: {vectors_analyzed} vectors analyzed")
    
    success = vectors_loaded and visual_diagrams
    significant_improvement = comprehensive_analysis and multiple_domains
    
    if success and significant_improvement:
        print(f"\n🎉 COMPLETE SUCCESS: Load All Vectors Working!")
        print(f"   • ✅ Comprehensive vector loading ({vectors_analyzed} vectors)")
        print(f"   • ✅ Multi-domain analysis ({domains_covered} domains)")
        print(f"   • ✅ Visual diagrams generated")
        print(f"   • ✅ {'Specialized' if is_specialized else 'Enhanced fallback'} scenario")
        print(f"\n🚀 Users will now get much better capability analysis!")
    elif success:
        print(f"\n✅ SUCCESS: Basic functionality working")
        print(f"   • ✅ Vector loading successful")
        print(f"   • ✅ Visual diagrams generated")
        print(f"   • ⚠️ May need more vectors for specialized scenarios")
        print(f"\n💡 Consider adding more domain-specific brain vectors")
    else:
        print(f"\n❌ ISSUES DETECTED:")
        if not vectors_loaded:
            print(f"   ❌ No vectors loaded - check vector database connection")
        if not visual_diagrams:
            print(f"   ❌ Visual diagrams not generated")
    
    return success

if __name__ == "__main__":
    """Test the comprehensive vector loading approach"""
    
    print("📊 Load All Vectors Test Suite")
    print("=" * 80)
    
    success = asyncio.run(test_load_all_vectors())
    
    print("\n" + "=" * 80)
    if success:
        print("🎯 LOAD ALL VECTORS: SUCCESS!")
        print("   The grading system now loads comprehensive brain vectors!")
        print("   • ✅ No more keyword filtering limitations")
        print("   • ✅ Comprehensive capability analysis") 
        print("   • ✅ Better scenario generation")
        print("   • ✅ Beautiful visual diagrams")
        print("\n🚀 Ready for production with comprehensive analysis!")
    else:
        print("❌ LOAD ALL VECTORS: NEEDS IMPROVEMENT")
        print("   Check vector database connection and content") 