#!/usr/bin/env python3
"""
Simple test script to demonstrate COTProcessor knowledge access.
This shows the most direct way to access comprehensive_profiling_skills externally.
"""

import asyncio
from grading import COTProcessorGrader

async def simple_comprehensive_skills_test():
    """Simple demonstration of accessing comprehensive skills."""
    
    print("🔬 Simple COTProcessor Comprehensive Skills Test")
    print("-" * 50)
    
    # Create and initialize grader
    grader = COTProcessorGrader()
    
    print("1. Initializing COTProcessor...")
    success = await grader.initialize_cot_processor()
    
    if not success:
        print("❌ Failed to initialize COTProcessor")
        return
    
    print("✅ COTProcessor initialized successfully")
    
    # The key part: accessing comprehensive_profiling_skills externally
    print("\n2. Accessing comprehensive_profiling_skills externally...")
    
    comprehensive_profiling = grader.get_comprehensive_profiling_skills()
    
    if "error" in comprehensive_profiling:
        print(f"❌ Error accessing profiling skills: {comprehensive_profiling['error']}")
        return
    
    # Display the results
    knowledge_context = comprehensive_profiling.get("knowledge_context", "")
    metadata = comprehensive_profiling.get("metadata", {})
    
    print("✅ Successfully accessed comprehensive_profiling_skills!")
    print(f"📊 Content length: {len(knowledge_context)} characters")
    print(f"🕒 Compiled at: {metadata.get('timestamp', 'unknown')}")
    print(f"📚 Source: {metadata.get('source', 'unknown')}")
    
    # Show first 300 characters as preview
    if knowledge_context:
        preview = knowledge_context[:300]
        print(f"\n📝 Preview of comprehensive profiling skills:")
        print(f"'{preview}...'")
    else:
        print("⚠️ No knowledge content found")
    
    # Also show the other comprehensive skills
    print("\n3. Accessing other comprehensive skills...")
    
    # Communication skills
    comm_skills = grader.get_comprehensive_communication_skills()
    if "error" not in comm_skills:
        comm_length = len(comm_skills.get("knowledge_context", ""))
        print(f"💬 Comprehensive Communication Skills: {comm_length} chars")
    
    # Business objectives
    business_obj = grader.get_comprehensive_business_objectives()
    if "error" not in business_obj:
        business_length = len(business_obj.get("knowledge_context", ""))
        print(f"🎯 Comprehensive Business Objectives: {business_length} chars")
    
    print("\n✨ Test completed!")

async def direct_access_demo():
    """Demonstrate direct access to the COTProcessor instance."""
    
    print("\n" + "=" * 50)
    print("🔧 Direct Access Demo")
    print("=" * 50)
    
    grader = COTProcessorGrader()
    await grader.initialize_cot_processor()
    
    # Direct access to the COTProcessor instance
    cot_processor = grader.cot_processor
    
    if cot_processor:
        print("✅ Direct access to COTProcessor instance:")
        print(f"Graph Version ID: {cot_processor.graph_version_id}")
        
        # Direct access to comprehensive skills (alternative method)
        direct_profiling = cot_processor.comprehensive_profiling_skills
        direct_comm = cot_processor.comprehensive_communication_skills
        direct_business = cot_processor.comprehensive_business_objectives
        
        print(f"📊 Direct profiling access: {len(direct_profiling.get('knowledge_context', ''))} chars")
        print(f"💬 Direct communication access: {len(direct_comm.get('knowledge_context', ''))} chars")
        print(f"🎯 Direct business access: {len(direct_business.get('knowledge_context', ''))} chars")
        
        # You can also manually trigger compilation
        print("\n🔄 Manually triggering _compile_self_awareness...")
        compilation_result = await cot_processor._compile_self_awareness()
        print(f"Manual compilation result: {compilation_result.get('status', 'unknown')}")

if __name__ == "__main__":
    # Run the simple test
    asyncio.run(simple_comprehensive_skills_test())
    
    # Run the direct access demo
    asyncio.run(direct_access_demo()) 