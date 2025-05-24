import asyncio
from grading import COTProcessorGrader, grade_cot_processor

async def main():
    """Example of how to use the grading module to validate COTProcessor."""
    
    print("=" * 60)
    print("COTProcessor Knowledge Validation Example")
    print("=" * 60)
    
    # Method 1: Using the quick grade function
    print("\n🚀 Method 1: Using grade_cot_processor() function")
    grader = await grade_cot_processor()
    
    # Method 2: Manual step-by-step validation
    print("\n🔧 Method 2: Manual step-by-step validation")
    
    # Create grader instance
    manual_grader = COTProcessorGrader()
    
    # Initialize COTProcessor
    print("Initializing COTProcessor...")
    success = await manual_grader.initialize_cot_processor()
    
    if success:
        print("✅ COTProcessor initialized successfully")
        
        # Manually trigger compilation (optional - already done in initialize)
        print("\nManually triggering compilation for testing...")
        compilation_result = await manual_grader.manually_trigger_compilation()
        print(f"Compilation status: {compilation_result.get('status', 'unknown')}")
        
        # Validate knowledge loading
        print("\nValidating knowledge loading...")
        validation_results = manual_grader.validate_knowledge_loading()
        
        # Access comprehensive skills externally
        print("\n🎯 Accessing comprehensive skills externally:")
        
        # Get comprehensive profiling skills
        profiling_skills = manual_grader.get_comprehensive_profiling_skills()
        if "error" not in profiling_skills:
            content_length = len(profiling_skills.get("knowledge_context", ""))
            metadata = profiling_skills.get("metadata", {})
            print(f"📊 Comprehensive Profiling Skills:")
            print(f"   - Content length: {content_length} characters")
            print(f"   - Source: {metadata.get('source', 'unknown')}")
            print(f"   - Timestamp: {metadata.get('timestamp', 'unknown')}")
            print(f"   - Language preserved: {metadata.get('language_preserved', 'unknown')}")
            
            # Preview first 200 characters
            preview = profiling_skills.get("knowledge_context", "")[:200]
            print(f"   - Preview: {preview}...")
        
        # Get comprehensive communication skills
        comm_skills = manual_grader.get_comprehensive_communication_skills()
        if "error" not in comm_skills:
            content_length = len(comm_skills.get("knowledge_context", ""))
            print(f"\n💬 Comprehensive Communication Skills:")
            print(f"   - Content length: {content_length} characters")
            
            # Preview first 200 characters
            preview = comm_skills.get("knowledge_context", "")[:200]
            print(f"   - Preview: {preview}...")
        
        # Get comprehensive business objectives
        business_obj = manual_grader.get_comprehensive_business_objectives()
        if "error" not in business_obj:
            content_length = len(business_obj.get("knowledge_context", ""))
            print(f"\n🎯 Comprehensive Business Objectives:")
            print(f"   - Content length: {content_length} characters")
            
            # Preview first 200 characters
            preview = business_obj.get("knowledge_context", "")[:200]
            print(f"   - Preview: {preview}...")
        
        # Print detailed knowledge summary
        manual_grader.print_knowledge_summary()
        
        # Export knowledge for inspection
        print("\n📁 Exporting knowledge for detailed inspection...")
        export_file = manual_grader.export_knowledge_for_inspection()
        
        # Print validation summary
        print(f"\n📋 Final Validation Summary:")
        print(f"Overall Status: {validation_results.get('overall_status', 'unknown')}")
        
        # Count valid knowledge bases
        basic_valid = sum(1 for v in validation_results.get('basic_knowledge', {}).values() if v.get('is_valid', False))
        comprehensive_valid = sum(1 for v in validation_results.get('comprehensive_knowledge', {}).values() if v.get('is_valid', False))
        
        print(f"Basic Knowledge: {basic_valid}/3 valid")
        print(f"Comprehensive Knowledge: {comprehensive_valid}/3 valid")
        
        if validation_results.get('overall_status') == 'success':
            print("🎉 All knowledge bases loaded and compiled successfully!")
        else:
            print("⚠️ Some issues detected. Check the detailed validation results above.")
    
    else:
        print("❌ Failed to initialize COTProcessor")

async def test_specific_knowledge():
    """Test accessing specific comprehensive knowledge types."""
    print("\n" + "=" * 60)
    print("Testing Specific Knowledge Access")
    print("=" * 60)
    
    grader = COTProcessorGrader()
    await grader.initialize_cot_processor()
    
    # Test accessing each type of comprehensive knowledge
    knowledge_types = [
        ("Profiling Skills", grader.get_comprehensive_profiling_skills),
        ("Communication Skills", grader.get_comprehensive_communication_skills),
        ("Business Objectives", grader.get_comprehensive_business_objectives)
    ]
    
    for name, getter_func in knowledge_types:
        print(f"\n🔍 Testing {name}:")
        knowledge = getter_func()
        
        if "error" in knowledge:
            print(f"   ❌ Error: {knowledge['error']}")
        else:
            # Analyze the knowledge structure
            context = knowledge.get("knowledge_context", "")
            metadata = knowledge.get("metadata", {})
            
            print(f"   ✅ Successfully accessed")
            print(f"   📏 Content length: {len(context)} characters")
            print(f"   🕒 Compiled at: {metadata.get('timestamp', 'unknown')}")
            print(f"   📚 Source: {metadata.get('source', 'unknown')}")
            
            # Check for Vietnamese content
            vietnamese_chars = sum(1 for char in context if ord(char) > 127)
            if vietnamese_chars > 0:
                print(f"   🇻🇳 Contains Vietnamese characters: {vietnamese_chars}")
            
            # Check content quality
            if len(context) > 1000:
                print(f"   💪 Rich content (good length)")
            elif len(context) > 100:
                print(f"   📝 Moderate content")
            else:
                print(f"   ⚠️ Limited content")

if __name__ == "__main__":
    # Run the main example
    asyncio.run(main())
    
    # Run specific knowledge testing
    asyncio.run(test_specific_knowledge()) 