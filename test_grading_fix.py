#!/usr/bin/env python3
"""
Test script to verify the grading detection fix
Ensures grading requests are properly detected and handled without SERPAPI
"""

import asyncio
from exec_tool import ExecutiveTool, ToolExecutionRequest

async def test_grading_detection_fix():
    """Test that grading requests are properly detected and handled"""
    
    print("🔧 TESTING GRADING DETECTION FIX")
    print("=" * 50)
    print("Verifying that grading requests are detected before provider executors")
    print()
    
    # Initialize executive tool
    executive_tool = ExecutiveTool()
    
    # Test grading request that previously failed
    test_query = "I want to try out my agent's capabilities and see how it performs in a comprehensive grading scenario"
    
    print(f"🧪 Test Query: '{test_query[:80]}...'")
    print()
    
    # Test 1: Grading detection
    print("1️⃣ Testing grading detection...")
    is_grading = await executive_tool.detect_grading_request(test_query)
    print(f"   ✅ Grading detected: {is_grading}")
    
    if not is_grading:
        print("   ❌ ERROR: Grading request should have been detected!")
        return
    
    # Test 2: Create request and test flow
    print("\n2️⃣ Testing grading request flow...")
    
    request = ToolExecutionRequest(
        llm_provider="openai",
        user_query=test_query,
        enable_tools=True,
        cursor_mode=True,
        org_id="test_fix_org",
        user_id="test_fix_user"
    )
    
    # Track response types to ensure grading flow is triggered
    response_types = []
    grading_flow_detected = False
    serpapi_error = False
    
    try:
        print("   📡 Streaming response...")
        
        # Test the fixed execution flow
        async for chunk in executive_tool.execute_tool_stream(request):
            chunk_type = chunk.get("type", "unknown")
            response_types.append(chunk_type)
            
            # Check for grading-specific responses
            if chunk_type == "thinking":
                thought_type = chunk.get("thought_type", "")
                content = chunk.get("content", "")
                
                if thought_type == "grading_intent":
                    grading_flow_detected = True
                    print(f"   ✅ Grading flow triggered: {content[:60]}...")
                elif thought_type == "grading_method":
                    print(f"   ✅ Using internal tools: {content[:60]}...")
                elif "SERPAPI" in content:
                    serpapi_error = True
                    print(f"   ❌ SERPAPI error detected: {content}")
            
            elif chunk_type == "grading_scenario_proposal":
                print(f"   ✅ Grading scenario proposed!")
                
            elif chunk_type == "error" and "SERPAPI" in chunk.get("content", ""):
                serpapi_error = True
                print(f"   ❌ SERPAPI error: {chunk.get('content')}")
            
            # Limit output for testing
            if len(response_types) > 20:  # Prevent infinite loops
                print("   ⏭️ Truncating response for testing...")
                break
    
    except Exception as e:
        print(f"   ❌ Error during streaming: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 3: Analyze results
    print("\n3️⃣ Analyzing results...")
    print(f"   Response types received: {set(response_types)}")
    print(f"   ✅ Grading flow detected: {grading_flow_detected}")
    print(f"   ✅ No SERPAPI errors: {not serpapi_error}")
    
    # Test 4: Summary
    print("\n📊 TEST SUMMARY:")
    if grading_flow_detected and not serpapi_error:
        print("   🎉 SUCCESS: Grading detection fix is working!")
        print("   ✅ Grading requests are properly detected")
        print("   ✅ No SERPAPI dependency for grading")
        print("   ✅ Uses internal brain vector analysis")
    else:
        print("   ❌ FAILED: Issues detected")
        if not grading_flow_detected:
            print("   ❌ Grading flow was not triggered")
        if serpapi_error:
            print("   ❌ SERPAPI errors still occurring")

async def test_non_grading_request():
    """Test that non-grading requests still work normally"""
    
    print("\n🔄 TESTING NON-GRADING REQUEST")
    print("=" * 50)
    
    executive_tool = ExecutiveTool()
    
    # Non-grading query
    test_query = "What is the weather like today?"
    
    print(f"🧪 Non-grading Query: '{test_query}'")
    
    # Should NOT be detected as grading
    is_grading = await executive_tool.detect_grading_request(test_query)
    print(f"   ✅ Grading detection: {is_grading} (should be False)")
    
    if is_grading:
        print("   ❌ ERROR: Non-grading request incorrectly detected as grading!")
    else:
        print("   ✅ SUCCESS: Non-grading request properly handled")

if __name__ == "__main__":
    """Run grading detection fix tests"""
    
    print("🚀 Testing Grading Detection Fix")
    print("=" * 80)
    
    # Test the fix
    asyncio.run(test_grading_detection_fix())
    
    # Test non-grading requests still work
    asyncio.run(test_non_grading_request())
    
    print("\n" + "=" * 80)
    print("🎯 Fix Summary:")
    print("   • Grading detection added BEFORE provider executors")
    print("   • No SERPAPI dependency for grading scenarios")
    print("   • Uses internal brain vector analysis only")
    print("   • Supports Cursor-style diagram generation")
    print("🚀 The grading system should now work without SERPAPI!") 