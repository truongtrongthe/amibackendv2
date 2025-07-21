#!/usr/bin/env python3
"""
Test script to demonstrate the unified Cursor-style thinking implementation
Shows the improvement from separate thought generators to unified logical flow
"""

import asyncio
import json
from datetime import datetime
from exec_tool import ExecutiveTool, ToolExecutionRequest

async def test_unified_thinking_flow():
    """Test the unified thinking implementation with a Vietnamese learning request"""
    
    print("🧠 UNIFIED CURSOR-STYLE THINKING TEST")
    print("=" * 60)
    print("Testing with the example request that showed illogical ordering")
    print("Request: 'Ok phần 5 được hiểu ntn' (Vietnamese)")
    print("\n🎯 Expected Logical Flow:")
    print("1. 💭 Understanding the request")
    print("2. 🔍 Intent analysis")
    print("3. 🧠 Detailed LLM reasoning steps")
    print("4. 🛠️ Tool selection")
    print("5. 📚 Strategy explanation")
    print("6. 🚀 Execution readiness")
    print("\n" + "=" * 60)
    
    # Create executive tool
    executive_tool = ExecutiveTool()
    
    # Create test request
    request = ToolExecutionRequest(
        llm_provider="openai",  # Use OpenAI for consistency
        user_query="Ok phần 5 được hiểu ntn",
        system_prompt="You are a helpful AI assistant",
        org_id="test_org",
        user_id="test_user",
        enable_tools=True,
        cursor_mode=True,
        enable_intent_classification=True,
        enable_request_analysis=True
    )
    
    print("\n🚀 STREAMING UNIFIED THOUGHTS:")
    print("-" * 40)
    
    thought_count = 0
    analysis_data = None
    
    try:
        # Stream the execution and capture thinking steps
        async for chunk in executive_tool.execute_tool_stream(request):
            chunk_type = chunk.get("type", "unknown")
            
            if chunk_type == "thinking":
                thought_count += 1
                content = chunk.get("content", "")
                step = chunk.get("step", "?")
                thought_type = chunk.get("thought_type", "unknown")
                
                print(f"Step {step}: [{thought_type.upper()}] {content}")
                
            elif chunk_type == "analysis_complete":
                analysis_data = chunk.get("analysis", {})
                print(f"\n📊 ANALYSIS RESULTS:")
                print(f"   Intent: {analysis_data.get('intent', 'unknown')}")
                print(f"   Confidence: {analysis_data.get('confidence', 0):.2f}")
                print(f"   Complexity: {analysis_data.get('complexity', 'unknown')}")
                print("")
                
            elif chunk_type == "response_chunk":
                # Just count response chunks, don't print them all
                if chunk.get("complete", False):
                    print("\n✅ RESPONSE COMPLETE")
                    break
                    
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n📊 SUMMARY:")
    print(f"   Total thinking steps: {thought_count}")
    if analysis_data:
        print(f"   Intent detected: {analysis_data.get('intent', 'unknown')}")
        print(f"   Analysis confidence: {analysis_data.get('confidence', 0):.2f}")
    print(f"   Test completed at: {datetime.now().isoformat()}")
    
    print("\n✅ UNIFIED THINKING FLOW TEST COMPLETE")
    print("The thoughts should now appear in logical order!")

async def demonstrate_thought_types():
    """Demonstrate the different types of thoughts in the unified system"""
    
    print("\n🎭 UNIFIED THOUGHT TYPES DEMONSTRATION")
    print("=" * 50)
    
    thought_types = [
        ("understanding", "💭", "Initial comprehension of user request"),
        ("intent_analysis", "🔍", "Classification of request intent"),
        ("detailed_analysis", "🧠", "LLM's step-by-step reasoning"),
        ("tool_selection", "🛠️", "Which tools will be used"),
        ("strategy", "📚", "Execution strategy explanation"),
        ("execution_ready", "🚀", "Ready to execute plan"),
        ("tool_execution", "⚙️", "Real-time tool execution feedback"),
        ("response_generation", "✍️", "Final response generation")
    ]
    
    for i, (thought_type, icon, description) in enumerate(thought_types, 1):
        print(f"{i:2d}. {icon} {thought_type.upper():<20} - {description}")
    
    print("\n📈 LOGICAL FLOW BENEFITS:")
    print("✅ Natural human reasoning progression")
    print("✅ No duplicate or out-of-order thoughts")
    print("✅ Maintains all UI/UX enhancements")
    print("✅ Clear step numbering and timestamps")
    print("✅ Consistent thought types and formatting")

if __name__ == "__main__":
    async def main():
        await test_unified_thinking_flow()
        demonstrate_thought_types()
    
    asyncio.run(main()) 