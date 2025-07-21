#!/usr/bin/env python3
"""
Test script to demonstrate comprehensive tool call logging
Shows exactly when tools are called, their parameters, execution times, and results
"""

import asyncio
import json
import logging
from datetime import datetime
from exec_tool import ExecutiveTool, ToolExecutionRequest

# Configure logging to see tool execution clearly
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

async def test_tool_logging_comprehensive():
    """Test comprehensive tool logging with various scenarios"""
    
    print("🔧 COMPREHENSIVE TOOL CALL LOGGING TEST")
    print("=" * 70)
    print("This test demonstrates detailed logging for all tool executions:")
    print("• Console logging for server-side debugging")
    print("• Streaming events for frontend visibility")
    print("• Execution time tracking")
    print("• Parameter and result logging")
    print("• Error handling and status tracking")
    print("\n" + "=" * 70)
    
    # Create executive tool
    executive_tool = ExecutiveTool()
    
    test_scenarios = [
        {
            "name": "Learning Request with Tool Calls",
            "query": "Công ty của tôi có 100 nhân viên và chúng tôi đang phát triển sản phẩm AI",
            "description": "Vietnamese learning request that should trigger learning tools",
            "expected_tools": ["search_learning_context", "analyze_learning_opportunity"]
        },
        {
            "name": "General Question with Search",
            "query": "What are the latest trends in artificial intelligence for 2024?", 
            "description": "General question that should trigger search tool",
            "expected_tools": ["search_google"]
        },
        {
            "name": "Context-Heavy Request",
            "query": "Can you help me understand our organization's current status?",
            "description": "Request that should trigger context tool",
            "expected_tools": ["get_context"]
        }
    ]
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n🧪 TEST SCENARIO {i}: {scenario['name']}")
        print("-" * 50)
        print(f"Query: {scenario['query']}")
        print(f"Description: {scenario['description']}")
        print(f"Expected tools: {', '.join(scenario['expected_tools'])}")
        print("\n📋 CONSOLE LOGGING OUTPUT:")
        print("-" * 30)
        
        # Create test request
        request = ToolExecutionRequest(
            llm_provider="openai",
            user_query=scenario['query'],
            system_prompt="You are a helpful AI assistant that can search for information and learn from user input.",
            org_id="test_org",
            user_id="test_user", 
            enable_tools=True,
            cursor_mode=True,
            enable_intent_classification=True,
            enable_request_analysis=True,
            tools_whitelist=None  # Allow all tools
        )
        
        tools_detected = []
        tool_executions = []
        error_count = 0
        
        try:
            print("\n🔄 STREAMING EVENTS:")
            print("-" * 20)
            
            # Stream the execution and capture tool events
            async for chunk in executive_tool.execute_tool_stream(request):
                chunk_type = chunk.get("type", "unknown")
                
                if chunk_type == "thinking":
                    # Show thinking steps but abbreviated
                    content = chunk.get("content", "")[:80] + "..."
                    thought_type = chunk.get("thought_type", "unknown")
                    step = chunk.get("step", "?")
                    print(f"  💭 Step {step} [{thought_type}]: {content}")
                
                elif chunk_type == "tool_execution":
                    tool_name = chunk.get("tool_name", "unknown")
                    status = chunk.get("status", "unknown")
                    execution_time = chunk.get("execution_time", 0)
                    content = chunk.get("content", "")
                    
                    tools_detected.append(tool_name)
                    tool_executions.append({
                        "name": tool_name,
                        "status": status, 
                        "execution_time": execution_time,
                        "content": content
                    })
                    
                    if status == "error":
                        error_count += 1
                        print(f"  🔧❌ TOOL FAILED: {content}")
                    else:
                        print(f"  🔧✅ TOOL SUCCESS: {content}")
                
                elif chunk_type == "tools_summary":
                    summary = chunk.get("content", "")
                    total_time = chunk.get("total_execution_time", 0)
                    tools_data = chunk.get("tools_executed", [])
                    
                    print(f"  📊 SUMMARY: {summary}")
                    print(f"  ⏱️  Total execution time: {total_time:.2f}s")
                    
                elif chunk_type == "response_chunk":
                    if chunk.get("complete", False):
                        print(f"  ✅ Response generation completed")
                        break
                
                elif chunk_type == "error":
                    error_msg = chunk.get("content", "Unknown error")
                    print(f"  ❌ ERROR: {error_msg}")
                    error_count += 1
                    break
        
        except Exception as e:
            print(f"  ❌ EXCEPTION: {e}")
            error_count += 1
        
        # Test Results Summary
        print(f"\n📈 TEST RESULTS:")
        print(f"  Expected tools: {scenario['expected_tools']}")
        print(f"  Detected tools: {list(set(tools_detected))}")
        print(f"  Tool executions: {len(tool_executions)}")
        print(f"  Errors: {error_count}")
        
        # Check if expected tools were called
        expected_set = set(scenario['expected_tools'])
        detected_set = set(tools_detected)
        
        if expected_set.issubset(detected_set):
            print(f"  ✅ SUCCESS: All expected tools were called")
        else:
            missing = expected_set - detected_set
            print(f"  ⚠️  PARTIAL: Missing tools: {list(missing)}")
        
        print(f"\n{'='*70}")
        
        # Brief pause between tests
        await asyncio.sleep(1)

async def test_error_scenarios():
    """Test tool logging with error scenarios"""
    
    print("\n🚨 ERROR SCENARIO TESTING")
    print("=" * 50)
    print("Testing tool logging when errors occur...")
    
    executive_tool = ExecutiveTool()
    
    # Test with invalid API key scenario (common error)
    request = ToolExecutionRequest(
        llm_provider="anthropic",  # Try Anthropic which might have API key issues
        user_query="Test query that should fail",
        system_prompt="You are a test assistant.",
        org_id="test_org",
        user_id="test_user",
        enable_tools=True
    )
    
    error_detected = False
    
    try:
        print("\n🔄 Testing error logging...")
        async for chunk in executive_tool.execute_tool_stream(request):
            chunk_type = chunk.get("type", "")
            
            if chunk_type == "error":
                error_content = chunk.get("content", "")
                print(f"✅ Error properly logged: {error_content[:100]}...")
                error_detected = True
                break
            elif chunk_type == "tool_execution":
                status = chunk.get("status", "")
                if status == "error":
                    content = chunk.get("content", "")
                    print(f"✅ Tool error properly logged: {content[:100]}...")
                    error_detected = True
    
    except Exception as e:
        print(f"✅ Exception properly caught: {str(e)[:100]}...")
        error_detected = True
    
    if error_detected:
        print("✅ ERROR LOGGING: Working properly")
    else:
        print("⚠️  ERROR LOGGING: No errors detected (might be working too well!)")

def demonstrate_log_format():
    """Show the logging format and structure"""
    
    print("\n📋 LOGGING FORMAT DEMONSTRATION")
    print("=" * 50)
    print("The tool logging system provides multiple levels of visibility:\n")
    
    log_examples = [
        {
            "level": "CONSOLE LOGGING",
            "format": "🔧 [TOOL] HH:MM:SS - MESSAGE",
            "examples": [
                "🔧 [ANTHROPIC_TOOL] 14:23:15 - 🚀 Starting tool execution for query: 'What is AI?'",
                "🔧 [ANTHROPIC_TOOL] 14:23:15 - 📋 Available tools: 3 tools", 
                "🔧 [ANTHROPIC_TOOL] 14:23:15 - 🔍 EXECUTING: search_google",
                "🔧 [ANTHROPIC_TOOL] 14:23:15 -    Parameters: {'query': 'What is AI?'}",
                "🔧 [ANTHROPIC_TOOL] 14:23:17 - ✅ SUCCESS: search_google completed in 1.85s",
                "🔧 [ANTHROPIC_TOOL] 14:23:17 -    Result preview: AI (Artificial Intelligence) refers to...",
                "🔧 [ANTHROPIC_TOOL] 14:23:17 - 📊 TOOL EXECUTION SUMMARY:",
                "🔧 [ANTHROPIC_TOOL] 14:23:17 -    Total tools: 1",
                "🔧 [ANTHROPIC_TOOL] 14:23:17 -    Successful: 1",
                "🔧 [ANTHROPIC_TOOL] 14:23:17 -    Failed: 0",
                "🔧 [ANTHROPIC_TOOL] 14:23:17 -    Total time: 1.85s"
            ]
        },
        {
            "level": "STREAMING EVENTS",
            "format": "JSON events sent to frontend",
            "examples": [
                '{"type": "tool_execution", "tool_name": "search_google", "status": "completed", "execution_time": 1.85}',
                '{"type": "tools_summary", "content": "🏁 Tools completed: 1/1 successful (1.9s total)"}',
                '{"type": "thinking", "content": "🔍 Search completed (1.9s) - Found 2847 chars of results"}'
            ]
        }
    ]
    
    for log_type in log_examples:
        print(f"\n{log_type['level']}:")
        print(f"Format: {log_type['format']}")
        print("Examples:")
        for example in log_type['examples']:
            print(f"  {example}")
    
    print(f"\n🎯 KEY BENEFITS:")
    print("✅ Real-time visibility into tool execution")
    print("✅ Detailed parameter logging for debugging")
    print("✅ Execution time tracking for performance")
    print("✅ Error handling with full context")
    print("✅ Both server-side and client-side visibility")
    print("✅ Summary statistics for analysis")

async def main():
    """Run all tool logging tests"""
    await test_tool_logging_comprehensive()
    await test_error_scenarios()
    demonstrate_log_format()
    
    print(f"\n🏁 TOOL LOGGING TEST COMPLETE")
    print(f"All tool calls are now comprehensively logged!")
    print(f"Check console output and streaming events for full visibility.")

if __name__ == "__main__":
    asyncio.run(main()) 