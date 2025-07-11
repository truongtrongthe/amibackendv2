"""
Test script for the new Context Tool integration

This demonstrates how the LLM can now call both search_google and get_context tools
"""

import asyncio
from exec_tool import execute_tool_stream


async def test_basic_context_usage():
    """Test basic usage where LLM decides whether to use context"""
    print("=== Test 1: Basic Context Usage ===")
    print("Query: 'What's my account status and what are your pricing plans?'")
    print()
    
    async for chunk in execute_tool_stream(
        llm_provider="openai",
        user_query="What's my account status and what are your pricing plans?",
        enable_tools=True  # Both search and context tools available
    ):
        if chunk["type"] == "response_chunk":
            print(chunk["content"], end="", flush=True)
    
    print("\n" + "="*70 + "\n")


async def test_context_only():
    """Test using only context tool (no web search)"""
    print("=== Test 2: Context Tool Only (No Web Search) ===")
    print("Query: 'Show me my recent activity and system status'")
    print()
    
    async for chunk in execute_tool_stream(
        llm_provider="anthropic", 
        user_query="Show me my recent activity and system status",
        enable_tools=True,
        tools_whitelist=["context"]  # Only context tool
    ):
        if chunk["type"] == "response_chunk":
            print(chunk["content"], end="", flush=True)
    
    print("\n" + "="*70 + "\n")


async def test_search_only():
    """Test using only search tool (no context)"""
    print("=== Test 3: Search Tool Only (No Context) ===")
    print("Query: 'What are the latest trends in AI development?'")
    print()
    
    async for chunk in execute_tool_stream(
        llm_provider="openai",
        user_query="What are the latest trends in AI development?", 
        enable_tools=True,
        tools_whitelist=["search"]  # Only search tool
    ):
        if chunk["type"] == "response_chunk":
            print(chunk["content"], end="", flush=True)
    
    print("\n" + "="*70 + "\n")


async def test_conversation_with_context():
    """Test conversation with history and context"""
    print("=== Test 4: Conversation with History and Context ===")
    print("Conversation about authentication issues...")
    print()
    
    conversation_history = [
        {"role": "user", "content": "I'm having trouble logging in"},
        {"role": "assistant", "content": "I can help with login issues. What specific error are you seeing?"},
        {"role": "user", "content": "It says my token has expired"}
    ]
    
    async for chunk in execute_tool_stream(
        llm_provider="anthropic",
        user_query="How do I fix this and prevent it in the future?",
        enable_tools=True,  # Both tools available
        conversation_history=conversation_history
    ):
        if chunk["type"] == "response_chunk":
            print(chunk["content"], end="", flush=True)
    
    print("\n" + "="*70 + "\n")


async def test_both_tools():
    """Test where LLM might use both search and context"""
    print("=== Test 5: Both Tools Available (LLM Chooses) ===")
    print("Query: 'I need help with API rate limiting best practices for my premium account'")
    print()
    
    async for chunk in execute_tool_stream(
        llm_provider="openai",
        user_query="I need help with API rate limiting best practices for my premium account",
        enable_tools=True,  # Both search and context available
        user_id="premium_user_123",
        org_id="enterprise_corp"
    ):
        if chunk["type"] == "response_chunk":
            print(chunk["content"], end="", flush=True)
    
    print("\n" + "="*70 + "\n")


async def main():
    """Run all tests"""
    print("🚀 Context Tool Integration Tests")
    print("=" * 70)
    print()
    
    try:
        await test_basic_context_usage()
        await test_context_only()
        await test_search_only()
        await test_conversation_with_context()
        await test_both_tools()
        
        print("✅ All tests completed!")
        print()
        print("Key Features Demonstrated:")
        print("- LLM explicitly calls get_context when needed")
        print("- Same architecture as search_tool")
        print("- Selective tool enabling via whitelist")
        print("- No frontend changes required")
        print("- Context from multiple sources (user, system, knowledge)")
        
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Testing the new Context Tool integration...")
    print("This shows how LLM can call both search_google and get_context tools")
    print()
    asyncio.run(main()) 