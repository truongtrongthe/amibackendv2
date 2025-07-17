"""
Tool execution coordinator for the LLM Tool Calling System
"""

from anthropic_tool import AnthropicTool
from openai_tool import OpenAITool
from search_tool import create_search_tool

def execute_tool(llm_provider: str, user_query: str) -> str:
    """
    Execute a tool using the specified LLM provider with appropriate search tool
    
    Args:
        llm_provider: Either 'anthropic' or 'openai'
        user_query: The user's input query
        
    Returns:
        Response from the LLM with tool execution results
    """
    
    # Create appropriate search tool based on LLM provider
    search_tool = create_search_tool(llm_provider)
    
    try:
        if llm_provider.lower() == "anthropic":
            llm = AnthropicTool()
            # For Anthropic, we use native web search (no external search tools needed)
            return llm.process_with_tools(user_query, available_tools=[], enable_web_search=True)
            
        elif llm_provider.lower() == "openai":
            llm = OpenAITool()
            # For OpenAI, we use external SERPAPI search tool
            return llm.process_with_tools(user_query, [search_tool])
            
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")
            
    except Exception as e:
        return f"Error executing tool: {str(e)}"

def get_available_tools():
    """
    Get list of available tools for different LLM providers
    
    Returns:
        Dict with available tools for each provider
    """
    return {
        "anthropic": {
            "web_search": "Native web search (built into Claude)",
            "context_retrieval": "Context and knowledge base access",
            "learning_tools": "Interactive learning capabilities"
        },
        "openai": {
            "search_google": "Google search via SERPAPI",
            "context_retrieval": "Context and knowledge base access", 
            "learning_tools": "Interactive learning capabilities"
        }
    } 