"""
Tool execution coordinator for the LLM Tool Calling System
"""

from anthropic_tool import AnthropicTool
from openai_tool import OpenAITool
from search_tool import SearchTool

def execute_tool(llm_provider: str, user_query: str) -> str:
    """
    Execute a tool using the specified LLM provider
    
    Args:
        llm_provider: Either 'anthropic' or 'openai'
        user_query: The user's input query
        
    Returns:
        Response from the LLM with tool execution results
    """
    
    # Initialize search tool (shared by all LLMs)
    search_tool = SearchTool()
    
    try:
        if llm_provider.lower() == "anthropic":
            llm = AnthropicTool()
            return llm.process_with_tools(user_query, [search_tool])
            
        elif llm_provider.lower() == "openai":
            llm = OpenAITool()
            return llm.process_with_tools(user_query, [search_tool])
            
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")
            
    except Exception as e:
        return f"Error executing tool: {str(e)}"

def get_available_tools():
    """
    Get list of available tools
    
    Returns:
        List of available tool names
    """
    return ["search_tool"] 