"""
Anthropic Claude Sonnet LLM implementation with tool calling
"""

import os
import json
from typing import List, Any, Dict
from anthropic import Anthropic

class AnthropicTool:
    def __init__(self):
        """Initialize Anthropic client"""
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required")
        
        self.client = Anthropic(api_key=self.api_key)
        self.model = "claude-3-5-sonnet-20241022"  # Latest Claude Sonnet model
    
    def process_with_tools(self, user_query: str, available_tools: List[Any]) -> str:
        """
        Process user query with available tools using Claude
        
        Args:
            user_query: The user's input query
            available_tools: List of available tool instances
            
        Returns:
            Response from Claude with tool execution results
        """
        
        # Define tools for Claude
        tools = []
        for tool in available_tools:
            tools.append({
                "name": "search_google",
                "description": "Search Google for information on any topic",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to look up"
                        }
                    },
                    "required": ["query"]
                }
            })
        
        try:
            # First API call to Claude
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                tools=tools,
                messages=[
                    {
                        "role": "user",
                        "content": user_query
                    }
                ]
            )
            
            # Check if Claude wants to use tools
            if response.stop_reason == "tool_use":
                return self._handle_tool_use(response, available_tools, user_query)
            else:
                return response.content[0].text
                
        except Exception as e:
            return f"Error with Anthropic API: {str(e)}"
    
    def _handle_tool_use(self, response: Any, available_tools: List[Any], original_query: str) -> str:
        """
        Handle tool use requests from Claude
        
        Args:
            response: Claude's response containing tool use requests
            available_tools: List of available tool instances
            original_query: The original user query
            
        Returns:
            Final response after tool execution
        """
        
        tool_results = []
        
        for content_block in response.content:
            if content_block.type == "tool_use":
                tool_name = content_block.name
                tool_input = content_block.input
                tool_use_id = content_block.id
                
                # Execute the tool
                if tool_name == "search_google" and available_tools:
                    search_tool = available_tools[0]  # First tool is search tool
                    result = search_tool.search(tool_input.get("query", ""))
                    
                    tool_results.append({
                        "tool_use_id": tool_use_id,
                        "type": "tool_result",
                        "content": result
                    })
        
        # Send tool results back to Claude
        try:
            messages = [
                {"role": "user", "content": original_query},
                {"role": "assistant", "content": response.content},
                {"role": "user", "content": tool_results}
            ]
            
            final_response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=messages
            )
            
            return final_response.content[0].text
            
        except Exception as e:
            return f"Error processing tool results: {str(e)}" 