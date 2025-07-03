"""
OpenAI GPT-4.1 LLM implementation with tool calling
"""

import os
import json
from typing import List, Any, Dict
from openai import OpenAI

class OpenAITool:
    def __init__(self):
        """Initialize OpenAI client"""
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = "gpt-4-1106-preview"  # GPT-4 Turbo with function calling
    
    def process_with_tools(self, user_query: str, available_tools: List[Any]) -> str:
        """
        Process user query with available tools using GPT-4
        
        Args:
            user_query: The user's input query
            available_tools: List of available tool instances
            
        Returns:
            Response from GPT-4 with tool execution results
        """
        
        # Define functions for OpenAI
        functions = []
        for tool in available_tools:
            functions.append({
                "type": "function",
                "function": {
                    "name": "search_google",
                    "description": "Search Google for information on any topic",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query to look up"
                            }
                        },
                        "required": ["query"]
                    }
                }
            })
        
        try:
            # First API call to GPT-4
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that can search for information when needed."
                    },
                    {
                        "role": "user",
                        "content": user_query
                    }
                ],
                tools=functions,
                tool_choice="auto"
            )
            
            message = response.choices[0].message
            
            # Check if GPT-4 wants to use tools
            if message.tool_calls:
                return self._handle_tool_calls(message, available_tools, user_query)
            else:
                return message.content
                
        except Exception as e:
            return f"Error with OpenAI API: {str(e)}"
    
    def _handle_tool_calls(self, message: Any, available_tools: List[Any], original_query: str) -> str:
        """
        Handle tool calls from GPT-4
        
        Args:
            message: GPT-4's message containing tool calls
            available_tools: List of available tool instances
            original_query: The original user query
            
        Returns:
            Final response after tool execution
        """
        
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that can search for information when needed."
            },
            {
                "role": "user",
                "content": original_query
            },
            message
        ]
        
        # Execute each tool call
        for tool_call in message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            # Execute the tool
            if function_name == "search_google" and available_tools:
                search_tool = available_tools[0]  # First tool is search tool
                result = search_tool.search(function_args.get("query", ""))
                
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": result
                })
        print("Tool result:", messages)
        # Send tool results back to GPT-4
        try:
            final_response = self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )
            
            return final_response.choices[0].message.content
            
        except Exception as e:
            return f"Error processing tool results: {str(e)}" 