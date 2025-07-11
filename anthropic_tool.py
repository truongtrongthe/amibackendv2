"""
Anthropic Claude Sonnet LLM implementation with tool calling
"""

import os
import json
from typing import List, Any, Dict, AsyncGenerator
from anthropic import Anthropic

class AnthropicTool:
    def __init__(self, model: str = "claude-3-5-sonnet-20241022"):
        """Initialize Anthropic client"""
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required")
        
        self.client = Anthropic(api_key=self.api_key)
        self.model = model  # Custom model name (e.g., "claude-3-5-haiku", "claude-3-5-sonnet")
    
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
    
    async def process_with_tools_stream(self, user_query: str, available_tools: List[Any], system_prompt: str = None, force_tools: bool = False) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process user query with available tools using Claude with streaming
        
        Args:
            user_query: The user's input query
            available_tools: List of available tool instances
            system_prompt: Optional custom system prompt
            force_tools: Whether to force tool usage (not directly supported by Anthropic, but influences behavior)
            
        Yields:
            Dict containing streaming response data
        """
        
        # Define tools for Claude only if tools are available
        tools = []
        if available_tools:
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

        # Prepare API call parameters
        api_params = {
            "model": self.model,
            "max_tokens": 1024,
            "stream": True
        }
        
        # Prepare messages with system prompt if provided
        messages = []
        if system_prompt:
            # For Claude, we can include system prompt in the user message
            messages.append({
                "role": "user",
                "content": f"System: {system_prompt}\n\nUser: {user_query}"
            })
        else:
            messages.append({
                "role": "user",
                "content": user_query
            })
        
        api_params["messages"] = messages
        
        # Only add tools if available
        if tools:
            api_params["tools"] = tools

        try:
            # First API call to Claude with streaming
            response_stream = self.client.messages.create(**api_params)
            
            content_buffer = ""
            tool_use_detected = False
            
            # Process streaming response
            for chunk in response_stream:
                if chunk.type == "content_block_start":
                    if chunk.content_block.type == "tool_use":
                        tool_use_detected = True
                        yield {
                            "type": "response_chunk",
                            "content": f"[Using tool: {chunk.content_block.name}]",
                            "complete": False
                        }
                elif chunk.type == "content_block_delta":
                    if chunk.delta.type == "text_delta":
                        content_buffer += chunk.delta.text
                        yield {
                            "type": "response_chunk",
                            "content": chunk.delta.text,
                            "complete": False
                        }
                elif chunk.type == "message_stop":
                    if tool_use_detected:
                        # Handle tool use with TRUE STREAMING like Cursor
                        yield {
                            "type": "response_chunk",
                            "content": "\n[Processing tool results...]",
                            "complete": False
                        }
                        
                        # Stream the tool execution and final response
                        async for final_chunk in self._stream_tool_execution(
                            user_query, available_tools, system_prompt
                        ):
                            yield final_chunk
                    else:
                        # Simple text response completed
                        yield {
                            "type": "response_complete",
                            "content": content_buffer,
                            "complete": True
                        }
                    break
                    
        except Exception as e:
            yield {
                "type": "error",
                "content": f"Error with Anthropic API: {str(e)}",
                "complete": True
            }
    
    async def _stream_tool_execution(
        self, 
        user_query: str, 
        available_tools: List[Any], 
        system_prompt: str = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Execute tools and stream the final response like Cursor does
        """
        try:
            # Step 1: Execute the tool search (non-streaming but fast)
            search_result = ""
            if available_tools:
                search_tool = available_tools[0]
                search_result = search_tool.search(user_query)
                
                yield {
                    "type": "response_chunk",
                    "content": "\n[Search completed, generating response...]",
                    "complete": False
                }
            
            # Step 2: Build conversation with tool results for Claude
            if system_prompt:
                final_query = f"System: {system_prompt}\n\nUser: {user_query}\n\nSearch Results:\n{search_result}\n\nBased on the search results above, please provide a comprehensive answer to the original question."
            else:
                final_query = f"User: {user_query}\n\nSearch Results:\n{search_result}\n\nBased on the search results above, please provide a comprehensive answer to the original question."
            
            # Step 3: Stream the final response using Claude
            final_response_params = {
                "model": self.model,
                "max_tokens": 1024,
                "messages": [
                    {
                        "role": "user",
                        "content": final_query
                    }
                ],
                "stream": True
            }
            
            response_stream = self.client.messages.create(**final_response_params)
            
            # Step 4: Stream the final response in real-time
            content_buffer = ""
            for chunk in response_stream:
                if chunk.type == "content_block_delta":
                    if chunk.delta.type == "text_delta":
                        content_buffer += chunk.delta.text
                        yield {
                            "type": "response_chunk",
                            "content": chunk.delta.text,
                            "complete": False
                        }
                elif chunk.type == "message_stop":
                    yield {
                        "type": "response_complete",
                        "content": content_buffer,
                        "complete": True
                    }
                    break
                    
        except Exception as e:
            yield {
                "type": "error",
                "content": f"Error in Anthropic tool execution streaming: {str(e)}",
                "complete": True
            }
    
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