"""
OpenAI GPT-4.1 LLM implementation with tool calling
"""

import os
import json
from typing import List, Any, Dict, AsyncGenerator
from openai import OpenAI

class OpenAITool:
    def __init__(self, model: str = "gpt-4-1106-preview"):
        """Initialize OpenAI client"""
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model  # Custom model name (e.g., "gpt-4o", "gpt-4-turbo")
    
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
    
    async def process_with_tools_stream(self, user_query: str, available_tools: List[Any], system_prompt: str = None, force_tools: bool = False, conversation_history: List[Dict[str, Any]] = None, max_history_messages: int = 25, max_history_tokens: int = 6000) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process user query with available tools using GPT-4 with streaming
        
        Args:
            user_query: The user's input query
            available_tools: List of available tool instances
            system_prompt: Optional custom system prompt
            force_tools: Whether to force tool usage (tool_choice="required")
            conversation_history: Previous conversation messages
            max_history_messages: Maximum number of history messages to include
            max_history_tokens: Maximum token count for history
            
        Yields:
            Dict containing streaming response data
        """
        
        # Use custom system prompt or default
        if system_prompt is None:
            system_prompt = "You are a helpful assistant that can search for information when needed."
        
        # Define functions for OpenAI only if tools are available
        functions = []
        tool_choice = "auto"
        
        if available_tools:
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
            
            # Set tool choice based on force_tools parameter
            if force_tools:
                tool_choice = "required"
            else:
                tool_choice = "auto"
        else:
            # No tools available, set to none
            tool_choice = "none"

        # Build messages with conversation history
        messages = [
            {
                "role": "system",
                "content": system_prompt
            }
        ]
        
        # Add conversation history if provided
        if conversation_history:
            history_messages = []
            token_count = 0
            message_count = 0
            
            # Process history in reverse order to get most recent messages first
            for msg in reversed(conversation_history):
                if message_count >= max_history_messages:
                    break
                    
                role = msg.get("role", "").lower()
                content = msg.get("content", "")
                
                # Convert role names to OpenAI format
                if role in ["user", "human"]:
                    openai_role = "user"
                elif role in ["assistant", "ai"]:
                    openai_role = "assistant"
                elif role == "system":
                    openai_role = "system"
                else:
                    continue  # Skip unknown roles
                
                # Estimate token count (rough approximation: 1 token ≈ 4 characters)
                estimated_tokens = len(content) // 4
                if token_count + estimated_tokens > max_history_tokens:
                    break
                
                history_messages.append({
                    "role": openai_role,
                    "content": content
                })
                token_count += estimated_tokens
                message_count += 1
            
            # Add messages in chronological order
            messages.extend(reversed(history_messages))
        
        # Add current user query
        messages.append({
            "role": "user",
            "content": user_query
        })
        
        # Prepare API call parameters
        api_params = {
            "model": self.model,
            "messages": messages,
            "stream": True
        }
        
        # Only add tools if available
        if functions:
            api_params["tools"] = functions
            api_params["tool_choice"] = tool_choice

        try:
            # First API call to GPT-4 with streaming
            response_stream = self.client.chat.completions.create(**api_params)
            
            content_buffer = ""
            tool_calls_detected = False
            
            # Process streaming response
            for chunk in response_stream:
                if chunk.choices and len(chunk.choices) > 0:
                    choice = chunk.choices[0]
                    delta = choice.delta
                    
                    # Handle content chunks
                    if delta.content:
                        content_buffer += delta.content
                        yield {
                            "type": "response_chunk",
                            "content": delta.content,
                            "complete": False
                        }
                    
                    # Handle tool calls
                    if delta.tool_calls:
                        tool_calls_detected = True
                        for tool_call in delta.tool_calls:
                            if tool_call.function and tool_call.function.name:
                                yield {
                                    "type": "response_chunk",
                                    "content": f"[Using tool: {tool_call.function.name}]",
                                    "complete": False
                                }
                    
                    # Check if response is finished
                    if choice.finish_reason == "stop":
                        yield {
                            "type": "response_complete",
                            "content": content_buffer,
                            "complete": True
                        }
                        break
                    elif choice.finish_reason == "tool_calls":
                        # Handle tool calls with TRUE STREAMING like Cursor
                        yield {
                            "type": "response_chunk",
                            "content": "\n[Processing tool results...]",
                            "complete": False
                        }
                        
                        # Stream the tool execution and final response
                        async for final_chunk in self._stream_tool_execution(
                            user_query, available_tools, system_prompt,
                            conversation_history, max_history_messages, max_history_tokens
                        ):
                            yield final_chunk
                        break
                    
        except Exception as e:
            yield {
                "type": "error",
                "content": f"Error with OpenAI API: {str(e)}",
                "complete": True
            }
    
    async def _stream_tool_execution(
        self, 
        user_query: str, 
        available_tools: List[Any], 
        system_prompt: str,
        conversation_history: List[Dict[str, Any]] = None,
        max_history_messages: int = 25,
        max_history_tokens: int = 6000
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Execute tools and stream the final response like Cursor does
        """
        try:
            # Step 1: Execute the tool search (non-streaming but fast)
            search_result = ""
            if available_tools:
                search_tool = available_tools[0]
                # Extract search query from the user query (simplified approach)
                search_result = search_tool.search(user_query)
                
                yield {
                    "type": "response_chunk",
                    "content": "\n[Search completed, generating response...]",
                    "complete": False
                }
            
            # Step 2: Build conversation with tool results and history
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                }
            ]
            
            # Add conversation history if provided
            if conversation_history:
                history_messages = []
                token_count = 0
                message_count = 0
                
                # Process history in reverse order to get most recent messages first
                for msg in reversed(conversation_history):
                    if message_count >= max_history_messages:
                        break
                        
                    role = msg.get("role", "").lower()
                    content = msg.get("content", "")
                    
                    # Convert role names to OpenAI format
                    if role in ["user", "human"]:
                        openai_role = "user"
                    elif role in ["assistant", "ai"]:
                        openai_role = "assistant"
                    elif role == "system":
                        openai_role = "system"
                    else:
                        continue  # Skip unknown roles
                    
                    # Estimate token count (rough approximation: 1 token ≈ 4 characters)
                    estimated_tokens = len(content) // 4
                    if token_count + estimated_tokens > max_history_tokens:
                        break
                    
                    history_messages.append({
                        "role": openai_role,
                        "content": content
                    })
                    token_count += estimated_tokens
                    message_count += 1
                
                # Add messages in chronological order
                messages.extend(reversed(history_messages))
            
            # Add current interaction with tool results
            messages.extend([
                {
                    "role": "user", 
                    "content": user_query
                },
                {
                    "role": "assistant",
                    "content": f"I'll search for information about this. Let me analyze the search results:\n\n{search_result}"
                },
                {
                    "role": "user",
                    "content": "Based on the search results above, please provide a comprehensive answer to my original question."
                }
            ])
            
            # Step 3: Stream the final response using OpenAI
            final_response_params = {
                "model": self.model,
                "messages": messages,
                "stream": True,
                "temperature": 0.7
            }
            
            response_stream = self.client.chat.completions.create(**final_response_params)
            
            # Step 4: Stream the final response in real-time
            content_buffer = ""
            for chunk in response_stream:
                if chunk.choices and len(chunk.choices) > 0:
                    choice = chunk.choices[0]
                    delta = choice.delta
                    
                    if delta.content:
                        content_buffer += delta.content
                        yield {
                            "type": "response_chunk",
                            "content": delta.content,
                            "complete": False
                        }
                    
                    if choice.finish_reason == "stop":
                        yield {
                            "type": "response_complete",
                            "content": content_buffer,
                            "complete": True
                        }
                        break
                        
        except Exception as e:
            yield {
                "type": "error",
                "content": f"Error in tool execution streaming: {str(e)}",
                "complete": True
            }
    
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