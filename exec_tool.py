"""
Executive Tool Module - API-ready wrapper for LLM tool execution
Provides dynamic parameter support and customizable system prompts for API endpoints
"""

import os
import json
import traceback
from typing import Dict, List, Any, Optional, Union, AsyncGenerator
from datetime import datetime
from dataclasses import dataclass

from anthropic_tool import AnthropicTool
from openai_tool import OpenAITool
from search_tool import SearchTool


@dataclass
class ToolExecutionRequest:
    """Request model for tool execution"""
    llm_provider: str  # 'anthropic' or 'openai'
    user_query: str
    system_prompt: Optional[str] = None
    model: Optional[str] = None  # Custom model name (e.g., "gpt-4o", "claude-3-5-haiku")
    model_params: Optional[Dict[str, Any]] = None
    tools_config: Optional[Dict[str, Any]] = None
    org_id: Optional[str] = "default"
    user_id: Optional[str] = "anonymous"
    # New parameters to control tool usage
    enable_tools: Optional[bool] = True  # Whether to enable tools at all
    force_tools: Optional[bool] = False  # Force tool usage (tool_choice="required")
    tools_whitelist: Optional[List[str]] = None  # Only allow specific tools
    # Conversation history support
    conversation_history: Optional[List[Dict[str, Any]]] = None  # Previous messages
    max_history_messages: Optional[int] = 25  # Maximum number of history messages to include
    max_history_tokens: Optional[int] = 6000  # Maximum token count for history


@dataclass
class ToolExecutionResponse:
    """Response model for tool execution"""
    success: bool
    result: str
    provider: str
    model_used: str
    execution_time: float
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ExecutiveTool:
    """Executive tool handler for API-ready LLM tool execution"""
    
    def __init__(self):
        """Initialize the executive tool handler"""
        self.available_tools = self._initialize_tools()
        self.default_system_prompts = {
            "anthropic": "You are a helpful assistant. Provide accurate, concise, and well-structured responses based on your knowledge.",
            "openai": "You are a helpful assistant. Provide accurate, concise, and well-structured responses based on your knowledge.",
            "anthropic_with_tools": "You are a helpful assistant that can search for information when needed. Provide accurate, concise, and well-structured responses.",
            "openai_with_tools": "You are a helpful assistant that can search for information when needed. Provide accurate, concise, and well-structured responses."
        }
    
    def _initialize_tools(self) -> Dict[str, Any]:
        """Initialize available tools"""
        tools = {}
        try:
            tools["search"] = SearchTool()
        except Exception as e:
            print(f"Warning: Could not initialize SearchTool: {e}")
        return tools
    
    async def execute_tool_async(self, request: ToolExecutionRequest) -> ToolExecutionResponse:
        """
        Asynchronously execute tool with the specified LLM provider
        
        Args:
            request: ToolExecutionRequest containing execution parameters
            
        Returns:
            ToolExecutionResponse with execution results
        """
        start_time = datetime.now()
        
        try:
            # Validate provider
            if request.llm_provider.lower() not in ["anthropic", "openai"]:
                raise ValueError(f"Unsupported LLM provider: {request.llm_provider}")
            
            # Execute tool
            if request.llm_provider.lower() == "anthropic":
                result = await self._execute_anthropic(request)
            else:
                result = await self._execute_openai(request)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return ToolExecutionResponse(
                success=True,
                result=result,
                provider=request.llm_provider,
                model_used=self._get_model_name(request.llm_provider, request.model),
                execution_time=execution_time,
                metadata={
                    "org_id": request.org_id,
                    "user_id": request.user_id,
                    "tools_used": list(self.available_tools.keys())
                }
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"Tool execution failed: {str(e)}"
            traceback.print_exc()
            
            return ToolExecutionResponse(
                success=False,
                result="",
                provider=request.llm_provider,
                model_used=self._get_model_name(request.llm_provider, request.model),
                execution_time=execution_time,
                error=error_msg
            )
    
    def execute_tool_sync(self, request: ToolExecutionRequest) -> ToolExecutionResponse:
        """
        Synchronously execute tool with the specified LLM provider
        
        Args:
            request: ToolExecutionRequest containing execution parameters
            
        Returns:
            ToolExecutionResponse with execution results
        """
        start_time = datetime.now()
        
        try:
            # Validate provider
            if request.llm_provider.lower() not in ["anthropic", "openai"]:
                raise ValueError(f"Unsupported LLM provider: {request.llm_provider}")
            
            # Execute tool
            if request.llm_provider.lower() == "anthropic":
                result = self._execute_anthropic_sync(request)
            else:
                result = self._execute_openai_sync(request)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return ToolExecutionResponse(
                success=True,
                result=result,
                provider=request.llm_provider,
                model_used=self._get_model_name(request.llm_provider, request.model),
                execution_time=execution_time,
                metadata={
                    "org_id": request.org_id,
                    "user_id": request.user_id,
                    "tools_used": list(self.available_tools.keys())
                }
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"Tool execution failed: {str(e)}"
            traceback.print_exc()
            
            return ToolExecutionResponse(
                success=False,
                result="",
                provider=request.llm_provider,
                model_used=self._get_model_name(request.llm_provider, request.model),
                execution_time=execution_time,
                error=error_msg
            )
    
    async def execute_tool_stream(self, request: ToolExecutionRequest) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream tool execution with the specified LLM provider using SSE format
        
        Args:
            request: ToolExecutionRequest containing execution parameters
            
        Yields:
            Dict containing streaming response data
        """
        start_time = datetime.now()
        
        try:
            # Validate provider
            if request.llm_provider.lower() not in ["anthropic", "openai"]:
                yield {
                    "type": "error",
                    "content": f"Unsupported LLM provider: {request.llm_provider}",
                    "provider": request.llm_provider,
                    "success": False
                }
                return
            
            # Yield initial processing status
            yield {
                "type": "status",
                "content": "Starting LLM tool execution...",
                "provider": request.llm_provider,
                "status": "processing"
            }
            
            # Execute tool with streaming
            if request.llm_provider.lower() == "anthropic":
                async for chunk in self._execute_anthropic_stream(request):
                    yield chunk
            else:
                async for chunk in self._execute_openai_stream(request):
                    yield chunk
            
            # Yield completion status
            execution_time = (datetime.now() - start_time).total_seconds()
            yield {
                "type": "complete",
                "content": "Tool execution completed successfully",
                "provider": request.llm_provider,
                "model_used": self._get_model_name(request.llm_provider, request.model),
                "execution_time": execution_time,
                "success": True,
                "metadata": {
                    "org_id": request.org_id,
                    "user_id": request.user_id,
                    "tools_used": list(self.available_tools.keys())
                }
            }
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"Tool execution failed: {str(e)}"
            traceback.print_exc()
            
            yield {
                "type": "error",
                "content": error_msg,
                "provider": request.llm_provider,
                "model_used": self._get_model_name(request.llm_provider, request.model),
                "execution_time": execution_time,
                "success": False
            }
    
    async def _execute_anthropic(self, request: ToolExecutionRequest) -> str:
        """Execute using Anthropic Claude with custom parameters"""
        # Use custom model if provided, otherwise use default
        model = request.model or self._get_default_model("anthropic")
        anthropic_tool = AnthropicTool(model=model)
        
        # Override system prompt if provided (Anthropic handles system prompts differently)
        if request.system_prompt:
            # For Anthropic, we'll prepend the system prompt to the user query
            enhanced_query = f"System: {request.system_prompt}\n\nUser: {request.user_query}"
        else:
            enhanced_query = request.user_query
        
        # Get available tools for execution
        tools_to_use = []
        if "search" in self.available_tools:
            tools_to_use.append(self.available_tools["search"])
        
        return anthropic_tool.process_with_tools(enhanced_query, tools_to_use)
    
    def _execute_anthropic_sync(self, request: ToolExecutionRequest) -> str:
        """Synchronously execute using Anthropic Claude"""
        # Use custom model if provided, otherwise use default
        model = request.model or self._get_default_model("anthropic")
        anthropic_tool = AnthropicTool(model=model)
        
        # Override system prompt if provided
        if request.system_prompt:
            enhanced_query = f"System: {request.system_prompt}\n\nUser: {request.user_query}"
        else:
            enhanced_query = request.user_query
        
        # Get available tools for execution
        tools_to_use = []
        if "search" in self.available_tools:
            tools_to_use.append(self.available_tools["search"])
        
        return anthropic_tool.process_with_tools(enhanced_query, tools_to_use)
    
    async def _execute_openai(self, request: ToolExecutionRequest) -> str:
        """Execute using OpenAI with custom system prompt"""
        # Use custom model if provided, otherwise use default
        model = request.model or self._get_default_model("openai")
        openai_tool = OpenAIToolWithCustomPrompt(model=model)
        
        # Set custom system prompt if provided
        system_prompt = request.system_prompt or self.default_system_prompts["openai"]
        
        # Get available tools for execution
        tools_to_use = []
        if "search" in self.available_tools:
            tools_to_use.append(self.available_tools["search"])
        
        return openai_tool.process_with_tools_and_prompt(
            request.user_query, 
            tools_to_use, 
            system_prompt,
            request.model_params
        )
    
    def _execute_openai_sync(self, request: ToolExecutionRequest) -> str:
        """Synchronously execute using OpenAI with custom system prompt"""
        # Use custom model if provided, otherwise use default
        model = request.model or self._get_default_model("openai")
        openai_tool = OpenAIToolWithCustomPrompt(model=model)
        
        # Set custom system prompt if provided
        system_prompt = request.system_prompt or self.default_system_prompts["openai"]
        
        # Get available tools for execution
        tools_to_use = []
        if "search" in self.available_tools:
            tools_to_use.append(self.available_tools["search"])
        
        return openai_tool.process_with_tools_and_prompt(
            request.user_query, 
            tools_to_use, 
            system_prompt,
            request.model_params
        )
    
    async def _execute_anthropic_stream(self, request: ToolExecutionRequest) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute using Anthropic Claude with streaming"""
        # Use custom model if provided, otherwise use default
        model = request.model or self._get_default_model("anthropic")
        anthropic_tool = AnthropicTool(model=model)
        
        # Get available tools for execution based on configuration
        tools_to_use = []
        if request.enable_tools and "search" in self.available_tools:
            # Check whitelist if provided
            if request.tools_whitelist is None or "search" in request.tools_whitelist:
                tools_to_use.append(self.available_tools["search"])
        
        # Set custom system prompt if provided, otherwise use appropriate default
        if request.system_prompt:
            system_prompt = request.system_prompt
        else:
            # Use tool-aware prompt if tools are enabled, otherwise use regular prompt
            if tools_to_use:
                system_prompt = self.default_system_prompts["anthropic_with_tools"]
            else:
                system_prompt = self.default_system_prompts["anthropic"]
        
        try:
            # Use the new streaming method from AnthropicTool with system prompt and tool config
            async for chunk in anthropic_tool.process_with_tools_stream(
                request.user_query, 
                tools_to_use, 
                system_prompt,
                force_tools=request.force_tools
            ):
                yield chunk
            
        except Exception as e:
            yield {
                "type": "error",
                "content": f"Anthropic execution error: {str(e)}",
                "complete": True
            }
    
    async def _execute_openai_stream(self, request: ToolExecutionRequest) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute using OpenAI with streaming"""
        # Use custom model if provided, otherwise use default
        model = request.model or self._get_default_model("openai")
        openai_tool = OpenAITool(model=model)
        
        # Get available tools for execution based on configuration
        tools_to_use = []
        if request.enable_tools and "search" in self.available_tools:
            # Check whitelist if provided
            if request.tools_whitelist is None or "search" in request.tools_whitelist:
                tools_to_use.append(self.available_tools["search"])
        
        # Set custom system prompt if provided, otherwise use appropriate default
        if request.system_prompt:
            system_prompt = request.system_prompt
        else:
            # Use tool-aware prompt if tools are enabled, otherwise use regular prompt
            if tools_to_use:
                system_prompt = self.default_system_prompts["openai_with_tools"]
            else:
                system_prompt = self.default_system_prompts["openai"]
        
        try:
            # Use the new streaming method from OpenAITool with system prompt and tool config
            async for chunk in openai_tool.process_with_tools_stream(
                request.user_query, 
                tools_to_use, 
                system_prompt,
                force_tools=request.force_tools,
                conversation_history=request.conversation_history,
                max_history_messages=request.max_history_messages,
                max_history_tokens=request.max_history_tokens
            ):
                yield chunk
            
        except Exception as e:
            yield {
                "type": "error",
                "content": f"OpenAI execution error: {str(e)}",
                "complete": True
            }
    
    def _get_model_name(self, provider: str, custom_model: str = None) -> str:
        """Get the model name for the specified provider"""
        if custom_model:
            return custom_model
        
        if provider.lower() == "anthropic":
            return "claude-3-5-sonnet-20241022"
        elif provider.lower() == "openai":
            return "gpt-4-1106-preview"
        return "unknown"
    
    def _get_default_model(self, provider: str) -> str:
        """Get the default model for the specified provider"""
        if provider.lower() == "anthropic":
            return "claude-3-5-sonnet-20241022"
        elif provider.lower() == "openai":
            return "gpt-4-1106-preview"
        return "unknown"
    
    def get_available_providers(self) -> List[str]:
        """Get list of available LLM providers"""
        return ["anthropic", "openai"]
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tools"""
        return list(self.available_tools.keys())


class OpenAIToolWithCustomPrompt(OpenAITool):
    """Extended OpenAI tool that supports custom system prompts"""
    
    def process_with_tools_and_prompt(
        self, 
        user_query: str, 
        available_tools: List[Any], 
        system_prompt: str,
        model_params: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Process user query with available tools using custom system prompt
        
        Args:
            user_query: The user's input query
            available_tools: List of available tool instances
            system_prompt: Custom system prompt
            model_params: Optional model parameters (temperature, max_tokens, etc.)
            
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
        
        # Prepare model parameters
        call_params = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_query
                }
            ],
            "tools": functions,
            "tool_choice": "auto"
        }
        
        # Add optional parameters
        if model_params:
            if "temperature" in model_params:
                call_params["temperature"] = model_params["temperature"]
            if "max_tokens" in model_params:
                call_params["max_tokens"] = model_params["max_tokens"]
            if "top_p" in model_params:
                call_params["top_p"] = model_params["top_p"]
        
        try:
            # First API call to GPT-4
            response = self.client.chat.completions.create(**call_params)
            
            message = response.choices[0].message
            
            # Check if GPT-4 wants to use tools
            if message.tool_calls:
                return self._handle_tool_calls_with_prompt(
                    message, available_tools, user_query, system_prompt
                )
            else:
                return message.content
                
        except Exception as e:
            return f"Error with OpenAI API: {str(e)}"
    
    def _handle_tool_calls_with_prompt(
        self, 
        message: Any, 
        available_tools: List[Any], 
        original_query: str,
        system_prompt: str
    ) -> str:
        """Handle tool calls with custom system prompt"""
        
        messages = [
            {
                "role": "system",
                "content": system_prompt
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
        
        # Send tool results back to GPT-4
        try:
            final_response = self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )
            
            return final_response.choices[0].message.content
            
        except Exception as e:
            return f"Error processing tool results: {str(e)}"


# Convenience functions for easy API integration
def create_tool_request(
    llm_provider: str,
    user_query: str,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    model_params: Optional[Dict[str, Any]] = None,
    org_id: str = "default",
    user_id: str = "anonymous",
    enable_tools: bool = True,
    force_tools: bool = False,
    tools_whitelist: Optional[List[str]] = None,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
    max_history_messages: Optional[int] = 25,
    max_history_tokens: Optional[int] = 6000
) -> ToolExecutionRequest:
    """
    Create a tool execution request
    
    Args:
        llm_provider: 'anthropic' or 'openai'
        user_query: User's input query
        system_prompt: Optional custom system prompt
        model: Optional custom model name (e.g., "gpt-4o", "claude-3-5-haiku")
        model_params: Optional model parameters (temperature, max_tokens, etc.)
        org_id: Organization ID
        user_id: User ID
        enable_tools: Whether to enable tools at all
        force_tools: Force tool usage (tool_choice="required")
        tools_whitelist: Only allow specific tools
        conversation_history: Previous conversation messages
        max_history_messages: Maximum number of history messages to include
        max_history_tokens: Maximum token count for history
        
    Returns:
        ToolExecutionRequest object
    """
    return ToolExecutionRequest(
        llm_provider=llm_provider,
        user_query=user_query,
        system_prompt=system_prompt,
        model=model,
        model_params=model_params,
        org_id=org_id,
        user_id=user_id,
        enable_tools=enable_tools,
        force_tools=force_tools,
        tools_whitelist=tools_whitelist,
        conversation_history=conversation_history,
        max_history_messages=max_history_messages,
        max_history_tokens=max_history_tokens
    )


async def execute_tool_async(
    llm_provider: str,
    user_query: str,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    model_params: Optional[Dict[str, Any]] = None,
    org_id: str = "default",
    user_id: str = "anonymous",
    enable_tools: bool = True,
    force_tools: bool = False,
    tools_whitelist: Optional[List[str]] = None
) -> ToolExecutionResponse:
    """
    Asynchronously execute tool with specified parameters
    
    Args:
        llm_provider: 'anthropic' or 'openai'
        user_query: User's input query
        system_prompt: Optional custom system prompt
        model: Optional custom model name (e.g., "gpt-4o", "claude-3-5-haiku")
        model_params: Optional model parameters
        org_id: Organization ID
        user_id: User ID
        enable_tools: Whether to enable tools at all
        force_tools: Force tool usage (tool_choice="required")
        tools_whitelist: Only allow specific tools
        
    Returns:
        ToolExecutionResponse with results
    """
    executive_tool = ExecutiveTool()
    request = create_tool_request(
        llm_provider, user_query, system_prompt, model, model_params, org_id, user_id,
        enable_tools, force_tools, tools_whitelist
    )
    return await executive_tool.execute_tool_async(request)


def execute_tool_sync(
    llm_provider: str,
    user_query: str,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    model_params: Optional[Dict[str, Any]] = None,
    org_id: str = "default",
    user_id: str = "anonymous",
    enable_tools: bool = True,
    force_tools: bool = False,
    tools_whitelist: Optional[List[str]] = None
) -> ToolExecutionResponse:
    """
    Synchronously execute tool with specified parameters
    
    Args:
        llm_provider: 'anthropic' or 'openai'
        user_query: User's input query
        system_prompt: Optional custom system prompt
        model: Optional custom model name (e.g., "gpt-4o", "claude-3-5-haiku")
        model_params: Optional model parameters
        org_id: Organization ID
        user_id: User ID
        enable_tools: Whether to enable tools at all
        force_tools: Force tool usage (tool_choice="required")
        tools_whitelist: Only allow specific tools
        
    Returns:
        ToolExecutionResponse with results
    """
    executive_tool = ExecutiveTool()
    request = create_tool_request(
        llm_provider, user_query, system_prompt, model, model_params, org_id, user_id, 
        enable_tools, force_tools, tools_whitelist
    )
    return executive_tool.execute_tool_sync(request) 

# Add convenience function for streaming
async def execute_tool_stream(
    llm_provider: str,
    user_query: str,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    model_params: Optional[Dict[str, Any]] = None,
    org_id: str = "default",
    user_id: str = "anonymous",
    enable_tools: bool = True,
    force_tools: bool = False,
    tools_whitelist: Optional[List[str]] = None,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
    max_history_messages: Optional[int] = 25,
    max_history_tokens: Optional[int] = 6000
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Stream tool execution with specified parameters
    
    Args:
        llm_provider: 'anthropic' or 'openai'
        user_query: User's input query
        system_prompt: Optional custom system prompt
        model: Optional custom model name (e.g., "gpt-4o", "claude-3-5-haiku")
        model_params: Optional model parameters
        org_id: Organization ID
        user_id: User ID
        enable_tools: Whether to enable tools at all
        force_tools: Force tool usage (tool_choice="required")
        tools_whitelist: Only allow specific tools
        conversation_history: Previous conversation messages
        max_history_messages: Maximum number of history messages to include
        max_history_tokens: Maximum token count for history
        
    Yields:
        Dict containing streaming response data
    """
    executive_tool = ExecutiveTool()
    request = ToolExecutionRequest(
        llm_provider=llm_provider,
        user_query=user_query,
        system_prompt=system_prompt,
        model=model,
        model_params=model_params,
        org_id=org_id,
        user_id=user_id,
        enable_tools=enable_tools,
        force_tools=force_tools,
        tools_whitelist=tools_whitelist,
        conversation_history=conversation_history,
        max_history_messages=max_history_messages,
        max_history_tokens=max_history_tokens
    )
    async for chunk in executive_tool.execute_tool_stream(request):
        yield chunk 