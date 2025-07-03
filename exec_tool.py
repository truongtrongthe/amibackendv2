"""
Executive Tool Module - API-ready wrapper for LLM tool execution
Provides dynamic parameter support and customizable system prompts for API endpoints
"""

import os
import json
import traceback
from typing import Dict, List, Any, Optional, Union
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
    model_params: Optional[Dict[str, Any]] = None
    tools_config: Optional[Dict[str, Any]] = None
    org_id: Optional[str] = "default"
    user_id: Optional[str] = "anonymous"


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
            "anthropic": "You are a helpful assistant that can search for information when needed. Provide accurate, concise, and well-structured responses.",
            "openai": "You are a helpful assistant that can search for information when needed. Provide accurate, concise, and well-structured responses."
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
                model_used=self._get_model_name(request.llm_provider),
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
                model_used=self._get_model_name(request.llm_provider),
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
                model_used=self._get_model_name(request.llm_provider),
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
                model_used=self._get_model_name(request.llm_provider),
                execution_time=execution_time,
                error=error_msg
            )
    
    async def _execute_anthropic(self, request: ToolExecutionRequest) -> str:
        """Execute using Anthropic Claude with custom parameters"""
        anthropic_tool = AnthropicTool()
        
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
        anthropic_tool = AnthropicTool()
        
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
        openai_tool = OpenAIToolWithCustomPrompt()
        
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
        openai_tool = OpenAIToolWithCustomPrompt()
        
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
    
    def _get_model_name(self, provider: str) -> str:
        """Get the model name for the specified provider"""
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
    model_params: Optional[Dict[str, Any]] = None,
    org_id: str = "default",
    user_id: str = "anonymous"
) -> ToolExecutionRequest:
    """
    Create a tool execution request
    
    Args:
        llm_provider: 'anthropic' or 'openai'
        user_query: User's input query
        system_prompt: Optional custom system prompt
        model_params: Optional model parameters (temperature, max_tokens, etc.)
        org_id: Organization ID
        user_id: User ID
        
    Returns:
        ToolExecutionRequest object
    """
    return ToolExecutionRequest(
        llm_provider=llm_provider,
        user_query=user_query,
        system_prompt=system_prompt,
        model_params=model_params,
        org_id=org_id,
        user_id=user_id
    )


async def execute_tool_async(
    llm_provider: str,
    user_query: str,
    system_prompt: Optional[str] = None,
    model_params: Optional[Dict[str, Any]] = None,
    org_id: str = "default",
    user_id: str = "anonymous"
) -> ToolExecutionResponse:
    """
    Asynchronously execute tool with specified parameters
    
    Args:
        llm_provider: 'anthropic' or 'openai'
        user_query: User's input query
        system_prompt: Optional custom system prompt
        model_params: Optional model parameters
        org_id: Organization ID
        user_id: User ID
        
    Returns:
        ToolExecutionResponse with results
    """
    executive_tool = ExecutiveTool()
    request = create_tool_request(
        llm_provider, user_query, system_prompt, model_params, org_id, user_id
    )
    return await executive_tool.execute_tool_async(request)


def execute_tool_sync(
    llm_provider: str,
    user_query: str,
    system_prompt: Optional[str] = None,
    model_params: Optional[Dict[str, Any]] = None,
    org_id: str = "default",
    user_id: str = "anonymous"
) -> ToolExecutionResponse:
    """
    Synchronously execute tool with specified parameters
    
    Args:
        llm_provider: 'anthropic' or 'openai'
        user_query: User's input query
        system_prompt: Optional custom system prompt
        model_params: Optional model parameters
        org_id: Organization ID
        user_id: User ID
        
    Returns:
        ToolExecutionResponse with results
    """
    executive_tool = ExecutiveTool()
    request = create_tool_request(
        llm_provider, user_query, system_prompt, model_params, org_id, user_id
    )
    return executive_tool.execute_tool_sync(request) 