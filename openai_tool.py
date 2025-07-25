"""
OpenAI GPT-4.1 LLM implementation with tool calling
"""

import os
import json
import logging
from typing import List, Any, Dict, AsyncGenerator
from openai import OpenAI
from datetime import datetime

logger = logging.getLogger(__name__)

# Configure detailed logging for tool calls
tool_logger = logging.getLogger("tool_calls")
tool_logger.setLevel(logging.INFO)
if not tool_logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('🔧 [OPENAI_TOOL] %(asctime)s - %(message)s', datefmt='%H:%M:%S')
    handler.setFormatter(formatter)
    tool_logger.addHandler(handler)

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
            available_tools: List of available tool instances (search, context, etc.)
            
        Returns:
            Response from GPT-4 with tool execution results
        """
        
        # Define functions for OpenAI only if tools are available
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
                tool_choice="auto",
                max_tokens=4000  # Use reasonable default that works with most models
            )
            
            message = response.choices[0].message
            
            # Check if GPT-4 wants to use tools
            if message.tool_calls:
                return self._handle_tool_calls(message, available_tools, user_query)
            else:
                return message.content
                
        except Exception as e:
            return f"Error with OpenAI API: {str(e)}"
    
    async def process_with_tools_stream(self, user_query: str, available_tools: List[Any], system_prompt: str = None, force_tools: bool = False, conversation_history: List[Dict[str, Any]] = None, max_history_messages: int = 25, max_history_tokens: int = 6000, model_params: Dict[str, Any] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process user query with available tools using GPT-4 with streaming
        
        Args:
            user_query: The user's input query
            available_tools: List of available tool instances (search, context, etc.)
            system_prompt: Optional custom system prompt
            force_tools: Whether to force tool usage (tool_choice="required")
            conversation_history: Previous conversation messages
            max_history_messages: Maximum number of history messages to include
            max_history_tokens: Maximum token count for history
            model_params: Optional model parameters (temperature, max_tokens, etc.)
            
        Yields:
            Dict containing streaming response data
        """
        
        # Use custom system prompt or default
        if system_prompt is None:
            system_prompt = "You are a helpful assistant that can search for information and retrieve relevant context when needed."
        
        # Define functions for OpenAI only if tools are available
        functions = []
        
        if available_tools:
            for tool in available_tools:
                # Check if tool has get_tool_description method (dynamic tool definitions)
                if hasattr(tool, 'get_tool_description'):
                    try:
                        tool_defs = tool.get_tool_description()
                        # Handle both single tool definition and list of tool definitions
                        if isinstance(tool_defs, list):
                            for tool_def in tool_defs:
                                functions.append({
                                    "type": "function",
                                    "function": tool_def
                                })
                        else:
                            # Single tool definition
                            functions.append({
                                "type": "function",
                                "function": tool_defs
                            })
                        print(f"DEBUG: Added {len(tool_defs) if isinstance(tool_defs, list) else 1} functions from {type(tool).__name__}")
                    except Exception as e:
                        print(f"Warning: Could not get tool description for {tool}: {e}")
                        continue
                # Legacy hardcoded tool definitions for backward compatibility
                elif hasattr(tool, 'search'):  # Search tool
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
                elif hasattr(tool, 'get_context'):  # Context tool
                    functions.append({
                        "type": "function",
                        "function": {
                            "name": "get_context",
                            "description": "Retrieve relevant context including user profile, system status, organization info, and knowledge base information",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "query": {
                                        "type": "string",
                                        "description": "The topic or question to get context for"
                                    },
                                    "source_types": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "Specific context sources to query: user_profile, system_status, organization_info, knowledge_base, recent_activity",
                                        "default": None
                                    }
                                },
                                "required": ["query"]
                            }
                        }
                    })
                elif hasattr(tool, 'search_learning_context'):  # Learning search tool
                    functions.append({
                        "type": "function",
                        "function": {
                            "name": "search_learning_context",
                            "description": "Search existing knowledge base for similar content to avoid duplicates",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "query": {
                                        "type": "string",
                                        "description": "The search query to find existing knowledge"
                                    },
                                    "depth": {
                                        "type": "string",
                                        "description": "Search depth: 'basic' for simple search, 'comprehensive' for detailed search",
                                        "enum": ["basic", "comprehensive"]
                                    }
                                },
                                "required": ["query"]
                            }
                        }
                    })
                elif hasattr(tool, 'analyze_learning_opportunity'):  # Learning analysis tool
                    functions.append({
                        "type": "function",
                        "function": {
                            "name": "analyze_learning_opportunity",
                            "description": "Analyze if a message contains valuable information that should be learned",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "user_message": {
                                        "type": "string",
                                        "description": "The user message to analyze for learning opportunities"
                                    }
                                },
                                "required": ["user_message"]
                            }
                        }
                    })
                elif hasattr(tool, 'request_learning_decision'):  # Human learning decision tool
                    functions.append({
                        "type": "function",
                        "function": {
                            "name": "request_learning_decision",
                            "description": "Request human decision on whether to learn specific information",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "information": {
                                        "type": "string",
                                        "description": "The information to potentially learn"
                                    },
                                    "reason": {
                                        "type": "string",
                                        "description": "Why this information should be learned"
                                    },
                                    "category": {
                                        "type": "string",
                                        "description": "Category of information (e.g., 'company_info', 'procedure', 'personal_data')"
                                    }
                                },
                                "required": ["information", "reason"]
                            }
                        }
                    })
                elif hasattr(tool, 'preview_knowledge_save'):  # Knowledge preview tool
                    functions.append({
                        "type": "function",
                        "function": {
                            "name": "preview_knowledge_save",
                            "description": "Preview what knowledge would be saved before actual saving",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "content": {
                                        "type": "string",
                                        "description": "The content to preview for saving"
                                    },
                                    "title": {
                                        "type": "string",
                                        "description": "Title for the knowledge item"
                                    }
                                },
                                "required": ["content"]
                            }
                        }
                    })
                elif hasattr(tool, 'save_knowledge'):  # Knowledge save tool
                    functions.append({
                        "type": "function",
                        "function": {
                            "name": "save_knowledge",
                            "description": "Save approved knowledge to the knowledge base",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "content": {
                                        "type": "string",
                                        "description": "The content to save"
                                    },
                                    "title": {
                                        "type": "string",
                                        "description": "Title for the knowledge item"
                                    },
                                    "category": {
                                        "type": "string",
                                        "description": "Category of knowledge"
                                    }
                                },
                                "required": ["content"]
                            }
                        }
                    })
        
        # Set tool choice based on force_tools parameter
        tool_choice = "auto"
        if force_tools:
            tool_choice = "required"
        else:
            tool_choice = "auto"

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
        
        # Add model parameters if provided
        if model_params:
            if "temperature" in model_params:
                api_params["temperature"] = model_params["temperature"]
            if "max_tokens" in model_params:
                api_params["max_tokens"] = model_params["max_tokens"]
            if "top_p" in model_params:
                api_params["top_p"] = model_params["top_p"]
        else:
            # Use reasonable defaults
            api_params["max_tokens"] = 4000
        
        # Only add tools if available
        if functions:
            api_params["tools"] = functions
            api_params["tool_choice"] = tool_choice
            print(f"DEBUG: API call with {len(functions)} functions, tool_choice={tool_choice}, force_tools={force_tools}")
        else:
            print(f"DEBUG: No functions available for API call")


        try:
            # First API call to GPT-4 with streaming
            response_stream = self.client.chat.completions.create(**api_params)
            
            content_buffer = ""
            tool_calls_detected = False
            collected_tool_calls = []  # Collect tool calls from streaming
            current_tool_calls = {}  # Track current tool calls being built
            
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
                            if tool_call.function:
                                # Get or create tool call ID
                                tool_call_id = tool_call.index if hasattr(tool_call, 'index') else len(current_tool_calls)
                                
                                # Initialize tool call if not exists
                                if tool_call_id not in current_tool_calls:
                                    current_tool_calls[tool_call_id] = {
                                        'id': tool_call.id if hasattr(tool_call, 'id') else f"call_{tool_call_id}",
                                        'function': {'name': '', 'arguments': ''}
                                    }
                                
                                # Update tool call with streaming data
                                if tool_call.function.name:
                                    current_tool_calls[tool_call_id]['function']['name'] = tool_call.function.name
                                    print(f"DEBUG: Tool call {tool_call_id} name: {tool_call.function.name}")
                                
                                if tool_call.function.arguments:
                                    current_tool_calls[tool_call_id]['function']['arguments'] += tool_call.function.arguments
                                    print(f"DEBUG: Tool call {tool_call_id} arguments chunk: '{tool_call.function.arguments}'")
                                
                                # Yield tool usage message
                                if tool_call.function.name:
                                    yield {
                                        "type": "response_chunk",
                                        "content": f"[Using tool: {tool_call.function.name}]",
                                        "complete": False
                                    }
                    
                    # Check if response is finished
                    print(f"DEBUG: finish_reason = {choice.finish_reason}")
                    if choice.finish_reason == "stop":
                        yield {
                            "type": "response_complete",
                            "content": content_buffer,
                            "complete": True
                        }
                        break
                    elif choice.finish_reason == "tool_calls":
                        print(f"DEBUG: ENTERING tool_calls branch")
                        try:
                            # Handle tool calls with TRUE STREAMING like Cursor
                            yield {
                                "type": "response_chunk",
                                "content": "\n[Processing tool results...]",
                                "complete": False
                            }
                            
                            # Use the collected tool calls from streaming
                            print(f"DEBUG: Processing {len(current_tool_calls)} collected tool calls")
                            print(f"DEBUG: Current tool calls: {current_tool_calls}")
                            
                            if not current_tool_calls:
                                print(f"DEBUG: No tool calls were collected during streaming")
                                break
                            
                            # Handle the actual tool calls from the LLM
                            print(f"DEBUG: About to execute tool calls")
                            
                            # Execute the tool calls directly
                            print(f"DEBUG: Executing {len(current_tool_calls)} tool calls")
                            
                            # Store tool results for second API call
                            tool_results_cache = {}
                            
                            for tool_call_data in current_tool_calls.values():
                                function_name = tool_call_data['function']['name']
                                function_args_str = tool_call_data['function']['arguments']
                                tool_call_id = tool_call_data['id']
                                
                                print(f"DEBUG: Executing {function_name} with args: {function_args_str}")
                                
                                try:
                                    # Parse arguments
                                    function_args = json.loads(function_args_str)
                                    
                                    # Find and execute the tool
                                    tool_executed = False
                                    for tool in available_tools:
                                        if hasattr(tool, function_name):
                                            method = getattr(tool, function_name)
                                            if callable(method):
                                                result = method(**function_args)
                                                print(f"DEBUG: Tool {function_name} executed successfully")
                                                
                                                # Store result for second API call
                                                tool_results_cache[function_name] = result
                                                
                                                # Yield the result
                                                yield {
                                                    "type": "response_chunk",
                                                    "content": f"\n[Tool Result: {function_name}]\n{result[:200]}{'...' if len(result) > 200 else ''}",
                                                    "complete": False
                                                }
                                                
                                                tool_executed = True
                                                break
                                    
                                    if not tool_executed:
                                        print(f"DEBUG: Tool {function_name} not found")
                                        yield {
                                            "type": "response_chunk",
                                            "content": f"\n[Error: Tool {function_name} not found]",
                                            "complete": False
                                        }
                                
                                except json.JSONDecodeError as e:
                                    print(f"DEBUG: JSON decode error for {function_name}: {e}")
                                    yield {
                                        "type": "response_chunk",
                                        "content": f"\n[Error: Invalid arguments for {function_name}]",
                                        "complete": False
                                    }
                                except Exception as e:
                                    print(f"DEBUG: Error executing {function_name}: {e}")
                                    yield {
                                        "type": "response_chunk",
                                        "content": f"\n[Error executing {function_name}: {str(e)}]",
                                        "complete": False
                                    }
                            
                            print(f"DEBUG: Finished executing tool calls")
                            print(f"DEBUG: EXITING tool_calls branch")
                            
                            # After tool execution, make a second API call with tool results
                            print(f"DEBUG: Making second API call with tool results")
                            
                            # Prepare tool results for the second call using cached results
                            tool_results_messages = []
                            
                            for tool_call_data in current_tool_calls.values():
                                function_name = tool_call_data['function']['name']
                                tool_call_id = tool_call_data['id']
                                
                                # Use cached result from first execution
                                if function_name in tool_results_cache:
                                    tool_result = tool_results_cache[function_name]
                                    tool_results_messages.append({
                                        "role": "tool",
                                        "tool_call_id": tool_call_id,
                                        "content": tool_result
                                    })
                            
                            # Make second API call with tool results
                            if tool_results_messages:
                                print(f"DEBUG: Making second API call with {len(tool_results_messages)} tool results")
                                
                                # Create the original message with tool_calls
                                original_tool_calls = []
                                for tool_call_data in current_tool_calls.values():
                                    original_tool_calls.append({
                                        "id": tool_call_data['id'],
                                        "type": "function",
                                        "function": {
                                            "name": tool_call_data['function']['name'],
                                            "arguments": tool_call_data['function']['arguments']
                                        }
                                    })
                                
                                # Prepare messages for second call with proper structure
                                second_messages = messages + [
                                    {
                                        "role": "assistant",
                                        "content": None,
                                        "tool_calls": original_tool_calls
                                    }
                                ] + tool_results_messages + [
                                    {"role": "user", "content": "Please analyze the document content and provide a comprehensive summary and analysis."}
                                ]
                                
                                # Ensure model_params is not None
                                safe_model_params = model_params or {}
                                
                                # Make the second API call
                                second_response = self.client.chat.completions.create(
                                    model=self.model,
                                    messages=second_messages,
                                    stream=True,
                                    **safe_model_params
                                )
                                
                                # Stream the final response
                                for chunk in second_response:
                                    if chunk.choices and len(chunk.choices) > 0:
                                        choice = chunk.choices[0]
                                        delta = choice.delta
                                        
                                        if delta.content:
                                            yield {
                                                "type": "response_chunk",
                                                "content": delta.content,
                                                "complete": False
                                            }
                                        
                                        if choice.finish_reason == "stop":
                                            yield {
                                                "type": "response_complete",
                                                "content": "",
                                                "complete": True
                                            }
                                            break
                            
                            break
                        except Exception as e:
                            print(f"DEBUG: Exception in tool_calls branch: {e}")
                            import traceback
                            traceback.print_exc()
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
            # Step 1: Execute all tools (non-streaming but fast)
            tool_results = []
            tools_executed = []
            

            if available_tools:
                tool_logger.info(f"🚀 Starting OpenAI streaming tool execution for query: '{user_query[:60]}{'...' if len(user_query) > 60 else ''}'")
                tool_logger.info(f"📋 Available tools: {len(available_tools)} tools")
                
                # Execute search tool if available
                search_tool = next((tool for tool in available_tools if hasattr(tool, 'search')), None)
                if search_tool:
                    tool_start_time = datetime.now()
                    query = user_query
                    tool_logger.info("🔍 EXECUTING: search_google (streaming)")
                    tool_logger.info(f"   Parameters: {{'query': '{query[:100]}{'...' if len(query) > 100 else ''}'}}")
                    
                    try:
                        search_result = search_tool.search(query)
                        tool_execution_time = (datetime.now() - tool_start_time).total_seconds()
                        result_preview = search_result[:200] + "..." if len(search_result) > 200 else search_result
                        
                        tool_logger.info(f"✅ SUCCESS: search_google completed in {tool_execution_time:.2f}s")
                        tool_logger.info(f"   Result preview: {result_preview}")
                        
                        tool_results.append(f"Search Results:\n{search_result}")
                        tools_executed.append({
                            "name": "search_google",
                            "status": "success",
                            "execution_time": tool_execution_time,
                            "result_length": len(search_result)
                        })
                        
                        yield {
                            "type": "tool_execution",
                            "content": f"🔍 Search completed ({tool_execution_time:.1f}s) - Found {len(search_result)} chars of results",
                            "tool_name": "search_google",
                            "status": "completed",
                            "execution_time": tool_execution_time
                        }
                        
                    except Exception as e:
                        tool_execution_time = (datetime.now() - tool_start_time).total_seconds()
                        tool_logger.error(f"❌ ERROR: search_google failed in {tool_execution_time:.2f}s - {str(e)}")
                        tools_executed.append({
                            "name": "search_google",
                            "status": "error",
                            "execution_time": tool_execution_time,
                            "error": str(e)
                        })
                        
                        yield {
                            "type": "tool_execution",
                            "content": f"❌ Search failed: {str(e)}",
                            "tool_name": "search_google",
                            "status": "error",
                            "execution_time": tool_execution_time
                        }
                
                # Execute dynamic tools (file_access, business_logic, etc.) if available
                for tool in available_tools:
                    if hasattr(tool, 'get_tool_description'):
                        # This is a dynamic tool - we need to let the LLM decide which specific method to call
                        # Skip automatic execution for dynamic tools
                        continue
                
                # Execute learning tools if available
                learning_search_tool = next((tool for tool in available_tools if hasattr(tool, 'search_learning_context')), None)
                if learning_search_tool:
                    tool_start_time = datetime.now()
                    tool_logger.info("📚 EXECUTING: search_learning_context (streaming)")
                    tool_logger.info(f"   Parameters: {{'query': '{user_query[:100]}{'...' if len(user_query) > 100 else ''}'}}")
                    
                    try:
                        learning_search_result = learning_search_tool.search_learning_context(query=user_query)
                        tool_execution_time = (datetime.now() - tool_start_time).total_seconds()
                        result_preview = learning_search_result[:200] + "..." if len(learning_search_result) > 200 else learning_search_result
                        
                        tool_logger.info(f"✅ SUCCESS: search_learning_context completed in {tool_execution_time:.2f}s")
                        tool_logger.info(f"   Result preview: {result_preview}")
                        
                        tool_results.append(f"Learning Context Search:\n{learning_search_result}")
                        tools_executed.append({
                            "name": "search_learning_context",
                            "status": "success",
                            "execution_time": tool_execution_time,
                            "result_length": len(learning_search_result)
                        })
                        
                        yield {
                            "type": "tool_execution",
                            "content": f"📚 Learning context search completed ({tool_execution_time:.1f}s)",
                            "tool_name": "search_learning_context",
                            "status": "completed",
                            "execution_time": tool_execution_time
                        }
                        
                    except Exception as e:
                        tool_execution_time = (datetime.now() - tool_start_time).total_seconds()
                        tool_logger.error(f"❌ ERROR: search_learning_context failed in {tool_execution_time:.2f}s - {str(e)}")
                        tools_executed.append({
                            "name": "search_learning_context",
                            "status": "error",
                            "execution_time": tool_execution_time,
                            "error": str(e)
                        })
                        
                        yield {
                            "type": "tool_execution",
                            "content": f"❌ Learning context search failed: {str(e)}",
                            "tool_name": "search_learning_context",
                            "status": "error",
                            "execution_time": tool_execution_time
                        }
                
                learning_analysis_tool = next((tool for tool in available_tools if hasattr(tool, 'analyze_learning_opportunity')), None)
                if learning_analysis_tool:
                    tool_start_time = datetime.now()
                    tool_logger.info("🧠 EXECUTING: analyze_learning_opportunity (streaming)")
                    tool_logger.info(f"   Parameters: {{'user_message': '{user_query[:100]}{'...' if len(user_query) > 100 else ''}'}}")
                    
                    try:
                        learning_analysis_result = await learning_analysis_tool.analyze_learning_opportunity(user_message=user_query)
                        tool_execution_time = (datetime.now() - tool_start_time).total_seconds()
                        result_preview = learning_analysis_result[:200] + "..." if len(learning_analysis_result) > 200 else learning_analysis_result
                        
                        tool_logger.info(f"✅ SUCCESS: analyze_learning_opportunity completed in {tool_execution_time:.2f}s")
                        tool_logger.info(f"   Result preview: {result_preview}")
                        
                        tool_results.append(f"Learning Analysis:\n{learning_analysis_result}")
                        tools_executed.append({
                            "name": "analyze_learning_opportunity",
                            "status": "success",
                            "execution_time": tool_execution_time,
                            "result_length": len(learning_analysis_result)
                        })
                        
                        yield {
                            "type": "tool_execution",
                            "content": f"🧠 Learning analysis completed ({tool_execution_time:.1f}s)",
                            "tool_name": "analyze_learning_opportunity",
                            "status": "completed",
                            "execution_time": tool_execution_time
                        }
                        
                    except Exception as e:
                        tool_execution_time = (datetime.now() - tool_start_time).total_seconds()
                        tool_logger.error(f"❌ ERROR: analyze_learning_opportunity failed in {tool_execution_time:.2f}s - {str(e)}")
                        tools_executed.append({
                            "name": "analyze_learning_opportunity",
                            "status": "error",
                            "execution_time": tool_execution_time,
                            "error": str(e)
                        })
                        
                        yield {
                            "type": "tool_execution",
                            "content": f"❌ Learning analysis failed: {str(e)}",
                            "tool_name": "analyze_learning_opportunity",
                            "status": "error",
                            "execution_time": tool_execution_time
                        }
                
                # Summary of all tool executions
                successful_tools = [t for t in tools_executed if t["status"] == "success"]
                failed_tools = [t for t in tools_executed if t["status"] == "error"]
                total_execution_time = sum(t["execution_time"] for t in tools_executed)
                
                tool_logger.info(f"📊 OPENAI STREAMING TOOL EXECUTION SUMMARY:")
                tool_logger.info(f"   Total tools: {len(tools_executed)}")
                tool_logger.info(f"   Successful: {len(successful_tools)}")
                tool_logger.info(f"   Failed: {len(failed_tools)}")
                tool_logger.info(f"   Total time: {total_execution_time:.2f}s")
                
                yield {
                    "type": "tools_summary",
                    "content": f"🏁 Tools completed: {len(successful_tools)}/{len(tools_executed)} successful ({total_execution_time:.1f}s total)",
                    "tools_executed": tools_executed,
                    "total_execution_time": total_execution_time
                }
            
            # Step 2: Build conversation with tool results and history
            # Enhance system prompt with context if provided and priority is high
            if system_prompt and "IMPORTANT CONTEXT:" in system_prompt:
                system_prompt += "\n\nPlease consider this context when answering questions."
            
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
            user_message_content = user_query
            
            # Combine all tool results
            combined_tool_results = "\n\n".join(tool_results) if tool_results else "No additional information found."
            
            messages.extend([
                {
                    "role": "user", 
                    "content": user_message_content
                },
                {
                    "role": "assistant",
                    "content": f"I'll analyze the available information about this. Let me process the tool results:\n\n{combined_tool_results}"
                },
                {
                    "role": "user",
                    "content": "Based on the tool results above, please provide a comprehensive answer to my original question."
                }
            ])
            
            # Step 3: Stream the final response using OpenAI
            final_response_params = {
                "model": self.model,
                "messages": messages,
                "stream": True,
                "temperature": 0.7,
                "max_tokens": 4000  # Use reasonable default that works with most models
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
        tools_executed = []
        tool_logger.info(f"🚀 Starting OpenAI tool execution for query: '{original_query[:60]}{'...' if len(original_query) > 60 else ''}'")
        tool_logger.info(f"📋 Tool calls requested: {len(message.tool_calls)} tools")
        
        for tool_call in message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            tool_start_time = datetime.now()
            tool_logger.info(f"🔍 EXECUTING: {function_name}")
            tool_logger.info(f"   Tool Call ID: {tool_call.id}")
            tool_logger.info(f"   Parameters: {function_args}")
            
            # Find the appropriate tool and execute it
            tool_executed = False
            
            for tool in available_tools:
                # Check if this tool has the requested method
                if hasattr(tool, function_name):
                    try:
                        method = getattr(tool, function_name)
                        # Call the method with the provided arguments
                        if callable(method):
                            result = method(**function_args)
                            tool_execution_time = (datetime.now() - tool_start_time).total_seconds()
                            result_preview = result[:200] + "..." if len(result) > 200 else result
                            
                            tool_logger.info(f"✅ SUCCESS: {function_name} completed in {tool_execution_time:.2f}s")
                            tool_logger.info(f"   Result preview: {result_preview}")
                            
                            messages.append({
                                "tool_call_id": tool_call.id,
                                "role": "tool",
                                "name": function_name,
                                "content": result
                            })
                            
                            tools_executed.append({
                                "name": function_name,
                                "status": "success",
                                "execution_time": tool_execution_time,
                                "result_length": len(result),
                                "tool_call_id": tool_call.id
                            })
                            tool_executed = True
                            break
                    except Exception as e:
                        tool_execution_time = (datetime.now() - tool_start_time).total_seconds()
                        tool_logger.error(f"❌ ERROR: {function_name} failed in {tool_execution_time:.2f}s - {str(e)}")
                        
                        messages.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": f"Error: {str(e)}"
                        })
                        
                        tools_executed.append({
                            "name": function_name,
                            "status": "error",
                            "execution_time": tool_execution_time,
                            "error": str(e),
                            "tool_call_id": tool_call.id
                        })
                        tool_executed = True
                        break
            
            # Handle fallback for search_google if no tool found
            if not tool_executed and function_name == "search_google":
                tool_logger.warning(f"⚠️  No search tool found for {function_name}, using fallback")
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool", 
                    "name": function_name,
                    "content": "Search functionality not available"
                })
                
                tools_executed.append({
                    "name": function_name,
                    "status": "fallback",
                    "execution_time": (datetime.now() - tool_start_time).total_seconds(),
                    "tool_call_id": tool_call.id
                })
            elif not tool_executed:
                tool_execution_time = (datetime.now() - tool_start_time).total_seconds()
                tool_logger.error(f"❌ UNKNOWN TOOL: {function_name} not found in available tools")
                
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name, 
                    "content": f"Error: Tool '{function_name}' not found"
                })
                
                tools_executed.append({
                    "name": function_name,
                    "status": "not_found",
                    "execution_time": tool_execution_time,
                    "error": f"Tool '{function_name}' not found",
                    "tool_call_id": tool_call.id
                })
        
        # Summary of all tool executions
        successful_tools = [t for t in tools_executed if t["status"] == "success"]
        failed_tools = [t for t in tools_executed if t["status"] in ["error", "not_found"]]
        total_execution_time = sum(t["execution_time"] for t in tools_executed)
        
        tool_logger.info(f"📊 OPENAI TOOL EXECUTION SUMMARY:")
        tool_logger.info(f"   Total tools: {len(tools_executed)}")
        tool_logger.info(f"   Successful: {len(successful_tools)}")
        tool_logger.info(f"   Failed: {len(failed_tools)}")
        tool_logger.info(f"   Total time: {total_execution_time:.2f}s")
        # Send tool results back to GPT-4
        try:
            final_response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=4000  # Use reasonable default that works with most models
            )
            
            return final_response.choices[0].message.content
            
        except Exception as e:
            return f"Error processing tool results: {str(e)}"
    
    async def _handle_tool_calls_streaming(self, message: Any, available_tools: List[Any], original_query: str) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Handle tool calls from GPT-4 with streaming response
        
        Args:
            message: GPT-4's message containing tool calls
            available_tools: List of available tool instances
            original_query: The original user query
            
        Yields:
            Dict containing streaming response data
        """
        
        print(f"DEBUG: _handle_tool_calls_streaming called with {len(available_tools)} tools")
        print(f"DEBUG: Message tool calls: {len(message.tool_calls) if message.tool_calls else 0}")
        print(f"DEBUG: METHOD START - About to process tool calls")
        
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
            print(f"DEBUG: Processing tool call: {function_name}")
            print(f"DEBUG: Arguments string: '{tool_call.function.arguments}'")
            print(f"DEBUG: Arguments type: {type(tool_call.function.arguments)}")
            
            try:
                function_args = json.loads(tool_call.function.arguments)
                print(f"DEBUG: Parsed arguments: {function_args}")
            except json.JSONDecodeError as e:
                print(f"DEBUG: JSON decode error: {e}")
                print(f"DEBUG: Raw arguments: {repr(tool_call.function.arguments)}")
                # Try to handle empty or malformed arguments
                if not tool_call.function.arguments or tool_call.function.arguments.strip() == "":
                    function_args = {}
                else:
                    # Skip this tool call if we can't parse it
                    continue
            
            # Find the appropriate tool and execute it
            tool_executed = False
            
            for tool in available_tools:
                # Check if this tool has the requested method
                if hasattr(tool, function_name):
                    try:
                        method = getattr(tool, function_name)
                        # Call the method with the provided arguments
                        if callable(method):
                            result = method(**function_args)
                            messages.append({
                                "tool_call_id": tool_call.id,
                                "role": "tool",
                                "name": function_name,
                                "content": result
                            })
                            tool_executed = True
                            break
                    except Exception as e:
                        print(f"Error executing tool {function_name}: {e}")
                        messages.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": f"Error: {str(e)}"
                        })
                        tool_executed = True
                        break
            
            # Legacy hardcoded tool handling for backward compatibility
            if not tool_executed:
                if function_name == "search_google":
                    # Find the search tool
                    search_tool = next((tool for tool in available_tools if hasattr(tool, 'search')), None)
                    if search_tool:
                        result = search_tool.search(function_args.get("query", ""))
                        messages.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": result
                        })
                elif function_name == "get_context":
                    # Find the context tool
                    context_tool = next((tool for tool in available_tools if hasattr(tool, 'get_context')), None)
                    if context_tool:
                        query = function_args.get("query", "")
                        source_types = function_args.get("source_types", None)
                        # Pass user info if available from the conversation
                        result = context_tool.get_context(query, source_types, user_id="unknown", org_id="unknown")
                        messages.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool", 
                            "name": function_name,
                            "content": result
                        })
        
        print("Tool result:", messages)
        
        # Send tool results back to GPT-4 with streaming
        try:
            response_stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                max_tokens=4000  # Use reasonable default that works with most models
            )
            
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
                "content": f"Error processing tool results: {str(e)}",
                "complete": True
            } 