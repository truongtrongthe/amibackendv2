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
                tool_choice="auto",
                max_tokens=10000  # Default to 10k tokens for longer responses
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
            available_tools: List of available tool instances (search, context, etc.)
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
            system_prompt = "You are a helpful assistant that can search for information and retrieve relevant context when needed."
        
        # Define functions for OpenAI only if tools are available
        functions = []
        
        if available_tools:
            for tool in available_tools:
                # Check if tool has get_tool_description method (dynamic tool definitions)
                if hasattr(tool, 'get_tool_description'):
                    try:
                        tool_def = tool.get_tool_description()
                        functions.append({
                            "type": "function",
                            "function": tool_def
                        })
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
                        
                        # Stream the tool execution and final response using the simple approach
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
            # Step 1: Execute all tools (non-streaming but fast)
            tool_results = []
            

            if available_tools:
                # Execute search tool if available
                search_tool = next((tool for tool in available_tools if hasattr(tool, 'search')), None)
                if search_tool:
                    search_result = search_tool.search(user_query)
                    tool_results.append(f"Search Results:\n{search_result}")
                
                # Execute learning tools if available
                learning_search_tool = next((tool for tool in available_tools if hasattr(tool, 'search_learning_context')), None)
                if learning_search_tool:
                    # Extract key terms from user query for learning search
                    learning_search_result = learning_search_tool.search_learning_context(query=user_query)
                    tool_results.append(f"Learning Context Search:\n{learning_search_result}")
                
                learning_analysis_tool = next((tool for tool in available_tools if hasattr(tool, 'analyze_learning_opportunity')), None)
                if learning_analysis_tool:
                    learning_analysis_result = await learning_analysis_tool.analyze_learning_opportunity(user_message=user_query)
                    tool_results.append(f"Learning Analysis:\n{learning_analysis_result}")
                    
                    # Check if learning decision should be created
                    if "MAYBE_LEARN" in learning_analysis_result or "SHOULD_LEARN" in learning_analysis_result:
                        human_learning_tool = next((tool for tool in available_tools if hasattr(tool, 'request_learning_decision')), None)
                        if human_learning_tool:
                            try:
                                # Extract key information for learning decision
                                decision_result = human_learning_tool.request_learning_decision(
                                    decision_type="save_new",
                                    context=f"Teaching content detected: {user_query}",
                                    options=["Save as new knowledge", "Skip learning", "Need more context"],
                                    additional_info="AI analysis detected teaching intent with factual content"
                                )
                                tool_results.append(f"Learning Decision Created:\n{decision_result}")
                            except Exception as e:
                                tool_results.append(f"Learning Decision Error:\n{str(e)}")
                
                yield {
                    "type": "response_chunk",
                    "content": "\n[Tools executed, generating response...]",
                    "complete": False
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
                "max_tokens": 10000  # Default to 10k tokens for longer responses
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
        # Send tool results back to GPT-4
        try:
            final_response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=10000  # Default to 10k tokens for longer responses
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
                max_tokens=10000  # Default to 10k tokens for longer responses
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