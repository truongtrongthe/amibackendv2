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
    
    async def process_with_tools_stream(self, user_query: str, available_tools: List[Any], system_prompt: str = None, force_tools: bool = False, conversation_history: List[Dict[str, Any]] = None, max_history_messages: int = 25, max_history_tokens: int = 6000) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process user query with available tools using Claude with streaming
        
        Args:
            user_query: The user's input query
            available_tools: List of available tool instances (search, context, etc.)
            system_prompt: Optional custom system prompt
            force_tools: Whether to force tool usage (not directly supported by Anthropic, but influences behavior)
            conversation_history: Previous conversation messages
            max_history_messages: Maximum number of history messages to include
            max_history_tokens: Maximum token count for history
            
        Yields:
            Dict containing streaming response data
        """
        
        # Use default system prompt if none provided
        if system_prompt is None:
            system_prompt = "You are a helpful assistant that can search for information and retrieve relevant context when needed."
        
        # Define tools for Claude only if tools are available
        tools = []
        if available_tools:
            for tool in available_tools:
                # Check tool type and add appropriate tool definition
                if hasattr(tool, 'search'):  # Search tool
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
                elif hasattr(tool, 'get_context'):  # Context tool
                    tools.append({
                        "name": "get_context", 
                        "description": "Retrieve relevant context including user profile, system status, organization info, and knowledge base information",
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "The topic or question to get context for"
                                },
                                "source_types": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Specific context sources to query: user_profile, system_status, organization_info, knowledge_base, recent_activity"
                                }
                            },
                            "required": ["query"]
                        }
                    })
                elif hasattr(tool, 'search_learning_context'):  # Learning search tool
                    tools.append({
                        "name": "search_learning_context",
                        "description": "Search existing knowledge base for similar content to avoid duplicates",
                        "input_schema": {
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
                    })
                elif hasattr(tool, 'analyze_learning_opportunity'):  # Learning analysis tool
                    tools.append({
                        "name": "analyze_learning_opportunity",
                        "description": "Analyze if a message contains valuable information that should be learned",
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "user_message": {
                                    "type": "string",
                                    "description": "The user message to analyze for learning opportunities"
                                }
                            },
                            "required": ["user_message"]
                        }
                    })
                elif hasattr(tool, 'request_learning_decision'):  # Human learning decision tool
                    tools.append({
                        "name": "request_learning_decision",
                        "description": "Request human decision on whether to learn specific information",
                        "input_schema": {
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
                    })
                elif hasattr(tool, 'preview_knowledge_save'):  # Knowledge preview tool
                    tools.append({
                        "name": "preview_knowledge_save",
                        "description": "Preview what knowledge would be saved before actual saving",
                        "input_schema": {
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
                    })
                elif hasattr(tool, 'save_knowledge'):  # Knowledge save tool
                    tools.append({
                        "name": "save_knowledge",
                        "description": "Save approved knowledge to the knowledge base",
                        "input_schema": {
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
                    })

        # Prepare API call parameters
        api_params = {
            "model": self.model,
            "max_tokens": 1024,
            "stream": True
        }
        
        # Build messages with conversation history
        messages = []
        
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
                
                # Convert role names to Anthropic format
                if role in ["user", "human"]:
                    anthropic_role = "user"
                elif role in ["assistant", "ai"]:
                    anthropic_role = "assistant"
                elif role == "system":
                    # System messages will be included in the first user message
                    continue
                else:
                    continue  # Skip unknown roles
                
                # Estimate token count (rough approximation: 1 token ≈ 4 characters)
                estimated_tokens = len(content) // 4
                if token_count + estimated_tokens > max_history_tokens:
                    break
                
                history_messages.append({
                    "role": anthropic_role,
                    "content": content
                })
                token_count += estimated_tokens
                message_count += 1
            
            # Add messages in chronological order
            messages.extend(reversed(history_messages))
        
        # Prepare current message with system prompt if provided
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
                        
                        # Stream the tool execution and final response using the simple approach
                        async for final_chunk in self._stream_tool_execution(
                            user_query, available_tools, system_prompt,
                            conversation_history, max_history_messages, max_history_tokens
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
        system_prompt: str = None,
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
                    learning_search_result = learning_search_tool.search_learning_context(query=user_query)
                    tool_results.append(f"Learning Context Search:\n{learning_search_result}")
                
                learning_analysis_tool = next((tool for tool in available_tools if hasattr(tool, 'analyze_learning_opportunity')), None)
                if learning_analysis_tool:
                    learning_analysis_result = learning_analysis_tool.analyze_learning_opportunity(user_message=user_query)
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
            
            # Step 2: Build conversation with tool results and history for Claude
            # Use default system prompt if none provided
            if system_prompt is None:
                system_prompt = "You are a helpful assistant that can search for information and retrieve relevant context when needed."
            
            messages = []
            
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
                    
                    # Convert role names to Anthropic format
                    if role in ["user", "human"]:
                        anthropic_role = "user"
                    elif role in ["assistant", "ai"]:
                        anthropic_role = "assistant"
                    elif role == "system":
                        # System messages will be included in the final query
                        continue
                    else:
                        continue  # Skip unknown roles
                    
                    # Estimate token count (rough approximation: 1 token ≈ 4 characters)
                    estimated_tokens = len(content) // 4
                    if token_count + estimated_tokens > max_history_tokens:
                        break
                    
                    history_messages.append({
                        "role": anthropic_role,
                        "content": content
                    })
                    token_count += estimated_tokens
                    message_count += 1
                
                # Add messages in chronological order
                messages.extend(reversed(history_messages))
            
            # Build final query with system prompt and tool results
            user_message_content = user_query
            
            # Combine all tool results
            combined_tool_results = "\n\n".join(tool_results) if tool_results else "No additional information found."
            
            if system_prompt:
                final_query = f"System: {system_prompt}\n\nUser: {user_message_content}\n\nTool Results:\n{combined_tool_results}\n\nBased on the tool results above, please provide a comprehensive answer to the original question."
            else:
                final_query = f"User: {user_message_content}\n\nTool Results:\n{combined_tool_results}\n\nBased on the tool results above, please provide a comprehensive answer to the original question."
            
            messages.append({
                "role": "user",
                "content": final_query
            })
            
            # Step 3: Stream the final response using Claude
            final_response_params = {
                "model": self.model,
                "max_tokens": 1024,
                "messages": messages,
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
                
                # Find the appropriate tool and execute it using reflection
                tool_executed = False
                
                for tool in available_tools:
                    # Check if this tool has the requested method
                    if hasattr(tool, tool_name):
                        try:
                            method = getattr(tool, tool_name)
                            # Call the method with the provided arguments
                            if callable(method):
                                result = method(**tool_input)
                                tool_results.append({
                                    "tool_use_id": tool_use_id,
                                    "type": "tool_result",
                                    "content": result
                                })
                                tool_executed = True
                                break
                        except Exception as e:
                            print(f"Error executing tool {tool_name}: {e}")
                            tool_results.append({
                                "tool_use_id": tool_use_id,
                                "type": "tool_result",
                                "content": f"Error: {str(e)}"
                            })
                            tool_executed = True
                            break
                
                # Legacy hardcoded tool handling for backward compatibility
                if not tool_executed:
                    if tool_name == "search_google" and available_tools:
                        search_tool = next((tool for tool in available_tools if hasattr(tool, 'search')), None)
                        if search_tool:
                            result = search_tool.search(tool_input.get("query", ""))
                            tool_results.append({
                                "tool_use_id": tool_use_id,
                                "type": "tool_result",
                                "content": result
                            })
                    elif tool_name == "get_context" and available_tools:
                        context_tool = next((tool for tool in available_tools if hasattr(tool, 'get_context')), None)
                        if context_tool:
                            query = tool_input.get("query", "")
                            source_types = tool_input.get("source_types", None)
                            result = context_tool.get_context(query, source_types, user_id="unknown", org_id="unknown")
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
    
    async def _handle_tool_use_streaming(self, response: Any, available_tools: List[Any], original_query: str) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Handle tool use requests from Claude with streaming response
        
        Args:
            response: Claude's response containing tool use requests
            available_tools: List of available tool instances
            original_query: The original user query
            
        Yields:
            Dict containing streaming response data
        """
        
        tool_results = []
        
        for content_block in response.content:
            if content_block.type == "tool_use":
                tool_name = content_block.name
                tool_input = content_block.input
                tool_use_id = content_block.id
                
                # Find the appropriate tool and execute it using reflection
                tool_executed = False
                
                for tool in available_tools:
                    # Check if this tool has the requested method
                    if hasattr(tool, tool_name):
                        try:
                            method = getattr(tool, tool_name)
                            # Call the method with the provided arguments
                            if callable(method):
                                result = method(**tool_input)
                                tool_results.append({
                                    "tool_use_id": tool_use_id,
                                    "type": "tool_result",
                                    "content": result
                                })
                                tool_executed = True
                                break
                        except Exception as e:
                            print(f"Error executing tool {tool_name}: {e}")
                            tool_results.append({
                                "tool_use_id": tool_use_id,
                                "type": "tool_result",
                                "content": f"Error: {str(e)}"
                            })
                            tool_executed = True
                            break
                
                # Legacy hardcoded tool handling for backward compatibility
                if not tool_executed:
                    if tool_name == "search_google" and available_tools:
                        search_tool = next((tool for tool in available_tools if hasattr(tool, 'search')), None)
                        if search_tool:
                            result = search_tool.search(tool_input.get("query", ""))
                            tool_results.append({
                                "tool_use_id": tool_use_id,
                                "type": "tool_result",
                                "content": result
                            })
                    elif tool_name == "get_context" and available_tools:
                        context_tool = next((tool for tool in available_tools if hasattr(tool, 'get_context')), None)
                        if context_tool:
                            query = tool_input.get("query", "")
                            source_types = tool_input.get("source_types", None)
                            result = context_tool.get_context(query, source_types, user_id="unknown", org_id="unknown")
                            tool_results.append({
                                "tool_use_id": tool_use_id,
                                "type": "tool_result",
                                "content": result
                            })
        
        # Send tool results back to Claude with streaming
        try:
            messages = [
                {"role": "user", "content": original_query},
                {"role": "assistant", "content": response.content},
                {"role": "user", "content": tool_results}
            ]
            
            response_stream = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=messages,
                stream=True
            )
            
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
                "content": f"Error processing tool results: {str(e)}",
                "complete": True
            } 