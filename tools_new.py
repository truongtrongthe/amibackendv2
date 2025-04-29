"""
Tool calling implementation for the MC system with optimized performance.
This file defines the tool schemas and handlers for LLM tool calling with combined operations,
parallel processing, and optimized prompts.
"""

import json
import re
import time
import asyncio
import traceback
from typing import List, Dict, Any, Optional, AsyncGenerator, Union, Tuple
from functools import lru_cache
from langchain_openai import ChatOpenAI
from utilities import logger

from brain_singleton import get_brain, get_current_graph_version
from tool_helpers import (
    extract_structured_data_from_raw,
    detect_language,
    ensure_brain_loaded,
    optimize_knowledge_context,
    format_knowledge_entry,
    extract_key_knowledge
)
from profile_helper import (
    build_user_profile,
    format_user_profile_for_prompt
)
from response_optimization import ResponseProcessor, ResponseFilter

# Use the same LLM instances as mc.py
LLM = ChatOpenAI(model="gpt-4o", streaming=False)
StreamLLM = ChatOpenAI(model="gpt-4o", streaming=True)

# Performance tracking
PERF_METRICS = {}

# Access the brain singleton once
brain = get_brain()

# Add a module-level variable to store results from the CoT handler
_last_cot_results = {
    "analysis_content": "",
    "next_actions_content": "",
    "knowledge_entries": [],
    "knowledge_context": ""
}

class ToolRegistry:
    """Registry for all available tools"""
    
    def __init__(self):
        self.tools = {}
        self.tool_schemas = {}
    
    def register_tool(self, name: str, handler, schema: Dict):
        """Register a tool with its handler and schema"""
        self.tools[name] = handler
        self.tool_schemas[name] = schema
        logger.info(f"Registered tool: {name}")
    
    def get_tool_schemas(self) -> List[Dict]:
        """Get all tool schemas in a format suitable for LLM tool calling"""
        return list(self.tool_schemas.values())
    
    def get_tool_handler(self, name: str):
        """Get a tool handler by name"""
        if name not in self.tools:
            raise ValueError(f"Tool not found: {name}")
        return self.tools[name]
    
    def get_tool_names(self) -> List[str]:
        """Get list of all registered tool names"""
        return list(self.tools.keys())


# Initialize the tool registry
tool_registry = ToolRegistry()


# OPTIMIZED TOOL SCHEMAS
# Combined analysis and next actions tool schema
COMBINED_ANALYSIS_ACTIONS_SCHEMA = {
    "name": "combined_analysis_actions_tool",
    "description": "Analyze the conversation context and determine next actions in a single operation",
    "parameters": {
        "type": "object",
        "properties": {
            "conversation_context": {
                "type": "string",
                "description": "The full conversation history to analyze"
            },
            "graph_version_id": {
                "type": "string",
                "description": "The version ID of the knowledge graph to query for profile information"
            },
            "knowledge_context": {
                "type": "string",
                "description": "The retrieved knowledge relevant to the conversation"
            },
            "additional_instructions": {
                "type": "string",
                "description": "Optional additional instructions for the analysis",
                "default": ""
            }
        },
        "required": ["conversation_context", "graph_version_id"]
    }
}

KNOWLEDGE_QUERY_SCHEMA = {
    "name": "knowledge_query_tool",
    "description": "Query the knowledge base for information relevant to the user's request",
    "parameters": {
        "type": "object",
        "properties": {
            "queries": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of search queries to run against the knowledge base"
            },
            "graph_version_id": {
                "type": "string",
                "description": "The version ID of the knowledge graph to query"
            },
            "top_k": {
                "type": "integer",
                "description": "Number of results to return per query",
                "default": 3
            }
        },
        "required": ["queries", "graph_version_id"]
    }
}

RESPONSE_GENERATION_SCHEMA = {
    "name": "response_generation_tool",
    "description": "Generate a response to the user based on conversation context, analysis, and knowledge",
    "parameters": {
        "type": "object",
        "properties": {
            "conversation_context": {
                "type": "string",
                "description": "The conversation history"
            },
            "analysis": {
                "type": "string",
                "description": "The analysis of the conversation"
            },
            "next_actions": {
                "type": "string",
                "description": "The planned next actions"
            },
            "knowledge_context": {
                "type": "string",
                "description": "The retrieved knowledge"
            },
            "personality_instructions": {
                "type": "string",
                "description": "Instructions for the AI's personality"
            },
            "knowledge_found": {
                "type": "boolean",
                "description": "Whether relevant knowledge was found",
                "default": False
            }
        },
        "required": ["conversation_context", "analysis", "next_actions", "personality_instructions", "knowledge_context"]
    }
}

# Register the CoT handler schema
COT_KNOWLEDGE_ANALYSIS_SCHEMA = {
    "name": "cot_knowledge_analysis_tool",
    "description": "Combine knowledge retrieval with analysis and next actions using Chain-of-Thought reasoning",
    "parameters": {
        "type": "object",
        "properties": {
            "conversation_context": {
                "type": "string",
                "description": "The full conversation history to analyze"
            },
            "graph_version_id": {
                "type": "string",
                "description": "The version ID of the knowledge graph to query"
            }
        },
        "required": ["conversation_context", "graph_version_id"]
    }
}

# OPTIMIZED TOOL HANDLERS
# Combined analysis and next actions handler
async def combined_analysis_actions_handler(params: Dict) -> AsyncGenerator[Dict, None]:
    """
    Combined handler for context analysis and next actions with improved performance.
    
    Args:
        params: Dictionary containing conversation_context, graph_version_id, 
                knowledge_context (optional), and additional_instructions (optional)
        
    Yields:
        Dict events with streaming results for both analysis and next actions
    """
    start_time = time.time()
    PERF_METRICS["combined_analysis_start"] = start_time
    
    conversation_context = params.get("conversation_context", "")
    graph_version_id = params.get("graph_version_id", "")
    knowledge_context = params.get("knowledge_context", "")
    additional_instructions = params.get("additional_instructions", "")
    thread_id = params.get("_thread_id")
    
    # OPTIMIZATION: Skip profiling completely for faster analysis
    profiling_instructions = {}
    
    # SUPER OPTIMIZATION: Just extract last user message for fastest analysis
    last_user_message = ""
    for line in conversation_context.strip().split('\n'):
        if line.startswith("User:"):
            last_user_message = line[5:].strip()
    
    if not last_user_message and conversation_context:
        # If we couldn't extract it using the method above, take the last line
        lines = conversation_context.strip().split('\n')
        if lines:
            last_line = lines[-1]
            if last_line.startswith("User:"):
                last_user_message = last_line[5:].strip()
            else:
                last_user_message = last_line
    
    # Ultra fast minimal prompt
    combined_prompt = f"""
    Analyze this user message and determine next actions.
    
    USER MESSAGE: {last_user_message}
    
    FORMAT YOUR RESPONSE EXACTLY AS FOLLOWS:
    
    [ANALYSIS]
    Brief analysis about the user's intent, needs, and emotional state. 
    [/ANALYSIS]
    
    [NEXT_ACTIONS]
    1. First action to take
    2. Second action to take
    3. Additional actions if needed
    [/NEXT_ACTIONS]
    """
    
    # Track prompt building time
    PERF_METRICS["prompt_built"] = time.time()
    logger.info(f"Ultra optimized minimal prompt built in {time.time() - start_time:.2f}s")
    
    # Track current section being processed
    current_section = None
    section_buffer = ""
    analysis_content = ""
    next_actions_content = ""
    
    # CRITICAL FIX: Set very short timeout for analysis
    ANALYSIS_TIMEOUT = 4  # 4 seconds maximum for analysis
    
    try:
        # Stream a starting event for the frontend to show something is happening
        # CRITICAL FIX: Format events exactly like knowledge events
        analysis_start_event = {
            "type": "analysis", 
            "content": "Analyzing your message...", 
            "complete": False,
            # Add these fields to match knowledge events format
            "status": "analyzing",
            "thread_id": thread_id
        }
        logger.info(f"Sending initial analysis event: {analysis_start_event}")
        yield analysis_start_event
        
        logger.info("Starting ultra-fast analysis")
        
        # FIX: Use a properly cancellable Task for the stream
        stream_task = None
        stream_gen = StreamLLM.astream(combined_prompt)
        
        try:
            # Process with timeout
            async with asyncio.timeout(ANALYSIS_TIMEOUT):
                async for chunk in stream_gen:
                    content = chunk.content
                    
                    # Process content to separate sections
                    for line in content.split('\n'):
                        if '[ANALYSIS]' in line:
                            current_section = 'analysis'
                            # Immediately send a starting event for the section
                            analysis_section_event = {
                                "type": "analysis", 
                                "content": "", 
                                "complete": False,
                                "thread_id": thread_id,
                                "status": "analyzing"
                            }
                            logger.info(f"Sending analysis section start event: {analysis_section_event}")
                            yield analysis_section_event
                            continue
                        elif '[/ANALYSIS]' in line:
                            # Complete analysis section
                            analysis_complete_event = {
                                "type": "analysis", 
                                "content": analysis_content, 
                                "complete": True,
                                "thread_id": thread_id,
                                "status": "complete"
                            }
                            logger.info(f"Sending analysis complete event with content length: {len(analysis_content)}")
                            yield analysis_complete_event
                            current_section = None
                            continue
                        elif '[NEXT_ACTIONS]' in line:
                            current_section = 'next_actions'
                            # Immediately send a starting event for the section
                            next_actions_start_event = {
                                "type": "next_actions", 
                                "content": "", 
                                "complete": False,
                                "thread_id": thread_id,
                                "status": "analyzing"
                            }
                            logger.info(f"Sending next_actions start event")
                            yield next_actions_start_event
                            continue
                        elif '[/NEXT_ACTIONS]' in line:
                            # Complete next actions section
                            next_actions_complete_event = {
                                "type": "next_actions", 
                                "content": next_actions_content, 
                                "complete": True,
                                "thread_id": thread_id,
                                "status": "complete"
                            }
                            logger.info(f"Sending next_actions complete event with content length: {len(next_actions_content)}")
                            yield next_actions_complete_event
                            current_section = None
                            continue
                        
                        # Add content to appropriate section and stream as we go
                        if current_section == 'analysis':
                            analysis_content += line + "\n"
                            # CRITICAL FIX: Stream every line immediately to frontend
                            analysis_line_event = {
                                "type": "analysis", 
                                "content": line, 
                                "complete": False,
                                "thread_id": thread_id,
                                "status": "analyzing"
                            }
                            logger.info(f"Sending analysis line event: {line[:30]}...")
                            yield analysis_line_event
                        elif current_section == 'next_actions':
                            next_actions_content += line + "\n"
                            # CRITICAL FIX: Stream every line immediately to frontend
                            next_actions_line_event = {
                                "type": "next_actions", 
                                "content": line, 
                                "complete": False,
                                "thread_id": thread_id,
                                "status": "analyzing"
                            }
                            logger.info(f"Sending next_actions line event: {line[:30]}...")
                            yield next_actions_line_event
        except asyncio.TimeoutError:
            logger.warning(f"Analysis timed out after {ANALYSIS_TIMEOUT}s, using partial results")
            # FIX: We don't need to and can't cancel a generator, it will be garbage collected
            # Just log the timeout and continue
            logger.info("Processing timed out, continuing with partial results")
        except Exception as e:
            logger.error(f"Error during streaming: {str(e)}")
        
        # Ensure we yield complete events for both sections if they weren't properly marked
        if not analysis_content or "[ANALYSIS]" in analysis_content:
            # Create simple analysis from user message
            simple_analysis = f"User is asking about: {last_user_message[:50]}"
            analysis_content = simple_analysis
            fallback_analysis_event = {
                "type": "analysis",
                "content": simple_analysis,
                "complete": True,
                "thread_id": thread_id,
                "status": "complete"
            }
            logger.info(f"Sending fallback analysis complete event: {simple_analysis}")
            yield fallback_analysis_event
        else:
            # Store complete analysis in state for later reference
            logger.info(f"Stored COMPLETE analysis in state: {analysis_content[:50]}...")
        
        if not next_actions_content or "[NEXT_ACTIONS]" in next_actions_content:
            # Create very basic next actions
            simple_actions = "1. Provide a helpful response\n2. Include relevant information\n3. Use appropriate tone"
            next_actions_content = simple_actions
            fallback_next_actions_event = {
                "type": "next_actions",
                "content": simple_actions,
                "complete": True,
                "thread_id": thread_id,
                "status": "complete"
            }
            logger.info(f"Sending fallback next_actions complete event")
            yield fallback_next_actions_event
        
        # Track completion time
        PERF_METRICS["combined_analysis_end"] = time.time()
        logger.info(f"Combined analysis completed in {time.time() - start_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Error in combined analysis: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Always provide at least some analysis and next actions
        simple_analysis = f"User is asking about: {last_user_message[:50]}"
        simple_actions = "1. Provide a helpful response\n2. Include relevant information\n3. Use appropriate tone"
        
        emergency_analysis_event = {
            "type": "analysis", 
            "content": simple_analysis, 
            "complete": True,
            "thread_id": thread_id,
            "status": "complete"
        }
        emergency_next_actions_event = {
            "type": "next_actions", 
            "content": simple_actions, 
            "complete": True,
            "thread_id": thread_id,
            "status": "complete"
        }
        
        logger.info("Sending emergency fallback events due to exception")
        yield emergency_analysis_event
        yield emergency_next_actions_event

# Replace the existing ensure_brain_loaded function with this fixed version
# that properly caches results of async functions, not the coroutines themselves
_brain_load_cache = {}

# Optimized knowledge query handler with batch processing and caching
async def knowledge_query_handler(params: Dict) -> AsyncGenerator[Dict, None]:
    """
    Optimized handler for knowledge query requests with improved batch processing.
    
    Args:
        params: Dictionary containing queries, graph_version_id, and optional top_k
        
    Yields:
        Dict events with streaming knowledge query results
    """
    start_time = time.time()
    PERF_METRICS["knowledge_query_start"] = start_time
    
    queries = params.get("queries", [])
    graph_version_id = params.get("graph_version_id", "")
    top_k = params.get("top_k", 3)
    thread_id = params.get("_thread_id")  # Internal param for WebSocket
    
    # OPTIMIZATION: Reduce number of queries and simplify
    # Limit number of queries to reduce processing time
    if len(queries) > 3:  # Reduce from 5 to 3 for better performance
        # Keep only the most specific queries (usually longer ones)
        queries = sorted(queries, key=len, reverse=True)[:3]
    
    # CRITICAL FIX: Send an initial event to frontend to show progress
    yield {"type": "knowledge", "content": [], "complete": False, "status": "searching"}
    
    # Handle empty queries
    if not queries:
        yield {"type": "knowledge", "content": [], "complete": True}
        PERF_METRICS["knowledge_query_end"] = time.time()
        return
        
    try:
        # OPTIMIZATION: Skip brain loading if it's taking too long
        # Set a very short timeout for brain loading
        try:
            async with asyncio.timeout(5.0):  # Increase from 3 to 5 second timeout
                brain_loaded = await ensure_brain_loaded(graph_version_id)
        except asyncio.TimeoutError:
            logger.warning("Brain loading timed out, continuing with limited functionality")
            brain_loaded = False
            
        if not brain_loaded:
            logger.error(f"Failed to load brain for version {graph_version_id}")
            yield {"type": "knowledge", "content": [], "complete": True, 
                   "error": f"Knowledge database unavailable at this time"}
            return
        
        # Get global brain instance
        global brain
        brain = get_brain()
        
        # OPTIMIZATION: Reduce batch size and top_k for faster processing
        # Use smaller top_k to reduce processing time
        optimized_top_k = min(top_k, 1)  # Limit to just 1 result per query for extreme speed
        
        # Process all queries at once for speed
        all_results = []
        seen_ids = set()
        
        # OPTIMIZATION: Set a timeout for the knowledge query
        KNOWLEDGE_TIMEOUT = 10  # Increase from 8 to 10 seconds
        
        try:
            # Process with timeout
            async with asyncio.timeout(KNOWLEDGE_TIMEOUT):
                try:
                    # Just use the first query for extreme speed
                    if queries:
                        # OPTIMIZATION: Just use first query
                        primary_query = queries[0]
                        logger.info(f"Using primary query only: {primary_query}")
                        
                        results = await brain.get_similar_vectors_by_text(primary_query, top_k=optimized_top_k)
                        
                        for vector_id, vector, metadata, similarity in results:
                            if vector_id not in seen_ids:
                                seen_ids.add(vector_id)
                                
                                # OPTIMIZATION: Simplify result format
                                # Only include essential fields
                                result = {
                                    "id": vector_id,
                                    "similarity": float(similarity),
                                    "raw": metadata.get("raw", ""),
                                    "query": primary_query,
                                }
                                all_results.append(result)
                        
                        # Stream results
                        if all_results:
                            yield {"type": "knowledge", "content": all_results, "complete": False}
                        
                except Exception as query_error:
                    logger.error(f"Primary query failed: {query_error}")
                
        except asyncio.TimeoutError:
            logger.warning(f"Knowledge query timed out after {KNOWLEDGE_TIMEOUT}s, using partial results")
        
        # Final complete event with stats
        yield {
            "type": "knowledge", 
            "content": all_results, 
            "complete": True, 
            "stats": {
                "total_results": len(all_results),
                "query_count": 1,  # We're only using one query now
                "graph_version": get_current_graph_version(),
                "processing_time": f"{time.time() - start_time:.2f}s"
            }
        }
        logger.info(f"Completed knowledge search with {len(all_results)} total results in {time.time() - start_time:.2f}s")
        PERF_METRICS["knowledge_query_end"] = time.time()
        
    except Exception as e:
        logger.error(f"Error in knowledge query: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        yield {"type": "knowledge", "content": [], "complete": True, "error": str(e)}
        PERF_METRICS["knowledge_query_end"] = time.time()

# Optimized response generation handler
async def response_generation_handler(params: Dict) -> AsyncGenerator[str, None]:
    """
    Optimized handler for response generation with user profile awareness.
    
    Args:
        params: Dictionary containing conversation_context, analysis, next_actions, 
                knowledge_context, personality_instructions, knowledge_found,
                and user_profile (optional)
                
    Yields:
        Streamed response chunks as strings
    """
    start_time = time.time()
    PERF_METRICS["response_generation_start"] = start_time
    
    conversation_context = params.get("conversation_context", "")
    analysis = params.get("analysis", "")
    next_actions = params.get("next_actions", "")
    knowledge_context = params.get("knowledge_context", "")
    personality_instructions = params.get("personality_instructions", "")
    knowledge_found = params.get("knowledge_found", False)
    user_profile = params.get("user_profile", _last_cot_results.get("user_profile", {}))
    
    # Get user profile information or use defaults
    language_preference = "vietnamese"
    
    # Extract basic language information from the portrait or fallback to detection
    if user_profile and "portrait" in user_profile:
        # Try to extract language from portrait
        portrait = user_profile.get("portrait", "")
        if "communicating in Vietnamese" in portrait:
            language_preference = "vietnamese"
        elif "communicating in English" in portrait:
            language_preference = "english"
        else:
            # Fall back to simple language detection
            language_preference = detect_language(conversation_context)
    else:
        # Fall back to simple language detection if no profile
        language_preference = detect_language(conversation_context)
    
    detected_language = language_preference
    
    # Get the user portrait to include in the prompt
    user_understanding = format_user_profile_for_prompt(user_profile) if user_profile else ""
    
    # Create simplified cultural instructions
    cultural_instructions = f"Respond in {detected_language}. Use appropriate tone and detail level based on the user understanding."
    
    # OPTIMIZATION: Simplify personality instructions if too long
    if personality_instructions and len(personality_instructions) > 500:
        personality_instructions = "You are a friendly and helpful AI assistant that responds appropriately to users' needs."
    
    # OPTIMIZATION: Drastically reduce knowledge context size
    if knowledge_context and len(knowledge_context) > 1500:
        # Extract only the most relevant parts
        knowledge_context = extract_key_knowledge(knowledge_context, conversation_context)
    
    # Get the last user message for better context
    last_message = ""
    for line in conversation_context.split('\n'):
        if line.startswith("User:"):
            last_message = line
    
    # OPTIMIZATION: Simplify the prompt structure but incorporate user understanding
    prompt = f"""
    # Context
    Latest message: {last_message}
    
    {user_understanding}
    
    # Analysis Summary
    {analysis[:300]}  
    
    # Actions Needed
    {next_actions[:300]}
    
    {f"# Knowledge\n{knowledge_context}" if knowledge_context else ""}
    
    # Personality
    {personality_instructions}
    
    # Task
    Create a helpful {detected_language} response that:
    1. Addresses the user's immediate need based on the user understanding
    2. Provides an appropriate level of detail and tone
    3. Uses knowledge effectively while maintaining conversational flow
    
    Begin your response immediately without preamble.
    """
    
    # OPTIMIZATION: Set a timeout for response generation
    RESPONSE_TIMEOUT = 7  # 7 seconds timeout
    
    # Response optimization components
    response_processor = ResponseProcessor()
    response_filter = ResponseFilter()
    
    # Check if this is the first message
    is_first_message = "User:" in conversation_context and conversation_context.count("User:") <= 1
    
    # OPTIMIZATION: Collect the full response with larger buffer size
    full_response = ""
    buffer = ""
    buffer_size = 30  # Increased buffer size for efficiency
    
    try:
        logger.info(f"Starting profile-aware response generation stream")
        PERF_METRICS["response_stream_start"] = time.time()
        
        # Set up streaming with timeout
        response_task = None  # Initialize to None to avoid reference errors
        
        try:
            # Process with timeout
            logger.info(f"Sending response generation request to OpenAI")
            async with asyncio.timeout(RESPONSE_TIMEOUT):
                async for chunk in StreamLLM.astream(prompt):
                    content = chunk.content
                    buffer += content
                    full_response += content
                    
                    # Log the first chunk received
                    if len(full_response) <= len(content):
                        logger.info(f"Received first response chunk from OpenAI: {content[:30]}...")
                    
                    # Only yield when buffer reaches threshold
                    if len(buffer) >= buffer_size or content.endswith((".", "!", "?", "\n")):
                        yield buffer
                        buffer = ""
        except asyncio.TimeoutError:
            logger.warning(f"Response generation timed out after {RESPONSE_TIMEOUT}s, using partial response")
            # Don't try to cancel the streaming task since it's not stored as a task
            # Just yield what we've got so far
            if not full_response:
                # If we have nothing, generate a simple response based on language
                if detected_language == "vietnamese":
                    full_response = "Tôi đang xử lý thông tin. Vui lòng cho tôi thêm chi tiết để hỗ trợ bạn tốt hơn."
                else:
                    full_response = "I'm processing your request. Could you provide more details so I can help you better?"
                yield full_response
        except Exception as e:
            logger.error(f"Error in response generation streaming: {str(e)}")
            logger.error(traceback.format_exc())
            # Fall back to language-specific error messages
            if not full_response:
                if detected_language == "vietnamese":
                    full_response = "Xin lỗi, tôi đang gặp sự cố khi xử lý yêu cầu của bạn. Vui lòng thử lại."
                else:
                    full_response = "I encountered an issue while generating a response. Let me try again."
                yield full_response
        
        # Yield any remaining buffer
        if buffer:
            yield buffer
        
        PERF_METRICS["response_stream_end"] = time.time()
        logger.info(f"Response streaming completed in {PERF_METRICS['response_stream_end'] - PERF_METRICS['response_stream_start']:.2f}s")
        
    except Exception as e:
        logger.error(f"Error in response generation: {str(e)}")
        # Fall back to language-specific error messages
        if detected_language == "vietnamese":
            yield "Xin lỗi, tôi đang gặp sự cố khi xử lý yêu cầu của bạn. Vui lòng thử lại."
        else:
            yield "I encountered an issue while generating a response. Let me try again."
    
    PERF_METRICS["response_generation_end"] = time.time()
    logger.info(f"Response generation completed in {time.time() - start_time:.2f}s")

# Add the Chain of Thought handler
async def cot_knowledge_analysis_actions_handler(params: Dict) -> AsyncGenerator[Dict, None]:
    """
    Chain-of-Thought handler that combines knowledge retrieval with analysis and next actions.
    Enhanced with user profiling for more targeted knowledge retrieval and analysis.
    
    Args:
        params: Dictionary containing conversation_context, graph_version_id
                
    Yields:
        Dict events with streaming results for knowledge, analysis, and next actions
    """
    start_time = time.time()
    PERF_METRICS["cot_handler_start"] = start_time
    
    # Access the global variable to store results
    global _last_cot_results
    _last_cot_results = {
        "analysis_content": "",
        "next_actions_content": "",
        "knowledge_entries": [],
        "knowledge_context": "",
        "user_profile": {}  # Add user profile to results
    }
    
    conversation_context = params.get("conversation_context", "")
    graph_version_id = params.get("graph_version_id", "")
    thread_id = params.get("_thread_id")
    
    # Extract user message for knowledge retrieval - IMPROVED to capture up to 30 recent messages
    # This better captures the conversation context and history
    recent_messages = []
    for line in conversation_context.strip().split('\n'):
        if line.startswith("User:") or line.startswith("AI:"):
            recent_messages.append(line)
    
    # Keep last 30 messages for richer context
    recent_messages = recent_messages[-30:] if len(recent_messages) > 30 else recent_messages
    
    # Get most recent user message for initial knowledge search
    last_user_message = ""
    for line in reversed(recent_messages):
        if line.startswith("User:"):
            last_user_message = line[5:].strip()
            break
    
    # Create a comprehensive context string from recent messages
    context_window = "\n".join(recent_messages)
    
    logger.info(f"Starting CoT processing with context of {len(recent_messages)} messages")
    logger.info(f"Most recent query: '{last_user_message[:30]}...'")
    
    # 1. First, emit initial events to show progress
    # Send analysis starting event
    analysis_start_event = {
        "type": "analysis",
        "content": "Analyzing your message and building knowledge-enhanced profile...",
        "complete": False,
        "thread_id": thread_id,
        "status": "analyzing"
    }
    logger.info(f"Sending initial CoT analysis event: {analysis_start_event}")
    yield analysis_start_event
    
    # NEW STEP: Build knowledge-enhanced user profile
    logger.info("Building knowledge-enhanced user profile")
    user_profile = await build_user_profile(context_window, last_user_message, graph_version_id)
    _last_cot_results["user_profile"] = user_profile  # Store for future use
    
    # Log profile enhancement results
    logger.info(f"User profile built with method: {user_profile.get('method', 'unknown')}")
    logger.info(f"Profile has {user_profile.get('knowledge_sources', 0)} knowledge sources")
    
    # Generate enhanced search queries based on the knowledge-enhanced profile
    emotional_states = user_profile.get("emotional_state", {}).get("current", [])
    emotional_state = emotional_states[0] if emotional_states else "neutral"
    #enhanced_queries = generate_profile_enhanced_queries(last_user_message, user_profile)
    enhanced_queries = []
    logger.info(f"Generated {len(enhanced_queries)} knowledge-enhanced queries: {enhanced_queries}")
    
    # Send knowledge search starting event
    knowledge_start_event = {
        "type": "knowledge",
        "content": [],
        "complete": False,
        "status": "searching",
        "thread_id": thread_id
    }
    logger.info(f"Sending knowledge start event for CoT: {knowledge_start_event}")
    yield knowledge_start_event
    
    # 2. Retrieve knowledge using profile-enhanced queries
    knowledge_entries = []
    structured_knowledge = []  # To store extracted structured data
    knowledge_context = ""
    
    try:
        # Make sure brain is loaded
        brain_loaded = False
        try:
            # Set short timeout for brain loading
            async with asyncio.timeout(5.0):  # Increase from 3 to 5 seconds
                brain_loaded = await ensure_brain_loaded(graph_version_id)
        except asyncio.TimeoutError:
            logger.warning("Brain loading timed out in CoT, continuing with limited functionality")
            brain_loaded = False
            
        if brain_loaded:
            # Get global brain instance
            global brain
            brain = get_brain()
            
            # Use profile-enhanced queries for more targeted search
            # Use a short timeout for knowledge query
            KNOWLEDGE_TIMEOUT = 10  # Increase from 8 to 10 seconds
            
            try:
                # Process with timeout
                async with asyncio.timeout(KNOWLEDGE_TIMEOUT):
                    # HYBRID APPROACH - Phase 1: Profile-enhanced knowledge retrieval
                    logger.info(f"CoT PHASE 1: Profile-enhanced knowledge queries: {enhanced_queries[:2]}")
                    
                    # Use top 2 profile-enhanced queries for better results
                    for query_idx, enhanced_query in enumerate(enhanced_queries[:2]):
                        initial_results = await brain.get_similar_vectors_by_text(enhanced_query, top_k=3)
                        
                        # Process initial knowledge results
                        for vector_id, vector, metadata, similarity in initial_results:
                            # Skip duplicates
                            if any(entry.get("id") == vector_id for entry in knowledge_entries):
                                continue
                                
                            raw_text = metadata.get("raw", "")
                            structured_data = extract_structured_data_from_raw(raw_text)
                            
                            entry = {
                                "id": vector_id,
                                "similarity": float(similarity),
                                "raw": raw_text,
                                "structured": structured_data,
                                "query": enhanced_query,
                                "phase": "initial",
                                "profile_match": True  # Mark as profile-matched
                            }
                            knowledge_entries.append(entry)
                            
                            if structured_data:
                                structured_knowledge.append(structured_data)
                    
                    # Stream initial knowledge results
                    if knowledge_entries:
                        knowledge_event = {
                            "type": "knowledge",
                            "content": knowledge_entries,
                            "complete": False,
                            "thread_id": thread_id,
                            "status": "searching"
                        }
                        logger.info(f"CoT PHASE 1: Found {len(knowledge_entries)} initial knowledge entries with profile-enhanced queries")
                        yield knowledge_event
                    
                    # HYBRID APPROACH - Phase 2: Initial Analysis with Profile + Knowledge
                    # Perform quick analysis on initial results before expanded search
                    analysis_from_initial = ""
                    if knowledge_entries:
                        # Extract key insights based on user profile and initial knowledge
                        segment = user_profile["segment"]["category"]
                        emotion = emotional_state
                        
                        # Generate a brief analysis text based on profile and initial knowledge
                        analysis_from_initial = f"User appears to be in {segment} segment with {emotion} emotional state. "
                        
                        # Extract relevant topics from knowledge
                        topics = []
                        for entry in knowledge_entries:
                            if entry.get("structured", {}).get("title"):
                                topics.append(entry["structured"]["title"])
                        
                        if topics:
                            analysis_from_initial += f"Initial knowledge suggests interest in: {', '.join(topics[:3])}"
                        
                        logger.info(f"Generated initial analysis for Phase 2: {analysis_from_initial}")
                    
                    # HYBRID APPROACH - Phase 3: Expanded Knowledge with Analysis + Profile
                    if knowledge_entries:
                        # Extract key concepts from initial knowledge to inform expanded search
                        search_terms = []
                        key_concepts = set()
                        
                        # 1. First extract from cluster connections with enhanced parsing for Vietnamese
                        for entry in knowledge_entries:
                            structured = entry.get("structured", {})
                            if structured and "cross_cluster_connections" in structured:
                                connections_text = structured["cross_cluster_connections"]
                                
                                # Try to extract cluster references (e.g., "Cluster 8") for Vietnamese
                                cluster_refs = re.findall(r'Cluster (\d+)', connections_text)
                                for cluster_num in cluster_refs[:3]:
                                    key_concepts.add(f"Cluster {cluster_num}")
                                
                                # Extract phrases that might be between quotes
                                key_phrases = re.findall(r'"([^"]+)"|"([^"]+)"', connections_text)
                                for phrase in key_phrases[:2]:
                                    if isinstance(phrase, tuple):
                                        phrase = next((p for p in phrase if p), "")
                                    if phrase:
                                        key_concepts.add(phrase)
                                
                                # If normal splitting would work better for other languages
                                if not cluster_refs and not key_phrases:
                                    concepts = connections_text.split(", ")
                                    for concept in concepts[:3]:
                                        key_concepts.add(concept)
                        
                        # 2. Then extract titles and themes
                        for entry in knowledge_entries:
                            structured = entry.get("structured", {})
                            if structured and "title" in structured:
                                key_concepts.add(structured["title"])
                        
                        # 3. Add key concepts as search terms
                        search_terms = list(key_concepts)[:3]
                        
                        # 4. Create search term from initial analysis
                        if analysis_from_initial:
                            # Extract the most information-rich part
                            if "interest in:" in analysis_from_initial:
                                interests_part = analysis_from_initial.split("interest in:")[1].strip()
                                search_terms.append(interests_part)
                            
                        # 5. Add user profile information as search terms
                        profile_term = ""
                        knowledge_areas = user_profile.get("query_characteristics", {}).get("knowledge_areas", [])
                        if knowledge_areas:
                            profile_term += " ".join(knowledge_areas)
                        if user_profile["segment"]["category"] != "general":
                            profile_term += f" {user_profile['segment']['category']}"
                        if profile_term:
                            search_terms.append(f"{last_user_message} {profile_term}")
                        
                        # 6. Also generate a search term based on context window
                        if context_window:
                            context_query = " ".join(recent_messages[-3:])
                            if context_query and context_query != last_user_message:
                                search_terms.append(context_query)
                        
                        logger.info(f"CoT PHASE 3: Expanded search with terms: {search_terms}")
                        
                        # Track existing IDs to avoid duplicates
                        existing_ids = {entry["id"] for entry in knowledge_entries}
                        
                        # Execute expanded searches
                        for term in search_terms:
                            expanded_results = await brain.get_similar_vectors_by_text(term, top_k=1)
                            
                            # Process additional knowledge results
                            for vector_id, vector, metadata, similarity in expanded_results:
                                # Skip if we already have this entry
                                if vector_id in existing_ids:
                                    continue
                                    
                                existing_ids.add(vector_id)
                                raw_text = metadata.get("raw", "")
                                structured_data = extract_structured_data_from_raw(raw_text)
                                
                                entry = {
                                    "id": vector_id,
                                    "similarity": float(similarity),
                                    "raw": raw_text,
                                    "structured": structured_data,
                                    "query": term,
                                    "phase": "expanded"
                                }
                                knowledge_entries.append(entry)
                                
                                if structured_data:
                                    structured_knowledge.append(structured_data)
                        
                        # Stream the expanded results
                        if len(knowledge_entries) > 0:
                            expanded_event = {
                                "type": "knowledge",
                                "content": knowledge_entries,
                                "complete": False,
                                "thread_id": thread_id,
                                "status": "searching"
                            }
                            
                            logger.info(f"CoT PHASE 3: Additional knowledge found, now {len(knowledge_entries)} total entries")
                            yield expanded_event
            
            except asyncio.TimeoutError:
                logger.warning(f"CoT knowledge query timed out after {KNOWLEDGE_TIMEOUT}s")
                # Ensure Phase 3 properly completes even with timeout
                if knowledge_entries:
                    yield {
                        "type": "knowledge",
                        "content": knowledge_entries,
                        "complete": True,
                        "thread_id": thread_id,
                        "status": "complete",
                        "stats": {"total_results": len(knowledge_entries)},
                        "note": "Partial results due to timeout"
                    }
            except Exception as e:
                logger.error(f"Error in CoT knowledge query: {e}")
                # Ensure Phase 3 completes with error status
                yield {
                    "type": "knowledge",
                    "content": knowledge_entries,
                    "complete": True,
                    "thread_id": thread_id,
                    "status": "complete",
                    "stats": {"total_results": len(knowledge_entries)},
                    "error": f"Error in knowledge retrieval: {str(e)}"
                }
            
            # Send knowledge complete event
            yield {
                "type": "knowledge",
                "content": knowledge_entries,
                "complete": True,
                "thread_id": thread_id,
                "status": "complete",
                "stats": {"total_results": len(knowledge_entries)}
            }
            
            # Format knowledge context for the prompt using the optimized function
            if knowledge_entries:
                # Use the new optimized formatter rather than the old approach
                knowledge_context = optimize_knowledge_context(knowledge_entries, last_user_message, max_chars=2500)
                logger.info(f"Created optimized knowledge context: {len(knowledge_context)} chars")
            else:
                # Fall back to empty knowledge context if no entries found
                knowledge_context = ""
                logger.warning("No knowledge entries found, knowledge context will be empty")
            
            # Now perform combined analysis and next actions with knowledge context and user profile
            current_section = None
            analysis_content = ""
            next_actions_content = ""
            
            # Build an enhanced CoT prompt with structured knowledge and user profile
            # Format the user profile for the prompt
            profile_summary = format_user_profile_for_prompt(user_profile)
            logger.info(f"User profile for CoT: {profile_summary}")
            
            cot_prompt = f"""
            You are an AI assistant using Implicit Chain-of-Thought reasoning to help users effectively.
            
            CONVERSATION CONTEXT:
            {context_window}
            
            {profile_summary}
            
            {f"RELEVANT KNOWLEDGE:\n{knowledge_context}" if knowledge_context else "NO RELEVANT KNOWLEDGE FOUND."}
            
            Based on the conversation context, user profile, and provided knowledge:
            
            [ANALYSIS]
            Analyze the user's core needs based on their message and the user understanding provided above.
            Consider the context of their query, their language preferences, communication style, and emotional state.
            
            Address how the knowledge can specifically help this user with their query.
            
            If this is a health-related concern, especially regarding sexual health, consider the sensitivity and importance of this topic to the user, potential embarrassment, and psychological impacts.
            [/ANALYSIS]
            
            [NEXT_ACTIONS]
            1. Most important action tailored to this user profile
            2. Secondary action considering their communication preferences  
            3. Additional action if needed based on their emotional state
            
            For health topics, especially sensitive ones like sexual health:
            - Prioritize factual medical information
            - Provide empathetic and non-judgmental responses
            - Consider suggesting professional medical consultation when appropriate
            - Maintain a respectful and compassionate tone throughout
            
            Each action should be concise, practical, and directly applicable to this specific user.
            [/NEXT_ACTIONS]
            """
            
            # Set timeout for combined analysis
            COT_TIMEOUT = 30  # seconds
            
            logger.info("Starting enhanced CoT LLM stream with structured knowledge and user profile")
            try:
                # Stream the CoT analysis with timeout
                stream_gen = StreamLLM.astream(cot_prompt)
                
                try:
                    # Process with timeout
                    async with asyncio.timeout(COT_TIMEOUT):
                        async for chunk in stream_gen:
                            content = chunk.content
                            
                            # Process content to separate sections
                            for line in content.split('\n'):
                                if '[ANALYSIS]' in line:
                                    current_section = 'analysis'
                                    # Send analysis section start event
                                    yield {
                                        "type": "analysis",
                                        "content": "",
                                        "complete": False,
                                        "thread_id": thread_id,
                                        "status": "analyzing"
                                    }
                                    continue
                                elif '[/ANALYSIS]' in line:
                                    # Complete analysis section
                                    analysis_complete_event = {
                                        "type": "analysis",
                                        "content": analysis_content,
                                        "complete": True,
                                        "thread_id": thread_id,
                                        "status": "complete",
                                        "search_terms": [],
                                        "user_profile": user_profile  # Include user profile in result
                                    }
                                    yield analysis_complete_event
                                    current_section = None
                                    continue
                                elif '[NEXT_ACTIONS]' in line:
                                    current_section = 'next_actions'
                                    # Send next_actions section start event
                                    yield {
                                        "type": "next_actions",
                                        "content": "",
                                        "complete": False,
                                        "thread_id": thread_id,
                                        "status": "analyzing"
                                    }
                                    continue
                                elif '[/NEXT_ACTIONS]' in line:
                                    # Complete next_actions section
                                    next_actions_complete_event = {
                                        "type": "next_actions",
                                        "content": next_actions_content,
                                        "complete": True,
                                        "thread_id": thread_id,
                                        "status": "complete",
                                        "user_profile": user_profile  # Include user profile in result
                                    }
                                    yield next_actions_complete_event
                                    current_section = None
                                    continue
                                
                                # Add content to appropriate section and stream
                                if current_section == 'analysis':
                                    analysis_content += line + "\n"
                                    # Stream analysis line
                                    yield {
                                        "type": "analysis",
                                        "content": line,
                                        "complete": False,
                                        "thread_id": thread_id,
                                        "status": "analyzing"
                                    }
                                elif current_section == 'next_actions':
                                    next_actions_content += line + "\n"
                                    # Stream next_actions line
                                    yield {
                                        "type": "next_actions",
                                        "content": line,
                                        "complete": False,
                                        "thread_id": thread_id,
                                        "status": "analyzing" 
                                    }
                
                except asyncio.TimeoutError:
                    logger.warning(f"CoT analysis timed out after {COT_TIMEOUT}s, using partial results")
                    logger.info("CoT processing timed out, continuing with partial results")
                    
                except Exception as e:
                    logger.error(f"Error in CoT analysis streaming: {e}")
                    logger.error(traceback.format_exc())
                
                # Now check the module-level variable for results if needed
                if not analysis_content:
                    # Instead of creating a simple fallback, use the actual user profile to create a more detailed analysis
                    if user_profile and "portrait" in user_profile:
                        # Extract key information from the portrait
                        portrait = user_profile.get("portrait", "")
                        
                        # Create a more informative analysis based on the user profile
                        analysis_content = f"Based on user profile: {portrait[:300]}..."
                    else:
                        # Only if no user profile is available, fall back to simple analysis
                        analysis_content = f"User is asking about: {last_user_message[:100]}"
                    
                    # Create a proper analysis event with the full user profile
                    fallback_analysis_event = {
                        "type": "analysis",
                        "content": analysis_content,
                        "complete": True,
                        "thread_id": thread_id,
                        "status": "complete",
                        "search_terms": [],
                        "user_profile": user_profile  # Always include the full user profile
                    }
                    logger.info(f"Sending enhanced fallback analysis with full user profile")
                    
                    # Use socketio_manager directly for WebSocket events
                    if thread_id:
                        try:
                            from socketio_manager import emit_analysis_event
                            was_delivered = emit_analysis_event(thread_id, fallback_analysis_event)
                        except Exception as e:
                            logger.error(f"Error emitting fallback analysis: {str(e)}")
                            yield fallback_analysis_event
                
                if not next_actions_content:
                    # Create more specific fallback actions based on the user profile
                    fallback_actions = []
                    
                    # If we have a user profile with a portrait, use it to create more tailored actions
                    if user_profile and "portrait" in user_profile:
                        # Extract classification info if available in the portrait
                        portrait = user_profile.get("portrait", "")
                        
                        if "Discouraged Group" in portrait:
                            fallback_actions.append("1. Provide reassurance and supportive information to build confidence")
                            fallback_actions.append("2. Offer practical strategies and techniques addressing the specific issue")
                            fallback_actions.append("3. Suggest professional resources while maintaining a compassionate tone")
                        elif "Confident Group" in portrait:
                            fallback_actions.append("1. Provide motivational information to build on existing confidence")
                            fallback_actions.append("2. Offer advanced strategies and techniques for further improvement")
                            fallback_actions.append("3. Suggest ways to maintain and enhance current capabilities")
                        else:
                            # Generic but still supportive actions
                            fallback_actions.append("1. Provide clear and relevant information based on user query")
                            fallback_actions.append("2. Offer helpful options and practical next steps")
                            fallback_actions.append("3. Maintain a supportive tone appropriate to the user's needs")
                    else:
                        # Very basic fallback if no profile available
                        fallback_actions.append("1. Provide clear and relevant information based on user query")
                        fallback_actions.append("2. Offer helpful options and next steps")
                        fallback_actions.append("3. Maintain a supportive tone appropriate to the user's needs")
                    
                    # Format the fallback actions
                    next_actions_content = "\n".join(fallback_actions)
                    
                    fallback_next_actions_event = {
                        "type": "next_actions",
                        "content": next_actions_content,
                        "complete": True,
                        "thread_id": thread_id,
                        "status": "complete",
                        "user_profile": user_profile  # Always include the full user profile
                    }
                    logger.info(f"Sending enhanced fallback next_actions with full user profile")
                    
                    # Use socketio_manager directly for WebSocket events
                    if thread_id:
                        try:
                            from socketio_manager import emit_next_action_event
                            was_delivered = emit_next_action_event(thread_id, fallback_next_actions_event)
                        except Exception as e:
                            logger.error(f"Error emitting fallback next_actions: {str(e)}")
                            yield fallback_next_actions_event
                    else:
                        yield fallback_next_actions_event
                
                # Before completing sections, store results in the module-level variable
                if analysis_content:
                    _last_cot_results["analysis_content"] = analysis_content
                
                if next_actions_content:
                    _last_cot_results["next_actions_content"] = next_actions_content
                
                if knowledge_entries:
                    _last_cot_results["knowledge_entries"] = knowledge_entries
                    _last_cot_results["knowledge_context"] = knowledge_context
                
                _last_cot_results["user_profile"] = user_profile
                
                # Track completion time
                PERF_METRICS["cot_handler_end"] = time.time()
                total_time = PERF_METRICS["cot_handler_end"] - PERF_METRICS["cot_handler_start"]
                logger.info(f"CoT handler completed in {total_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Error in CoT processing: {e}")
                logger.error(traceback.format_exc())
                
                # Create fallback content
                if not analysis_content:
                    emotional_state_text = ', '.join(user_profile['emotional_state']['current']) if user_profile['emotional_state']['current'] else 'neutral'
                    analysis_content = f"User with {emotional_state_text} emotional state is asking about: {last_user_message[:30]}"
                    fallback_analysis_event = {
                        "type": "analysis",
                        "content": analysis_content,
                        "complete": True,
                        "thread_id": thread_id,
                        "status": "complete",
                        "search_terms": [],
                        "user_profile": user_profile  # Always include the full user profile
                    }
                    logger.info(f"Sending enhanced fallback analysis with full user profile")
                    
                    # Use socketio_manager directly for WebSocket events
                    if thread_id:
                        try:
                            from socketio_manager import emit_analysis_event
                            was_delivered = emit_analysis_event(thread_id, fallback_analysis_event)
                        except Exception as e:
                            logger.error(f"Error emitting fallback analysis: {str(e)}")
                            yield fallback_analysis_event
                
                if not next_actions_content:
                    next_actions_results = "1. Provide helpful information\n2. Use appropriate tone"
                    fallback_next_actions = {
                        "type": "next_actions", 
                        "content": next_actions_results, 
                        "complete": True,
                        "thread_id": thread_id,
                        "status": "complete",
                        "user_profile": user_profile  # Always include the full user profile
                    }
                    logger.info(f"Sending enhanced fallback next_actions with full user profile")
                    
                    # Use socketio_manager directly for WebSocket events
                    if thread_id:
                        try:
                            from socketio_manager import emit_next_action_event
                            was_delivered = emit_next_action_event(thread_id, fallback_next_actions)
                        except Exception as e:
                            logger.error(f"Error emitting fallback next_actions: {str(e)}")
                            yield fallback_next_actions
                
                # Yield the final state
                yield {"state": _last_cot_results}
        else:
            # Send empty knowledge complete event if brain not loaded
            yield {
                "type": "knowledge",
                "content": [],
                "complete": True,
                "thread_id": thread_id,
                "status": "complete",
                "error": "Could not access knowledge database"
            }
    except Exception as e:
        logger.error(f"Error in CoT knowledge retrieval: {e}")
        # Send knowledge error event
        yield {
            "type": "knowledge",
            "content": [],
            "complete": True,
            "thread_id": thread_id,
            "status": "complete",
            "error": f"Error retrieving knowledge: {str(e)}"
        }

# Register the new CoT tool
tool_registry.register_tool(
    "cot_knowledge_analysis_tool",
    cot_knowledge_analysis_actions_handler,
    COT_KNOWLEDGE_ANALYSIS_SCHEMA
)

# Now modify the process_llm_with_tools function to use the new CoT handler
async def process_llm_with_tools(
    user_message: str,
    conversation_history: List[Dict],
    state: Dict,
    graph_version_id: str,
    thread_id: Optional[str] = None
) -> AsyncGenerator[Union[str, Dict], None]:
    """
    Process a user message using LLM with tool calling.
    Optimized implementation with more parallel processing for better performance.
    
    Args:
        user_message: The user's message
        conversation_history: List of previous messages
        state: Current conversation state
        graph_version_id: Version ID for the knowledge graph
        thread_id: Optional thread ID for WebSocket streaming
        
    Yields:
        Tool results and final response as they become available
    """
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    import traceback
    
    overall_start_time = time.time()
    PERF_METRICS.clear()  # Reset metrics for this request
    PERF_METRICS["overall_start"] = overall_start_time
    
    # CRITICAL OPTIMIZATION: Just use the current message and last AI message for context
    # This drastically reduces the context size
    last_ai_message = ""
    
    if conversation_history:
        for msg in reversed(conversation_history):
            if msg.get("role") == "assistant":
                last_ai_message = msg.get("content", "")
                break
    
    conversation_context = ""
    if last_ai_message:
        conversation_context += f"AI: {last_ai_message}\n"
    conversation_context += f"User: {user_message}\n"
    
    # Track history processing time
    PERF_METRICS["history_processed"] = time.time()
    logger.info(f"Conversation history processed in {time.time() - overall_start_time:.2f}s")
    
    use_websocket = thread_id is not None
    
    # Track events for debugging
    analysis_events_sent = 0
    knowledge_events_sent = 0
    next_actions_events_sent = 0
    
    # Reset the last CoT results
    global _last_cot_results
    _last_cot_results = {
        "analysis_content": "",
        "next_actions_content": "",
        "knowledge_entries": [],
        "knowledge_context": ""
    }
    
    try:
        # CRITICAL OPTIMIZATION: Skip personality loading completely
        personality_instructions = """
        You are a friendly and helpful AI assistant with these key traits:
        - Warm and professional tone
        - Clear and concise communication
        - Helpful and informative responses
        """
        
        # CRITICAL FIX: Send an initial event to the frontend
        status_event = {
            "type": "status", 
            "status": "processing", 
            "message": "Processing your request..."
        }
        logger.info(f"Sending initial status event: {status_event}")
        yield status_event
        
        # ==================== NEW CODE: USE CoT APPROACH ====================
        # Use the new CoT handler for combined knowledge retrieval, analysis, and next actions
        logger.info("Using Chain-of-Thought approach for combined knowledge and analysis")
        
        # Setup CoT parameters
        cot_params = {
            "conversation_context": conversation_context,
            "graph_version_id": graph_version_id
        }
        if thread_id:
            cot_params["_thread_id"] = thread_id
        
        # Process all events from the CoT handler
        analysis_results = ""
        next_actions_results = ""
        knowledge_context = ""
        knowledge_entries = []
        knowledge_found = False
        
        # Forward all events from the CoT handler
        try:
            async for result in cot_knowledge_analysis_actions_handler(cot_params):
                # Track event types
                if result.get("type") == "analysis":
                    analysis_events_sent += 1
                elif result.get("type") == "knowledge":
                    knowledge_events_sent += 1
                elif result.get("type") == "next_actions":
                    next_actions_events_sent += 1
                
                # CRITICAL FIX: Ensure each event has the thread_id for socket.io routing
                if thread_id and "thread_id" not in result:
                    result["thread_id"] = thread_id
                
                # Log the event being forwarded
                event_type = result.get("type", "unknown")
                is_complete = result.get("complete", False)
                event_content_preview = str(result.get("content", ""))[:30]
                logger.info(f"Forwarding CoT {event_type} event (complete: {is_complete}): {event_content_preview}...")
                
                # CRITICAL FIX: Use socketio_manager directly for WebSocket events
                if use_websocket:
                    try:
                        # Import the correct emit function based on event type
                        if event_type == "analysis":
                            from socketio_manager import emit_analysis_event
                            logger.info(f"Emitting analysis event to thread {thread_id} (1 active sessions)")
                            was_delivered = emit_analysis_event(thread_id, result)
                            logger.info(f"Emitted analysis event via socketio_manager for thread {thread_id}")
                        elif event_type == "knowledge":
                            from socketio_manager import emit_knowledge_event
                            logger.info(f"Emitting knowledge event to thread {thread_id} (1 active sessions)")
                            was_delivered = emit_knowledge_event(thread_id, result)
                            logger.info(f"Emitted knowledge event via socketio_manager for thread {thread_id}")
                        elif event_type == "next_actions":
                            from socketio_manager import emit_next_action_event
                            logger.info(f"Emitting next_actions event to thread {thread_id}")
                            was_delivered = emit_next_action_event(thread_id, result)
                            logger.info(f"Emitted next_actions event via socketio_manager for thread {thread_id}")
                        else:
                            # For other event types, just yield them
                            yield result
                    except Exception as e:
                        logger.error(f"Error emitting {event_type} event via socketio_manager: {str(e)}")
                        logger.error(traceback.format_exc())
                        # Fall back to yielding the event
                        yield result
                else:
                    # Not using WebSockets, just yield the event
                    yield result
                
                # Collect results as they come (for backward compatibility)
                if isinstance(result, dict):
                    # Collect analysis results
                    if result.get("type") == "analysis" and result.get("complete", True):
                        analysis_results = result.get("content", "")
                        # Store user profile if available in the analysis result
                        if "user_profile" in result:
                            _last_cot_results["user_profile"] = result["user_profile"]
                    
                    # Collect next_actions results
                    elif result.get("type") == "next_actions" and result.get("complete", True):
                        next_actions_results = result.get("content", "")
                        # Store user profile if available in the next_actions result
                        if "user_profile" in result:
                            _last_cot_results["user_profile"] = result["user_profile"]
                    
                    # Collect knowledge entries
                    elif result.get("type") == "knowledge" and not result.get("complete", False):
                        entries = result.get("content", [])
                        if isinstance(entries, list):
                            knowledge_entries.extend(entries)
        
        except Exception as cot_error:
            logger.error(f"Error in CoT processing: {cot_error}")
            logger.error(traceback.format_exc())
            
            # Make sure we have the user profile even if CoT fails
            if not _last_cot_results.get("user_profile") and "user_profile" in state:
                _last_cot_results["user_profile"] = state["user_profile"]
            
            # Create fallback content
            if not analysis_results:
                # Create a meaningful analysis from the user profile if available
                if _last_cot_results.get("user_profile") and "portrait" in _last_cot_results["user_profile"]:
                    portrait = _last_cot_results["user_profile"]["portrait"]
                    analysis_results = f"Based on user profile: {portrait[:300]}..."
                else:
                    analysis_results = f"User is asking about: {user_message[:100]}"
                
                fallback_analysis_event = {
                    "type": "analysis", 
                    "content": analysis_results, 
                    "complete": True,
                    "thread_id": thread_id,
                    "status": "complete",
                    "search_terms": [],  # Add search_terms to match expected format
                    "user_profile": _last_cot_results.get("user_profile", {})  # Include user profile
                }
                logger.info(f"Sending enhanced fallback analysis with full user profile")
                
                # CRITICAL FIX: Use socketio_manager directly for WebSocket events
                if use_websocket:
                    try:
                        from socketio_manager import emit_analysis_event
                        was_delivered = emit_analysis_event(thread_id, fallback_analysis_event)
                    except Exception as e:
                        logger.error(f"Error emitting fallback analysis: {str(e)}")
                        yield fallback_analysis_event
            
            analysis_events_sent += 1
            
            if not next_actions_results:
                # Create more tailored next actions based on user profile if available
                if _last_cot_results.get("user_profile") and "portrait" in _last_cot_results["user_profile"]:
                    portrait = _last_cot_results["user_profile"]["portrait"]
                    if "Discouraged Group" in portrait:
                        next_actions_results = "1. Provide reassurance and supportive information\n2. Offer practical strategies for improvement\n3. Suggest professional resources while maintaining empathy"
                    elif "Confident Group" in portrait:
                        next_actions_results = "1. Provide motivational information\n2. Offer advanced strategies and techniques\n3. Suggest ways to maintain progress"
                    else:
                        next_actions_results = "1. Provide helpful information\n2. Offer practical next steps\n3. Use supportive tone"
                else:
                    next_actions_results = "1. Provide helpful information\n2. Use appropriate tone\n3. Offer practical solutions"
                
                fallback_next_actions = {
                    "type": "next_actions", 
                    "content": next_actions_results, 
                    "complete": True,
                    "thread_id": thread_id,
                    "status": "complete",
                    "user_profile": _last_cot_results.get("user_profile", {})  # Include user profile
                }
                logger.info(f"Sending enhanced fallback next_actions with full user profile")
                
                # CRITICAL FIX: Use socketio_manager directly for WebSocket events
                if use_websocket:
                    try:
                        from socketio_manager import emit_next_action_event
                        was_delivered = emit_next_action_event(thread_id, fallback_next_actions)
                    except Exception as e:
                        logger.error(f"Error emitting fallback next_actions: {str(e)}")
                        yield fallback_next_actions
        
        # Store the analysis in the state as a dictionary
        if analysis_results:
            # Create a structured analysis object for the state
            analysis_dict = {
                "content": analysis_results,
                "user_profile": _last_cot_results.get("user_profile", {})
            }
            state["analysis"] = analysis_dict
            logger.info("Final analysis stored in state as a dictionary")
        
        # IMPORTANT: Restore response generation code
        # Now check the module-level variable for results if needed
        if not analysis_results and _last_cot_results["analysis_content"]:
            analysis_results = _last_cot_results["analysis_content"]
            logger.info(f"Using analysis from _last_cot_results: {analysis_results[:50]}...")
        
        if not next_actions_results and _last_cot_results["next_actions_content"]:
            next_actions_results = _last_cot_results["next_actions_content"]
            logger.info(f"Using next actions from _last_cot_results")
        
        # For knowledge, prefer what we've collected during streaming, but fall back to module var
        if not knowledge_entries and _last_cot_results["knowledge_entries"]:
            knowledge_entries = _last_cot_results["knowledge_entries"]
            logger.info(f"Using {len(knowledge_entries)} knowledge entries from _last_cot_results")
        
        # Format knowledge context from entries
        if knowledge_entries:
            # First check if we have a pre-formatted knowledge context from _last_cot_results
            if _last_cot_results["knowledge_context"]:
                knowledge_context = _last_cot_results["knowledge_context"]
                logger.info(f"Using pre-formatted structured knowledge context from _last_cot_results")
            else:
                # Use our optimized formatter
                knowledge_context = optimize_knowledge_context(knowledge_entries, user_message, max_chars=2500)
                logger.info(f"Created optimized knowledge context: {len(knowledge_context)} chars")
                
            knowledge_found = True
            logger.info(f"Using knowledge context for response generation: {len(knowledge_context)} chars")
        
        # Log completion of CoT processing
        PERF_METRICS["cot_processing_end"] = time.time()
        cot_time = PERF_METRICS["cot_processing_end"] - PERF_METRICS["history_processed"]
        logger.info(f"CoT processing completed in {cot_time:.2f}s")
        
        # Generate response
        # Let the user know we're generating a response
        response_status = {
            "type": "status", 
            "status": "generating", 
            "message": "Generating response..."
        }
        logger.info(f"Sending response generation status: {response_status}")
        yield response_status
        
        response_params = {
            "conversation_context": conversation_context,
            "analysis": analysis_results,
            "next_actions": next_actions_results,
            "knowledge_context": knowledge_context,
            "personality_instructions": personality_instructions,
            "knowledge_found": knowledge_found,
            "user_profile": _last_cot_results.get("user_profile", {})  # Add user profile to response params
        }
        
        # Add debug logs
        logger.info(f"Starting response generation with params: analysis={len(analysis_results)} chars, next_actions={len(next_actions_results)} chars, knowledge={len(knowledge_context)} chars")
        if _last_cot_results.get("user_profile"):
            # Try to extract key information from the user_profile for logging
            if "portrait" in _last_cot_results["user_profile"]:
                portrait = _last_cot_results["user_profile"]["portrait"]
                # Check for classification in the portrait
                if "Discouraged Group" in portrait:
                    user_classification = "Discouraged Group"
                elif "Confident Group" in portrait:
                    user_classification = "Confident Group"
                else:
                    user_classification = "Unclassified"
                logger.info(f"Including user profile in response generation with classification: {user_classification}")
            else:
                logger.info(f"Including user profile in response generation")
        
        # Stream the response
        response_buffer = ""
        try:
            async for chunk in response_generation_handler(response_params):
                # Response chunks are streamed as strings
                if isinstance(chunk, str):
                    response_buffer += chunk
                    yield chunk
                    # Log chunks as they come in
                    logger.info(f"Response chunk received: {chunk[:30]}...")
        except Exception as response_error:
            logger.error(f"Error in response generation: {response_error}")
            # If we haven't yielded anything yet, provide a fallback response
            if not response_buffer:
                if "portrait" in _last_cot_results.get("user_profile", {}) and "Discouraged Group" in _last_cot_results["user_profile"]["portrait"]:
                    # Tailored fallback for Discouraged Group
                    fallback = "Xuất tinh sớm là tình trạng phổ biến, và rất nhiều nam giới gặp phải. Đây không phải là điều gì đáng xấu hổ. Có nhiều phương pháp hữu ích để cải thiện tình trạng này như các bài tập Kegel, kỹ thuật thở sâu, và phương pháp start-stop. Bạn có thể thử những phương pháp này hoặc tham khảo ý kiến chuyên gia y tế."
                else:
                    # Generic fallback
                    fallback = "Tôi hiểu vấn đề bạn đang gặp phải. Có nhiều phương pháp và kỹ thuật có thể giúp cải thiện tình trạng này. Bạn có thể thử một số bài tập thư giãn và kỹ thuật kiểm soát. Nếu cần thiết, tham khảo ý kiến bác sĩ sẽ giúp bạn có phương pháp điều trị phù hợp."
                
                yield fallback
                response_buffer = fallback
        
        # Log the final response buffer
        logger.info(f"Final response buffer: {response_buffer[:50]}...")
        
        # After all processing, update the state
        if response_buffer:
            state["messages"].append({"role": "assistant", "content": response_buffer})
            state["prompt_str"] = response_buffer
            logger.info(f"Updated state with response of {len(response_buffer)} chars")
        else:
            logger.warning("Response buffer is empty, state not updated")
        
        # Log event counts
        logger.info(f"Total analysis events sent from trigger: {analysis_events_sent}")
        logger.info(f"Total knowledge events sent from trigger: {knowledge_events_sent}")
        logger.info(f"Total next actions events sent from trigger: {next_actions_events_sent}")
        
        # Yield the final state
        yield {"state": state}
        
        # Log overall performance metrics
        PERF_METRICS["overall_end"] = time.time()
        total_time = PERF_METRICS["overall_end"] - PERF_METRICS["overall_start"]
        logger.info(f"Total processing time: {total_time:.2f}s")
        
        # Log detailed performance metrics
        logger.info("Performance metrics:")
        for key, value in PERF_METRICS.items():
            if key.endswith("_start"):
                end_key = key.replace("_start", "_end")
                if end_key in PERF_METRICS:
                    duration = PERF_METRICS[end_key] - value
                    logger.info(f"  {key[:-6]}: {duration:.2f}s")
            
    except Exception as e:
        logger.error(f"Error in LLM tool calling: {str(e)}")
        logger.error(traceback.format_exc())
        yield "I encountered an issue while processing your request. Please try again."

async def execute_tool_call(tool_call: Dict, params: Dict, thread_id: Optional[str] = None) -> AsyncGenerator[Dict, None]:
    """
    Execute a single tool call and handle the results.
    
    Args:
        tool_call: The tool call dictionary containing name and parameters
        params: Common parameters to include with the tool call
        thread_id: Optional thread ID for WebSocket events
        
    Yields:
        Results from the tool handler
    """
    try:
        tool_name = tool_call.get("name", "")
        tool_args = tool_call.get("arguments", {})
        
        # Get handler for this tool
        handler = tool_registry.get_tool_handler(tool_name)
        if not handler:
            logger.error(f"No handler found for tool: {tool_name}")
            yield {"error": f"Tool not found: {tool_name}"}
            return
        
        # Merge args with common params
        if isinstance(tool_args, str):
            try:
                # Parse if it's a JSON string
                tool_args = json.loads(tool_args)
            except:
                # Fallback: create a "query" parameter if parsing fails
                tool_args = {"query": tool_args}
        
        # Ensure tool_args is a dict
        if not isinstance(tool_args, dict):
            tool_args = {"input": str(tool_args)}
        
        # Add common params
        call_params = {**tool_args, **params}
        
        # Add thread ID for WebSocket routing if provided
        if thread_id:
            call_params["_thread_id"] = thread_id
            
        logger.info(f"Executing tool call: {tool_name} with params: {str(call_params)[:100]}...")
        
        # Track tool call start time
        start_time = time.time()
        
        # Execute the tool handler
        try:
            async for result in handler(call_params):
                # Log and yield each result
                if isinstance(result, dict) and "type" in result:
                    event_type = result.get("type")
                    is_complete = result.get("complete", False)
                    logger.info(f"Tool {tool_name} returned {event_type} event (complete: {is_complete})")
                yield result
        except Exception as handler_error:
            logger.error(f"Error in tool handler {tool_name}: {str(handler_error)}")
            yield {"error": f"Tool execution error: {str(handler_error)}"}
        
        # Log completion
        execution_time = time.time() - start_time
        logger.info(f"Tool call {tool_name} completed in {execution_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Error executing tool call: {str(e)}")
        logger.error(traceback.format_exc())
        yield {"error": f"Tool execution error: {str(e)}"}
