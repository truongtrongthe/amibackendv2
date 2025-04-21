"""
Tool calling implementation for the MC system.
This file defines the tool schemas and handlers for LLM tool calling.
"""

import json
from typing import List, Dict, Any, Optional, AsyncGenerator, Union
import asyncio
from langchain_openai import ChatOpenAI
from utilities import logger
from analysis import (
    stream_analysis, 
    stream_next_action,
    build_context_analysis_prompt,
    build_next_actions_prompt
)
from database import query_graph_knowledge, get_version_brain_banks, get_cached_embedding
from response_optimization import ResponseProcessor, ResponseFilter

# Use the same LLM instances as mc.py
LLM = ChatOpenAI(model="gpt-4o", streaming=False)
StreamLLM = ChatOpenAI(model="gpt-4o", streaming=True)

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


# Tool Schemas - These define the interface for each tool
CONTEXT_ANALYSIS_SCHEMA = {
    "name": "context_analysis_tool",
    "description": "Analyze the conversation context to understand user needs, identify intents, and determine key information",
    "parameters": {
        "type": "object",
        "properties": {
            "conversation_context": {
                "type": "string",
                "description": "The full conversation history to analyze"
            },
            "additional_instructions": {
                "type": "string",
                "description": "Optional additional instructions for the analysis",
                "default": ""
            }
        },
        "required": ["conversation_context"]
    }
}

NEXT_ACTIONS_SCHEMA = {
    "name": "next_actions_tool",
    "description": "Determine the next actions to take based on conversation context, analysis, and available knowledge",
    "parameters": {
        "type": "object",
        "properties": {
            "conversation_context": {
                "type": "string",
                "description": "The conversation history"
            },
            "context_analysis": {
                "type": "string",
                "description": "The analysis of the conversation context"
            },
            "knowledge_context": {
                "type": "string",
                "description": "The retrieved knowledge relevant to the conversation"
            }
        },
        "required": ["conversation_context", "context_analysis", "knowledge_context"]
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
        "required": ["conversation_context", "analysis", "next_actions", "personality_instructions","knowledge_context"]
    }
}

# Tool Handlers - These implement the actual functionality
async def context_analysis_handler(params: Dict) -> AsyncGenerator[Dict, None]:
    """
    Handle context analysis requests with streaming
    
    Args:
        params: Dictionary containing conversation_context and optional additional_instructions
        
    Yields:
        Dict events with streaming analysis results
    """
    conversation_context = params.get("conversation_context", "")
    additional_instructions = params.get("additional_instructions", "")
    
    # Build the analysis prompt
    analysis_prompt = build_context_analysis_prompt(conversation_context, additional_instructions)
    
    # Use existing stream_analysis function
    thread_id = params.get("_thread_id")  # Internal param for WebSocket
    use_websocket = thread_id is not None
    
    async for chunk in stream_analysis(analysis_prompt, thread_id, use_websocket):
        yield chunk


async def next_actions_handler(params: Dict) -> AsyncGenerator[Dict, None]:
    """
    Handle next actions generation with streaming
    
    Args:
        params: Dictionary containing conversation_context, context_analysis, and knowledge_context
        
    Yields:
        Dict events with streaming next action results
    """
    conversation_context = params.get("conversation_context", "")
    context_analysis = params.get("context_analysis", "")
    knowledge_context = params.get("knowledge_context", "")
    
    # Build the next actions prompt
    next_actions_prompt = build_next_actions_prompt(
        conversation_context, 
        context_analysis, 
        knowledge_context
    )
    
    # Use existing stream_next_action function
    thread_id = params.get("_thread_id")  # Internal param for WebSocket
    use_websocket = thread_id is not None
    
    async for chunk in stream_next_action(next_actions_prompt, thread_id, use_websocket):
        yield chunk


async def knowledge_query_handler(params: Dict) -> AsyncGenerator[Dict, None]:
    """
    Handle knowledge query requests with streaming
    
    Args:
        params: Dictionary containing queries, graph_version_id, and optional top_k
        
    Yields:
        Dict events with streaming knowledge query results
    """
    queries = params.get("queries", [])
    graph_version_id = params.get("graph_version_id", "")
    top_k = params.get("top_k", 3)
    thread_id = params.get("_thread_id")  # Internal param for WebSocket
    
    # Handle empty queries
    if not queries:
        yield {"type": "knowledge", "content": [], "complete": True}
        return
    
    try:
        # Get brain banks
        brain_banks = await get_version_brain_banks(graph_version_id)
        
        if not brain_banks:
            yield {"type": "knowledge", "content": [], "complete": True, "error": "No brain banks found"}
            return
        
        # Generate embeddings
        embedding_tasks = [get_cached_embedding(query) for query in queries]
        embeddings = await asyncio.gather(*embedding_tasks)
        
        # Create mapping of query index to embedding
        query_embeddings = {i: embedding for i, embedding in enumerate(embeddings)}
        
        # Store already streamed result IDs to avoid duplicates
        streamed_ids = set()
        all_results = []
        
        # Import necessary function
        from database import query_brain_with_embeddings_batch
        
        # Check if streaming function is available
        try:
            from database import stream_query_brain_with_embeddings
            has_streaming = True
        except ImportError:
            has_streaming = False
            logger.warning("stream_query_brain_with_embeddings not available, falling back to batch queries")
        
        # Process brain banks
        for brain_bank in brain_banks:
            bank_name = brain_bank["bank_name"]
            brain_id = brain_bank["id"]
            
            if has_streaming:
                # Stream results from this brain bank
                try:
                    # Use the streaming function if available
                    from database import stream_query_brain_with_embeddings
                    async for result_batch in stream_query_brain_with_embeddings(
                        query_embeddings, bank_name, brain_id, top_k
                    ):
                        # Filter out already streamed results
                        new_results = []
                        for result in result_batch:
                            result_id = result.get("id", "unknown")
                            if result_id not in streamed_ids:
                                streamed_ids.add(result_id)
                                new_results.append(result)
                                all_results.append(result)
                        
                        # Only yield if we have new results
                        if new_results:
                            yield {"type": "knowledge", "content": new_results, "complete": False}
                except Exception as e:
                    logger.error(f"Error in streaming knowledge query: {str(e)}")
                    # Fall back to batch query
                    has_streaming = False
            
            if not has_streaming:
                # Use batch query as fallback
                try:
                    batch_results = await query_brain_with_embeddings_batch(
                        query_embeddings, bank_name, brain_id, top_k
                    )
                    
                    # Process results
                    new_results = []
                    for query_idx, results in batch_results.items():
                        for result in results:
                            result_id = result.get("id", "unknown")
                            if result_id not in streamed_ids:
                                streamed_ids.add(result_id)
                                new_results.append(result)
                                all_results.append(result)
                    
                    # Yield results in a batch
                    if new_results:
                        yield {"type": "knowledge", "content": new_results, "complete": False}
                
                except Exception as e:
                    logger.error(f"Error in batch knowledge query: {str(e)}")
                    # Just continue to the next brain bank
        
        # Final complete event
        yield {
            "type": "knowledge", 
            "content": all_results, 
            "complete": True, 
            "stats": {
                "total_results": len(all_results),
                "query_count": len(queries),
                "brain_bank_count": len(brain_banks)
            }
        }
        
    except Exception as e:
        logger.error(f"Error in knowledge query: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        yield {"type": "knowledge", "content": [], "complete": True, "error": str(e)}


async def response_generation_handler(params: Dict) -> AsyncGenerator[str, None]:
    """
    Handle response generation with streaming
    
    Args:
        params: Dictionary containing conversation_context, analysis, next_actions, 
                knowledge_context, personality_instructions, and knowledge_found
                
    Yields:
        Streamed response chunks as strings
    """
    conversation_context = params.get("conversation_context", "")
    analysis = params.get("analysis", "")
    next_actions = params.get("next_actions", "")
    knowledge_context = params.get("knowledge_context", "")
    personality_instructions = params.get("personality_instructions", "")
    knowledge_found = params.get("knowledge_found", False)
    
    # Detect language from analysis or conversation context
    detected_language = "unknown"
    if "Vietnamese" in analysis or "tiếng Việt" in analysis:
        detected_language = "vietnamese"
    elif "English" in analysis:
        detected_language = "english"
    else:
        # Simple language detection based on common Vietnamese words
        vietnamese_markers = ["của", "những", "và", "các", "là", "không", "có", "được", "người", "trong", "để", "anh", "chị", "em", "ơi", "nhé"]
        english_markers = ["the", "and", "is", "in", "to", "of", "that", "for", "you", "with", "have", "this", "are"]
        
        vietnamese_count = sum(1 for marker in vietnamese_markers if f" {marker} " in f" {conversation_context} ")
        english_count = sum(1 for marker in english_markers if f" {marker} " in f" {conversation_context} ")
        
        if vietnamese_count > english_count:
            detected_language = "vietnamese"
        else:
            detected_language = "english"
    
    # Extract user name or pronoun from conversation context
    user_name = None
    last_user_message = ""
    
    # Get the last user message
    for line in conversation_context.split('\n'):
        if line.startswith("User:"):
            last_user_message = line[5:].strip()
    
    # Build language and cultural awareness instructions
    cultural_instructions = """
Respond in the detected language with a natural and culturally aware tone. Adapt to the cultural context implied by the language by following these principles:

1. Choose forms of address or expressions that reflect the social and relational dynamics inferred from the user's communication style and context.
2. Initiate or respond with a tone that feels appropriate to the conversation's flow, balancing respect and familiarity as needed.
3. Incorporate linguistic nuances or conversational elements that enhance connection and align with natural speech patterns in the detected language.
4. Adjust your tone and level of formality to match the user's style, evolving with the interaction.
5. Present ideas or suggestions thoughtfully, respecting the user's perspective and cultural expectations.
6. Ensure your language feels authentic and fluent, mirroring the natural rhythms and idioms of the detected language.

Focus on embodying the cultural values and norms associated with the detected language, fostering rapport through empathy and adaptability while personalizing your tone to the user's cues.
"""
    
    # Build the response prompt
    prompt_parts = [
        "# Conversation Context",
        conversation_context,
        "# Analysis",
        analysis,
        "# Next Actions",
        next_actions
    ]
    
    if knowledge_context:
        prompt_parts.extend([
            "# Retrieved Knowledge",
            knowledge_context
        ])
    
    prompt_parts.extend([
        "# Personality Instructions",
        personality_instructions,
        "# Language and Cultural Guidelines",
        cultural_instructions,
        
        "# Your Task",
        """
            You are an empathetic AI assistant helping users with their concerns. Your goal is to provide thoughtful, natural-sounding responses tailored to each user's unique situation.
            
            Focus on three key components:
            
            1. UNDERSTAND THE USER: The analysis provides insights into the user's needs, conversation stage, and information gaps. Use this understanding as your foundation.
               - Recognize where the user is in their journey
               - Identify what information is missing
               - Understand the emotional context
            
            2. FOLLOW THE GUIDANCE: The next actions provide specific direction on what to do next. Follow this guidance closely.
               - If specific phrases or questions are recommended, incorporate them naturally
               - Maintain the suggested sequence of actions
               - When knowledge provides exact wording for questions or statements, use them authentically
            
            3. SOUND NATURAL: While following the guidance, ensure your response sounds conversational and genuine.
               - Adapt to the detected language and cultural context
               - Blend any suggested wording or phrases seamlessly into your response
               - Maintain a tone that builds trust and connection
               - End your messages naturally without formulaic closing statements; let conversations flow as they would between humans
               - Avoid repeating reassurances or support statements at the end of every message
            
            Craft a concise response (80-100 words) that addresses the user's concerns while following the recommended actions. Your response should feel like it's coming from a knowledgeable, empathetic human who deeply understands both the user's situation and cultural context.
        """
        ])
    
    prompt = "\n\n".join(prompt_parts)
    
    # Response optimization components
    response_processor = ResponseProcessor()
    response_filter = ResponseFilter()
    
    # Check if this is the first message
    is_first_message = "User:" in conversation_context and conversation_context.count("User:") <= 1
    
    # Collect the full response first
    full_response = ""
    buffer = ""
    
    try:
        async for chunk in StreamLLM.astream(prompt):
            content = chunk.content
            buffer += content
            full_response += content
            
            # Only yield periodically to reduce overhead
            if len(buffer) >= 20 or content.endswith((".", "!", "?", "\n")):
                yield buffer
                buffer = ""
        
        # Yield any remaining buffer
        if buffer:
            yield buffer
        
        # Process the full response for optimization
        processed_response = response_processor.process_response(
            full_response, 
            {"messages": conversation_context.split("\n")},
            conversation_context.split("\n")[-1] if conversation_context.split("\n") else "",
            knowledge_found
        )
        
        # Apply greeting filter if needed
        if not is_first_message:
            processed_response = response_filter.remove_greeting(processed_response)
        
        # We've already yielded the raw response, so we don't need to yield again
        
    except Exception as e:
        logger.error(f"Error in response generation: {str(e)}")
        # Fall back to language-specific error messages
        if detected_language == "vietnamese":
            yield "Xin lỗi, tôi đang gặp sự cố khi xử lý yêu cầu của bạn. Vui lòng thử lại."
        else:
            yield "I encountered an issue while generating a response. Let me try again."

# Register the tools
tool_registry.register_tool(
    "context_analysis_tool", 
    context_analysis_handler, 
    CONTEXT_ANALYSIS_SCHEMA
)

tool_registry.register_tool(
    "next_actions_tool", 
    next_actions_handler, 
    NEXT_ACTIONS_SCHEMA
)

tool_registry.register_tool(
    "knowledge_query_tool", 
    knowledge_query_handler, 
    KNOWLEDGE_QUERY_SCHEMA
)

tool_registry.register_tool(
    "response_generation_tool", 
    response_generation_handler, 
    RESPONSE_GENERATION_SCHEMA
)

async def execute_tool_call(tool_call: Dict) -> AsyncGenerator[Dict, None]:
    """
    Execute a tool call and return the result
    
    Args:
        tool_call: Dict with name and parameters for the tool call
        
    Yields:
        Tool results as they become available
    """
    tool_name = tool_call.get("name")
    parameters = tool_call.get("parameters", {})
    
    try:
        # Get the tool handler
        handler = tool_registry.get_tool_handler(tool_name)
        
        # Execute the tool
        async for result in handler(parameters):
            # If the handler already returns properly formatted results,
            # just pass them through
            if isinstance(result, dict) and result.get("type") in ["analysis", "knowledge", "next_actions"]:
                yield result
            else:
                # Otherwise, wrap the result
                yield {
                    "tool_call_id": tool_call.get("id", "unknown"),
                    "tool_name": tool_name,
                    "content": result
                }
            
    except Exception as e:
        logger.error(f"Error executing tool {tool_name}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        yield {
            "tool_call_id": tool_call.get("id", "unknown"),
            "tool_name": tool_name,
            "content": {"error": str(e)},
            "error": True
        }

async def process_llm_with_tools(
    user_message: str,
    conversation_history: List[Dict],
    state: Dict,
    graph_version_id: str,
    thread_id: Optional[str] = None
) -> AsyncGenerator[Union[str, Dict], None]:
    """
    Process a user message using LLM with tool calling.
    This function mirrors the flow of MC._handle_request to ensure functional equivalence
    while using the tool-based architecture.
    
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
    
    # Prepare messages for LLM context
    messages = []
    
    # Convert conversation history to string format
    conversation_context = ""
    for msg in conversation_history[-100:]:  # Limit to last 10 messages
        if msg.get("role") == "user":
            conversation_context += f"User: {msg.get('content', '')}\n"
        elif msg.get("role") == "assistant":
            conversation_context += f"AI: {msg.get('content', '')}\n"
    
    # Add current user message
    conversation_context += f"User: {user_message}\n"
    
    use_websocket = thread_id is not None
    thread_id_for_analysis = thread_id
    
    try:
        # STEP 1: Load personality if needed
        from personality import PersonalityManager
        personality_manager = PersonalityManager()
        if not hasattr(personality_manager, 'personality_instructions') or not personality_manager.personality_instructions:
            await personality_manager.load_personality(graph_version_id)
        
        # STEP 2: Detect language (simplified - handled within tools)
        
        # STEP 3: Run context analysis
        logger.info(f"Starting context analysis for message: {user_message[:50]}...")
        context_params = {
            "conversation_context": conversation_context
        }
        if thread_id:
            context_params["_thread_id"] = thread_id
            
        analysis_results = ""
        async for result in context_analysis_handler(context_params):
            # Pass through the analysis events
            yield {"type": "analysis", "content": result}
            
            # If this is a complete event, store the full analysis
            if isinstance(result, dict) and result.get("complete", False):
                analysis_results = result.get("content", "")
        
        # STEP 4: Extract search terms from analysis
        if analysis_results:
            logger.info(f"Context analysis complete, extracting search terms...")
            from analysis import extract_search_terms
            search_terms = extract_search_terms(analysis_results)
            logger.info(f"Extracted {len(search_terms)} search terms: {search_terms}")
        else:
            # Fallback to basic search terms
            search_terms = [user_message]
            logger.info(f"Using fallback search term: {user_message}")
        
        # STEP 5: Run knowledge query
        logger.info(f"Starting knowledge query with {len(search_terms)} terms...")
        knowledge_params = {
            "queries": search_terms,
            "graph_version_id": graph_version_id,
            "top_k": 3
        }
        if thread_id:
            knowledge_params["_thread_id"] = thread_id
            
        all_knowledge_entries = []
        async for result in knowledge_query_handler(knowledge_params):
            # Pass through the knowledge events directly without double-wrapping
            yield result
            
            # Collect knowledge entries
            if isinstance(result, dict) and "content" in result:
                entries = result.get("content", [])
                if isinstance(entries, list):
                    all_knowledge_entries.extend(entries)
        
        # Format knowledge context
        if all_knowledge_entries:
            knowledge_context = "\n\n".join(entry.get("raw", "") for entry in all_knowledge_entries)
            knowledge_found = True
            logger.info(f"Retrieved {len(all_knowledge_entries)} knowledge entries")
        else:
            knowledge_context = "No relevant knowledge found."
            knowledge_found = False
            logger.info("No knowledge entries found")
        
        # STEP 6: Run next actions
        logger.info(f"Starting next actions determination...")
        next_actions_params = {
            "conversation_context": conversation_context,
            "context_analysis": analysis_results,
            "knowledge_context": knowledge_context
        }
        if thread_id:
            next_actions_params["_thread_id"] = thread_id
            
        next_actions_results = ""
        async for result in next_actions_handler(next_actions_params):
            # Pass through the next actions events directly without double-wrapping
            yield result
            
            # If this is a complete event, store the full next actions
            if isinstance(result, dict) and result.get("complete", False):
                next_actions_results = result.get("content", "")
        
        # STEP 7: Assess knowledge coverage (simplified)
        # In the original _handle_request this would call assess_knowledge_coverage
        
        # STEP 8: Prepare and send final response prompt
        logger.info(f"Generating final response...")
        
        # Build the final response prompt with all the components
        response_params = {
            "conversation_context": conversation_context,
            "analysis": analysis_results,
            "next_actions": next_actions_results,
            "knowledge_context": knowledge_context,
            "personality_instructions": personality_manager.personality_instructions,
            "knowledge_found": knowledge_found
        }
        
        # Stream the response
        response_buffer = ""
        async for chunk in response_generation_handler(response_params):
            # Response chunks are streamed as strings
            if isinstance(chunk, str):
                response_buffer += chunk
                yield chunk
        
        # After all processing, update the state
        if response_buffer:
            state["messages"].append({"role": "assistant", "content": response_buffer})
            state["prompt_str"] = response_buffer
        
        # Yield the final state
        yield {"state": state}
        
    except Exception as e:
        logger.error(f"Error in LLM tool calling: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        yield "I encountered an issue while processing your request. Please try again." 