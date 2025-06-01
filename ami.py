import logging
import json
import re
import time
from typing import TypedDict, Annotated, List, Dict, Any, AsyncGenerator
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph.message import add_messages

from mc_tools import MCWithTools  # Standard tools
from utilities import logger
import asyncio

# Import socketio manager for WebSocket support
try:
    from socketio_manager_async import emit_analysis_event, emit_knowledge_event, emit_next_action_event, emit_learning_intent_event, emit_learning_knowledge_event
    socketio_imports_success = True
    logger.info("Successfully imported socketio_manager_async functions in ami.py")
except ImportError:
    socketio_imports_success = False
    logger.warning("Could not import socketio_manager_async in ami.py - WebSocket events may not be delivered")

# Tool execution helper functions for AMI
async def execute_save_knowledge_tool(parameters: dict, user_id: str, thread_id: str) -> dict:
    """Execute save_knowledge tool call with proper error handling and vector ID capture."""
    from pccontroller import save_knowledge
    try:
        # Get required parameters
        input_text = parameters.get("query", "")
        if not input_text and "content" in parameters:
            input_text = parameters.get("content", "")
        
        # If we still don't have input text, try to use both query and content
        if not input_text and "query" in parameters and "content" in parameters:
            input_text = f"{parameters['query']} {parameters['content']}".strip()
        
        if not input_text:
            logger.error("Missing required parameter for save_knowledge: query or content")
            return {"success": False, "error": "Missing required content"}
            
        # Get optional parameters
        user_id_param = parameters.get("user_id", user_id)
        bank_name = parameters.get("bank_name", "default")
        thread_id_param = parameters.get("thread_id", thread_id)
        topic = parameters.get("topic", None)
        categories = parameters.get("categories", ["general"])
        
        # Check if this is a health-related query and adjust bank_name
        if not parameters.get("bank_name"):
            if any(term in input_text.lower() for term in ["rối loạn cương dương", "xuất tinh sớm", "phân nhóm khách hàng", "phân tích chân dung khách hàng"]):
                bank_name = "health"
                if "health_segmentation" not in categories:
                    categories.append("health_segmentation")
        
        logger.info(f"Executing save_knowledge tool: '{input_text[:50]}...' for user {user_id_param}")
        
        # Execute save_knowledge
        result = await save_knowledge(
            input=input_text,
            user_id=user_id_param,
            bank_name=bank_name,
            thread_id=thread_id_param,
            topic=topic,
            categories=categories
        )
        logger.info(f"Save knowledge tool result: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Error executing save_knowledge tool: {str(e)}")
        return {"success": False, "error": str(e)}

async def execute_save_teaching_synthesis_tool(parameters: dict, user_id: str, thread_id: str) -> dict:
    """Execute save_teaching_synthesis tool call for human-in-the-loop scenarios."""
    try:
        from alpha import save_teaching_synthesis
        
        # Extract parameters
        conversation_turns = parameters.get("conversation_turns", [])
        final_synthesis = parameters.get("final_synthesis", "")
        topic = parameters.get("topic", "user_teaching")
        priority_topic_name = parameters.get("priority_topic_name", "")
        
        logger.info(f"Executing save_teaching_synthesis tool for topic: {topic}")
        
        # Execute save_teaching_synthesis
        result = await save_teaching_synthesis(
            conversation_turns=conversation_turns,
            final_synthesis=final_synthesis,
            topic=topic,
            user_id=user_id,
            thread_id=thread_id,
            priority_topic_name=priority_topic_name
        )
        logger.info(f"Save teaching synthesis tool result: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Error executing save_teaching_synthesis tool: {str(e)}")
        return {"success": False, "error": str(e)}

async def handle_save_approval_request(parameters: dict, user_id: str, thread_id: str) -> dict:
    """Handle save approval request for human-in-the-loop scenarios."""
    try:
        # Future implementation for human-in-the-loop
        preview = parameters.get("preview", "")
        content = parameters.get("content", "")
        requires_approval = parameters.get("requires_approval", True)
        
        logger.info(f"Creating save approval request for user {user_id}")
        
        # For now, return a placeholder structure
        # In the future, this would:
        # 1. Store the pending save request
        # 2. Send approval request to frontend
        # 3. Wait for human response
        # 4. Execute save based on approval
        
        request_id = f"approval_{thread_id}_{int(time.time())}"
        
        return {
            "success": True,
            "request_id": request_id,
            "status": "pending_approval",
            "preview": preview,
            "message": "Save approval request created (future feature)"
        }
        
    except Exception as e:
        logger.error(f"Error handling save approval request: {str(e)}")
        return {"success": False, "error": str(e)}

async def handle_update_decision_tool(parameters: dict, user_id: str, thread_id: str) -> dict:
    """Handle UPDATE vs CREATE decision from human."""
    try:
        request_id = parameters.get("request_id", "")
        action = parameters.get("action", "")
        target_id = parameters.get("target_id", "")
        
        if not request_id:
            return {"success": False, "error": "Missing request_id parameter"}
        
        if not action:
            return {"success": False, "error": "Missing action parameter"}
        
        logger.info(f"Processing UPDATE vs CREATE decision: {action} for request {request_id}")
        
        # Get the AVA instance to handle the decision
        # Note: In a production system, this would be handled differently
        # For now, we'll create a temporary instance
        from ava import AVA
        ava = AVA()
        await ava.initialize()
        
        # Prepare user decision
        user_decision = {
            "action": action,
            "target_id": target_id if action == "UPDATE_EXISTING" else None
        }
        
        # Handle the decision
        result = await ava.handle_update_decision(
            request_id=request_id,
            user_decision=user_decision,
            user_id=user_id,
            thread_id=thread_id
        )
        
        logger.info(f"UPDATE vs CREATE decision result: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Error handling UPDATE vs CREATE decision: {str(e)}")
        return {"success": False, "error": str(e)}

class State(TypedDict):
    messages: Annotated[list, add_messages]
    prompt_str: str
    convo_id: str
    user_id: str
    preset_memory: str
    instinct: str
    graph_version_id: str  # Added graph_version_id
    analysis: dict  # Add analysis field to the state schema
    stream_events: list  # Add stream_events for real-time events

mc = MCWithTools(user_id="thefusionlab")  # Use MCWithTools instead of MC


async def mc_node(state: State, config=None):
    
    start_time = time.time()
    user_id = config.get("configurable", {}).get("user_id", "thefusionlab") if config else "thefusionlab"
    graph_version_id = config.get("configurable", {}).get("graph_version_id", "") if config else ""
    
    logger.info(f"MC Node - User ID: {user_id}, Graph Version: {graph_version_id}")
    
    if not mc.instincts:
        await mc.initialize()
    
    # Process the async generator output from trigger
    final_state = state  # Default to input state
    last_analysis = None  # Track the last analysis event

    # Log initial state
    logger.info(f"Initial state keys: {list(state.keys())}")
    
    # Create stream_events list to capture events that need to be streamed to client
    stream_events = []
    
    # Count the analysis events for debugging
    analysis_count = 0
    complete_analysis_count = 0

    async for output in mc.trigger(state=state, user_id=user_id, graph_version_id=graph_version_id, config=config):
        if isinstance(output, dict) and "state" in output:
            final_state = output["state"]  # Capture the final state
            logger.info(f"Received state update with keys: {list(final_state.keys())}")
        elif isinstance(output, dict) and output.get("type") == "analysis":
            # Handle analysis event
            analysis_count += 1
            
            # Store analysis event to be streamed
            stream_events.append({
                "event_type": "analysis",
                "data": output
            })
            
            # Also store in state for persistence
            last_analysis = output
            if output.get("complete", False):
                # Only store the complete analysis in the state
                complete_analysis_count += 1
                final_state["analysis"] = output
                logger.info(f"Stored COMPLETE analysis in state: {output.get('content', '')[:100] if isinstance(output.get('content', ''), str) else str(output.get('content', ''))[:100]}...")
            else:
                # Safely handle content that might be a dictionary or other non-string type
                content = output.get('content', '')
                content_str = content if isinstance(content, str) else str(content)
                logger.debug(f"Received partial analysis chunk: {content_str[:50]}...")
        else:
            # Regular response chunk
            logger.debug(f"Received response chunk: {output}")
    
    # Make sure the final analysis is stored in the state
    if last_analysis and last_analysis.get("complete", False):
        final_state["analysis"] = last_analysis
        logger.info(f"Final analysis stored in state at end of processing")
    
    logger.info(f"Analysis events received: {analysis_count} (complete: {complete_analysis_count})")
    
    # Handle analysis in final state with proper indentation
    if 'analysis' in final_state:
        # CRITICAL FIX: Add proper type checking for analysis field
        if isinstance(final_state['analysis'], dict):
            analysis_content = final_state['analysis'].get('content', '')
            analysis_str = analysis_content if isinstance(analysis_content, str) else str(analysis_content)
            logger.info(f"Final analysis content: {analysis_str[:100]}...")
        else:
            # Handle case where analysis is a string or other non-dict type
            analysis_str = str(final_state['analysis'])
            logger.warning(f"Analysis is not a dict type (got {type(final_state['analysis']).__name__}). Content: {analysis_str[:100]}...")
            # Convert to proper format to avoid issues later
            final_state['analysis'] = {
                "content": analysis_str,
                "complete": True,
                "type": "analysis"
            }
    
    # Make ABSOLUTELY sure analysis is included in the final state
    if 'analysis' not in final_state:
        if last_analysis:
            final_state["analysis"] = last_analysis
            logger.warning("Had to add analysis to final state before returning")
        else:
            logger.warning("No analysis available to add to final state")
    
    # Store stream events in the state for convo_stream to process
    final_state["stream_events"] = stream_events
    logger.info(f"Stored {len(stream_events)} stream events in state")
    
    logger.debug(f"Mc node took {time.time() - start_time:.2f}s")
    return final_state


graph_builder = StateGraph(State)
graph_builder.add_node("mc", mc_node)
graph_builder.set_entry_point("mc")
graph_builder.add_edge("mc", END)

checkpointer = MemorySaver()
convo_graph = graph_builder.compile(checkpointer=checkpointer)
logger.info(f"Graph nodes: {convo_graph.nodes}")


async def convo_stream(user_input: str = None, user_id: str = None, thread_id: str = None, 
              graph_version_id: str = "", mode: str = "mc", 
              use_websocket: bool = False, thread_id_for_analysis: str = None):
    """
    Process user input and stream the response.
    
    Args:
        user_input: The user's message
        user_id: User identifier
        thread_id: Conversation thread identifier
        graph_version_id: Graph version identifier for knowledge retrieval
        mode: Processing mode ('mc', 'pilot', etc.)
        use_websocket: Whether to use WebSocket for analysis streaming
        thread_id_for_analysis: Thread ID to use for WebSocket analysis events
    
    Returns:
        A generator yielding response chunks
    """
    start_time = time.time()
    thread_id = thread_id or f"mc_thread_{int(time.time())}"
    user_id = user_id or "thefusionlab"
    logger.info(f"convo_stream init - Input: '{user_input}', Convo ID: {thread_id}")
    
    # Load or init state with intent_history
    checkpoint = checkpointer.get({"configurable": {"thread_id": thread_id}})
    default_state = {
        "messages": [],
        "prompt_str": "",
        "convo_id": thread_id,
        "last_response": "",
        "user_id": user_id,
        "preset_memory": "Be friendly",
        "instinct": "",
        "graph_version_id": graph_version_id,
        "analysis": {},  # Add a field for analysis
        "stream_events": [],  # Add field for stream events
        "use_websocket": use_websocket,  # Flag to use WebSocket
        "thread_id_for_analysis": thread_id_for_analysis  # Thread ID for WebSocket analysis
    }
    state = {**default_state, **(checkpoint.get("channel_values", {}) if checkpoint else {})}

    # Add user input
    if user_input:
        state["messages"] = add_messages(state["messages"], [HumanMessage(content=user_input)])

    logger.debug(f"convo_stream init - Input: '{user_input}', Convo ID: {thread_id}")

    # Configuration for graph
    config = {
        "configurable": {
            "thread_id": thread_id,
            "user_id": user_id,
            "graph_version_id": graph_version_id,
            "use_websocket": use_websocket,
            "thread_id_for_analysis": thread_id_for_analysis
        }
    }
    
    # Process the state asynchronously
    async def process_state():
        # Invoke the graph with the state
        logger.info(f"Invoking graph with state: {state}")
        updated_state = await convo_graph.ainvoke(state, config)
        logger.info(f"After ainvoke, keys in state: {list(updated_state.keys())}")
        await convo_graph.aupdate_state({"configurable": {"thread_id": thread_id}}, updated_state, as_node="mc")
        return updated_state
    
    # Use await directly
    updated_state = await process_state()
    
    logger.debug(f"convo_stream processing took {time.time() - start_time:.2f}s")
    
    # When using WebSockets for analysis, only stream the final response here
    # Analysis is handled directly via WebSocket in the MC class
    if not use_websocket:
        # First stream all analysis events in order (only for non-WebSocket mode)
        if "stream_events" in updated_state and updated_state["stream_events"]:
            logger.info(f"Streaming {len(updated_state['stream_events'])} events to client via SSE")
            
            # Process events in sequence
            for event in updated_state["stream_events"]:
                if event["event_type"] == "analysis":
                    analysis_data = event["data"]
                    
                    # Safely handle content that might be a dictionary or other non-string type
                    content = analysis_data.get("content", "")
                    if not isinstance(content, (str, dict, list, bool, int, float, type(None))):
                        content = str(content)
                    
                    # Format the analysis event for streaming
                    analysis_json = json.dumps({
                        "type": "analysis", 
                        "content": content,
                        "complete": analysis_data.get("complete", False)
                    })
                    # Send analysis chunk in real-time
                    yield f"data: {analysis_json}\n\n"
                    # Small delay to let frontend process event
                    await asyncio.sleep(0.05)
        else:
            logger.warning("No stream_events found in updated state")
            
            # Fallback: Try to send the analysis from state if available
            if "analysis" in updated_state and isinstance(updated_state["analysis"], dict):
                logger.info(f"Fallback: Streaming analysis from state directly")
                
                # Safely handle content that might be a dictionary or other non-string type
                content = updated_state["analysis"].get("content", "")
                if not isinstance(content, (str, dict, list, bool, int, float, type(None))):
                    content = str(content)
                
                analysis_json = json.dumps({
                    "type": "analysis", 
                    "content": content,
                    "complete": updated_state["analysis"].get("complete", False)
                })
                yield f"data: {analysis_json}\n\n"
            elif "analysis" in updated_state:
                # Handle case where analysis is not a dictionary
                logger.warning(f"Analysis in state is not a dictionary, converting to proper format")
                analysis_str = str(updated_state["analysis"])
                analysis_json = json.dumps({
                    "type": "analysis", 
                    "content": analysis_str,
                    "complete": True
                })
                yield f"data: {analysis_json}\n\n"
    else:
        logger.info("Using WebSockets for analysis streaming - analysis events sent directly")
    
    # Stream the regular response
    response_lines = updated_state["prompt_str"].split('\n')
    for line in response_lines:
        if line.strip():
            yield f"data: {json.dumps({'message': line.strip()})}\n\n"
            await asyncio.sleep(0.05)
    
    logger.debug(f"convo_stream total took {time.time() - start_time:.2f}s")

async def convo_stream_learning(user_input: str = None, user_id: str = None, thread_id: str = None, 
              graph_version_id: str = "", mode: str = "learning", 
              use_websocket: bool = False, thread_id_for_analysis: str = None):
    """
    Process user input using the learning-based tools and stream the response.
    
    This function uses tool_learning.py's process_llm_with_tools function which implements
    active learning: understanding user messages, checking similarity with existing knowledge,
    and offering to save new knowledge.
    
    Args:
        user_input: The user's message
        user_id: User identifier
        thread_id: Conversation thread identifier
        graph_version_id: Graph version identifier for knowledge retrieval
        mode: Processing mode ('learning')
        use_websocket: Whether to use WebSocket for analysis streaming
        thread_id_for_analysis: Thread ID to use for WebSocket analysis events
    
    Returns:
        A generator yielding response chunks
    """
    start_time = time.time()
    thread_id = thread_id or f"learning_thread_{int(time.time())}"
    user_id = user_id or "prof"

    # Validate user_input
    if not isinstance(user_input, str):
        logger.error(f"Invalid user_input type: {type(user_input)}, value: {user_input}")
        user_input = str(user_input) if user_input else ""
    if not user_input.strip():
        logger.error("Empty user_input")
        yield f"data: {json.dumps({'error': 'Empty message provided'})}\n\n"
        return
    
    # Load or init state following the same pattern as convo_stream
    checkpoint = checkpointer.get({"configurable": {"thread_id": thread_id}})
    default_state = {
        "messages": [],
        "prompt_str": "",
        "convo_id": thread_id,
        "last_response": "",
        "user_id": user_id,
        "preset_memory": "Be friendly",
        "instinct": "",
        "graph_version_id": graph_version_id,
        "analysis": {},
        "stream_events": [],
        "use_websocket": use_websocket,
        "thread_id_for_analysis": thread_id_for_analysis
    }
    state = {**default_state, **(checkpoint.get("channel_values", {}) if checkpoint else {})}

    # Add user input to state messages (this is how conversation history is maintained)
    if user_input:
        state["messages"] = add_messages(state["messages"], [HumanMessage(content=user_input)])

    logger.info(f"convo_stream_learning - User: {user_id}, Input: '{user_input}', Thread: {thread_id}")
    logger.info(f"DEBUG AMI: Loaded state with {len(state['messages'])} messages from checkpoint")
    
    # Convert messages to conversation_context format for tool_learning compatibility
    conversation_context = ""
    if state["messages"]:
        recent_messages = []
        message_count = 0
        max_messages = 50
        
        # Process messages in reverse to get most recent first, then reverse back for chronological order
        for msg in reversed(state["messages"]):
            try:
                if hasattr(msg, 'content'):
                    content = msg.content
                    if isinstance(msg, HumanMessage):
                        recent_messages.append(f"User: {content.strip()}")
                        message_count += 1
                    elif isinstance(msg, AIMessage):
                        recent_messages.append(f"AI: {content.strip()}")
                        message_count += 1
                    
                    if message_count >= max_messages:
                        if len(state["messages"]) > max_messages:
                            recent_messages.append(f"[Note: Conversation history truncated. Total messages: {len(state['messages'])}]")
                        break
            except Exception as e:
                logger.warning(f"Error processing message in conversation history: {e}")
                continue
        
        if recent_messages:
            conversation_history_text = "\n\n".join(reversed(recent_messages))
            conversation_context = f"{conversation_history_text}"
            logger.info(f"DEBUG AMI: Built conversation_context with {len(recent_messages)} messages")

    # Configuration for learning processor
    config = {
        "configurable": {
            "thread_id": thread_id,
            "user_id": user_id,
            "graph_version_id": graph_version_id,
            "use_websocket": use_websocket,
            "thread_id_for_analysis": thread_id_for_analysis
        }
    }
    
    # Use tool_learning approach but with proper state management
    try:
        from ava import AVA
        
        # Create AVA instance for this request (no singleton needed with proper state management)
        ava = AVA()
        await ava.initialize()
        
        # Process the message using the streaming learning processor
        final_response = None
        logger.info(f"DEBUG AMI: About to call ava.read_human_input with conversation_context length: {len(conversation_context)}")
        
        async for chunk in ava.read_human_input(
            user_input,
            conversation_context,
            user_id=user_id,
            thread_id=thread_id,
            use_websocket=use_websocket,
            thread_id_for_analysis=thread_id_for_analysis
        ):
            if chunk.get("type") == "response_chunk":
                # Stream chunks immediately
                yield f"data: {json.dumps({'status': 'streaming', 'content': chunk['content'], 'complete': False})}\n\n"
            elif chunk.get("type") == "response_complete":
                # Store final response for post-processing
                final_response = chunk
            elif chunk.get("type") == "error":
                yield f"data: {json.dumps(chunk)}\n\n"
                return

        # Add AI response to state messages
        if final_response and isinstance(final_response, dict) and "message" in final_response:
            message_content = final_response["message"]
            # Strip knowledge queries from saved message
            message_content = re.split(r'<knowledge_queries>', message_content)[0].strip()
            state["messages"] = add_messages(state["messages"], [AIMessage(content=message_content)])
            state["prompt_str"] = message_content
            
            # Update the checkpoint with new state (this maintains conversation history)
            await convo_graph.aupdate_state({"configurable": {"thread_id": thread_id}}, state, as_node="mc")
            logger.info(f"DEBUG AMI: Updated state with new messages, total count: {len(state['messages'])}")

        # Execute any tool calls found in the response (simplified version)
        if final_response and isinstance(final_response, dict) and "metadata" in final_response:
            metadata = final_response.get("metadata", {})
            if "tool_calls" in metadata:
                tool_calls = metadata["tool_calls"]
                logger.info(f"Found {len(tool_calls)} tool calls to execute")
                
                for tool_call in tool_calls:
                    if isinstance(tool_call, dict) and "name" in tool_call:
                        tool_name = tool_call.get("name")
                        logger.info(f"Executing tool call: {tool_name}")
                        # Tool execution can happen in background
                        # For now, just log that we found tool calls

        # Yield final response
        if final_response:
            yield f"data: {json.dumps(final_response)}\n\n"
        
    except Exception as e:
        logger.error(f"Error in convo_stream_learning: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    logger.debug(f"convo_stream_learning took {time.time() - start_time:.2f}s")