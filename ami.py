import json
import time
import re  # Add re import for regex
from typing import Annotated, List, Dict
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage

from pilot import Pilot
from mc_tools import MCWithTools  # Standard tools
from utilities import logger
import asyncio


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
graph_builder = StateGraph(State)

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
    logger.info(f"Final state has analysis: {'analysis' in final_state}")
    
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

graph_builder.add_node("mc", mc_node)
graph_builder.add_edge(START, "mc")
graph_builder.add_edge("mc", END)

checkpointer = MemorySaver()
convo_graph = graph_builder.compile(checkpointer=checkpointer)

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

# Add new learning-based stream function
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
    user_id = user_id or "user"

    # Validate user_input
    if not isinstance(user_input, str):
        logger.error(f"Invalid user_input type: {type(user_input)}, value: {user_input}")
        user_input = str(user_input) if user_input else ""
    if not user_input.strip():
        logger.error("Empty user_input")
        yield f"data: {json.dumps({'error': 'Empty message provided'})}\n\n"
        return
    
    # Convert conversation history to the format expected by tool_learning
    conversation_history = []
    
    # Load conversation checkpoint if available
    checkpoint = checkpointer.get({"configurable": {"thread_id": thread_id}})
    if checkpoint and "channel_values" in checkpoint and "messages" in checkpoint["channel_values"]:
        messages = checkpoint["channel_values"]["messages"]
        for msg in messages:
            if isinstance(msg, HumanMessage):
                conversation_history.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                conversation_history.append({"role": "assistant", "content": msg.content})
    
    # Initialize state
    state = {
        "messages": conversation_history,
        "user_id": user_id,
        "graph_version_id": graph_version_id,
        "use_websocket": use_websocket,
        "thread_id_for_analysis": thread_id_for_analysis
    }
    
    logger.info(f"convo_stream_learning - User: {user_id}, Input: '{user_input}', Thread: {thread_id}")
    
    # Use our tool_learning.py's process_llm_with_tools function to process the message
    try:
        # Import directly from tool_learning to ensure we're using the updated version
        from tool_learning import LearningProcessor
        
        # Create a learning processor instance directly
        learning_processor = LearningProcessor()
        await learning_processor.initialize()
        
        # Process the message using the learning processor
        response = await learning_processor.process_incoming_message(
            user_input,
            conversation_context="",  # Initialize with empty context
            user_id=user_id,
            thread_id=thread_id
        )
        
        # Execute any tool calls found in the response
        if isinstance(response, dict) and "metadata" in response and "tool_calls" in response["metadata"]:
            tool_calls = response["metadata"]["tool_calls"]
            logger.info(f"Found {len(tool_calls)} tool calls to execute")
            
            # Execute each tool call
            for tool_call in tool_calls:
                if isinstance(tool_call, dict) and "name" in tool_call and "parameters" in tool_call:
                    tool_name = tool_call["name"]
                    parameters = tool_call["parameters"]
                    
                    # Make sure user_id is included in parameters
                    if "user_id" not in parameters:
                        parameters["user_id"] = user_id
                    
                    # Make sure thread_id is included in parameters if applicable
                    if "thread_id" not in parameters and tool_name != "knowledge_query":
                        parameters["thread_id"] = thread_id
                    
                    logger.info(f"Executing tool call: {tool_name} with parameters: {parameters}")
                    
                    # Execute the tool call
                    if tool_name == "save_knowledge":
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
                                continue
                                
                            # Get optional parameters
                            user_id = parameters.get("user_id", user_id)
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
                            
                            logger.info(f"Saving knowledge: '{input_text[:50]}...' for user {user_id}")
                            
                            # Execute save_knowledge
                            success = await save_knowledge(
                                input=input_text,
                                user_id=user_id,
                                bank_name=bank_name,
                                thread_id=thread_id_param,
                                topic=topic,
                                categories=categories
                            )
                            logger.info(f"Save knowledge result: {success}")
                            
                        except Exception as e:
                            logger.error(f"Error executing save_knowledge: {str(e)}")
                    
                    elif tool_name == "knowledge_query":
                        logger.info("Skipping knowledge_query tool call in response processing")
                    
                    else:
                        logger.warning(f"Unknown tool call: {tool_name}")
                else:
                    logger.warning(f"Invalid tool call format: {tool_call}")

        # Yield the response
        if isinstance(response, dict) and "message" in response:
            # Strip out knowledge queries section from the message
            message_content = response["message"]
            # Split at knowledge_queries tag and take only the first part
            message_content = re.split(r'<knowledge_queries>', message_content)[0].strip()
            logger.info("Stripped knowledge_queries from message before sending to frontend")
            
            yield f"data: {json.dumps({'message': message_content})}\n\n"
        else:
            message_content = str(response)
            # Also strip knowledge queries from the string representation
            message_content = re.split(r'<knowledge_queries>', message_content)[0].strip()
            yield f"data: {json.dumps({'message': message_content})}\n\n"
        
    except Exception as e:
        logger.error(f"Error in convo_stream_learning: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    logger.debug(f"convo_stream_learning took {time.time() - start_time:.2f}s")