import json
import time
from typing import Annotated, List, Dict
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from training import Training
from pilot import Pilot
from mc_tools import MCWithTools  # Import MCWithTools instead of MC
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