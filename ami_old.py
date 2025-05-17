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
import uuid


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
              use_websocket: bool = False, thread_id_for_analysis: str = None,
              request_id: str = None):
    """
    Process user input and stream the response.
    
    Args:
        user_input: The user's message
        user_id: User identifier
        thread_id: Conversation thread identifier
        graph_version_id: Graph version identifier for knowledge retrieval
        mode: Processing mode ('mc' or 'conversation')
        use_websocket: Whether to use WebSocket for analysis streaming
        thread_id_for_analysis: Thread ID to use for WebSocket analysis events
        request_id: Unique identifier for this specific request (for concurrent processing)
    
    Returns:
        A generator yielding response chunks
    """
    start_time = time.time()
    thread_id = thread_id or f"chat_thread_{int(time.time())}"
    user_id = user_id or "user"
    request_id = request_id or str(uuid.uuid4())  # Ensure we have a request ID

    # Load conversation checkpoint if available
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
        "thread_id_for_analysis": thread_id_for_analysis,  # Thread ID for WebSocket analysis events
        "request_id": request_id  # Add the request ID for tracking
    }
    state = {**default_state, **(checkpoint.get("channel_values", {}) if checkpoint else {})}
    
    # Always ensure request_id is set in state
    state["request_id"] = request_id

    # Add user input
    if user_input:
        state["messages"] = add_messages(state["messages"], [HumanMessage(content=user_input)])

    logger.debug(f"convo_stream init - Request: {request_id}, Input: '{user_input}', Convo ID: {thread_id}")

    # Configuration for graph
    config = {
        "configurable": {
            "thread_id": thread_id,
            "user_id": user_id,
            "graph_version_id": graph_version_id,
            "use_websocket": use_websocket,
            "thread_id_for_analysis": thread_id_for_analysis,
            "request_id": request_id
        }
    }
    
    # Process state in a completely isolated thread
    import queue
    import threading
    
    result_queue = queue.Queue()
    
    def worker_thread():
        # Create a new event loop for this thread
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            async def process_graph():
                # Invoke the graph with the state
                updated_state = await convo_graph.ainvoke(state, config)
                
                # Use request_id to create unique checkpoint identity
                # This allows concurrent updates to state without conflicts
                checkpoint_id = {"configurable": {
                    "thread_id": thread_id,
                    "request_id": request_id
                }}
                
                # Update state with the unique checkpoint ID to avoid conflicts
                await convo_graph.aupdate_state(checkpoint_id, updated_state, as_node="mc")
                return updated_state
            
            # Run the async function on this thread's event loop
            result = loop.run_until_complete(process_graph())
            
            # Put the result in the queue
            result_queue.put({"success": True, "state": result})
            
        except Exception as e:
            logger.error(f"Error in worker thread: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            result_queue.put({"success": False, "error": str(e)})
            
        finally:
            # Clean up any pending tasks
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            
            # Run loop until tasks are cancelled
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                
            # Close the loop
