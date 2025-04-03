# ami.py
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
from mc import MC  # Import MC directly
from utilities import logger
import asyncio

STATE_STORE = {}  # Thread-based state storage

class State(TypedDict):
    messages: Annotated[list, add_messages]
    prompt_str: str
    convo_id: str
    user_id: str
    intent_history: List[str]
    preset_memory: str
    instinct: str

training_ami = Training(user_id="thefusionlab")
pilot_ami = Pilot(user_id="brian")
mc = MC(user_id="thefusionlab")

graph_builder = StateGraph(State)

async def training_node(state: State, config=None):
    start_time = time.time()
    user_id = config.get("configurable", {}).get("user_id", "thefusionlab") if config else "thefusionlab"
    logger.info(f"training_node - User ID: {user_id}")
    if not training_ami.instincts:
        await training_ami.initialize()
    updated_state = await training_ami.training(state=state, user_id=user_id)
    logger.debug(f"training_node took {time.time() - start_time:.2f}s")
    return updated_state

async def pilot_node(state: State, config=None):
    start_time = time.time()
    user_id = config.get("configurable", {}).get("user_id", "brian") if config else "brian"
    logger.info(f"pilot_node - User ID: {user_id}")
    if not pilot_ami.instincts:
        await pilot_ami.initialize()
    updated_state = await pilot_ami.pilot(state=state, user_id=user_id)
    logger.debug(f"pilot_node took {time.time() - start_time:.2f}s")
    return updated_state
# ami.py (partial update)
async def mc_node(state: State, config=None):
    start_time = time.time()
    config = config or {}
    user_id = config.get("configurable", {}).get("user_id", "thefusionlab")
    bank_name = config.get("configurable", {}).get("bank_name", "")
    thread_id = config.get("configurable", {}).get("thread_id", "default_thread")
    logger.info(f"MC node - User ID: {user_id}")

    if not mc.instincts:
        await mc.initialize()
    
    # Ensure bank_name in mc.state
    mc.state["bank_name"] = bank_name
    
    async for response_chunk in mc.trigger(state=state, user_id=user_id, bank_name=bank_name, config=config):
        state["prompt_str"] = response_chunk
        state["unresolved_requests"] = mc.state["unresolved_requests"]  # From mc.state
        state["bank_name"] = mc.state["bank_name"]  # Sync from mc.state
        logger.info(f"Streaming chunk from mc_node: {response_chunk}")
        yield {
            "prompt_str": response_chunk,
            "unresolved_requests": state["unresolved_requests"],
            "bank_name": state["bank_name"]
        }
        # Persist state to graph
        await convo_graph.aupdate_state({"configurable": {"thread_id": thread_id}}, state, as_node="mc")
    
    logger.debug(f"mc node took {time.time() - start_time:.2f}s")


graph_builder.add_node("training", training_node)
graph_builder.add_node("pilot", pilot_node)
graph_builder.add_node("mc", mc_node)

def route_by_mode(state: State, config=None):
    mode = config.get("configurable", {}).get("mode", "training")
    if mode == "training":
        return "training"
    elif mode == "pilot":
        return "pilot"
    else:
        return "mc"

graph_builder.add_conditional_edges(START, route_by_mode, {"training": "training", "pilot": "pilot", "mc": "mc"})
graph_builder.add_edge("training", END)
graph_builder.add_edge("pilot", END)
graph_builder.add_edge("mc", END)

checkpointer = MemorySaver()
convo_graph = graph_builder.compile(checkpointer=checkpointer)

async def convo_stream(user_input: str = None, user_id: str = None, thread_id: str = None, bank_name: str = "", mode: str = "mc"):
    start_time = time.time()
    thread_id = thread_id or f"thread_{int(time.time())}"
    user_id = user_id or "thefusionlab"

    logger.info(f"Running in {mode} mode for user {user_id}, thread {thread_id}")

    # Minimal state, avoid messages channel
    state = {
        "prompt_str": "",
        "convo_id": thread_id,
        "user_id": user_id,
        "intent_history": [],
        "preset_memory": "Be friendly",
        "instinct": "",
        "bank_name": bank_name,
        "unresolved_requests": mc.state.get("unresolved_requests", [])
    }
    messages = [HumanMessage(content=user_input)] if user_input else []  # Manage messages separately

    config = {"configurable": {"thread_id": thread_id, "user_id": user_id, "mode": mode, "bank_name": bank_name}}

    logger.info(f"State before astream: {json.dumps(state, default=str)}")
    async for event in convo_graph.astream(state, config, stream_mode="updates"):
        logger.info(f"Raw event: {event}")
        if mode == "mc" and "mc" in event:
            if not isinstance(event["mc"], dict):
                logger.warning(f"Skipping invalid 'mc' event: {event['mc']}")
                continue
            
            # Unwrap prompt_str
            prompt_data = event["mc"].get("prompt_str", "")
            while isinstance(prompt_data, dict) and "mc" in prompt_data:
                prompt_data = prompt_data["mc"].get("prompt_str", "")
            prompt_str = prompt_data if isinstance(prompt_data, str) else ""
            
            unresolved_requests = event["mc"].get("unresolved_requests", mc.state["unresolved_requests"])
            event_bank_name = event["mc"].get("bank_name", bank_name)

            if unresolved_requests and isinstance(unresolved_requests, list):
                for req in unresolved_requests:
                    if "bank_name" not in req or not req["bank_name"]:
                        req["bank_name"] = event_bank_name
            
            # Update state and messages
            if prompt_str:
                state["prompt_str"] = prompt_str
                messages.append(AIMessage(content=prompt_str))  # Append to separate list
                logger.info(f"Streaming prompt_str: {prompt_str}")
                yield f"data: {json.dumps({'message': prompt_str, 'bank_name': event_bank_name})}\n\n"
                await asyncio.sleep(0.01)
            else:
                logger.debug("No prompt_str to stream")
            
            state["unresolved_requests"] = unresolved_requests
            state["bank_name"] = event_bank_name
            # Update state without messages channel
            await convo_graph.aupdate_state({"configurable": {"thread_id": thread_id}}, state, as_node="mc")
        
        logger.debug(f"State after event: {json.dumps(state, default=str)}")
        logger.debug(f"Messages list: {json.dumps(messages, default=str)}")

    yield "data: [DONE]\n\n"
    # Final state sync with messages
    state["messages"] = messages
    STATE_STORE[thread_id] = state
    logger.debug(f"Final state saved: {json.dumps(state, default=str)}")