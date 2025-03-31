# ami.py
# Purpose: Simplified state and graph harness for Training, Pilot, and Funny modes
# Date: March 28, 2025 (Updated March 31, 2025)

import json
import time
from typing import Annotated, List, Dict
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from training import Training
from pilot import Pilot
from fun import Fun
from utilities import logger
import asyncio

# Define the state structure
class State(TypedDict):
    messages: Annotated[list, add_messages]
    prompt_str: str
    convo_id: str
    user_id: str
    intent_history: List[str]
    preset_memory: str
    instinct: str

# Initialize instances
training_ami = Training(user_id="thefusionlab")
pilot_ami = Pilot(user_id="brian")
funnyguy = Fun(user_id="thefusionlab")

# Set up the graph
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

async def fun_node(state: State, config=None):
    start_time = time.time()
    user_id = config.get("configurable", {}).get("user_id", "thefusionlab")
    logger.info(f"funny node - User ID: {user_id}")
    if not funnyguy.instincts:
        await funnyguy.initialize()
    
    # Yield each response chunk directly as it’s generated
    async for response_chunk in funnyguy.havefun(state=state, user_id=user_id):
        state["prompt_str"] = response_chunk
        logger.info(f"Streaming chunk from fun_node: {response_chunk}")
        yield {"prompt_str": response_chunk}  # Yield a dict compatible with astream
    logger.debug(f"fun node took {time.time() - start_time:.2f}s")

# Add nodes to the graph
graph_builder.add_node("training", training_node)
graph_builder.add_node("pilot", pilot_node)
graph_builder.add_node("funny", fun_node)

# Define routing logic based on mode
def route_by_mode(state: State, config=None):
    mode = config.get("configurable", {}).get("mode", "training")
    if mode == "training":
        return "training"
    elif mode == "pilot":
        return "pilot"
    else:
        return "funny"

# Add edges
graph_builder.add_conditional_edges(START, route_by_mode, {"training": "training", "pilot": "pilot", "funny": "funny"})
graph_builder.add_edge("training", END)
graph_builder.add_edge("pilot", END)
graph_builder.add_edge("funny", END)

# Set up memory persistence
checkpointer = MemorySaver()
convo_graph = graph_builder.compile(checkpointer=checkpointer)

async def convo_stream(user_input: str = None, user_id: str = None, thread_id: str = None, mode: str = "training"):
    start_time = time.time()
    thread_id = thread_id or f"thread_{int(time.time())}"
    user_id = user_id or ("thefusionlab" if mode == "training" else "thefusionlab")

    logger.info(f"Running in {mode} mode for user {user_id}, thread {thread_id}")

    checkpoint = checkpointer.get({"configurable": {"thread_id": thread_id}})
    default_state = {
        "messages": [],
        "prompt_str": "",
        "convo_id": thread_id,
        "user_id": user_id,
        "intent_history": [],
        "preset_memory": "Be friendly",
        "instinct": ""
    }
    state = {**default_state, **(checkpoint.get("channel_values", {}) if checkpoint else {})}

    if user_input:
        state["messages"] = add_messages(state["messages"], [HumanMessage(content=user_input)])

    logger.debug(f"convo_stream init - Input: '{user_input}', Convo ID: {thread_id}, Mode: {mode}")

    config = {"configurable": {"thread_id": thread_id, "user_id": user_id, "mode": mode}}

    async for event in convo_graph.astream(state, config):
        if mode == "funny":
            # Extract prompt_str from fun_node’s yielded dict
            prompt_str = event.get("funny", {}).get("prompt_str", "")
        else:
            prompt_str = (
                event.get("training", {}).get("prompt_str") or
                event.get("pilot", {}).get("prompt_str") or
                ""
            )
        
        if prompt_str:
            logger.info(f"Streaming prompt_str: {prompt_str}")
            yield f"data: {json.dumps({'message': prompt_str})}\n\n"
            await asyncio.sleep(0.01)  # Small delay for smooth streaming
            # Update state after each chunk
            await convo_graph.aupdate_state({"configurable": {"thread_id": thread_id}}, {"prompt_str": prompt_str}, as_node=mode)

    yield "data: [DONE]\n\n"
    logger.debug(f"convo_stream total took {time.time() - start_time:.2f}s")