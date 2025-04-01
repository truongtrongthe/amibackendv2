# ami.py
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
from mc import MC  # Import MC directly
from utilities import logger
import asyncio

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

async def mc_node(state: State, config=None):
    start_time = time.time()
    user_id = config.get("configurable", {}).get("user_id", "thefusionlab")
    logger.info(f"MC node - User ID: {user_id}")
    
    if not mc.instincts:
        await mc.initialize()
    
    async for response_chunk in mc.trigger(state=state, user_id=user_id):
        state["prompt_str"] = response_chunk
        state["unresolved_requests"] = mc.state["unresolved_requests"]  # Sync from MC instance
        logger.info(f"Streaming chunk from mc_node: {response_chunk}")
        yield {"prompt_str": response_chunk, "unresolved_requests": state["unresolved_requests"]}
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


# ami.py (partial update)


async def convo_stream(user_input: str = None, user_id: str = None, thread_id: str = None, mode: str = "mc"):
    start_time = time.time()
    thread_id = thread_id or f"thread_{int(time.time())}"
    user_id = user_id or "thefusionlab"

    logger.info(f"Running in {mode} mode for user {user_id}, thread {thread_id}")

    checkpoint = checkpointer.get({"configurable": {"thread_id": thread_id}})
    default_state = {
        "messages": [],
        "prompt_str": "",
        "convo_id": thread_id,
        "user_id": user_id,
        "intent_history": [],
        "preset_memory": "Be friendly",
        "instinct": "",
        "unresolved_requests": mc.state.get("unresolved_requests", [])  # Initialize from STATE_STORE
    }
    state = {**default_state, **(checkpoint.get("channel_values", {}) if checkpoint else {})}

    if user_input:
        state["messages"] = add_messages(state["messages"], [HumanMessage(content=user_input)])

    config = {"configurable": {"thread_id": thread_id, "user_id": user_id, "mode": mode}}

    async for event in convo_graph.astream(state, config, stream_mode="updates"):
        logger.info(f"Raw event: {event}")
        if mode == "mc" and "mc" in event:
            prompt_str = event["mc"].get("prompt_str", "")
            unresolved_requests = event["mc"].get("unresolved_requests", mc.state["unresolved_requests"])
            if prompt_str:
                logger.info(f"Streaming prompt_str: {prompt_str}")
                yield f"data: {json.dumps({'message': prompt_str})}\n\n"
                await asyncio.sleep(0.01)
                state["prompt_str"] = prompt_str
                state["unresolved_requests"] = unresolved_requests
                await convo_graph.aupdate_state({"configurable": {"thread_id": thread_id}}, state, as_node="mc")

    yield "data: [DONE]\n\n"
    logger.debug(f"convo_stream total took {time.time() - start_time:.2f}s")