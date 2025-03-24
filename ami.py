# ami.py
# Purpose: State and graph harness with dual nodes for Teaching and Copilot modes
# Date: March 23, 2025

import json
import time
from typing import Annotated, List, Dict
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from ami_core import Ami
from utilities import logger
import asyncio

class State(TypedDict):
    messages: Annotated[list, add_messages]
    prompt_str: str
    convo_id: str
    user_id: str
    intent_history: List[Dict[str, float]]
    preset_memory: str  # Added to state

teaching_ami = Ami(mode="teaching")
copilot_ami = Ami(mode="copilot")
pretrain_ami = Ami(mode="pretrain")
graph_builder = StateGraph(State)

async def teaching_node(state: State, config=None):
    start_time = time.time()
    user_id = config.get("configurable", {}).get("user_id", "unknown") if config else "unknown"
    logger.info(f"teaching_node - User ID: {user_id}")
    updated_state = await teaching_ami.do(state, user_id=user_id)
    logger.debug(f"teaching_node took {time.time() - start_time:.2f}s")
    return updated_state

async def pretrain_node(state: State, config=None):
    start_time = time.time()
    user_id = config.get("configurable", {}).get("user_id", "thefusionlab") if config else "thefusionlab"
    logger.info(f"pretrain_node - User ID: {user_id}")
    updated_state = await pretrain_ami.pretrain(state, user_id=user_id)
    logger.debug(f"pretrain_node took {time.time() - start_time:.2f}s")
    return updated_state

async def copilot_node(state: State, config=None):
    start_time = time.time()
    user_id = config.get("configurable", {}).get("user_id", "unknown") if config else "unknown"
    logger.info(f"copilot_node - User ID: {user_id}")
    updated_state = await copilot_ami.do(state, user_id=user_id)
    logger.debug(f"copilot_node took {time.time() - start_time:.2f}s")
    return updated_state

graph_builder.add_node("teaching", teaching_node)
graph_builder.add_node("copilot", copilot_node)
graph_builder.add_node("pretrain", pretrain_node)

def route_mode(state: State, config: Dict) -> str:
    mode = config.get("configurable", {}).get("mode", "teaching")
    return mode

graph_builder.add_conditional_edges(START, route_mode, {"teaching": "teaching", "copilot": "copilot", "pretrain": "pretrain"})
graph_builder.add_edge("teaching", END)
graph_builder.add_edge("copilot", END)
graph_builder.add_edge("pretrain", END)

checkpointer = MemorySaver()
convo_graph = graph_builder.compile(checkpointer=checkpointer)

def convo_stream(user_input=None, user_id=None, thread_id=None, mode="teaching"):
    start_time = time.time()
    thread_id = thread_id or f"thread_{int(time.time())}"
    user_id = user_id or "thefusionlab"

    logger.info(f"running at MODE={mode}")
    checkpoint = checkpointer.get({"configurable": {"thread_id": thread_id}})
    default_state = {
        "messages": [],
        "prompt_str": "",
        "convo_id": thread_id,
        "user_id": user_id,
        "intent_history": [],
        "preset_memory": teaching_ami.state["preset_memory"] if mode == "teaching" else copilot_ami.state["preset_memory"] if mode == "copilot" else pretrain_ami.state["preset_memory"]
    }
    state = {**default_state, **(checkpoint.get("channel_values", {}) if checkpoint else {})}

    if user_input:
        state["messages"] = add_messages(state["messages"], [HumanMessage(content=user_input)])

    logger.debug(f"convo_stream init - Input: '{user_input}', Convo ID: {thread_id}, Mode: {mode}")

    async def process_state():
        config = {"configurable": {"thread_id": thread_id, "user_id": user_id, "mode": mode}}
        updated_state = await convo_graph.ainvoke(state, config)
        await convo_graph.aupdate_state({"configurable": {"thread_id": thread_id}}, updated_state, as_node=mode)
        return updated_state

    state = asyncio.run(process_state())

    logger.debug(f"convo_stream total took {time.time() - start_time:.2f}s")

    response_lines = state["prompt_str"].split('\n')
    for line in response_lines:
        if line.strip():
            yield f"data: {json.dumps({'message': line.strip()})}\n\n"
            time.sleep(0.05)