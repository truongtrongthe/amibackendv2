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
from ami_core import Ami  # Points to latest ami_core.py
from utilities import logger
import asyncio

# State - Aligned with AmiCore
class State(TypedDict):
    messages: Annotated[list, add_messages]
    prompt_str: str
    convo_id: str
    user_id: str
    intent_history: List[Dict[str, float]]

# Graph Setup
teaching_ami = Ami(mode="teaching")  # Instance for Teaching Mode
copilot_ami = Ami(mode="copilot")    # Instance for Copilot Mode
graph_builder = StateGraph(State)

# Node: Teaching Mode
async def teaching_node(state: State, config=None):
    start_time = time.time()
    user_id = config.get("configurable", {}).get("user_id", "unknown") if config else "unknown"
    logger.info(f"teaching_node - User ID: {user_id}")
    updated_state = await teaching_ami.do(state, user_id=user_id)
    logger.debug(f"teaching_node took {time.time() - start_time:.2f}s")
    return updated_state

# Node: Copilot Mode
async def copilot_node(state: State, config=None):
    start_time = time.time()
    user_id = config.get("configurable", {}).get("user_id", "unknown") if config else "unknown"
    logger.info(f"copilot_node - User ID: {user_id}")
    updated_state = await copilot_ami.do(state, user_id=user_id)
    logger.debug(f"copilot_node took {time.time() - start_time:.2f}s")
    return updated_state

# Define graph with routing
graph_builder.add_node("teaching", teaching_node)
graph_builder.add_node("copilot", copilot_node)

# Conditional routing based on config mode
def route_mode(state: State, config: Dict) -> str:
    mode = config.get("configurable", {}).get("mode", "teaching")  # Default to teaching
    return "teaching" if mode == "teaching" else "copilot"

graph_builder.add_conditional_edges(START, route_mode, {"teaching": "teaching", "copilot": "copilot"})
graph_builder.add_edge("teaching", END)
graph_builder.add_edge("copilot", END)

checkpointer = MemorySaver()
convo_graph = graph_builder.compile(checkpointer=checkpointer)

# Convo Stream Function
def convo_stream(user_input=None, user_id=None, thread_id=None, mode="teaching"):
    start_time = time.time()
    thread_id = thread_id or f"test_thread_{int(time.time())}"
    user_id = user_id or "tfl_default"

    # Load or init state
    checkpoint = checkpointer.get({"configurable": {"thread_id": thread_id}})
    default_state = {
        "messages": [],
        "prompt_str": "",
        "convo_id": thread_id,
        "user_id": user_id,
        "intent_history": []
    }
    state = {**default_state, **(checkpoint.get("channel_values", {}) if checkpoint else {})}

    # Add user input if provided
    if user_input:
        state["messages"] = add_messages(state["messages"], [HumanMessage(content=user_input)])

    logger.debug(f"convo_stream init - Input: '{user_input}', Convo ID: {thread_id}, Mode: {mode}")

    # Run async graph synchronously
    async def process_state():
        config = {"configurable": {"thread_id": thread_id, "user_id": user_id, "mode": mode}}
        updated_state = await convo_graph.ainvoke(state, config)
        await convo_graph.aupdate_state({"configurable": {"thread_id": thread_id}}, updated_state, as_node=mode)
        return updated_state

    state = asyncio.run(process_state())

    logger.debug(f"convo_stream total took {time.time() - start_time:.2f}s")

    # Stream response
    response_lines = state["prompt_str"].split('\n')
    for line in response_lines:
        if line.strip():
            yield f"data: {json.dumps({'message': line.strip()})}\n\n"
            time.sleep(0.05)

# Test both modes
if __name__ == "__main__":
    # Teaching Mode Test
    teaching_turns = [
        "Xin chào Ami!",
        "Khi bán nhà, nhớ kiểm tra tài sản trước.",
        "Bán nhà cho Mike đi.",
    ]
    user_id = "TFL"
    thread_id = "teaching_test"
    print("\n=== Teaching Mode Test ===")
    for i, msg in enumerate(teaching_turns, 1):
        print(f"\nTurn {i}: {msg}")
        for chunk in convo_stream(msg, user_id, thread_id, mode="teaching"):
            data = json.loads(chunk.split("data: ")[1])
            print(f"Response: {data['message']}")

    # Copilot Mode Test
    copilot_turns = [
        "Xin chào Ami!",
        "Bán nhà cho Mike đi.",
        "Chill chút đi nào!",
        "Hạ Long đẹp không?",
    ]
    thread_id = "copilot_test"
    print("\n=== Copilot Mode Test ===")
    for i, msg in enumerate(copilot_turns, 1):
        print(f"\nTurn {i}: {msg}")
        for chunk in convo_stream(msg, user_id, thread_id, mode="copilot"):
            data = json.loads(chunk.split("data: ")[1])
            print(f"Response: {data['message']}")