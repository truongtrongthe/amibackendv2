# ami.py
# Purpose: Simple state and graph harness for intent testing with AmiCore
# Date: March 22, 2025

import json
import time
from typing import Annotated, List, Dict
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from ami_core import Ami  # Ensure this points to the correct file
from utilities import logger
import asyncio

# State - Aligned with AmiCore, including intent_history
class State(TypedDict):
    messages: Annotated[list, add_messages]
    prompt_str: str
    convo_id: str
    active_terms: dict
    pending_knowledge: dict
    last_response: str
    user_id: str
    needs_confirmation: bool
    intent: str  # Kept for compatibility, though less critical with intent_history
    current_focus: str
    intent_history: List[Dict[str, float]]  # New: Tracks soft intent history

# Graph
ami_core = Ami()
graph_builder = StateGraph(State)

# Node: Async AmiCore.do with dynamic confirmation
async def ami_node(state: State, config=None):
    start_time = time.time()
    user_id = config.get("configurable", {}).get("user_id", "unknown") if config else "unknown"
    logger.info(f"ami_node - User ID: {user_id}")

    def confirm_callback(prompt):
        if "lưu không bro?" in prompt.lower():
            # Check current_focus or latest message for context
            if state.get("current_focus", "").lower() in state["messages"][-1].content.lower() or "hito" in state["messages"][-1].content.lower():
                logger.debug(f"Confirming save for {state.get('current_focus', 'unknown')}")
                return "yes"
            return "yes"  # Default to yes for teaching unless explicitly contradicted
        logger.debug("No confirmation needed")
        return "no"

    updated_state = await ami_core.do(state, not state.get("messages", []), confirm_callback=confirm_callback, user_id=user_id)
    logger.debug(f"ami_node took {time.time() - start_time:.2f}s")
    return updated_state

graph_builder.add_node("ami", ami_node)
graph_builder.add_edge(START, "ami")
graph_builder.add_edge("ami", END)
checkpointer = MemorySaver()
convo_graph = graph_builder.compile(checkpointer=checkpointer)

# Simple convo stream
def convo_stream(user_input=None, user_id=None, thread_id=None):
    start_time = time.time()
    thread_id = thread_id or f"test_thread_{int(time.time())}"
    user_id = user_id or "tfl_default"

    # Load or init state with intent_history
    checkpoint = checkpointer.get({"configurable": {"thread_id": thread_id}})
    default_state = {
        "messages": [],
        "prompt_str": "",
        "convo_id": thread_id,
        "active_terms": {},
        "pending_knowledge": {},
        "last_response": "",
        "user_id": user_id,
        "needs_confirmation": False,
        "intent": "",
        "current_focus": "",
        "intent_history": []  # New: Initialize intent_history
    }
    state = {**default_state, **(checkpoint.get("channel_values", {}) if checkpoint else {})}

    # Add user input
    if user_input:
        state["messages"] = add_messages(state["messages"], [HumanMessage(content=user_input)])

    logger.debug(f"convo_stream init - Input: '{user_input}', Convo ID: {thread_id}")

    # Run async graph synchronously
    async def process_state():
        config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
        updated_state = await convo_graph.ainvoke(state, config)
        await convo_graph.aupdate_state({"configurable": {"thread_id": thread_id}}, updated_state, as_node="ami")
        return updated_state

    state = asyncio.run(process_state())

    logger.debug(f"convo_stream total took {time.time() - start_time:.2f}s")

    # Stream response
    response_lines = state["prompt_str"].split('\n')
    for line in response_lines:
        if line.strip():
            yield f"data: {json.dumps({'message': line.strip()})}\n\n"
            time.sleep(0.05)

# Test the full 6-turn flow (circling Hạ Long)
if __name__ == "__main__":
    turns = [
        "Xin chào Ami!",
        "Hạ Long đẹp không?",
        "Ý anh là cảnh ở đó ấy",
        "Nói thêm về Hạ Long đi",
        "Mấy cái vịnh ở đó thế nào?",
        "Hạ Long có gì đặc biệt nữa không?"
    ]
    user_id = "TFL"
    thread_id = "teaching"
    print("\n=== 6-Turn Flow Test ===")
    for i, msg in enumerate(turns, 1):
        print(f"\nTurn {i}: {msg}")
        for chunk in convo_stream(msg, user_id, thread_id):
            data = json.loads(chunk.split("data: ")[1])
            print(f"Response: {data['message']}")