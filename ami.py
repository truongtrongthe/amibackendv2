# ami.py
# Purpose: State, graph, and test harness for intent stress testing with Ami Blue Print 3.4 Mark 3
# Date: March 15, 2025 (Updated for live on March 16, 2025)

import json
import time
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from ami_core import AmiCore

# State - Aligned with AmiCore and utilities.py
class State(TypedDict):
    messages: Annotated[list, add_messages]
    prompt_str: str
    convo_id: str
    active_terms: dict
    pending_node: dict
    pending_knowledge: dict
    brain: list
    sales_stage: str
    last_response: str  # For confirmation feedback

# Graph
ami_core = AmiCore()
graph_builder = StateGraph(State)

# Node: AmiCore.do with confirmation callback
def ami_node(state,config=None):
    # Pass a callback for confirmation—defaults to "yes" for testing
    #confirm_callback = lambda x: "yes" if "test" in state.get("convo_id", "") else None
    confirm_callback = lambda x: "yes"  # Always confirm for testing
    force_copilot = config.get("force_copilot", False) if config else False
    return ami_core.do(state, not state.get("messages", []), confirm_callback=confirm_callback,force_copilot=force_copilot)

graph_builder.add_node("ami", ami_node)
graph_builder.add_edge(START, "ami")
graph_builder.add_edge("ami", END)
checkpointer = MemorySaver()
convo_graph = graph_builder.compile(checkpointer=checkpointer)

def convo_stream(user_input=None, thread_id=f"test_thread_{int(time.time())}"):
    # Load or init state
    checkpoint = checkpointer.get({"configurable": {"thread_id": thread_id}})
    default_state = {
        "messages": [],
        "prompt_str": "",
        "convo_id": thread_id,  # Tie thread_id to convo_id for persistence
        "active_terms": {},
        "pending_node": {"pieces": [], "primary_topic": "Miscellaneous"},
        "pending_knowledge": {},
        "brain": ami_core.brain,
        "sales_stage": ami_core.sales_stages[0],
        "last_response": ""
    }
    state = {**default_state, **(checkpoint.get("channel_values", {}) if checkpoint else {})}
    
    # Add user input if provided
    if user_input:
        state["messages"] = add_messages(state["messages"], [HumanMessage(content=user_input)])
    
    # Debug state before processing
    print(f"Debug: Starting convo_stream - Input: '{user_input}', Stage: {state['sales_stage']}, Convo ID: {state['convo_id']}")
    
    # Pass to AmiCore.do for processing via graph
    state = convo_graph.invoke(state, {"configurable": {"thread_id": thread_id}})
    
    # Debug state after processing
    print(f"Debug: State after invoke - Prompt: '{state['prompt_str']}', Stage: {state['sales_stage']}, Last Response: {state.get('last_response', '')}")

    # Stream response
    response_lines = state["prompt_str"].split('\n')
    for line in response_lines:
        if line.strip():
            yield f"data: {json.dumps({'message': line.strip()})}\n\n"
    
    # Persist updated state
    convo_graph.update_state({"configurable": {"thread_id": thread_id}}, state, as_node="ami")

def pilot_stream(user_input=None, thread_id=f"copilot_thread_{int(time.time())}"):
    # Load or init state
    checkpoint = checkpointer.get({"configurable": {"thread_id": thread_id}})
    default_state = {
        "messages": [],
        "prompt_str": "",
        "convo_id": thread_id,
        "active_terms": {},
        "pending_node": {"pieces": [], "primary_topic": "Miscellaneous"},
        "pending_knowledge": {},
        "brain": ami_core.brain,
        "sales_stage": ami_core.sales_stages[0],
        "last_response": "",
        "copilot_task": user_input if user_input else None
    }
    state = {**default_state, **(checkpoint.get("channel_values", {}) if checkpoint else {})}
    
    if user_input:
        state["messages"] = add_messages(state["messages"], [HumanMessage(content=user_input)])
    
    print(f"Debug: Starting pilot_stream - Input: '{user_input}', Stage: {state['sales_stage']}, CoPilot Task: {state['copilot_task']}")
    
    # Combine checkpoint config and force_copilot into one config dict
    config = {
        "configurable": {"thread_id": thread_id},
        "force_copilot": True
    }
    state = convo_graph.invoke(state, config=config)
    
    print(f"Debug: State after invoke - Prompt: '{state['prompt_str']}', Stage: {state['sales_stage']}, Last Response: {state.get('last_response', '')}")
    
    response_lines = state["prompt_str"].split('\n')
    for line in response_lines:
        if line.strip():
            yield f"data: {json.dumps({'message': line.strip()})}\n\n"
    
    convo_graph.update_state({"configurable": {"thread_id": thread_id}}, state, as_node="ami")