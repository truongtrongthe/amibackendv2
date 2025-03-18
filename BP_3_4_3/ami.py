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
from utilities import logger,EMBEDDINGS
import textwrap
from pinecone_datastores import index
from utilities import detect_intent
# State - Aligned with AmiCore and utilities.py
from datetime import datetime
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
    force_copilot = config.get("configurable", {}).get("force_copilot", False) if config else False
    user_id = config.get("configurable", {}).get("user_id", "unknown")  # Grab from config
    logger.info(f"ami_node received config: {config}, extracted force_copilot: {force_copilot}, user_id: {user_id}")
    confirm_callback = lambda x: "yes"
    updated_state = ami_core.do(state, not state.get("messages", []), confirm_callback=confirm_callback, force_copilot=force_copilot, user_id=user_id)
    logger.info(f"ami_node returning state with prompt_str for {user_id}: '{updated_state['prompt_str']}'")
    return updated_state
    
graph_builder.add_node("ami", ami_node)
graph_builder.add_edge(START, "ami")
graph_builder.add_edge("ami", END)
checkpointer = MemorySaver()
convo_graph = graph_builder.compile(checkpointer=checkpointer)

def convo_stream(user_input=None,user_id=None, thread_id=f"test_thread_{int(time.time())}"):
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
        "last_response": "",
        "user_id":user_id
    }
    state = {**default_state, **(checkpoint.get("channel_values", {}) if checkpoint else {})}
    
    # Add user input if provided
    if user_input:
        state["messages"] = add_messages(state["messages"], [HumanMessage(content=user_input)])
    
    # Debug state before processing
    print(f"Debug: Starting convo_stream - Input: '{user_input}', Stage: {state['sales_stage']}, Convo ID: {state['convo_id']}")
    
    # Pass to AmiCore.do for processing via graph
    state = convo_graph.invoke(state, {"configurable": {"thread_id": thread_id,"user_id": user_id}})
    
    # Debug state after processing
    print(f"Debug: State after invoke - Prompt: '{state['prompt_str']}', Stage: {state['sales_stage']}, Last Response: {state.get('last_response', '')}")

    # Stream response
    response_lines = state["prompt_str"].split('\n')
    for line in response_lines:
        if line.strip():
            yield f"data: {json.dumps({'message': line.strip()})}\n\n"
    
    # Persist updated state
    convo_graph.update_state({"configurable": {"thread_id": thread_id}}, state, as_node="ami")



def pilot_stream(user_input=None, user_id=None, thread_id=None):
    if not user_id:
        raise ValueError("user_id is required for personalized CoPilot chats")
    thread_id = thread_id or f"{user_id}_copilot_{int(time.time())}"
    
    checkpoint = checkpointer.get({"configurable": {"thread_id": thread_id}})
    default_state = {
        "messages": [],
        "prompt_str": "",
        "convo_id": thread_id,
        "user_id": user_id,
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
    
    print(f"Debug: Starting pilot_stream - User: '{user_id}', Input: '{user_input}', Stage: {state['sales_stage']}, CoPilot Task: {state['copilot_task']}")
    
    config = {
        "configurable": {
            "thread_id": thread_id,
            "user_id": user_id
        },
        "force_copilot": True
    }
    logger.info(f"Invoking graph with config for {user_id}: {config}")
    state = convo_graph.invoke(state, config=config)
    
    print(f"Debug: State after invoke - User: '{user_id}', Prompt: '{state['prompt_str']}', Stage: {state['sales_stage']}, Last Response: {state.get('last_response', '')}")
    
    response = state["prompt_str"].strip()
    if not response:
        response = f"{user_id.split('_')[0]}, Ami đây—cho bro cái task đi!"
        logger.warning(f"prompt_str empty after invoke for {user_id}, using fallback")
    
    for chunk in textwrap.wrap(response, width=80):
        print(f"Debug: Streaming chunk for {user_id}: '{chunk}'")
        yield f"data: {json.dumps({'message': chunk, 'user_id': user_id})}\n\n"
        time.sleep(0.2)
    
    intent = state.get("intent", "unknown")
    if intent == "unknown":
        intent_result = detect_intent(state)
        intent = intent_result[0] if isinstance(intent_result, tuple) else intent_result
    
    # Use latest_msg from state, not user_input
    latest_msg = state["messages"][-1].content if state["messages"] else ""
    input_to_save = latest_msg if latest_msg else "[no input]"  # Handle empty input
    
    chat_content = f"Input: {input_to_save}\nResponse: {response}"
    embedding = EMBEDDINGS.embed_query(chat_content)
    vector_id = f"node_{thread_id}_{int(time.time())}"
    metadata = {
        "user_id": user_id,
        "thread_id": thread_id,
        "input": input_to_save,  # Use processed input
        "response": response,
        "intent": intent,
        "timestamp": datetime.now().isoformat(),
        "primary_topic": "CoPilot Chat"
    }

    logger.info(f"Attempting upsert to {user_id}_pilot_nodes with node: {vector_id}")
    index.upsert([(vector_id, embedding, metadata)], namespace=f"{user_id}_pilot_nodes")  # Scope to user_id
    logger.info(f"Successfully stored convo node in {user_id}_pilot_nodes: {vector_id}")
    
    convo_graph.update_state({"configurable": {"thread_id": thread_id}}, state, as_node="ami")