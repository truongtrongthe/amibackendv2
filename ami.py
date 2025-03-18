# ami.py
# Purpose: State, graph, and test harness for intent stress testing with Ami Blue Print 3.4 Mark 3
# Date: March 15, 2025 (Updated for live on March 16, 2025, Optimized March 18, 2025)

import json
import time
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from ami_core import AmiCore
from utilities import logger, EMBEDDINGS, detect_intent
import textwrap
from pinecone_datastores import index
import asyncio
from datetime import datetime

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
    user_id: str  # Added for consistency

# Graph
ami_core = AmiCore()
graph_builder = StateGraph(State)

# Node: Async AmiCore.do with confirmation callback and timing
async def ami_node(state: State, config=None):
    start_time = time.time()
    force_copilot = config.get("configurable", {}).get("force_copilot", False) if config else False
    user_id = config.get("configurable", {}).get("user_id", "unknown") if config else "unknown"
    logger.info(f"ami_node received config: {config}, extracted force_copilot: {force_copilot}, user_id: {user_id}")
    confirm_callback = lambda x: "yes"
    updated_state = await ami_core.do(state, not state.get("messages", []), confirm_callback=confirm_callback, 
                                      force_copilot=force_copilot, user_id=user_id)
    logger.debug(f"ami_node took {time.time() - start_time:.2f}s for user_id: {user_id}")
    return updated_state

graph_builder.add_node("ami", ami_node)
graph_builder.add_edge(START, "ami")
graph_builder.add_edge("ami", END)
checkpointer = MemorySaver()
convo_graph = graph_builder.compile(checkpointer=checkpointer)

def convo_stream(user_input=None, user_id=None, thread_id=None):
    start_time = time.time()
    thread_id = thread_id or f"test_thread_{int(time.time())}"
    user_id = user_id or "global_thread"  # Default user_id if not provided
    
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
        "user_id": user_id
    }
    state = {**default_state, **(checkpoint.get("channel_values", {}) if checkpoint else {})}
    
    # Add user input if provided
    if user_input:
        state["messages"] = add_messages(state["messages"], [HumanMessage(content=user_input)])
    
    print(f"Debug: Starting convo_stream - Input: '{user_input}', Stage: {state['sales_stage']}, Convo ID: {state['convo_id']}")
    logger.debug(f"convo_stream state init took {time.time() - start_time:.2f}s")
    
    # Run async logic synchronously
    async def process_state():
        config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
        start_invoke = time.time()
        updated_state = await convo_graph.ainvoke(state, config)
        logger.debug(f"convo_graph.ainvoke took {time.time() - start_invoke:.2f}s")
        start_update = time.time()
        await convo_graph.aupdate_state({"configurable": {"thread_id": thread_id}}, updated_state, as_node="ami")
        logger.debug(f"convo_graph.aupdate_state took {time.time() - start_update:.2f}s")
        return updated_state
    
    state = asyncio.run(process_state())
    
    print(f"Debug: State after invoke - Prompt: '{state['prompt_str']}', Stage: {state['sales_stage']}, Last Response: {state.get('last_response', '')}")
    logger.debug(f"convo_stream total processing took {time.time() - start_time:.2f}s")
    
    # Yield response synchronously
    response_lines = state["prompt_str"].split('\n')
    for line in response_lines:
        if line.strip():
            yield f"data: {json.dumps({'message': line.strip()})}\n\n"
            time.sleep(0.05)  # Small delay for streaming effect

def pilot_stream(user_input=None, user_id=None, thread_id=None):
    start_time = time.time()
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
    logger.debug(f"pilot_stream state init took {time.time() - start_time:.2f}s")
    
    async def process_pilot_state():
        config = {
            "configurable": {"thread_id": thread_id, "user_id": user_id},
            "force_copilot": True
        }
        start_invoke = time.time()
        updated_state = await convo_graph.ainvoke(state, config=config)
        logger.debug(f"convo_graph.ainvoke took {time.time() - start_invoke:.2f}s")
        start_update = time.time()
        await convo_graph.aupdate_state({"configurable": {"thread_id": thread_id}}, updated_state, as_node="ami")
        logger.debug(f"convo_graph.aupdate_state took {time.time() - start_update:.2f}s")
        return updated_state
    
    state = asyncio.run(process_pilot_state())
    
    print(f"Debug: State after invoke - User: '{user_id}', Prompt: '{state['prompt_str']}', Stage: {state['sales_stage']}, Last Response: {state.get('last_response', '')}")
    logger.debug(f"pilot_stream total processing took {time.time() - start_time:.2f}s")
    
    response = state["prompt_str"].strip()
    if not response:
        response = f"{user_id.split('_')[0]}, Ami đây—cho bro cái task đi!"
        logger.warning(f"prompt_str empty after invoke for {user_id}, using fallback")
    
    # Stream using natural \n splits
    response_lines = response.split('\n')
    for line in response_lines:
        if line.strip():
            print(f"Debug: Streaming chunk for {user_id}: '{line.strip()}'")
            yield f"data: {json.dumps({'message': line.strip(), 'user_id': user_id})}\n\n"
            time.sleep(0.2)
    # Signal stream end
    yield "data: [DONE]\n\n"
    
    intent = state.get("intent", "unknown")
    if intent == "unknown":
        intent = detect_intent(state)
    
    latest_msg = state["messages"][-1].content if state["messages"] else ""
    input_to_save = latest_msg if latest_msg else "[no input]"
    
    chat_content = f"Input: {input_to_save}\nResponse: {response}"
    embedding = EMBEDDINGS.embed_query(chat_content)
    vector_id = f"node_{thread_id}_{int(time.time())}"
    metadata = {
        "user_id": user_id,
        "thread_id": thread_id,
        "input": input_to_save,
        "response": response,
        "intent": intent,
        "timestamp": datetime.now().isoformat(),
        "primary_topic": "CoPilot Chat"
    }

    try:
        start_upsert = time.time()
        index.upsert([(vector_id, embedding, metadata)], namespace=f"{user_id}_pilot_nodes")
        logger.debug(f"Pinecone upsert took {time.time() - start_upsert:.2f}s")
    except Exception as e:
        logger.error(f"Upsert failed for {user_id}_pilot_nodes: {e}")