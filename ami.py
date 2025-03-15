# ami.py
# Purpose: State, graph, and test harness for intent stress testing

import json
import time
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from ami_core import AmiCore
from utilities import LLM

# State
class State(TypedDict):
    messages: Annotated[list, add_messages]
    prompt_str: str
    brain: list
    sales_stage: str

# Graph
ami_core = AmiCore()
graph_builder = StateGraph(State)
graph_builder.add_node("ami", lambda state: ami_core.do(state, not state.get("messages", [])))
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
        "brain": ami_core.brain,
        "sales_stage": ami_core.sales_stages[0]
    }
    state = {**default_state, **(checkpoint.get("channel_values", {}) if checkpoint else {})}
    
    # Add user input if provided
    if user_input:
        state["messages"] = add_messages(state["messages"], [HumanMessage(content=user_input)])
    
    # Debug state before processing
    print(f"Debug: Starting convo_stream - Input: '{user_input}', Stage: {state['sales_stage']}")
    
    # Pass to AmiCore.do for processing
    state = convo_graph.invoke(state, {"configurable": {"thread_id": thread_id}})
    
    # Debug state after processing
    print(f"Debug: State after invoke - Prompt: '{state['prompt_str']}', Stage: {state['sales_stage']}")

    response_lines = state["prompt_str"].split('\n')
    for line in response_lines:
        if line.strip():
            yield f"data: {json.dumps({'message': line.strip()})}\n\n"
    
    # Persist updated state
    convo_graph.update_state({"configurable": {"thread_id": thread_id}}, state, as_node="ami")

# Test
if __name__ == "__main__":
    thread_id = "stress_test_intent_1"
    print("\nAmi starts:")
    for chunk in convo_stream(thread_id=thread_id):
        print(chunk)
    test_inputs = [
        "Hello",
        "Hey, just chilling today",  # Casual
        "HITO is a calcium supplement",  # Teaching
    ]
    for input in test_inputs:
        print(f"\nYou: {input}")
        for chunk in convo_stream(input, thread_id=thread_id):
            print(chunk)
    current_state = convo_graph.get_state({"configurable": {"thread_id": thread_id}})
    print(f"\nFinal state: {json.dumps(current_state, default=str)}")