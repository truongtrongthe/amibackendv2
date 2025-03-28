# ami.py
# Purpose: Simplified state and graph harness for Training mode
# Date: March 28, 2025

import json
import time
from typing import Annotated, List, Dict
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from training import Training
from utilities import logger
import asyncio

# Define the state structure
class State(TypedDict):
    messages: Annotated[list, add_messages]
    prompt_str: str
    convo_id: str
    user_id: str
    intent_history: List[str]  # Simplified to a list of strings
    preset_memory: str

# Initialize Training instance
training_ami = Training(user_id="thefusionlab")  # Default user_id set here

# Set up the graph
graph_builder = StateGraph(State)

async def training_node(state: State, config=None):
    """Process the state through the Training class."""
    start_time = time.time()
    user_id = config.get("configurable", {}).get("user_id", "thefusionlab") if config else "thefusionlab"
    logger.info(f"training_node - User ID: {user_id}")
    
    # Ensure Training is initialized (called only once if needed)
    if not training_ami.instincts:  # Check if initialized
        await training_ami.initialize()
    
    # Call the training method
    updated_state = await training_ami.training(state=state, user_id=user_id)
    logger.debug(f"training_node took {time.time() - start_time:.2f}s")
    return updated_state

# Add the training node to the graph
graph_builder.add_node("training", training_node)

# Simplified routing: always go to training
graph_builder.add_edge(START, "training")
graph_builder.add_edge("training", END)

# Set up memory persistence
checkpointer = MemorySaver()
convo_graph = graph_builder.compile(checkpointer=checkpointer)

async def convo_stream(user_input: str = None, user_id: str = None, thread_id: str = None):
    """Stream the conversation response."""
    start_time = time.time()
    thread_id = thread_id or f"thread_{int(time.time())}"
    user_id = user_id or "thefusionlab"

    logger.info(f"Running in training mode for user {user_id}, thread {thread_id}")

    # Load or initialize state
    checkpoint = checkpointer.get({"configurable": {"thread_id": thread_id}})
    default_state = {
        "messages": [],
        "prompt_str": "",
        "convo_id": thread_id,
        "user_id": user_id,
        "intent_history": [],
        "preset_memory": "Be friendly"  # Default preset
    }
    state = {**default_state, **(checkpoint.get("channel_values", {}) if checkpoint else {})}

    # Add user input to messages if provided
    if user_input:
        state["messages"] = add_messages(state["messages"], [HumanMessage(content=user_input)])

    logger.debug(f"convo_stream init - Input: '{user_input}', Convo ID: {thread_id}")

    # Process the state through the graph
    config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
    updated_state = await convo_graph.ainvoke(state, config)
    await convo_graph.aupdate_state({"configurable": {"thread_id": thread_id}}, updated_state, as_node="training")

    logger.debug(f"convo_stream total took {time.time() - start_time:.2f}s")

    # Stream the response
    response_lines = updated_state["prompt_str"].split('\n')
    for line in response_lines:
        if line.strip():
            yield f"data: {json.dumps({'message': line.strip()})}\n\n"
            await asyncio.sleep(0.05)  # Non-blocking sleep
    yield "data: [DONE]\n\n"

async def test_convo():
    """Test the conversation stream with a sequence of messages."""
    test_messages = [
        "Hello em!",
        "Em hãy nhớ em tên là Linh Chi!",
        "You're curious and truthful",
        "Người tò mò là người luôn chủ động tìm kiếm thông tin để tìm ra bản chất vấn đề!",
        "Thành thật là chân thành với mọi người xung quanh!",
        "Bán hàng là nghệ thuật chăn dắt khách hàng!",
        "Tell me about sales",
        "Đầu tiên em cần hiểu về các đại đô thị quy mô như Vinhomes Ocean Park, tính chất đặc điểm về loại hình, phân khúc, giá cả và những thứ liên quan nhé. Anh sẽ hướng dẫn em dần"
    ]
    
    thread_id = "test_thread"
    for i, msg in enumerate(test_messages):
        print(f"\nUser ({i+1}/{len(test_messages)}): {msg}")
        async for line in convo_stream(user_input=msg, thread_id=thread_id):
            print(line.strip())

if __name__ == "__main__":
    asyncio.run(test_convo())