import json
import time
from typing import Annotated, List, Dict
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage
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
    bank_name: str
    unresolved_requests: List[Dict]  # Corrected from List[str]
    brain_uuid:str

mc = MC(user_id="thefusionlab")

graph_builder = StateGraph(State)

async def mc_node(state: State, config=None):
    start_time = time.time()
    user_id = config.get("configurable", {}).get("user_id", "thefusionlab") if config else "thefusionlab"
    bank_name = config.get("configurable", {}).get("bank_name", "") if config else ""
    brain_uuid = config.get("configurable", {}).get("brain_uuid", "") if config else ""
    
    logger.info(f"MC Node - User ID: {user_id}, Bank Name: {bank_name}")
    
    if not mc.instincts:
        await mc.initialize()
    
    # Process the async generator output from trigger
    final_state = state  # Default to input state
    async for output in mc.trigger(state=state, user_id=user_id, bank_name=bank_name,brain_uuid=brain_uuid, config=config):
        if isinstance(output, dict) and "state" in output:
            final_state = output["state"]  # Capture the final state
        else:
            logger.debug(f"Received response chunk: {output}")
    
    logger.debug(f"Mc node took {time.time() - start_time:.2f}s")
    return final_state

graph_builder.add_node("mc", mc_node)
graph_builder.add_edge(START, "mc")
graph_builder.add_edge("mc", END)

checkpointer = MemorySaver()
convo_graph = graph_builder.compile(checkpointer=checkpointer)

def convo_stream(user_input: str = None, user_id: str = None, thread_id: str = None, bank_name: str = "", brain_uuid: str ="", mode: str = "mc"):
    start_time = time.time()
    thread_id = thread_id or f"mc_thread_{int(time.time())}"
    user_id = user_id or "thefusionlab"

    # Load or init state with intent_history
    checkpoint = checkpointer.get({"configurable": {"thread_id": thread_id}})
    default_state = {
        "messages": [],
        "prompt_str": "",
        "convo_id": thread_id,
        "last_response": "",
        "user_id": user_id,
        "intent": "",
        "intent_history": [],
        "preset_memory": "Be friendly",
        "instinct": "",
        "bank_name": bank_name,
        "unresolved_requests": [],
        "brain_uuid":brain_uuid
    }
    state = {**default_state, **(checkpoint.get("channel_values", {}) if checkpoint else {})}

    # Add user input
    if user_input:
        state["messages"] = add_messages(state["messages"], [HumanMessage(content=user_input)])

    logger.debug(f"convo_stream init - Input: '{user_input}', Convo ID: {thread_id}")

    # Run async graph synchronously
    async def process_state():
        config = {"configurable": {"thread_id": thread_id, "user_id": user_id, "bank_name": bank_name,"brain_uuid": brain_uuid}}  # Added bank adn brain
        updated_state = await convo_graph.ainvoke(state, config)
        await convo_graph.aupdate_state({"configurable": {"thread_id": thread_id}}, updated_state, as_node="mc")
        return updated_state

    state = asyncio.run(process_state())

    logger.debug(f"convo_stream total took {time.time() - start_time:.2f}s")

    # Stream response
    response_lines = state["prompt_str"].split('\n')
    for line in response_lines:
        if line.strip():
            yield f"data: {json.dumps({'message': line.strip()})}\n\n"
            time.sleep(0.05)