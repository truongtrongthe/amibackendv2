# ami.py
# Purpose: Simple state and graph harness for intent testing with AmiCore
# Date: March 22, 2025

import json
import time
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from ami_core import Ami
from utilities import logger
import asyncio

# State - Aligned with AmiCore
class State(TypedDict):
    messages: Annotated[list, add_messages]
    prompt_str: str
    convo_id: str
    active_terms: dict
    pending_knowledge: dict
    last_response: str
    user_id: str
    needs_confirmation: bool  # Added to match ami_core_4_0.py

# Graph
ami_core = Ami()
graph_builder = StateGraph(State)

# Node: Async AmiCore.do with fixed confirmation
async def ami_node(state: State, config=None):
    start_time = time.time()
    user_id = config.get("configurable", {}).get("user_id", "unknown") if config else "unknown"
    logger.info(f"ami_node - User ID: {user_id}")
    confirm_callback = lambda x: "yes"
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

    # Load or init state
    checkpoint = checkpointer.get({"configurable": {"thread_id": thread_id}})
    default_state = {
        "messages": [],
        "prompt_str": "",
        "convo_id": thread_id,
        "active_terms": {},
        "pending_knowledge": {},
        "last_response": "",
        "user_id": user_id,
        "needs_confirmation": False  # Added to match ami_core_4_0.py
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

# Test the 6-turn flow
if __name__ == "__main__":
    turns = [
        "Human hello Ami!"
        #"HITO là sản phẩm bổ sung canxi hỗ trợ phát triển chiều cao (từ 2 tuổi trở lên, đăc biệt dành cho người trưởng thành),Đối tượng KH: Việt kiều 20-30 tuổi (cốm viên) Và mẹ có con từ 12-18 tuổi ở VN (sữa, thạch). Sản phẩm cao cấp, công thức toàn diện. Được đội ngũ chuyên viên đồng hành, cung cấp thông tin chuyên khoa, cá nhân hóa. Bộ tứ canxi hữu cơ kết hợp giúp hệ xương phát tri. ển toàn diện: Canxi cá tuyết, canxi tảo đỏ, canxi Gluconate, bột nhung hươu, ở trên bảng thành phần sp A+. Sản phẩm được CLB hàng đầu VN tín nhiệm và đưa vào chế độ dinh dưỡng cho các lứa cầu thủ chuyên nghiệp. Sản phẩm canxi duy nhất được CLB Bóng đá Hoàng Anh Gia Lai tin dùng. Website: https://hitovietnam.com/. Canxi cá tuyết: cá tuyết sống ở mực nước sâu hàng nghìn mét dưới mực nước biển nên có hệ xương vững chắc, mật độ xương cao. Theo chuyên gia Hito thì xương cá tuyết có cầu tạo gần giống hệ xương người, dồi dào canxi hữu cơ (gấp 9-10 lần canxi so với các nguồn khác), tương thích sinh học cao, tăng hấp thụ tối đa canxi vào xương",
        #"It's made in Japan"
        #"Yes",
        #"Tell me about HITO",
        #"No, HITO’s from Vietnam",
        #"What’s HITO’s price?"
        #"Giá của HITO là combo1 $500, combo2 $800 và combo3 $1200",
        #"Giá HITO là bao nhiêu?"
    ]
    user_id = "TFL"
    thread_id = "teaching"
    print("\n=== 6-Turn Flow Test ===")
    for i, msg in enumerate(turns, 1):
        print(f"\nTurn {i}: {msg}")
        for chunk in convo_stream(msg, user_id, thread_id):
            data = json.loads(chunk.split("data: ")[1])
            print(f"Response: {data['message']}")