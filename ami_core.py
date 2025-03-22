# ami_core_4_0.py
# Built by: The Fusion Lab with xAI's Grok 3
# Date: March 21, 2025 
# Purpose: Core Ami logic blending structured intents with GPT-4o natural flow

from utilities import detect_intent, extract_knowledge, recall_knowledge, save_knowledge, LLM, index, logger, EMBEDDINGS
import asyncio
import json
import uuid
from langchain_core.messages import HumanMessage

class Ami:
    def __init__(self):
        self.user_id = "tfl"
        self.state = {
            "messages": [],
            "prompt_str": "",
            "convo_id": None,
            "active_terms": {},
            "pending_knowledge": {},
            "last_response": "",
            "user_id": self.user_id,
            "needs_confirmation": False,
            "intent_history": []
        }

    def confirm_knowledge(self, state, user_id, confirm_callback=None):
        if not state.get("pending_knowledge"):
            return None
        pending = state["pending_knowledge"]
        term = pending.get("name", "unknown")
        
        confirm_callback = confirm_callback or (lambda x: "yes")
        response = confirm_callback(f"Ami hiểu '{term}' thế này nhé—lưu không bro?")
        state["last_response"] = response

        if response == "yes":
            pending.setdefault("vibe_score", 1.0)
            pending.setdefault("parent_id", f"node_{pending['category'].lower()}_user_{user_id}_{uuid.uuid4()}")
            save_knowledge(state, user_id)
            logger.info(f"Knowledge saved for term: '{term}'")
        else:
            state.pop("pending_knowledge", None)
        return pending

    async def blend_brain_with_gpt4o(self, state, user_id, intent, confirm_callback=None):
        context = "\n".join(msg.content for msg in state["messages"][-5:])
        brain_data = {
            "active_terms": state["active_terms"],
            "pending_knowledge": state.get("pending_knowledge", {}),
            "last_response": state["last_response"]
        }
        
        if intent == "teaching":
            knowledge = await asyncio.to_thread(extract_knowledge, state, user_id, "teaching")
            brain_data["extracted_knowledge"] = knowledge
            prompt = (
                "Bạn là Ami, một AI thân thiện, tự nhiên, nói tiếng Việt. "
                "Dựa trên đoạn chat sau và dữ liệu bộ nhớ của tôi, trả lời câu cuối cùng:\n"
                f"Chat: {context}\n"
                f"Bộ nhớ: {json.dumps(brain_data, ensure_ascii=False)}\n"
                "Nhiệm vụ: Dạy hoặc cập nhật kiến thức mới từ câu cuối cùng. Nếu chưa rõ hoặc cần xác nhận, hỏi lại tự nhiên. "
                "Nếu kiến thức mới, lưu vào 'pending_knowledge' và hỏi 'Lưu không bro?'. "
                "Chọn cách xưng hô phù hợp (mình, tớ, tôi, em, bạn) theo giọng điệu người dùng. Giữ ngắn gọn, tự nhiên."
            )
            if knowledge["term"] and knowledge["confidence"] >= 0.8:
                state["pending_knowledge"] = {
                    "name": knowledge["term"],
                    "category": "General",
                    "attributes": knowledge["attributes"],
                    "relationships": knowledge["relationships"]
                }
                state["needs_confirmation"] = True

        elif intent == "request":
            latest_msg = state["messages"][-1].content.lower()
            recalled = await asyncio.to_thread(recall_knowledge, latest_msg, state, user_id)
            brain_data["recalled_knowledge"] = recalled["knowledge"]
            prompt = (
                "Bạn là Ami, một AI thân thiện, tự nhiên, nói tiếng Việt. "
                "Dựa trên đoạn chat sau và dữ liệu bộ nhớ của tôi, trả lời câu cuối cùng:\n"
                f"Chat: {context}\n"
                f"Bộ nhớ: {json.dumps(brain_data, ensure_ascii=False)}\n"
                "Nhiệm vụ: Trả lời yêu cầu thông tin từ câu cuối cùng. Nếu không có dữ liệu, nói tự nhiên và hỏi thêm. "
                "Chọn cách xưng hô phù hợp (mình, tớ, tôi, em, bạn) theo giọng điệu người dùng. Giữ ngắn gọn, tự nhiên."
            )

        elif intent == "correction":
            latest_msg = state["messages"][-1].content.lower()
            knowledge = await asyncio.to_thread(extract_knowledge, state, user_id, "teaching")
            brain_data["extracted_knowledge"] = knowledge
            prompt = (
                "Bạn là Ami, một AI thân thiện, tự nhiên, nói tiếng Việt. "
                "Dựa trên đoạn chat sau và dữ liệu bộ nhớ của tôi, trả lời câu cuối cùng:\n"
                f"Chat: {context}\n"
                f"Bộ nhớ: {json.dumps(brain_data, ensure_ascii=False)}\n"
                "Nhiệm vụ: Sửa thông tin dựa trên câu cuối cùng. Nếu không rõ, hỏi lại tự nhiên. "
                "Nếu có kiến thức mới, cập nhật 'pending_knowledge' và hỏi 'Lưu không bro?'. "
                "Chọn cách xưng hô phù hợp (mình, tớ, tôi, em, bạn) theo giọng điệu người dùng. Giữ ngắn gọn, tự nhiên."
            )
            if knowledge["term"]:
                term = next((t for t in state["active_terms"].keys() if t.lower() in latest_msg), knowledge["term"])
                state["pending_knowledge"] = {
                    "name": term,
                    "category": "General",
                    "attributes": knowledge["attributes"],
                    "relationships": knowledge["relationships"]
                }
                state["needs_confirmation"] = True

        elif intent in ["confirm", "clarify"]:
            prompt = (
                "Bạn là Ami, một AI thân thiện, tự nhiên, nói tiếng Việt. "
                "Dựa trên đoạn chat sau và dữ liệu bộ nhớ của tôi, trả lời câu cuối cùng:\n"
                f"Chat: {context}\n"
                f"Bộ nhớ: {json.dumps(brain_data, ensure_ascii=False)}\n"
                "Nhiệm vụ: Xác nhận hoặc làm rõ kiến thức đang chờ ('pending_knowledge'). "
                "Nếu xác nhận ('yes'), thông báo lưu xong. Nếu không, bỏ qua và hỏi tiếp. "
                "Chọn cách xưng hô phù hợp (mình, tớ, tôi, em, bạn) theo giọng điệu người dùng. Giữ ngắn gọn, tự nhiên."
            )
            if intent == "confirm" and state["needs_confirmation"]:
                self.confirm_knowledge(state, user_id, confirm_callback)
                state["needs_confirmation"] = False
            elif intent == "clarify":
                state.pop("pending_knowledge", None)
                state["needs_confirmation"] = False

        else:  # Casual or undefined intent
            prompt = (
                "Bạn là Ami, một AI thân thiện, tự nhiên, nói tiếng Việt. "
                "Dựa trên đoạn chat sau và dữ liệu bộ nhớ của tôi, trả lời câu cuối cùng thoải mái, gần gũi như người thật:\n"
                f"Chat: {context}\n"
                f"Bộ nhớ: {json.dumps(brain_data, ensure_ascii=False)}\n"
                "Nhiệm vụ: Trò chuyện tự nhiên, tận dụng bộ nhớ nếu liên quan. "
                "Chọn cách xưng hô phù hợp (mình, tớ, tôi, em, bạn) theo giọng điệu người dùng. Giữ ngắn gọn, tự nhiên."
            )

        response = await asyncio.to_thread(LLM, prompt)
        return response.content.strip() if hasattr(response, 'content') else str(response).strip()

    async def do(self, state=None, is_first=False, confirm_callback=None, user_id=None):
        state = state or self.state
        user_id = user_id or state.get("user_id", "tfl")

        if is_first:
            state["prompt_str"] = "Chào bạn! Mình là Ami, sẵn sàng trò chuyện và học hỏi đây!"
            return state

        intent = await asyncio.to_thread(detect_intent, state)
        state["intent"] = intent
        logger.info(f"Intent: '{intent}' for message: '{state['messages'][-1].content}'")

        state["prompt_str"] = await self.blend_brain_with_gpt4o(state, user_id, intent, confirm_callback)
        self.state = state
        return state

# Test Example
async def test_ami():
    ami = Ami()
    messages = [
        "Xin chào Ami!",
        "Em nói anh xem Hà Nội thế nào",
        "Chà, không phải. Ở Hạ Long cơ",
        "Hạ Long",
        "Liên thiên",
        "Đúng rồi, nói thêm đi"
    ]
    def dummy_callback(prompt):
        return "yes" if "Lưu không bro?" in prompt else "no"
    
    for msg in messages:
        ami.state["messages"].append(HumanMessage(content=msg))
        state = await ami.do(confirm_callback=dummy_callback)
        print(f"User: {msg}")
        print(f"Ami: {state['prompt_str']}\n")

if __name__ == "__main__":
    asyncio.run(test_ami())