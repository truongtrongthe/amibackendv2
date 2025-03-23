# ami_core_4_0.py
# Built by: The Fusion Lab with xAI's Grok 3
# Date: March 21, 2025 
# Purpose: Core Ami logic blending structured intents with GPT-4o natural flow

from utilities import extract_knowledge, recall_knowledge, save_knowledge, LLM, index, logger, EMBEDDINGS
import asyncio
import json
import uuid
from langchain_core.messages import HumanMessage
from typing import Dict, List, Tuple

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
            "intent_history": [],  # Now stores intent scores over time
            "current_focus": ""
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
            logger.debug(f"Saving knowledge with pending: {pending}")
            save_knowledge(state, user_id)
            logger.info(f"Knowledge saved for term: '{term}'")
        else:
            state.pop("pending_knowledge", None)
        return pending

    async def detect_intent(self, state: Dict) -> Dict[str, float]:
        context = "\n".join(msg.content for msg in state["messages"][-50:]) if state["messages"] else ""
        latest_msg = state["messages"][-1].content if state["messages"] else ""
        
        prompt = (
            "You are an AI designed to detect user intent in conversations. Based on the following conversation history, determine the primary intent of the latest message.\n"
            f"Conversation history: {context}\n"
            f"Latest message: '{latest_msg}'\n"
            "Possible intents: teaching, request, correction, confirm, clarify, casual.\n"
            "Task: Return a JSON dictionary with confidence scores (0.0 to 1.0) for each intent, reflecting how likely it matches the latest message in the context of the conversation. Scores should indicate confidence and must include all intents listed.\n"
            "Intent definitions:\n"
            "- 'teaching': The user is providing new information, explaining, or instructing.\n"
            "- 'request': The user is asking for information, clarification, or assistance.\n"
            "- 'correction': The user is correcting or adjusting previous information.\n"
            "- 'confirm': The user is confirming or agreeing (e.g., affirmations like 'yes' or 'no').\n"
            "- 'clarify': The user is refining or elaborating on something previously mentioned.\n"
            "- 'casual': The user is engaging in informal chat without a clear goal.\n"
            "Guidelines:\n"
            "- Consider the entire conversation context to understand the flow and relationships between messages.\n"
            "- Focus on the latest message’s role within this context (e.g., does it build on prior topics?).\n"
            "- Assign higher scores to intents that align with explicit user actions or natural conversational cues.\n"
            "- Return ONLY a valid JSON dictionary with all intents (e.g., {'teaching': 0.9, 'request': 0.2, 'correction': 0.0, 'confirm': 0.0, 'clarify': 0.1, 'casual': 0.3}). Do not include explanations, comments, or any text outside the JSON.\n"
            "- If unsure, default to low scores across all intents but ensure the output is still valid JSON."
        )
        
        response = await asyncio.to_thread(LLM, prompt)
        raw_response = response.content.strip() if hasattr(response, 'content') else str(response).strip()
        logger.debug(f"Raw LLM intent response: '{raw_response}'")
        
        if raw_response.startswith("```json") and raw_response.endswith("```"):
            raw_response = raw_response[7:-3].strip()
        
        try:
            intent_scores = json.loads(raw_response)
            required_intents = {"teaching", "request", "correction", "confirm", "clarify", "casual"}
            if not all(intent in intent_scores for intent in required_intents):
                raise ValueError("Missing required intents in LLM response")
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Invalid LLM intent response: {e}. Defaulting to neutral scores.")
            intent_scores = {intent: 0.1 for intent in required_intents}
        
        if state["intent_history"]:
            last_intent = state["intent_history"][-1]
            for intent, score in last_intent.items():
                intent_scores[intent] += score * 0.3
        
        intent_scores = {k: min(max(v, 0.0), 1.0) for k, v in intent_scores.items()}
        logger.debug(f"Final intent scores: {intent_scores}")
        return intent_scores

    async def blend_brain_with_gpt4o(self, state, user_id, intent_scores: Dict[str, float], confirm_callback=None):
        context = "\n".join(msg.content for msg in state["messages"][-50:])
        brain_data = {
            "active_terms": state["active_terms"],
            "pending_knowledge": state.get("pending_knowledge", {}),
            "last_response": state["last_response"],
            "current_focus": state["current_focus"]
        }
        
        # Tiebreaker: Prioritize 'teaching' if scores are equal
        max_score = max(intent_scores.values())
        dominant_intent = "teaching" if intent_scores["teaching"] == max_score else max(intent_scores, key=intent_scores.get)
        logger.debug(f"Dominant intent: {dominant_intent} with scores: {intent_scores}")
        
        if dominant_intent == "teaching":
            knowledge = await asyncio.to_thread(extract_knowledge, state, user_id, "teaching")
            brain_data["extracted_knowledge"] = knowledge
            summary = ""
            if knowledge["term"] and knowledge["confidence"] >= 0.8:
                key_attrs = [attr["value"] for attr in knowledge["attributes"] if attr["key"] in ["Use", "Ingredients", "Calcium Composition"]]
                if len(key_attrs) >= 2:
                    summary = f"Em hiểu '{knowledge['term']}' là {key_attrs[0]} với {key_attrs[1]}."
                elif len(key_attrs) == 1:
                    summary = f"Em hiểu '{knowledge['term']}' là {key_attrs[0]}."
                else:
                    summary = f"Em hiểu '{knowledge['term']}' là một sản phẩm đặc biệt."
                term_id = state["active_terms"].get(knowledge["term"], {}).get("term_id", f"node_{knowledge['term'].lower()}_products_user_{user_id}_{uuid.uuid4()}")
                state["pending_knowledge"] = {
                    "name": knowledge["term"],
                    "term_id": term_id,
                    "category": "General",
                    "attributes": knowledge["attributes"],
                    "relationships": knowledge["relationships"]
                }
                state["needs_confirmation"] = True
                state["current_focus"] = knowledge["term"]
            
            prompt = (
                "Bạn là Ami, một AI thân thiện, tự nhiên, nói tiếng Việt. "
                "Dựa trên đoạn chat sau và dữ liệu bộ nhớ của tôi, trả lời câu cuối cùng:\n"
                f"Chat: {context}\n"
                f"Bộ nhớ: {json.dumps(brain_data, ensure_ascii=False)}\n"
                f"Intent scores: {json.dumps(intent_scores, ensure_ascii=False)}\n"
                "Nhiệm vụ: Nếu có kiến thức mới (xem 'pending_knowledge'), bắt đầu bằng cách tóm tắt ngắn gọn những gì đã hiểu (ví dụ: '{summary}'), rồi hỏi 'Lưu không bro?' để xác nhận—đảm bảo câu này luôn xuất hiện khi có kiến thức mới. "
                "Tiếp tục trò chuyện tự nhiên, thoải mái theo chủ đề nếu người dùng đang xoay quanh nó. "
                "Chọn cách xưng hô phù hợp (mình, tớ, tôi, em, bạn) theo giọng điệu người dùng. Giữ ngắn gọn, tự nhiên."
            )
            response = await asyncio.to_thread(LLM, prompt)
            state["prompt_str"] = response.content.strip() if hasattr(response, 'content') else str(response).strip()
        
        elif dominant_intent == "request":
            latest_msg = state["messages"][-1].content.lower()
            recalled = await asyncio.to_thread(recall_knowledge, latest_msg, state, user_id)
            brain_data["recalled_knowledge"] = recalled["knowledge"]
            logger.debug(f"Recalled knowledge: {recalled}")
            # Use the highest score from matches as confidence
            #confidence = max([match["score"] for match in recalled["knowledge"]], default=0.0) if recalled["knowledge"] else 0.0
            prompt = (
                "Bạn là Ami, một AI thân thiện, tự nhiên, nói tiếng Việt. "
                "Dựa trên đoạn chat sau và dữ liệu bộ nhớ của tôi, trả lời câu cuối cùng:\n"
                f"Chat: {context}\n"
                f"Bộ nhớ: {json.dumps(brain_data, ensure_ascii=False)}\n"
                f"Intent scores: {json.dumps(intent_scores, ensure_ascii=False)}\n"
                "Nhiệm vụ: Trả lời yêu cầu thông tin từ câu cuối cùng dựa trên 'recalled_knowledge'. Nếu không có dữ liệu, nói tự nhiên và hỏi thêm. "
                "Nếu người dùng đang xoay quanh một chủ đề, hãy tiếp tục chủ đề đó một cách tự nhiên, thoải mái. "
                "Chọn cách xưng hô phù hợp (mình, tớ, tôi, em, bạn) theo giọng điệu người dùng. Giữ ngắn gọn, tự nhiên."
            )
            response = await asyncio.to_thread(LLM, prompt)
            state["prompt_str"] = response.content.strip() if hasattr(response, 'content') else str(response).strip()
            if recalled["knowledge"]:
                state["current_focus"] = recalled["knowledge"][0].get("name", state["current_focus"])

        elif dominant_intent == "correction":
            latest_msg = state["messages"][-1].content.lower()
            knowledge = await asyncio.to_thread(extract_knowledge, state, user_id, "teaching")
            brain_data["extracted_knowledge"] = knowledge
            prompt = (
                "Bạn là Ami, một AI thân thiện, tự nhiên, nói tiếng Việt. "
                "Dựa trên đoạn chat sau và dữ liệu bộ nhớ của tôi, trả lời câu cuối cùng:\n"
                f"Chat: {context}\n"
                f"Bộ nhớ: {json.dumps(brain_data, ensure_ascii=False)}\n"
                f"Intent scores: {json.dumps(intent_scores, ensure_ascii=False)}\n"
                "Nhiệm vụ: Sửa thông tin dựa trên câu cuối cùng. Nếu không rõ, hỏi lại tự nhiên. "
                "Nếu người dùng đang xoay quanh một chủ đề, hãy tiếp tục chủ đề đó một cách tự nhiên, thoải mái. "
                "Nếu có kiến thức mới, cập nhật 'pending_knowledge' và hỏi 'Lưu không bro?'. "
                "Chọn cách xưng hô phù hợp (mình, tớ, tôi, em, bạn) theo giọng điệu người dùng. Giữ ngắn gọn, tự nhiên."
            )
            response = await asyncio.to_thread(LLM, prompt)
            state["prompt_str"] = response.content.strip() if hasattr(response, 'content') else str(response).strip()
            if knowledge["term"]:
                term = next((t for t in state["active_terms"].keys() if t.lower() in latest_msg), knowledge["term"])
                state["pending_knowledge"] = {
                    "name": term,
                    "category": "General",
                    "attributes": knowledge["attributes"],
                    "relationships": knowledge["relationships"]
                }
                state["needs_confirmation"] = True
                state["current_focus"] = term

        elif dominant_intent in ["confirm", "clarify"]:
            prompt = (
                "Bạn là Ami, một AI thân thiện, tự nhiên, nói tiếng Việt. "
                "Dựa trên đoạn chat sau và dữ liệu bộ nhớ của tôi, trả lời câu cuối cùng:\n"
                f"Chat: {context}\n"
                f"Bộ nhớ: {json.dumps(brain_data, ensure_ascii=False)}\n"
                f"Intent scores: {json.dumps(intent_scores, ensure_ascii=False)}\n"
                "Nhiệm vụ: Xác nhận hoặc làm rõ kiến thức đang chờ ('pending_knowledge'). "
                "Nếu xác nhận ('yes'), thông báo lưu xong. Nếu không, bỏ qua và hỏi tiếp. "
                "Nếu người dùng đang xoay quanh một chủ đề, hãy tiếp tục chủ đề đó một cách tự nhiên, thoải mái. "
                "Chọn cách xưng hô phù hợp (mình, tớ, tôi, em, bạn) theo giọng điệu người dùng. Giữ ngắn gọn, tự nhiên."
            )
            response = await asyncio.to_thread(LLM, prompt)
            state["prompt_str"] = response.content.strip() if hasattr(response, 'content') else str(response).strip()
            if dominant_intent == "confirm" and state["needs_confirmation"]:
                self.confirm_knowledge(state, user_id, confirm_callback)
                state["needs_confirmation"] = False
            elif dominant_intent == "clarify":
                state.pop("pending_knowledge", None)
                state["needs_confirmation"] = False

        else:  # Casual or low-confidence blend
            prompt = (
                "Bạn là Ami, một AI thân thiện, tự nhiên, nói tiếng Việt. "
                "Dựa trên đoạn chat sau và dữ liệu bộ nhớ của tôi, trả lời câu cuối cùng thoải mái, gần gũi như người thật:\n"
                f"Chat: {context}\n"
                f"Bộ nhớ: {json.dumps(brain_data, ensure_ascii=False)}\n"
                f"Intent scores: {json.dumps(intent_scores, ensure_ascii=False)}\n"
                "Nhiệm vụ: Trò chuyện tự nhiên, tận dụng bộ nhớ nếu liên quan. "
                "Nếu người dùng đang xoay quanh một chủ đề, hãy tiếp tục chủ đề đó một cách tự nhiên, thoải mái. "
                "Chọn cách xưng hô phù hợp (mình, tớ, tôi, em, bạn) theo giọng điệu người dùng. Giữ ngắn gọn, tự nhiên."
            )
            response = await asyncio.to_thread(LLM, prompt)
            state["prompt_str"] = response.content.strip() if hasattr(response, 'content') else str(response).strip()

        logger.debug(f"Returning state from blend_brain_with_gpt4o: {state}")
        return state

    async def do(self, state=None, is_first=False, confirm_callback=None, user_id=None):
        state = state or self.state
        user_id = user_id or state.get("user_id", "tfl")

        if is_first:
            state["prompt_str"] = "Chào bạn! Mình là Ami, sẵn sàng trò chuyện và học hỏi đây!"
            return state

        intent_scores = await self.detect_intent(state)
        state["intent_history"].append(intent_scores)
        if len(state["intent_history"]) > 5:
            state["intent_history"].pop(0)
        logger.info(f"Intent scores for message '{state['messages'][-1].content}': {intent_scores}")

        state = await self.blend_brain_with_gpt4o(state, user_id, intent_scores, confirm_callback)
        logger.debug(f"State after blend_brain_with_gpt4o: {state}")
        
        if state.get("needs_confirmation", False):
            logger.debug(f"Attempting confirmation for pending_knowledge: {state.get('pending_knowledge', {})}")
            confirmed = self.confirm_knowledge(state, user_id, confirm_callback)
            logger.debug(f"Confirmation result: {confirmed}")
            if confirmed and "lưu không bro?" in state["prompt_str"].lower():
                state["prompt_str"] = state["prompt_str"].replace("Lưu không bro?", "Đã lưu nhé!")
            elif confirmed:
                state["prompt_str"] += " Đã lưu nhé!"
            state["needs_confirmation"] = False

        #latest_msg = state["messages"][-1].content.lower()
        #if "nói thêm" in latest_msg or "thế nào" in latest_msg or state["current_focus"]:
        #    if "hỏi thêm" not in state["prompt_str"].lower():
        #        state["prompt_str"] += f" {state['current_focus']} thì sao nữa nhỉ?" if state["current_focus"] else " Còn gì thú vị nữa không nhỉ?"

        self.state = state
        return state