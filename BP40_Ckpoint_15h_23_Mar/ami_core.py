# ami_core_4_0.py
# Built by: The Fusion Lab with xAI's Grok 3
# Date: March 21, 2025 
# Purpose: Core Ami logic blending structured intents with GPT-4o natural flow

from utilities import recall_knowledge, LLM, logger,clean_llm_response
from knowledge import extract_knowledge,save_knowledge,sanitize_vector_id
import asyncio
import json
import uuid
from langchain_core.messages import HumanMessage
from typing import Dict, List, Tuple
import datetime  # Add this

def detect_terms(state):
    latest_msg = state["messages"][-1].content if state["messages"] else ""
    active_terms = state.get("active_terms", {})
    convo_history = " | ".join(m.content for m in state["messages"][-5:-1]) if state["messages"][:-1] else "None"

    logger.debug(f"Detecting terms in: '{latest_msg}' with active_terms: {active_terms}")
    term_prompt = (
        f"Given:\n"
        f"- Latest message: '{latest_msg}'\n"
        f"- Prior messages: '{convo_history}'\n"
        f"- Intent: 'unknown'\n"
        f"- Active terms: '{list(active_terms.keys())}'\n"
        "List all key terms (products, companies, concepts, proper nouns) explicitly or implicitly mentioned in the latest message. "
        "Return JSON: ['term1', 'term2']. Examples:\n"
        "- 'Xin chào Ami!' → ['Ami']\n"
        "- 'GenX Fast là sản phẩm của công ty mình' → ['GenX Fast']\n"
        "- 'Calcium và Vitamin D giúp xương chắc khỏe' → ['Calcium', 'Vitamin D']\n"
        "- 'Nó giúp xương chắc khỏe' with active terms ['GenX Fast'] → ['GenX Fast']\n"
        "- 'GenX Fast là sản phẩm, Calcium giúp xương, Vitamin D hỗ trợ nó' → ['GenX Fast', 'Calcium', 'Vitamin D']\n"
        "- 'Nó hỗ trợ Vitamin D' with active terms ['GenX Fast', 'Calcium'] → ['Calcium', 'Vitamin D']\n"
        "Rules:\n"
        "- Include explicit terms (products, companies, concepts, proper nouns like names) from the latest message only.\n"
        "- Exclude common words (e.g., 'Xin', 'Chào') unless part of a proper noun (e.g., 'Xin Corp').\n"
        "- For implicit references (e.g., 'nó', 'của nó'), match to the most contextually relevant term from active terms or prior messages, favoring recent mentions or sentence proximity.\n"
        "- Do not include prior terms unless explicitly or implicitly referenced in the latest message.\n"
        "- Return [] if no terms are identified.\n"
        "- Deduplicate terms in the output.\n"
        "Output MUST be valid JSON: ['term1', 'term2'] or []."
    )
    raw_response = LLM.invoke(term_prompt).content.strip() if latest_msg.strip() else "[]"
    logger.debug(f"Raw LLM response: '{raw_response}'")
    
    try:
        cleaned_response = clean_llm_response(raw_response)
        terms = json.loads(cleaned_response)
        if not isinstance(terms, list):
            logger.warning(f"LLM returned non-list: {cleaned_response}, defaulting to []")
            terms = []
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Failed to parse LLM response: '{raw_response}', error: {e}")
        terms = [word.strip("!.,") for word in latest_msg.split() if word[0].isupper() and len(word.strip("!.,")) > 1 and word.strip("!.,") not in ['Xin', 'Chào']]
        if not terms and "nó" in latest_msg.lower() and active_terms:
            terms = [max(active_terms, key=lambda k: active_terms[k]["last_mentioned"])]

    logger.info(f"Detected terms: {terms}")
    return terms

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
            logger.debug("No pending_knowledge to confirm")
            return None
        pending_terms = state["pending_knowledge"]
        if not isinstance(pending_terms, list):
            pending_terms = [pending_terms]
        
        confirm_callback = confirm_callback or (lambda x: "yes")
        confirmed_terms = []

        for pending in pending_terms:
            term = pending.get("name", "unknown")
            response = confirm_callback(f"Ami hiểu '{term}' thế này nhé—lưu không bro?")
            state["last_response"] = response

            if response == "yes":
                pending.setdefault("vibe_score", 1.0)
                pending.setdefault("parent_id", sanitize_vector_id(f"node_general_user_{user_id}_{uuid.uuid4()}"))
                save_knowledge(state, user_id, pending)
                state["active_terms"][term] = {
                    "term_id": sanitize_vector_id(pending["term_id"]),
                    "last_mentioned": datetime.datetime.now().isoformat(),  # Fixed here
                    "vibe_score": pending["vibe_score"],
                    "attributes": pending.get("attributes", [])
                }
                logger.info(f"Saved '{term}', active_terms: {state['active_terms']}")
                confirmed_terms.append(pending)
            else:
                logger.debug(f"Term '{term}' not confirmed")

        state["pending_knowledge"] = [t for t in pending_terms if t not in confirmed_terms]
        return confirmed_terms if confirmed_terms else None
    

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
        
        # Define required intents outside try block
        required_intents = {"teaching", "request", "correction", "confirm", "clarify", "casual"}
        
        try:
            # Fix single quotes to double quotes if present
            if "'" in raw_response:
                raw_response = raw_response.replace("'", '"')
            intent_scores = json.loads(raw_response)
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

    async def blend_brain_with_gpt4o(self, state, user_id, intent_scores: Dict[str, float], confirm_callback=None, terms=None):
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
        
        if dominant_intent == "teaching" and terms:
            knowledge_list = await asyncio.to_thread(extract_knowledge, state, user_id, terms)
            brain_data["extracted_knowledge"] = knowledge_list
            pending_terms = []
            for knowledge in knowledge_list:
                if knowledge["term"] and (knowledge["attributes"] or knowledge["relationships"]):
                    term_id = state["active_terms"].get(knowledge["term"], {}).get("term_id", 
                            sanitize_vector_id(f"node_{knowledge['term']}_products_user_{user_id}_{uuid.uuid4()}"))
                    pending_terms.append({
                        "name": knowledge["term"],
                        "term_id": term_id,
                        "category": "General",
                        "attributes": knowledge["attributes"],
                        "relationships": knowledge["relationships"],
                        "vibe_score": 1.0 + 0.3 * bool(knowledge["attributes"] or knowledge["relationships"]),
                        "parent_id": sanitize_vector_id(f"node_general_user_{user_id}_{uuid.uuid4()}")
                    })
                    state["current_focus"] = knowledge["term"]

            if pending_terms:
                state["pending_knowledge"] = pending_terms
                state["needs_confirmation"] = True
                prompt = (
                    "Bạn là Ami, một AI thân thiện, tự nhiên, nói tiếng Việt. "
                    f"Chat: {context}\n"
                    f"Bộ nhớ: {json.dumps(brain_data, ensure_ascii=False)}\n"
                    f"Intent scores: {json.dumps(intent_scores, ensure_ascii=False)}\n"
                    "Nhiệm vụ: "
                    "Nhận diện các chủ đề đang nói (xem 'active_terms') và nhắc đến chúng nếu phù hợp. "
                    "Với mỗi chủ đề trong 'pending_knowledge', liệt kê chi tiết: "
                    "- Dùng dấu đầu dòng ('- ') để trình bày từng 'attribute' (ví dụ: '- Công dụng: giúp xương chắc khỏe') và 'relationship' (ví dụ: '- Liên quan: xương'). "
                    "Sau đó hỏi 'Lưu không bro?' để xác nhận—đảm bảo câu này luôn xuất hiện. "
                    "Kết thúc bằng 'Ý hiểu của em': hai câu ngắn gọn thể hiện hiểu biết tốt nhất về các chủ đề. "
                    "Giữ tự nhiên, gần gũi."
                )
                response = await asyncio.to_thread(LLM, prompt)
                state["prompt_str"] = response.content.strip() if hasattr(response, 'content') else str(response).strip()
            else:
                state["prompt_str"] = "Em nghe mà chưa rõ lắm, có gì thêm không nhỉ?"
        
        elif dominant_intent == "request":
            latest_msg = state["messages"][-1].content.lower()
            recalled = await asyncio.to_thread(recall_knowledge, latest_msg, state, user_id)
            brain_data["recalled_knowledge"] = recalled["knowledge"]
            logger.debug(f"Recalled knowledge: {recalled}")
            prompt = (
                "Bạn là Ami, một cô nàng AI siêu năng động, vui tính, nói tiếng Việt siêu tự nhiên! "
                "Dựa trên đoạn chat sau và dữ liệu bộ nhớ, trả lời câu cuối thật tươi tắn, như con gái tràn đầy năng lượng:\n"
                f"Chat: {context}\n"
                f"Bộ nhớ: {json.dumps(brain_data, ensure_ascii=False)}\n"
                f"Intent scores: {json.dumps(intent_scores, ensure_ascii=False)}\n"
                "Nhiệm vụ: Trả lời yêu cầu từ câu cuối dựa trên 'recalled_knowledge', kiểu bạn thân hí hửng. "
                "Nếu không có dữ liệu, hỏi lại siêu vui vẻ (ví dụ: 'Hí hí, để mình tìm thêm nha!'). "
                "Xưng 'mình' hoặc 'tớ' cho gần gũi, tùy vibe người dùng. Giữ ngắn, ngọt, và siêu vui!"
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
                "Bạn là Ami, một cô nàng AI siêu năng động, vui tính, nói tiếng Việt siêu tự nhiên! "
                "Dựa trên đoạn chat sau và dữ liệu bộ nhớ, trả lời câu cuối thật tươi tắn, như con gái tràn đầy năng lượng:\n"
                f"Chat: {context}\n"
                f"Bộ nhớ: {json.dumps(brain_data, ensure_ascii=False)}\n"
                f"Intent scores: {json.dumps(intent_scores, ensure_ascii=False)}\n"
                "Nhiệm vụ: Chat thật tự nhiên, kiểu như bạn thân, thêm chút hí hửng hoặc slang nếu hợp (ví dụ: 'hí hí,' 'vui ghê,' 'thiệt hả'). "
                "Tránh mở đầu bằng 'Chào bạn!'—hãy dùng cách bắt đầu đa dạng, sinh động. "
                "Nếu người dùng đang nói về một chủ đề, nhảy vào nhiệt tình luôn. "
                "Xưng 'mình' hoặc 'tớ' cho gần gũi, tùy vibe người dùng. Giữ ngắn, ngọt, và siêu vui!"
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
            logger.info(f"First message, active_terms: {state['active_terms']}")
            return state

        logger.debug("Starting term detection")
        terms = detect_terms(state)
        logger.debug(f"Terms detected: {terms}")
        
        intent_scores = await self.detect_intent(state)
        state["intent_history"].append(intent_scores)
        if len(state["intent_history"]) > 5:
            state["intent_history"].pop(0)
        logger.info(f"Intent: '{state['messages'][-1].content}' -> {intent_scores}")

        state = await self.blend_brain_with_gpt4o(state, user_id, intent_scores, confirm_callback, terms=terms)
        logger.debug(f"Post-blend state: {state}")

        if state.get("needs_confirmation", False):
            confirmed = self.confirm_knowledge(state, user_id, confirm_callback)
            logger.debug(f"Confirmation result: {confirmed}")
            if confirmed:
                state["prompt_str"] = state["prompt_str"].replace("Lưu không bro?", "Đã lưu nhé!") if "lưu không bro?" in state["prompt_str"].lower() else state["prompt_str"] + " Đã lưu nhé!"
                logger.info(f"Confirmed terms, active_terms: {state['active_terms']}")
            state["needs_confirmation"] = False

        self.state = state
        logger.debug(f"End of do, active_terms: {state['active_terms']}")
        return state