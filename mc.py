import asyncio
from typing import List, Dict
from langchain_openai import ChatOpenAI
from utilities import logger
from langchain_core.messages import HumanMessage
import json
from sentence_transformers import SentenceTransformer, util
import numpy as np
from database import query_knowledge
LLM = ChatOpenAI(model="gpt-4o", streaming=False)
StreamLLM = ChatOpenAI(model="gpt-4o", streaming=True)

FEEDBACKTYPE = ["correction", "adjustment", "new", "confirmation", "clarification",
                "rejection", "elaboration", "satisfaction", "confusion"]

STATE_STORE = {}

class ResponseBuilder:
    def __init__(self):
        self.parts = []
        self.error_flag = False

    def add(self, text: str, is_error: bool = False):
        if text:
            self.parts.append(text)
        self.error_flag = self.error_flag or is_error
        return self

    def build(self, separator: str = None) -> str:
        if not self.parts:
            return "Tôi không biết nói gì cả!"
        separator = separator or ""
        return separator.join(part for part in self.parts if part)

class MC:
    embedder = SentenceTransformer('sentence-transformers/LaBSE')

    def __init__(self, user_id: str = "thefusionlab", convo_id: str = None, 
                 similarity_threshold: float = 0.55, max_active_requests: int = 5):
        self.user_id = user_id
        self.convo_id = convo_id or "default_thread"
        self.name = "Grok 3"
        self.instincts = {"friendly": "Be nice"}
        self.similarity_threshold = similarity_threshold
        self.max_active_requests = max_active_requests
        self.state = STATE_STORE.get(self.convo_id, {
            "messages": [],
            "intent_history": [],
            "preset_memory": "Be friendly",
            "unresolved_requests": [],
            "convo_id": self.convo_id,
            "user_id": self.user_id,
            "prompt_str": ""
        })

    async def initialize(self):
        if not self.instincts:
            self.instincts = {"friendly": "Be nice"}

    
    # mc.py (in detect_intent)
    async def detect_intent(self, message: str, context: str = "") -> str:
        prompt = (
                f"Message: '{message}'\nContext: '{context}'\n"
                "Classify intent as 'teaching', 'request', or 'casual'. "
                "Return one word."
            )
        try:
            response = await asyncio.to_thread(LLM.invoke, prompt)
            intent = response.content.strip().lower()
            return intent if intent in ["teaching", "request", "casual"] else "casual"
        except Exception as e:
            logger.info(f"Intent detection failed: {e}")
            return "casual"
    async def detect_feedback_type(self, message: str, context: str) -> str:
        prompt = (
            f"Context: '{context}'\nMessage: '{message}'\n"
            f"Classify feedback as: {', '.join(FEEDBACKTYPE)}. Return one word."
        )
        try:
            response = await asyncio.to_thread(LLM.invoke, prompt)
            feedback = response.content.strip().replace("'", "")
            return feedback if feedback in FEEDBACKTYPE else "new"
        except Exception as e:
            logger.info(f"Feedback detection failed: {e}")
            return "new"

    async def resolve_related_request(self, message: str, feedback_type: str, state: Dict, context: str) -> Dict[str, any]:
        unresolved = [r for r in state.get("unresolved_requests", []) if not r.get("resolved", False)]
        if not unresolved:
            logger.info("No unresolved requests found")
            return None

        if feedback_type in ["clarification", "elaboration", "confusion"]:
            message_embedding = self.embedder.encode(message, convert_to_tensor=True)
            
            # Approach 1: Concatenation (request + response)
            concat_texts = [f"{r['message']} {r.get('response', '')}" for r in unresolved]
            concat_embeddings = self.embedder.encode(concat_texts, convert_to_tensor=True)
            concat_similarities = util.cos_sim(message_embedding, concat_embeddings)[0]
            
            # Approach 2: Weighted Average (0.3 request + 0.7 response)
            request_texts = [r["message"] for r in unresolved]
            response_texts = [r.get("response", "") for r in unresolved]
            request_embeddings = self.embedder.encode(request_texts, convert_to_tensor=True)
            response_embeddings = self.embedder.encode(response_texts, convert_to_tensor=True)
            weighted_similarities = 0.3 * util.cos_sim(message_embedding, request_embeddings)[0] + \
                                0.7 * util.cos_sim(message_embedding, response_embeddings)[0]
            
            # Token boosts (applied to both)
            message_tokens = set(message.lower().split())
            concat_boosts = [len(message_tokens.intersection(set(t.lower().split()))) * 0.1 for t in concat_texts]
            weighted_boosts = [len(message_tokens.intersection(set(r["message"].lower().split() + r.get("response", "").lower().split()))) * 0.1 for r in unresolved]
            
            # Adjusted similarities
            concat_adjusted = concat_similarities + np.array(concat_boosts)
            weighted_adjusted = weighted_similarities + np.array(weighted_boosts)
            
            # Pick an approach (toggle here or test both!)
            # Using concatenation for now—swap to weighted_adjusted to try the other
            adjusted_similarities = concat_adjusted
            threshold = 0.3 if feedback_type in ["elaboration", "clarification"] else self.similarity_threshold
            
            # Log scores for debugging
            logger.info(f"Concat similarities: {list(zip(concat_texts, concat_adjusted.tolist()))}")
            logger.info(f"Weighted similarities: {list(zip(request_texts, weighted_adjusted.tolist()))}")
            
            max_similarity_idx = np.argmax(adjusted_similarities)
            similarity_score = adjusted_similarities[max_similarity_idx].item()
            latest_active = max(
                [(i, r) for i, r in enumerate(unresolved) if r["active"]],
                key=lambda x: len(state["messages"]) - x[1]["turn"],
                default=(None, None)
            )[1]

            # Linking logic
            if latest_active and adjusted_similarities[unresolved.index(latest_active)] >= threshold - 0.1:
                related_request = latest_active
                logger.info(f"Linked to latest active: {related_request['message']} (score: {adjusted_similarities[unresolved.index(latest_active)]:.2f})")
            elif similarity_score >= threshold:
                related_request = unresolved[max_similarity_idx]
                logger.info(f"Linked to: {related_request['message']} (score: {similarity_score:.2f})")
            else:
                logger.info(f"No match: highest score {similarity_score:.2f} below threshold {threshold}")
                return None
            
            # Update request based on feedback type
            if feedback_type == "clarification":
                related_request["status"] = "CLARIFIED"
            elif feedback_type == "elaboration":
                related_request["message"] = f"{related_request['message']} - {message}"
                related_request["status"] = "ELABORATED"
            elif feedback_type == "confusion":
                related_request["status"] = "NEEDS_CLARIFICATION"
            return related_request
        
        # Fallback for other feedback types (e.g., confirmation)
        message_tokens = set(message.lower().split())
        related_request = next(
            (r for r in unresolved if len(message_tokens.intersection(set(r["message"].lower().split()))) >= 1),
            None
        )
        if related_request and feedback_type in ["confirmation", "satisfaction"]:
            related_request["resolved"] = True
            related_request["satisfied"] = True
            related_request["status"] = "RESOLVED"
        return related_request

    async def stream_response(self, prompt: str, builder: ResponseBuilder):
        if len(self.state["messages"]) > 20:
            prompt += "\nKeep it short and sweet."
        buffer = ""
        try:
            async for chunk in StreamLLM.astream(prompt):
                buffer += chunk.content
                # Split on sentence boundaries or size limit
                if "\n" in buffer or buffer.endswith((".", "!", "?")) or len(buffer) > 500:
                    # Take the complete part, leave the rest in buffer
                    parts = buffer.split("\n", 1) if "\n" in buffer else [buffer, ""]
                    complete_part = parts[0].strip()
                    if complete_part:  # Only add if there's something meaningful
                        builder.add(complete_part)
                        yield builder.build(separator="\n")  # Use newline for natural flow
                    buffer = parts[1] if len(parts) > 1 else ""
            # Flush any remaining buffer
            if buffer.strip():
                builder.add(buffer.strip())
                yield builder.build(separator="\n")
        except Exception as e:
            logger.info(f"Streaming failed: {e}")
            builder.add("Có lỗi nhỏ, thử lại nhé!")
            yield builder.build(separator="\n")

    # mc.py (in trigger)
    async def trigger(self, state: Dict = None, user_id: str = None, bank_name: str = None, config: Dict = None):
        state = state or self.state.copy()
        user_id = user_id or self.user_id
        config = config or {}
        bank_name = (
            bank_name if bank_name is not None 
            else state.get("bank_name", 
                config.get("configurable", {}).get("bank_name", 
                    self.bank_name if hasattr(self, 'bank_name') else ""))
        )
        if not bank_name:
            logger.warning(f"bank_name is empty! State: {state}, Config: {config}")
        
        # Sync with mc.state
        self.state["bank_name"] = bank_name
        if "unresolved_requests" not in state:
            state["unresolved_requests"] = self.state.get("unresolved_requests", [])

        logger.info(f"Starting Fun - User: {user_id}, Bank: {bank_name}, State: {json.dumps({k: v if k != 'messages' else [{'content': m.content if isinstance(m, HumanMessage) else m} for m in v] for k, v in state.items()}, ensure_ascii=False)}")
        
        latest_msg = state["messages"][-1] if state["messages"] else HumanMessage(content="")
        latest_msg_content = latest_msg.content.strip() if isinstance(latest_msg, HumanMessage) else latest_msg.strip()
        context = "\n".join(f"User: {msg.content}" if isinstance(msg, HumanMessage) else f"AI: {msg}" for msg in state["messages"][-30:])

        if state["unresolved_requests"]:
            current_embedding = self.embedder.encode(latest_msg_content, convert_to_tensor=True)
            unresolved_embeddings = self.embedder.encode([r["message"] for r in state["unresolved_requests"]], convert_to_tensor=True)
            similarities = util.cos_sim(current_embedding, unresolved_embeddings)[0]
            for i, req in enumerate(state["unresolved_requests"]):
                req["score"] = similarities[i].item()
                req["active"] = req["score"] > 0.3 or (len(state["messages"]) - req["turn"] < 10)

        intent = await self.detect_intent(latest_msg_content, context)
        feedback = await self.detect_feedback_type(latest_msg_content, context)
        if feedback in ["correction", "adjustment", "clarification", "confusion", "elaboration", "rejection"]:
            intent = "request"
        
        state["intent_history"].append(intent)
        state["intent_history"] = state["intent_history"][-5:]
        builder = ResponseBuilder()
        
        logger.info(f"intent in trigger={intent}")
        if intent == "request":
            async for response_chunk in self._handle_request(latest_msg_content, user_id, context, builder, feedback, state, intent, bank_name=bank_name):
                state["prompt_str"] = response_chunk
                logger.debug(f"Yielding from trigger: {state['prompt_str']}")
                yield {"mc": {"prompt_str": state["prompt_str"]}}  # Explicit dict
        else:
            async for response_chunk in self._handle_casual(latest_msg_content, context, builder, state, bank_name=bank_name):
                state["prompt_str"] = response_chunk
                logger.debug(f"Yielding from trigger: {state['prompt_str']}")
                yield {"mc": {"prompt_str": state["prompt_str"]}}  # Explicit dict
        
        state["messages"].append(state["prompt_str"])
        self.state.update(state)
        STATE_STORE[self.convo_id] = self.state
        logger.info(f"Final response: {state['prompt_str']}")
    
    async def _handle_request(self, message: str, user_id: str, context: str, builder: "ResponseBuilder", 
                         feedback_type: str, state: Dict, intent: str, bank_name: str = ""):
        related_request = await self.resolve_related_request(message, feedback_type, state, context)
        logger.info(f"Handling request for user {user_id} with bank_name: {bank_name}")
        
        try:
            # Query knowledge with full context for broader product relevance
            knowledge = await query_knowledge(context, bank_name=bank_name) or []
            kwcontext = "\n\n".join(entry["raw"] for entry in knowledge)
        except Exception as e:
            logger.error(f"Knowledge query failed: {e}")
            builder.add("Oops, có lỗi khi tìm thông tin, thử lại nhé!")
            yield builder.build()
            return

        # Sync LLM to build customer profile
        
        profile_prompt = (
            f"Based on this conversation:\n{context}\n"
            f"Infer the customer's interests, needs, or preferences (e.g., price-sensitive, looking for combos, curious about features). "
            f"Summarize it in 1-2 sentences."
        )
        customer_profile = LLM.invoke(profile_prompt).content  # Sync call, full response
        logger.info(f"Customer profile: {customer_profile}")

        # Main response prompt with profile and product info
        base_prompt = (
                    f"AI: {self.name}\n"
                    f"Context: {context}\n"
                    f"Message: '{message}'\n"
                    f"Customer Profile: {customer_profile}\n"
                    f"Product Info and Rules: {kwcontext}\n"
                    f"Task: Reply in Vietnamese. Start casual and friendly, avoid greeting if already greeted. "
                    f"Reason from the recent conversation in Context and align with the Customer Profile to pick the most relevant products or tips for their needs (e.g., stamina, fertility), limiting to 3 distinct items unless nudging to a sale, then use 3-5. "
                    f"Extract and prioritize testimonials and skills explicitly labeled or clearly implied in Product Info and Rules, pulling the most relevant ones matching the Customer Profile and recent Context (e.g., testimonials for trust, closing actions for purchase intent). "
                    f"Prioritize testimonial links when trust-building is needed (e.g., doubts about effectiveness). "
                    f"If nudging to a sale, highlight specific benefits matching their goals in 3-5 sentences, weaving in testimonials or skills. Otherwise, hint at options casually."
                )

        if feedback_type in ["satisfaction", "confirmation"]:
            if related_request:
                related_request.update({"resolved": True, "satisfied": True, "status": "RESOLVED"})
            prompt = base_prompt + " Customer seems happy—add a light push to close the deal with a product matching their profile."
            async for chunk in self.stream_response(prompt, builder):
                yield builder.build()
            return
        
        active_unresolved = [r for r in state["unresolved_requests"] if r["active"] and not r["resolved"]]
        if len(active_unresolved) > self.max_active_requests:
            oldest = min(active_unresolved, key=lambda r: r["turn"])
            builder.add(f"Nhiều câu hỏi chưa xong, như '{oldest['message']}'. Chọn cái nào hoặc 'dọn' để xóa bớt?")
            yield builder.build()
            if message.lower() in ["dọn", "bỏ"]:
                state["unresolved_requests"] = [r for r in state["unresolved_requests"] if r["active"]]
                builder.add("Đã dọn dẹp, tiếp tục nào!")
                yield builder.build()
            return
        
        if feedback_type == "new" or not related_request:
            async for chunk in self.stream_response(base_prompt, builder):
                yield builder.build()
            response_text = builder.build()
            turn = sum(1 for msg in state["messages"] if isinstance(msg, HumanMessage))
            state["unresolved_requests"].append({
                "message": message, "turn": turn, "resolved": False, "response": response_text,
                "status": "RECEIVED", "satisfied": False, "score": 0.0, "active": True,
                "bank_name": bank_name
            })
        
        elif feedback_type in ["elaboration", "clarification"]:
            action = "clarify" if feedback_type == "clarification" else "elaborate on"
            prompt = base_prompt + f" {action.capitalize()} the prior response, using product info that fits the customer profile if it flows naturally."
            async for chunk in self.stream_response(prompt, builder):
                yield builder.build()
            if related_request:
                related_request["response"] = builder.build()
    
    async def _handle_casual(self, message: str, context: str, builder: "ResponseBuilder", state: Dict, bank_name: str = ""):
        logger.info(f"Handling casual with bank_name: {bank_name}")
        
        # Fetch knowledge with fallback to empty list
        knowledge = await query_knowledge(message, bank_name=bank_name) or []
        kwcontext = "\n\n".join(entry["raw"] for entry in knowledge)
        
        # Tightened prompt mirroring _handle_request structure
        prompt = (
            f"AI: {self.name} (smart, chill)\n"
            f"Context: {context}\n"
            f"Message: '{message}'\n"
            f"Product Info: {kwcontext}\n"
            f"Task: Reply in Vietnamese. Keep it short, casual, and fun—no formal greetings. "
            f"Reason from the Context and Message to give a light, relevant response. "
            f"If Product Info is available, weave in a subtle hint about 1-2 items matching the Message, "
            f"then nudge toward sales talk naturally (e.g., 'nếu thích thì thử xem sao'). "
            f"Keep it under 3 sentences unless Product Info is empty, then just vibe with the Message."
        )
        logger.info(f"casual prompt={prompt}")
        
        async for chunk in self.stream_response(prompt, builder):
            yield chunk