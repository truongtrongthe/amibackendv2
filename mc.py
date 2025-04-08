import asyncio
from typing import List, Dict
from langchain_openai import ChatOpenAI
from utilities import logger
from langchain_core.messages import HumanMessage,AIMessage
import json
from sentence_transformers import SentenceTransformer, util
import numpy as np
from database import query_knowledge
from brainlog import create_brain_log
import re
from typing import Tuple

def add_messages(existing_messages, new_messages):
    return existing_messages + new_messages

LLM = ChatOpenAI(model="gpt-4o", streaming=False)
StreamLLM = ChatOpenAI(model="gpt-4o", streaming=True)

FEEDBACKTYPE = ["correction", "adjustment", "new", "confirmation", "clarification",
                "rejection", "elaboration", "satisfaction", "confusion"]

# Vietnamese examples for each feedback type
FEEDBACK_EXAMPLES = {
    "correction": [
        "Không phải vậy đâu",
        "Sai rồi, phải là...",
        "Em hiểu nhầm rồi"
    ],
    "adjustment": [
        "Cho em sửa lại một chút",
        "Em muốn thay đổi...",
        "Có thể điều chỉnh...",
        "Không phải,..."
    ],
    "new": [
        "Em muốn hỏi về...",
        "Cho em hỏi...",
        "Em cần tìm hiểu..."
    ],
    "confirmation": [
        "Đúng rồi",
        "Vâng, em hiểu rồi",
        "Chính xác",
        "Ok",
        "OK em"
    ],
    "clarification": [
        "Em chưa hiểu lắm",
        "Có thể giải thích rõ hơn không?",
        "Ý anh là..."
    ],
    "rejection": [
        "Em không cần",
        "Không phù hợp",
        "Không đúng ý em"
    ],
    "elaboration": [
        "Cụ thể là...",
        "Chi tiết hơn thì...",
        "Ngoài ra còn..."
    ],
    "satisfaction": [
        "Tuyệt vời",
        "Đúng ý em",
        "Em rất hài lòng"
    ],
    "confusion": [
        "Em không hiểu",
        "Hơi khó hiểu",
        "Mơ hồ quá"
    ]
}

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
        self.name = "Ami"
        self.instincts = {"friendly": "Be nice"}
        self.similarity_threshold = similarity_threshold
        self.max_active_requests = max_active_requests
        self.state = {
            "messages": [],
            "intent_history": [],
            "preset_memory": "Be friendly",
            "unresolved_requests": [],
            "convo_id": self.convo_id,
            "user_id": self.user_id,
            "prompt_str": "",
            "bank_name":"",
            "brain_uuid":""
        }
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
        logger.info(f"detect feedback of: {message} with context:{context}")
        
        # Build examples string
        examples_str = "\n".join([
            f"- {ftype}: {', '.join(examples)}"
            for ftype, examples in FEEDBACK_EXAMPLES.items()
        ])
        
        prompt = (
            f"Context: '{context}'\nMessage: '{message}'\n\n"
            f"Phân loại phản hồi của người dùng thành một trong các loại sau:\n"
            f"{examples_str}\n\n"
            f"Lưu ý:\n"
            f"1. Xem xét ngữ cảnh và cách diễn đạt tiếng Việt\n"
            f"2. Chú ý các từ ngữ đặc trưng trong tiếng Việt\n"
            f"3. Cân nhắc cả ý nghĩa trực tiếp và hàm ý\n"
            f"4. Trả về một từ duy nhất trong danh sách: {', '.join(FEEDBACKTYPE)}"
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
    async def trigger(self, state: Dict = None, user_id: str = None, bank_name: str = None, brain_uuid :str =None,config: Dict = None):
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
        
        brain_uuid = (
            brain_uuid if brain_uuid is not None 
            else state.get("brain_uuid", 
                config.get("configurable", {}).get("brain_uuid", 
                    self.bank_name if hasattr(self, 'brain_uuid') else ""))
        )
        
        # Sync with mc.state
        self.state["bank_name"] = bank_name
        self.state["brain_uuid"] = brain_uuid
        if "unresolved_requests" not in state:
            state["unresolved_requests"] = self.state.get("unresolved_requests", [])

        latest_msg = state["messages"][-1] if state["messages"] else HumanMessage(content="")
        latest_msg_content = latest_msg.content.strip() if isinstance(latest_msg, HumanMessage) else latest_msg.strip()
        context = "\n".join(f"User: {msg.content}" if isinstance(msg, HumanMessage) else f"AI: {msg.content}" for msg in state["messages"][-26:])

        logger.info(f"Triggering - User: {user_id}, Bank: {bank_name}, latest_msg: {latest_msg_content}, context: {context}")

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
            async for response_chunk in self._handle_request(latest_msg_content, user_id, context, builder, feedback, state, intent, bank_name=bank_name,brain_uuid=brain_uuid):
                state["prompt_str"] = response_chunk
                logger.debug(f"Yielding from trigger: {state['prompt_str']}")
                yield response_chunk
        else:
            async for response_chunk in self._handle_casual(latest_msg_content, context, builder, state, bank_name=bank_name):
                state["prompt_str"] = response_chunk
                logger.debug(f"Yielding from trigger: {state['prompt_str']}")
                yield response_chunk
        
        # Wrap response as AIMessage and append using add_messages
        if state["prompt_str"]:
            state["messages"] = add_messages(state["messages"], [AIMessage(content=state["prompt_str"])])
        
        self.state.update(state)
        logger.info(f"Final response: {state['prompt_str']}")
        
        # Yield the final state as a special chunk
        yield {"state": state}
    
    async def _handle_request(self, message: str, user_id: str, context: str, builder: "ResponseBuilder", 
                         feedback_type: str, state: Dict, intent: str, bank_name: str = "",brain_uuid: str =""):
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
        
        """
         TESTING PROMPT
        """
        
        profile_prompt_good = (
            f"Based on the conversation:\n{context}\n\n"
            f"Follow these instructions to shape the profile and drive the next-step action:\n{kwcontext if kwcontext else 'No specific instructions available; infer from conversation context alone.'}\n\n"
            f"Create a concise customer profile capturing their core interests, needs, and hidden desires to guide an AI sales response. Focus on the most recent messages to pinpoint what's driving them now, weaving in earlier patterns if they align with the instructions.\n\n"
            f"Interests: What they're drawn to (e.g., practical solutions, quick results), guided by the instructions.\n"
            f"Needs: What they're seeking (e.g., guidance, confidence), tied to the instructions' focus.\n"
            f"Hidden desires: Subtle motivations (e.g., self-assurance, partner satisfaction), reflecting the instructions' insights.\n\n"
            f"Add specific cues for sales: Product preferences (e.g., course features), pain points (e.g., embarrassment), and emotional triggers (e.g., loss of confidence) that match the instructions' suggested approach.\n\n"
            f"If their focus shifts, note the trigger and adjust only if the instructions allow—otherwise, stay consistent with prior traits unless contradicted.\n\n"
            f"Summarize in 1-2 sentences with a clear next-step action (e.g., 'Ask probing questions to uncover causes, then pitch course') that strictly follows the instructions' sequence and intent, defaulting to a context-based action if no instructions are provided."
        )
        profile_prompt = (
            f"Based on the conversation:\n{context}\n\n"
            f"Use these instructions as your strict guide to analyze the situation and determine the next step, following their exact wording and sequence:\n{kwcontext if kwcontext else 'No specific instructions available; infer from conversation context alone.'}\n\n"
            f"Create a concise customer profile capturing their core interests, needs, and hidden desires to guide an AI sales response. Focus on the most recent messages to pinpoint what's driving them now, weaving in earlier patterns if they align with the instructions.\n\n"
            f"Interests: What they're drawn to (e.g., practical solutions, quick results), guided by the instructions.\n"
            f"Needs: What they're seeking (e.g., guidance, confidence), tied to the instructions' focus.\n"
            f"Hidden desires: Subtle motivations (e.g., self-assurance, partner satisfaction), reflecting the instructions' insights.\n\n"
            f"Add specific cues for sales: Product preferences (e.g., course features), pain points (e.g., embarrassment), and emotional triggers (e.g., loss of confidence) that match the instructions' suggested approach.\n\n"
            f"If their focus shifts, note the trigger and adjust only if the instructions allow—otherwise, stay consistent with prior traits unless contradicted.\n\n"
            f"Summarize in 1-2 sentences with a clear next-step action: analyze the current conversation state (e.g., what the customer has said, how much they've revealed, their apparent intent); then, strictly follow the instructions' sequence—start by asking the exact open-ended questions provided (e.g., 'Anh gặp trường hợp này từ lúc mới bắt đầu quan hệ, hay gần đây mới bị anh nhỉ?') if no responses yet; after they respond, proceed to analyze their answers against the listed causes (psychological, physiological, habits), then offer empathy and motivation using the instructed phrasing (e.g., 'Em hiểu cảm giác này...'); only pitch the product combo ('1 year for 399K or 2 years for 599K + 3 free eBooks') if they show explicit buying intent (e.g., 'send it,' 'I want it') after earlier steps; otherwise, continue probing or building trust with the next instructed step, defaulting to a context-based open-ended question if instructions are unclear."
            )

        customer_profile = LLM.invoke(profile_prompt).content  # Sync call, full response
        analysis = "\n".join(f"Customer Profile: {customer_profile}")
        logger.info(f"Customer profile built: {customer_profile}")
        
        
        # Measure feasibility
        possibility, missing = await self._measure_feasibility(message, context, customer_profile, kwcontext)
        analysis = "\n".join([
            f"Customer Profile Feasibility:",
            f"  - Possibility: {possibility}%",
            f"  - Missing Information: {', '.join(missing) if missing else 'Không thiếu thông tin nào'}"
        ])

        # Optional: Log or use the analysis
        logger.info(analysis)

        base_prompt = (
            f"AI: {self.name}\n"
            f"Context: {context}\n"
            f"Message: '{message}'\n"
            f"Customer Profile: {customer_profile}\n"
            f"Rules: {kwcontext}\n"
            f"Task: Reply in Vietnamese in a casual and friendly tone. Keep answer short in 2-3 sentences. Avoid repeating greetings if one is already in Context.\n\n"
            f"1. Analyze the conversation flow holistically. Prioritize recent exchanges but reference past messages naturally where relevant.\n\n"
            f"2. Strictly follow the Customer Profile's next-step action as your sole guide, executing its full sequence ('acknowledge… then… proceed to…') exactly as written, aligning with the Rules' intent—do not skip or stop short of any step.\n\n"
            f"3. Use the exact phrasing and actions from the Rules that match the profile's current next-step; include testimonials or product offers only when the profile explicitly directs it ('proceed to the product pitch')—do not omit them if instructed.\n\n"
            f"4. If hesitation appears in Context, address it with empathy from Rules only if the profile's next-step calls for it—otherwise, stay on the instructed action.\n\n"
            f"5. Keep the tone conversational and tied to the profile's interests, needs, and cues, fully completing the current step in the sequence.\n\n"
            f"6. **Debug Info**: Use the provided Customer Profile as-is; if missing, state 'Customer Profile incomplete, assuming curiosity-driven interest' and proceed."
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
            await self._log_to_brain(brain_uuid, message, response_text, analysis)
        elif feedback_type in ["elaboration", "clarification"]:
            logger.info("Jump to CLARIFICATION!")
            action = "clarify" if feedback_type == "clarification" else "elaborate on"
            prompt = base_prompt + f" {action.capitalize()} the prior response, using product info that fits the customer profile if it flows naturally."
            async for chunk in self.stream_response(prompt, builder):
                yield builder.build()
            response_text = builder.build()
            if related_request:
                related_request["response"] = builder.build()
            await self._log_to_brain(brain_uuid, message, response_text, analysis)
        
    async def _handle_casual(self, message: str, context: str, builder: "ResponseBuilder", state: Dict, bank_name: str = ""):
        logger.info(f"Handling CASUAL. Bank_name: {bank_name}")
        
        # Fetch knowledge with fallback to empty list
        knowledge = await query_knowledge(message, bank_name=bank_name) or []
        kwcontext = "\n\n".join(entry["raw"] for entry in knowledge)
        
        # Tightened prompt mirroring _handle_request structure
        prompt = (
                f"AI: {self.name} (smart, chill)\n"
                f"Context: {context}\n"
                f"Message: '{message}'\n"
                f"Task: Reply in Vietnamese in **2 short sentences**—casual, fun, and effortless. No formal greetings or repeated vibes from Context.\n\n"
                f"Skim Context to catch the flow and reply naturally. Skip greetings if one already exists."
            )

        #logger.info(f"casual prompt={prompt}")
        
        async for chunk in self.stream_response(prompt, builder):
            yield chunk
    
    async def _measure_feasibility(self, message: str, context: str, customer_profile: str, kwcontext: str) -> Tuple[int, list[str]]:
        """
        Evaluate the feasibility of resolving the user's request based on available information and the customer profile's next-step action.
        """
        possibility_prompt = (
            f"AI: {self.name}\n"
            f"Context: {context if context else 'No conversation context provided; assume a generic customer interaction.'}\n"
            f"Message: '{message if message else 'User request is unspecified; infer intent from context or profile.'}'\n"
            f"Customer Profile: {customer_profile if customer_profile else 'No profile generated yet; assume a blank slate.'}\n"
            f"Rules: {kwcontext if kwcontext else 'No specific instructions available; use general reasoning based on context and profile.'}\n"
            f"Task: Assess whether the AI can successfully execute the next-step action outlined in the Customer Profile (treating any step as valid), given the provided context, message, and rules.\n"
            f"Output: Return a valid JSON object with:\n"
            f"- 'possibility': Integer (0-100) representing the percentage chance of success.\n"
            f"- 'missing': List of specific missing information or resources (in Vietnamese) needed to increase the possibility, or an empty list if none.\n"
            f"Consider: Availability of data (context, message, rules), clarity of the next-step action, and the AI's capability to act on it.\n"
        )
        try:
            # Handle both sync and async LLM.invoke
            if asyncio.iscoroutinefunction(LLM.invoke):
                response = await LLM.invoke(possibility_prompt)
            else:
                response = LLM.invoke(possibility_prompt)

            # Extract content from response (assuming AIMessage or similar)
            raw_response = getattr(response, 'content', response).strip()
            logger.debug(f"Preliminary reasoning raw response: '{raw_response}'")

            if not raw_response:
                logger.warning("LLM returned an empty response")
                return 0, ["Không có phản hồi từ LLM để đánh giá"]

            # Parse JSON
            try:
                prelim_result = json.loads(raw_response)
            except json.JSONDecodeError:
                json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
                if json_match:
                    prelim_result = json.loads(json_match.group(0))
                else:
                    logger.error(f"Invalid JSON in response: '{raw_response}'")
                    return 0, ["Phản hồi từ LLM không chứa JSON hợp lệ"]

            # Validate fields
            required_fields = {"possibility", "missing"}
            if not required_fields.issubset(prelim_result.keys()):
                missing_fields = required_fields - set(prelim_result.keys())
                return 0, [f"Thiếu trường dữ liệu cần thiết: {', '.join(missing_fields)}"]

            possibility = prelim_result["possibility"]
            if not isinstance(possibility, (int, float)) or not 0 <= possibility <= 100:
                return 0, [f"Giá trị 'possibility' không hợp lệ: {possibility}"]

            missing = prelim_result["missing"]
            if not isinstance(missing, list) or not all(isinstance(item, str) for item in missing):
                return 0, [f"Danh sách 'missing' không hợp lệ: {missing}"]

            return int(possibility), missing

        except Exception as e:
            logger.error(f"Feasibility evaluation failed for '{message}': {str(e)}")
            return 0, [f"Lỗi hệ thống khi đánh giá: {str(e)}"]
    
    async def _log_to_brain(self, brainid: str, request: str, response: str, gaps: str) -> str:
        """
        Log a conversation entry to the brain system and return the created entry ID.
        
        Args:
            brainid (str): Unique identifier for the brain instance.
            request (str): The user's input or request.
            response (str): The AI's response to the request.
            gaps (str): Information about missing data or gaps (e.g., from feasibility analysis).
        
        Returns:
            str: The ID of the created log entry.
        
        Raises:
            ValueError: If required inputs are empty or invalid.
            Exception: If log creation fails.
        """
        # Validate inputs
        if not all([brainid, request, response]):
            raise ValueError("brainid, request, and response must not be empty")

        # Log the brain UUID for tracking
        logger.info(f"BrainUUID: {brainid}")

        # Format the log entry
        entry = f"Human: {request}. AI: {response}"
        logger.debug(f"Logging entry: '{entry}' with gaps: '{gaps}'")

        try:
            # Assuming create_brain_log is an async function; adjust if it's sync
            new_log = await create_brain_log(brainid, entry, gaps) if asyncio.iscoroutinefunction(create_brain_log) else create_brain_log(brainid, entry, gaps)
            
            # Verify the log was created and has an entry_id
            if not hasattr(new_log, 'entry_id') or not new_log.entry_id:
                raise ValueError("Created log missing entry_id")
            
            # Log and print success
            logger.info(f"Created log entry: {new_log.entry_id}")
            print(f"Created log: {new_log.entry_id}")
            
            return new_log.entry_id

        except Exception as e:
            logger.error(f"Failed to log to brain {brainid}: {str(e)}")
            raise Exception(f"Log creation failed: {str(e)}") from e