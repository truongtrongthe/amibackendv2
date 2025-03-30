import asyncio
from typing import List, Dict
from langchain_openai import ChatOpenAI
from database import save_training, load_instincts, query_knowledge, find_knowledge
from utilities import logger
from langchain_core.messages import HumanMessage
import json

# Initialize the LLM
LLM = ChatOpenAI(model="gpt-4o-mini", streaming=False)
LLM_small = ChatOpenAI(model="gpt-4o-mini", streaming=False)

class ResponseBuilder:
    """A utility to build natural, flexible responses."""
    def __init__(self):
        self.parts = []
        self.error_flag = False

    def add(self, text: str, is_error: bool = False):
        if text:
            self.parts.append(text.strip())
            self.error_flag = self.error_flag or is_error
        return self

    def build(self, separator: str = None) -> str:
        if not self.parts:
            return "Tôi không biết nói gì cả!"
        separator = separator or (" " if self.error_flag else "\n")
        return separator.join(part for part in self.parts if part).strip()

class Fun:
    def __init__(self, user_id: str = "thefusionlab", state: Dict = None):
        """Initialize the Fun class with user ID and optional state."""
        self.user_id = user_id
        self.instincts = {}
        self.instinct_descriptions = {}
        self.state = state or {
            "messages": [],
            "intent_history": [],
            "preset_memory": "Be friendly",
            "unresolved_requests": [],
            "greeted": False
        }

    async def initialize(self):
        """Load instincts and descriptions from the database with fallback."""
        instincts = await load_instincts(self.user_id)
        logger.info(f"Instinct at INIT = {json.dumps(instincts, ensure_ascii=False)}")
        self.instincts = instincts.copy() if instincts else {"friendliness": "Be warm and approachable"}
        
        if "name" in self.instincts:
            self.name = self.instincts["name"]
            logger.info(f"Set AI name to: {self.name}")
        
        seen_desc = set()
        for trait, instruction in self.instincts.items():
            desc = await find_knowledge(self.user_id, primary=trait, special="description")
            if desc and desc[0]["raw"] not in seen_desc:
                self.instinct_descriptions[trait] = desc[0]["raw"]
                seen_desc.add(desc[0]["raw"])
            else:
                self.instinct_descriptions[trait] = instruction
        
        logger.info(f"Initialized - Instincts: {self.instincts}, True Self: {self.instinct_descriptions}")

    async def detect_intent(self, message: str, context: str = "") -> str:
        """Classify message intent using LLM, with feedback check for unresolved requests."""
        feedback = await self.detect_feedback_type(message, context)
        if feedback in ["correction", "adjustment", "clarification", "confusion", "elaboration"]:
            return "request"
        
        prompt = (
            f"Message: '{message}'\n"
            f"Context: '{context}'\n"
            "Classify intent as 'teaching', 'request', or 'casual'. Return one word: 'teaching', 'request', or 'casual'. "
            "If it’s a question, refinement of a prior topic, or asks for advice, classify as 'request'. "
            "If it’s conversational or lacks clear directive, classify as 'casual'."
        )
        try:
            response = await asyncio.to_thread(LLM.invoke, prompt)
            intent = response.content.strip().lower()
            return intent if intent in ["teaching", "request", "casual"] else "casual"
        except Exception as e:
            logger.info(f"Intent detection failed: {e}")
            return "casual"

    async def detect_feedback_type(self, message: str, context: str) -> str:
        """Classify user feedback based on context and message tone."""
        prompt = (
               f"Context: '{context}'\n"
                f"Message: '{message}'\n"
                f"Classify the user's feedback as one of the following: "
                f"'correction', 'adjustment', 'new', 'confirmation', 'clarification', "
                f"'rejection', 'elaboration', 'satisfaction', 'confusion'. "
                f"'new' (unrelated to prior response, introduces a fresh topic), "
                f"'confirmation' (agrees with the AI’s prior response, e.g., 'Đúng rồi,' 'Ừ' when affirming AI info, 'Chính xác'), "
                f"'clarification' (seeks more detail or rephrasing, e.g., 'Ý là gì?', 'Cái nào?'), "
                f"'rejection' (dismisses prior info or topic, e.g., 'Thôi,' 'Không cần', 'Bỏ đi'), "
                f"'elaboration' (adds details to the prior topic without changing its core intent, e.g., 'Mà,' 'Còn,' or simply more specifics tied to the same subject or entity; includes mentions of specific projects, locations, or preferences related to the ongoing topic), "
                f"'satisfaction' (signals completion or approval, e.g., 'Được rồi,' 'Ok', 'Xong'), "
                f"'confusion' (shows uncertainty or misunderstanding, e.g., 'Cái gì vậy?', 'Không hiểu'). "
                f"Analyze the message in context to detect its relationship to the prior response or user request. "
                f"Focus on tone and intent: negation for correction, agreement with AI for confirmation, questions for clarification. "
                f"Classify as 'elaboration' if the message provides additional details (e.g., budget, location, specific projects) about the same entity (e.g., 'a Minh') or topic (e.g., 'mua nhà', 'chung cư') as the prior user request or AI response, unless it explicitly negates prior info or shifts to an unrelated intent. "
                f"Return one word: 'correction', 'adjustment', 'new', 'confirmation', 'clarification', 'rejection', 'elaboration', 'satisfaction', or 'confusion'."
            )   
        try:
            response = await asyncio.to_thread(LLM_small.invoke, prompt)
            feedback = response.content.strip().replace("'", "")  # Remove quotes
            logger.info(f"feedback={feedback}")
            valid_types = [
                "correction", "adjustment", "new", "confirmation", "clarification",
                "rejection", "elaboration", "satisfaction", "confusion"
            ]
            if feedback not in valid_types:
                logger.info(f"Invalid feedback type '{feedback}', defaulting to 'new'")
                return "new"
            return feedback
        except Exception as e:
            logger.info(f"Feedback detection failed: {e}")
            return "new"

    async def resolve_related_request(self, message: str, feedback_type: str, state: Dict) -> Dict[str, any]:
        """Resolve or update a related unresolved request based on feedback type."""
        unresolved = [r for r in state.get("unresolved_requests", []) if not r["resolved"]]
        if not unresolved:
            return None

        message_tokens = set(message.lower().split())
        related_request = next(
            (r for r in unresolved if len(message_tokens.intersection(set(r["message"].lower().split()))) >= 1),
            None
        )

        if not related_request:
            return None

        if feedback_type == "confirmation" or feedback_type == "satisfaction":
            related_request["resolved"] = True
            related_request["satisfied"] = True
            related_request["status"] = "RESOLVED"
            logger.info(f"Resolved request '{related_request['message']}' as {related_request['status']}")
        elif feedback_type == "rejection":
            related_request["resolved"] = True
            related_request["satisfied"] = False
            related_request["status"] = "REJECTED"
            logger.info(f"Rejected request '{related_request['message']}'")
        elif feedback_type == "correction" or feedback_type == "adjustment":
            related_request["status"] = "CORRECTED" if feedback_type == "correction" else "ADJUSTED"
            related_request["resolved"] = False
            related_request["satisfied"] = False
            logger.info(f"Updated request '{related_request['message']}' to {related_request['status']}")
        elif feedback_type == "elaboration":
            related_request["message"] = f"{related_request['message']} - {message}"
            related_request["status"] = "ELABORATED"
            logger.info(f"Elaborated request '{related_request['message']}'")

        return related_request

    async def havefun(self, state: Dict = None, user_id: str = None) -> Dict:
        """Handle the conversation flow, ensuring state persistence."""
        state = state or self.state
        user_id = user_id or self.user_id

        if "unresolved_requests" not in state:
            state["unresolved_requests"] = []
        if "greeted" not in state:
            state["greeted"] = False
        log_state = state.copy()
        if "messages" in log_state:
            log_state["messages"] = [{"content": msg.content} if isinstance(msg, HumanMessage) else {"content": msg} for msg in log_state["messages"]]
        logger.info(f"Starting Fun - User: {user_id}, State: {json.dumps(log_state, ensure_ascii=False)}")

        if not self.instincts:
            await self.initialize()

        latest_msg = state["messages"][-1] if state["messages"] else HumanMessage(content="")
        latest_msg_content = latest_msg.content.strip() if isinstance(latest_msg, HumanMessage) else latest_msg.strip()
        context = "\n".join(f"User: {msg.content}" if isinstance(msg, HumanMessage) else f"AI: {msg}" for msg in state["messages"][-10:])  # Include latest message in context
        intent = await self.detect_intent(latest_msg_content, context)

        feedback = await self.detect_feedback_type(latest_msg_content, context)
        if feedback in ["correction", "adjustment", "clarification", "confusion", "elaboration", "new", "satisfaction"] or (feedback == "confirmation" and any(word in latest_msg_content.lower() for word in ["thích", "muốn", "chọn"])):
            intent = "request"
        else:
            intent = await self.detect_intent(latest_msg_content, context)
            
        state["intent_history"].append(intent)
        state["intent_history"] = state["intent_history"][-5:]
        state["instinct"] = " ".join(self.instincts.keys()) or "No character traits defined."
        state["unresolved_requests"] = [r for r in state["unresolved_requests"] if len(state["messages"]) - r["turn"] < 10]

        builder = ResponseBuilder()
        true_self = self._build_true_self()

        if intent == "request":
            await self._handle_request(latest_msg_content, user_id, context, builder, true_self, state)
        else:
            await self._handle_casual(latest_msg_content, context, builder, true_self, state)

        state["prompt_str"] = builder.build()
        state["messages"].append(state["prompt_str"])  # Add AI response to messages
        logger.info(f"Generated response: {state['prompt_str']}")
        self.state = state  # Update instance state
        return dict(state)

    async def _handle_request(self, message: str, user_id: str, context: str, builder: "ResponseBuilder", true_self: str, state: Dict):
        """Handle request intents with updated feedback logic."""
        is_first_message = not context and not state["greeted"]
        instinct_guidance = f"Reflect instincts: {', '.join(self.instincts.keys()) or 'none'}."
        unresolved = [r for r in state["unresolved_requests"] if not r["resolved"]]
        unresolved_str = " | ".join(f"{r['message']} (Response: {r['response']}, Status: {r['status']})" for r in unresolved) if unresolved else "Không có yêu cầu nào chưa giải quyết."

        feedback_type = await self.detect_feedback_type(message, context)
        related_request = await self.resolve_related_request(message, feedback_type, state)

        if feedback_type == "elaboration":
            logger.info("Request Elaboration Detected")
            original_msg = related_request["message"] if related_request else (state["messages"][0].content if state["messages"] and isinstance(state["messages"][0], HumanMessage) else state["messages"][0])
            prev_response = related_request["response"] if related_request else state.get("prompt_str", "")
            prompt = (
                f"You are: {self.name} a smart and funny AI\n"
                f"True self: {true_self}\n"
                f"Context: {context}\n"
                f"Original request: '{original_msg}'\n"
                f"Previous response: '{prev_response}'\n"
                f"Additional details: '{message}'\n"
                f"Task: Reply in Vietnamese. Incorporate the new details into the prior response. "
                f"Match user's tone, no greeting unless first message. End with 'ELABORATED'."
                f"{instinct_guidance}"
            )
        elif feedback_type in ["correction", "adjustment", "clarification", "confusion"]:
            logger.info(f"Request {feedback_type.capitalize()} Detected")
            original_msg = related_request["message"] if related_request else (state["messages"][0].content if state["messages"] and isinstance(state["messages"][0], HumanMessage) else state["messages"][0])
            prev_response = related_request["response"] if related_request else state.get("prompt_str", "")
            prompt = (
                f"You are: {self.name} a smart and funny AI\n"
                f"True self: {true_self}\n"
                f"Context: {context}\n"
                f"Original request: '{original_msg}'\n"
                f"Previous response: '{prev_response}'\n"
                f"User feedback: '{message}'\n"
                f"Task: Reply in Vietnamese. "
                f"{'Correct your previous response' if feedback_type == 'correction' else 'Adjust based on feedback' if feedback_type == 'adjustment' else 'Clarify or simplify based on feedback'} "
                f"with analysis (why it was wrong/needed tweak/needed clarification, how it’s improved). "
                f"Match user's tone, no greeting unless first message, no questions unless invited. "
                f"End with '{feedback_type.upper()}'."
                f"{instinct_guidance}"
            )
        elif feedback_type == "confirmation":
            if any(word in message.lower() for word in ["thích", "muốn", "chọn"]):  # Preference indicators
                logger.info("Request Elaboration Detected")
                original_msg = related_request["message"] if related_request else (state["messages"][0].content if state["messages"] and isinstance(state["messages"][0], HumanMessage) else state["messages"][0])
                prev_response = related_request["response"] if related_request else state.get("prompt_str", "")
                prompt = (
                    f"You are: {self.name} a smart and funny AI\n"
                    f"True self: {true_self}\n"
                    f"Context: {context}\n"
                    f"Original request: '{original_msg}'\n"
                    f"Previous response: '{prev_response}'\n"
                    f"Additional details: '{message}'\n"
                    f"Task: Reply in Vietnamese. Incorporate the new preference into the prior response. "
                    f"Match user's tone, no greeting unless first message. End with 'ELABORATED'."
                    f"{instinct_guidance}"
                )
            else:
                builder.add("Tuyệt, em đúng ý anh/chị rồi nhé! Có gì cần thêm không?")
                return
        elif feedback_type == "satisfaction":
            builder.add("Vậy là xong, mừng quá! Còn gì thú vị nữa không?")
            return
        elif feedback_type == "rejection":
            builder.add("OK, bỏ qua cái đó. Anh/chị muốn gì tiếp theo?")
            return
        elif feedback_type == "new":
            logger.info("Pure Request Intent")
            prompt = (
                f"You are: {self.name} a smart and funny AI\n"
                f"True self: {true_self}\n"
                f"Context: {context}\n"
                f"Unresolved requests: '{unresolved_str}'\n"
                f"Current request: '{message}'\n"
                f"Task: Reply in Vietnamese. Follow the user's intent strictly. "
                f"Provide a detailed response with analysis if applicable, match user's tone. "
                f"End with 'RECEIVED'."
                f"{instinct_guidance}"
            )
        else:
            logger.info(f"Unhandled feedback type '{feedback_type}', treating as elaboration")
            original_msg = related_request["message"] if related_request else (state["messages"][0].content if state["messages"] and isinstance(state["messages"][0], HumanMessage) else state["messages"][0])
            prev_response = related_request["response"] if related_request else state.get("prompt_str", "")
            prompt = (
                f"You are: {self.name} a smart and funny AI\n"
                f"True self: {true_self}\n"
                f"Context: {context}\n"
                f"Original request: '{original_msg}'\n"
                f"Previous response: '{prev_response}'\n"
                f"Additional details: '{message}'\n"
                f"Task: Reply in Vietnamese. Incorporate the new details into the prior response. "
                f"Match user's tone, no greeting unless first message. End with 'ELABORATED'."
                f"{instinct_guidance}"
            )

        try:
            response = await asyncio.to_thread(LLM_small.invoke, prompt)
            response_text = response.content.strip()
            builder.add(response_text)

            if not related_request and feedback_type == "new":
                state["unresolved_requests"].append({
                    "message": message,
                    "turn": len(state["messages"]),
                    "resolved": False,
                    "response": response_text,
                    "status": "RECEIVED",
                    "satisfied": False
                })
                logger.info(f"Marked request '{message}' as RECEIVED.")
            elif related_request and feedback_type in ["correction", "adjustment", "elaboration"]:
                related_request["response"] = response_text

            if is_first_message:
                state["greeted"] = True
        except Exception as e:
            logger.info(f"LLM failed for request '{message}': {e}")
            builder.add(f"Hơi lúng túng tí, mình nói lại nhé?")


    async def _handle_casual(self, message: str, context: str, builder: "ResponseBuilder", true_self: str, state: Dict):
        is_first_message = not context and not state["greeted"]
        instinct_guidance = f"Reflect instincts: {', '.join(self.instincts.keys()) or 'none'}."
        logger.info("Casual Intent")
        prompt = (
            f"You are: {self.name} a smart and funny AI\n"
            f"True self: {true_self}\n"
            f"Context: {context}\n"
            f"Message: '{message}'\n"
            f"Task: Reply in Vietnamese. "
            f"Keep it short and natural, match the user's tone, no greeting unless first message, "
            f"avoid questions unless the message invites elaboration."
            f"{'Xin chào mình!' if is_first_message else ''}"
            f"{instinct_guidance}"
        )
        try:
            response = await asyncio.to_thread(LLM_small.invoke, prompt)
            builder.add(response.content.strip())
            if is_first_message:
                state["greeted"] = True
        except Exception as e:
            logger.info(f"LLM failed for casual '{message}': {e}")
            builder.add(f"Hơi lag tí, mình nói lại nhé?")

    def _build_true_self(self) -> str:
        """Construct a string describing the AI's true self."""
        if not self.instinct_descriptions:
            return "Tôi đang học cách trở nên đặc biệt hơn."
        return "Tôi với bản năng gốc: " + ", ".join(
            f"{key} ({value})" for key, value in self.instinct_descriptions.items()
        )