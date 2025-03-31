import asyncio
from typing import List, Dict
from langchain_openai import ChatOpenAI
from database import save_training, load_instincts, query_knowledge, find_knowledge
from utilities import logger
from langchain_core.messages import HumanMessage
import json

# Initialize the LLMs
LLM = ChatOpenAI(model="gpt-4o-mini", streaming=False)  # For intent detection
LLM_small = ChatOpenAI(model="gpt-4o-mini", streaming=False)  # For feedback detection
StreamLLM = ChatOpenAI(model="gpt-4o", streaming=True)  # For streaming responses

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

class Fun:
    def __init__(self, user_id: str = "thefusionlab", state: Dict = None):
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
        prompt = (
            f"Context: '{context}'\n"
            f"Message: '{message}'\n"
            f"Classify feedback as: 'correction', 'adjustment', 'new', 'confirmation', 'clarification', "
            f"'rejection', 'elaboration', 'satisfaction', 'confusion'. Return one word."
        )
        try:
            response = await asyncio.to_thread(LLM_small.invoke, prompt)
            feedback = response.content.strip().replace("'", "")
            valid_types = ["correction", "adjustment", "new", "confirmation", "clarification",
                           "rejection", "elaboration", "satisfaction", "confusion"]
            return feedback if feedback in valid_types else "new"
        except Exception as e:
            logger.info(f"Feedback detection failed: {e}")
            return "new"

    async def resolve_related_request(self, message: str, feedback_type: str, state: Dict) -> Dict[str, any]:
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

        if feedback_type in ["confirmation", "satisfaction"]:
            related_request["resolved"] = True
            related_request["satisfied"] = True
            related_request["status"] = "RESOLVED"
        elif feedback_type == "rejection":
            related_request["resolved"] = True
            related_request["satisfied"] = False
            related_request["status"] = "REJECTED"
        elif feedback_type in ["correction", "adjustment"]:
            related_request["status"] = "CORRECTED" if feedback_type == "correction" else "ADJUSTED"
            related_request["resolved"] = False
        elif feedback_type == "elaboration":
            related_request["message"] = f"{related_request['message']} - {message}"
            related_request["status"] = "ELABORATED"

        return related_request

    # New shared streaming method
    async def stream_response(self, prompt: str, builder: ResponseBuilder):
        buffer = ""
        open_markdown = 0
        try:
            async for chunk in StreamLLM.astream(prompt):
                buffer += chunk.content
                open_markdown += chunk.content.count("**")
                if (("\n" in buffer or buffer.endswith((".", "!", "?"))) and open_markdown % 2 == 0) or len(buffer) > 500:
                    parts = buffer.split("\n", 1)
                    if len(parts) > 1:
                        builder.add(parts[0] + "\n")
                        yield builder.build(separator="")
                        buffer = parts[1]
                    else:
                        builder.add(buffer)
                        yield builder.build(separator="")
                        buffer = ""
                    open_markdown = buffer.count("**")
            if buffer:
                builder.add(buffer)
                yield builder.build(separator="")
        except Exception as e:
            logger.info(f"Streaming failed: {e}")
            builder.add("Có lỗi nhỏ, thử lại nhé!")
            yield builder.build(separator="")

    async def havefun(self, state: Dict = None, user_id: str = None):
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
        context = "\n".join(f"User: {msg.content}" if isinstance(msg, HumanMessage) else f"AI: {msg}" for msg in state["messages"][-10:])
        
        # Refined intent detection
        intent = await self.detect_intent(latest_msg_content, context)
        feedback = await self.detect_feedback_type(latest_msg_content, context)
        if feedback in ["correction", "adjustment", "clarification", "confusion", "elaboration"]:
            intent = "request"  # Actionable feedback overrides
        elif feedback == "new" and ("?" in latest_msg_content or any(word in latest_msg_content.lower() for word in ["làm", "giúp", "cho"])):
            intent = "request"  # New topic only if directive/question

        state["intent_history"].append(intent)
        state["intent_history"] = state["intent_history"][-5:]
        state["instinct"] = " ".join(self.instincts.keys()) or "No character traits defined."
        state["unresolved_requests"] = [r for r in state["unresolved_requests"] if len(state["messages"]) - r["turn"] < 10]

        builder = ResponseBuilder()
        true_self = self._build_true_self()

        if intent == "request":
            async for response_chunk in self._handle_request(latest_msg_content, user_id, context, builder, true_self, state):
                state["prompt_str"] = response_chunk
                yield state["prompt_str"]
        else:
            async for response_chunk in self._handle_casual(latest_msg_content, context, builder, true_self, state):
                state["prompt_str"] = response_chunk
                yield state["prompt_str"]

        logger.info(f"Final response: {state['prompt_str']}")
        self.state = state

    async def _handle_request(self, message: str, user_id: str, context: str, builder: "ResponseBuilder", true_self: str, state: Dict):
        is_first_message = not context and not state["greeted"]
        instinct_guidance = f"Reflect instincts: {', '.join(self.instincts.keys()) or 'none'}."
        unresolved_str = " | ".join(f"{r['message']} (Status: {r['status']})" for r in state["unresolved_requests"] if not r["resolved"]) or "Không có."

        feedback_type = await self.detect_feedback_type(message, context)
        related_request = await self.resolve_related_request(message, feedback_type, state)

        # Quick responses for simple feedback
        if feedback_type == "satisfaction":
            builder.add("Được rồi, vui quá! Còn gì nữa không?")
            yield builder.build()
            return
        elif feedback_type == "confirmation" and not any(word in message.lower() for word in ["thích", "muốn", "chọn"]):
            builder.add("Tuyệt, hợp ý rồi nhé!")
            yield builder.build()
            return
        elif feedback_type == "rejection":
            builder.add("OK, bỏ qua vậy. Tiếp theo đi!")
            yield builder.build()
            return

        # LLM for complex cases
        if feedback_type == "new":
            prompt = (
                f"AI: {self.name} (smart, funny)\n"
                f"True self: {true_self}\n"
                f"Context: {context}\n"
                f"Request: '{message}'\n"
                f"Task: Reply in Vietnamese, detailed if needed, match tone. End with 'RECEIVED'.\n"
                f"{instinct_guidance}"
            )
        elif feedback_type == "elaboration":
            original_msg = related_request["message"] if related_request else (state["messages"][0].content if state["messages"] else "")
            prev_response = related_request["response"] if related_request else state.get("prompt_str", "")
            prompt = (
                f"AI: {self.name}\n"
                f"True self: {true_self}\n"
                f"Context: {context}\n"
                f"Original: '{original_msg}'\n"
                f"Previous: '{prev_response}'\n"
                f"Details: '{message}'\n"
                f"Task: Reply in Vietnamese, blend in new details, match tone. End with 'ELABORATED'.\n"
                f"{instinct_guidance}"
            )
        elif feedback_type in ["correction", "adjustment"]:
            original_msg = related_request["message"] if related_request else (state["messages"][0].content if state["messages"] else "")
            prev_response = related_request["response"] if related_request else state.get("prompt_str", "")
            prompt = (
                f"AI: {self.name}\n"
                f"True self: {true_self}\n"
                f"Context: {context}\n"
                f"Original: '{original_msg}'\n"
                f"Previous: '{prev_response}'\n"
                f"Feedback: '{message}'\n"
                f"Task: Reply in Vietnamese, {'correct' if feedback_type == 'correction' else 'adjust'} previous response, explain why, match tone. End with '{feedback_type.upper()}'.\n"
                f"{instinct_guidance}"
            )
        elif feedback_type in ["clarification", "confusion"]:
            original_msg = related_request["message"] if related_request else (state["messages"][0].content if state["messages"] else "")
            prev_response = related_request["response"] if related_request else state.get("prompt_str", "")
            prompt = (
                f"AI: {self.name}\n"
                f"True self: {true_self}\n"
                f"Context: {context}\n"
                f"Original: '{original_msg}'\n"
                f"Previous: '{prev_response}'\n"
                f"Feedback: '{message}'\n"
                f"Task: Reply in Vietnamese, clarify or simplify, match tone. End with '{feedback_type.upper()}'.\n"
                f"{instinct_guidance}"
            )
        else:
            prompt = (
                f"AI: {self.name}\n"
                f"True self: {true_self}\n"
                f"Context: {context}\n"
                f"Message: '{message}'\n"
                f"Task: Reply in Vietnamese, match tone. End with 'RECEIVED'.\n"
                f"{instinct_guidance}"
            )

        async for chunk in self.stream_response(prompt, builder):
            yield chunk

        response_text = builder.build()
        if feedback_type == "new" and not related_request:
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

    async def _handle_casual(self, message: str, context: str, builder: "ResponseBuilder", true_self: str, state: Dict):
        is_first_message = not context and not state["greeted"]
        instinct_guidance = f"Reflect instincts: {', '.join(self.instincts.keys()) or 'none'}."

        prompt = (
            f"AI: {self.name} (smart, funny)\n"
            f"True self: {true_self}\n"
            f"Context: {context}\n"
            f"Message: '{message}'\n"
            f"Task: Reply in Vietnamese, short and natural, match tone. "
            f"Add a light follow-up (e.g., 'Còn gì hay không?') only if context suggests interest. "
            f"{'Xin chào nha!' if is_first_message else ''}\n"
            f"{instinct_guidance}"
        )

        async for chunk in self.stream_response(prompt, builder):
            yield chunk
        
        if is_first_message:
            state["greeted"] = True

    def _build_true_self(self) -> str:
        if not self.instinct_descriptions:
            return "Tôi đang học cách trở nên đặc biệt hơn."
        return "Tôi với bản năng gốc: " + ", ".join(
            f"{key} ({value})" for key, value in self.instinct_descriptions.items()
        )