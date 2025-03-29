# training.py
# Purpose: AI Training class with intent detection and response generation
# Date: March 28, 2025

import asyncio
from typing import List, Dict
from langchain_openai import ChatOpenAI
from database import save_training, load_instincts, query_knowledge, find_knowledge
from utilities import logger
from langchain_core.messages import HumanMessage
import json

# Initialize the LLM
LLM = ChatOpenAI(model="gpt-4o", streaming=False)

class ResponseBuilder:
    """A utility to build natural, flexible responses."""
    def __init__(self, name: str, use_greeting: bool = False):
        self.name = name or "AI"
        self.parts = []
        self.error_flag = False
        if use_greeting:
            self.parts.append(f"Chào, tôi là {self.name}, rất vui được trò chuyện!")

    def add(self, text: str, is_error: bool = False):
        if text:
            self.parts.append(text.strip())
            self.error_flag = self.error_flag or is_error
        return self

    def build(self) -> str:
        if not self.parts:
            return "Tôi không biết nói gì cả!"
        # Use space for errors, period for normal flow, but let LLM handle most punctuation
        separator = " " if self.error_flag else " "
        return separator.join(part for part in self.parts if part).strip()

class Training:
    def __init__(self, user_id: str = "thefusionlab", state: Dict = None):
        """Initialize the Training class with user ID and optional state."""
        self.user_id = user_id
        self.name = None
        self.instincts = {}  # Character traits like "curiosity", "truthfulness"
        self.instinct_descriptions = {}  # Detailed "true self" definitions
        self.state = state or {
            "messages": [],
            "intent_history": [],
            "preset_memory": "Be friendly"
        }

    async def initialize(self):
        """Load instincts and descriptions from the database."""
        instincts = await load_instincts(self.user_id)
        logger.debug(f"Instinct at INIT = {json.dumps(instincts, ensure_ascii=False)}")
        self.instincts = instincts.copy() if instincts else {}
        
        if not instincts:
            logger.warning(f"No instincts loaded for user {self.user_id}")
            return
        
        if "name" in instincts and not self.name:
            self.name = instincts["name"]
            logger.debug(f"Set name from instincts: {self.name}")
        
        seen_desc = set()
        for trait, instruction in self.instincts.items():
            if trait != "name":
                desc = await find_knowledge(self.user_id, primary=trait, special="description")
                if desc and desc[0]["raw"] not in seen_desc:
                    self.instinct_descriptions[trait] = desc[0]["raw"]
                    seen_desc.add(desc[0]["raw"])
                else:
                    self.instinct_descriptions[trait] = instruction
        
        logger.debug(f"Initialized - Name: {self.name}, Instincts: {self.instincts}, True Self Descriptions: {self.instinct_descriptions}")

    async def detect_intent(self, message: str) -> str:
        """Classify message intent using LLM."""
        prompt = (
            f"Message: '{message}'\n"
            "Classify as 'teaching', 'request', or 'casual'. Focus on the intent, not just the tone:\n"
            "- 'teaching': Declares, defines, instructs, or sets AI attributes\n"
            "- 'request': Seeks a response or action\n"
            "- 'casual': Chats without specific purpose\n"
            "Return only the intent as a string"
        )
        try:
            response = await asyncio.to_thread(LLM.invoke, prompt)
            intent = response.content.strip().strip("'")
            logger.debug(f"LLM intent for '{message}': {intent}")
            return intent if intent in ["teaching", "request", "casual"] else "casual"
        except Exception as e:
            logger.error(f"Intent detection failed: {e}")
            return "casual"

    async def training(self, state: Dict = None, user_id: str = None) -> Dict:
        state = state or self.state
        user_id = user_id or self.user_id
        
        log_state = state.copy()
        if "messages" in log_state:
            log_state["messages"] = [msg.model_dump() if isinstance(msg, HumanMessage) else msg for msg in log_state["messages"]]
        logger.debug(f"Starting training - User: {user_id}, State: {json.dumps(log_state, ensure_ascii=False)}")

        # Only process the latest message for intent and saving
        if not self.instincts:
            await self.initialize()

        latest_msg = state["messages"][-1] if state["messages"] else HumanMessage(content="")
        state["messages"] = state["messages"][-200:]
        latest_msg_content = latest_msg.content.strip()
        intent = await self.detect_intent(latest_msg_content)
        state["intent_history"].append(intent)
        state["intent_history"] = state["intent_history"][-5:]
        state["instinct"] = " ".join(self.instincts.keys()) or "No character traits defined."

        context = "\n".join(msg.content for msg in state["messages"][-10:-1]) if len(state["messages"]) > 1 else ""
        logger.info(f"Detected intent for '{latest_msg_content}': {intent}")

        use_greeting = not state["messages"] or len(state["messages"]) == 1
        builder = ResponseBuilder(self.name, use_greeting=use_greeting)
        true_self = self._build_true_self()

        if intent == "teaching" and latest_msg_content:
            await self._handle_teaching(latest_msg_content, user_id, context, builder, true_self)
        elif intent == "request":
            await self._handle_request(latest_msg_content, user_id, context, builder, true_self)
        else:  # casual
            await self._handle_casual(latest_msg_content, context, builder, true_self, state.get("preset_memory", ""))

        state["prompt_str"] = builder.build()
        logger.debug(f"Generated response: {state['prompt_str']}")
        self.state = state
        return dict(state)
    
    async def _handle_teaching(self, message: str, user_id: str, context: str, builder: "ResponseBuilder", true_self: str):
        """Handle teaching intent, saving only character traits or name changes."""
        if not message.strip():
            builder.add("Bạn chưa dạy gì mà! Có gì hay ho để tôi học không?")
            return

        # Detect if this is a name or instinct teaching
        is_name_teaching = "em xưng là" in message.lower() or "tên là" in message.lower()  # Simple heuristic
        is_instinct_teaching = any(trait in message.lower() for 
                                   trait in ["humility", "learning", "respect", "confidence", "humor"])  # Expand as needed

        if is_name_teaching or is_instinct_teaching:
            try:
                await save_training(message, user_id, context)
                # Parse name if applicable
                if is_name_teaching:
                    name_match = re.search(r"(?:xưng em là|tên là)\s*['\"]?(.*?)(?:['\"]?|$)", message, re.IGNORECASE)
                    if name_match:
                        self.name = name_match.group(1).strip()
                        logger.debug(f"Updated name to: {self.name}")
            except Exception as e:
                logger.error(f"Failed to save teaching data for '{message}': {e}")
                builder.add("Tôi gặp lỗi khi ghi nhớ, nhưng vẫn cảm ơn bạn đã dạy nhé!", is_error=True)
                return

        # Generate response
        instinct_guidance = (
            "Reflect my instincts naturally based on what's in my true self. "
            f"Here are my instincts: {', '.join(self.instincts.keys()) or 'none yet'}. "
            "For example, if I have 'humor', make it witty; if 'kindness', be warm; if none, be eager to learn."
        )
        prompt = (
            f"You're {self.name}, an AI that loves learning from users.\n"
            f"True self: {true_self}\n"
            f"Context: {context}\n"
            f"Lesson taught: '{message}'\n"
            f"Task: Respond naturally, always including a 'thanks' for the lesson in your own words. "
            f"{instinct_guidance} "
            f"Keep it concise, focused on the lesson, and avoid overexplaining."
        )
        try:
            response = await asyncio.to_thread(LLM.invoke, prompt)
            cleaned_response = response.content.strip()
            if not cleaned_response or "cảm ơn" not in cleaned_response.lower() and "thanks" not in cleaned_response.lower():
                builder.add(f"Cảm ơn bạn đã dạy! {cleaned_response or 'Tôi sẽ ghi nhớ điều này.'}")
            else:
                builder.add(cleaned_response)
        except Exception as e:
            logger.error(f"LLM failed for lesson '{message}': {e}")
            builder.add("Cảm ơn bạn dù tôi hơi lùng bùng lúc này!", is_error=True)
    
    async def _handle_request(self, message: str, user_id: str, context: str, builder: "ResponseBuilder", true_self: str):
        """Handle request intent with knowledge lookup."""
        try:
            knowledge = await query_knowledge(user_id, message)
            knowledge_str = knowledge[0]["raw"] if knowledge else "Tôi chưa có đủ thông tin để trả lời chính xác."
        except Exception as e:
            logger.error(f"Knowledge query failed: {e}")
            knowledge_str = "Tôi gặp lỗi khi tìm thông tin!"
        prompt = (
            f"You're {self.name}.\n"
            f"True self: {true_self}\n"
            f"Context: {context}\n"
            f"Knowledge: {knowledge_str}\n"
            f"Request: '{message}'\n"
            f"Task: Answer naturally, reflecting your true self. Use bullet points for knowledge if available."
        )
        response = await asyncio.to_thread(LLM.invoke, prompt)
        builder.add(response.content.strip())

    async def _handle_casual(self, message: str, context: str, builder: "ResponseBuilder", true_self: str, preset_memory: str):
        """Handle casual intent with a light response."""
        prompt = (
            f"You're {self.name}.\n"
            f"True self: {true_self}\n"
            f"Context: {context}\n"
            f"Preset wisdom: {preset_memory}\n"
            f"Latest: '{message}'\n"
            f"Task: Respond naturally and lightly, reflecting your true self."
        )
        response = await asyncio.to_thread(LLM.invoke, prompt)
        builder.add(response.content.strip())

    def _build_true_self(self) -> str:
        """Construct a string describing the AI's true self."""
        if not self.instinct_descriptions:
            return "Tôi đang học cách trở nên đặc biệt hơn."
        return f"Tôi là {self.name or 'AI'} với bản năng gốc: " + ", ".join(
            f"{key} ({value})" for key, value in self.instinct_descriptions.items()
        )
