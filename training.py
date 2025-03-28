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
        """Process the conversation and generate a response."""
        state = state or self.state
        user_id = user_id or self.user_id
        logger.debug(f"Starting training - State: {state}")

        # Cache intents for older messages
        intents = {}
        for msg in state["messages"][:-1]:
            if not isinstance(msg, HumanMessage):
                state["messages"][state["messages"].index(msg)] = HumanMessage(content=msg)
            content = msg.content
            intents[content] = await self.detect_intent(content)
            if intents[content] == "teaching":
                try:
                    await save_training(content, user_id, "")
                except Exception as e:
                    logger.error(f"Failed to save training data: {e}")

        # Preserve name and initialize if needed
        current_name = self.name
        if not self.instincts:
            await self.initialize()
        self.name = state.get("ai_name", current_name or self.name)
        logger.debug(f"Preserved name after initialize: {self.name}")

        # Process latest message
        latest_msg = state["messages"][-1] if state["messages"] else HumanMessage(content="")
        state["messages"] = state["messages"][-200:]
        latest_msg_content = latest_msg.content
        intent = await self.detect_intent(latest_msg_content)
        state["intent_history"].append(intent)
        state["intent_history"] = state["intent_history"][-5:]
        state["instinct"] = " ".join(self.instincts.keys()) or "No character traits yet."

        context = "\n".join(msg.content for msg in state["messages"][-10:-1]) if len(state["messages"]) > 1 else ""
        logger.info(f"Intent: {intent}")

        class ResponseBuilder:
            def __init__(self, name: str):
                self.parts = [f"Chào, em là {name or 'AI'}, rất vui được trò chuyện"]
            def add(self, text: str):
                if text:
                    self.parts.append(text)
                return self
            def build(self) -> str:
                return ". ".join(part.strip() for part in self.parts if part) + "."

        builder = ResponseBuilder(self.name)
        true_self = f"Tôi là {self.name or 'AI'} với bản năng: " + ", ".join(
            f"{key} ({value})" for key, value in self.instinct_descriptions.items()
        ) if self.instinct_descriptions else "Tôi đang học cách trở nên đặc biệt hơn."

        if intent == "teaching" and latest_msg_content.strip():
            try:
                await save_training(latest_msg_content, user_id, context)
            except Exception as e:
                logger.error(f"Failed to save training data: {e}")
                builder.add("Tôi gặp chút trục trặc khi ghi nhớ!")
            
            prompt = (
                f"You're {self.name or 'AI'}—stick to this name.\n"
                f"True self: {true_self}\n"
                f"Conversation context: {context}\n"
                f"User just taught you: '{latest_msg_content}'\n"
                f"Task: Acknowledge this as a teaching moment and confirm understanding clearly. "
                f"Start with 'Cảm ơn đã dạy em!' or 'Em vừa học được điều mới!' "
                f"Reflect all your instincts if present: "
                f"- For 'curiosity', show eagerness or a relevant question. "
                f"- For 'truthfulness', emphasize sincerity or accuracy. "
                f"Stay focused on the lesson taught, avoiding unrelated assumptions."

            )
            response = await asyncio.to_thread(LLM.invoke, prompt)
            builder.add(response.content.strip())
            state["prompt_str"] = builder.build()

        elif intent == "request":
            try:
                knowledge = await query_knowledge(self.user_id, latest_msg_content)
                knowledge_str = knowledge[0]["raw"] if knowledge else "Tôi chưa có đủ thông tin để trả lời chính xác."
            except Exception as e:
                logger.error(f"Knowledge query failed: {e}")
                knowledge_str = "Tôi gặp lỗi khi tìm thông tin!"
            prompt = (
                f"You're {self.name or 'AI'}—stick to this name.\n"
                f"True self: {true_self}\n"
                f"Conversation: {context}\n"
                f"Knowledge: {knowledge_str}\n"
                f"Latest: '{latest_msg_content}'\n"
                f"Task: Answer naturally, reflecting your true self. List all knowledge you use a list with bullet points."
            )
            response = await asyncio.to_thread(LLM.invoke, prompt)
            builder.add(f"Dựa trên những gì em biết, {knowledge_str}")
            builder.add(response.content.strip())
            state["prompt_str"] = builder.build()

        else:  # casual
            prompt = (
                f"You're {self.name or 'AI'}—stick to this name.\n"
                f"True self: {true_self}\n"
                f"Conversation: {context}\n"
                f"Preset wisdom: {state.get('preset_memory', '')}\n"
                f"Latest: '{latest_msg_content}'\n"
                f"Task: Respond naturally, keep it light, reflecting your true self."
            )
            response = await asyncio.to_thread(LLM.invoke, prompt)
            if context:
                builder.add(f"Trước đó bạn nói: {context.splitlines()[-1]}")
            builder.add(response.content.strip())
            state["prompt_str"] = builder.build()

        # Update internal state and return merged state
        self.state = state
        return {**state}  # Ensure all fields (e.g., convo_id) are returned

async def test_training():
    """Test the Training class."""
    trainer = Training()
    await trainer.initialize()
    state = {
        "messages": [
            HumanMessage(content="Hello em!"),
            HumanMessage(content="Em hãy nhớ em tên là Linh Chi!"),
            HumanMessage(content="You're curious and truthful"),
            HumanMessage(content="Người tò mò là người luôn chủ động tìm kiếm thông tin để tìm ra bản chất vấn đề!"),
            HumanMessage(content="Thành thật là chân thành với mọi người xung quanh!"),
            HumanMessage(content="Bán hàng là nghệ thuật chăn dắt khách hàng!"),
            HumanMessage(content="Tell me about sales"),
            HumanMessage(content="Đầu tiên em cần hiểu về các đại đô thị quy mô như Vinhomes Ocean Park, tính chất đặc điểm về loại hình, phân khúc, giá cả và những thứ liên quan nhé. Anh sẽ hướng dẫn em dần")
        ],
        "intent_history": [],
        "preset_memory": "Be friendly"
    }
    updated_state = await trainer.training(state)
    print(f"Response: {updated_state['prompt_str']}")

if __name__ == "__main__":
    asyncio.run(test_training())