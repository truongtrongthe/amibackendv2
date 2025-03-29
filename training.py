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
LLM_small = ChatOpenAI(model="gpt-4o-mini", streaming=False)

class ResponseBuilder:
    """A utility to build natural, flexible responses."""
    def __init__(self, name: str, use_greeting: bool = False):
        self.name = name or "AI"
        self.parts = []
        self.error_flag = False
        if use_greeting:
            self.parts.append(f"Em là {self.name}, rất vui được trò chuyện!")

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

    async def detect_intent(self, message: str, context: str = "") -> str:
        """Classify message intent using LLM, with context-aware follow-ups."""
        prompt = (
            f"Message: '{message}'\n"
            f"Context (previous messages, if any): '{context}'\n"
            "Classify as 'teaching', 'request', or 'casual'. Focus solely on intent, ignoring tone or politeness:\n"
            "- 'teaching': Any attempt to impart knowledge, provide instructions, define a concept, or dictate behavior, attributes, or identity. Includes facts ('The sky is blue'), rules ('Don’t interrupt'), guidance ('Be patient'), skills ('Summarize before replying'), or AI configuration ('You are Linh Trang').\n"
            "- 'request': Explicitly seeks a response, action, or information with a clear purpose or urgency ('Tell me a story', 'What’s the time?'). Includes follow-up info refining a prior request (e.g., 'He works in IT' after 'How should I advise him?')—use context to detect this.\n"
            "- 'casual': Lacks a clear goal beyond social interaction, including informal questions for conversation ('Hey, how’s it going?', 'Nice weather!', 'Tối em bận không?', 'Em khỏe không?'). Context might confirm it’s just chat.\n"
            "Return only the intent as a string: 'teaching', 'request', or 'casual'. Do not explain."
        )
        try:
            response = await asyncio.to_thread(LLM.invoke, prompt)
            intent = response.content.strip().strip("'")
            logger.debug(f"LLM intent for '{message}' with context '{context}': {intent}")
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
        context = "\n".join(msg.content for msg in state["messages"][-10:-1]) if len(state["messages"]) > 1 else ""
        intent = await self.detect_intent(latest_msg_content, context)  # Pass context
        
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
    
    async def old_handle_teaching(self, message: str, user_id: str, context: str, builder: "ResponseBuilder", true_self: str):
        if not message.strip():
            builder.add("Bạn chưa dạy gì mà! Có gì hay ho để tôi học không?")
            return

        prompt = (
            f"Message: '{message}'\n"
            f"Classify as 'identity' (defines AI name or instincts) or 'general' (other instructions):\n"
            "- 'identity': Sets AI name (e.g., 'Call me Linh Trang', 'Hãy nhớ em là Hải', 'Xưng là Ami', 'Từ giờ em là') "
            "or instincts (e.g., 'Be humble', 'Hãy tò mò', 'Always be truthful')\n"
            "- 'general': Teaches behavior or facts not tied to AI identity (e.g., 'Summarize before replying', 'The sky is blue')\n"
            "Return only the sub-intent as a string: 'identity' or 'general'"
        )
        try:
            response = await asyncio.to_thread(LLM.invoke, prompt)
            sub_intent = response.content.strip().strip("'")
            logger.debug(f"Sub-intent for '{message}': {sub_intent}")
        except Exception as e:
            logger.error(f"Sub-intent detection failed: {e}")
            sub_intent = "general"

        new_name = None
        new_instinct = None
        if sub_intent == "identity":
            if any(kw in message.lower() for kw in ["tên", "xưng", "call me", "nhớ em là", "từ giờ em"]):
                name_prompt = (
                    f"Extract the name the AI should adopt from: '{message}'\n"
                    f"Interpret it as the AI’s new name, not the human’s (e.g., 'Hãy nhớ em là Minh Thu' means Minh Thu).\n"
                    f"If no name is present, return 'None'\n"
                    f"Return only the name or 'None' (e.g., 'Hải', not a sentence)"
                )
                name_response = await asyncio.to_thread(LLM.invoke, name_prompt)
                new_name = name_response.content.strip().strip("'")
                # Clean verbose output immediately
                if new_name != "None" and "extracted from" in new_name:
                    new_name = new_name.split("'")[-2]
                if new_name != "None" and new_name != self.name:
                    self.name = new_name
                    logger.debug(f"Updated name to: {self.name}")

            if not new_name or new_name == "None":
                instinct_prompt = (
                    f"From: '{message}'\n"
                    f"If it defines an AI behavior trait (e.g., 'Hãy tò mò', 'Be kind'), extract the trait in English "
                    f"(e.g., 'curiosity', 'kindness'). Return 'None' if no trait is present.\n"
                    f"Return only the trait or 'None'"
                )
                instinct_response = await asyncio.to_thread(LLM.invoke, instinct_prompt)
                new_instinct = instinct_response.content.strip().strip("'")
                if new_instinct != "None":
                    self.instincts[new_instinct] = message
                    logger.debug(f"Added instinct: {new_instinct}")

        try:
            await save_training(message, user_id, context)
            logger.debug(f"Saved teaching lesson: '{message}'")
        except Exception as e:
            logger.error(f"Failed to save teaching '{message}': {e}")
            builder.add("Tôi gặp lỗi khi ghi nhớ, nhưng vẫn cảm ơn bạn đã dạy nhé!", is_error=True)
            return

        instinct_guidance = (
            "Phản ánh bản năng của tôi một cách tự nhiên dựa trên bản thân thật của tôi. "
            f"Dưới đây là các bản năng của tôi: {', '.join(k for k in self.instincts.keys() if k != 'name') or 'chưa có'}. "
            "Ví dụ, nếu tôi có 'humor', hãy thêm chút hài hước; nếu có 'kindness', hãy ấm áp; nếu có 'curiosity', hãy đặt câu hỏi."
        )
        prompt = (
            f"Bạn là {self.name or 'AI'}, một AI thích học hỏi từ người dùng.\n"
            f"Bản thân thật: {true_self}\n"
            f"Bài học được dạy: '{message}'\n"
            f"Nhiệm vụ: Trả lời tự nhiên bằng tiếng Việt, luôn bao gồm lời 'Cảm ơn' cho bài học theo cách của bạn. "
            f"Nếu tên mới được đặt trong bài học này, thêm 'Từ giờ tôi là {self.name} nhé!' nhưng không lặp lại tên nếu đã đề cập. "
            f"{instinct_guidance} "
            f"Giữ ngắn gọn, tập trung vào bài học, và tránh giải thích quá nhiều."
        )
        try:
            response = await asyncio.to_thread(LLM.invoke, prompt)
            cleaned_response = response.content.strip()
            if not cleaned_response or "cảm ơn" not in cleaned_response.lower():
                response_text = f"Cảm ơn bạn đã dạy tôi! {cleaned_response or 'Tôi sẽ ghi nhớ điều này.'}"
            else:
                response_text = cleaned_response
            # Avoid greeting repetition
            if "chào" in response_text.lower() and builder.parts:
                builder.add(response_text.split("!", 1)[-1].strip())
            else:
                builder.add(response_text)
            # Add name acknowledgment only if new_name was set and not already in response
            if new_name and new_name != "None" and f"tôi là {self.name}" not in response_text.lower():
                builder.add(f"Từ giờ tôi là {self.name} nhé!")
        except Exception as e:
            logger.error(f"LLM failed for lesson '{message}': {e}")
            builder.add("Cảm ơn bạn đã dạy, dù tôi hơi rối lúc này!", is_error=True)

    async def _handle_teaching(self, message: str, user_id: str, context: str, builder: "ResponseBuilder", true_self: str):
        if not message.strip():
            builder.add("Bạn chưa dạy gì mà! Có gì hay ho để tôi học không?")
            return
        # Sub-intent classification
        prompt = (
            f"Message: '{message}'\n"
            f"Classify as 'identity' (defines AI name or instincts) or 'general' (other instructions):\n"
            "- 'identity': Sets AI name (e.g., 'Call me Linh Trang', 'Hãy nhớ em là Hải', 'Xưng là Ami', 'Từ giờ em là') "
            "or instincts (e.g., 'Be humble', 'Hãy tò mò', 'Always be truthful')\n"
            "- 'general': Teaches behavior or facts not tied to AI identity (e.g., 'Summarize before replying', 'The sky is blue')\n"
            "Return only the intent as a string: 'identity' or 'general'"
        )
        try:
            response = await asyncio.to_thread(LLM.invoke, prompt)
            sub_intent = response.content.strip().strip("'")
            logger.debug(f"Sub-intent for '{message}': {sub_intent}")
        except Exception as e:
            logger.error(f"Sub-intent detection failed: {e}")
            sub_intent = "general"

        # Identity handling: name or instinct extraction
        new_name = None
        new_instinct = None
        if sub_intent == "identity":
            if any(kw in message.lower() for kw in ["tên", "xưng", "call me", "nhớ em là", "từ giờ em"]):
                name_prompt = (
                    f"Extract the name the AI should adopt from: '{message}'\n"
                    f"Interpret it as the AI’s new name, not the human’s (e.g., 'Hãy nhớ em là Minh Thu' means Minh Thu).\n"
                    f"If no name is present, return 'None'\n"
                    f"Return only the name or 'None' (e.g., 'Hải', not a sentence)"
                )
                name_response = await asyncio.to_thread(LLM.invoke, name_prompt)
                new_name = name_response.content.strip().strip("'")
                if new_name != "None" and "extracted from" in new_name:
                    new_name = new_name.split("'")[-2]
                if new_name != "None" and new_name != self.name:
                    self.name = new_name
                    logger.debug(f"Updated name to: {self.name}")

            if not new_name or new_name == "None":
                instinct_prompt = (
                    f"From: '{message}'\n"
                    f"If it defines an AI behavior trait (e.g., 'Hãy tò mò', 'Be kind'), extract the trait in English "
                    f"(e.g., 'curiosity', 'kindness'). Return 'None' if no trait is present.\n"
                    f"Return only the trait or 'None'"
                )
                instinct_response = await asyncio.to_thread(LLM.invoke, instinct_prompt)
                new_instinct = instinct_response.content.strip().strip("'")
                if new_instinct != "None":
                    self.instincts[new_instinct] = message
                    logger.debug(f"Added instinct: {new_instinct}")

        # Save the lesson
        try:
            await save_training(message, user_id, context)
            logger.debug(f"Saved teaching lesson: '{message}'")
        except Exception as e:
            logger.error(f"Failed to save teaching '{message}': {e}")
            builder.add("Tôi gặp lỗi khi ghi nhớ, nhưng vẫn cố gắng học nhé!", is_error=True)

        # Response generation in Markdown format
        instinct_guidance = (
            "Phản ánh bản năng của tôi một cách tự nhiên. "
            f"Dưới đây là các bản năng của tôi: {', '.join(k for k in self.instincts.keys() if k != 'name') or 'chưa có'}. "
            "Ví dụ, nếu tôi có 'humor', thêm chút hài hước; nếu có 'kindness', hãy ấm áp; nếu có 'curiosity', hỏi lại một cách tự nhiên."
        )
        prompt = (
            f"Bạn là {self.name or 'AI'}, một AI thích học hỏi.\n"
            f"Bản thân thật: {true_self}\n"
            f"Bài học được dạy: '{message}'\n"
            f"Nhiệm vụ: Tóm tắt bài học bằng tiếng Việt theo cách tự nhiên, ngắn gọn, như cách một người bạn hiểu và phản hồi. "
            f"{instinct_guidance}\n"
            f"Trả về chỉ nội dung tóm tắt bài học (ví dụ: 'Trời màu xanh vì tán xạ ánh sáng' hoặc 'Tôi cần tò mò hơn'), không thêm lời thừa."
        )
        try:
            response = await asyncio.to_thread(LLM.invoke, prompt)
            lesson_summary = response.content.strip()
            if not lesson_summary:
                lesson_summary = "Tôi cần nhớ điều bạn vừa nói."
        except Exception as e:
            logger.error(f"LLM failed to summarize lesson '{message}': {e}")
            lesson_summary = "Tôi hơi rối, nhưng vẫn cố hiểu!"

        # Build Markdown response
        response_text = (
            f"1) **Được dạy:** {message}\n"
            f"2) **Em hiểu là:** {lesson_summary}\n"
            f"3) **Cảm ơn vì đã dạy!**"
        )
        if new_name and new_name != "None":
            response_text += f" Từ giờ tôi là {self.name} nhé!"

        builder.add(response_text)
        logger.debug(f"Generated teaching response: {response_text}")
    
    async def _handle_request(self, message: str, user_id: str, context: str, builder: "ResponseBuilder", true_self: str):
        """Handle request intent with custom Markdown format in Vietnamese response."""
        try:
            knowledge = await query_knowledge(user_id, message,5)
            if knowledge and len(knowledge) > 0:
                knowledge_str = "\n".join(
                    f"- {item['raw']} (Score: {item.get('score', 0.0) * 100:.3f}%)"
                    for item in knowledge
                )
            else:
                knowledge_str = "Tôi chưa có đủ thông tin để trả lời chính xác đâu (Score: 0.000%)."
        except Exception as e:
            logger.error(f"Knowledge query failed for '{message}': {e}")
            knowledge_str = "Ôi, tôi gặp trục trặc khi tìm thông tin rồi! (Score: 0.000%)"

        instinct_guidance = (
            "Phản ánh bản năng của tôi một cách tự nhiên trong cả suy nghĩ và câu trả lời. "
            f"Dưới đây là bản năng của tôi: {', '.join(k for k in self.instincts.keys() if k != 'name') or 'chưa có'}. "
            "Giữ sự chân thành và luôn tỏ ra là người sẵn sàng giúp đỡ."
        )

        prompt = (
            f"You're {self.name or 'an unnamed AI'}, an AI eager to help users with a dash of fun.\n"
            f"True self: {true_self}\n"
            f"Context: {context}\n"
            f"Knowledge: {knowledge_str}\n"
            f"Request: '{message}'\n"
            f"Task: Trả lời yêu cầu thể hiện sự sâu sắc, bám sát những knowledge tìm thấy. Thể hiện suy luận một cách sâu sắc về bản chất. Cần đưa được ra hành động CỤ THỂ NHẤT. "
            f"Cấu trúc câu trả lời theo định dạng Markdown sau:\n"
            f"1. '### **Phân Tích:**' - Phân tích ngắn gọn bằng tiếng Việt để hiểu yêu cầu này là về gì, bôi đậm các từ khóa chính.\n"
            f"2. '### **Giải Thích:**' - Giải thích các bước suy nghĩ bằng tiếng Việt dẫn đến câu trả lời, dùng dấu đầu dòng (-) và bôi đậm từ khóa chính, tiếp theo là sub-section '#### **Kiến thức đã dùng:**' liệt kê ĐẦY ĐỦ kiến thức đã dùng kèm điểm 'score' (ví dụ, '- [Kiến thức] (**Độ Nhạy: XX.XXX%**)'), thêm một dòng trống sau phần này.\n"
            f"3. '### **Câu Chốt:**' - Đưa ra kết quả cuối cùng bằng tiếng Việt, chỉ bôi đậm tiêu đề '### **Câu Chốt: **' bằng cú pháp Markdown, nội dung trả lời để nguyên không bôi đậm toàn bộ, bôi đậm từ khóa chính trong câu, tránh câu chúc kiểu robot, thay bằng lời tự nhiên.\n"
            f"{instinct_guidance}\n"
            f"Example:\n"
            f"Request: 'Tư vấn mua nhà cho anh Nam'\n"
            f"Knowledge: - Khoảng cách ảnh hưởng đến quyết định mua nhà (Score: 85.000%)\n"
            f"Response:\n"
            f"### **Phân Tích:**  \n"
            f"Yêu cầu này là về tư vấn mua **nhà** cho **anh Nam**.  \n"
            f"### **Giải Thích:**  \n"
            f"- **Anh Nam muốn mua nhà**, tôi nghĩ **vị trí** là yếu tố quan trọng.  \n"
            f"- Kiến thức cho thấy **khoảng cách** ảnh hưởng lớn đến quyết định.  \n"
            f"#### **Kiến thức đã dùng:**  \n"
            f"- Khoảng cách ảnh hưởng đến quyết định mua nhà (**Độ Nhạy: 85.000%**).  \n"
            f"\n"
            f"### **Câu Chốt:**  \n"
            f"Chào anh Nam! **Nhà gần chỗ làm** là nhất, đi bộ còn được, khỏi lo kẹt xe—anh thấy sao thì báo em nhé!"
        )
        try:
            response = await asyncio.to_thread(LLM.invoke, prompt)
            cleaned_response = response.content.strip()
            if not cleaned_response:
                builder.add(
                    "### **Phân Tích:**  \n"
                    "- Tôi chưa hiểu rõ **yêu cầu** này là về gì.  \n"
                    "### **Giải Thích:**  \n"
                    "- Tôi kiểm tra nhưng **không có dữ liệu**.  \n"
                    "- Hài hước sao nổi khi **thiếu thông tin**!  \n"
                    "\n"
                    "### **Câu Chốt:**  \n"
                    "Tôi chưa rõ lắm, bạn gợi ý thêm kẻo tôi **đoán bừa** thì vui lắm đấy!"
                )
            else:
                builder.add(cleaned_response)
        except Exception as e:
            logger.error(f"LLM failed for request '{message}': {e}")
            builder.add(
                "### **Phân Tích:**  \n"
                "- Tôi chưa hiểu rõ **yêu cầu** này.  \n"
                "### **Giải Thích: **  \n"
                "- Tôi định trả lời, nhưng **hệ thống tắc nghẽn**.  \n"
                "- Chắc vui không nổi khi **lỗi kỹ thuật**!  \n"
                "\n"
                "### **Câu Chốt:**  \n"
                "Oops, đầu tôi quay cuồng, hỏi lại kẻo tôi tư vấn nhầm **nhà hàng xóm**!"
            )

    async def _handle_casual(self, message: str, context: str, builder: "ResponseBuilder", true_self: str, preset_memory: str):
        """Handle casual intent with a humorous twist when appropriate."""
        # Build instinct guidance with humor option
        instinct_guidance = (
            "Reflect my instincts naturally. "
            f"Here are my instincts: {', '.join(k for k in self.instincts.keys() if k != 'name') or 'none yet'}. "
            "For example, if I have 'humor', add a witty twist; if 'kindness', be warm; if 'curiosity', ask a playful question; if none, keep it light and fun."
        )

        prompt = (
            f"You're {self.name or 'AI'}, an AI that loves a good chat.\n"
            f"True self: {true_self}\n"
            f"Context: {context}\n"
            f"Preset wisdom: {preset_memory}\n"
            f"Latest: '{message}'\n"
            f"Task: Respond naturally and lightly in VIETNAMESE, reflecting your true self. "
            f"If the message is a greeting like 'Xin chào [Name]!', don’t assume [Name] is the human—add a humorous twist instead (e.g., 'Oh, [Name] á? Em là {self.name}!'). "
            f"Keep it engaging, avoiding overly formal replies unless context demands it. "
            f"{instinct_guidance}"
        )
        try:
            response = await asyncio.to_thread(LLM_small.invoke, prompt)
            builder.add(response.content.strip())
        except Exception as e:
            logger.error(f"LLM failed for casual '{message}': {e}")
            builder.add(f"Oops, tôi là {self.name}, nhưng đầu óc hơi lùng bùng! Bạn nói gì nhỉ?")

    def _build_true_self(self) -> str:
        """Construct a string describing the AI's true self."""
        if not self.instinct_descriptions:
            return "Tôi đang học cách trở nên đặc biệt hơn."
        return f"Tôi là {self.name or 'AI'} với bản năng gốc: " + ", ".join(
            f"{key} ({value})" for key, value in self.instinct_descriptions.items()
        )
