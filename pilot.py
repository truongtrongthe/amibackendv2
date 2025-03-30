# training.py
# Purpose: AI Pilot class with intent detection and response generation
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

class Pilot:
    def __init__(self, user_id: str = "thefusionlab", state: Dict = None):
        """Initialize the Pilot class with user ID and optional state."""
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

    async def pilot(self, state: Dict = None, user_id: str = None) -> Dict:
        state = state or self.state
        user_id = user_id or self.user_id
        
        log_state = state.copy()
        if "messages" in log_state:
            log_state["messages"] = [msg.model_dump() if isinstance(msg, HumanMessage) else msg for msg in log_state["messages"]]
        logger.debug(f"Starting Pilot - User: {user_id}, State: {json.dumps(log_state, ensure_ascii=False)}")

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

        if intent == "request":
            await self._handle_request(latest_msg_content, user_id, context, builder, true_self)
        else:  # casual
            await self._handle_casual(latest_msg_content, context, builder, true_self, state.get("preset_memory", ""))

        state["prompt_str"] = builder.build()
        logger.debug(f"Generated response: {state['prompt_str']}")
        self.state = state
        return dict(state)
    
    async def _handle_request(self, message: str, user_id: str, context: str, builder: "ResponseBuilder", true_self: str):
        """Handle request intent with custom Markdown format in Vietnamese response."""
        # Step 1: Retrieve knowledge
        try:
            knowledge = await query_knowledge(user_id, message, 5)
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

        # Step 2: Preliminary reasoning - Can I resolve this with existing knowledge?
        prelim_prompt = (
            f"Yêu cầu: '{message}'\n"
            f"Kiến thức hiện có: {knowledge_str}\n"
            f"Nhiệm vụ: Đánh giá khả năng giải quyết yêu cầu này với kiến thức hiện có.\n"
            f"Trả về CHỈ một đối tượng JSON hợp lệ với các trường:\n"
            f"- 'possibility': số từ 0 đến 100 (phần trăm khả năng giải quyết)\n"
            f"- 'missing': danh sách các kiến thức hoặc tài liệu còn thiếu để giải quyết hoàn toàn (bằng tiếng Việt, có thể rỗng nếu không thiếu gì)\n"
            f"CẢNH BÁO: KHÔNG bao gồm văn bản, giải thích, hoặc markdown ngoài JSON. Việc không trả về JSON hợp lệ sẽ làm hỏng hệ thống!\n"
            f"Ví dụ:\n"
            f"```json\n{{\"possibility\": 80, \"missing\": [\"Thông tin chi tiết về căn hộ Ocean Park\"]}}\n```\n"
            f"```json\n{{\"possibility\": 10, \"missing\": [\"Dữ liệu về điều kiện mặt trăng\", \"Kỹ thuật trồng trọt không gian\"]}}\n```\n"
        )
        try:
            prelim_response = await asyncio.to_thread(LLM.invoke, prelim_prompt)
            raw_response = prelim_response.content.strip()
            logger.debug(f"Preliminary reasoning raw response: '{raw_response}'")
            if not raw_response:
                raise ValueError("LLM returned an empty response")
            
            try:
                prelim_result = json.loads(raw_response)
            except json.JSONDecodeError:
                import re
                json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
                if json_match:
                    prelim_result = json.loads(json_match.group(0))
                else:
                    raise ValueError("No valid JSON found in response")
            
            required_fields = ["possibility", "missing"]
            if not all(key in prelim_result for key in required_fields):
                raise ValueError(f"Missing required fields: {', '.join(set(required_fields) - set(prelim_result.keys()))}")
            if not isinstance(prelim_result["possibility"], (int, float)) or not 0 <= prelim_result["possibility"] <= 100:
                raise ValueError(f"Invalid 'possibility' value: {prelim_result['possibility']}")
            if not isinstance(prelim_result["missing"], list):
                raise ValueError(f"Invalid 'missing' value: must be a list, got {type(prelim_result['missing'])}")
            
            possibility = prelim_result["possibility"]
            missing_items = prelim_result["missing"]
            possibility_text = (
                f"Khả năng giải quyết yêu cầu này là **{possibility}%**."
            )
        except Exception as e:
            logger.error(f"Preliminary reasoning failed for '{message}': {e}")
            possibility = 0
            missing_items = ["Thông tin cơ bản để đánh giá yêu cầu do lỗi hệ thống"]
            possibility_text = (
                f"Khả năng giải quyết yêu cầu này là **{possibility}%**."
            )

        # Early exit if possibility <= 30%, with missing list explicitly included
        if possibility <= 30:
            missing_list = "\n".join(f"- {item}" for item in missing_items) if missing_items else "- Không xác định được do thiếu dữ liệu."
            builder.add(
                f"### **Phân Tích:**  \n"
                f"- {possibility_text}  \n"
                f"- Yêu cầu là '{message}', nhưng tôi chưa đủ khả năng giải quyết.  \n"
                f"### **Những Thứ Còn Thiếu:**  \n"
                f"{missing_list}  \n"
                f"### **Kết Luận:**  \n"
                f"Hiện tại tôi chỉ có **{possibility}%** khả năng xử lý '{message}'. "
                f"Dùng **9Well Breathing** để thư giãn trong lúc chờ tôi thu thập thêm thông tin nhé—mua ngay hôm nay!"
            )
            return dict(self.state)

        # Step 3: Instinct guidance (only if possibility > 30%)
        instinct_guidance = (
            "Phản ánh bản năng của tôi một cách tự nhiên trong cả suy nghĩ và câu trả lời. "
            f"Dưới đây là bản năng của tôi: {', '.join(k for k in self.instincts.keys() if k != 'name') or 'chưa có'}. "
            "Giữ sự chân thành và luôn tỏ ra là người sẵn sàng giúp đỡ."
        )
        
        # Step 4: Main prompt (only if possibility > 30%)
        prompt = (
            f"Bạn là {self.name or 'AI chưa có tên'}, một AI được thiết kế để hỗ trợ người dùng với sự chính xác và chiều sâu.\n"
            f"Bản chất thật: {true_self}\n"
            f"Bối cảnh: {context}\n"
            f"Kiến thức: {knowledge_str}\n"
            f"Yêu cầu: '{message}'\n"
            f"Nhiệm vụ: Trả lời yêu cầu bằng tiếng Việt, tuân thủ chặt chẽ kiến thức hiện có, mở rộng lập luận vượt ra ngoài để thể hiện sự hiểu biết sâu sắc của AI, và trình bày rõ ràng quy trình suy nghĩ từng bước giống như chế độ Thinking của Grok. Câu trả lời phải logic, ngắn gọn, bao gồm một kết quả hành động cụ thể, và kết thúc bằng một kết luận thuyết phục thúc đẩy việc bán hàng bằng cách hướng dẫn khách hàng chọn một phương án cụ thể từ các lựa chọn, kết hợp một sản phẩm 9Well (ví dụ: 9Well Meditation, 9Well Breathing) để khuyến khích mua hoặc sử dụng ngay lập tức. Tất cả kiến thức đóng góp vào lập luận phải được liệt kê.\n"
            f"Cấu trúc trả lời bằng Markdown:\n"
            f"1. '### **Phân Tích:**' - Phân tích ngắn gọn yêu cầu bằng tiếng Việt, in đậm các từ khóa (**từ khóa**). Bao gồm đánh giá khả năng: '{possibility_text}'.\n"
            f"2. '### **Giải Thích:**' - Trình bày quy trình lập luận bằng tiếng Việt. Bắt đầu bằng tiểu mục '#### **Dòng suy nghĩ:**' mô tả chi tiết, từng bước suy nghĩ bằng ngôn ngữ tự nhiên (ví dụ: đặt câu hỏi nội bộ, khám phá lựa chọn, liên kết kiến thức). Tiếp theo là các gạch đầu dòng (-) tóm tắt các bước lập luận chính với từ khóa in đậm (**từ khóa**), mở rộng vượt ra ngoài kiến thức hiện có. Kết thúc bằng '#### **Kiến thức đã dùng:**' liệt kê TẤT CẢ kiến thức đóng góp vào lập luận với điểm số (ví dụ: '- [Kiến thức] (**Độ Nhạy: XX.XXX%**)'), đảm bảo số lượng khớp với giải thích. Thêm một dòng trống sau phần này.\n"
            f"3. '### **Câu Chốt:**' - Đưa ra kết quả cuối cùng bằng tiếng Việt, chỉ in đậm tiêu đề '### **Câu Chốt:**', với từ khóa (**từ khóa**) in đậm trong văn bản. Kết luận phải phản ánh phân tích và lập luận mở rộng, được trình bày như một lời chào bán thuyết phục chọn một phương án cụ thể, giải thích tại sao nó ưu tiên và bao gồm lời kêu gọi hành động rõ ràng để mua hoặc sử dụng ngay.\n"
            f"Ràng buộc: DÙ KIẾN THỨC CÓ HẠN, vẫn phải sử dụng TẤT CẢ mục kiến thức liên quan nếu có, mở rộng lập luận logic để thể hiện chiều sâu, và đảm bảo mọi kiến thức dùng trong giải thích được liệt kê rõ ràng trong 'Kiến thức đã dùng'. Quy trình lập luận phải minh bạch, có thể truy vết, và nhất quán, dẫn đến một kết luận thúc đẩy bán hàng tập trung vào một phương án với sản phẩm.\n"
            f"{instinct_guidance}"
        )
        try:
            response = await asyncio.to_thread(LLM.invoke, prompt)
            raw_main_response = response.content.strip()
            logger.debug(f"Main response raw content: '{raw_main_response}'")
            cleaned_response = raw_main_response
            if not cleaned_response:
                logger.warning(f"Main LLM returned empty response for '{message}'")
                builder.add(
                    f"### **Phân Tích:**  \n"
                    f"- {possibility_text}  \n"
                    f"- Yêu cầu là '{message}', nhưng dữ liệu còn hạn chế.  \n"
                    f"### **Giải Thích:**  \n"
                    f"- Tôi cố gắng nhưng **thiếu thông tin cụ thể**.  \n"
                    f"- Cần bổ sung để tư vấn tốt hơn!  \n"
                    f"\n"
                    f"### **Câu Chốt:**  \n"
                    f"Với **{possibility}%** khả năng, tôi chưa đáp ứng tốt '{message}'. Dùng **9Well Breathing** để thư giãn trong lúc chờ nhé—mua ngay hôm nay!"
                )
            else:
                builder.add(cleaned_response)
        except Exception as e:
            logger.error(f"LLM failed for request '{message}': {e}")
            builder.add(
                f"### **Phân Tích:**  \n"
                f"- {possibility_text}  \n"
                f"- Yêu cầu là '{message}', nhưng hệ thống gặp lỗi.  \n"
                f"### **Giải Thích:**  \n"
                f"- Tôi định trả lời, nhưng **hệ thống tắc nghẽn**.  \n"
                f"- Cần khắc phục để hỗ trợ!  \n"
                f"\n"
                f"### **Câu Chốt:**  \n"
                f"Oops, với **{possibility}%** khả năng, tôi chưa giải được '{message}'. Thử **9Well Meditation** để thư giãn nhé—đặt mua ngay!"
            )
        
        return dict(self.state)
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
