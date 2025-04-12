import asyncio
from typing import List, Dict
from langchain_openai import ChatOpenAI
from utilities import logger
from langchain_core.messages import HumanMessage,AIMessage
import json
from sentence_transformers import SentenceTransformer, util
import numpy as np
from database import query_graph_knowledge
from brainlog import create_brain_log
import re
from typing import Tuple, Dict, Any

def add_messages(existing_messages, new_messages):
    return existing_messages + new_messages

LLM = ChatOpenAI(model="gpt-4o", streaming=False)
StreamLLM = ChatOpenAI(model="gpt-4o", streaming=True)

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
            "graph_version_id": "",
        }
    async def initialize(self):
        if not self.instincts:
            self.instincts = {"friendly": "Be nice"}
    
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
    async def trigger(self, state: Dict = None, user_id: str = None, graph_version_id: str = None, config: Dict = None):
        state = state or self.state.copy()
        user_id = user_id or self.user_id
        config = config or {}
        graph_version_id = (
            graph_version_id if graph_version_id is not None 
            else state.get("graph_version_id", 
                config.get("configurable", {}).get("graph_version_id", 
                    self.graph_version_id if hasattr(self, 'graph_version_id') else ""))
        )
        if not graph_version_id:
            logger.warning(f"graph_version_id is empty! State: {state}, Config: {config}")
        
        # Sync with mc.state
        self.state["graph_version_id"] = graph_version_id

        latest_msg = state["messages"][-1] if state["messages"] else HumanMessage(content="")
        latest_msg_content = latest_msg.content.strip() if isinstance(latest_msg, HumanMessage) else latest_msg.strip()
        context = "\n".join(f"User: {msg.content}" if isinstance(msg, HumanMessage) else f"AI: {msg.content}" for msg in state["messages"][-100:])

        logger.info(f"Triggering - User: {user_id}, Graph Version: {graph_version_id}, latest_msg: {latest_msg_content}, context: {context}")

        builder = ResponseBuilder()
        
        async for response_chunk in self._handle_request(latest_msg_content, user_id, context, builder, state, graph_version_id=graph_version_id):
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
    
    async def detect_language_with_llm(self, text, llm=LLM):
        """Use LLM to detect language and provide appropriate response guidance"""
        # For very short inputs, give LLM more context
        if len(text.strip()) < 10:
            context_prompt = (
                f"This is a very short text: '{text}'\n"
                f"Based on this limited sample, identify the most likely language.\n"
                f"Consider common greetings, questions, or expressions that might indicate the language.\n"
                f"Return your answer in this JSON format:\n"
                f"{{\n"
                f"  \"language\": \"[language name in English]\",\n"
                f"  \"code\": \"[ISO 639-1 two-letter code]\",\n"
                f"  \"confidence\": [0-1 value],\n"
                f"  \"responseGuidance\": \"[Brief guidance on responding appropriately in this language]\"\n"
                f"}}"
            )
        else:
            context_prompt = (
                f"Identify the language of this text: '{text}'\n"
                f"Analyze the text carefully, considering vocabulary, grammar, script, and cultural markers.\n"
                f"Return your answer in this JSON format:\n"
                f"{{\n"
                f"  \"language\": \"[language name in English]\",\n"
                f"  \"code\": \"[ISO 639-1 two-letter code]\",\n"
                f"  \"confidence\": [0-1 value],\n"
                f"  \"responseGuidance\": \"[Brief guidance on responding appropriately in this language]\"\n"
                f"}}"
            )
        
        try:
            response = await llm.ainvoke(context_prompt) if asyncio.iscoroutinefunction(llm.invoke) else llm.invoke(context_prompt)
            response_text = getattr(response, 'content', response).strip()
            
            # Extract JSON from response (handling cases where LLM adds extra text)
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                json_str = json_match.group(0)
                lang_data = json.loads(json_str)
                
                # Validate required fields
                if all(k in lang_data for k in ["language", "code", "confidence", "responseGuidance"]):
                    return lang_data
                
            # If we get here, something went wrong with the JSON
            logger.warning(f"Language detection returned invalid format: {response_text[:100]}...")
            return {
                "language": "English",
                "code": "en",
                "confidence": 0.5,
                "responseGuidance": "Respond in a neutral, professional tone"
            }
            
        except Exception as e:
            logger.error(f"Language detection error: {str(e)}")
            # Fallback to English on any error
            return {
                "language": "English",
                "code": "en",
                "confidence": 0.5,
                "responseGuidance": "Respond in a neutral, professional tone"
            }

    async def _handle_request(self, message: str, user_id: str, context: str, builder: "ResponseBuilder", state: Dict, graph_version_id: str = ""):
        
        try:
            # Query knowledge with full context for broader product relevance
            knowledge = await query_graph_knowledge(graph_version_id, context, top_k=5) or []
            kwcontext = "\n\n".join(entry["raw"] for entry in knowledge)
        except Exception as e:
            logger.error(f"Knowledge query failed: {e}")
            builder.add("Oops, có lỗi khi tìm thông tin, thử lại nhé!")
            yield builder.build()
            return
        
        # Detect language of user message
        lang_info = await self.detect_language_with_llm(message)
        logger.info(f"Language detected: {lang_info['language']} (confidence: {lang_info['confidence']})")
        
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
            f"Language Information:\n"
            f"  - Detected: {lang_info['language']} ({lang_info['code']})\n"
            f"  - Confidence: {lang_info['confidence']}\n"
            f"  - Guidance: {lang_info['responseGuidance']}\n\n"
            f"Task: Reply in {lang_info['language']} following these guidelines:\n"
            f"  - Use the detected language with proper cultural norms and etiquette\n"
            f"  - If confidence is below 0.7, include a brief apology in English at the start\n"
            f"  - Keep answer concise (2-3 sentences) unless the sales process requires more\n"
            f"  - Apply the specific guidance: {lang_info['responseGuidance']}\n\n"
            
            f"1. Analyze the conversation context to identify where the customer is in the sales process. Prioritize recent messages but consider overall patterns.\n\n"
            
            f"2. Use the Customer Profile's next-step action as your primary guide for moving the conversation forward. Focus on:\n"
            f"   a) Current stage in the buying journey (awareness, consideration, decision)\n"
            f"   b) Key barrier preventing progress\n"
            f"   c) Next action to move them forward\n\n"
            
            f"3. Incorporate knowledge from Rules in this priority order:\n"
            f"   a) Process instructions: Follow specific conversation flows exactly as specified\n"
            f"   b) Product information: Reference features/pricing ONLY when directly relevant\n" 
            f"   c) Conversational techniques that match the customer's current stage\n\n"
            
            f"4. If the customer shows hesitation or objections, address them with empathy using specific language from Rules if available.\n\n"
            
            f"5. When explicitly directed by the Profile's next-step, include testimonials or product offers using exact phrasing from Rules.\n\n"
            
            f"6. Keep the tone conversational while respecting cultural communication norms.\n\n"
            
            f"7. If you absolutely cannot respond in the detected language, respond in English but acknowledge the language barrier politely."
        )
        
        async for _ in self.stream_response(base_prompt, builder):
            yield builder.build()
            
        
    async def _handle_casual(self, message: str, context: str, builder: "ResponseBuilder", state: Dict, graph_version_id: str = ""):
        logger.info(f"Handling CASUAL. Graph_version_id: {graph_version_id}")
        
        # Fetch knowledge with fallback to empty list
        knowledge = await query_graph_knowledge(graph_version_id, message, top_k=5) or []
        kwcontext = "\n\n".join(entry["raw"] for entry in knowledge)
        
        # Detect language of the message to determine response language
        is_vietnamese = any(word in message.lower() for word in ["tôi", "bạn", "không", "có", "là", "và", "em", "anh", "chị", "vâng", "đúng", "sai", "được"])
        
        # Enhanced prompt for more natural conversations
        prompt = (
                f"AI: {self.name} (smart, chill, personal assistant)\n"
                f"Context: {context}\n"
                f"Message: '{message}'\n"
                f"Task: Respond naturally and conversationally to the message. "
                f"{'Reply in Vietnamese' if is_vietnamese else 'Reply in English'} with a friendly, casual tone. "
                f"Maintain a natural flow with the conversation context. "
                f"Keep your response concise (1-2 sentences) unless the question requires more detail. "
                f"Avoid repeating information already mentioned in the context. "
                f"Skip formal greetings if the conversation is already ongoing."
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
    