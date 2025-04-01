import asyncio
from typing import List, Dict
from langchain_openai import ChatOpenAI
from utilities import logger
from langchain_core.messages import HumanMessage
import json
from sentence_transformers import SentenceTransformer, util
import numpy as np

LLM = ChatOpenAI(model="gpt-4o-mini", streaming=False)
StreamLLM = ChatOpenAI(model="gpt-4o-mini", streaming=True)

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

class CoreFuncs:
    # Load LaBSE once at class level
    embedder = SentenceTransformer('sentence-transformers/LaBSE')

    def __init__(self, user_id: str = "thefusionlab", convo_id: str = None, similarity_threshold: float = 0.55):
        self.user_id = user_id
        self.convo_id = convo_id or "default_thread"
        self.name = "Grok 3"
        self.instincts = {"friendly": "Be nice"}
        self.similarity_threshold = similarity_threshold
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

    async def detect_intent(self, message: str, context: str = "") -> str:
        prompt = (
            f"Message: '{message}'\n"
            f"Context: '{context}'\n"
            "Classify intent as 'teaching', 'request', or 'casual'. Return one word: 'teaching', 'request', or 'casual'. "
            "If it’s a clear question, refinement of a prior topic, or asks for advice, classify as 'request'. "
            "If it’s a greeting, conversational statement, or lacks clear directive, classify as 'casual'. "
            "Weigh context heavily to avoid misclassifying greetings as requests."
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
            f"'rejection', 'elaboration', 'satisfaction', 'confusion'. Return one word. "
            f"If the message is a greeting or introduces a new topic or question unrelated to prior context, classify as 'new'. "
            f"If it refines or expands on a prior response (e.g., 'how' after 'what'), classify as 'elaboration'. "
            f"If it seeks to clear up ambiguity (e.g., 'what do you mean'), classify as 'clarification'. "
            f"If it expresses gratitude or understanding (e.g., 'thanks', 'got it'), classify as 'satisfaction'. "
            f"If it confirms or agrees without preference (e.g., 'yes', 'ok'), classify as 'confirmation'."
        )
        try:
            response = await asyncio.to_thread(LLM.invoke, prompt)
            feedback = response.content.strip().replace("'", "")
            return feedback if feedback in FEEDBACKTYPE else "new"
        except Exception as e:
            logger.info(f"Feedback detection failed: {e}")
            return "new"

    async def resolve_related_request(self, message: str, feedback_type: str, state: Dict, context: str) -> Dict[str, any]:
        unresolved = [r for r in state.get("unresolved_requests", []) if not r["resolved"]]
        if not unresolved:
            logger.info("No unresolved requests found")
            return None

        if feedback_type in ["clarification", "elaboration", "confusion"]:
            # Use full context (including AI responses) to enrich message embedding
            message_with_context = f"{context} {message}" if context else message
            message_embedding = self.embedder.encode(message_with_context, convert_to_tensor=True)

            # Compare against unresolved request messages
            unresolved_texts = [r["message"] for r in unresolved]
            unresolved_embeddings = self.embedder.encode(unresolved_texts, convert_to_tensor=True)

            # Compute cosine similarities
            similarities = util.cos_sim(message_embedding, unresolved_embeddings)[0]
            max_similarity_idx = np.argmax(similarities)
            similarity_score = similarities[max_similarity_idx].item()
            latest_request = unresolved[-1]  # Most recent unresolved request
            latest_score = similarities[-1].item() if unresolved else 0

            # Log scores for debugging
            logger.info(f"LaBSE similarity score: {similarity_score:.2f} for '{message}' against '{unresolved[max_similarity_idx]['message']}'")
            logger.info(f"Latest request score: {latest_score:.2f} for '{latest_request['message']}'")

            # Prefer latest request if within 0.05 of max score and above threshold
            if latest_score >= self.similarity_threshold and (latest_score >= similarity_score - 0.05):
                related_request = latest_request
                logger.info(f"LaBSE linked to latest: {related_request['message']} (score: {latest_score:.2f}, threshold: {self.similarity_threshold})")
            elif similarity_score > self.similarity_threshold:
                related_request = unresolved[max_similarity_idx]
                logger.info(f"LaBSE linked to: {related_request['message']} (score: {similarity_score:.2f}, threshold: {self.similarity_threshold})")
            else:
                logger.info(f"No match: highest score {similarity_score:.2f} below threshold {self.similarity_threshold}")
                return None

            # Update request status based on feedback type
            if feedback_type == "clarification":
                related_request["status"] = "CLARIFIED"
            elif feedback_type == "elaboration":
                related_request["message"] = f"{related_request['message']} - {message}"
                related_request["status"] = "ELABORATED"
            elif feedback_type == "confusion":
                related_request["status"] = "NEEDS_CLARIFICATION"
            return related_request

        # Fallback to token matching for other feedback types
        message_tokens = set(message.lower().split())
        related_request = next(
            (r for r in unresolved 
            if len(message_tokens.intersection(
                set(r["message"].lower().split()) | set(r["response"].lower().split()))) >= 1),
            None
        )
        if related_request:
            logger.info(f"Linked via tokens to: {related_request['message']}")
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
            return related_request
        
        logger.info("No match found via tokens")
        return None

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

    async def triggercore(self, state: Dict = None, user_id: str = None):
        state = state or self.state.copy()
        user_id = user_id or self.user_id

        if "unresolved_requests" not in state:
            state["unresolved_requests"] = self.state.get("unresolved_requests", [])

        log_state = state.copy()
        if "messages" in log_state:
            log_state["messages"] = [{"content": msg.content} if isinstance(msg, HumanMessage) else {"content": msg} for msg in log_state["messages"]]
        logger.info(f"Starting Fun - User: {user_id}, State: {json.dumps(log_state, ensure_ascii=False)}")

        if not self.instincts:
            await self.initialize()

        latest_msg = state["messages"][-1] if state["messages"] else HumanMessage(content="")
        latest_msg_content = latest_msg.content.strip() if isinstance(latest_msg, HumanMessage) else latest_msg.strip()
        
        # Include both user and AI messages in context
        context = "\n".join(
            f"User: {msg.content}" if isinstance(msg, HumanMessage) else f"AI: {msg}"
            for msg in state["messages"][-10:]
        )
        logger.info(f"Context={context}")
        
        intent = await self.detect_intent(latest_msg_content, context)
        feedback = await self.detect_feedback_type(latest_msg_content, context)
        logger.info(f"Initial intent: {intent}, Feedback: {feedback}")

        if feedback in ["correction", "adjustment", "clarification", "confusion", "elaboration", "rejection"]:
            intent = "request"
            logger.info(f"Intent overridden to: {intent}")
        elif feedback in ["satisfaction", "confirmation"] and intent != "casual":
            intent = "request"
            logger.info(f"Intent overridden to: {intent}")

        state["intent_history"].append(intent)
        state["intent_history"] = state["intent_history"][-5:]
        state["unresolved_requests"] = [r for r in state["unresolved_requests"] if len(state["messages"]) - r["turn"] < 10]
        
        builder = ResponseBuilder()
        
        if intent == "request":
            async for response_chunk in self._handle_request(latest_msg_content, user_id, context, builder, feedback, state, intent):
                state["prompt_str"] = response_chunk
                yield state["prompt_str"]
        else:
            async for response_chunk in self._handle_casual(latest_msg_content, context, builder, state):
                state["prompt_str"] = response_chunk
                yield state["prompt_str"]
        
        final_response = state["prompt_str"]
        state["messages"].append(final_response)  # Add this line
        
        self.state.update(state)
        STATE_STORE[self.convo_id] = self.state
        logger.info(f"Final response: {state['prompt_str']}")

    async def _handle_request(self, message: str, user_id: str, context: str, builder: "ResponseBuilder", feedback_type: str, state: Dict, intent: str):
        instinct_guidance = f"Reflect instincts: {', '.join(self.instincts.keys()) or 'none'}."
        related_request = await self.resolve_related_request(message, feedback_type, state, context)
        logger.info(f"Intent at request: {intent}, Feedback type: {feedback_type}, Related request: {related_request is not None if related_request else 'None'}")

        if feedback_type == "satisfaction":
            unresolved = [r for r in state["unresolved_requests"] if not r["resolved"]]
            for req in unresolved:
                if req["message"] in context:
                    req["resolved"] = True
                    req["satisfied"] = True
                    req["status"] = "RESOLVED"
            builder.add("Được rồi, vui quá! Còn gì nữa không?")
            yield builder.build()
            return
        elif feedback_type == "confirmation" and not any(word in message.lower() for word in ["thích", "muốn", "chọn"]):
            builder.add("Tuyệt, hợp ý rồi nhé!")
            yield builder.build()
            return
        elif feedback_type == "rejection":
            if related_request:
                related_request["resolved"] = True
                related_request["satisfied"] = False
                related_request["status"] = "REJECTED"
            builder.add("OK, bỏ qua vậy. Tiếp theo đi!")
            yield builder.build()
            return
        elif feedback_type in ["elaboration", "clarification", "confusion"] and related_request:
            logger.info(f"Handling {feedback_type} for '{message}' with related request: {related_request['message']}")
            prompt = (
                f"AI: {self.name}\n"
                f"Context: {context}\n"
                f"Message: '{message}'\n"
                f"Task: {'Clarify' if feedback_type in ['clarification', 'confusion'] else 'Elaborate on'} prior response in Vietnamese, keep it concise. END with CLARIFICATION. Do not start with 'Hi!' or 'Hello, <name>!'\n"
                f"{instinct_guidance}"
            )
            builder = ResponseBuilder()  # Fresh builder to avoid old chunks
            async for chunk in self.stream_response(prompt, builder):
                yield builder.build()
            related_request["response"] = builder.build()
            return
        elif feedback_type == "new" or (not related_request and intent == "request"):
            logger.info(f"Processing as new request: '{message}'")
            prompt = (
                f"AI: {self.name}\n"
                f"Context: {context}\n"
                f"Message: '{message}'\n"
                f"Task: Reply in Vietnamese, explain if needed. End with 'NEW'. Do not start with 'Hi!' or 'Hello, <name>!'\n"
                f"{instinct_guidance}"
            )
            builder = ResponseBuilder()  # Fresh builder
            async for chunk in self.stream_response(prompt, builder):
                yield builder.build()
            response_text = builder.build()
            turn = sum(1 for msg in state["messages"] if isinstance(msg, HumanMessage))
            state["unresolved_requests"].append({
                "message": message,
                "turn": turn,
                "resolved": False,
                "response": response_text,
                "status": "RECEIVED",
                "satisfied": False
            })
            logger.info(f"Marked request '{message}' as RECEIVED")
            return
        elif feedback_type in ["correction", "adjustment"] and related_request:
            prompt = (
                f"AI: {self.name}\n"
                f"Context: {context}\n"
                f"Message: '{message}'\n"
                f"Task: Adjust or correct prior response in Vietnamese based on feedback.End with ADJUSTMENT. Do not start with 'Hi!' or 'Hello, <name>!'\n"
                f"{instinct_guidance}"
            )
            builder = ResponseBuilder()  # Fresh builder
            async for chunk in self.stream_response(prompt, builder):
                yield builder.build()
            related_request["response"] = builder.build()
            return

    async def _handle_casual(self, message: str, context: str, builder: "ResponseBuilder", state: Dict):
        instinct_guidance = f"Reflect instincts: {', '.join(self.instincts.keys()) or 'none'}."
        prompt = (
            f"{self.name} (smart, funny)\n"
            f"Context: {context}\n"
            f"Message: '{message}'\n"
            f"Task: Reply in Vietnamese, short and natural, show the sense of humor. Do not start with 'Hi!' or 'Hello, <name>!'\n"
            f"{instinct_guidance}"
        )
        async for chunk in self.stream_response(prompt, builder):
            yield chunk

    async def test_conversation(self):
        test_messages = [
            "Xin chào!",
            "Hôm nay em thế nào?",
            "Bạn có thể giải thích AI là gì không?",
            "Nó hoạt động thế nào?",
            "Cảm ơn, hiểu rồi!",
            "Bạn khỏe không?"
        ]
        print("Starting conversation test with LaBSE...")
        for i, msg in enumerate(test_messages):
            print(f"\nTurn {i + 1} - User: {msg}")
            self.state["messages"].append(HumanMessage(content=msg))
            full_response = ""
            async for response in self.triggercore():
                full_response = response
            print(f"AI: {full_response}")
            self.state["messages"].append(full_response)
            print(f"State after turn: {json.dumps(self.state, default=str, ensure_ascii=False)}")
        print("\nConversation test completed.")

if __name__ == "__main__":
    core = CoreFuncs(user_id="test_user", convo_id="chat_thread")
    asyncio.run(core.test_conversation())