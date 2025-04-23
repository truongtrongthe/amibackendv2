import asyncio
import json
import logging
import os
import re
from typing import List, Dict, Optional, AsyncGenerator
from langchain_openai import ChatOpenAI
from utilities import logger
from langchain_core.messages import HumanMessage, AIMessage
from database import (
    query_knowledge, 
    save_training, 
    # These functions are now imported from hotbrain instead
    # query_brain_with_embeddings_batch,
    # get_cached_embedding,
    # get_version_brain_banks,
    clean_text
)

# Import the functions from hotbrain module 
from hotbrain import (
    query_brain_with_embeddings_batch,
    get_cached_embedding,
    get_version_brain_banks,
    batch_get_embeddings
)

from analysis import (
    stream_analysis, stream_next_action, build_context_analysis_prompt,
    build_next_actions_prompt, process_analysis_result, process_next_actions_result,
    extract_search_terms_from_next_actions
)
from personality import PersonalityManager
from response_optimization import ResponseFilter, ResponseProcessor
import time
import nltk

# Configure NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if os.getenv("VERBOSE_LOGGING", "false").lower() == "true" else logging.INFO)

# Initialize LLMs
LLM = ChatOpenAI(model="gpt-4o-mini", streaming=False)
StreamLLM = ChatOpenAI(model="gpt-4o-mini", streaming=True)

def add_messages(existing_messages: List, new_messages: List) -> List:
    """Append new messages to existing message list."""
    return existing_messages + new_messages

def deduplicate_knowledge(entries: List[Dict]) -> List[Dict]:
    """Remove duplicate knowledge entries based on their IDs."""
    seen_ids = set()
    unique_entries = []
    for entry in entries:
        entry_id = entry.get('id', 'unknown')
        if entry_id not in seen_ids:
            seen_ids.add(entry_id)
            unique_entries.append(entry)
    return unique_entries

class ResponseBuilder:
    def __init__(self):
        """Initialize a response builder for constructing responses."""
        self.parts = []
        self.error_flag = False

    def add(self, text: str, is_error: bool = False) -> 'ResponseBuilder':
        """Add a text part to the response."""
        if text:
            self.parts.append(text)
        self.error_flag = self.error_flag or is_error
        return self

    def build(self, separator: str = None) -> str:
        """Build the final response string."""
        if not self.parts:
            return "Tôi không biết nói gì cả!"
        separator = separator or ""
        return separator.join(part for part in self.parts if part)

class MC:
    def __init__(self, user_id: str = "thefusionlab", convo_id: str = None,
                 similarity_threshold: float = 0.55, max_active_requests: int = 5):
        """
        Initialize the conversational AI system.

        Args:
            user_id: Unique user identifier.
            convo_id: Conversation thread ID.
            similarity_threshold: Threshold for knowledge relevance.
            max_active_requests: Maximum concurrent requests.
        """
        self.user_id = user_id
        self.convo_id = convo_id or "default_thread"
        self.similarity_threshold = similarity_threshold
        self.max_active_requests = max_active_requests
        self.personality_manager = PersonalityManager()
        self.response_filter = ResponseFilter()
        self.response_processor = ResponseProcessor()
        self._cache = {
            'personality': {}, 'language': {}, 'knowledge': {},
            'analysis': {}, 'query_embeddings': {}
        }
        self.state = {
            "messages": [], "intent_history": [], "preset_memory": "Be friendly",
            "unresolved_requests": [], "convo_id": self.convo_id,
            "user_id": self.user_id, "prompt_str": "", "graph_version_id": "",
            "analysis": {"english": "", "vietnamese": ""}, "instinct": {},
            "stream_events": []
        }
        # Load language patterns
        try:
            with open("language_patterns.json", "r") as f:
                self.language_patterns = json.load(f)
        except FileNotFoundError:
            logger.warning("language_patterns.json not found, using default patterns")
            self.language_patterns = {
                "vi": {
                    "question_particles": ["không", "nhỉ", "nhé", "chứ", "hả", "à", "sao"],
                    "continuity_markers": ["ngoài ra", "thêm nữa", "bên cạnh đó", "hơn nữa", "cũng", "còn"]
                },
                "ms": {
                    "question_particles": ["kah", "tak", "ke", "apa", "siapa", "bila", "mana"],
                    "continuity_markers": ["juga", "tambahan pula", "selain itu", "lagi", "dan"]
                },
                "en": {
                    "question_particles": ["what", "how", "why", "where", "when"],
                    "continuity_markers": ["also", "another", "additionally", "furthermore", "moreover"]
                }
            }

    @property
    def name(self) -> str:
        """Get the AI's name from personality manager."""
        return self.personality_manager.name

    @property
    def personality_instructions(self) -> str:
        """Get the personality instructions."""
        return self.personality_manager.personality_instructions

    @property
    def instincts(self) -> Dict:
        """Get the personality instincts."""
        return self.personality_manager.instincts

    async def initialize(self):
        """Initialize personality manager if not already set."""
        if not hasattr(self, 'personality_manager'):
            self.personality_manager = PersonalityManager()

    async def stream_response(self, prompt: str, builder: ResponseBuilder, knowledge_found: bool = False) -> AsyncGenerator[str, None]:
        """
        Stream a response from the LLM with optimized processing.

        Args:
            prompt: The prompt for the LLM.
            builder: ResponseBuilder instance.
            knowledge_found: Whether knowledge was retrieved.

        Yields:
            str: Streamed response chunks.
        """
        if len(self.state["messages"]) > 20:
            prompt += "\nKeep it short and sweet."

        is_first_message = len(self.state["messages"]) <= 1
        buffer = ""

        try:
            full_response = ""
            async for chunk in StreamLLM.astream(prompt):
                buffer += chunk.content
                full_response += chunk.content

            processed_response = self.response_processor.process_response(
                full_response, self.state,
                self.state["messages"][-1].content if self.state["messages"] else "",
                knowledge_found
            )

            if not is_first_message:
                processed_response = self.response_filter.remove_greeting(processed_response)

            sentences = processed_response.split(". ")
            for i, sentence in enumerate(sentences):
                if i < len(sentences) - 1 or not sentence.endswith((".", "!", "?")):
                    sentence = sentence + "."
                builder.add(sentence.strip())
                yield builder.build(separator=" ")

            self.response_filter.increment_turn()

        except Exception as e:
            logger.error(f"Streaming failed: {e}", exc_info=True)
            builder.add("Có lỗi nhỏ, thử lại nhé!")
            yield builder.build(separator="\n")

    async def stream_analysis(self, prompt: str, thread_id_for_analysis: Optional[str] = None, use_websocket: bool = False) -> AsyncGenerator[Dict, None]:
        """
        Stream context analysis from the LLM with optimized chunking.

        Args:
            prompt: The prompt to analyze.
            thread_id_for_analysis: WebSocket thread ID for analysis events.
            use_websocket: Whether to use WebSocket for streaming.

        Yields:
            Dict: Analysis event with content and completion status.
        """
        analysis_buffer = ""
        chunk_buffer = ""
        chunk_size_threshold = 250

        try:
            async for chunk in StreamLLM.astream(prompt):
                chunk_content = chunk.content
                analysis_buffer += chunk_content
                chunk_buffer += chunk_content

                if (len(chunk_buffer) >= chunk_size_threshold or
                        (len(chunk_buffer) > 50 and chunk_buffer.endswith((".", "!", "?", "\n")))):
                    analysis_event = {"type": "analysis", "content": chunk_buffer, "complete": False}
                    if use_websocket and thread_id_for_analysis:
                        try:
                            from socketio_manager import emit_analysis_event
                            emit_analysis_event(thread_id_for_analysis, analysis_event)
                        except Exception as e:
                            logger.error(f"WebSocket delivery error: {str(e)}")
                    yield analysis_event
                    chunk_buffer = ""

            if chunk_buffer:
                intermediate_event = {"type": "analysis", "content": chunk_buffer, "complete": False}
                if use_websocket and thread_id_for_analysis:
                    try:
                        from socketio_manager import emit_analysis_event
                        emit_analysis_event(thread_id_for_analysis, intermediate_event)
                    except Exception as e:
                        logger.error(f"WebSocket delivery error: {str(e)}")
                yield intermediate_event

            complete_event = {"type": "analysis", "content": analysis_buffer, "complete": True}
            if use_websocket and thread_id_for_analysis:
                try:
                    from socketio_manager import emit_analysis_event
                    emit_analysis_event(thread_id_for_analysis, complete_event)
                except Exception as e:
                    logger.error(f"WebSocket delivery error: {str(e)}")
            yield complete_event

        except Exception as e:
            logger.error(f"Analysis streaming failed: {e}")
            error_event = {"type": "analysis", "content": "Error in analysis process", "complete": True, "error": True}
            if use_websocket and thread_id_for_analysis:
                try:
                    from socketio_manager import emit_analysis_event
                    emit_analysis_event(thread_id_for_analysis, error_event)
                except Exception as e:
                    logger.error(f"WebSocket delivery error: {str(e)}")
            yield error_event

    async def trigger(self, state: Dict = None, user_id: str = None, graph_version_id: str = None, config: Dict = None) -> AsyncGenerator[Dict, None]:
        """
        Process a user request and generate a response.

        Args:
            state: Current conversation state.
            user_id: User identifier.
            graph_version_id: Knowledge graph version ID.
            config: Configuration overrides.

        Yields:
            Dict: Response chunks, analysis events, or final state.
        """
        state = state or self.state.copy()
        user_id = user_id or self.user_id
        config = config or {}
        graph_version_id = (
            graph_version_id or state.get("graph_version_id",
                config.get("configurable", {}).get("graph_version_id", ""))
        )
        use_websocket = state.get("use_websocket", False) or config.get("configurable", {}).get("use_websocket", False)
        thread_id_for_analysis = state.get("thread_id_for_analysis") or config.get("configurable", {}).get("thread_id_for_analysis")

        if not graph_version_id:
            logger.warning(f"graph_version_id is empty! State: {state}, Config: {config}")

        self.state["graph_version_id"] = graph_version_id
        latest_msg = state["messages"][-1] if state["messages"] else HumanMessage(content="")
        latest_msg_content = latest_msg.content.strip() if isinstance(latest_msg, HumanMessage) else latest_msg.strip()
        context = "\n".join(f"User: {msg.content}" if isinstance(msg, HumanMessage) else f"AI: {msg.content}" for msg in state["messages"][-50:])

        logger.info(f"Triggering - User: {user_id}, Graph Version: {graph_version_id}, Message: {latest_msg_content[:50]}...")

        builder = ResponseBuilder()
        async for response_chunk in self._handle_request(
            latest_msg_content, user_id, context, builder, state, graph_version_id,
            use_websocket, thread_id_for_analysis
        ):
            if isinstance(response_chunk, dict) and response_chunk.get("type") in ["analysis", "knowledge"]:
                state["stream_events"] = state.get("stream_events", []) + [response_chunk]
                yield response_chunk
            else:
                state["prompt_str"] = response_chunk
                yield response_chunk

        if state["prompt_str"]:
            state["messages"] = add_messages(state["messages"], [AIMessage(content=state["prompt_str"])])
        self.state.update(state)
        yield {"state": state}

    async def detect_language_with_llm(self, text: str, llm=LLM) -> Dict:
        """
        Detect the language of the input text using the LLM.

        Args:
            text: Input text to analyze.
            llm: Language model instance.

        Returns:
            Dict: Language information with name, code, confidence, and response guidance.
        """
        context_prompt = (
            f"Identify the language of this text: '{text}'\n"
            f"Return in JSON format:\n"
            f"{{\n"
            f"  \"language\": \"[language name in English]\",\n"
            f"  \"code\": \"[ISO 639-1 code]\",\n"
            f"  \"confidence\": [0-1],\n"
            f"  \"responseGuidance\": \"[Guidance for responding]\"\n"
            f"}}"
        )

        try:
            response = await llm.ainvoke(context_prompt) if asyncio.iscoroutinefunction(llm.invoke) else llm.invoke(context_prompt)
            response_text = getattr(response, 'content', response).strip()
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                lang_data = json.loads(json_match.group(0))
                if all(k in lang_data for k in ["language", "code", "confidence", "responseGuidance"]):
                    return lang_data
            logger.warning(f"Invalid language detection format: {response_text[:100]}...")
        except Exception as e:
            logger.error(f"Language detection error: {str(e)}")
        return {
            "language": "English", "code": "en", "confidence": 0.5,
            "responseGuidance": "Respond in a neutral, professional tone"
        }

    async def detect_conversation_language(self, conversation: List[str]) -> Optional[str]:
        """
        Determine the primary language of the conversation.

        Args:
            conversation: List of conversation lines.

        Returns:
            Optional[str]: Detected language or None if insufficient data.
        """
        user_messages = [line[5:].strip() for line in conversation if line.startswith("User:") and line[5:].strip()]
        if len(user_messages) < 2:
            return None

        lang_counts = {}
        for message in user_messages[-3:]:
            try:
                lang_info = await self.detect_language_with_llm(message)
                lang = lang_info.get("language", "Unknown")
                confidence = lang_info.get("confidence", 0)
                if confidence > 0.6:
                    lang_counts[lang] = lang_counts.get(lang, 0) + 1
            except Exception:
                continue

        return max(lang_counts.items(), key=lambda x: x[1])[0] if lang_counts else "English"

    async def maintain_contextual_memory(self, context: str, message: str, knowledge_context: str, lang_info: Optional[Dict] = None) -> Dict:
        """
        Maintain contextual memory for conversation continuity.

        Args:
            context: Conversation history.
            message: Current user message.
            knowledge_context: Retrieved knowledge.
            lang_info: Language information dictionary.

        Returns:
            Dict: Contextual memory with topics, preferences, and language support.
        """
        language_confident = lang_info and lang_info.get("confidence", 0) > 0.7
        language_code = lang_info.get("code", "en") if language_confident else "en"

        context_memory = {
            "mentioned_topics": [], "user_preferences": {}, "answered_questions": [],
            "unresolved_topics": [], "emotional_markers": {}, "conversation_history_summary": "",
            "continuity_hints": [], "language_support": {
                "code": language_code, "full_support": language_code == "en",
                "partial_support": language_code in ["vi", "ms", "zh", "fr", "es"],
                "confidence": lang_info.get("confidence", 1.0) if lang_info else 1.0
            }
        }

        # Parse conversation
        turns = []
        current_turn = {"speaker": None, "text": ""}
        for line in context.split('\n'):
            if line.startswith("User:"):
                if current_turn["speaker"] == "AI": turns.append(current_turn)
                current_turn = {"speaker": "User", "text": line[5:].strip()}
            elif line.startswith("AI:"):
                if current_turn["speaker"] == "User": turns.append(current_turn)
                current_turn = {"speaker": "AI", "text": line[4:].strip()}
            elif line.strip() and current_turn["speaker"]:
                current_turn["text"] += " " + line.strip()
        if current_turn["speaker"] and current_turn["text"]: turns.append(current_turn)

        # Extract topics from knowledge
        if knowledge_context:
            words = knowledge_context.lower().split()
            topic_counts = {word: words.count(word) for word in set(words) if len(word) > 3 and word not in ["this", "that", "with"]}
            context_memory["mentioned_topics"] = [word for word, _ in sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:5]]

        # Detect questions
        recent_user_turns = [t for t in turns[-6:] if t["speaker"] == "User"]
        for turn in recent_user_turns:
            questions = [s.strip() + "?" for s in turn["text"].split("?") if s.strip()]
            if language_confident and language_code in self.language_patterns:
                sentences = [s.strip() for s in turn["text"].split(".") if s.strip() and "?" not in s]
                particles = self.language_patterns[language_code].get("question_particles", [])
                questions.extend(s for s in sentences if any(p in s.lower() for p in particles))

            for question in questions:
                question_words = set(question.lower().split())
                is_answered = False
                for ai_turn in [t for t in turns if t["speaker"] == "AI"]:
                    if len(question_words.intersection(set(ai_turn["text"].lower().split()))) >= (2 if language_code in ["vi", "zh"] else 3):
                        is_answered = True
                        context_memory["answered_questions"].append(question)
                        break
                if not is_answered:
                    context_memory["unresolved_topics"].append(question)

        # Continuity hints
        if turns and language_code in self.language_patterns:
            last_ai_turns = [t for t in turns if t["speaker"] == "AI"]
            if last_ai_turns:
                markers = self.language_patterns[language_code].get("continuity_markers", [])
                for sentence in last_ai_turns[-1]["text"].split("."):
                    if any(marker in sentence.lower() for marker in markers):
                        context_memory["continuity_hints"].append({"type": "continue_thread", "thread": sentence.strip()})

        # Summary
        if len(turns) > 4:
            topics = ", ".join(context_memory["mentioned_topics"][:3]) or "general topics"
            context_memory["conversation_history_summary"] = f"Conversation with {len(turns)} turns covering {topics}."
            if context_memory["unresolved_topics"]:
                context_memory["conversation_history_summary"] += f" {len(context_memory['unresolved_topics'])} unresolved questions."

        return context_memory

    async def _handle_request(self, message: str, user_id: str, context: str, builder: ResponseBuilder, state: Dict, graph_version_id: str, use_websocket: bool, thread_id_for_analysis: str) -> AsyncGenerator[Dict, None]:
        """
        Handle a user request by processing language, knowledge, analysis, and response generation.

        Args:
            message: User's input message.
            user_id: Unique user identifier.
            context: Conversation history.
            builder: ResponseBuilder instance.
            state: Current conversation state.
            graph_version_id: Knowledge graph version ID.
            use_websocket: Whether to use WebSocket for streaming.
            thread_id_for_analysis: WebSocket thread ID for analysis events.

        Yields:
            Dict: Analysis, knowledge, or response chunks.
        """
        try:
            await self._load_personality(graph_version_id)
            semaphore = asyncio.Semaphore(3)

            async def limited_task(coro):
                async with semaphore:
                    return await coro

            conversation = [line.strip() for line in context.split("\n") if line.strip()]
            lang_info, conversation_language, profile_entries, dynamics_insights, naturalness_guidance = await asyncio.gather(
                limited_task(self.detect_language_with_llm(message)),
                limited_task(self.detect_conversation_language(conversation)),
                limited_task(query_graph_knowledge(graph_version_id, "contact profile building information gathering customer understanding", top_k=5)),
                limited_task(self.track_conversation_dynamics(context, message)),
                limited_task(self.enhance_response_naturalness(context))
            )

            if conversation_language and conversation_language != lang_info["language"] and len(conversation) > 5:
                logger.debug(f"Overriding language {lang_info['language']} with {conversation_language}")
                lang_info["language"] = conversation_language
                lang_info["confidence"] = 0.9
                lang_info["responseGuidance"] = f"Respond in {conversation_language} consistently."

            language_to_code = {"english": "en", "vietnamese": "vi", "malay": "ms", "chinese": "zh", "french": "fr", "spanish": "es"}
            lang_info["code"] = lang_info.get("code", language_to_code.get(lang_info["language"].lower(), "en"))

            profile_instructions = "\n\n".join(entry["raw"] for entry in profile_entries)
            context_analysis_prompt = build_context_analysis_prompt(context, profile_instructions)
            context_analysis, initial_search_terms = "", []
            async for chunk in stream_analysis(context_analysis_prompt, thread_id_for_analysis, use_websocket):
                yield chunk
                if chunk.get("type") == "analysis" and chunk.get("complete"):
                    context_analysis = chunk.get("content", "")
                    analysis_result = process_analysis_result(context_analysis)
                    initial_search_terms = analysis_result.get("search_terms", [])

            search_terms = initial_search_terms or [message]
            if len(message.split()) <= 10 and message not in search_terms:
                search_terms.append(message)
            filtered_queries = [term for term in set(term.lower().strip() for term in search_terms)
                               if len(term) > 3 and term not in ['the', 'and', 'or', 'but', 'for', 'with', 'that']]
            knowledge_context = await self._retrieve_knowledge(graph_version_id, filtered_queries, use_websocket, thread_id_for_analysis)

            next_actions_prompt = build_next_actions_prompt(context, context_analysis, knowledge_context)
            next_action_content = ""
            async for chunk in stream_next_action(next_actions_prompt, thread_id_for_analysis, use_websocket):
                yield chunk
                if chunk.get("complete") and chunk.get("content"):
                    next_action_content = chunk.get("content", "")

            additional_knowledge_context = await self._retrieve_next_action_knowledge(
                graph_version_id, next_action_content, use_websocket, thread_id_for_analysis
            )
            if additional_knowledge_context:
                knowledge_context += "\n\n" + additional_knowledge_context

            contextual_memory = await self.maintain_contextual_memory(context, message, knowledge_context, lang_info)

            response_prompt = self._build_response_prompt(
                message, context, context_analysis, next_action_content, knowledge_context, lang_info,
                dynamics_insights, naturalness_guidance, contextual_memory
            )
            async for _ in self.stream_response(response_prompt, builder, bool(knowledge_context)):
                yield builder.build()

        except Exception as e:
            logger.error(f"Error handling request: {str(e)}", exc_info=True)
            builder.add("Oops, something went wrong. Please try again!")
            yield builder.build()

    async def _load_personality(self, graph_version_id: str) -> None:
        """
        Load personality instructions for the given graph version.

        Args:
            graph_version_id: Knowledge graph version ID.
        """
        await self.load_personality_instructions(graph_version_id)
        logger.info(f"Loaded personality for graph_version_id: {graph_version_id}, AI name: {self.personality_manager.name}")

    async def _retrieve_knowledge(self, graph_version_id: str, queries: List[str], use_websocket: bool, thread_id: str) -> str:
        """
        Retrieve and deduplicate knowledge for given queries.

        Args:
            graph_version_id: Knowledge graph version ID.
            queries: List of query strings.
            use_websocket: Whether to use WebSocket for streaming.
            thread_id: WebSocket thread ID.

        Returns:
            str: Concatenated knowledge context.
        """
        all_knowledge = []
        for i in range(0, len(queries), 5):
            batch = queries[i:i + 5]
            if use_websocket:
                async for event in self._stream_batch_query_knowledge(graph_version_id, batch, top_k=3, thread_id=thread_id):
                    if not event.get("complete") and event.get("content"):
                        all_knowledge.extend(event.get("content", []))
            else:
                batch_results = await self._batch_query_knowledge(graph_version_id, batch, top_k=3)
                for results in batch_results:
                    all_knowledge.extend(results)
        unique_knowledge = deduplicate_knowledge(all_knowledge)
        logger.debug(f"Retrieved {len(unique_knowledge)} unique knowledge entries")
        return "\n\n".join(entry.get("raw", "") for entry in unique_knowledge)

    async def _retrieve_next_action_knowledge(self, graph_version_id: str, next_action_content: str, use_websocket: bool, thread_id: str) -> str:
        """
        Retrieve additional knowledge based on next action content.

        Args:
            graph_version_id: Knowledge graph version ID.
            next_action_content: Next action text.
            use_websocket: Whether to use WebSocket for streaming.
            thread_id: WebSocket thread ID.

        Returns:
            str: Additional knowledge context.
        """
        if not next_action_content or len(next_action_content) < 50:
            return ""
        try:
            next_actions_data = process_next_actions_result(next_action_content)
            english_next_actions = next_actions_data.get("next_action_english", "")
            if not english_next_actions:
                return ""

            sentences = re.findall(r'[^.!?]+[.!?]', english_next_actions)
            filtered_sentences = [s.strip() for s in sentences if len(s.split()) > 5 and not all(c.isupper() for c in s if c.isalpha())]
            bullet_points = [p.strip() for p in re.findall(r'[-•*]\s*[^-•*\n]+', english_next_actions) if len(p.split()) > 5]
            quoted_questions = [q for q in re.findall(r'"([^"]*\?)"', english_next_actions) if len(q.split()) > 3]
            action_patterns = [
                r'(Ask about [^.!?]+[.!?])', r'(Inquire about [^.!?]+[.!?])', r'(Explain [^.!?]+[.!?])',
                r'(Suggest [^.!?]+[.!?])', r'(Recommend [^.!?]+[.!?])', r'(Provide [^.!?]+[.!?])'
            ]
            action_sentences = []
            for pattern in action_patterns:
                action_sentences.extend(a for a in re.findall(pattern, english_next_actions) if len(a.split()) > 3)

            queries = list(set(filtered_sentences + bullet_points + quoted_questions + action_sentences))
            if not queries:
                return ""

            additional_knowledge = []
            for i in range(0, len(queries), 3):
                batch = queries[i:i + 3]
                if use_websocket:
                    async for event in self._stream_batch_query_knowledge(graph_version_id, batch, top_k=3, thread_id=thread_id):
                        if not event.get("complete") and event.get("content"):
                            additional_knowledge.extend(event.get("content", []))
                else:
                    batch_results = await self._batch_query_knowledge(graph_version_id, batch, top_k=3)
                    for results in batch_results:
                        additional_knowledge.extend(results)

            unique_additional_knowledge = deduplicate_knowledge(additional_knowledge)
            return "\n\n".join(entry["raw"] for entry in unique_additional_knowledge)

        except Exception as e:
            logger.error(f"Error retrieving next action knowledge: {str(e)}")
            return ""

    def _build_response_prompt(self, message: str, context: str, context_analysis: str, next_action_content: str, knowledge_context: str, lang_info: Dict, dynamics_insights: Dict, naturalness_guidance: Dict, contextual_memory: Dict) -> str:
        """
        Build a concise prompt for response generation.

        Args:
            message: User message.
            context: Conversation history.
            context_analysis: Analyzed context.
            next_action_content: Suggested next actions.
            knowledge_context: Retrieved knowledge.
            lang_info: Language information.
            dynamics_insights: Conversation dynamics.
            naturalness_guidance: Response naturalness guidance.
            contextual_memory: Contextual memory.

        Returns:
            str: Formatted prompt.
        """
        return (
            f"AI: {self.name}\n"
            f"Role: Friendly sales assistant\n"
            f"Language: {lang_info['language']}\n"
            f"Message: {message}\n"
            f"Context: {context}\n"
            f"Analysis: {context_analysis}\n"
            f"Next Actions: {next_action_content}\n"
            f"Knowledge: {knowledge_context}\n"
            f"Personality: {self.personality_instructions}\n"
            f"Dynamics: {json.dumps(dynamics_insights, indent=2)}\n"
            f"Naturalness: {json.dumps(naturalness_guidance, indent=2)}\n"
            f"Memory: {json.dumps(contextual_memory, indent=2)}\n\n"
            f"Task:\n"
            f"- Respond based on next actions and context analysis.\n"
            f"- Use knowledge or workarounds if information is missing.\n"
            f"- Maintain a concise, empathetic, culturally sensitive tone in {lang_info['language']}.\n"
            f"- Avoid repetition or premature actions.\n"
            f"- Adapt to cultural needs (e.g., calls instead of meetings for Vietnamese users).\n"
        )

    async def load_personality_instructions(self, graph_version_id: str = "") -> str:
        """
        Load personality instructions with caching.

        Args:
            graph_version_id: Knowledge graph version ID.

        Returns:
            str: Personality instructions.
        """
        cache_key = f"personality_{graph_version_id}"
        if cache_key in self._cache.get('personality', {}):
            cached_data = self._cache['personality'][cache_key]
            self.personality_manager.personality_instructions = cached_data.get('instructions', '')
            self.personality_manager.name = cached_data.get('name', 'Ami')
            self.personality_manager.instincts = cached_data.get('instincts', {})
            self.personality_manager.is_loaded_from_knowledge = True
            self.personality_manager.loaded_graph_version_id = graph_version_id
            logger.debug(f"Loaded personality from cache for graph_version_id: {graph_version_id}")
            return self.personality_manager.personality_instructions

        await self.personality_manager.load_personality_instructions(graph_version_id)
        self._cache.setdefault('personality', {})[cache_key] = {
            'instructions': self.personality_manager.personality_instructions,
            'name': self.personality_manager.name,
            'instincts': self.personality_manager.instincts,
            'timestamp': time.time()
        }
        logger.debug(f"Cached personality for graph_version_id: {graph_version_id}")
        return self.personality_manager.personality_instructions

    async def _batch_query_knowledge(self, graph_version_id: str, queries: List[str], top_k: int = 3) -> List[List[Dict]]:
        """
        Batch process multiple knowledge queries using vector search.

        Args:
            graph_version_id: Knowledge graph version ID.
            queries: List of query strings.
            top_k: Maximum number of results per query.

        Returns:
            List[List[Dict]]: List of results for each query.
        """
        if not queries:
            logger.warning("Empty queries list provided")
            return []

        logger.debug(f"Starting batch query for {len(queries)} queries")
        try:
            brain_banks = await get_version_brain_banks(graph_version_id)
            if not brain_banks:
                logger.warning(f"No brain banks found for graph version {graph_version_id}")
                return [[] for _ in queries]

            embedding_tasks = [get_cached_embedding(query) for query in queries]
            embeddings = await asyncio.gather(*embedding_tasks)
            query_embeddings = {i: embedding for i, embedding in enumerate(embeddings)}

            all_results = [[] for _ in queries]
            async def process_brain_bank(brain_bank):
                bank_name, brain_id = brain_bank["bank_name"], brain_bank["id"]
                brain_results = await query_brain_with_embeddings_batch(query_embeddings, bank_name, brain_id, top_k)
                for query_idx, results in brain_results.items():
                    if 0 <= query_idx < len(all_results):
                        all_results[query_idx].extend(results)

            await asyncio.gather(*[process_brain_bank(bank) for bank in brain_banks])
            final_results = [
                sorted(results, key=lambda x: x.get("score", 0), reverse=True)[:top_k]
                for results in all_results
            ]
            total_results = sum(len(results) for results in final_results)
            logger.debug(f"Batch processing complete: {total_results} total results")
            return final_results

        except Exception as e:
            logger.error(f"Error in batch query processing: {str(e)}")
            return await asyncio.gather(*[query_graph_knowledge(graph_version_id, query, top_k) for query in queries])

    async def _stream_batch_query_knowledge(self, graph_version_id: str, queries: List[str], top_k: int = 3, thread_id: Optional[str] = None) -> AsyncGenerator[Dict, None]:
        """
        Stream knowledge query results in real-time for multiple queries.

        Args:
            graph_version_id: Knowledge graph version ID.
            queries: List of query strings.
            top_k: Maximum number of results per query.
            thread_id: Optional WebSocket thread ID for streaming.

        Yields:
            Dict: Knowledge event with results or completion status.
        """
        if not queries:
            logger.warning("Empty queries list provided")
            yield {"type": "knowledge", "content": [], "complete": True}
            return

        logger.debug(f"Starting batch query for {len(queries)} queries")
        brain_banks = await get_version_brain_banks(graph_version_id)
        if not brain_banks:
            logger.warning(f"No brain banks found for graph version {graph_version_id}")
            yield {"type": "knowledge", "content": [], "complete": True}
            return

        embedding_tasks = [get_cached_embedding(query) for query in queries]
        embeddings = await asyncio.gather(*embedding_tasks)
        query_embeddings = {i: embedding for i, embedding in enumerate(embeddings)}

        all_results = [[] for _ in queries]
        streamed_ids = set()

        async def process_brain_bank(brain_bank):
            bank_name, brain_id = brain_bank["bank_name"], brain_bank["id"]
            brain_results = await query_brain_with_embeddings_batch(query_embeddings, bank_name, brain_id, top_k)
            new_results = []
            for query_idx, results in brain_results.items():
                if 0 <= query_idx < len(all_results):
                    for result in results:
                        if result['id'] not in streamed_ids:
                            streamed_ids.add(result['id'])
                            result_with_query = result.copy()
                            result_with_query['query'] = queries[query_idx]
                            result_with_query['query_idx'] = query_idx
                            new_results.append(result_with_query)
                    all_results[query_idx].extend(results)
            return new_results

        brain_bank_results = await asyncio.gather(*[process_brain_bank(bank) for bank in brain_banks])
        for results_batch in brain_bank_results:
            if results_batch:
                event = {"type": "knowledge", "content": results_batch, "complete": False}
                if thread_id:
                    try:
                        from socketio_manager import emit_knowledge_event
                        emit_knowledge_event(thread_id, event)
                    except Exception as e:
                        logger.error(f"WebSocket delivery error: {str(e)}")
                yield event

        final_results = [
            sorted(results, key=lambda x: x.get("score", 0), reverse=True)[:top_k]
            for results in all_results
        ]
        total_results = sum(len(results) for results in final_results)
        yield {
            "type": "knowledge", "content": [], "complete": True,
            "summary": {"total_results": total_results, "empty_queries": sum(1 for r in final_results if not r)}
        }

    async def adapt_response_to_culture(self, text: str, culture: str = "default") -> str:
        """
        Adapt a response to specific cultural contexts.

        Args:
            text: Response text to adapt.
            culture: Target culture code (e.g., "vi", "en").

        Returns:
            str: Culturally adapted response.
        """
        cultural_adaptations = {
            "vi": {
                "greeting_patterns": ["Xin chào", "Chào bạn", "Kính chào"],
                "politeness_markers": ["ạ", "nhé", "vui lòng"]
            },
            "en": {
                "greeting_patterns": ["Hello", "Hi", "Good day"],
                "politeness_markers": ["please", "thank you"]
            }
        }
        return text if culture not in cultural_adaptations else text  # Placeholder for future enhancements

    async def track_conversation_dynamics(self, conversation: str, current_message: str) -> Dict:
        """
        Track conversation dynamics to improve flow and reduce repetition.

        Args:
            conversation: Conversation history.
            current_message: Current user message.

        Returns:
            Dict: Insights about conversation dynamics.
        """
        user_messages = [line[5:].strip() for line in conversation.split('\n') if line.startswith("User:") and line[5:].strip()]
        if current_message not in user_messages:
            user_messages.append(current_message)

        insights = {
            "repetition_risk": False, "user_engagement": "medium", "conversation_pace": "normal",
            "topic_shifts": [], "repeated_themes": [], "suggested_approaches": []
        }

        if len(user_messages) >= 3:
            recent_messages = user_messages[-3:]
            word_counts = {}
            for msg in recent_messages:
                for word in msg.lower().split():
                    if len(word) > 3 and word not in ["what", "when", "where", "how", "why"]:
                        word_counts[word] = word_counts.get(word, 0) + 1
            insights["repeated_themes"] = [word for word, count in word_counts.items() if count >= 2]
            if insights["repeated_themes"]:
                insights["repetition_risk"] = True
                insights["suggested_approaches"].append("introduce_new_aspects")

        avg_msg_length = sum(len(msg) for msg in user_messages[-3:]) / min(3, len(user_messages))
        if avg_msg_length < 15:
            insights["user_engagement"] = "low"
            insights["suggested_approaches"].append("ask_engaging_questions")
        elif avg_msg_length > 100:
            insights["user_engagement"] = "high"
            insights["suggested_approaches"].append("provide_concise_responses")

        if len(user_messages) >= 2:
            last_msg = user_messages[-1].lower()
            if any(indicator in last_msg for indicator in ["instead", "by the way", "different"]):
                insights["topic_shifts"].append("user_initiated")
                insights["suggested_approaches"].append("acknowledge_shift")

        return insights

    async def enhance_response_naturalness(self, context: str) -> Dict:
        """
        Generate guidance for natural response patterns.

        Args:
            context: Conversation context.

        Returns:
            Dict: Guidance for sentence variations and patterns.
        """
        message_count = len([line for line in context.split('\n') if line.strip()])
        is_early_conversation = message_count < 6

        guidance = {
            "sentence_variations": [], "transitional_phrases": [],
            "recommended_patterns": [], "avoid_patterns": []
        }

        if is_early_conversation:
            guidance["sentence_variations"] = ["shorter_sentences", "questions"]
            guidance["transitional_phrases"] = ["first", "to start with"]
            guidance["recommended_patterns"] = ["personal_greeting", "open_ended_questions"]
            guidance["avoid_patterns"] = ["complex_explanations", "technical_jargon"]
        else:
            guidance["sentence_variations"] = ["mix_short_and_long", "reflective_observations"]
            guidance["transitional_phrases"] = ["additionally", "considering what you mentioned"]
            guidance["recommended_patterns"] = ["refer_to_previous_points", "add_personal_perspective"]
            guidance["avoid_patterns"] = ["repetitive_structure", "overuse_of_questions"]

        positive_count = sum(1 for word in ["thanks", "good", "great"] if word in context.lower())
        negative_count = sum(1 for word in ["no", "not", "problem"] if word in context.lower())
        if positive_count > negative_count * 2:
            guidance["recommended_patterns"].append("upbeat_friendly")
        elif negative_count > positive_count * 2:
            guidance["recommended_patterns"].append("empathetic_listening")
            guidance["avoid_patterns"].append("overly_cheerful")

        return guidance