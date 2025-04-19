import asyncio
from typing import List, Dict, Optional
from langchain_openai import ChatOpenAI
from utilities import logger
from langchain_core.messages import HumanMessage,AIMessage
import json
#from sentence_transformers import SentenceTransformer, util
import numpy as np
from database import query_graph_knowledge, get_version_brain_banks, get_cached_embedding
import re
from typing import Tuple, Dict, Any
from analysis import (
    stream_analysis, 
    stream_next_action,
    build_context_analysis_prompt, 
    build_next_actions_prompt,
    process_analysis_result, 
    process_next_actions_result,
    extract_search_terms, 
    extract_search_terms_from_next_actions
)
from personality import PersonalityManager
from response_optimization import ResponseFilter, ResponseStructure, ResponseProcessor
import time
import datetime
import os
import random
import uuid

import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

def add_messages(existing_messages, new_messages):
    return existing_messages + new_messages

LLM = ChatOpenAI(model="gpt-4o-mini", streaming=False)
StreamLLM = ChatOpenAI(model="gpt-4o-mini", streaming=True)

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
    #embedder = SentenceTransformer('sentence-transformers/LaBSE')

    def __init__(self, user_id: str = "thefusionlab", convo_id: str = None, 
                 similarity_threshold: float = 0.55, max_active_requests: int = 5):
        self.user_id = user_id
        self.convo_id = convo_id or "default_thread"
        self.similarity_threshold = similarity_threshold
        self.max_active_requests = max_active_requests
        self.personality_manager = PersonalityManager()
        self.response_filter = ResponseFilter()  # Add response filter
        self.response_processor = ResponseProcessor()  # Add response processor
        self.state = {
            "messages": [],
            "intent_history": [],
            "preset_memory": "Be friendly",
            "unresolved_requests": [],
            "convo_id": self.convo_id,
            "user_id": self.user_id,
            "prompt_str": "",
            "graph_version_id": "",
            "analysis": {
                "english": "",
                "vietnamese": ""
            }
        }

    @property
    def name(self):
        return self.personality_manager.name

    @property
    def personality_instructions(self):
        return self.personality_manager.personality_instructions

    @property
    def instincts(self):
        return self.personality_manager.instincts

    async def initialize(self):
        if not hasattr(self, 'personality_manager'):
            self.personality_manager = PersonalityManager()
        # No longer load personality here, will be loaded on first request
    
    async def stream_response(self, prompt: str, builder: ResponseBuilder, knowledge_found: bool = False):
        if len(self.state["messages"]) > 20:
            prompt += "\nKeep it short and sweet."
        
        # Check if this is the first message or a continuation
        is_first_message = len(self.state["messages"]) <= 1
        buffer = ""
        
        try:
            # First collect the full response
            full_response = ""
            async for chunk in StreamLLM.astream(prompt):
                buffer += chunk.content
                full_response += chunk.content
            
            # Process the full response to optimize structure
            processed_response = self.response_processor.process_response(
                full_response, 
                self.state, 
                self.state["messages"][-1].content if self.state["messages"] else "",
                knowledge_found
            )
            
            # Then apply greeting filter if needed
            if not is_first_message:
                processed_response = self.response_filter.remove_greeting(processed_response)
            
            # Break into sentences for streaming
            sentences = processed_response.split(". ")
            for i, sentence in enumerate(sentences):
                # Add period back unless it's the last sentence and already has punctuation
                if i < len(sentences) - 1 or not sentence.endswith((".", "!", "?")):
                    sentence = sentence + "."
                
                builder.add(sentence.strip())
                yield builder.build(separator=" ")
            
            # Update conversation state after successful response
            self.response_filter.increment_turn()
            
        except Exception as e:
            logger.error(f"Streaming failed: {e}", exc_info=True)
            builder.add("Có lỗi nhỏ, thử lại nhé!")
            yield builder.build(separator="\n")

    async def stream_analysis(self, prompt: str, thread_id_for_analysis=None, use_websocket=False):
        """
        Stream the context analysis from the LLM with optimized chunking to reduce overhead
        
        Args:
            prompt: The prompt to analyze
            thread_id_for_analysis: Thread ID to use for WebSocket analysis events
            use_websocket: Whether to use WebSocket for streaming
        """
        analysis_buffer = ""
        chunk_buffer = ""
        chunk_size_threshold = 250  # Larger chunk size reduces message count
        
        try:
            async for chunk in StreamLLM.astream(prompt):
                chunk_content = chunk.content
                analysis_buffer += chunk_content
                chunk_buffer += chunk_content
                
                # Only emit chunks when they reach a significant size or contain a natural break
                if (len(chunk_buffer) >= chunk_size_threshold or 
                    (len(chunk_buffer) > 50 and chunk_buffer.endswith((".", "!", "?", "\n")))):
                    
                    # Create analysis event with current buffer
                    analysis_event = {
                        "type": "analysis", 
                        "content": chunk_buffer, 
                        "complete": False
                    }
                    
                    # If using WebSocket and thread ID is provided, emit to that room
                    if use_websocket and thread_id_for_analysis:
                        try:
                            # Import from socketio_manager module
                            from socketio_manager import emit_analysis_event
                            emit_analysis_event(thread_id_for_analysis, analysis_event)
                        except Exception as e:
                            logger.error(f"Error in socketio_manager websocket delivery: {str(e)}")
                    
                    # Always yield for the standard flow too
                    yield analysis_event
                    
                    # Reset chunk buffer
                    chunk_buffer = ""
            
            # Send any remaining buffer content
            if chunk_buffer:
                intermediate_event = {
                    "type": "analysis",
                    "content": chunk_buffer,
                    "complete": False
                }
                
                if use_websocket and thread_id_for_analysis:
                    try:
                        from socketio_manager import emit_analysis_event
                        emit_analysis_event(thread_id_for_analysis, intermediate_event)
                    except Exception as e:
                        logger.error(f"Error in socketio_manager delivery: {str(e)}")
                        
                yield intermediate_event
            
            # Final complete message with the full analysis
            complete_event = {
                "type": "analysis", 
                "content": analysis_buffer, 
                "complete": True
            }
            
            # Send via WebSocket if configured
            if use_websocket and thread_id_for_analysis:
                try:
                    # Import from socketio_manager module
                    from socketio_manager import emit_analysis_event
                    emit_analysis_event(thread_id_for_analysis, complete_event)
                except Exception as e:
                    logger.error(f"Error in socketio_manager delivery of complete event: {str(e)}")
            
            # Always yield for standard flow
            yield complete_event
            
        except Exception as e:
            logger.error(f"Analysis streaming failed: {e}")
            # Error event
            error_event = {
                "type": "analysis", 
                "content": "Error in analysis process", 
                "complete": True, 
                "error": True
            }
            
            # Send via WebSocket if configured
            if use_websocket and thread_id_for_analysis:
                try:
                    # Import from socketio_manager module
                    from socketio_manager import emit_analysis_event
                    emit_analysis_event(thread_id_for_analysis, error_event)
                except Exception as e:
                    logger.error(f"Error in socketio_manager delivery of error event: {str(e)}")
            
            # Always yield for standard flow
            yield {"type": "analysis", "content": "Error in analysis process", "complete": True, "error": True}

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
        
        # Get WebSocket configuration
        use_websocket = state.get("use_websocket", False) or config.get("configurable", {}).get("use_websocket", False)
        thread_id_for_analysis = state.get("thread_id_for_analysis") or config.get("configurable", {}).get("thread_id_for_analysis")
        
        if not graph_version_id:
            logger.warning(f"graph_version_id is empty! State: {state}, Config: {config}")
        
        # Sync with mc.state
        self.state["graph_version_id"] = graph_version_id

        latest_msg = state["messages"][-1] if state["messages"] else HumanMessage(content="")
        latest_msg_content = latest_msg.content.strip() if isinstance(latest_msg, HumanMessage) else latest_msg.strip()
        context = "\n".join(f"User: {msg.content}" if isinstance(msg, HumanMessage) else f"AI: {msg.content}" for msg in state["messages"][-50:])

        logger.info(f"Triggering - User: {user_id}, Graph Version: {graph_version_id}, latest_msg: {latest_msg_content}, WebSocket: {use_websocket}")

        builder = ResponseBuilder()
        
        analysis_events_sent = 0
        knowledge_events_sent = 0
        
        async for response_chunk in self._handle_request(latest_msg_content, user_id, context, builder, state, 
                                                       graph_version_id=graph_version_id,
                                                       use_websocket=use_websocket,
                                                       thread_id_for_analysis=thread_id_for_analysis):
            # Check if this is a special analysis chunk type
            if isinstance(response_chunk, dict) and response_chunk.get("type") == "analysis":
                # For analysis chunks, pass them through directly
                analysis_events_sent += 1
                #logger.info(f"Yielding analysis chunk #{analysis_events_sent} from trigger: {response_chunk.get('content', '')[:50]}...")
                
                # When using WebSockets, we don't need to yield the analysis chunks through the regular flow
                # They're already sent via WebSocket, but we still yield them for proper counting and state management
                yield response_chunk
            # Check if this is a knowledge event
            elif isinstance(response_chunk, dict) and response_chunk.get("type") == "knowledge":
                # For knowledge chunks, pass them through directly
                knowledge_events_sent += 1
                logger.info(f"Yielding knowledge chunk #{knowledge_events_sent} from trigger: {len(response_chunk.get('content', []))} results")
                
                # When using WebSockets, we don't need to yield the knowledge chunks through the regular flow
                # They're already sent via WebSocket, but we still yield them for proper counting and state management
                yield response_chunk
            else:
                # For regular response chunks, update the state's prompt_str
                state["prompt_str"] = response_chunk
                logger.debug(f"Yielding from trigger: {state['prompt_str']}")
                yield response_chunk
        
        # Wrap response as AIMessage and append using add_messages
        if state["prompt_str"]:
            state["messages"] = add_messages(state["messages"], [AIMessage(content=state["prompt_str"])])
        
        self.state.update(state)
        logger.info(f"Final response: {state['prompt_str']}")
        logger.info(f"Total analysis events sent from trigger: {analysis_events_sent}")
        logger.info(f"Total knowledge events sent from trigger: {knowledge_events_sent}")
        
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
    async def detect_conversation_language(self, conversation):
        """Determine the primary language of the conversation based on history"""
        # Count language occurrences in conversation
        lang_counts = {}
        
        # Extract user messages
        user_messages = []
        for line in conversation:
            if line.startswith("User:"):
                user_message = line[5:].strip()
                if user_message:
                    user_messages.append(user_message)
        
        # If conversation is too short, return None to indicate insufficient data
        if len(user_messages) < 2:
            return None
        
        # Detect language for each message
        for message in user_messages[-3:]:  # Use last 3 messages for efficiency
            try:
                lang_info = await self.detect_language_with_llm(message)
                lang = lang_info.get("language", "Unknown")
                confidence = lang_info.get("confidence", 0)
                
                # Only count if confidence is reasonable
                if confidence > 0.6:
                    lang_counts[lang] = lang_counts.get(lang, 0) + 1
            except Exception:
                continue
        
        # Find most common language
        if lang_counts:
            primary_lang = max(lang_counts.items(), key=lambda x: x[1])[0]
            return primary_lang
        
        return "English"  # Default to English if no clear determination
        
    async def assess_knowledge_coverage(self, query_results, context_analysis, message):
        """
        Assess whether the retrieved knowledge adequately covers the user's query or context
        
        Args:
            query_results: Dictionary mapping queries to result counts
            context_analysis: The analyzed conversation context
            message: The user's message
            
        Returns:
            tuple: (coverage_score, missing_aspects, requires_feedback)
        """
        # Start with a neutral score
        coverage_score = 0.5
        
        # Check if we have any results at all
        total_results = sum(query_results.values())
        if total_results == 0:
            logger.warning(f"No knowledge results found for any query")
            return 0.1, ["complete information"], True
            
        # Extract key aspects from the context_analysis
        aspects = []
        
        # Look for specific sections in the analysis
        next_actions = re.search(r'NEXT ACTIONS?[:\s]*(.*?)(?:\n\n|\Z)', context_analysis, re.IGNORECASE | re.DOTALL)
        if next_actions:
            aspects.append(("next_actions", next_actions.group(1).strip()))
            
        requirements = re.search(r'REQUIREMENTS?[:\s]*(.*?)(?:\n\n|\Z)', context_analysis, re.IGNORECASE | re.DOTALL)
        if requirements:
            aspects.append(("requirements", requirements.group(1).strip()))
            
        # Look for rejection indicators in the message
        rejection_patterns = [
            r'\b(?:no|nope|incorrect|wrong|not right|not correct|disagree|rejected|refuse|won\'t)\b',
            r'\bdon\'t\s+(?:agree|think|believe|want|like)\b',
            r'\bthat\'s not\b',
            r'\b(?:bull|nonsense|ridiculous|crazy)\b'
        ]
        
        has_rejection = any(re.search(pattern, message.lower()) for pattern in rejection_patterns)
        
        # Check knowledge relevance for each aspect
        missing_aspects = []
        
        # If rejection detected, check if we have knowledge about handling objections
        if has_rejection:
            objection_keywords = ["objection", "rejection", "disagreement", "negative feedback"]
            has_objection_knowledge = False
            
            for query, count in query_results.items():
                if any(keyword in query.lower() for keyword in objection_keywords) and count > 0:
                    has_objection_knowledge = True
                    break
                    
            if not has_objection_knowledge:
                missing_aspects.append("handling rejection or objection")
                coverage_score -= 0.2
        
        # Generic knowledge gap assessment
        if total_results < 3:
            coverage_score -= 0.2
            
        # Check for knowledge confidence indicators in analysis
        uncertain_patterns = [
            r'\buncertain\b', r'\bunclear\b', r'\bunknown\b', r'\bmay(?:be)?\b',
            r'\bpossibly\b', r'\bperhaps\b', r'\bnot (?:sure|certain|clear)\b',
            r'\b(?:insufficient|inadequate|limited) (?:information|data|details)\b',
            r'\bcannot (?:determine|establish|ascertain)\b'
        ]
        
        uncertainty_count = 0
        for pattern in uncertain_patterns:
            matches = re.findall(pattern, context_analysis.lower())
            uncertainty_count += len(matches)
            
        if uncertainty_count > 3:
            coverage_score -= 0.15
            missing_aspects.append("clear information")
        
        # Determine if feedback is required
        requires_feedback = coverage_score < 0.3 or (has_rejection and not missing_aspects)
        
        # If we can't identify specific missing aspects but coverage is low
        if requires_feedback and not missing_aspects:
            missing_aspects.append("relevant information")
            
        return coverage_score, missing_aspects, requires_feedback
    
    async def generate_feedback_request(self, missing_aspects, message, context):
        """
        Generate a natural request for more information based on missing knowledge aspects
        
        Args:
            missing_aspects: List of missing knowledge aspects
            message: The user's message
            context: Conversation context
            
        Returns:
            str: A natural language request for clarification
        """
        # Map aspect types to question templates
        aspect_questions = {
            "handling rejection or objection": [
                "Could you tell me more about what specifically doesn't work for you?",
                "I'd like to understand better what you disagree with. Could you elaborate?",
                "What part of that didn't address your needs?",
                "To better help you, could you share what aspect you're rejecting?"
            ],
            "clear information": [
                "I'm not sure I have enough information. Could you provide more details?",
                "To give you the best response, could you clarify what you're looking for?",
                "I'd like to help, but need a bit more context. Could you elaborate?",
                "I'm missing some details to properly assist you. What specifically are you interested in?"
            ],
            "relevant information": [
                "I don't seem to have the specific information you need. Could you rephrase or clarify?",
                "I want to make sure I address your question properly. Could you provide more details?",
                "I'd like to help with that, but I need a bit more context. Could you elaborate?",
                "To give you the best answer, could you share more about what you're looking for?"
            ],
            "complete information": [
                "I don't have enough information about that topic. Could you share what specific aspects you're interested in?",
                "I'd like to help with that, but it seems I need more information. What exactly would you like to know?",
                "I don't have complete details on that. Could you clarify what you're looking for?",
                "I want to make sure I understand correctly. Could you provide more details about your question?"
            ]
        }
        
        # Build a response
        if not missing_aspects:
            missing_aspects = ["relevant information"]
            
        # Select the first missing aspect as the primary one
        primary_aspect = missing_aspects[0]
        
        # Get question templates for this aspect
        templates = aspect_questions.get(primary_aspect, aspect_questions["relevant information"])
        
        # Select a template based on a hash of the message (for variety)
        template_index = hash(message) % len(templates)
        question = templates[template_index]
        
        # If the message is very short, add encouragement for more details
        if len(message.strip()) < 10:
            follow_up = "Even a few more details would help me provide a better response."
            question = f"{question} {follow_up}"
            
        return question

    async def _handle_request(self, message: str, user_id: str, context: str, builder: "ResponseBuilder", state: Dict, graph_version_id: str = "", use_websocket=False, thread_id_for_analysis=None):
        try:
            # Initialize cache if not exists
            if not hasattr(self, '_cache'):
                self._cache = {
                    'personality': {},
                    'language': {},
                    'knowledge': {},
                    'analysis': {},
                    'query_embeddings': {}  # Add cache for query embeddings
                }

            # Load personality if not already loaded, if graph_version_id changed, or if we have it cached
            personality_loaded = (hasattr(self, 'personality_instructions') and self.personality_instructions and 
                                 hasattr(self.personality_manager, 'loaded_graph_version_id') and 
                                 self.personality_manager.loaded_graph_version_id == graph_version_id)

            # Check if we have this personality in cache
            cache_key = f"personality_{graph_version_id}"
            personality_in_cache = cache_key in self._cache.get('personality', {})

            # Log detailed personality status
            logger.info(f"[PERSONALITY] Check load status - has_attr: {hasattr(self, 'personality_instructions')}, " +
                       f"loaded_gvid: {getattr(self.personality_manager, 'loaded_graph_version_id', None)}, " +
                       f"current_gvid: {graph_version_id}, " +
                       f"in_cache: {personality_in_cache}, " +
                       f"is_loaded_from_kb: {getattr(self.personality_manager, 'is_loaded_from_knowledge', False)}")
                                 
            if not personality_loaded or personality_in_cache:
                logger.info(f"[PERSONALITY] Loading personality for graph_version_id: {graph_version_id}, from_cache: {personality_in_cache}")
                await self.load_personality_instructions(graph_version_id)
                self.state["graph_version_id"] = graph_version_id
            else:
                logger.info(f"[PERSONALITY] Personality instructions already loaded for graph_version_id: {graph_version_id}")
                # Log the already loaded personality for debugging
                if hasattr(self, 'personality_instructions') and self.personality_instructions:
                    logger.info(f"[PERSONALITY] Current loaded personality (first 500 chars): {self.personality_instructions[:500]}...")
                    logger.info(f"[PERSONALITY] Current AI name: {self.name}")

            # Break conversation into lines
            conversation = [line.strip() for line in context.split("\n") if line.strip()]
            
            logger.info(f"Processing message: '{message}' from user_id: {user_id}")
            
            # Run language detection and context analysis in parallel
            language_task = self.detect_language_with_llm(message)
            conversation_language_task = self.detect_conversation_language(conversation)
            
            # Get profile building skills from knowledge base in parallel
            profile_query = "contact profile building information gathering customer understanding"
            profile_task = query_graph_knowledge(graph_version_id, profile_query, top_k=5)
            
            # IMPROVEMENT 3: Enhance response naturalness
            naturalness_task = self.enhance_response_naturalness(context)
            
            # Wait for all parallel tasks
            lang_info, conversation_language, profile_entries,naturalness_guidance = await asyncio.gather(
                language_task,
                conversation_language_task,
                profile_task,
                naturalness_task
            )
            
            # Override language confidence if conversation history establishes a pattern
            if conversation_language and conversation_language != lang_info["language"] and len(conversation) > 5:
                logger.info(f"Overriding detected language ({lang_info['language']}) with conversation language ({conversation_language})")
                lang_info["language"] = conversation_language
                lang_info["confidence"] = 0.9
                lang_info["responseGuidance"] = f"Respond in {conversation_language}, maintaining consistency with previous messages"
            
            # Add language code to lang_info if not present
            if "code" not in lang_info:
                # Map common language names to ISO codes
                language_to_code = {
                    "english": "en",
                    "vietnamese": "vi",
                    "malay": "ms",
                    "indonesian": "id",
                    "chinese": "zh",
                    "french": "fr",
                    "spanish": "es"
                }
                # Default to English if language not in mapping
                lang_info["code"] = language_to_code.get(lang_info["language"].lower(), "en")
                logger.info(f"Added language code: {lang_info['code']} for {lang_info['language']}")
            
            # IMPROVEMENT 1: Set culture code for selective cultural adaptation
            culture_code = "default"
            if lang_info["language"].lower() == "vietnamese":
                culture_code = "vi"
            elif lang_info["language"].lower() == "english":
                culture_code = "en"
            
            # Process profile instructions
            profile_instructions = "\n\n".join(entry["raw"] for entry in profile_entries)
            process_instructions = profile_instructions
            
            # STEP 1: Build and process initial context analysis (parts 1-3)
            context_analysis_prompt = build_context_analysis_prompt(context, process_instructions)
            
            # Run initial analysis
            logger.info("Starting initial analysis (parts 1-3)")
            # Use stream_analysis from analysis.py which has important post-processing logic
            from analysis import stream_analysis
            initial_analysis_task = stream_analysis(context_analysis_prompt, thread_id_for_analysis, use_websocket)
            
            # Process initial analysis results
            context_analysis = ""
            analysis_result = None
            initial_search_terms = []
            
            # Collect results from initial analysis
            async for analysis_chunk in initial_analysis_task:
                # Pass through all chunks to caller
                yield analysis_chunk
                
                # Store the analysis content when it's complete
                if analysis_chunk.get("type") == "analysis" and analysis_chunk.get("complete", False):
                    context_analysis = analysis_chunk.get("content", "")
                    # Process the analysis result to extract search terms
                    logger.info(f"[DEBUG] Processing initial analysis with length: {len(context_analysis)}")
                    analysis_result = process_analysis_result(context_analysis)
                    initial_search_terms = analysis_result.get("search_terms", [])
                    logger.info(f"[DEBUG] Received complete initial analysis, extracted {len(initial_search_terms)} search terms: {initial_search_terms}")
            
            # STEP 2: Use search terms from initial analysis for knowledge queries
            logger.info("Using search terms from initial analysis for knowledge retrieval")
            search_terms = initial_search_terms
            
            # If we couldn't extract any search terms, use the message as fallback
            if not search_terms:
                search_terms = [message]
                logger.info(f"[DEBUG] No search terms extracted, using message as fallback search term: {message}")
            
            # Also add the message itself as a search term if short enough
            if len(message.split()) <= 10 and message not in search_terms:
                search_terms.append(message)
                logger.info(f"[DEBUG] Added original message as additional search term: {message}")
                
            # Pre-filter queries based on relevance
            filtered_queries = []
            seen_terms = set()  # For deduplication
            
            for term in search_terms:
                # Skip very short or common terms and deduplicate
                normalized_term = term.lower().strip()
                if (len(normalized_term) > 3 and
                    normalized_term not in seen_terms and
                    not normalized_term in ['the', 'and', 'or', 'but', 'for', 'with', 'that']):
                    seen_terms.add(normalized_term)
                    filtered_queries.append(term)
            
            logger.info(f"[DEBUG] After filtering, {len(filtered_queries)} search terms remain: {filtered_queries}")
            
            # Batch process queries with vector search
            unique_queries = []
            cache_hits = 0
            batch_size = 5  # Process queries in batches

            # Check cache for each query
            for query in filtered_queries:
                cache_key = f"{graph_version_id}:{query}"
                if cache_key in self._cache['knowledge']:
                    cache_hits += 1
                    continue
                unique_queries.append(query)

            # Process queries in batches
            all_knowledge = []
            for i in range(0, len(unique_queries), batch_size):
                batch = unique_queries[i:i + batch_size]
                
                # Use vector search for batch processing
                try:
                    # Call with streaming and thread ID if websocket is enabled
                    if use_websocket:
                        batch_results = []
                        async for knowledge_event in self._stream_batch_query_knowledge(
                            graph_version_id, batch, top_k=3, thread_id=thread_id_for_analysis
                        ):
                            # Pass through knowledge events to trigger method
                            yield knowledge_event
                            
                            # Extract content from non-complete events (contains actual knowledge)
                            if not knowledge_event.get("complete", False) and knowledge_event.get("content"):
                                logger.info(f"[DEBUG] Received knowledge event with {len(knowledge_event.get('content', []))} results")
                                for result in knowledge_event.get("content", []):
                                    all_knowledge.append(result)
                                    logger.info(f"[DEBUG] Added knowledge result: {result.get('id', 'unknown-id')[:10]}...")
                            
                            # If this is the final event with complete flag, save the results
                            if knowledge_event.get("complete") and not knowledge_event.get("error"):
                                # The complete event doesn't include results, as they've been streamed
                                # Results should be saved via standard method from other events
                                logger.info(f"[DEBUG] Received complete knowledge event")
                                pass
                    else:
                        # Normal non-streaming call
                        batch_results = await self._batch_query_knowledge(graph_version_id, batch)
                        logger.info(f"[DEBUG] Received batch results with {sum(len(results) for results in batch_results)} total items")
                        # Update cache with new results
                        for query, results in zip(batch, batch_results):
                            cache_key = f"{graph_version_id}:{query}"
                            self._cache['knowledge'][cache_key] = results
                            all_knowledge.extend(results)
                except Exception as e:
                    logger.error(f"Error in batch processing: {str(e)}")
                    # Fallback to individual queries
                    for query in batch:
                        try:
                            results = await query_graph_knowledge(graph_version_id, query, top_k=3)
                            cache_key = f"{graph_version_id}:{query}"
                            self._cache['knowledge'][cache_key] = results
                            all_knowledge.extend(results)
                        except Exception as e:
                            logger.error(f"Error processing query '{query}': {str(e)}")

            # Combine cached and new results
            logger.info(f"[DEBUG] all_knowledge has {len(all_knowledge)} items before adding cached results")
            for query in filtered_queries:
                cache_key = f"{graph_version_id}:{query}"
                if cache_key in self._cache['knowledge']:
                    cached_results = self._cache['knowledge'][cache_key]
                    logger.info(f"[DEBUG] Adding {len(cached_results)} cached results for query '{query}'")
                    all_knowledge.extend(cached_results)

            # Remove duplicates using a more efficient method
            logger.info(f"[DEBUG] all_knowledge has {len(all_knowledge)} items before deduplication")
            unique_knowledge = []
            seen_ids = set()
            for entry in all_knowledge:
                entry_id = entry.get('id', 'unknown')
                if entry_id not in seen_ids:
                    seen_ids.add(entry_id)
                    unique_knowledge.append(entry)
                    logger.info(f"[DEBUG] Added unique knowledge entry: {entry_id[:10]}...")
                else:
                    logger.info(f"[DEBUG] Skipped duplicate knowledge entry: {entry_id[:10]}...")

            # Create knowledge context
            knowledge_context = "\n\n".join(entry.get("raw", "") for entry in unique_knowledge)
            logger.info(f"[DEBUG] Final unique_knowledge has {len(unique_knowledge)} items with IDs: {[entry.get('id', 'unknown')[:8] for entry in unique_knowledge]}")
            logger.info(f"Retrieved {len(unique_knowledge)} unique knowledge entries")
            logger.info(f"Knowledge context: {knowledge_context[:200]}..." if len(knowledge_context) > 200 else f"Knowledge context: {knowledge_context}")
            
            # STEP 3: Generate next actions using the retrieved knowledge
            logger.info("Generating next actions with retrieved knowledge")
            next_actions_prompt = build_next_actions_prompt(context, context_analysis, knowledge_context)
            
            # Run next actions generation using the dedicated stream_next_action function
            from analysis import stream_next_action
            next_actions_task = stream_next_action(next_actions_prompt, thread_id_for_analysis, use_websocket)
            next_action_content = ""
            next_action_error = False
            
            # Process next actions results
            async for next_action_chunk in next_actions_task:
                # Pass through all chunks to caller
                yield next_action_chunk
                
                # Check for error flag
                if next_action_chunk.get("error", False):
                    next_action_error = True
                    logger.warning(f"Received error in next action: {next_action_chunk.get('content', '')}")
                
                # Store the complete next action content
                if next_action_chunk.get("complete", False) and next_action_chunk.get("content"):
                    next_action_content = next_action_chunk.get("content", "")
                    logger.info(f"Received complete next actions, length: {len(next_action_content)}")
            
            # Now we should have:
            # 1. context_analysis - from the initial analysis (parts 1-3)
            # 2. next_action_content - from the next actions generation
            
            # STEP 3.5: Extract search terms from next actions and retrieve more targeted knowledge
            # Only proceed if we have valid next_action_content without errors
            additional_knowledge = []
            
            if not next_action_error and next_action_content and len(next_action_content) > 50:
                try:
                    logger.info("Extracting content from next actions for targeted knowledge retrieval")
                    from analysis import extract_search_terms_from_next_actions, process_next_actions_result
                    
                    # First, process the next_action_content to get structured data
                    next_actions_data = process_next_actions_result(next_action_content)
                    
                    # Make sure we have valid data and handle potential errors
                    if not next_actions_data or not isinstance(next_actions_data, dict):
                        logger.warning(f"Invalid next_actions_data format: {next_actions_data}")
                        next_actions_data = {"next_action_english": "", "next_action_vietnamese": "", "next_action_full": next_action_content}
                    
                    # Check if unattributed questions were detected
                    if "warning" in next_actions_data and next_actions_data.get("unattributed_questions", []):
                        logger.warning(f"DETECTED UNATTRIBUTED QUESTIONS IN NEXT ACTIONS: {next_actions_data['warning']}")
                        logger.warning(f"Unattributed questions: {next_actions_data['unattributed_questions']}")
                        
                        # If we have unattributed questions, modify the next_actions content to indicate this
                        english_next_actions = next_actions_data.get("next_action_english", "")
                        if english_next_actions:
                            warning_note = "\n\nNOTE: Some suggested questions may not directly come from knowledge content and should be reviewed."
                            next_actions_data["next_action_english"] = english_next_actions + warning_note
                            next_actions_data["next_action_full"] = next_actions_data["next_action_full"] + warning_note
                    
                    # Extract the English next actions
                    english_next_actions = next_actions_data.get("next_action_english", "")
                    
                    if english_next_actions:
                        logger.info(f"Using English next actions for knowledge queries: {english_next_actions[:100]}...")
                        
                        # Split the English next actions into sentences for more contextual queries
                        import re
                        # Split by sentence boundaries while preserving punctuation
                        sentences = re.findall(r'[^.!?]+[.!?]', english_next_actions)
                        
                        # Filter sentences to avoid very short ones
                        filtered_sentences = []
                        for sentence in sentences:
                            sentence = sentence.strip()
                            # Only include meaningful sentences (more than 5 words and not just a heading)
                            if len(sentence.split()) > 5 and not all(c.isupper() for c in sentence if c.isalpha()):
                                filtered_sentences.append(sentence)
                                logger.info(f"[DEBUG] Added next action sentence as query: {sentence}")
                        
                        # If we have bullet points, those might not end with periods
                        bullet_points = re.findall(r'[-•*]\s*[^-•*\n]+', english_next_actions)
                        for point in bullet_points:
                            point = point.strip()
                            if len(point.split()) > 5 and point not in filtered_sentences:
                                filtered_sentences.append(point)
                                logger.info(f"[DEBUG] Added bullet point as query: {point}")
                        
                        # Add quoted questions (these are important as they're directly from knowledge)
                        quoted_questions = re.findall(r'"([^"]*\?)"', english_next_actions)
                        for question in quoted_questions:
                            if question not in filtered_sentences and len(question.split()) > 3:
                                filtered_sentences.append(question)
                                logger.info(f"[DEBUG] Added quoted question as query: {question}")
                        
                        # Extract instructions and actions (imperative sentences)
                        action_patterns = [
                            r'(Ask about [^.!?]+[.!?])',
                            r'(Inquire about [^.!?]+[.!?])',
                            r'(Explain [^.!?]+[.!?])',
                            r'(Suggest [^.!?]+[.!?])',
                            r'(Recommend [^.!?]+[.!?])',
                            r'(Provide [^.!?]+[.!?])',
                            r'(Clarify [^.!?]+[.!?])',
                            r'(Address [^.!?]+[.!?])',
                            r'(Acknowledge [^.!?]+[.!?])',
                            r'(Follow up [^.!?]+[.!?])'
                        ]
                        
                        for pattern in action_patterns:
                            action_sentences = re.findall(pattern, english_next_actions)
                            for action in action_sentences:
                                if action not in filtered_sentences and len(action.split()) > 3:
                                    filtered_sentences.append(action)
                                    logger.info(f"[DEBUG] Added action sentence as query: {action}")
                        
                        logger.info(f"Extracted {len(filtered_sentences)} sentences from next actions")
                        
                        # Use these sentences directly as queries without further processing
                        next_action_queries = filtered_sentences
                        
                        # Deduplicate but keep the original sentence structure
                        unique_queries = []
                        seen_normalized = set()
                        
                        for query in next_action_queries:
                            # Normalize for comparison but keep original for querying
                            normalized = query.lower().strip()
                            if normalized not in seen_normalized:
                                seen_normalized.add(normalized)
                                unique_queries.append(query)
                        
                        logger.info(f"Using {len(unique_queries)} unique contextual queries from next actions")
                        
                        # Retrieve additional knowledge based on these contextual queries
                        if unique_queries:
                            logger.info(f"Retrieving additional knowledge based on {len(unique_queries)} next action queries")
                            
                            # Process next action queries in batches
                            batch_size = 3  # Smaller batch size for longer queries
                            for i in range(0, len(unique_queries), batch_size):
                                batch = unique_queries[i:i + batch_size]
                                
                                # Use vector search for batch processing
                                try:
                                    # Call with streaming and thread ID if websocket is enabled
                                    if use_websocket:
                                        next_action_batch_results = []
                                        async for knowledge_event in self._stream_batch_query_knowledge(
                                            graph_version_id, batch, top_k=3, thread_id=thread_id_for_analysis
                                        ):
                                            # Pass through knowledge events to trigger method
                                            yield knowledge_event
                                            
                                            # Extract content from non-complete events (contains actual knowledge)
                                            if not knowledge_event.get("complete", False) and knowledge_event.get("content"):
                                                # Log which queries got results
                                                for result in knowledge_event.get("content", []):
                                                    query_idx = result.get("query_idx", -1)
                                                    query = batch[query_idx] if 0 <= query_idx < len(batch) else "unknown"
                                                    logger.info(f"[NEXT_ACTION_PINECONE_QUERY] Streaming sentence: \"{query}\" → Retrieved knowledge: {result.get('id', 'unknown-id')[:10]}")
                                                    
                                                    if result not in additional_knowledge:
                                                        additional_knowledge.append(result)
                                                        logger.info(f"[DEBUG] Added knowledge result from next action sentence: {result.get('id', 'unknown-id')[:10]}...")
                                    else:
                                        # Normal non-streaming call
                                        next_action_batch_results = await self._batch_query_knowledge(graph_version_id, batch)
                                        logger.info(f"[DEBUG] Received batch results with {sum(len(results) for results in next_action_batch_results)} total items")
                                        for batch_idx, results in enumerate(next_action_batch_results):
                                            query = batch[batch_idx]
                                            logger.info(f"[NEXT_ACTION_PINECONE_QUERY] Sentence: \"{query}\" → Retrieved {len(results)} knowledge results")
                                            for result in results:
                                                additional_knowledge.append(result)
                                                logger.info(f"[DEBUG] Added knowledge result from next action sentence: {result.get('id', 'unknown-id')[:10]}...")
                                except Exception as e:
                                    logger.error(f"Error in next action batch processing: {str(e)}")
                                    import traceback
                                    logger.error(traceback.format_exc())
                    else:
                        logger.warning("No English next actions found for knowledge queries")
                except Exception as e:
                    logger.error(f"Error processing next actions for knowledge queries: {str(e)}")
                    import traceback
                    logger.error(traceback.format_exc())
            else:
                if next_action_error:
                    logger.warning("Skipping next action knowledge queries due to error in next_action generation")
                elif not next_action_content:
                    logger.warning("Skipping next action knowledge queries due to empty next_action_content")
                else:
                    logger.warning(f"Skipping next action knowledge queries due to short content: {len(next_action_content)} chars")
            
            # Process any additional knowledge found
            if additional_knowledge:
                # Remove duplicates with original knowledge
                unique_additional_knowledge = []
                seen_ids = set(entry['id'] for entry in unique_knowledge)
                
                for entry in additional_knowledge:
                    if entry['id'] not in seen_ids:
                        seen_ids.add(entry['id'])
                        unique_additional_knowledge.append(entry)
                
                # Add new unique knowledge to the knowledge context
                if unique_additional_knowledge:
                    additional_knowledge_context = "\n\n".join(entry["raw"] for entry in unique_additional_knowledge)
                    knowledge_context += "\n\n" + additional_knowledge_context
                    logger.info(f"Added {len(unique_additional_knowledge)} additional knowledge entries from next action queries")
            
            # STEP 3: Build Prompt
            # Generate ultra-simplified response prompt
            response_prompt = (
                f"AI: {self.name}\n"
                f"Message: '{message}'\n"
                f"Context: {context}\n"
                f"Context Analysis: {context_analysis}\n"
                f"NEXT ACTIONS: {next_action_content}\n"
                f"Knowledge: {knowledge_context}\n"
                f"Language: {lang_info['language']}\n\n"
                f"PERSONALITY: {self.personality_instructions}\n\n"
                f"TASK:\n"
                f"Create a sophisticated response that precisely executes the NEXT ACTIONS while applying the KNOWLEDGE and sounding completely natural:\n\n"
                f"1. CAREFUL REVIEW: Understand exactly what stage the conversation is in and what specific action NEXT ACTIONS recommends.\n\n"
                f"2. APPLY KNOWLEDGE: Directly incorporate relevant information from KNOWLEDGE to support your response. Always ground your answers in the provided knowledge. If there's specific data, facts, skills, products and service informInformation or guidance in the knowledge section, use it explicitly.\n\n"
                f"3. AUTHENTIC VOICE: Use your own natural voice rather than formulaic phrases or templates. Avoid sounding like you're following a script.\n\n"
                f"4. LANGUAGE-SPECIFIC CONVERSATION PATTERNS: Use casual, warm conversation patterns that are authentic and natural to {lang_info['language']}. Adapt your relationship terms, address forms, and speech patterns to match how native speakers would naturally communicate in this context.\n\n"
                f"5. NATURAL BEGINNINGS AND ENDINGS:\n"
                f"   - Start with warm, casual greetings appropriate to the language and relationship\n"
                f"   - End naturally without asking for feedback or using formulaic closings\n"
                f"   - Never reintroduce yourself after the initial introduction\n\n"
                f"6. PRECISE EXECUTION: Execute what the NEXT ACTIONS specifies without deviation or unnecessary additions, but always integrate relevant KNOWLEDGE.\n\n"
                f"7. CULTURAL ATTUNEMENT: Draw on natural conversation patterns native to {lang_info['language']}, including culturally appropriate address forms, conversation rhythm, and communication norms. Reflect the natural speaking style of the language.\n\n"
                f"Your response should feel like a warm, casual conversation with a knowledgeable professional in the user's native language, while precisely following the NEXT ACTIONS guidance and incorporating the provided KNOWLEDGE.\n"
            )
            
            # Check if knowledge was found
            knowledge_found = bool(knowledge_context and knowledge_context.strip())
            
            # Log the knowledge context for debugging
            logger.info(f"[KNOWLEDGE_CONTEXT] For message: '{message[:100]}...' (if longer)")
            if knowledge_context:
                logger.info(f"[KNOWLEDGE_CONTEXT] Content: {knowledge_context[:1500]}..." if len(knowledge_context) > 1500 else f"[KNOWLEDGE_CONTEXT] Content: {knowledge_context}")
            else:
                logger.info(f"[KNOWLEDGE_CONTEXT] No knowledge context found")
            
            async for _ in self.stream_response(response_prompt, builder, knowledge_found):
                yield builder.build()
                
        except Exception as e:
            logger.error(f"Error handling request: {str(e)}", exc_info=True)
            builder.add("Oops, có lỗi khi tìm thông tin, thử lại nhé!")
            yield builder.build()

    async def load_personality_instructions(self, graph_version_id: str = ""):
        """
        Load personality instructions using the PersonalityManager with caching.
        """
        # Check if we have this personality in cache
        cache_key = f"personality_{graph_version_id}"
        
        if cache_key in self._cache.get('personality', {}):
            cached_data = self._cache['personality'][cache_key]
            # Apply cached personality data
            self.personality_manager.personality_instructions = cached_data.get('instructions', '')
            self.personality_manager.name = cached_data.get('name', 'Ami')
            self.personality_manager.instincts = cached_data.get('instincts', {})
            self.personality_manager.is_loaded_from_knowledge = True
            self.personality_manager.loaded_graph_version_id = graph_version_id
            
            logger.info(f"[PERSONALITY] Loaded personality from cache for graph_version_id: {graph_version_id}")
            logger.info(f"[PERSONALITY] Cached AI name: {self.personality_manager.name}")
            
            return self.personality_manager.personality_instructions
        
        # Not in cache, load from knowledge base
        await self.personality_manager.load_personality_instructions(graph_version_id)
        
        # Cache the personality for future use
        self._cache.setdefault('personality', {})[cache_key] = {
            'instructions': self.personality_manager.personality_instructions,
            'name': self.personality_manager.name,
            'instincts': self.personality_manager.instincts,
            'timestamp': time.time()
        }
        
        logger.info(f"[PERSONALITY] Cached personality for graph_version_id: {graph_version_id}")
        
        return self.personality_manager.personality_instructions

    async def _batch_query_knowledge(self, graph_version_id: str, queries: List[str], top_k: int = 3, should_stream: bool = False, thread_id: str = None):
        """
        Batch process multiple queries using vector search for efficiency.
        For streaming, use _stream_batch_query_knowledge instead.
        
        Args:
            graph_version_id: The version ID of the knowledge graph
            queries: List of queries to process
            top_k: Number of results to return per query
            should_stream: DEPRECATED - Use _stream_batch_query_knowledge for streaming
            thread_id: DEPRECATED - Use _stream_batch_query_knowledge for streaming
            
        Returns:
            List[List[Dict]]: Results for each query
        """
        # Handle streaming mode by delegating to the streaming-specific function
        if should_stream:
            logger.warning("Using should_stream=True with _batch_query_knowledge is deprecated. Use _stream_batch_query_knowledge instead.")
            # Use a list to collect all the streamed results
            final_results = [[] for _ in range(len(queries))]
            async for event in self._stream_batch_query_knowledge(graph_version_id, queries, top_k, thread_id):
                # Process the event if needed
                pass
            # Return the collected results from streaming
            return final_results

        # Early return for empty queries
        if not queries:
            logger.warning("[KNOWLEDGE_DEBUG] Empty queries list provided to _batch_query_knowledge")
            return []
        
        logger.info(f"[KNOWLEDGE_DEBUG] Starting batch query for {len(queries)} queries: {queries[:5]}{'...' if len(queries) > 5 else ''}")
        
        try:
            # Get brain banks once for all queries
            brain_banks = await get_version_brain_banks(graph_version_id)
            
            if not brain_banks:
                logger.warning(f"[KNOWLEDGE_DEBUG] No brain banks found for graph version {graph_version_id}")
                return [[] for _ in queries]  # Return empty results for all queries
            
            logger.info(f"[KNOWLEDGE_DEBUG] Found {len(brain_banks)} brain banks for graph version {graph_version_id}")
            
            # Generate all embeddings in parallel (with caching)
            logger.info(f"[KNOWLEDGE_DEBUG] Generating embeddings for {len(queries)} queries")
            embedding_tasks = [get_cached_embedding(query) for query in queries]
            embeddings = await asyncio.gather(*embedding_tasks)
            
            # Create a mapping of query index to embedding
            query_embeddings = {i: embedding for i, embedding in enumerate(embeddings)}
            logger.info(f"[KNOWLEDGE_DEBUG] Successfully generated {len(query_embeddings)} embeddings")
            
            # Process each brain bank in parallel, but send all queries at once
            all_results = [[] for _ in range(len(queries))]  # Initialize result containers
            
            async def process_brain_bank(brain_bank):
                bank_name = brain_bank["bank_name"]
                brain_id = brain_bank["id"]
                
                logger.info(f"[KNOWLEDGE_DEBUG] Processing brain bank: {bank_name} (ID: {brain_id})")
                
                # Import necessary function
                from database import query_brain_with_embeddings_batch
                
                # Send all embeddings to this brain bank at once
                brain_results = await query_brain_with_embeddings_batch(
                    query_embeddings, bank_name, brain_id, top_k
                )
                
                # Log results for this brain bank
                result_counts = {query_idx: len(results) for query_idx, results in brain_results.items()}
                logger.info(f"[KNOWLEDGE_DEBUG] Brain {bank_name} results: {result_counts}")
                
                # Merge results
                for query_idx, results in brain_results.items():
                    # We ensure query_idx is a valid index in all_results
                    if 0 <= query_idx < len(all_results):
                        # Add all results to the full results list
                        all_results[query_idx].extend(results)
                        
                        # Log score ranges to help debug potential filtering issues
                        if results:
                            scores = [result.get("score", 0) for result in results]
                            logger.info(f"[KNOWLEDGE_DEBUG] Query {query_idx} ({queries[query_idx][:30]}...) scores in {bank_name}: min={min(scores):.4f}, max={max(scores):.4f}, count={len(scores)}")
                    else:
                        logger.warning(f"[KNOWLEDGE_DEBUG] Received invalid query_idx {query_idx} from brain {bank_name}")
            
            # Process all brain banks in parallel
            await asyncio.gather(*[process_brain_bank(brain_bank) for brain_bank in brain_banks])
            
            # Log detailed results after processing all brain banks
            for query_idx in range(len(queries)):
                result_count = len(all_results[query_idx])
                logger.info(f"[KNOWLEDGE_DEBUG] After merging, query {query_idx} ({queries[query_idx][:30]}...) has {result_count} results")
            
            # Sort results by score and trim to top_k
            final_results = []
            for query_idx in range(len(queries)):
                results = all_results[query_idx]
                if results:  # Simplified check, just see if the list has any items
                    # Sort by score and take top results
                    sorted_results = sorted(
                        results,
                        key=lambda x: x.get("score", 0),
                        reverse=True
                    )[:top_k]
                    
                    # Log filtering effect
                    original_count = len(results)
                    final_count = len(sorted_results)
                    logger.info(f"[KNOWLEDGE_DEBUG] Query {query_idx} ({queries[query_idx][:30]}...): filtered from {original_count} to {final_count} results")
                    
                    # Log score details
                    if sorted_results:
                        scores = [result.get("score", 0) for result in sorted_results]
                        logger.info(f"[KNOWLEDGE_DEBUG] Final scores for query {query_idx}: min={min(scores):.4f}, max={max(scores):.4f}")
                    
                    final_results.append(sorted_results)
                else:
                    logger.warning(f"[KNOWLEDGE_DEBUG] Query {query_idx} ({queries[query_idx][:30]}...) has NO results after processing")
                    final_results.append([])
            
            # Count total results
            total_results = sum(len(results) for results in final_results)
            empty_queries = sum(1 for results in final_results if not results)
            
            logger.info(f"[KNOWLEDGE_DEBUG] Batch processing complete: {total_results} total results, {empty_queries}/{len(queries)} queries with no results")
            
            # Warning if no results found for any query
            if total_results == 0:
                logger.warning("[KNOWLEDGE_DEBUG] No knowledge results found for any query")
            
            return final_results
            
        except Exception as e:
            logger.error(f"[KNOWLEDGE_DEBUG] Error in batch query processing: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Fallback to standard parallel execution
            logger.info("[KNOWLEDGE_DEBUG] Falling back to individual query processing")
            return await asyncio.gather(*[
                query_graph_knowledge(graph_version_id, query, top_k)
                for query in queries
            ])

    async def _stream_batch_query_knowledge(self, graph_version_id: str, queries: List[str], top_k: int = 3, thread_id: str = None):
        """
        Stream batch query knowledge results for real-time updates.
        This is an async generator function that yields results as they're found.
        
        Args:
            graph_version_id: The version ID of the knowledge graph
            queries: List of queries to process
            top_k: Number of results to return per query
            thread_id: Thread ID for WebSocket delivery of streamed results
            
        Yields:
            Dict events with streaming knowledge results
        """
        try:
            # Early return for empty queries
            if not queries:
                logger.warning("[KNOWLEDGE_DEBUG] Empty queries list provided to _stream_batch_query_knowledge")
                yield {"type": "knowledge", "content": [], "complete": True}
                return
            
            logger.info(f"[KNOWLEDGE_DEBUG] Starting batch query for {len(queries)} queries: {queries[:5]}{'...' if len(queries) > 5 else ''}")
            
            # Get brain banks once for all queries
            brain_banks = await get_version_brain_banks(graph_version_id)
            
            if not brain_banks:
                logger.warning(f"[KNOWLEDGE_DEBUG] No brain banks found for graph version {graph_version_id}")
                yield {"type": "knowledge", "content": [], "complete": True}
                return
            
            logger.info(f"[KNOWLEDGE_DEBUG] Found {len(brain_banks)} brain banks for graph version {graph_version_id}")
            
            # Generate all embeddings in parallel (with caching)
            logger.info(f"[KNOWLEDGE_DEBUG] Generating embeddings for {len(queries)} queries")
            embedding_tasks = [get_cached_embedding(query) for query in queries]
            embeddings = await asyncio.gather(*embedding_tasks)
            
            # Create a mapping of query index to embedding
            query_embeddings = {i: embedding for i, embedding in enumerate(embeddings)}
            logger.info(f"[KNOWLEDGE_DEBUG] Successfully generated {len(query_embeddings)} embeddings")
            
            # Process each brain bank in parallel, but send all queries at once
            all_results = [[] for _ in range(len(queries))]  # Initialize result containers
            
            # For streaming: track already streamed knowledge to avoid duplicates
            streamed_ids = set()
            
            # Create a list to collect all results for streaming
            streaming_results = []
            
            # Streaming version - uses generators
            async def process_brain_bank_streaming(brain_bank):
                bank_name = brain_bank["bank_name"]
                brain_id = brain_bank["id"]
                
                logger.info(f"[KNOWLEDGE_DEBUG] Processing brain bank: {bank_name} (ID: {brain_id})")
                
                # Import necessary function
                from database import query_brain_with_embeddings_batch
                
                # Send all embeddings to this brain bank at once
                brain_results = await query_brain_with_embeddings_batch(
                    query_embeddings, bank_name, brain_id, top_k
                )
                
                # Log results for this brain bank
                result_counts = {query_idx: len(results) for query_idx, results in brain_results.items()}
                logger.info(f"[KNOWLEDGE_DEBUG] Brain {bank_name} results: {result_counts}")
                
                # Prepare new results to stream
                new_results_to_stream = []
                
                # Merge results
                for query_idx, results in brain_results.items():
                    # We ensure query_idx is a valid index in all_results
                    if 0 <= query_idx < len(all_results):
                        # Collect new results for streaming
                        for result in results:
                            if result['id'] not in streamed_ids:
                                streamed_ids.add(result['id'])
                                # Add query info to help frontend organize results
                                result_with_query = result.copy()
                                result_with_query['query'] = queries[query_idx]
                                result_with_query['query_idx'] = query_idx
                                new_results_to_stream.append(result_with_query)
                        
                        # Add all results to the full results list
                        all_results[query_idx].extend(results)
                        
                        # Log score ranges to help debug potential filtering issues
                        if results:
                            scores = [result.get("score", 0) for result in results]
                            logger.info(f"[KNOWLEDGE_DEBUG] Query {query_idx} ({queries[query_idx][:30]}...) scores in {bank_name}: min={min(scores):.4f}, max={max(scores):.4f}, count={len(scores)}")
                    else:
                        logger.warning(f"[KNOWLEDGE_DEBUG] Received invalid query_idx {query_idx} from brain {bank_name}")
                
                # Return the results to be streamed (no longer a generator)
                return new_results_to_stream
            
            # Process all brain banks in parallel, but now with regular coroutines
            brain_bank_tasks = [process_brain_bank_streaming(brain_bank) for brain_bank in brain_banks]
            
            # Use asyncio.gather instead of as_completed for simplicity
            brain_bank_results = await asyncio.gather(*brain_bank_tasks)
            
            # Process and yield each batch of results
            for results_batch in brain_bank_results:
                if results_batch:
                    # Create a knowledge event with current results
                    knowledge_event = {
                        "type": "knowledge", 
                        "content": results_batch,
                        "complete": False
                    }
                    
                    # If using WebSocket and thread ID is provided, emit to that room
                    if thread_id:
                        try:
                            # Import from socketio_manager module
                            from socketio_manager import emit_knowledge_event
                            emit_knowledge_event(thread_id, knowledge_event)
                        except Exception as e:
                            logger.error(f"Error in socketio_manager websocket delivery for knowledge: {str(e)}")
                            logger.error(f"Make sure you implement emit_knowledge_event in socketio_manager.py similar to: def emit_knowledge_event(thread_id, event): socketio.emit('knowledge', event, room=thread_id)")
                    
                    # Yield the event for standard flow
                    yield knowledge_event
            
            # Log detailed results after processing all brain banks
            for query_idx in range(len(queries)):
                result_count = len(all_results[query_idx])
                logger.info(f"[KNOWLEDGE_DEBUG] After merging, query {query_idx} ({queries[query_idx][:30]}...) has {result_count} results")
            
            # Sort results by score and trim to top_k
            final_results = []
            for query_idx in range(len(queries)):
                results = all_results[query_idx]
                if results:  # Simplified check, just see if the list has any items
                    # Sort by score and take top results
                    sorted_results = sorted(
                        results,
                        key=lambda x: x.get("score", 0),
                        reverse=True
                    )[:top_k]
                    
                    # Log filtering effect
                    original_count = len(results)
                    final_count = len(sorted_results)
                    logger.info(f"[KNOWLEDGE_DEBUG] Query {query_idx} ({queries[query_idx][:30]}...): filtered from {original_count} to {final_count} results")
                    
                    # Log score details
                    if sorted_results:
                        scores = [result.get("score", 0) for result in sorted_results]
                        logger.info(f"[KNOWLEDGE_DEBUG] Final scores for query {query_idx}: min={min(scores):.4f}, max={max(scores):.4f}")
                    
                    final_results.append(sorted_results)
                else:
                    logger.warning(f"[KNOWLEDGE_DEBUG] Query {query_idx} ({queries[query_idx][:30]}...) has NO results after processing")
                    final_results.append([])
            
            # Count total results
            total_results = sum(len(results) for results in final_results)
            empty_queries = sum(1 for results in final_results if not results)
            
            logger.info(f"[KNOWLEDGE_DEBUG] Batch processing complete: {total_results} total results, {empty_queries}/{len(queries)} queries with no results")
            
            # Warning if no results found for any query
            if total_results == 0:
                logger.warning("[KNOWLEDGE_DEBUG] No knowledge results found for any query")
            
            # Send final "complete" event for streaming
            complete_event = {
                "type": "knowledge", 
                "content": [],  # Empty as we've already streamed all content
                "complete": True,
                "summary": {
                    "total_results": total_results,
                    "empty_queries": empty_queries,
                    "total_queries": len(queries)
                }
            }
            
            # Send via WebSocket if configured
            if thread_id:
                try:
                    from socketio_manager import emit_knowledge_event
                    emit_knowledge_event(thread_id, complete_event)
                except Exception as e:
                    logger.error(f"Error in socketio_manager delivery of complete event: {str(e)}")
            
            # Yield for standard flow
            yield complete_event
            
        except Exception as e:
            logger.error(f"[KNOWLEDGE_DEBUG] Error in stream batch query processing: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Send error event for streaming
            error_event = {
                "type": "knowledge", 
                "content": [],
                "complete": True,
                "error": True,
                "error_message": str(e)
            }
            
            if thread_id:
                try:
                    from socketio_manager import emit_knowledge_event
                    emit_knowledge_event(thread_id, error_event)
                except Exception as socket_e:
                    logger.error(f"Error in socketio_manager delivery of error event: {str(socket_e)}")
            
            yield error_event

    
    async def track_conversation_dynamics(self, conversation, current_message):
        """
        Track conversation dynamics to improve natural flow and reduce repetition.
        
        Args:
            conversation: The conversation history
            current_message: The current user message
            
        Returns:
            dict: Insights about conversation dynamics
        """
        # Extract only the user messages for analysis
        user_messages = []
        for line in conversation.split('\n'):
            if line.startswith("User:"):
                user_message = line[5:].strip()
                if user_message:
                    user_messages.append(user_message)
        
        # Add the current message if it's not already included
        if current_message not in user_messages:
            user_messages.append(current_message)
            
        # Initialize insights dictionary
        insights = {
            "repetition_risk": False,
            "user_engagement": "medium",
            "conversation_pace": "normal",
            "topic_shifts": [],
            "repeated_themes": [],
            "suggested_approaches": []
        }
        
        # Check for repetition
        if len(user_messages) >= 3:
            # Check for repeated questions or phrases from the user
            recent_messages = user_messages[-3:]
            repeated_words = set()
            
            # Extract key nouns and verbs
            for msg in recent_messages:
                words = msg.lower().split()
                for word in words:
                    if len(word) > 3 and word not in ["what", "when", "where", "how", "why", "which", "this", "that", "with", "from", "have", "about"]:
                        repeated_words.add(word)
            
            # Count occurrences of each word
            word_counts = {}
            for msg in recent_messages:
                for word in repeated_words:
                    if word in msg.lower():
                        word_counts[word] = word_counts.get(word, 0) + 1
            
            # Identify words that appear in multiple messages
            repeated_themes = [word for word, count in word_counts.items() if count >= 2]
            insights["repeated_themes"] = repeated_themes
            
            if repeated_themes:
                insights["repetition_risk"] = True
                insights["suggested_approaches"].append("introduce_new_aspects")
        
        # Analyze message length for user engagement
        avg_msg_length = sum(len(msg) for msg in user_messages[-3:]) / min(3, len(user_messages))
        if avg_msg_length < 15:
            insights["user_engagement"] = "low"
            insights["suggested_approaches"].append("ask_engaging_questions")
        elif avg_msg_length > 100:
            insights["user_engagement"] = "high"
            insights["suggested_approaches"].append("provide_concise_responses")
        
        # Detect potential topic shifts
        if len(user_messages) >= 2:
            last_msg = user_messages[-1].lower()
            prev_msg = user_messages[-2].lower()
            
            shift_indicators = ["instead", "actually", "by the way", "speaking of", "different", "change", "another", "forget", "what about"]
            
            if any(indicator in last_msg for indicator in shift_indicators):
                insights["topic_shifts"].append("user_initiated")
                insights["suggested_approaches"].append("acknowledge_shift")
        
        return insights

    async def enhance_response_naturalness(self, context):
        """
        Generate guidance for varying sentence structures and speech patterns
        to make responses more natural based on conversation context.
        
        Args:
            context: The conversation context
            
        Returns:
            dict: Guidance for natural language patterns
        """
        # Analyze conversation context
        message_count = len([line for line in context.split('\n') if line.strip()])
        is_early_conversation = message_count < 6
        
        # Init naturalness guidance
        guidance = {
            "sentence_variations": [],
            "transitional_phrases": [],
            "recommended_patterns": [],
            "avoid_patterns": []
        }
        
        # Generate sentence structure variations appropriate for conversation stage
        if is_early_conversation:
            # Early conversation - more direct and engaging
            guidance["sentence_variations"] = [
                "shorter_sentences",
                "questions",
                "simple_statements"
            ]
            guidance["transitional_phrases"] = [
                "first",
                "to start with",
                "I'd like to know"
            ]
            guidance["recommended_patterns"] = [
                "personal_greeting",
                "open_ended_questions",
                "brief_self_intro"
            ]
            guidance["avoid_patterns"] = [
                "complex_explanations",
                "multiple_questions_at_once",
                "technical_jargon"
            ]
        else:
            # Established conversation - more varied and complex
            guidance["sentence_variations"] = [
                "mix_short_and_long",
                "conditional_statements",
                "reflective_observations"
            ]
            guidance["transitional_phrases"] = [
                "additionally",
                "that said",
                "considering what you mentioned",
                "building on that"
            ]
            guidance["recommended_patterns"] = [
                "refer_to_previous_points",
                "add_personal_perspective",
                "strategic_pauses"
            ]
            guidance["avoid_patterns"] = [
                "repetitive_structure",
                "overuse_of_questions",
                "obvious_templated_responses"
            ]
        
        # Analyze conversational mood
        positive_indicators = ["thanks", "good", "great", "appreciate", "help", "useful", "like"]
        negative_indicators = ["no", "not", "don't", "isn't", "wrong", "bad", "problem", "issue", "mistake"]
        
        positive_count = sum(1 for indicator in positive_indicators if indicator in context.lower())
        negative_count = sum(1 for indicator in negative_indicators if indicator in context.lower())
        
        # Adjust guidance based on conversational mood
        if positive_count > negative_count * 2:
            # Very positive conversation
            guidance["recommended_patterns"].append("upbeat_friendly")
            guidance["sentence_variations"].append("enthusiastic_expressions")
        elif negative_count > positive_count * 2:
            # Challenging conversation
            guidance["recommended_patterns"].append("empathetic_listening")
            guidance["recommended_patterns"].append("solution_focused")
            guidance["avoid_patterns"].append("overly_cheerful")
        
        # Add variation for long conversations to avoid monotony
        if message_count > 10:
            guidance["recommended_patterns"].append("introduce_new_elements")
            guidance["sentence_variations"].append("vary_sentence_structure")
            guidance["avoid_patterns"].append("conversation_fatigue_markers")
        
        return guidance
