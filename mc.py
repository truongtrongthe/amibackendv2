import asyncio
from typing import List, Dict
from langchain_openai import ChatOpenAI
from utilities import logger
from langchain_core.messages import HumanMessage,AIMessage
import json
#from sentence_transformers import SentenceTransformer, util
import numpy as np
from database import query_graph_knowledge
import re
from typing import Tuple, Dict, Any
from analysis import stream_analysis, build_context_analysis_prompt, process_analysis_result
from personality import PersonalityManager

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
    #embedder = SentenceTransformer('sentence-transformers/LaBSE')

    def __init__(self, user_id: str = "thefusionlab", convo_id: str = None, 
                 similarity_threshold: float = 0.55, max_active_requests: int = 5):
        self.user_id = user_id
        self.convo_id = convo_id or "default_thread"
        self.similarity_threshold = similarity_threshold
        self.max_active_requests = max_active_requests
        self.personality_manager = PersonalityManager()
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

    async def stream_analysis(self, prompt: str, thread_id_for_analysis=None, use_websocket=False):
        """
        Stream the context analysis from the LLM
        
        Args:
            prompt: The prompt to analyze
            thread_id_for_analysis: Thread ID to use for WebSocket analysis events
            use_websocket: Whether to use WebSocket for streaming
        """
        analysis_buffer = ""
        try:
            async for chunk in StreamLLM.astream(prompt):
                chunk_content = chunk.content
                analysis_buffer += chunk_content
                
                # Create analysis event
                analysis_event = {
                    "type": "analysis", 
                    "content": chunk_content, 
                    "complete": False
                }
                
                # If using WebSocket and thread ID is provided, emit to that room
                if use_websocket and thread_id_for_analysis:
                    try:
                        # Import from socketio_manager module instead of socket
                        from socketio_manager import emit_analysis_event
                        was_delivered = emit_analysis_event(thread_id_for_analysis, analysis_event)
                        if was_delivered:
                            logger.info(f"Sent analysis chunk via socketio_manager to room {thread_id_for_analysis}, length: {len(chunk_content)}")
                        else:
                            logger.warning(f"Analysis chunk NOT DELIVERED via socketio_manager to room {thread_id_for_analysis}, length: {len(chunk_content)} - No active sessions")
                    except Exception as e:
                        logger.error(f"Error in socketio_manager websocket delivery: {str(e)}")
                        was_delivered = False
                
                # Always yield for the standard flow too
                yield {"type": "analysis", "content": chunk_content, "complete": False}
            
            # Send a final complete message with the full analysis
            logger.info(f"Streaming complete analysis, length: {len(analysis_buffer)}")
            
            # Final complete event
            complete_event = {
                "type": "analysis", 
                "content": analysis_buffer, 
                "complete": True
            }
            
            # Send via WebSocket if configured
            if use_websocket and thread_id_for_analysis:
                try:
                    # Import from socketio_manager module instead of socket
                    from socketio_manager import emit_analysis_event
                    was_delivered = emit_analysis_event(thread_id_for_analysis, complete_event)
                    if was_delivered:
                        logger.info(f"Sent complete analysis via socketio_manager to room {thread_id_for_analysis}")
                    else:
                        logger.warning(f"Complete analysis NOT DELIVERED via socketio_manager to room {thread_id_for_analysis} - No active sessions")
                except Exception as e:
                    logger.error(f"Error in socketio_manager delivery of complete event: {str(e)}")
                    was_delivered = False

                if was_delivered:
                    logger.info(f"Sent complete analysis via WebSocket to room {thread_id_for_analysis}")
                else:
                    logger.warning(f"Complete analysis NOT DELIVERED via WebSocket to room {thread_id_for_analysis} - No active sessions")
            
            # Always yield for standard flow
            yield {"type": "analysis", "content": analysis_buffer, "complete": True}
            
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
                    # Import from socketio_manager module instead of socket
                    from socketio_manager import emit_analysis_event
                    was_delivered = emit_analysis_event(thread_id_for_analysis, error_event)
                    if was_delivered:
                        logger.info(f"Sent error event via socketio_manager to room {thread_id_for_analysis}")
                    else:
                        logger.warning(f"Error event NOT DELIVERED via socketio_manager to room {thread_id_for_analysis} - No active sessions")
                except Exception as e:
                    logger.error(f"Error in socketio_manager delivery of error event: {str(e)}")
                    was_delivered = False

                if was_delivered:
                    logger.info(f"Sent error event via WebSocket to room {thread_id_for_analysis}")
                else:
                    logger.warning(f"Error event NOT DELIVERED via WebSocket to room {thread_id_for_analysis} - No active sessions")
            
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
        
        async for response_chunk in self._handle_request(latest_msg_content, user_id, context, builder, state, 
                                                       graph_version_id=graph_version_id,
                                                       use_websocket=use_websocket,
                                                       thread_id_for_analysis=thread_id_for_analysis):
            # Check if this is a special analysis chunk type
            if isinstance(response_chunk, dict) and response_chunk.get("type") == "analysis":
                # For analysis chunks, pass them through directly
                analysis_events_sent += 1
                logger.info(f"Yielding analysis chunk #{analysis_events_sent} from trigger: {response_chunk.get('content', '')[:50]}...")
                
                # When using WebSockets, we don't need to yield the analysis chunks through the regular flow
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
            # Load personality if not already loaded or if graph_version_id changed
            if not hasattr(self, 'personality_instructions') or self.state.get("graph_version_id") != graph_version_id:
                logger.info(f"[PERSONALITY] Loading personality instructions for graph_version_id: {graph_version_id}")
                logger.info(f"[PERSONALITY] Current name before loading: {self.name}")
                await self.load_personality_instructions(graph_version_id)
                logger.info(f"[PERSONALITY] Name after loading: {self.name}")
                logger.info(f"[PERSONALITY] Personality instructions length: {len(self.personality_instructions)}")
                self.state["graph_version_id"] = graph_version_id
            
            # Break conversation into lines
            conversation = [line.strip() for line in context.split("\n") if line.strip()]
            
            logger.info(f"Processing message: '{message}' from user_id: {user_id}")
            
            # Determine conversation language preference
            conversation_language = await self.detect_conversation_language(conversation)
            
            # Detect language of current message
            lang_info = await self.detect_language_with_llm(message)
            
            # Override language confidence if conversation history establishes a pattern
            if conversation_language and conversation_language != lang_info["language"] and len(conversation) > 5:
                logger.info(f"Overriding detected language ({lang_info['language']}) with conversation language ({conversation_language})")
                lang_info["language"] = conversation_language
                lang_info["confidence"] = 0.9  # High confidence based on conversation history
                lang_info["responseGuidance"] = f"Respond in {conversation_language}, maintaining consistency with previous messages"
            
            # STEP 1: Get profile building skills from knowledge base - the core functionality
            profile_query = "contact profile building information gathering customer understanding"
            profile_entries = await query_graph_knowledge(graph_version_id, profile_query, top_k=5)
            profile_instructions = "\n\n".join(entry["raw"] for entry in profile_entries)
            
            # Combine instructions
            process_instructions = profile_instructions
            
            # STEP 2: Analyze conversation context based on knowledge
            logger.info("Building context analysis prompt")
            context_analysis_prompt = build_context_analysis_prompt(context, process_instructions)
            
            # Use streaming for context analysis
            logger.info("Streaming LLM for context analysis")
            context_analysis = ""
            full_analysis = None  # Track the complete analysis
            
            async for analysis_chunk in stream_analysis(context_analysis_prompt, thread_id_for_analysis, use_websocket):
                # Pass the analysis chunks through to the caller
                yield analysis_chunk
                
                # If this is the complete analysis, store it for further processing
                if analysis_chunk.get("complete", False):
                    logger.info(f"Received complete analysis chunk, storing for processing")
                    full_analysis = analysis_chunk.get("content", "")
            
            # Process the analysis
            try:
                # Process the analysis result to get English and Vietnamese parts
                analysis_parts = process_analysis_result(full_analysis)
                english_analysis = analysis_parts["english"]
                vietnamese_analysis = analysis_parts["vietnamese"]
                
                # Use English analysis as context_analysis for backward compatibility
                context_analysis = english_analysis
                
                # Store both versions in state
                if "analysis" not in state:
                    state["analysis"] = {}
                state["analysis"]["english"] = english_analysis
                state["analysis"]["vietnamese"] = vietnamese_analysis
                
                # Also store the complete analysis for backward compatibility
                state["context_analysis"] = context_analysis
                
            except Exception as e:
                logger.error(f"Error processing analysis: {str(e)}")
                # Fallback to original behavior
                context_analysis = full_analysis
                state["context_analysis"] = context_analysis
                state["analysis"] = {
                    "english": context_analysis,
                    "vietnamese": ""
                }
            
            # Use the full analysis if available, otherwise use an empty string
            context_analysis = context_analysis or ""
            logger.info(f"Final context_analysis length: {len(context_analysis)}")
            logger.info(f"Analysis summary: {context_analysis[:500]}...")
            
            # Log analysis for requirements and completeness
            completeness_match = re.search(r'completeness[\s\:]*([\d]+)%', context_analysis.lower())
            if completeness_match:
                completeness = completeness_match.group(1)
                logger.info(f"Detected information completeness: {completeness}%")
            
            missing_info = "missing" in context_analysis.lower() or "incomplete" in context_analysis.lower()
            logger.info(f"Missing information detected: {missing_info}")
            
            # STEP 3: Extract relevant search terms to find appropriate knowledge
            logger.info("Building entity extraction prompt")
            entity_extraction_prompt = (
                f"Based on the conversation and analysis:\n\n"
                f"CONVERSATION:\n{context}\n\n"
                f"ANALYSIS:\n{context_analysis}\n\n"
                f"Extract all relevant search terms that would help find information in a knowledge base:\n\n"
                f"1. CONTACT INFORMATION: Information about the person (demographics, preferences, etc.)\n"
                f"2. TOPICS: Main topics, product categories, or services mentioned\n"
                f"3. QUESTIONS: Specific questions or information requests\n"
                f"4. STAGE INDICATORS: Terms that indicate where in a process the conversation is\n"
                f"5. REACTIONS: Terms indicating agreement, disagreement, satisfaction, frustration\n\n"
                
                f"Format your response as a JSON object with these categories and 1-3 specific search terms for each that would help retrieve relevant knowledge."
            )
            
            logger.info("Invoking LLM for entity extraction")
            entity_response = LLM.invoke(entity_extraction_prompt).content
            logger.info(f"Entity extraction complete ({len(entity_response)} characters)")
            
            # Extract JSON from the response
            entity_match = re.search(r'\{[\s\S]*\}', entity_response)
            if entity_match:
                try:
                    entity_json = json.loads(entity_match.group(0))
                    logger.info(f"Successfully parsed entity JSON with {len(entity_json.keys())} categories")
                    # Flatten all entities into search terms
                    search_terms = []
                    for category, terms in entity_json.items():
                        if isinstance(terms, list):
                            search_terms.extend(terms)
                            logger.info(f"Category {category}: {len(terms)} terms - {', '.join(terms)}")
                        elif isinstance(terms, str):
                            search_terms.append(terms)
                            logger.info(f"Category {category}: single term - {terms}")
                except Exception as json_error:
                    logger.error(f"Failed to parse entity JSON: {str(json_error)}")
                    logger.error(f"Raw match: {entity_match.group(0)[:100]}...")
                    search_terms = [message]  # Fallback to original message
                    logger.info(f"Using fallback search term: {message}")
            else:
                logger.warning("No JSON pattern found in entity extraction response")
                search_terms = [message]  # Fallback to original message
                logger.info(f"Using fallback search term: {message}")
            
            # Create targeted queries based on context analysis and extracted entities
            targeted_queries = []
            
            # Add priority query based on analysis
            if missing_info:
                priority_query = "profile information gathering techniques"
                targeted_queries.append(priority_query)
                logger.info(f"Added priority query due to missing information: '{priority_query}'")
            
            # Check for rejection patterns in the message
            rejection_patterns = [
                r'\b(?:no|nope|incorrect|wrong|not right|not correct|disagree|rejected|refuse|won\'t)\b',
                r'\bdon\'t\s+(?:agree|think|believe|want|like)\b',
                r'\bthat\'s not\b',
                r'\b(?:bull|nonsense|ridiculous|crazy)\b'
            ]
            
            has_rejection = any(re.search(pattern, message.lower()) for pattern in rejection_patterns)
            if has_rejection:
                rejection_query = "handling objection disagreement rejection conversation"
                targeted_queries.append(rejection_query)
                logger.info(f"Added rejection handling query: '{rejection_query}'")
            
            # Add entity-based queries
            entity_queries_count = 0
            for term in search_terms:
                if term and len(term.strip()) > 2:  # Avoid very short terms
                    targeted_queries.append(term)
                    entity_queries_count += 1
            
            logger.info(f"Added {entity_queries_count} entity-based queries")
            
            # Add message context query
            targeted_queries.append(message)
            logger.info(f"Added message as final query: '{message}'")
            
            # Add language preference to ensure appropriate content
            language_code = lang_info.get("code", "en").lower()
            lang_query = f"content in {language_code} {' '.join(search_terms[:2])}"
            targeted_queries.append(lang_query)
            logger.info(f"Added language preference query: '{lang_query}'")
            
            # Remove duplicates while preserving order
            unique_queries = []
            for query in targeted_queries:
                if query not in unique_queries:
                    unique_queries.append(query)
            
            logger.info(f"Final unique queries: {len(unique_queries)} queries")
            for i, query in enumerate(unique_queries[:7]):
                logger.info(f"Query {i+1}: '{query}'")
            
            # Retrieve knowledge for each unique query
            logger.info("Starting knowledge retrieval for queries")
            all_knowledge = []
            query_results = {}
            
            for query in unique_queries:  # Process all unique queries
                try:
                    logger.info(f"Retrieving knowledge for query: '{query}'")
                    results = await query_graph_knowledge(graph_version_id, query, top_k=3)
                    if results:
                        query_results[query] = len(results)
                        all_knowledge.extend(results)
                        logger.info(f"Retrieved {len(results)} results for query '{query}'")
                    else:
                        logger.info(f"No results for query '{query}'")
                        query_results[query] = 0
                except Exception as e:
                    logger.error(f"Error retrieving knowledge for query '{query}': {e}")
                    query_results[query] = 0
            
            logger.info(f"Knowledge retrieval complete: {len(all_knowledge)} total entries before deduplication")
            logger.info(f"Results by query: {json.dumps(query_results)}")
            
            # Remove duplicate knowledge entries
            unique_knowledge = []
            seen_ids = set()
            for entry in all_knowledge:
                if entry['id'] not in seen_ids:
                    seen_ids.add(entry['id'])
                    unique_knowledge.append(entry)
            
            # Create knowledge context from unique entries
            knowledge_context = "\n\n".join(entry["raw"] for entry in unique_knowledge)
            logger.info(f"Retrieved {len(unique_knowledge)} unique knowledge entries from {len(unique_queries)} queries")
            logger.info(f"Knowledge context: {len(knowledge_context)} characters")
            
            # Extract recent topics to avoid repeating
            recent_topics = []
            if len(conversation) > 2:
                # Extract last 2-3 AI messages to check for repeated content
                ai_messages = [msg[4:] for msg in conversation if msg.startswith("AI:")][-3:]
                # Use simple keyword extraction to identify repeated phrases
                for msg in ai_messages:
                    words = re.findall(r'\b\w{4,}\b', msg.lower())
                    for word in words:
                        if words.count(word) > 1 and word not in recent_topics and len(word) > 4:
                            recent_topics.append(word)
            
            logger.info(f"Identified potential repetitive topics: {recent_topics}")
            
            # Assess knowledge coverage and determine if feedback is needed
            coverage_score, missing_aspects, requires_feedback = await self.assess_knowledge_coverage(
                query_results, context_analysis, message
            )
            
            logger.info(f"Knowledge coverage assessment: score={coverage_score}, missing_aspects={missing_aspects}, requires_feedback={requires_feedback}")
            
            # If feedback is required, generate a specific request for more information
            if requires_feedback:
                feedback_request = await self.generate_feedback_request(missing_aspects, message, context)
                logger.info(f"Generated feedback request: {feedback_request}")
                
                # Simplified response prompt for feedback requests
                feedback_prompt = (
                    f"AI: {self.name}\n"
                    f"Context: {context}\n"
                    f"Message: '{message}'\n"
                    f"Language: {lang_info['language']}\n"
                    f"CRITICAL PERSONALITY INSTRUCTIONS - YOU MUST FOLLOW THESE EXACTLY:\n{self.personality_instructions}\n\n"
                    
                    f"Instructions:\n"
                    f"1. PERSONALITY IS YOUR TOP PRIORITY: You MUST embody the exact role, expertise, tone, and positioning specified in the PERSONALITY INSTRUCTIONS above.\n"
                    f"2. The system has determined that you need more information to properly respond.\n"
                    f"3. Craft a response that genuinely seeks clarification, in a friendly and conversational way.\n\n"
                    
                    f"Here is the specific feedback request: \"{feedback_request}\"\n\n"
                    
                    f"Your response should:\n"
                    f"1. Acknowledge the user's message briefly\n"
                    f"2. Express that you want to help but need more specific information\n"
                    f"3. Include the feedback request, making it sound natural and aligned with your personality\n"
                    f"4. Be concise and friendly\n\n"
                    
                    f"Always respond in {lang_info['language']}."
                )
                
                logger.info("Starting feedback request response generation")
                async for _ in self.stream_response(feedback_prompt, builder):
                    yield builder.build()
                
                logger.info("Feedback request response complete")
                return
            
            # STEP 4: Generate response based on context analysis and knowledge
            logger.info(f"Responder name: {self.name}")
            logger.info(f"Responder personality instructions: {self.personality_instructions}")
            logger.info("Building response prompt")
            response_prompt = (
                f"AI: {self.name}\n"
                f"Context: {context}\n"
                f"Message: '{message}'\n"
                f"Context Analysis: {context_analysis}\n"
                f"Knowledge: {knowledge_context}\n"
                f"Language: {lang_info['language']} (confidence: {lang_info['confidence']})\n\n"
                
                f"CRITICAL PERSONALITY INSTRUCTIONS - YOU MUST FOLLOW THESE EXACTLY:\n{self.personality_instructions}\n\n"
                
                f"Instructions:\n"
                f"1. PERSONALITY IS YOUR TOP PRIORITY: You MUST embody the exact role, expertise, tone, and positioning specified in the PERSONALITY INSTRUCTIONS above.\n"
                f"2. Your primary guidance comes from the knowledge base. Follow the NEXT ACTIONS from the Context Analysis.\n"
                f"3. ALWAYS reply in {lang_info['language']} - this is mandatory regardless of what language appears in the knowledge base.\n"
                f"4. Keep responses concise and conversational.\n" 
                f"5. GREETING GUIDELINES:\n"
                f"   - For first message only: Use a natural greeting appropriate to the time and context\n"
                f"   - For all subsequent messages: Skip greetings entirely and respond directly to the content\n"
                f"6. If the user expressed disagreement or rejection, acknowledge it respectfully.\n"
                f"7. AVOID REPETITION: Do not repeat the same information or questions from previous exchanges.\n\n"
                
                f"Response Structure:\n"
                f"1. If first message: Natural greeting, otherwise respond directly to content\n"
                f"2. ADDRESS the current context appropriately:\n"
                f"   a) If gathering information is needed, ask specific questions (but not ones already asked)\n"
                f"   b) If providing information is needed, present relevant details in a fresh way\n"
                f"   c) If addressing concerns, provide targeted responses\n"
                f"3. PROGRESS the conversation with something new that hasn't been discussed yet\n\n"
                
                f"CRITICAL: Maintain a natural conversation flow that doesn't feel repetitive. Introduce fresh angles or questions rather than repeating previous points.\n\n"
                
                + (f"AVOID these repetitive topics/phrasings that have appeared multiple times: {', '.join(recent_topics)}\n\n" if recent_topics else "")
                
                + f"Always respond in {lang_info['language']}. Use knowledge when relevant, but prioritize a natural conversation flow."
            )
            
            logger.info("Starting response generation")
            async for _ in self.stream_response(response_prompt, builder):
                yield builder.build()
            
            logger.info("Response generation complete")
                
        except Exception as e:
            logger.error(f"Error handling request: {str(e)}", exc_info=True)
            builder.add("Oops, có lỗi khi tìm thông tin, thử lại nhé!")
            yield builder.build()

    async def load_personality_instructions(self, graph_version_id: str = ""):
        """
        Load personality instructions using the PersonalityManager.
        """
        return await self.personality_manager.load_personality_instructions(graph_version_id)
