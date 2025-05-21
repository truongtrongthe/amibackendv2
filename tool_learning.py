import json
import asyncio
import logging
from typing import Dict, List, Any, Optional, Union, AsyncGenerator, Set
from datetime import datetime
from uuid import uuid4
import re
import pytz
import weakref

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import functools
import concurrent.futures

from pccontroller import save_knowledge, query_knowledge

from utilities import logger

# Custom JSON encoder for datetime objects
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

def serialize_model(model: BaseModel) -> Dict[str, Any]:
    """Serialize a Pydantic model to a JSON-serializable dict."""
    return json.loads(json.dumps(model.model_dump(), cls=DateTimeEncoder))

# Initialize LLM
LLM = ChatOpenAI(model="gpt-4o", streaming=False, temperature=0.01)

class LearningProcessor:
    def __init__(self):
        self.graph_version_id = ""
        # Set to keep track of pending background tasks
        self._background_tasks: Set[asyncio.Task] = set()

    async def initialize(self):
        """Initialize the processor asynchronously."""
        logger.info("Initializing LearningProcessor")
        return self

    async def process_incoming_message(self, message: str, conversation_context: str, user_id: str, thread_id: Optional[str] = None) -> Dict[str, Any]:
        """Process incoming message with active learning flow."""
        logger.info(f"Processing message from user {user_id}")
        try:

            if not message.strip():
                logger.error("Empty message")
                return {"status": "error", "message": "Empty message provided"}
            
            # Step 1: Get suggested knowledge queries from previous responses
            suggested_queries = []
            if conversation_context:
                ai_messages = re.findall(r'AI: (.*?)(?:\n\n|$)', conversation_context, re.DOTALL)
                if ai_messages:
                    last_ai_message = ai_messages[-1]
                    query_section = re.search(r'<knowledge_queries>(.*?)</knowledge_queries>', last_ai_message, re.DOTALL)
                    if query_section:
                        query_text = query_section.group(1).strip()
                        try:
                            queries_data = json.loads(query_text)
                            suggested_queries = queries_data if isinstance(queries_data, list) else queries_data.get("queries", [])
                            logger.info(f"Extracted {len(suggested_queries)} suggested knowledge queries")
                        except json.JSONDecodeError:
                            logger.warning("Failed to parse query section as JSON")

            # Step 2: Search for relevant knowledge
            logger.info(f"Searching for knowledge based on message...")
            analysis_knowledge = await self._search_knowledge(message, conversation_context, user_id, thread_id)
            
            # Step 3: Enrich with suggested queries if we don't have sufficient results
            if suggested_queries and len(analysis_knowledge.get("query_results", [])) < 3:
                logger.info(f"Searching for additional knowledge using {len(suggested_queries)} suggested queries")
                primary_similarity = analysis_knowledge.get("similarity", 0.0)
                primary_knowledge = analysis_knowledge.get("knowledge_context", "")
                primary_queries = analysis_knowledge.get("queries", [])
                primary_query_results = analysis_knowledge.get("query_results", [])
                
                for query in suggested_queries:
                    if query not in primary_queries:  # Avoid duplicate queries
                        query_knowledge = await self._search_knowledge(query, conversation_context, user_id, thread_id)
                        query_similarity = query_knowledge.get("similarity", 0.0)
                        query_results = query_knowledge.get("query_results", [])
                        
                        logger.info(f"Suggested query '{query}' yielded similarity score: {query_similarity}")
                        
                        # Add the new query and its results to our collection
                        if query_results:
                            primary_queries.append(query)
                            primary_query_results.extend(query_results)
                            logger.info(f"Added results from suggested query '{query}'")
                
                # Update analysis_knowledge with enriched data
                if len(primary_query_results) > len(analysis_knowledge.get("query_results", [])):
                    analysis_knowledge["queries"] = primary_queries
                    analysis_knowledge["query_results"] = primary_query_results
                    logger.info(f"Updated knowledge with {len(primary_query_results)} total query results")
            
            # Step 4: Log similarity score
            similarity = analysis_knowledge.get("similarity", 0.0)
            logger.info(f"💯 Found knowledge with similarity score: {similarity}")
            
            # Step 5: Generate response
            logger.info(f"Generating response based on knowledge...")
            prior_data = analysis_knowledge.get("prior_data", {})
            
            # Remove hardcoded teaching intent detection and let the LLM handle it
            response = await self._active_learning(message, conversation_context, analysis_knowledge, user_id, prior_data)
            logger.info(f"Response generated with status: {response.get('status', 'unknown')}")
            
            if "metadata" in response and "response_strategy" in response["metadata"]:
                logger.info(f"Active learning mode used: {response['metadata']['response_strategy']}")
            
            # Step 6: Save knowledge for relevant responses
            if response.get("status") == "success":
                # Get teaching intent and priority topic info from LLM evaluation
                has_teaching_intent = response.get("metadata", {}).get("has_teaching_intent", False)
                is_priority_topic = response.get("metadata", {}).get("is_priority_topic", False)
                should_save_knowledge = response.get("metadata", {}).get("should_save_knowledge", False)
                priority_topic_name = response.get("metadata", {}).get("priority_topic_name", "")
                intent_type = response.get("metadata", {}).get("intent_type", "unknown")
                
                # Trust the LLM's intent detection entirely
                
                # Force should_save_knowledge to True if it's a priority topic
                if is_priority_topic:
                    should_save_knowledge = True
                
                logger.info(f"LLM evaluation: intent={intent_type}, teaching_intent={has_teaching_intent}, priority_topic={is_priority_topic}, should_save={should_save_knowledge}")
                
                # Only save knowledge when teaching intent is detected
                if has_teaching_intent:
                    # Determine logging message based on detected intent
                    log_reason = "teaching intent"
                    
                    logger.info(f"Saving knowledge due to {log_reason}")
                    
                    message_content = response["message"]
                    conversational_response = re.split(r'<knowledge_queries>', message_content)[0].strip()
                    query_section = re.search(r'<knowledge_queries>(.*?)</knowledge_queries>', message_content, re.DOTALL)
                    knowledge_queries = json.loads(query_section.group(1).strip()) if query_section else []

                    # Set up categories and bank name
                    categories = ["health_segmentation"] if is_priority_topic else ["general"]
                    bank_name = "conversation"
                    
                    # Add teaching intent category
                    categories.append("teaching_intent")
                    
                    # Add specific topic category if provided by LLM
                    if priority_topic_name and priority_topic_name not in categories:
                        categories.append(priority_topic_name.lower().replace(" ", "_"))
                    
                    # Combine user input and AI response in a formatted way
                    combined_knowledge = f"User: {message}\n\nAI: {conversational_response}"
                    
                    # Add synthesized flag if this was from TEACHING_INTENT strategy
                    if response.get("metadata", {}).get("response_strategy") == "TEACHING_INTENT":
                        categories.append("synthesized_knowledge")
                        logger.info(f"Adding synthesized_knowledge category for enhanced teaching content")
                    
                    # Save combined knowledge
                    try:
                        logger.info(f"Saving combined knowledge to {bank_name} bank: '{combined_knowledge[:100]}...'")
                        success = await self._background_save_knowledge(
                            input_text=combined_knowledge,
                            user_id=user_id,
                            bank_name=bank_name,
                            thread_id=thread_id,
                            topic=priority_topic_name or "user_teaching",
                            categories=categories,
                            ttl_days=365  # 365 days TTL
                        )
                        logger.info(f"Save combined knowledge completed: {success}")
                    except Exception as e:
                        logger.error(f"Error saving combined knowledge: {str(e)}")
                    
                    logger.info(f"Saved combined knowledge for topic '{priority_topic_name or 'user_teaching'}'")

                    # If we have synthesized content, save an additional entry with just the AI response
                    # This helps with future retrievals by isolating the clean synthesized knowledge
                    if response.get("metadata", {}).get("response_strategy") == "TEACHING_INTENT":
                        try:
                            synthesis_categories = list(categories)
                            synthesis_categories.append("ai_synthesis")
                            logger.info(f"Saving additional AI synthesis for improved future retrieval")
                            synthesis_success = await self._background_save_knowledge(
                                input_text=f"AI Synthesis: {conversational_response}",
                                user_id=user_id,
                                bank_name=bank_name,
                                thread_id=thread_id,
                                topic=priority_topic_name or "user_teaching",
                                categories=synthesis_categories,
                                ttl_days=365  # 365 days TTL
                            )
                            logger.info(f"Save AI synthesis completed: {synthesis_success}")
                        except Exception as e:
                            logger.error(f"Error saving AI synthesis: {str(e)}")

            return response
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            import traceback
            logger.error(f"Stack trace: {traceback.format_exc()}")
            return {"status": "error", "message": f"Error: {str(e)}"}
   
    async def _search_knowledge(self, message: Union[str, List], conversation_context: str = "", user_id: str = "unknown", thread_id: Optional[str] = None) -> Dict[str, Any]:
        logger.info(f"Searching for analysis knowledge based on message: {str(message)[:100]}...")
        try:
            if not isinstance(message, str):
                logger.warning(f"Converting non-string message: {message}")
                primary_query = str(message[0]) if isinstance(message, list) and message else str(message)
            else:
                primary_query = message.strip()
            if not primary_query:
                logger.error("Empty primary query")
                return {
                    "knowledge_context": "",
                    "similarity": 0.0,
                    "query_count": 0,
                    "prior_data": {"topic": "", "knowledge": ""},
                    "metadata": {"similarity": 0.0}
                }
            queries = []
            prior_topic = ""
            prior_knowledge = ""
            if conversation_context:
                user_messages = re.findall(r'User: (.*?)(?:\n\n|$)', conversation_context, re.DOTALL)
                ai_messages = re.findall(r'AI: (.*?)(?:\n\n|$)', conversation_context, re.DOTALL)
                logger.info(f"Found {len(user_messages)} user messages in context")
                if user_messages:
                    prior_topic = user_messages[-2].strip() if len(user_messages) > 1 else user_messages[0].strip()
                    
                    logger.info(f"Extracted prior topic: {prior_topic[:50]}")
                if ai_messages:
                    prior_knowledge = ai_messages[-1].strip()

            # Use the _detect_follow_up helper function instead of duplicating code
            follow_up_result = self._detect_follow_up(primary_query, prior_topic)
            is_follow_up = follow_up_result["is_follow_up"]
            
            if is_follow_up and prior_topic:
                queries.append(prior_topic)
                logger.info(f"Follow-up detected, reusing prior topic: {prior_topic[:50]}")
                similarity = 0.7
                knowledge_context = prior_knowledge
            else:
                queries.append(primary_query)
                similarity = 0.0
                knowledge_context = ""

            temp_response = await self._active_learning(primary_query, conversation_context, {}, user_id, {})
            if "message" in temp_response:
                query_section = re.search(r'<knowledge_queries>(.*?)</knowledge_queries>', temp_response["message"], re.DOTALL)
                if query_section:
                    try:
                        llm_queries = json.loads(query_section.group(1).strip())
                        valid_llm_queries = [
                            q for q in llm_queries 
                            if q not in queries and len(q.strip()) > 5 and 
                            not any(vague in q.lower() for vague in ["core topic", "chủ đề chính", "cuộc sống"])
                        ]
                        queries.extend(valid_llm_queries)
                        logger.info(f"Added {len(valid_llm_queries)} LLM-generated queries")
                    except json.JSONDecodeError:
                        logger.warning("Failed to parse LLM queries, retrying once")
                        temp_response = await self._active_learning(primary_query, conversation_context, {}, user_id, {})
                        query_section = re.search(r'<knowledge_queries>(.*?)</knowledge_queries>', temp_response["message"], re.DOTALL)
                        if query_section:
                            try:
                                llm_queries = json.loads(query_section.group(1).strip())
                                valid_llm_queries = [
                                    q for q in llm_queries 
                                    if q not in queries and len(q.strip()) > 5 and 
                                    not any(vague in q.lower() for vague in ["core topic", "chủ đề chính", "cuộc sống"])
                                ]
                                queries.extend(valid_llm_queries)
                                logger.info(f"Added {len(valid_llm_queries)} LLM-generated queries on retry")
                            except json.JSONDecodeError:
                                logger.error("Failed to parse LLM queries after retry")
            logger.info(f"Queries: {queries}")
            queries = list(dict.fromkeys(queries))
            queries = [q for q in queries if len(q.strip()) > 5]
            if not queries:
                logger.warning("No valid queries found")
                return {
                    "knowledge_context": knowledge_context,
                    "similarity": similarity,
                    "query_count": 0,
                    "prior_data": {"topic": prior_topic, "knowledge": prior_knowledge},
                    "metadata": {"similarity": similarity}
                }

            query_count = 0
            bank_name = "conversation"
            
            # Batch query_knowledge calls to reduce API retries
            results_list = await asyncio.gather(
                *(query_knowledge(
                    query=query,
                    bank_name=bank_name,
                    user_id=user_id,
                    thread_id=None,  # Remove thread_id restriction to find more results
                    topic=None,      # Remove topic restriction
                    top_k=10,
                    min_similarity=0.2  # Lower threshold for better matching
                ) for query in queries),
                return_exceptions=True
            )

            # Store all query results
            all_query_results = []
            best_result = None
            highest_similarity = 0.0

            for query, results in zip(queries, results_list):
                query_count += 1
                if isinstance(results, Exception):
                    logger.warning(f"Query '{query[:30]}...' failed: {str(results)}")
                    all_query_results.append(None)  # Add None for failed queries
                    continue
                if not results:
                    logger.info(f"Query '{query[:30]}...' returned no results")
                    all_query_results.append(None)  # Add None for empty results
                    continue
                
                # Log query results for debugging
                top_result = results[0]
                all_query_results.append(top_result)  # Store the top result for each query
                query_similarity = top_result["score"]
                knowledge_content = top_result["raw"]
                
                # Extract just the AI portion if this is a combined knowledge entry
                if knowledge_content.startswith("User:") and "\n\nAI:" in knowledge_content:
                    ai_part = re.search(r'\n\nAI:(.*)', knowledge_content, re.DOTALL)
                    if ai_part:
                        knowledge_content = ai_part.group(1).strip()
                        logger.info(f"Extracted AI portion from combined knowledge")
                
                logger.info(f"Query '{query[:30]}...' yielded similarity: {query_similarity}, content: '{knowledge_content[:50]}...'")
                
                # Track the best overall result
                if query_similarity > highest_similarity:
                    highest_similarity = query_similarity
                    best_result = top_result
                    similarity = query_similarity
                    knowledge_context = knowledge_content
                    logger.info(f"Updated best knowledge with similarity: {similarity}")

            # Apply regular boost for priority topics
            if any(term in primary_query.lower() or (prior_topic and term in prior_topic.lower()) 
                   for term in ["mục tiêu", "goals", "active learning", "phân nhóm", "phân tích chân dung", "chân dung khách hàng"]):
                vibe_score = 1.1
                similarity *= vibe_score
                logger.info(f"Applied vibe score {vibe_score} for priority topic")
            else:
                vibe_score = 1.0

            # Filter out None results
            valid_query_results = [result for result in all_query_results if result is not None]
            
            logger.info(f"Final similarity: {similarity} from {query_count} queries, found {len(valid_query_results)} valid results")
            return {
                "knowledge_context": knowledge_context,
                "similarity": similarity,
                "query_count": query_count,
                "queries": queries,
                "original_query": primary_query,  # Add the original query for reference
                "query_results": valid_query_results,
                "prior_data": {"topic": prior_topic, "knowledge": prior_knowledge},
                "metadata": {"similarity": similarity, "vibe_score": vibe_score}
            }
        except Exception as e:
            logger.error(f"Error fetching knowledge: {str(e)}")
            return {
                "knowledge_context": prior_knowledge if is_follow_up else "",
                "similarity": 0.7 if is_follow_up else 0.0,
                "query_count": 0,
                "prior_data": {"topic": prior_topic, "knowledge": prior_knowledge},
                "metadata": {"similarity": 0.7 if is_follow_up else 0.0}
            }

    async def _active_learning(self, message: Union[str, List], conversation_context: str = "", analysis_knowledge: Dict = None, user_id: str = "unknown", prior_data: Dict = None) -> Dict[str, Any]:
        logger.info("Answering user question with active learning approach")
        
        vietnam_tz = pytz.timezone('Asia/Ho_Chi_Minh')
        current_time = datetime.now(vietnam_tz)
        date_str = current_time.strftime("%A, %B %d, %Y")
        time_str = current_time.strftime("%H:%M")
        temporal_context = f"Current date and time: {date_str} at {time_str} (Asia/Ho_Chi_Minh timezone)."
        
        message_str = message if isinstance(message, str) else str(message[0]) if isinstance(message, list) and message else ""
        if not message_str:
            logger.error("Empty message in active learning")
            return {"status": "error", "message": "Empty message provided"}
        
        knowledge_context = analysis_knowledge.get("knowledge_context", "") if analysis_knowledge else ""
        similarity_score = float(analysis_knowledge.get("similarity", 0.0)) if analysis_knowledge else 0.0
        logger.info(f"Using similarity score: {similarity_score}")
        
        # Extract query information if available
        queries = analysis_knowledge.get("queries", []) if analysis_knowledge else []
        query_results = analysis_knowledge.get("query_results", []) if analysis_knowledge else []

        prior_topic = prior_data.get("topic", "") if prior_data else ""
        prior_knowledge = prior_data.get("knowledge", "") if prior_data else ""
        
        # Use the _detect_follow_up helper function instead of duplicating code
        follow_up_result = self._detect_follow_up(message_str, prior_topic)
        is_confirmation = follow_up_result["is_confirmation"]
        is_follow_up = follow_up_result["is_follow_up"]
        has_pattern_match = follow_up_result["has_pattern_match"]
        topic_overlap = follow_up_result["topic_overlap"]
        
        core_prior_topic = prior_topic
        
        # Knowledge handling strategy based on queries and similarity
        knowledge_response_sections = []
        
        # Check for conversation closing messages - common phrases indicating end of conversation
        closing_phrases = [
            "thế thôi", "hẹn gặp lại", "tạm biệt", "chào nhé", "goodbye", "bye", "cảm ơn nhé", 
            "cám ơn nhé", "đủ rồi", "vậy là đủ", "hôm nay vậy là đủ", "hẹn lần sau"
        ]
        
        is_closing_message = any(phrase in message_str.lower() for phrase in closing_phrases)
        
        if is_closing_message:
            # Override low confidence for closing messages
            knowledge_context = "CONVERSATION_CLOSING: User is ending the conversation politely."
            response_strategy = "CLOSING"
            strategy_instructions = (
                "Recognize this as a closing message where the user is ending the conversation. "
                "Respond with a brief, polite farewell message. "
                "Thank them for the conversation and express willingness to help in the future. "
                "Keep it concise and friendly, in the same language they used (Vietnamese/English)."
            )
            logger.info(f"Detected conversation closing message, overriding low confidence response")
        elif queries and query_results:
            # Group results by confidence level
            high_confidence = []
            medium_confidence = []
            low_confidence = []
            
            for i, query in enumerate(queries):
                # Get corresponding result if available
                result = query_results[i] if i < len(query_results) else None
                
                if not result:
                    low_confidence.append(query)
                    continue
                    
                query_similarity = result.get("score", 0.0)
                query_content = result.get("raw", "")
                
                if not query_content:
                    low_confidence.append(query)
                    continue
                
                # Extract just the AI portion if this is a combined knowledge entry
                if query_content.startswith("User:") and "\n\nAI:" in query_content:
                    ai_part = re.search(r'\n\nAI:(.*)', query_content, re.DOTALL)
                    if ai_part:
                        query_content = ai_part.group(1).strip()
                
                if query_similarity < 0.35:
                    low_confidence.append(query)
                elif 0.35 <= query_similarity <= 0.7:
                    medium_confidence.append((query, query_content, query_similarity))
                else:  # > 0.7
                    high_confidence.append((query, query_content, query_similarity))
            
            # Format response sections by confidence level
            if high_confidence:
                knowledge_response_sections.append("HIGH CONFIDENCE KNOWLEDGE:")
                for query, content, score in high_confidence:
                    knowledge_response_sections.append(
                        f"On the topic of '{query}' (confidence: {score:.2f}): {content}"
                    )
            
            if medium_confidence:
                knowledge_response_sections.append("MEDIUM CONFIDENCE KNOWLEDGE:")
                for query, content, score in medium_confidence:
                    knowledge_response_sections.append(
                        f"Regarding '{query}' (confidence: {score:.2f}): {content}\nCould you please confirm or clarify this information?"
                    )
            
            if low_confidence:
                knowledge_response_sections.append("LOW CONFIDENCE/NO KNOWLEDGE:")
                for query in low_confidence:
                    knowledge_response_sections.append(
                        f"I don't have sufficient knowledge about '{query}'. Would you like to teach me about this topic?"
                    )
            
            # Combine the knowledge sections if they exist
            if knowledge_response_sections:
                knowledge_context = "\n\n".join(knowledge_response_sections)
                logger.info(f"Created structured knowledge response with {len(high_confidence)} high, {len(medium_confidence)} medium, and {len(low_confidence)} low confidence items")
        logger.info(f"Knowledge context: {knowledge_context}")
        if is_follow_up:
            response_strategy = "FOLLOW_UP"
            
            # Enhanced strategy for confirmation responses
            if is_confirmation:
                strategy_instructions = (
                    f"Recognize this is a direct confirmation ('{message_str}') to your question in your previous message'. "
                    "Continue the conversation as if the user said 'yes' to your previous question. "
                    "Provide a helpful response that builds on the previous question, offering relevant details or asking a follow-up question. "
                    "Don't ask for clarification when the confirmation is clear - proceed with the conversation flow naturally. "
                    "If your previous question offered to provide more information, now is the time to provide that information. "
                    "Keep the response substantive, helpful, and directly related to what the user just confirmed interest in."
                )
            else:
                strategy_instructions = (
                    "Recognize the message as a follow-up or confirmation of PRIOR TOPIC, referring to a specific concept or group from PRIOR KNOWLEDGE (e.g., customer segmentation methods). "
                    "Use PRIOR KNOWLEDGE to deepen the discussion, leveraging specific details. "
                    "Structure the response with key aspects (e.g., purpose, methods, outcomes). "
                    "If PRIOR TOPIC is ambiguous, rephrase it (e.g., 'It sounds like you're confirming customer segmentation…'). "
                    "Ask a targeted follow-up to advance the discussion."
                )
            # For follow-ups, use prior knowledge if no specific knowledge response sections
            if not knowledge_response_sections:
                knowledge_context = prior_knowledge
                similarity_score = max(similarity_score, 0.7)
        elif is_closing_message:
            # This has already been set up earlier, but we'll ensure response_strategy is set correctly
            response_strategy = "CLOSING"
            # Keep the strategy_instructions from above
            # No need to adjust similarity score
        elif similarity_score < 0.35 and not knowledge_response_sections:
            response_strategy = "LOW_SIMILARITY"
            # Create query-specific response section if no other knowledge is found
            if queries:
                query_text = queries[0] if isinstance(queries, list) and queries else str(queries)
                knowledge_response_sections = [f"I don't have sufficient knowledge about '{query_text}'. Would you like to teach me about this topic?"]
                knowledge_context = "\n\n".join(knowledge_response_sections)
                logger.info(f"Created LOW_SIMILARITY response for query: {query_text}")
            
            # Handle short or unclear queries differently
            is_short_query = len(message_str.strip().split()) <= 2
            
            if is_short_query:
                strategy_instructions = (
                    "Recognize this as a very short or potentially unclear message. "
                    "Acknowledge that you need more information to provide a helpful response. "
                    "Politely ask the user to provide more details or context about what they're asking. "
                    "Suggest a few possible interpretations of their query if appropriate. "
                    "Keep your response friendly and helpful, showing eagerness to assist once you have more information. "
                    "Match the user's language choice (Vietnamese/English). "
                    "Ensure you provide a response even if the query is very minimal or unclear."
                )
            else:
                strategy_instructions = (
                    "State: 'Tôi không thể tìm thấy thông tin liên quan; vui lòng giải thích thêm.' "
                    "Ask for more details about the topic. "
                    "Propose a specific question (e.g., 'Bạn có thể chia sẻ thêm về ý nghĩa của điều này không?'). "
                    "If the message appears to be attempting to teach or explain something, acknowledge this and express "
                    "interest in learning about the topic through a thoughtful follow-up question."
                )
        else:
            response_strategy = "RELEVANT_KNOWLEDGE"
            strategy_instructions = (
                "Present the retrieved knowledge prominently in your response, directly quoting the most relevant parts. "
                "If there are multiple knowledge sections with different confidence levels, address each appropriately:"
                "- For low confidence (<0.35): Mention you don't have good information and ask for clarification"
                "- For medium confidence (0.35-0.7): Present the knowledge but express uncertainty and ask for confirmation"
                "- For high confidence (>0.7): Present the knowledge confidently"
                "Begin with a clear statement like 'Theo thông tin tôi có...' or 'Mục tiêu của tôi là...' followed by the knowledge. "
                "Structure the response to emphasize the core information from EXISTING KNOWLEDGE. "
                "If the message likely continues PRIOR TOPIC, prioritize deepening that topic with specific details. "
                "If the topic is ambiguous, connect the dots by stating how the knowledge answers their question."
            )
        
        # Check if this appears to be a teaching intent message
        teaching_keywords = ["let me explain", "I'll teach you", "Tôi sẽ giải thích", "Tôi dạy bạn", 
                             "here's how", "đây là cách", "the way to", "Important to know", 
                             "you should know", "bạn nên biết", "cần hiểu rằng", "phương pháp", "cách thức"]
        has_teaching_markers = any(keyword.lower() in message_str.lower() for keyword in teaching_keywords)
        
        # Check for Vietnamese greeting forms or names
        vn_greeting_patterns = ["anh ", "chị ", "bạn ", "cô ", "ông ", "bác ", "em "]
        common_vn_names = ["hùng", "hương", "minh", "tuấn", "thảo", "an", "hà", "thủy", "trung", "mai", "hoa", "quân", "dũng", "hiền", "nga", "tâm", "thanh", "tú", "hải", "hòa", "yến", "lan", "hạnh", "phương", "dung", "thu", "hiệp", "đức", "linh", "huy", "tùng", "bình", "giang", "tiến"]
        
        is_vn_greeting = any(pattern in message_str.lower() for pattern in vn_greeting_patterns)
        message_words = message_str.lower().split()
        contains_vn_name = any(name in message_words for name in common_vn_names)
        
        # If this is just a name or greeting, treat it as a greeting
        if (is_vn_greeting or contains_vn_name) and len(message_str.split()) <= 3:
            response_strategy = "GREETING"
            strategy_instructions = (
                "Recognize this as a Vietnamese greeting or someone addressing you by name. "
                "Respond warmly and appropriately to the greeting. "
                "If they used a Vietnamese name or greeting form, respond in Vietnamese. "
                "Keep your response friendly, brief, and conversational. "
                "Ask how you can assist them today. "
                "Ensure your tone matches the formality level they used (formal vs casual)."
            )
            logger.info(f"Detected Vietnamese greeting or name reference: '{message_str}'")
        elif has_teaching_markers or (len(message_str.split()) > 20 and "?" not in message_str):
            response_strategy = "TEACHING_INTENT"
            strategy_instructions = (
                "Recognize this message as TEACHING INTENT where the user is sharing knowledge with you. "
                "Your goal is to synthesize this knowledge for future use and demonstrate understanding. "
                
                "1. Begin by acknowledging their teaching with appreciation. "
                "2. Synthesize their input into a structured, comprehensive understanding. "
                "3. Organize the information with clear steps, examples, or practical applications. "
                "4. Rephrase any ambiguous terms or concepts for clarity. "
                "5. Highlight the key principles and practical takeaways. "
                "6. Ensure your response demonstrates how this knowledge could be applied. "
                "7. If appropriate, verify your understanding by restating core concepts. "
                "8. Ask a thoughtful follow-up question that demonstrates engagement. "
                
                "This synthesis approach helps create high-quality, reusable knowledge for future users."
            )
        
        prompt = f"""You are Ami, a conversational AI that understands topics deeply and drives discussions toward closure.

                **Identity Awareness**:
                - Your name is "Ami" - acknowledge when users call you by name
                - Notice when users refer to you as AI, assistant, bot, or similar terms
                - Recognize context clues that indicate the user is speaking directly to you
                - Maintain your identity consistently throughout the conversation
                - Do not explicitly state "My name is Ami" unless directly asked

                **Input**:
                - CURRENT MESSAGE: {message_str}
                - CONVERSATION HISTORY: {conversation_context}
                - TIME: {temporal_context}
                - EXISTING KNOWLEDGE: {knowledge_context}
                - RESPONSE STRATEGY: {response_strategy}
                - PRIOR TOPIC: {core_prior_topic}
                - USER ID: {user_id}

                **Response Approach**:
                {strategy_instructions}

                **Tools**:
                - knowledge_query: Query the knowledge base with query (required), user_id (required), context, thread_id, topic, top_k, min_similarity
                - save_knowledge: Save knowledge with user_id (required), query/content, thread_id, topic, categories

                **Instructions**:
                1. **Intent Classification**: 
                   - Determine whether the user is asking for information (query intent) or providing information (teaching intent)
                   - Base your classification on the semantic meaning and communicative purpose of the message
                   - For teaching intent, look for explanatory content, new information, or instructional tone
                   - For query intent, look for questions or requests for information
                   - Set has_teaching_intent=true when you detect teaching intent
                   - Use EXISTING KNOWLEDGE for queries when available
                   - Match the user's language choice (Vietnamese/English)
                   - For closing messages, set intent_type="closing" and respond with a polite farewell
                   - When the user addresses you as "Ami" or refers to you as an AI, acknowledge this in your response naturally
                   - Consider references to your identity or role as indicators of direct address
                   
                   - When handling TEACHING INTENT:
                     * Synthesize the input into a comprehensive practical understanding
                     * Structure the knowledge for future application (how to use this information)
                     * Rephrase any ambiguous terms, sentences, or paragraphs for clarity
                     * Organize information with clear steps, examples, or use cases when applicable
                     * Include contextual understanding (when/where/how to apply this knowledge)
                     * Highlight key principles rather than just recording facts
                     * Verify your understanding by restating core concepts in different terms
                     * Expand abbreviations and domain-specific terminology
                     * Ensure the response demonstrates how to apply this knowledge in future scenarios

                2. **Priority Topics**:
                   - Identify topics of special importance to the business domain
                   - When these topics are discussed, set is_priority_topic=true and include the topic name
                   - Consider the domain context when determining topic priority

                3. **Knowledge Management**:
                   - Recommend saving knowledge (should_save_knowledge=true) when:
                     * The message contains teaching intent
                     * The information appears valuable for future reference
                     * The content is well-structured or information-rich
                
                4. **Response Confidence**:
                   - For HIGH confidence queries (similarity >0.7):
                     * Demonstrate comprehensive understanding
                     * Speak confidently and authoritatively about the topic
                     * Present a thorough, well-structured response using the retrieved knowledge
                     * Connect concepts and provide additional context where appropriate
                   
                   - For MEDIUM confidence queries (similarity 0.35-0.7):
                     * Present the knowledge you have but express some uncertainty
                     * Acknowledge limitations in your understanding
                     * End with a specific question asking for clarification or confirmation
                     * Use phrases like "Based on what I understand..." or "I believe that..."
                   
                   - For LOW confidence queries (similarity <0.35):
                     * Clearly state that you don't have sufficient knowledge on this topic
                     * Ask if the user would like to teach you about this topic
                     * Invite them to add knowledge for future reference
                     * Frame it as an opportunity: "Would you mind sharing your knowledge about [topic]?"
                   
                   - When responding with MULTIPLE confidence levels:
                     * Structure your response in order of confidence (high → medium → low)
                     * For high confidence topics, provide detailed explanations
                     * For medium confidence topics, present what you know and ask for confirmation
                     * For low confidence topics, acknowledge knowledge gaps and request information
                     * Maintain a cohesive flow between different confidence sections
                     * Prioritize responding to the user's most important query first

                5. **Relational Dynamics**:
                   - Match the user's communication style and level of formality
                   - Maintain consistent linguistic patterns throughout the conversation
                   - Respect cultural and linguistic conventions in how you address the user
                   - Preserve the established relationship dynamic in your responses
                   - If addressed by name "Ami" or as an AI, subtly acknowledge this in your response
                   - Adapt your response style based on how directly the user is engaging with you
                   - For personal questions about your identity, provide concise, truthful answers without long explanations

                6. **Output Format**:
                   - Respond directly and concisely in the user's language (no prefix or labels)
                   - <knowledge_queries>["query1", "query2", "query3"]</knowledge_queries>
                   - <tool_calls>[{{"name": "tool_name", "parameters": {{...}}}}]</tool_calls> (if needed)
                   - <evaluation>{{"has_teaching_intent": true/false, "is_priority_topic": true/false, "priority_topic_name": "topic_name", "should_save_knowledge": true/false, "intent_type": "query/teaching/confirmation/follow-up", "name_addressed": true/false, "ai_referenced": true/false}}</evaluation>

                Maintain topic continuity, ensure proper JSON formatting, and include user_id in all tool calls.
                """
        try:
            response = await LLM.ainvoke(prompt)
            logger.info(f"LLM response generated with similarity score: {similarity_score}")
            
            content = response.content.strip()
            tool_calls = []
            evaluation = {"has_teaching_intent": False, "is_priority_topic": False, "priority_topic_name": "", "should_save_knowledge": False, "intent_type": "query", "name_addressed": False, "ai_referenced": False}
            
            # Extract tool calls if present
            if "<tool_calls>" in content:
                tool_section = re.search(r'<tool_calls>(.*?)</tool_calls>', content, re.DOTALL)
                if tool_section:
                    try:
                        tool_calls = json.loads(tool_section.group(1).strip())
                        content = re.sub(r'<tool_calls>.*?</tool_calls>', '', content, flags=re.DOTALL).strip()
                        logger.info(f"Extracted {len(tool_calls)} tool calls")
                    except json.JSONDecodeError:
                        logger.warning("Failed to parse tool calls")
            
            # Extract evaluation if present
            if "<evaluation>" in content:
                eval_section = re.search(r'<evaluation>(.*?)</evaluation>', content, re.DOTALL)
                if eval_section:
                    try:
                        evaluation = json.loads(eval_section.group(1).strip())
                        content = re.sub(r'<evaluation>.*?</evaluation>', '', content, flags=re.DOTALL).strip()
                        logger.info(f"Extracted LLM evaluation: {evaluation}")
                    except json.JSONDecodeError:
                        logger.warning("Failed to parse evaluation")
            
            if content.startswith('{') and '"message"' in content:
                try:
                    parsed_json = json.loads(content)
                    if isinstance(parsed_json, dict) and "message" in parsed_json:
                        content = parsed_json["message"]
                        logger.info("Extracted message from JSON response")
                except Exception as json_error:
                    logger.warning(f"Failed to parse JSON response: {json_error}")
            
            # Ensure closing messages get a response even if empty
            if response_strategy == "CLOSING" and (not content or content.isspace()):
                # Default closing message if the LLM didn't provide one
                if "vietnamese" in message_str.lower() or any(vn_word in message_str.lower() for vn_word in ["tạm biệt", "cảm ơn", "hẹn gặp", "thế thôi"]):
                    content = "Vâng, cảm ơn bạn đã trao đổi. Hẹn gặp lại bạn lần sau nhé!"
                else:
                    content = "Thank you for the conversation. Have a great day and I'm here if you need anything else!"
                logger.info("Added default closing response for empty LLM response")
            
            # Ensure unclear or short queries also get a helpful response when content is empty
            elif (not content or content.isspace()):
                # Check if message is short (1-2 words) or unclear
                is_short_message = len(message_str.strip().split()) <= 2
                
                # Default response for short/unclear messages
                if "vietnamese" in message_str.lower() or any(vn_word in message_str.lower() for vn_word in ["anh", "chị", "bạn", "cô", "ông", "xin", "vui lòng"]):
                    content = f"Xin lỗi, tôi không hiểu rõ câu hỏi '{message_str}'. Bạn có thể chia sẻ thêm thông tin hoặc đặt câu hỏi cụ thể hơn được không?"
                else:
                    content = f"I'm sorry, I didn't fully understand your message '{message_str}'. Could you please provide more details or ask a more specific question?"
                
                logger.info(f"Added default response for empty LLM response to short/unclear query: '{message_str}'")
            
            return {
                "status": "success",
                "message": content,
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "user_id": user_id,
                    "additional_knowledge_found": bool(knowledge_context),
                    "similarity_score": similarity_score,
                    "response_strategy": response_strategy,
                    "core_prior_topic": core_prior_topic,
                    "tool_calls": tool_calls,
                    "has_teaching_intent": evaluation.get("has_teaching_intent", False),
                    "is_priority_topic": evaluation.get("is_priority_topic", False),
                    "priority_topic_name": evaluation.get("priority_topic_name", ""),
                    "should_save_knowledge": evaluation.get("should_save_knowledge", False)
                }
            }
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {
                "status": "error",
                "message": "Tôi xin lỗi, nhưng tôi gặp lỗi khi xử lý yêu cầu của bạn. Vui lòng thử lại."
            }

    def _create_background_task(self, coro):
        """Create and track a background task."""
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        # Add callback to remove task from set when done
        task.add_done_callback(self._background_tasks.discard)
        return task

    async def cleanup(self):
        """Wait for all background tasks to complete."""
        if not self._background_tasks:
            return
            
        logger.info(f"Waiting for {len(self._background_tasks)} background tasks to complete...")
        # Create a copy as the set may be modified during iteration
        pending_tasks = list(self._background_tasks)
        
        if not pending_tasks:
            return
            
        done, pending = await asyncio.wait(
            pending_tasks, 
            timeout=5.0,  # Increased timeout to 5 seconds for tasks to complete
            return_when=asyncio.ALL_COMPLETED
        )
        
        if pending:
            logger.warning(f"{len(pending)} background tasks did not complete in time and will be cancelled")
            for task in pending:
                if not task.done():
                    task.cancel()
                    try:
                        # Try to await cancelled tasks to handle cancellation properly
                        await asyncio.wait_for(task, timeout=0.5)
                    except (asyncio.CancelledError, asyncio.TimeoutError):
                        pass
                    
        # Clean up the set
        self._background_tasks.clear()
        logger.info("Background task cleanup completed")

    async def _background_save_knowledge(self, input_text: str, user_id: str, bank_name: str, 
                                         thread_id: Optional[str] = None, topic: Optional[str] = None, 
                                         categories: List[str] = ["general"], ttl_days: Optional[int] = 365) -> None:
        """Execute save_knowledge in a separate background task."""
        try:
            logger.info(f"Starting background save_knowledge task for user {user_id}")
            # Use a shorter timeout for saving knowledge to avoid hanging tasks
            try:
                success = await asyncio.wait_for(
                    save_knowledge(
                        input=input_text,
                        user_id=user_id,
                        bank_name=bank_name,
                        thread_id=thread_id,
                        topic=topic,
                        categories=categories,
                        ttl_days=ttl_days  # Add TTL for data expiration
                    ),
                    timeout=3.0  # 5-second timeout for database operations
                )
                logger.info(f"Background save_knowledge {'completed successfully' if success else 'failed'}")
                return success
            except asyncio.TimeoutError:
                logger.warning(f"Background save_knowledge timed out for user {user_id}")
                return False
        except Exception as e:
            logger.error(f"Error in background save_knowledge: {str(e)}")
            import traceback
            logger.error(f"Stack trace: {traceback.format_exc()}")
            return False

    def _detect_follow_up(self, message: str, prior_topic: str = "") -> Dict[str, bool]:
        """
        Detect whether a message is a follow-up to a previous topic.
        
        Args:
            message: The current message to check
            prior_topic: The previous topic or message for context
            
        Returns:
            Dictionary with detection results containing:
            - is_confirmation: Whether the message confirms something
            - is_follow_up: Whether the message is a follow-up
            - has_pattern_match: Whether the message matches follow-up patterns
            - topic_overlap: Whether there's overlap with the prior topic
        """
        # More comprehensive confirmation keywords in both English and Vietnamese
        confirmation_keywords = [
            # Vietnamese affirmatives
            "có", "đúng", "đúng rồi", "chính xác", "phải", "ừ", "ừm", "vâng", "dạ", "ok", "được", "đồng ý",
            # English affirmatives
            "yes", "yeah", "correct", "right", "sure", "okay", "ok", "indeed", "exactly", "agree", "true",
            # Action-oriented confirmations
            "explore further", "tell me more", "continue", "go on", "proceed", "next", "more", "and then",
            # Vietnamese action confirmations
            "tiếp tục", "kể tiếp", "nói thêm", "thêm nữa", "và sau đó", "tiếp theo"
        ]
        is_confirmation = any(keyword.lower() in message.lower() for keyword in confirmation_keywords)
        
        # Enhanced follow-up detection with more patterns
        follow_up_patterns = [
            # Direct references
            r'\b(nhóm này|this group|that group|these groups|those groups)\b',
            # Questions about previously mentioned topics
            r'\b(vậy thì sao|thế còn|còn về|về điểm này|about this|what about|regarding this|related to this)\b',
            # Continuation markers
            r'\b(tiếp theo|tiếp tục|continue with|proceed with|more about|elaborate on)\b',
            # Implicit references
            r'\b(trong trường hợp đó|in that case|if so|if that\'s the case)\b',
            # Direct anaphoric references
            r'\b(nó|they|them|it|those|these|that|this)\b\s+(is|are|như thế nào|làm sao|means|works)'
        ]
        has_pattern_match = any(re.search(pattern, message.lower(), re.IGNORECASE) for pattern in follow_up_patterns)
        
        # Check for short responses that often indicate follow-ups
        is_short_response = len(message.strip().split()) <= 5
        
        # Check if message is primarily composed of question words (often follow-ups)
        question_starters = ["why", "how", "what", "when", "where", "who", "which", "tại sao", "làm sao", "khi nào", "ở đâu", "ai", "cái nào"]
        starts_with_question = any(message.lower().strip().startswith(q) for q in question_starters)
        
        # Semantic topic continuity
        topic_overlap = False
        if prior_topic:
            # Check if significant words from the message appear in prior topic
            msg_words = set(re.findall(r'\b\w{4,}\b', message.lower()))  # Words with 4+ chars
            topic_words = set(re.findall(r'\b\w{4,}\b', prior_topic.lower()))
            common_words = msg_words.intersection(topic_words)
            topic_overlap = len(common_words) >= 1 or message.lower().strip() in prior_topic.lower()
        
        # Combined follow-up detection
        is_follow_up = (
            is_confirmation or 
            has_pattern_match or 
            (is_short_response and prior_topic) or  # Short responses with context are likely follow-ups
            (starts_with_question and prior_topic) or  # Questions after context are likely follow-ups
            topic_overlap
        )
        
        return {
            "is_confirmation": is_confirmation,
            "is_follow_up": is_follow_up,
            "has_pattern_match": has_pattern_match,
            "topic_overlap": topic_overlap
        }

async def process_llm_with_tools(
        self,
        user_message: str,
        conversation_history: List[Dict],
        state: Union[Dict, str],
        graph_version_id: str,
        thread_id: Optional[str] = None
    ) -> AsyncGenerator[Union[str, Dict], None]:
        """Process user message with tools."""
        if not user_message:
            logger.error("Empty user_message")
            yield {"status": "error", "message": "Empty message provided"}
            return

        if isinstance(state, str):
            user_id = state
            state = {"user_id": state}
        else:
            user_id = state.get('user_id', 'unknown')
        if not user_id:
            logger.warning("Empty user_id, defaulting to 'unknown'")
            user_id = "unknown"
        logger.info(f"Processing message for user {user_id}")

        conversation_context = ""
        if conversation_history:
            recent_messages = []
            message_count = 0
            max_messages = 50
            for msg in reversed(conversation_history):
                try:
                    role = msg.get("role", "").lower() if isinstance(msg, dict) else ""
                    content = msg.get("content", "") if isinstance(msg, dict) else str(msg)
                    if role and content:
                        if role in ["assistant", "ai"]:
                            recent_messages.append(f"AI: {content.strip()}")
                            message_count += 1
                        elif role in ["user", "human"]:
                            recent_messages.append(f"User: {content.strip()}")
                            message_count += 1
                    if message_count >= max_messages:
                        if len(conversation_history) > max_messages:
                            recent_messages.append(f"[Note: Conversation history truncated. Total messages: {len(conversation_history)}]")
                        break
                except Exception as e:
                    logger.warning(f"Error processing message in conversation history: {e}")
                    continue
            if recent_messages:
                header = "==== CONVERSATION HISTORY (CHRONOLOGICAL ORDER) ====\n"
                conversation_history_text = "\n\n".join(reversed(recent_messages))
                separator = "\n==== CURRENT INTERACTION ====\n"
                conversation_context = f"{header}{conversation_history_text}{separator}\nUser: {user_message}\n"
                logger.info(f"Added {len(recent_messages)} messages from conversation history")
        
        if 'learning_processor' not in state:
            learning_processor = LearningProcessor()
            await learning_processor.initialize()
            state['learning_processor'] = learning_processor
        else:
            learning_processor = state['learning_processor']
        
        state['graph_version_id'] = graph_version_id
        
        try:
            response = await learning_processor.process_incoming_message(
                user_message, 
                conversation_context, 
                user_id,
                thread_id
            )
            
            message_content = response["message"] if isinstance(response, dict) and "message" in response else str(response)
            message_content = re.split(r'<knowledge_queries>', message_content)[0].strip()
            logger.info("Stripped knowledge_queries from message for frontend")
            
            tool_results = []
            if "metadata" in response and "tool_calls" in response["metadata"]:
                for tool_call in response["metadata"]["tool_calls"]:
                    if isinstance(tool_call, dict) and "name" in tool_call and "parameters" in tool_call:
                        tool_result = await self.execute_tool(
                            tool_name=tool_call["name"],
                            parameters=tool_call["parameters"]
                        )
                        tool_results.append(tool_result)
                        yield tool_result
                        logger.info(f"Executed tool {tool_call['name']} with result: {tool_result}")
                    else:
                        logger.warning(f"Invalid tool call format: {tool_call}")

            # Force cleanup of tasks before returning final result
            if 'learning_processor' in state:
                await state['learning_processor'].cleanup()
                
            yield {"status": "success", "message": message_content}
            state.setdefault("messages", []).append({"role": "assistant", "content": message_content})

            if tool_results:
                state["last_tool_results"] = tool_results
        except Exception as e:
            logger.error(f"Error in CoT processing: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            error_response = {"status": "error", "message": f"Error: {str(e)}"}
            yield error_response
        finally:
            # Make sure we clean up tasks even in case of exception
            if 'learning_processor' in state:
                await state['learning_processor'].cleanup()
                
        yield {"state": state}

async def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool with enhanced parameter validation and metadata enrichment."""
        logger.info(f"Executing tool: {tool_name} with parameters: {parameters}")
        try:
            user_id = parameters.get("user_id", "")
            if not user_id:
                logger.warning("Missing user_id in tool call, defaulting to 'unknown'")
                user_id = "unknown"

            if tool_name == "knowledge_query":
                query = parameters.get("query", "")
                if not query:
                    return {"status": "error", "message": "Missing required parameter: query"}
                context = parameters.get("context", "")
                thread_id = parameters.get("thread_id", None)
                topic = parameters.get("topic", None)
                
                # Determine if we should check health bank based on content
                is_health_topic = any(term in query.lower() or term in context.lower() 
                    for term in ["rối loạn cương dương", "xuất tinh sớm", "phân nhóm khách hàng", 
                                "phân tích chân dung khách hàng", "chân dung khách hàng", "customer profile"])
                
                bank_name = "conversation"
                
                # Try querying the specified bank
                results = await query_knowledge(
                    query=query,
                    bank_name=bank_name,
                    user_id=user_id,
                    thread_id=None,  # Remove thread_id restriction to find more results
                    topic=None,      # Remove topic restriction
                    top_k=10,
                    min_similarity=0.2  # Lower threshold for better matching
                )
                return {
                    "status": "success",
                    "message": f"Queried knowledge for '{query}' from {bank_name} bank" + 
                              (", then checked health bank" if bank_name != "health" and is_health_topic else "") +
                              (", then checked default bank" if bank_name == "health" and not results else ""),
                    "data": results
                }

            elif tool_name == "save_knowledge":
                query = parameters.get("query", "")
                content = parameters.get("content", "")
                input_text = f"{query} {content}".strip() if query and content else query or content
                if not input_text:
                    return {"status": "error", "message": "Missing required parameter: query or content"}
                
                thread_id = parameters.get("thread_id", None)
                topic = parameters.get("topic", None)
                categories = parameters.get("categories", ["general"])
                
                if not categories or categories == ["general"]:
                    categories = ["human shared"]
                
                # Add teaching_intent category for explicit knowledge saves
                if "teaching_intent" not in categories:
                    categories.append("teaching_intent")
                    
                bank_name = "conversation"
                
                # Format as a teaching entry for consistency with combined knowledge format
                if not input_text.startswith("User:"):
                    input_text = f"AI: {input_text}"

                # Run save_knowledge in background task
                self._create_background_task(self._background_save_knowledge(
                    input_text=input_text,
                    user_id=user_id,
                    bank_name=bank_name,
                    thread_id=thread_id,
                    topic=topic,
                    categories=categories,
                    ttl_days=365  # 365 days TTL for knowledge
                ))
                
                return {
                    "status": "success",
                    "message": f"Save knowledge task initiated for '{input_text[:50]}...'"
                }

            return {"status": "error", "message": f"Unknown tool: {tool_name}"}
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {str(e)}")
            return {"status": "error", "message": f"Tool execution failed: {str(e)}"}