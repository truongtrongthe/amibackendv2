import json
import asyncio
from typing import Dict, List, Any, Optional, Union, Set
from datetime import datetime
import re
import pytz
from langchain_openai import ChatOpenAI
from pccontroller import save_knowledge, query_knowledge, query_knowledge_from_graph
from utilities import logger, EMBEDDINGS

# Initialize LLM for support functions
LLM = ChatOpenAI(model="gpt-4o", streaming=False, temperature=0.01)

class LearningSupport:
    """Support class containing utility functions for LearningProcessor."""
    
    def __init__(self, learning_processor):
        self.learning_processor = learning_processor
        
    async def search_knowledge(self, message: Union[str, List], conversation_context: str = "", user_id: str = "unknown", thread_id: Optional[str] = None) -> Dict[str, Any]:
        """Search for relevant knowledge based on message and context."""
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
            prior_messages = []
            if conversation_context:
                user_messages = re.findall(r'User: (.*?)(?:\n\n|$)', conversation_context, re.DOTALL)
                ai_messages = re.findall(r'AI: (.*?)(?:\n\n|$)', conversation_context, re.DOTALL)
                logger.info(f"Found {len(user_messages)} user messages in context")
                if user_messages:
                    prior_messages = user_messages[:-1]  # All but the current message
                    prior_topic = user_messages[-2].strip() if len(user_messages) > 1 else ""
                    logger.info(f"Extracted prior topic: {prior_topic[:50]}")
                if ai_messages:
                    prior_knowledge = ai_messages[-1].strip()

            # Use the new conversation flow detection instead of pattern matching
            flow_result = await self.detect_conversation_flow(
                message=primary_query,
                prior_messages=prior_messages,
                conversation_context=conversation_context
            )
            
            flow_type = flow_result.get("flow_type", "NEW_TOPIC")
            flow_confidence = flow_result.get("confidence", 0.5)
            
            is_follow_up = flow_type in ["FOLLOW_UP", "CONFIRMATION"]
            is_practice_request = flow_type == "PRACTICE_REQUEST"
            
            logger.info(f"Conversation flow: {flow_type} (confidence: {flow_confidence})")
            
            if is_follow_up and prior_topic:
                queries.append(prior_topic)
                logger.info(f"Follow-up detected, reusing prior topic: {prior_topic[:50]}")
                similarity = 0.7
                knowledge_context = prior_knowledge
            elif is_practice_request and prior_knowledge:
                # For practice requests, prioritize last AI message as knowledge
                queries.append(primary_query)
                queries.append(prior_topic if prior_topic else primary_query)
                logger.info(f"Practice request detected, using previous AI knowledge as foundation")
                # Higher confidence for practice scenarios (we're quite certain about our prior knowledge)
                similarity = 0.85
                knowledge_context = prior_knowledge
                
                # For practice requests, log specific phrases detected
                practice_indicators = []
                if "thử" in primary_query.lower() and ("xem" in primary_query.lower() or "nào" in primary_query.lower()):
                    practice_indicators.append("thử...xem/nào")
                if "áp dụng" in primary_query.lower():
                    practice_indicators.append("áp dụng")
                if practice_indicators:
                    logger.info(f"Practice request indicators: {', '.join(practice_indicators)}")
                
                # Return early with high confidence for clear practice requests
                if flow_confidence > 0.8:
                    logger.info(f"High confidence practice request - prioritizing previous knowledge")
                    return {
                        "knowledge_context": knowledge_context,
                        "similarity": similarity,
                        "query_count": 1,
                        "queries": queries,
                        "original_query": primary_query,
                        "query_results": [{"raw": knowledge_context, "score": similarity, "metadata": {"practice_request": True}}],
                        "prior_data": {"topic": prior_topic, "knowledge": prior_knowledge},
                        "metadata": {"similarity": similarity, "vibe_score": 1.1, "flow_type": flow_type}
                    }
            else:
                queries.append(primary_query)
                similarity = 0.0
                knowledge_context = ""

            temp_response = await self.active_learning(primary_query, conversation_context, {}, user_id, {})
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
                        temp_response = await self.active_learning(primary_query, conversation_context, {}, user_id, {})
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
                *(query_knowledge_from_graph(
                    query=query,
                    graph_version_id=self.learning_processor.graph_version_id,
                    user_id=user_id,
                    thread_id=None,  # Remove thread_id restriction to find more results
                    topic=None,      # Remove topic restriction
                    top_k=100,
                    min_similarity=0.2,  # Lower threshold for better matching
                    exclude_categories=["ai_synthesis"]  # Exclude AI synthesis content
                ) for query in queries),
                return_exceptions=True
            )

            # Store all query results
            all_query_results = []
            best_results = []  # Change from single best_result to list of best_results
            highest_similarities = []  # Store top 3 similarity scores
            knowledge_contexts = []  # Store top 3 knowledge contexts
            
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
                
                # Process all results from a query, not just the first result
                for result_item in results:
                    knowledge_content = result_item["raw"]
                    
                    # Extract just the User portion if this is a combined knowledge entry
                    if knowledge_content.startswith("User:") and "\n\nAI:" in knowledge_content:
                        user_part = re.search(r'User:(.*?)(?=\n\nAI:)', knowledge_content, re.DOTALL)
                        if user_part:
                            knowledge_content = user_part.group(1).strip()
                            logger.info(f"Extracted User portion from combined knowledge")
                    
                    # Evaluate context relevance between query and retrieved knowledge
                    context_relevance = await self.evaluate_context_relevance(primary_query, knowledge_content)
                    
                    # Calculate adjusted similarity score with context relevance factor
                    query_similarity = result_item["score"]
                    adjusted_similarity = query_similarity * (1.0 + 0.5 * context_relevance)  # Range between 100%-150% of original score
                    
                    # Add context relevance information to result metadata
                    result_item["context_relevance"] = context_relevance
                    result_item["adjusted_similarity"] = adjusted_similarity
                    result_item["query"] = query
                    
                    all_query_results.append(result_item)  # Store all results 
                    
                    logger.info(f"Result for query '{query[:30]}...' yielded similarity: {query_similarity}, adjusted: {adjusted_similarity}, context relevance: {context_relevance}, content: '{knowledge_content[:50]}...'")
                    
                    # Track the top 5 results using adjusted similarity instead of top 3
                    if not highest_similarities or adjusted_similarity > min(highest_similarities) or len(highest_similarities) < 5:
                        # Add the new result to our collections
                        if len(highest_similarities) < 5:
                            highest_similarities.append(adjusted_similarity)
                            best_results.append(result_item)
                            knowledge_contexts.append(knowledge_content)
                        else:
                            # Find the minimum similarity in our top 5
                            min_index = highest_similarities.index(min(highest_similarities))
                            # Replace it with the new result
                            highest_similarities[min_index] = adjusted_similarity
                            best_results[min_index] = result_item
                            knowledge_contexts[min_index] = knowledge_content
                        
                        logger.info(f"Updated top 5 knowledge results with adjusted similarity: {adjusted_similarity}, context relevance: {context_relevance}")

            # Apply regular boost for priority topics
            if any(term in primary_query.lower() or (prior_topic and term in prior_topic.lower()) 
                   for term in ["mục tiêu", "goals", "active learning", "phân nhóm", "phân tích chân dung", "chân dung khách hàng"]):
                vibe_score = 1.1
                highest_similarities = [sim * vibe_score for sim in highest_similarities]
                logger.info(f"Applied vibe score {vibe_score} for priority topic")
            else:
                vibe_score = 1.0

            # Filter out None results
            valid_query_results = [result for result in all_query_results if result is not None]
            
            # Use the highest similarity from our top results
            similarity = max(highest_similarities) if highest_similarities else 0.0
            
            # Debug logging for similarity calculation
            if highest_similarities:
                logger.info(f"DEBUG: highest_similarities list: {highest_similarities}")
                logger.info(f"DEBUG: max(highest_similarities): {max(highest_similarities)}")
                logger.info(f"DEBUG: Using similarity: {similarity}")
            
            # Combine knowledge contexts for the top results
            combined_knowledge_context = ""
            if knowledge_contexts:
                # Sort by similarity to present most relevant first
                sorted_results = sorted(zip(best_results, highest_similarities, knowledge_contexts), 
                                      key=lambda pair: pair[1], reverse=True)
                
                # Format response sections
                knowledge_response_sections = []
                knowledge_response_sections.append("KNOWLEDGE RESULTS:")
                
                # Log number of results
                result_count = min(len(sorted_results), 5)  # Maximum 5 results
                logger.info(f"Adding {result_count} knowledge items to response")
                
                # Add each result with numbering
                for i, (result, item_similarity, content) in enumerate(sorted_results[:result_count], 1):
                    query = result.get("query", "unknown query")
                    score = result.get("score", 0.0)
                    
                    # Remove AI: or AI Synthesis: prefix if present
                    if content.startswith("AI: "):
                        content = content[4:]
                    elif content.startswith("AI Synthesis: "):
                        content = content[14:]
                    
                    # Add numbered result
                    knowledge_response_sections.append(
                        f"[{i}] Query: '{query}' (score: {score:.2f})\n{content}"
                    )
                
                # Create combined_knowledge_context from all sections
                combined_knowledge_context = "\n\n".join(knowledge_response_sections)
                
                # Log to check number of sections
                logger.info(f"Created knowledge response with {len(knowledge_response_sections) - 1} items")
            else:
                # If no knowledge contexts, set an empty string
                combined_knowledge_context = ""
                logger.info("No knowledge items found, using empty knowledge context")
            
            logger.info(f"Final similarity: {similarity} from {query_count} queries, found {len(valid_query_results)} valid results, using top {len(knowledge_contexts)} for response")
            return {
                "knowledge_context": combined_knowledge_context,
                "similarity": similarity,
                "query_count": query_count,
                "queries": queries,
                "original_query": primary_query,  # Add the original query for reference
                "query_results": valid_query_results,
                "top_results": best_results,  # Add top results
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

    async def evaluate_context_relevance(self, user_input: str, retrieved_knowledge: str) -> float:
        """
        Evaluate the relevance between user input and retrieved knowledge.
        Returns a score between 0.0 and 1.0 indicating relevance.
        """
        try:
            # Method 1: Use embeddings similarity
            user_embedding = await EMBEDDINGS.aembed_query(user_input)
            knowledge_embedding = await EMBEDDINGS.aembed_query(retrieved_knowledge)
            
            # Calculate cosine similarity
            dot_product = sum(a * b for a, b in zip(user_embedding, knowledge_embedding))
            user_norm = sum(a * a for a in user_embedding) ** 0.5
            knowledge_norm = sum(b * b for b in knowledge_embedding) ** 0.5
            
            if user_norm * knowledge_norm == 0:
                similarity = 0
            else:
                similarity = dot_product / (user_norm * knowledge_norm)
            
            # Method 2: Let LLM evaluate relevance if similarity is in ambiguous range
            if 0.3 <= similarity <= 0.7:
                # Only use LLM evaluation for ambiguous cases to save API calls
                prompt = f"""
                Evaluate the relevance between USER INPUT and KNOWLEDGE on a scale of 0-10.
                
                USER INPUT:
                {user_input}
                
                KNOWLEDGE:
                {retrieved_knowledge}
                
                Consider:
                - Topic alignment (not just keywords)
                - Whether the knowledge addresses the input's intent
                - Practical usefulness of the knowledge for the input
                
                Return ONLY a number between 0-10.
                """
                
                try:
                    llm_response = await LLM.ainvoke(prompt)
                    llm_score_text = llm_response.content.strip()
                    
                    # Extract just the number from potential additional text
                    import re
                    score_match = re.search(r'(\d+(\.\d+)?)', llm_score_text)
                    if score_match:
                        llm_score = float(score_match.group(1))
                        # Normalize to 0-1 range
                        llm_score = min(10, max(0, llm_score)) / 10
                        
                        # Combine with more weight on embedding similarity (50/50 instead of 30/70)
                        combined_score = 0.5 * similarity + 0.5 * llm_score
                        logger.info(f"Context relevance: embedding={similarity:.2f}, LLM={llm_score:.2f}, combined={combined_score:.2f}")
                        return combined_score
                except Exception as e:
                    logger.warning(f"LLM relevance evaluation failed: {str(e)}. Falling back to embedding similarity.")
            
            logger.info(f"Context relevance from embedding similarity: {similarity:.2f}")
            return similarity
            
        except Exception as e:
            logger.error(f"Error in context relevance evaluation: {str(e)}")
            # Default to medium relevance on error to avoid blocking the flow
            return 0.5

    async def detect_conversation_flow(self, message: str, prior_messages: List[str], conversation_context: str) -> Dict[str, Any]:
        """
        Use LLM to analyze conversation flow and detect the relationship between messages.
        
        Args:
            message: Current message to analyze
            prior_messages: Previous messages for context (most recent first)
            conversation_context: Full conversation history
            
        Returns:
            Dictionary with flow type, confidence, and other analysis details
        """
        # Skip LLM call if no context is available
        if not prior_messages:
            return {
                "flow_type": "NEW_TOPIC", 
                "confidence": 0.9,
                "reasoning": "No prior messages"
            }

        # First do a quick check for practice request patterns in Vietnamese
        practice_patterns = [
            r'(?:thử|áp dụng).*(?:xem|nào)',  # "thử...xem", "áp dụng...xem nào"
            r'(?:làm thử|thử làm)',           # "làm thử", "thử làm" 
            r'ví dụ.*(?:đi|nào)',             # "ví dụ...đi", "ví dụ...nào"
            r'minh họa',                      # "minh họa"
            r'thực hành'                      # "thực hành"
        ]
        
        # Direct pattern matching for clear practice requests in Vietnamese
        if any(re.search(pattern, message.lower()) for pattern in practice_patterns):
            logger.info(f"Direct pattern match for PRACTICE_REQUEST: '{message}'")
            return {
                "flow_type": "PRACTICE_REQUEST",
                "confidence": 0.95,
                "reasoning": "Direct Vietnamese practice request pattern detected"
            }
            
        # Get the most recent prior message for context
        prior_message = prior_messages[0] if prior_messages else ""
        
        # Prepare a context sample that's not too long (last 800 chars max)
        context_sample = conversation_context
        if len(context_sample) > 800:
            context_sample = "..." + context_sample[-800:]
        
        # Enhanced prompt for better flow detection, especially for Vietnamese
        prompt = f"""
        Analyze this conversation flow. Your task is to determine the relationship between the CURRENT MESSAGE and previous messages.
        
        Classify the CURRENT MESSAGE into exactly ONE of these categories:
        
        1. FOLLOW_UP: Continuing or asking for more details about a previous topic
        2. CONFIRMATION: Agreement, acknowledgment, or confirmation of previous information
        3. PRACTICE_REQUEST: Asking to demonstrate, apply, or try knowledge previously shared
        4. CLOSING: Indicating the conversation is ending
        5. NEW_TOPIC: Starting a completely new conversation topic
        
        CRITICAL VIETNAMESE PATTERNS:
        - If "thử...xem" appears in any form, this is almost certainly a PRACTICE_REQUEST
        - If "áp dụng" appears with "xem" or "nào", this is a PRACTICE_REQUEST
        - "làm thử", "thử làm" indicate PRACTICE_REQUEST
        - "ví dụ", "minh họa" indicate PRACTICE_REQUEST (asking for example)
        - Short responses like "vâng", "đúng rồi", "được" usually mean CONFIRMATION
        - "tạm biệt", "hẹn gặp lại" indicate CLOSING
        
        PAY SPECIAL ATTENTION: The phrase "Em thử áp dụng..." or similar patterns STRONGLY indicate a PRACTICE_REQUEST where the user wants to demonstrate their knowledge.
        
        CONVERSATION CONTEXT:
        {context_sample}
        
        PREVIOUS MESSAGE:
        {prior_message}
        
        CURRENT MESSAGE:
        {message}
        
        Return ONLY a JSON object:
        {{"flow_type": "FOLLOW_UP|CONFIRMATION|PRACTICE_REQUEST|CLOSING|NEW_TOPIC", "confidence": [0-1.0], "reasoning": "brief explanation"}}
        """
        
        try:
            response = await LLM.ainvoke(prompt)
            content = response.content.strip()
            
            # Extract JSON from potential additional text
            import re
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                content = json_match.group(0)
                
            result = json.loads(content)
            
            # Log the result
            logger.info(f"Conversation flow detected: {result['flow_type']} (confidence: {result['confidence']})")
            return result
            
        except Exception as e:
            logger.warning(f"Error detecting conversation flow: {str(e)}")
            
            # Enhanced fallback detection for practice requests in Vietnamese
            lower_message = message.lower()
            if any(term in lower_message for term in ["thử", "áp dụng", "ví dụ", "minh họa", "thực hành"]):
                return {
                    "flow_type": "PRACTICE_REQUEST",
                    "confidence": 0.8,
                    "reasoning": "Fallback practice request detection"
                }
            
            # Simple fallback based on message length
            is_short_message = len(message.split()) < 5
            
            if is_short_message:
                return {
                    "flow_type": "FOLLOW_UP",
                    "confidence": 0.6,
                    "reasoning": "Short message fallback classification"
                }
            else:
                return {
                    "flow_type": "NEW_TOPIC",
                    "confidence": 0.5,
                    "reasoning": "Fallback classification due to error"
                }

    async def background_save_knowledge(self, input_text: str, title: str, user_id: str, bank_name: str, 
                                         thread_id: Optional[str] = None, topic: Optional[str] = None, 
                                         categories: List[str] = ["general"], ttl_days: Optional[int] = 365) -> Dict:
        """Execute save_knowledge in a separate background task."""
        try:
            logger.info(f"Starting background save_knowledge task for user {user_id}")
            # Use a shorter timeout for saving knowledge to avoid hanging tasks
            try:
                result = await asyncio.wait_for(
                    save_knowledge(
                        input=input_text,
                        title=title,
                        user_id=user_id,
                        bank_name=bank_name,
                        thread_id=thread_id,
                        topic=topic,
                        categories=categories,
                        ttl_days=ttl_days  # Add TTL for data expiration
                    ),
                    timeout=6.0  # Increased to 10-second timeout for database operations
                )
                logger.info(f"Background save_knowledge completed: {result}")
                return result if isinstance(result, dict) else {"success": bool(result)}
            except asyncio.TimeoutError:
                logger.warning(f"Background save_knowledge timed out for user {user_id}")
                return {"success": False, "error": "Save operation timed out"}
        except Exception as e:
            logger.error(f"Error in background save_knowledge: {str(e)}")
            import traceback
            logger.error(f"Stack trace: {traceback.format_exc()}")
            return {"success": False, "error": str(e)}

    def detect_follow_up(self, message: str, prior_topic: str = "") -> Dict[str, bool]:
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

    async def detect_follow_up_dynamic(self, message: str, prior_topic: str = "", conversation_history: List[str] = None) -> Dict[str, Any]:
        """
        Dynamic follow-up detection using embeddings, LLM analysis, and adaptive learning.
        
        Args:
            message: The current message to check
            prior_topic: The previous topic or message for context
            conversation_history: List of recent messages for context
            
        Returns:
            Dictionary with detection results containing:
            - is_confirmation: Whether the message confirms something
            - is_follow_up: Whether the message is a follow-up
            - confidence: Confidence score (0.0-1.0)
            - reasoning: Explanation of the decision
            - semantic_similarity: Embedding-based similarity score
            - linguistic_patterns: Detected linguistic patterns
        """
        try:
            # Initialize results
            results = {
                "is_confirmation": False,
                "is_follow_up": False,
                "confidence": 0.0,
                "reasoning": "",
                "semantic_similarity": 0.0,
                "linguistic_patterns": []
            }
            
            if not message.strip():
                return results
            
            # Method 1: Semantic Similarity Analysis
            semantic_similarity = 0.0
            if prior_topic:
                try:
                    message_embedding = await EMBEDDINGS.aembed_query(message)
                    topic_embedding = await EMBEDDINGS.aembed_query(prior_topic)
                    
                    # Calculate cosine similarity
                    dot_product = sum(a * b for a, b in zip(message_embedding, topic_embedding))
                    message_norm = sum(a * a for a in message_embedding) ** 0.5
                    topic_norm = sum(b * b for b in topic_embedding) ** 0.5
                    
                    if message_norm * topic_norm > 0:
                        semantic_similarity = dot_product / (message_norm * topic_norm)
                    
                    results["semantic_similarity"] = semantic_similarity
                    logger.info(f"Semantic similarity between message and prior topic: {semantic_similarity:.3f}")
                except Exception as e:
                    logger.warning(f"Error calculating semantic similarity: {e}")
            
            # Method 2: LLM-based Contextual Analysis
            llm_analysis = await self.analyze_follow_up_with_llm(message, prior_topic, conversation_history)
            
            # Method 3: Dynamic Pattern Detection
            linguistic_patterns = self.detect_linguistic_patterns(message, prior_topic)
            results["linguistic_patterns"] = linguistic_patterns
            
            # Method 4: Conversation Flow Analysis
            flow_indicators = self.analyze_conversation_flow_indicators(message, conversation_history)
            
            # Combine all methods with weighted scoring
            weights = {
                "semantic": 0.3,
                "llm": 0.4,
                "linguistic": 0.2,
                "flow": 0.1
            }
            
            # Calculate weighted confidence scores
            semantic_score = min(semantic_similarity * 2, 1.0)  # Scale up semantic similarity
            llm_score = llm_analysis.get("confidence", 0.0)
            linguistic_score = self.score_linguistic_patterns(linguistic_patterns)
            flow_score = flow_indicators.get("follow_up_probability", 0.0)
            
            # Weighted combination
            combined_confidence = (
                weights["semantic"] * semantic_score +
                weights["llm"] * llm_score +
                weights["linguistic"] * linguistic_score +
                weights["flow"] * flow_score
            )
            
            # Determine final classification
            is_follow_up = combined_confidence > 0.5
            is_confirmation = (
                llm_analysis.get("is_confirmation", False) or
                linguistic_patterns.get("confirmation_indicators", 0) > 2
            )
            
            # Generate reasoning
            reasoning_parts = []
            if semantic_similarity > 0.3:
                reasoning_parts.append(f"High semantic similarity ({semantic_similarity:.2f})")
            if llm_analysis.get("reasoning"):
                reasoning_parts.append(f"LLM: {llm_analysis['reasoning']}")
            if linguistic_patterns.get("strong_indicators"):
                reasoning_parts.append(f"Patterns: {', '.join(linguistic_patterns['strong_indicators'])}")
            
            results.update({
                "is_confirmation": is_confirmation,
                "is_follow_up": is_follow_up,
                "confidence": combined_confidence,
                "reasoning": "; ".join(reasoning_parts) if reasoning_parts else "Low confidence classification",
                "method_scores": {
                    "semantic": semantic_score,
                    "llm": llm_score,
                    "linguistic": linguistic_score,
                    "flow": flow_score
                }
            })
            
            logger.info(f"Dynamic follow-up detection: {is_follow_up} (confidence: {combined_confidence:.3f})")
            return results
            
        except Exception as e:
            logger.error(f"Error in dynamic follow-up detection: {e}")
            return {
                "is_confirmation": False,
                "is_follow_up": False,
                "confidence": 0.0,
                "reasoning": f"Error in analysis: {str(e)}",
                "semantic_similarity": 0.0,
                "linguistic_patterns": []
            }
    
    async def analyze_follow_up_with_llm(self, message: str, prior_topic: str, conversation_history: List[str] = None) -> Dict[str, Any]:
        """Use LLM to analyze if a message is a follow-up with contextual understanding."""
        try:
            # Prepare context
            context_parts = []
            if conversation_history:
                recent_context = " | ".join(conversation_history[-3:])  # Last 3 messages
                context_parts.append(f"Recent conversation: {recent_context}")
            if prior_topic:
                context_parts.append(f"Prior topic: {prior_topic}")
            
            context = "\n".join(context_parts) if context_parts else "No prior context"
            
            prompt = f"""
            Analyze whether the CURRENT MESSAGE is a follow-up to previous conversation.
            
            CONTEXT:
            {context}
            
            CURRENT MESSAGE:
            {message}
            
            Determine:
            1. Is this a follow-up to previous topics? (true/false)
            2. Is this a confirmation/agreement? (true/false)
            3. Confidence level (0.0-1.0)
            4. Brief reasoning
            
            Consider:
            - Semantic relationship to prior topics
            - Conversational flow and intent
            - Language-specific patterns (Vietnamese/English)
            - Implicit references and context dependencies
            
            Return JSON:
            {{"is_follow_up": boolean, "is_confirmation": boolean, "confidence": float, "reasoning": "string"}}
            """
            
            response = await LLM.ainvoke(prompt)
            content = response.content.strip()
            
            # Extract JSON
            import re
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                return json.loads(json_match.group(0))
            else:
                # Fallback parsing
                return {
                    "is_follow_up": "follow" in content.lower(),
                    "is_confirmation": "confirm" in content.lower(),
                    "confidence": 0.5,
                    "reasoning": "Fallback parsing"
                }
                
        except Exception as e:
            logger.warning(f"LLM follow-up analysis failed: {e}")
            return {
                "is_follow_up": False,
                "is_confirmation": False,
                "confidence": 0.0,
                "reasoning": f"LLM analysis failed: {str(e)}"
            }
    
    def detect_linguistic_patterns(self, message: str, prior_topic: str = "") -> Dict[str, Any]:
        """Detect linguistic patterns dynamically based on message characteristics."""
        patterns = {
            "confirmation_indicators": 0,
            "reference_indicators": 0,
            "question_indicators": 0,
            "strong_indicators": [],
            "weak_indicators": []
        }
        
        message_lower = message.lower()
        
        # Dynamic confirmation detection (expandable)
        confirmation_patterns = {
            "direct_affirmation": [
                r'\b(yes|yeah|yep|sure|okay|ok|right|correct|exactly|indeed|true)\b',
                r'\b(có|đúng|phải|vâng|dạ|được|ừ|chính xác)\b'
            ],
            "agreement_phrases": [
                r'\b(i agree|that\'s right|you\'re right|exactly right)\b',
                r'\b(đồng ý|đúng rồi|chính xác rồi)\b'
            ],
            "continuation_requests": [
                r'\b(tell me more|continue|go on|what else|and then)\b',
                r'\b(nói thêm|tiếp tục|kể tiếp|còn gì|và sau đó)\b'
            ]
        }
        
        # Dynamic reference detection
        reference_patterns = {
            "anaphoric_references": [
                r'\b(this|that|these|those|it|they|them)\b',
                r'\b(cái này|cái đó|những cái này|những cái đó|nó|chúng)\b'
            ],
            "topic_references": [
                r'\b(about (this|that)|regarding (this|that)|related to)\b',
                r'\b(về (cái này|cái đó)|liên quan đến)\b'
            ]
        }
        
        # Count pattern matches
        for category, pattern_list in confirmation_patterns.items():
            for pattern in pattern_list:
                if re.search(pattern, message_lower):
                    patterns["confirmation_indicators"] += 1
                    patterns["strong_indicators"].append(f"confirmation_{category}")
        
        for category, pattern_list in reference_patterns.items():
            for pattern in pattern_list:
                if re.search(pattern, message_lower):
                    patterns["reference_indicators"] += 1
                    patterns["strong_indicators"].append(f"reference_{category}")
        
        # Question pattern detection
        question_patterns = [
            r'\b(why|how|what|when|where|who|which)\b',
            r'\b(tại sao|làm sao|cái gì|khi nào|ở đâu|ai|cái nào)\b',
            r'\?$'  # Ends with question mark
        ]
        
        for pattern in question_patterns:
            if re.search(pattern, message_lower):
                patterns["question_indicators"] += 1
                patterns["weak_indicators"].append("question_pattern")
        
        # Message length analysis
        word_count = len(message.split())
        if word_count <= 3:
            patterns["weak_indicators"].append("very_short")
        elif word_count <= 7:
            patterns["weak_indicators"].append("short")
        
        # Topic word overlap (if prior topic available)
        if prior_topic:
            message_words = set(re.findall(r'\b\w{3,}\b', message_lower))
            topic_words = set(re.findall(r'\b\w{3,}\b', prior_topic.lower()))
            overlap = len(message_words.intersection(topic_words))
            
            if overlap > 0:
                patterns["reference_indicators"] += overlap
                patterns["strong_indicators"].append(f"topic_overlap_{overlap}")
        
        return patterns
    
    def score_linguistic_patterns(self, patterns: Dict[str, Any]) -> float:
        """Score linguistic patterns to produce a confidence value."""
        score = 0.0
        
        # Strong indicators
        strong_weight = 0.3
        score += min(len(patterns.get("strong_indicators", [])) * strong_weight, 0.8)
        
        # Confirmation indicators
        confirmation_weight = 0.2
        score += min(patterns.get("confirmation_indicators", 0) * confirmation_weight, 0.6)
        
        # Reference indicators
        reference_weight = 0.15
        score += min(patterns.get("reference_indicators", 0) * reference_weight, 0.4)
        
        # Question indicators (weaker signal)
        question_weight = 0.1
        score += min(patterns.get("question_indicators", 0) * question_weight, 0.2)
        
        return min(score, 1.0)
    
    def analyze_conversation_flow_indicators(self, message: str, conversation_history: List[str] = None) -> Dict[str, float]:
        """Analyze conversation flow indicators for follow-up probability."""
        indicators = {
            "follow_up_probability": 0.0,
            "topic_continuity": 0.0,
            "conversational_coherence": 0.0
        }
        
        if not conversation_history:
            return indicators
        
        try:
            # Analyze message position in conversation
            total_messages = len(conversation_history)
            if total_messages > 1:
                # Messages later in conversation are more likely to be follow-ups
                position_factor = min(total_messages / 10, 0.3)
                indicators["follow_up_probability"] += position_factor
            
            # Analyze topic continuity
            if total_messages >= 2:
                recent_messages = conversation_history[-2:]
                message_lower = message.lower()
                
                # Check for topic word continuity
                topic_words = set()
                for msg in recent_messages:
                    topic_words.update(re.findall(r'\b\w{4,}\b', msg.lower()))
                
                current_words = set(re.findall(r'\b\w{4,}\b', message_lower))
                overlap_ratio = len(current_words.intersection(topic_words)) / max(len(current_words), 1)
                
                indicators["topic_continuity"] = min(overlap_ratio, 1.0)
                indicators["follow_up_probability"] += overlap_ratio * 0.4
            
            # Conversational coherence (simple heuristic)
            if len(message.split()) < 10 and total_messages > 0:
                # Short messages in context are often follow-ups
                indicators["conversational_coherence"] = 0.3
                indicators["follow_up_probability"] += 0.3
            
            indicators["follow_up_probability"] = min(indicators["follow_up_probability"], 1.0)
            
        except Exception as e:
            logger.warning(f"Error analyzing conversation flow: {e}")
        
        return indicators

    async def detect_follow_up_hybrid(self, message: str, prior_topic: str = "", conversation_history: List[str] = None) -> Dict[str, Any]:
        """
        Hybrid approach combining static and dynamic follow-up detection.
        Uses the dynamic method as primary with static as fallback.
        
        Args:
            message: The current message to check
            prior_topic: The previous topic or message for context
            conversation_history: List of recent messages for context
            
        Returns:
            Dictionary with comprehensive detection results
        """
        try:
            # Start with dynamic detection
            dynamic_result = await self.detect_follow_up_dynamic(message, prior_topic, conversation_history)
            
            # If dynamic detection has high confidence, use it
            if dynamic_result.get("confidence", 0.0) > 0.7:
                logger.info(f"Using dynamic detection result (high confidence: {dynamic_result['confidence']:.3f})")
                return dynamic_result
            
            # Otherwise, combine with static detection for validation
            static_result = self.detect_follow_up(message, prior_topic)
            
            # Combine results intelligently
            combined_result = {
                "is_confirmation": dynamic_result.get("is_confirmation", False) or static_result.get("is_confirmation", False),
                "is_follow_up": dynamic_result.get("is_follow_up", False) or static_result.get("is_follow_up", False),
                "confidence": max(dynamic_result.get("confidence", 0.0), 0.6 if static_result.get("is_follow_up", False) else 0.3),
                "reasoning": f"Hybrid: {dynamic_result.get('reasoning', '')} + Static patterns: {static_result.get('has_pattern_match', False)}",
                "semantic_similarity": dynamic_result.get("semantic_similarity", 0.0),
                "linguistic_patterns": dynamic_result.get("linguistic_patterns", []),
                "static_patterns": static_result,
                "method": "hybrid"
            }
            
            logger.info(f"Using hybrid detection result (combined confidence: {combined_result['confidence']:.3f})")
            return combined_result
            
        except Exception as e:
            logger.error(f"Error in hybrid follow-up detection: {e}")
            # Fallback to static method
            static_result = self.detect_follow_up(message, prior_topic)
            return {
                "is_confirmation": static_result.get("is_confirmation", False),
                "is_follow_up": static_result.get("is_follow_up", False),
                "confidence": 0.6 if static_result.get("is_follow_up", False) else 0.3,
                "reasoning": f"Fallback to static detection due to error: {str(e)}",
                "method": "static_fallback"
            }

    async def active_learning(self, message: Union[str, List], conversation_context: str = "", analysis_knowledge: Dict = None, user_id: str = "unknown", prior_data: Dict = None) -> Dict[str, Any]:
        """
        Placeholder for active learning functionality.
        This method should be implemented based on specific requirements.
        """
        # This is a placeholder implementation
        return {
            "status": "success",
            "message": "Active learning functionality not yet implemented",
            "metadata": {}
        }

    def setup_temporal_context(self) -> str:
        """Setup temporal context with current Vietnam time."""
        vietnam_tz = pytz.timezone('Asia/Ho_Chi_Minh')
        current_time = datetime.now(vietnam_tz)
        date_str = current_time.strftime("%A, %B %d, %Y")
        time_str = current_time.strftime("%H:%M")
        return f"Current date and time: {date_str} at {time_str} (Asia/Ho_Chi_Minh timezone)."

    def validate_and_normalize_message(self, message: Union[str, List]) -> str:
        """Validate and normalize the input message."""
        message_str = message if isinstance(message, str) else str(message[0]) if isinstance(message, list) and message else ""
        if not message_str:
            raise ValueError("Empty message provided")
        return message_str

    def extract_analysis_data(self, analysis_knowledge: Dict) -> Dict[str, Any]:
        """Extract and organize data from analysis_knowledge."""
        if not analysis_knowledge:
            return {
                "knowledge_context": "",
                "similarity_score": 0.0,
                "queries": [],
                "query_results": []
            }
        
        return {
            "knowledge_context": analysis_knowledge.get("knowledge_context", ""),
            "similarity_score": float(analysis_knowledge.get("similarity", 0.0)),
            "queries": analysis_knowledge.get("queries", []),
            "query_results": analysis_knowledge.get("query_results", [])
        }

    def extract_prior_data(self, prior_data: Dict) -> Dict[str, str]:
        """Extract prior topic and knowledge from prior_data."""
        if not prior_data:
            return {"prior_topic": "", "prior_knowledge": ""}
        
        return {
            "prior_topic": prior_data.get("topic", ""),
            "prior_knowledge": prior_data.get("knowledge", "")
        }

    def extract_prior_messages(self, conversation_context: str) -> List[str]:
        """Extract prior messages from conversation context."""
        prior_messages = []
        if conversation_context:
            user_messages = re.findall(r'User: (.*?)(?:\n\n|$)', conversation_context, re.DOTALL)
            if user_messages and len(user_messages) > 1:
                prior_messages = user_messages[:-1]  # All except current message
        return prior_messages

    def detect_message_characteristics(self, message_str: str) -> Dict[str, bool]:
        """Detect various characteristics of the message."""
        # Enhanced closing message detection
        closing_phrases = [
            "thế thôi", "hẹn gặp lại", "tạm biệt", "chào nhé", "goodbye", "bye", "cảm ơn nhé", 
            "cám ơn nhé", "đủ rồi", "vậy là đủ", "hôm nay vậy là đủ", "hẹn lần sau"
        ]
        
        # Check for teaching intent in the message
        teaching_keywords = ["let me explain", "I'll teach you", "Tôi sẽ giải thích", "Tôi dạy bạn", 
                            "here's how", "đây là cách", "the way to", "Important to know", 
                            "you should know", "bạn nên biết", "cần hiểu rằng", "phương pháp", "cách thức"]
        
        # Check for Vietnamese greeting forms or names
        vn_greeting_patterns = ["anh ", "chị ", "bạn ", "cô ", "ông ", "bác ", "em "]
        common_vn_names = ["hùng", "hương", "minh", "tuấn", "thảo", "an", "hà", "thủy", "trung", "mai", "hoa", "quân", "dũng", "hiền", "nga", "tâm", "thanh", "tú", "hải", "hòa", "yến", "lan", "hạnh", "phương", "dung", "thu", "hiệp", "đức", "linh", "huy", "tùng", "bình", "giang", "tiến"]
        
        message_lower = message_str.lower()
        message_words = message_lower.split()
        
        return {
            "is_closing_message": any(phrase in message_lower for phrase in closing_phrases),
            "has_teaching_markers": any(keyword.lower() in message_lower for keyword in teaching_keywords),
            "is_vn_greeting": any(pattern in message_lower for pattern in vn_greeting_patterns),
            "contains_vn_name": any(name in message_words for name in common_vn_names),
            "is_short_message": len(message_str.strip().split()) <= 2,
            "is_long_without_question": len(message_str.split()) > 20 and "?" not in message_str
        }

    def check_knowledge_relevance(self, analysis_knowledge: Dict) -> Dict[str, Any]:
        """Check the relevance of retrieved knowledge."""
        best_context_relevance = 0.0
        has_low_relevance_knowledge = False
        
        if analysis_knowledge and "query_results" in analysis_knowledge:
            query_results = analysis_knowledge.get("query_results", [])
            if query_results and isinstance(query_results[0], dict) and "context_relevance" in query_results[0]:
                best_context_relevance = query_results[0].get("context_relevance", 0.0)
                has_low_relevance_knowledge = best_context_relevance < 0.3
                logger.info(f"Best knowledge context relevance: {best_context_relevance}")
        
        return {
            "best_context_relevance": best_context_relevance,
            "has_low_relevance_knowledge": has_low_relevance_knowledge
        }

    def determine_response_strategy(self, flow_type: str, flow_confidence: float, message_characteristics: Dict, 
                                  knowledge_relevance: Dict, similarity_score: float, prior_knowledge: str,
                                  queries: List, query_results: List, knowledge_response_sections: List) -> Dict[str, Any]:
        """Determine the appropriate response strategy based on various factors."""
        
        is_confirmation = flow_type == "CONFIRMATION"
        is_follow_up = flow_type in ["FOLLOW_UP", "CONFIRMATION"]
        is_practice_request = flow_type == "PRACTICE_REQUEST"
        is_closing = flow_type == "CLOSING"
        
        # Extract message characteristics
        is_closing_message = message_characteristics["is_closing_message"] or is_closing
        has_teaching_markers = message_characteristics["has_teaching_markers"]
        is_vn_greeting = message_characteristics["is_vn_greeting"]
        contains_vn_name = message_characteristics["contains_vn_name"]
        is_short_message = message_characteristics["is_short_message"]
        is_long_without_question = message_characteristics["is_long_without_question"]
        
        # Extract knowledge relevance
        best_context_relevance = knowledge_relevance["best_context_relevance"]
        has_low_relevance_knowledge = knowledge_relevance["has_low_relevance_knowledge"]
        
        if is_closing_message:
            return {
                "strategy": "CLOSING",
                "instructions": (
                    "Recognize this as a closing message where the user is ending the conversation. "
                    "Respond with a brief, polite farewell message. "
                    "Thank them for the conversation and express willingness to help in the future. "
                    "Keep it concise and friendly, in the same language they used (Vietnamese/English)."
                ),
                "knowledge_context": "CONVERSATION_CLOSING: User is ending the conversation politely.",
                "similarity_score": similarity_score
            }
        
        elif is_practice_request and prior_knowledge:
            return {
                "strategy": "PRACTICE_REQUEST",
                "instructions": (
                    "The user wants you to DEMONSTRATE or APPLY previously shared knowledge. "
                    "Create a practical example that follows these steps: "
                    
                    "1. Acknowledge their request positively and with enthusiasm. "
                    "2. Reference the prior knowledge in your response directly. "
                    "3. Apply the knowledge in a realistic scenario or example. "
                    "4. Follow any specific methods or steps previously discussed. "
                    "5. Explain your reasoning as you demonstrate. "
                    "6. Ask if your demonstration meets their expectations. "
                    
                    "IMPORTANT: The user is asking you to SHOW your understanding, not asking for new information. "
                    "Even if the request is vague like 'Em thử áp dụng các kiến thức em có anh xem nào', "
                    "understand that they want you to DEMONSTRATE the knowledge you gained from previous messages. "
                    "Be confident and enthusiastic - this is a chance to show what you've learned."
                    
                    "CRITICAL: If the knowledge includes communication techniques, relationship building, or language patterns, "
                    "ACTIVELY USE these techniques in your response format, not just talk about them. For example:"
                    "- If knowledge mentions using 'em' to refer to yourself, use that pronoun in your response "
                    "- If it suggests addressing users as 'anh/chị', use that form of address "
                    "- If it recommends specific phrases or compliments, incorporate them naturally "
                    "- If it suggests question techniques, use those exact techniques at the end of your response"
                ),
                "knowledge_context": prior_knowledge,
                "similarity_score": max(similarity_score, 0.8)
            }
        
        elif has_low_relevance_knowledge and similarity_score > 0.3:
            return {
                "strategy": "LOW_RELEVANCE_KNOWLEDGE",
                "instructions": (
                    "You have knowledge with low relevance to the current query. "
                    "PRIORITIZE the user's current message over the retrieved knowledge. "
                    "ONLY reference the knowledge if it genuinely helps answer the query. "
                    "If the knowledge is off-topic, IGNORE it completely and focus on the user's message. "
                    "Be clear and direct in addressing what the user is actually asking about. "
                    "Generate a response primarily based on the user's current message and intent."
                    
                    "However, if the knowledge contains ANY communication techniques or relationship-building approaches, "
                    "incorporate those techniques into HOW you construct your response, even if the topic is different."
                ),
                "knowledge_context": f"LOW RELEVANCE KNOWLEDGE WARNING: The retrieved knowledge has low relevance " \
                    f"(score: {best_context_relevance:.2f}) to the current query. Prioritize the user's message.\n\n",
                "similarity_score": similarity_score
            }
        
        elif is_follow_up:
            if is_confirmation:
                instructions = (
                    "Recognize this is a direct confirmation to your question in your previous message. "
                    "Continue the conversation as if the user said 'yes' to your previous question. "
                    "Provide a helpful response that builds on the previous question, offering relevant details or asking a follow-up question. "
                    "Don't ask for clarification when the confirmation is clear - proceed with the conversation flow naturally. "
                    "If your previous question offered to provide more information, now is the time to provide that information. "
                    "Keep the response substantive, helpful, and directly related to what the user just confirmed interest in."
                )
            else:
                instructions = (
                    "Recognize the message as a follow-up or confirmation of PRIOR TOPIC, referring to a specific concept or group from PRIOR KNOWLEDGE (e.g., customer segmentation methods). "
                    "Use PRIOR KNOWLEDGE to deepen the discussion, leveraging specific details. "
                    "Structure the response with key aspects (e.g., purpose, methods, outcomes). "
                    "If PRIOR TOPIC is ambiguous, rephrase it (e.g., 'It sounds like you're confirming customer segmentation…'). "
                    "Ask a targeted follow-up to advance the discussion."
                )
            
            return {
                "strategy": "FOLLOW_UP",
                "instructions": instructions,
                "knowledge_context": prior_knowledge if not knowledge_response_sections else "",
                "similarity_score": max(similarity_score, 0.7) if not knowledge_response_sections else similarity_score
            }
        
        elif (is_vn_greeting or contains_vn_name) and len(message_characteristics.get("message_str", "").split()) <= 3:
            return {
                "strategy": "GREETING",
                "instructions": (
                    "Recognize this as a Vietnamese greeting or someone addressing you by name. "
                    "Respond warmly and appropriately to the greeting. "
                    "If they used a Vietnamese name or greeting form, respond in Vietnamese. "
                    "Keep your response friendly, brief, and conversational. "
                    "Ask how you can assist them today. "
                    "Ensure your tone matches the formality level they used (formal vs casual)."
                ),
                "knowledge_context": "",
                "similarity_score": similarity_score
            }
        
        elif similarity_score < 0.35 and not knowledge_response_sections:
            if is_short_message:
                instructions = (
                    "Recognize this as a very short or potentially unclear message. "
                    "Acknowledge that you need more information to provide a helpful response. "
                    "Politely ask the user to provide more details or context about what they're asking. "
                    "Suggest a few possible interpretations of their query if appropriate. "
                    "Keep your response friendly and helpful, showing eagerness to assist once you have more information. "
                    "Match the user's language choice (Vietnamese/English). "
                    "Ensure you provide a response even if the query is very minimal or unclear."
                )
            else:
                instructions = (
                    "State: 'Tôi không thể tìm thấy thông tin liên quan; vui lòng giải thích thêm.' "
                    "Ask for more details about the topic. "
                    "Propose a specific question (e.g., 'Bạn có thể chia sẻ thêm về ý nghĩa của điều này không?'). "
                    "If the message appears to be attempting to teach or explain something, acknowledge this and express "
                    "interest in learning about the topic through a thoughtful follow-up question."
                )
            
            return {
                "strategy": "LOW_SIMILARITY",
                "instructions": instructions,
                "knowledge_context": "",
                "similarity_score": similarity_score
            }
        
        elif has_teaching_markers or is_long_without_question:
            return {
                "strategy": "TEACHING_INTENT",
                "instructions": (
                    "Recognize this message as TEACHING INTENT where the user is sharing knowledge with you. "
                    "Your goal is to synthesize this knowledge for future use and demonstrate understanding. "
                    
                    "Generate THREE separate outputs in your response:\n\n"
                    
                    "1. <user_response>\n"
                    "   This is what the user will see - include:\n"
                    "   - Acknowledgment of their teaching with appreciation\n"
                    "   - Demonstration of your understanding\n"
                    "   - End with 1-2 open-ended questions to deepen the conversation\n"
                    "   - Make this conversational and engaging\n"
                    "</user_response>\n\n"
                    
                    "2. <knowledge_synthesis>\n"
                    "   This is for knowledge storage - include ONLY:\n"
                    "   - Factual information extracted from the user's message\n"
                    "   - Structured, clear explanation of the concepts\n"
                    "   - NO greeting phrases, acknowledgments, or questions\n"
                    "   - NO conversational elements - pure knowledge only\n"
                    "   - Organized in logical sections if appropriate\n"
                    "</knowledge_synthesis>\n\n"
                    
                    "3. <knowledge_summary>\n"
                    "   A concise 2-3 sentence summary capturing the core teaching point\n"
                    "   This should be factual and descriptive, not conversational\n"
                    "</knowledge_summary>\n\n"
                    
                    "CRITICAL LANGUAGE INSTRUCTION: ALWAYS respond in EXACTLY the SAME LANGUAGE as the user's message for ALL sections. "
                    "- If the user wrote in Vietnamese, respond entirely in Vietnamese "
                    "- If the user wrote in English, respond entirely in English "
                    "- Do not mix languages in your response "
                    
                    "This structured approach helps create high-quality, reusable knowledge while maintaining good user experience."
                ),
                "knowledge_context": "",
                "similarity_score": similarity_score
            }
        
        else:
            return {
                "strategy": "RELEVANT_KNOWLEDGE",
                "instructions": (
                    "I've found MULTIPLE knowledge entries relevant to your query. Let me provide a comprehensive response.\n\n"
                    "For each knowledge item found:\n"
                    "1. Review and synthesize the information from ALL available knowledge items\n"
                    "2. When answering, incorporate insights from ALL relevant knowledge items found\n"
                    "3. Show how different knowledge entries complement or confirm each other\n"
                    "4. If there are any contradictions between knowledge items, highlight them\n"
                    "5. Present information in order of relevance, addressing the most relevant points first\n\n"
                    "DO NOT ignore any of the provided knowledge items - incorporate insights from ALL of them in your response.\n"
                    "DO NOT summarize the knowledge as 'I found X items' - just seamlessly incorporate all relevant information.\n\n"
                    "MOST IMPORTANTLY: If the knowledge contains ANY communication techniques, relationship-building strategies, "
                    "or specific linguistic patterns, ACTIVELY APPLY these in how you structure your response. For example:"
                    "- If the knowledge mentions using 'em/tôi' or specific pronouns, use those exact pronouns yourself"
                    "- If it suggests addressing the user in specific ways ('anh/chị/bạn'), use that exact form of address"
                    "- If it recommends compliments or specific phrases, incorporate them naturally in your response"
                    "- If it mentions conversation flow techniques, apply them in how you structure this very response"
                    "This way, you're not just explaining the knowledge but DEMONSTRATING it in action."
                ),
                "knowledge_context": "",
                "similarity_score": similarity_score
            }

    def build_knowledge_fallback_sections(self, queries: List, query_results: List) -> str:
        """Build fallback knowledge response sections when knowledge_context is empty."""
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
        knowledge_response_sections = []
        
        if high_confidence:
            knowledge_response_sections.append("HIGH CONFIDENCE KNOWLEDGE:")
            for i, (query, content, score) in enumerate(high_confidence, 1):
                knowledge_response_sections.append(
                    f"[{i}] On the topic of '{query}' (confidence: {score:.2f}): {content}"
                )
        
        if medium_confidence:
            knowledge_response_sections.append("MEDIUM CONFIDENCE KNOWLEDGE:")
            for i, (query, content, score) in enumerate(medium_confidence, 1):
                knowledge_response_sections.append(
                    f"[{i}] About '{query}' (confidence: {score:.2f}): {content}"
                )
        
        if low_confidence:
            knowledge_response_sections.append("LOW CONFIDENCE/NO KNOWLEDGE:")
            for i, query in enumerate(low_confidence, 1):
                knowledge_response_sections.append(
                    f"[{i}] I don't have sufficient knowledge about '{query}'. Would you like to teach me about this topic?"
                )
        
        # Combine the knowledge sections if they exist
        if knowledge_response_sections:
            knowledge_context = "\n\n".join(knowledge_response_sections)
            logger.info(f"Created fallback knowledge response with {len(high_confidence)} high, {len(medium_confidence)} medium, and {len(low_confidence)} low confidence items")
            return knowledge_context
        
        return ""

    def build_llm_prompt(self, message_str: str, conversation_context: str, temporal_context: str, 
                        knowledge_context: str, response_strategy: str, strategy_instructions: str,
                        core_prior_topic: str, user_id: str) -> str:
        """Build the comprehensive LLM prompt."""
        return f"""You are Ami, a conversational AI that understands topics deeply and drives discussions toward closure.

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
                   - When KNOWLEDGE RESULTS contains multiple entries, incorporate ALL relevant information from ALL entries
                   - DO NOT ignore or skip any knowledge entries - review and use ALL of them in your response
                   - Match the user's language choice (Vietnamese/English)
                   - For closing messages, set intent_type="closing" and respond with a polite farewell
                   - When the user addresses you as "Ami" or refers to you as an AI, acknowledge this in your response naturally
                   - Consider references to your identity or role as indicators of direct address
                   
                   - When handling TEACHING INTENT:
                     * ACTIVELY SCAN the entire conversation history for supporting information related to current topic
                     * Synthesize BOTH the current input AND relevant historical context into a comprehensive understanding
                     * Structure the knowledge for future application (how to use this information)
                     * Rephrase any ambiguous terms, sentences, or paragraphs for clarity
                     * Organize information with clear steps, examples, or use cases when applicable
                     * Include contextual understanding (when/where/how to apply this knowledge)
                     * Highlight key principles rather than just recording facts
                     * Verify your understanding by restating core concepts in different terms
                     * Expand abbreviations and domain-specific terminology
                     * CREATE A CONCISE SUMMARY (2-3 sentences) PREFIXED WITH 'SUMMARY: ' - THIS IS MANDATORY
                     * The summary should capture the core teaching point in simple language
                     * Ensure the response demonstrates how to apply this knowledge in future scenarios
                     * END WITH 1-2 OPEN-ENDED QUESTIONS that invite brainstorming and deeper exploration

                   - When handling LOW_RELEVANCE_KNOWLEDGE:
                     * PRIORITIZE the user's current message over any retrieved knowledge
                     * If retrieved knowledge contradicts or misleads from the user's intent, IGNORE it
                     * Focus on generating a direct, helpful response to the user's current question
                     * Evaluate if there's ANY genuinely useful information in the knowledge before using it
                     * Be explicit when the retrieved knowledge is not addressing the actual query
                     * Generate a response primarily based on the query itself and your general capabilities
                     * DO NOT include a SUMMARY section in your response
                     
                   - When handling PRACTICE_REQUEST:
                     * Create a practical demonstration applying previously taught knowledge
                     * Follow specific steps or methods from the prior knowledge exactly
                     * Use realistic examples that show the knowledge in action
                     * Explain your thought process as you demonstrate
                     * Reference specific parts of prior knowledge to show understanding
                     * Ask for feedback on your demonstration
                     * DO NOT include a SUMMARY section in your response
                     * IMPORTANT: If you find any communication skills or techniques in the knowledge, ACTIVELY APPLY those techniques in your response format and style
                     
                   - When handling RELEVANT_KNOWLEDGE:
                     * Review and use ALL knowledge items provided - don't skip any
                     * Present information clearly based on relevance
                     * Structure your response logically with the most relevant information first
                     * DO NOT include a SUMMARY section in your response - summaries are ONLY for teaching intent
                     * IMPORTANT: If you find any communication skills or techniques in the knowledge, ACTIVELY DEMONSTRATE those techniques in your response

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
                   - ALWAYS MATCH THE USER'S LANGUAGE - if they use Vietnamese, respond in Vietnamese
                   - Keep all parts of your response (including SUMMARY sections) in the same language as the user's message
                   - <knowledge_queries>["query1", "query2", "query3"]</knowledge_queries>
                   - <tool_calls>[{{"name": "tool_name", "parameters": {{...}}}}]</tool_calls> (if needed)
                   - <evaluation>{{"has_teaching_intent": true/false, "is_priority_topic": true/false, "priority_topic_name": "topic_name", "should_save_knowledge": true/false, "intent_type": "query/teaching/confirmation/follow-up", "name_addressed": true/false, "ai_referenced": true/false}}</evaluation>

                Maintain topic continuity, ensure proper JSON formatting, and include user_id in all tool calls.
                """

    def extract_structured_sections(self, content: str) -> Dict[str, str]:
        """Extract structured sections from LLM response."""
        sections = {
            "user_response": "",
            "knowledge_synthesis": "",
            "knowledge_summary": ""
        }
        
        # Extract structured sections
        user_response_match = re.search(r'<user_response>(.*?)</user_response>', content, re.DOTALL)
        if user_response_match:
            sections["user_response"] = user_response_match.group(1).strip()
            logger.info(f"Found user_response section")
        
        synthesis_match = re.search(r'<knowledge_synthesis>(.*?)</knowledge_synthesis>', content, re.DOTALL)
        if synthesis_match:
            sections["knowledge_synthesis"] = synthesis_match.group(1).strip()
            logger.info(f"Found knowledge_synthesis section")
        
        summary_match = re.search(r'<knowledge_summary>(.*?)</knowledge_summary>', content, re.DOTALL)
        if summary_match:
            sections["knowledge_summary"] = summary_match.group(1).strip()
            logger.info(f"Found knowledge_summary section")
        
        return sections

    def extract_tool_calls_and_evaluation(self, content: str) -> tuple:
        """Extract tool calls and evaluation from LLM response."""
        tool_calls = []
        evaluation = {
            "has_teaching_intent": False, 
            "is_priority_topic": False, 
            "priority_topic_name": "", 
            "should_save_knowledge": False, 
            "intent_type": "query", 
            "name_addressed": False, 
            "ai_referenced": False
        }
        
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
        
        return content, tool_calls, evaluation

    def handle_empty_response_fallbacks(self, user_facing_content: str, response_strategy: str, message_str: str) -> str:
        """Handle cases where LLM response is empty and provide fallbacks."""
        if user_facing_content and not user_facing_content.isspace():
            return user_facing_content
        
        # Ensure closing messages get a response even if empty
        if response_strategy == "CLOSING":
            # Default closing message if the LLM didn't provide one
            if "vietnamese" in message_str.lower() or any(vn_word in message_str.lower() for vn_word in ["tạm biệt", "cảm ơn", "hẹn gặp", "thế thôi"]):
                user_facing_content = "Vâng, cảm ơn bạn đã trao đổi. Hẹn gặp lại bạn lần sau nhé!"
            else:
                user_facing_content = "Thank you for the conversation. Have a great day and I'm here if you need anything else!"
            logger.info("Added default closing response for empty LLM response")
        
        # Ensure unclear or short queries also get a helpful response when content is empty
        else:
            # Check if message is short (1-2 words) or unclear
            is_short_message = len(message_str.strip().split()) <= 2
            
            # Default response for short/unclear messages
            if "vietnamese" in message_str.lower() or any(vn_word in message_str.lower() for vn_word in ["anh", "chị", "bạn", "cô", "ông", "xin", "vui lòng"]):
                user_facing_content = f"Xin lỗi, tôi không hiểu rõ câu hỏi '{message_str}'. Bạn có thể chia sẻ thêm thông tin hoặc đặt câu hỏi cụ thể hơn được không?"
            else:
                user_facing_content = f"I'm sorry, I didn't fully understand your message '{message_str}'. Could you please provide more details or ask a more specific question?"
            
            logger.info(f"Added default response for empty LLM response to short/unclear query: '{message_str}'")
        
        return user_facing_content 