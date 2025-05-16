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
from brain_singleton import get_current_graph_version
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
        self.graph_version_id = get_current_graph_version() or str(uuid4())
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

            confirmation_keywords = ["có", "yes", "correct", "right", "explore further", "đúng rồi", "nhóm này"]
            is_confirmation = any(keyword.lower() in primary_query.lower() for keyword in confirmation_keywords)
            # Check topic overlap for follow-ups
            topic_overlap = any(term in primary_query.lower() and term in prior_topic.lower() 
                            for term in ["phân tích chân dung khách hàng", "phân nhóm khách hàng", "chân dung khách hàng"])
            is_follow_up = is_confirmation or topic_overlap or re.search(r'\b(nhóm này|this group|vậy thì sao)\b', primary_query.lower(), re.IGNORECASE) or (prior_topic and primary_query.lower().strip() in prior_topic.lower())
            
            if is_follow_up and prior_topic:
                queries.append(prior_topic)
                logger.info(f"Follow-up detected, reusing prior topic: {prior_topic[:50]}")
                similarity = 0.7
                knowledge_context = prior_knowledge
            else:
                queries.append(primary_query)
                similarity = 0.0
                knowledge_context = ""
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
        
        confirmation_keywords = ["có", "yes", "correct", "right", "explore further", "đúng rồi", "nhóm này"]
        is_confirmation = any(keyword.lower() in message_str.lower() for keyword in confirmation_keywords)
        
        # Improved follow-up detection that looks for patterns in the conversation history
        is_follow_up = is_confirmation or re.search(r'\b(nhóm này|this group|vậy thì sao)\b', message_str.lower(), re.IGNORECASE) or (prior_topic and message_str.lower().strip() in prior_topic.lower())
        
        core_prior_topic = prior_topic
        
        # Knowledge handling strategy based on queries and similarity
        knowledge_response_sections = []
        if queries and query_results:
            for query, result in zip(queries, query_results):
                query_similarity = result.get("score", 0.0)
                query_content = result.get("raw", "")
                
                if query_similarity < 0.35:
                    knowledge_response_sections.append(
                        f"I can't find knowledge relevant to '{query}'. Can you elaborate?"
                    )
                elif 0.35 <= query_similarity <= 0.7:
                    knowledge_response_sections.append(
                        f"I found some knowledge relevant to '{query}': {query_content}. But I'm not really confident. Can you justify?"
                    )
                else:  # > 0.7
                    knowledge_response_sections.append(
                        f"Here is what I know about '{query}': {query_content}"
                    )
            
            # Combine the knowledge sections if they exist
            if knowledge_response_sections:
                knowledge_context = "\n\n".join(knowledge_response_sections)
                logger.info(f"Created multi-query knowledge response with {len(knowledge_response_sections)} sections")
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
        elif similarity_score < 0.35 and not knowledge_response_sections:
            response_strategy = "LOW_SIMILARITY"
            # Create query-specific response section if no other knowledge is found
            if queries:
                query_text = queries[0] if isinstance(queries, list) and queries else str(queries)
                knowledge_response_sections = [f"I can't find knowledge relevant to '{query_text}'. Can you elaborate or teach me about this topic?"]
                knowledge_context = "\n\n".join(knowledge_response_sections)
                logger.info(f"Created LOW_SIMILARITY response for query: {query_text}")
            
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
        
        prompt = f"""You are Ami, a conversational AI that understands topics deeply and drives discussions toward closure.

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
                1. **Intent Classification** (critical for proper response): 
                   - Carefully analyze if the user is asking a question (query) or providing/initiating information (teaching)
                   - Use semantic understanding to detect teaching intent:
                     * The user introduces a new topic with specific details
                     * The user describes how to do something or explains concepts
                     * The message has an instructional or explanatory tone
                     * The user seems to be sharing knowledge rather than seeking it
                   - In Vietnamese conversations, consider cultural context in your analysis
                   - For queries: Use EXISTING KNOWLEDGE as foundation if available
                   - For teaching intent: Set has_teaching_intent=true and acknowledge user is sharing knowledge
                   - Use Vietnamese if the user does

                2. **Priority Topics** (set is_priority_topic=true AND should_save_knowledge=true):
                   - Customer segmentation ("chân dung khách hàng")
                   - Health-related topics ("rối loạn cương dương")

                3. **Knowledge Management**:
                   - IMPORTANT: You should recommend saving knowledge (should_save_knowledge=true) when:
                     * The user shows teaching intent through their communication style and content
                     * Priority topics are mentioned
                     * The user initiates a conversation about a topic they seem knowledgeable about
                     * The information shared appears valuable for future conversations
                
                4. **Confidence-Based Responses**:
                   - Low confidence (<0.35): "I can't find knowledge relevant to [query]. Can you elaborate?"
                   - Medium confidence (0.35-0.7): "I found some knowledge relevant to [query]: [content]. But I'm not really confident. Can you justify?"
                   - High confidence (>0.7): "Here is what I know about [query]: [content]"

                5. **Language and Relational Dynamics**:
                   - Recognize when you're being directly addressed (terms like "Em", "You", "Bạn")
                   - Match your response style to the user's speech register and level of formality
                   - In Vietnamese conversations:
                     * If addressed as "Em", respond using "Em" as self-reference and appropriate counterpart (like "Anh/Chị" for the user)
                     * Maintain consistent relationship pronouns throughout the conversation
                   - In English conversations:
                     * Use personal pronouns that match the conversation's established formality level
                   - Always recognize the cultural/linguistic context of addressing terms

                6. **Output Format**:
                   - Conversational Response (100-150 words, use user's language)
                   - <knowledge_queries>["query1", "query2", "query3"]</knowledge_queries>
                   - <tool_calls>[{{"name": "tool_name", "parameters": {{...}}}}]</tool_calls> (if needed)
                   - <evaluation>{{"has_teaching_intent": true/false, "is_priority_topic": true/false, "priority_topic_name": "topic_name", "should_save_knowledge": true/false, "intent_type": "query/teaching/confirmation/follow-up"}}</evaluation>

                Remember to maintain topic continuity for follow-ups, include user_id in all tool calls, and ensure proper JSON formatting.
                """
        try:
            response = await LLM.ainvoke(prompt)
            logger.info(f"LLM response generated with similarity score: {similarity_score}")
            
            content = response.content.strip()
            tool_calls = []
            evaluation = {"has_teaching_intent": False, "is_priority_topic": False, "priority_topic_name": "", "should_save_knowledge": False}
            
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
                    timeout=5.0  # 5-second timeout for database operations
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
                    categories = ["health_segmentation"] if any(term in input_text.lower() for term in ["rối loạn cương dương", "xuất tinh sớm", "phân nhóm khách hàng", "phân tích chân dung khách hàng"]) else ["general"]
                
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