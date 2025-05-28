"""
I want to implement a system that can:
1. When it detects teaching intent, it will find relevant knowledge. 
2. Syntheisize and store in a list with structure: [list of input in multiturn] and a the final synthesis.
3. Ask human to adjust
5. ask human to save
"""
import json
import asyncio
from typing import Dict, List, Any, Optional, Union, AsyncGenerator, Set
from datetime import datetime
from uuid import uuid4
import re
import pytz
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from pccontroller import save_knowledge, query_knowledge, query_knowledge_from_graph
from utilities import logger, EMBEDDINGS
from tool_learning_support import LearningSupport
from curiosity import KnowledgeExplorer
from alpha import save_teaching_synthesis

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

class AVA:
    def __init__(self):
        self.graph_version_id = ""
        # Set to keep track of pending background tasks
        self._background_tasks: Set[asyncio.Task] = set()
        # Initialize support class
        self.support = LearningSupport(self)
        # Initialize knowledge explorer
        self.knowledge_explorer = None

    async def initialize(self):
        """Initialize the processor asynchronously."""
        logger.info("Initializing AVA")
        # Initialize knowledge explorer with support module
        self.knowledge_explorer = KnowledgeExplorer(self.graph_version_id, self.support)
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

            # Step 2: Search for relevant knowledge using iterative exploration
            logger.info(f"Searching for knowledge based on message using iterative exploration...")
            analysis_knowledge = await self.knowledge_explorer.explore(message, conversation_context, user_id, thread_id)
            
            # Step 3: Enrich with suggested queries if iterative exploration didn't achieve high similarity
            if suggested_queries and analysis_knowledge.get("similarity", 0.0) < 0.70:
                logger.info(f"Searching for additional knowledge using {len(suggested_queries)} suggested queries")
                primary_similarity = analysis_knowledge.get("similarity", 0.0)
                primary_knowledge = analysis_knowledge.get("knowledge_context", "")
                primary_queries = analysis_knowledge.get("queries", [])
                primary_query_results = analysis_knowledge.get("query_results", [])
                
                for query in suggested_queries:
                    if query not in primary_queries:  # Avoid duplicate queries
                        query_knowledge = await self.support.search_knowledge(query, conversation_context, user_id, thread_id)
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
            # Step 5: Generate response
            prior_data = analysis_knowledge.get("prior_data", {})
            
            # Remove hardcoded teaching intent detection and let the LLM handle it
            response = await self._active_learning(message, conversation_context, analysis_knowledge, user_id, prior_data)
            
            # Step 6: Enhanced knowledge saving with similarity gating
            if response.get("status") == "success":
                # Get teaching intent and priority topic info from LLM evaluation
                has_teaching_intent = response.get("metadata", {}).get("has_teaching_intent", False)
                is_priority_topic = response.get("metadata", {}).get("is_priority_topic", False)
                should_save_knowledge = response.get("metadata", {}).get("should_save_knowledge", False)
                priority_topic_name = response.get("metadata", {}).get("priority_topic_name", "")
                intent_type = response.get("metadata", {}).get("intent_type", "unknown")
                
                # Get similarity score from analysis
                similarity_score = analysis_knowledge.get("similarity", 0.0) if analysis_knowledge else 0.0
                
                # Use new similarity-based gating system
                save_decision = await self.should_save_knowledge_with_similarity_gate(response, similarity_score, message)
                
                # Enhance response to guide conversation toward higher quality knowledge
                response = await self.enhance_response_for_knowledge_quality(response, similarity_score, save_decision, message)
                
                # Override response_strategy if LLM detected teaching_intent
                if has_teaching_intent and response.get("metadata", {}).get("response_strategy") != "TEACHING_INTENT":
                    original_strategy = response.get("metadata", {}).get("response_strategy", "unknown")
                    logger.info(f"LLM detected teaching_intent=True, regenerating response with TEACHING_INTENT format (was {original_strategy})")
                    
                    # Log original response for debugging
                    logger.info(f"ORIGINAL response before regeneration: {response.get('message', '')[:200]}...")
                    
                    # Generate a new response with teaching intent instructions
                    message_content = response.get("message", "")
                    teaching_prompt = f"""IMPORTANT: The user is TEACHING you something. Your job is to synthesize this knowledge.
                    
                    Original message from user: {message}
                    
                    Your original response: {message_content}
                    
                    Instructions:
                    1. Acknowledge their teaching with appreciation
                    2. Synthesize the knowledge into a clear, structured format
                    3. Create a SUMMARY section (2-3 sentences) prefixed with "SUMMARY: "
                    4. Ask 1-2 open-ended follow-up questions to deepen understanding
                    
                    CRITICAL: RESPOND IN THE SAME LANGUAGE AS THE USER'S MESSAGE.
                    - If the user wrote in Vietnamese, respond in Vietnamese
                    - If the user wrote in English, respond in English
                    - Match the language exactly - do not mix languages
                    
                    DO NOT say this is a teaching message or make meta-references to teaching.
                    DO include a "SUMMARY: " section with 2-3 concise sentences capturing the core point.
                    
                    Your revised synthesized response:
                    """
                    
                    try:
                        teaching_response = await LLM.ainvoke(teaching_prompt)
                        teaching_content = teaching_response.content.strip()
                        
                        # Check if there's a SUMMARY section, add one if missing
                        if "SUMMARY:" not in teaching_content:
                            topic_extract = message[:50] + ("..." if len(message) > 50 else "")
                            teaching_content += f"\n\nSUMMARY: {topic_extract} là một chiến lược quan trọng khi tương tác với khách hàng, giúp cải thiện trải nghiệm của họ."
                        
                        # Update the response
                        response["message"] = teaching_content
                        response["metadata"]["response_strategy"] = "TEACHING_INTENT"
                        logger.info("Successfully regenerated response with TEACHING_INTENT format including SUMMARY section")
                        logger.info(f"NEW response after regeneration: {teaching_content[:200]}...")
                    except Exception as e:
                        logger.error(f"Failed to regenerate teaching response: {str(e)}")
                        # Add a summary to the existing response if regeneration fails
                
                logger.info(f"Knowledge saving decision: {save_decision}")
                logger.info(f"LLM evaluation: intent={intent_type}, teaching_intent={has_teaching_intent}, priority_topic={is_priority_topic}, similarity={similarity_score:.2f}")
                
                # Only save knowledge when similarity-based gating approves
                if save_decision.get("should_save", False):
                    await self.handle_teaching_intent(message, response, user_id, thread_id, priority_topic_name)

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
            flow_result = await self.support.detect_conversation_flow(
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
                *(query_knowledge_from_graph(
                    query=query,
                    graph_version_id=self.graph_version_id,
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
            
            # Collect all result items for parallel processing
            all_result_items = []
            for query, results in zip(queries, results_list):
                query_count += 1
                if isinstance(results, Exception):
                    logger.warning(f"Query '{query[:30]}...' failed: {str(results)}")
                    continue
                if not results:
                    logger.info(f"Query '{query[:30]}...' returned no results")
                    continue
                
                # Collect all result items with their query info
                for result_item in results:
                    knowledge_content = result_item["raw"]
                    
                    # Extract just the User portion if this is a combined knowledge entry
                    if knowledge_content.startswith("User:") and "\n\nAI:" in knowledge_content:
                        user_part = re.search(r'User:(.*?)(?=\n\nAI:)', knowledge_content, re.DOTALL)
                        if user_part:
                            knowledge_content = user_part.group(1).strip()
                            logger.info(f"Extracted User portion from combined knowledge")
                    
                    # Store for parallel processing
                    all_result_items.append({
                        "result_item": result_item,
                        "knowledge_content": knowledge_content,
                        "query": query
                    })
            
            # Parallel context relevance evaluation
            if all_result_items:
                logger.info(f"Evaluating context relevance for {len(all_result_items)} results in parallel")
                
                # Create parallel tasks for context relevance evaluation
                relevance_tasks = [
                    self.support.evaluate_context_relevance(primary_query, item["knowledge_content"])
                    for item in all_result_items
                ]
                
                # Execute all context relevance evaluations in parallel
                context_relevances = await asyncio.gather(*relevance_tasks, return_exceptions=True)
                
                # Process results with their context relevance scores
                for item_data, context_relevance in zip(all_result_items, context_relevances):
                    if isinstance(context_relevance, Exception):
                        logger.warning(f"Context relevance evaluation failed: {context_relevance}")
                        context_relevance = 0.0  # Default to 0 if evaluation fails
                    
                    result_item = item_data["result_item"]
                    knowledge_content = item_data["knowledge_content"]
                    query = item_data["query"]
                    
                    # Calculate adjusted similarity score with context relevance factor
                    query_similarity = result_item["score"]
                    adjusted_similarity = query_similarity * (1.0 + 0.5 * context_relevance)  # Range between 100%-150% of original score
                    
                    # Add context relevance information to result metadata
                    result_item["context_relevance"] = context_relevance
                    result_item["adjusted_similarity"] = adjusted_similarity
                    result_item["query"] = query
                    
                    all_query_results.append(result_item)  # Store all results 
                    
                    logger.info(f"Result for query '{query[:30]}...' yielded similarity: {query_similarity}, adjusted: {adjusted_similarity}, context relevance: {context_relevance}, content: '{knowledge_content[:50]}...'")
                    
                    # Track the top 100 results using adjusted similarity
                    if not highest_similarities or adjusted_similarity > min(highest_similarities) or len(highest_similarities) < 100:
                        # Add the new result to our collections
                        if len(highest_similarities) < 100:
                            highest_similarities.append(adjusted_similarity)
                            best_results.append(result_item)
                            knowledge_contexts.append(knowledge_content)
                        else:
                            # Find the minimum similarity in our top 100
                            min_index = highest_similarities.index(min(highest_similarities))
                            # Replace it with the new result
                            highest_similarities[min_index] = adjusted_similarity
                            best_results[min_index] = result_item
                            knowledge_contexts[min_index] = knowledge_content
                        
                        logger.info(f"Updated top 100 knowledge results with adjusted similarity: {adjusted_similarity}, context relevance: {context_relevance}")
                
                logger.info(f"Completed parallel context relevance evaluation for {len(all_result_items)} results")

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
                # Đơn giản hóa cách xây dựng combined_knowledge_context
                # Lấy top 100 vectors có liên quan nhất (hoặc ít hơn nếu không đủ)
                # Sort by similarity to present most relevant first
                sorted_results = sorted(zip(best_results, highest_similarities, knowledge_contexts), 
                                      key=lambda pair: pair[1], reverse=True)
                
                # Format response sections
                knowledge_response_sections = []
                knowledge_response_sections.append("KNOWLEDGE RESULTS:")
                
                # Log số lượng kết quả
                result_count = min(len(sorted_results), 100)  # Tối đa 100 kết quả
                logger.info(f"Adding {result_count} knowledge items to response")
                
                # Thêm từng kết quả với số thứ tự
                for i, (result, item_similarity, content) in enumerate(sorted_results[:result_count], 1):
                    query = result.get("query", "unknown query")
                    score = result.get("score", 0.0)
                    
                    # Loại bỏ tiếp đầu ngữ "AI:" hoặc "AI Synthesis:" nếu có
                    if content.startswith("AI: "):
                        content = content[4:]
                    elif content.startswith("AI Synthesis: "):
                        content = content[14:]
                    
                    # Thêm kết quả có số thứ tự
                    knowledge_response_sections.append(
                        f"[{i}] Query: '{query}' (score: {score:.2f})\n{content}"
                    )
                
                # Tạo combined_knowledge_context từ tất cả sections
                combined_knowledge_context = "\n\n".join(knowledge_response_sections)
                
                # Log để kiểm tra số lượng sections
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

    async def _active_learning(self, message: Union[str, List], conversation_context: str = "", analysis_knowledge: Dict = None, user_id: str = "unknown", prior_data: Dict = None) -> Dict[str, Any]:
        """Simplified active learning method using helper functions from support class."""
        logger.info("Answering user question with active learning approach")
        
        try:
            # Step 1: Setup and validation
            temporal_context = self.support.setup_temporal_context()
            message_str = self.support.validate_and_normalize_message(message)
            
            # Step 2: Extract and organize data
            analysis_data = self.support.extract_analysis_data(analysis_knowledge)
            prior_data_extracted = self.support.extract_prior_data(prior_data)
            prior_messages = self.support.extract_prior_messages(conversation_context)
            
            knowledge_context = analysis_data["knowledge_context"]
            similarity_score = analysis_data["similarity_score"]
            queries = analysis_data["queries"]
            query_results = analysis_data["query_results"]
            prior_topic = prior_data_extracted["prior_topic"]
            prior_knowledge = prior_data_extracted["prior_knowledge"]
            
            logger.info(f"Using similarity score: {similarity_score}")
            
            # Step 3: Detect conversation flow and message characteristics
            flow_result = await self.support.detect_conversation_flow(
                message=message_str,
                prior_messages=prior_messages,
                conversation_context=conversation_context
            )
            
            flow_type = flow_result.get("flow_type", "NEW_TOPIC")
            flow_confidence = flow_result.get("confidence", 0.5)
            logger.info(f"Active learning conversation flow: {flow_type} (confidence: {flow_confidence})")
            
            message_characteristics = self.support.detect_message_characteristics(message_str)
            knowledge_relevance = self.support.check_knowledge_relevance(analysis_knowledge)
            
            # Step 4: Determine response strategy
            knowledge_response_sections = []
            strategy_result = self.support.determine_response_strategy(
                flow_type=flow_type,
                flow_confidence=flow_confidence,
                message_characteristics=message_characteristics,
                knowledge_relevance=knowledge_relevance,
                similarity_score=similarity_score,
                prior_knowledge=prior_knowledge,
                queries=queries,
                query_results=query_results,
                knowledge_response_sections=knowledge_response_sections
            )
            
            response_strategy = strategy_result["strategy"]
            strategy_instructions = strategy_result["instructions"]
            
            # Update knowledge context and similarity score from strategy
            if "knowledge_context" in strategy_result and strategy_result["knowledge_context"]:
                knowledge_context = strategy_result["knowledge_context"]
            if "similarity_score" in strategy_result:
                similarity_score = strategy_result["similarity_score"]
            
            # Handle fallback knowledge sections if needed
            if not knowledge_context and queries and query_results:
                fallback_context = self.support.build_knowledge_fallback_sections(queries, query_results)
                if fallback_context:
                    knowledge_context = fallback_context
            
            logger.info(f"Knowledge context: {knowledge_context}")
            
            # Step 5: Build and execute LLM prompt
            prompt = self.support.build_llm_prompt(
                message_str=message_str,
                conversation_context=conversation_context,
                temporal_context=temporal_context,
                knowledge_context=knowledge_context,
                response_strategy=response_strategy,
                strategy_instructions=strategy_instructions,
                core_prior_topic=prior_topic,
                user_id=user_id
            )
            
            # Step 6: Get LLM response and process it
            response = await LLM.ainvoke(prompt)
            logger.info(f"LLM response generated with similarity score: {similarity_score}")
            
            content = response.content.strip()
            
            # Step 7: Extract structured sections and metadata
            structured_sections = self.support.extract_structured_sections(content)
            content, tool_calls, evaluation = self.support.extract_tool_calls_and_evaluation(content)
            
            # Step 8: Handle teaching intent regeneration if needed
            if evaluation.get("has_teaching_intent", False) and response_strategy != "TEACHING_INTENT":
                content, response_strategy = await self._handle_teaching_intent_regeneration(
                    message_str, content, response_strategy
                )
            
            # Step 9: Extract user-facing content
            user_facing_content = self._extract_user_facing_content(content, response_strategy, structured_sections)
            
            # Step 10: Handle empty response fallbacks
            user_facing_content = self.support.handle_empty_response_fallbacks(
                user_facing_content, response_strategy, message_str
            )
            
            # Step 11: Build final response
            return {
                "status": "success",
                "message": user_facing_content,
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "user_id": user_id,
                    "additional_knowledge_found": bool(knowledge_context),
                    "similarity_score": similarity_score,
                    "response_strategy": response_strategy,
                    "core_prior_topic": prior_topic,
                    "tool_calls": tool_calls,
                    "has_teaching_intent": evaluation.get("has_teaching_intent", False),
                    "is_priority_topic": evaluation.get("is_priority_topic", False),
                    "priority_topic_name": evaluation.get("priority_topic_name", ""),
                    "should_save_knowledge": evaluation.get("should_save_knowledge", False),
                    "full_structured_response": content
                }
            }
            
        except ValueError as e:
            logger.error(f"Validation error in active learning: {str(e)}")
            return {"status": "error", "message": "Empty message provided"}
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {
                "status": "error",
                "message": "Tôi xin lỗi, nhưng tôi gặp lỗi khi xử lý yêu cầu của bạn. Vui lòng thử lại."
            }

    async def _handle_teaching_intent_regeneration(self, message_str: str, content: str, response_strategy: str) -> tuple:
        """Handle regeneration of response when teaching intent is detected."""
        original_strategy = response_strategy
        response_strategy = "TEACHING_INTENT"
        logger.info(f"LLM detected teaching intent, changing response_strategy from {original_strategy} to TEACHING_INTENT")
        
        teaching_prompt = f"""IMPORTANT: The user is TEACHING you something. Your job is to synthesize this knowledge.
        
        Original message: {message_str}
        
        Instructions:
        Generate THREE separate outputs in your response:
        
        1. <user_response>
           This is what the user will see - include:
           - Acknowledgment of their teaching with appreciation
           - Demonstration of your understanding
           - End with 1-2 open-ended questions to deepen the conversation
           - Make this conversational and engaging
        </user_response>
        
        2. <knowledge_synthesis>
           This is for knowledge storage - include ONLY:
           - Factual information extracted from the user's message
           - Structured, clear explanation of the concepts
           - NO greeting phrases, acknowledgments, or questions
           - NO conversational elements - pure knowledge only
           - Organized in logical sections if appropriate
        </knowledge_synthesis>
        
        3. <knowledge_summary>
           A concise 2-3 sentence summary capturing the core teaching point
           This should be factual and descriptive, not conversational
        </knowledge_summary>
        
        CRITICAL: RESPOND IN THE SAME LANGUAGE AS THE USER'S MESSAGE.
        - If the user wrote in Vietnamese, respond entirely in Vietnamese
        - If the user wrote in English, respond entirely in English
        - Match the language exactly - do not mix languages
        
        Your structured response:
        """
        
        try:
            teaching_response = await LLM.ainvoke(teaching_prompt)
            content = teaching_response.content.strip()
            logger.info("Successfully regenerated response with structured TEACHING_INTENT format")
        except Exception as e:
            logger.error(f"Failed to regenerate teaching response: {str(e)}")
            # Keep original content if regeneration fails
        
        return content, response_strategy

    def _extract_user_facing_content(self, content: str, response_strategy: str, structured_sections: Dict) -> str:
        """Extract the user-facing content from the response."""
        user_facing_content = content
        
        if response_strategy == "TEACHING_INTENT":
            # Extract user response (this is what should be sent to the user)
            if structured_sections.get("user_response"):
                user_facing_content = structured_sections["user_response"]
                logger.info(f"Extracted user-facing content from structured response")
            else:
                # Remove knowledge_synthesis and knowledge_summary sections if they exist
                user_facing_content = re.sub(r'<knowledge_synthesis>.*?</knowledge_synthesis>', '', content, flags=re.DOTALL).strip()
                user_facing_content = re.sub(r'<knowledge_summary>.*?</knowledge_summary>', '', user_facing_content, flags=re.DOTALL).strip()
                logger.info(f"Cleaned non-user sections from response")
        
        # Handle JSON responses
        if content.startswith('{') and '"message"' in content:
            try:
                parsed_json = json.loads(content)
                if isinstance(parsed_json, dict) and "message" in parsed_json:
                    user_facing_content = parsed_json["message"]
                    logger.info("Extracted message from JSON response")
            except Exception as json_error:
                logger.warning(f"Failed to parse JSON response: {json_error}")
        
        return user_facing_content

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
                self._create_background_task(self.support.background_save_knowledge(
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
            # Use streaming version for real-time response
            final_response = None
            tool_results = []
            
            async for chunk in learning_processor.process_incoming_message_streaming(
                user_message, 
                conversation_context, 
                user_id,
                thread_id
            ):
                if chunk.get("type") == "response_chunk":
                    # Stream chunks immediately to frontend
                    yield {"status": "streaming", "content": chunk["content"], "complete": False}
                elif chunk.get("type") == "response_complete":
                    # Store final response for post-processing
                    final_response = chunk
                    
                    # Extract message content and clean it
                    message_content = chunk["message"] if "message" in chunk else str(chunk)
                    message_content = re.split(r'<knowledge_queries>', message_content)[0].strip()
                    logger.info("Stripped knowledge_queries from message for frontend")
                    
                    # Execute any tool calls
                    if "metadata" in chunk and "tool_calls" in chunk["metadata"]:
                        for tool_call in chunk["metadata"]["tool_calls"]:
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
                    
                    # Yield final complete response with metadata
                    final_response_data = {
                        "status": "success", 
                        "message": message_content, 
                        "complete": True
                    }
                    
                    # Include metadata if available
                    if "metadata" in chunk:
                        final_response_data["metadata"] = chunk["metadata"]
                    
                    yield final_response_data
                    state.setdefault("messages", []).append({"role": "assistant", "content": message_content})
                    
                elif chunk.get("type") == "error":
                    # Handle errors
                    yield chunk
                    return

            # Force cleanup of tasks before returning final result
            if 'learning_processor' in state:
                await state['learning_processor'].cleanup()

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

    async def handle_teaching_intent(self, message: str, response: Dict[str, Any], user_id: str, thread_id: Optional[str], priority_topic_name: str) -> None:
        """Handle teaching intent detection and knowledge saving."""
        logger.info("Saving knowledge due to teaching intent")
        
        message_content = response["message"]
        conversational_response = re.split(r'<knowledge_queries>', message_content)[0].strip()
        query_section = re.search(r'<knowledge_queries>(.*?)</knowledge_queries>', message_content, re.DOTALL)
        knowledge_queries = json.loads(query_section.group(1).strip()) if query_section else []

        # NEW: Create conversation turns structure for multi-turn synthesis
        conversation_turns = [
            {
                "user": message,
                "ai": conversational_response
            }
        ]
        
        # Extract final synthesis content
        final_synthesis = conversational_response
        if response.get("metadata", {}).get("response_strategy") == "TEACHING_INTENT":
            # Try to extract knowledge_synthesis if available
            synthesis_match = re.search(r'<knowledge_synthesis>(.*?)</knowledge_synthesis>', message_content, re.DOTALL)
            if synthesis_match:
                final_synthesis = synthesis_match.group(1).strip()
        
        # Call the new alpha.py function to save teaching synthesis
        synthesis_result = await save_teaching_synthesis(
            conversation_turns=conversation_turns,
            final_synthesis=final_synthesis,
            topic=priority_topic_name or "user_teaching",
            user_id=user_id,
            thread_id=thread_id,
            priority_topic_name=priority_topic_name
        )
        
        logger.info(f"Teaching synthesis result: {synthesis_result}")
        
        # Store synthesis result in response metadata for tracking
        response["metadata"]["teaching_synthesis_result"] = synthesis_result

        # Set up categories and bank name
        categories = ["general"]
        bank_name = "conversation"
        
        # Add teaching intent category
        categories.append("teaching_intent")
        
        # Add specific topic category if provided by LLM
        if priority_topic_name and priority_topic_name not in categories:
            categories.append(priority_topic_name.lower().replace(" ", "_"))
        
        # Create synthesis_content early so it's available for combined_knowledge
        synthesis_content = conversational_response
        
        # Extract summary if present using regex
        summary = ""
        summary_match = re.search(r'SUMMARY:\s*(.*?)(?:\n|$)', conversational_response, re.IGNORECASE)
        if summary_match:
            summary = summary_match.group(1).strip()
            logger.info(f"Topic Summary Gen by LLM: {summary}")
        
        # Check for structured format sections
        user_response = ""
        knowledge_synthesis = ""
        knowledge_summary = ""
        
        # Extract structured sections from message_content
        user_response_match = re.search(r'<user_response>(.*?)</user_response>', message_content, re.DOTALL)
        synthesis_match = re.search(r'<knowledge_synthesis>(.*?)</knowledge_synthesis>', message_content, re.DOTALL)
        summary_match = re.search(r'<knowledge_summary>(.*?)</knowledge_summary>', message_content, re.DOTALL)
        
        # Try to extract from full response if available in metadata
        if not user_response_match and "full_structured_response" in response.get("metadata", {}):
            full_response = response["metadata"]["full_structured_response"]
            user_response_match = re.search(r'<user_response>(.*?)</user_response>', full_response, re.DOTALL)
            synthesis_match = re.search(r'<knowledge_synthesis>(.*?)</knowledge_synthesis>', full_response, re.DOTALL)
            summary_match = re.search(r'<knowledge_summary>(.*?)</knowledge_summary>', full_response, re.DOTALL)
            logger.info("Extracted structured sections from full_structured_response metadata")
        
        if user_response_match:
            user_response = user_response_match.group(1).strip()
            logger.info(f"Found structured user response section")
        
        if synthesis_match:
            knowledge_synthesis = synthesis_match.group(1).strip()
            logger.info(f"Found structured knowledge synthesis section")
            # Use the structured synthesis content if available
            synthesis_content = knowledge_synthesis
        
        if summary_match:
            knowledge_summary = summary_match.group(1).strip()
            logger.info(f"Found structured knowledge summary section: {knowledge_summary}")
            summary = knowledge_summary  # Use the explicit summary if available
        
        # Format synthesis_content with summary if available
        if summary and not synthesis_match:
            # Only add summary prefix if we don't have structured synthesis
            synthesis_content = f"SUMMARY: {summary}\n\nAI Synthesis: {conversational_response}"
        elif not synthesis_match:
            # Default format when no structured synthesis is found
            synthesis_content = f"AI Synthesis: {conversational_response}"
        
        logger.info(f"Final synthesis_content: {synthesis_content[:100]}...")
        
        # Combine user input and AI synthesis content in a formatted way
        combined_knowledge = f"User: {message}\n\nAI: {synthesis_content}"
        logger.info(f"Combined knowledge: {combined_knowledge}")
        
        # Add synthesized flag if this was from TEACHING_INTENT strategy
        if response.get("metadata", {}).get("response_strategy") == "TEACHING_INTENT":
            categories.append("synthesized_knowledge")
            logger.info(f"Adding synthesized_knowledge category for enhanced teaching content")
        
        # Check if we have structured format available
        has_structured_format = False
        if response.get("metadata", {}).get("full_structured_response", ""):
            full_response = response["metadata"]["full_structured_response"]
            if ("<user_response>" in full_response and 
                "<knowledge_synthesis>" in full_response and 
                "<knowledge_summary>" in full_response):
                has_structured_format = True
                logger.info("Detected structured format in response, will use that instead of combined knowledge")
        
        # Save combined knowledge only if we don't have structured format
        if not has_structured_format:
            await self._save_combined_knowledge(
                combined_knowledge, user_id, bank_name, thread_id, 
                priority_topic_name, categories, response
            )
        else:
            logger.info(f"Skipping combined knowledge save since structured format is available")

        # Save additional AI synthesis for TEACHING_INTENT strategy
        if response.get("metadata", {}).get("response_strategy") == "TEACHING_INTENT":
            await self._save_ai_synthesis(
                message, synthesis_content, knowledge_synthesis, knowledge_summary,
                summary, conversational_response, user_id, bank_name, thread_id,
                priority_topic_name, categories, response
            )

    async def _save_combined_knowledge(self, combined_knowledge: str, user_id: str, bank_name: str, 
                                     thread_id: Optional[str], priority_topic_name: str, 
                                     categories: List[str], response: Dict[str, Any]) -> None:
        """Save combined knowledge to the knowledge base."""
        try:
            logger.info(f"Saving combined knowledge to {bank_name} bank: '{combined_knowledge[:100]}...'")
            save_result = await self.support.background_save_knowledge(
                input_text=combined_knowledge,
                title="", # Add empty title parameter 
                user_id=user_id,
                bank_name=bank_name,
                thread_id=thread_id,
                topic=priority_topic_name or "user_teaching",
                categories=categories,
                ttl_days=365  # 365 days TTL
            )
            logger.info(f"Save combined knowledge completed: {save_result}")
            logger.info(f"Save result type: {type(save_result)}, content: {save_result}")
            
            # Store vector ID for frontend response
            if isinstance(save_result, dict):
                logger.info(f"Save result is dict, success: {save_result.get('success')}")
                if save_result.get("success"):
                    vector_id = save_result.get("vector_id")
                    response["metadata"]["combined_knowledge_vector_id"] = vector_id
                    logger.info(f"✅ Captured combined knowledge vector ID: {vector_id}")
                else:
                    logger.warning(f"Save operation failed: {save_result.get('error')}")
            else:
                logger.warning(f"Save result is not dict, got: {type(save_result)}")
        except Exception as e:
            logger.error(f"Error saving combined knowledge: {str(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
        
        logger.info(f"Saved combined knowledge for topic '{priority_topic_name or 'user_teaching'}'")

    async def _save_ai_synthesis(self, message: str, synthesis_content: str, knowledge_synthesis: str,
                               knowledge_summary: str, summary: str, conversational_response: str,
                               user_id: str, bank_name: str, thread_id: Optional[str],
                               priority_topic_name: str, categories: List[str], response: Dict[str, Any]) -> None:
        """Save AI synthesis and summary for improved future retrieval."""
        try:
            # Create categories list
            synthesis_categories = list(categories)
            synthesis_categories.append("ai_synthesis")
            
            logger.info(f"Saving additional AI synthesis for improved future retrieval")
            
            # Use the synthesis_content that was already created earlier
            # Determine storage format and title
            if knowledge_synthesis and knowledge_summary:
                logger.info(f"Using structured sections for knowledge storage")
                # Format for storage with clear separation
                final_synthesis_content = f"User: {message}\n\nAI: {knowledge_synthesis}"
                summary_title = knowledge_summary
                synthesis_categories = list(categories)
            else:
                # Use the synthesis_content that was already built
                logger.info(f"Using synthesis_content for knowledge storage")
                final_synthesis_content = f"User: {message}\n\nAI: {synthesis_content}"
                summary_title = summary or ""
                synthesis_categories = list(categories)
            
            logger.info(f"Final synthesis content to save: {final_synthesis_content[:100]}...")
            synthesis_result = await self.support.background_save_knowledge(
                input_text=final_synthesis_content,
                title=summary_title,
                user_id=user_id,
                bank_name=bank_name,
                thread_id=thread_id,
                topic=priority_topic_name or "user_teaching",
                categories=synthesis_categories,
                ttl_days=365  # 365 days TTL
            )
            logger.info(f"Save AI synthesis completed: {synthesis_result}")
            logger.info(f"Synthesis result type: {type(synthesis_result)}, content: {synthesis_result}")
            
            # Store synthesis vector ID
            if isinstance(synthesis_result, dict):
                logger.info(f"Synthesis result is dict, success: {synthesis_result.get('success')}")
                if synthesis_result.get("success"):
                    vector_id = synthesis_result.get("vector_id")
                    response["metadata"]["synthesis_vector_id"] = vector_id
                    logger.info(f"✅ Captured synthesis vector ID: {vector_id}")
                else:
                    logger.warning(f"Synthesis save failed: {synthesis_result.get('error')}")
            else:
                logger.warning(f"Synthesis result is not dict, got: {type(synthesis_result)}")
            
            # Save standalone summary if available
            await self._save_standalone_summary(
                knowledge_synthesis, knowledge_summary, summary, conversational_response,
                user_id, bank_name, thread_id, priority_topic_name, categories
            )
        except Exception as e:
            logger.error(f"Error saving AI synthesis: {str(e)}")

    async def _save_standalone_summary(self, knowledge_synthesis: str, knowledge_summary: str,
                                     summary: str, conversational_response: str, user_id: str,
                                     bank_name: str, thread_id: Optional[str], priority_topic_name: str,
                                     categories: List[str]) -> None:
        """Save standalone summary for quick retrieval."""
        if knowledge_summary:
            # Define categories for summary - but DON'T add "summary" to avoid filtering
            summary_categories = list(categories)
            summary_categories.append("knowledge_summary")  # Use a different category name
            summary_categories.append("ai_synthesis")  # Add ai_synthesis to summary instead
            logger.info(f"Saving standalone summary for quick retrieval")
            summary_success = await self.support.background_save_knowledge(
                input_text=f"AI: {knowledge_synthesis}",
                title=knowledge_summary,
                user_id=user_id,
                bank_name=bank_name,
                thread_id=thread_id,
                topic=priority_topic_name or "user_teaching",
                categories=summary_categories,
                ttl_days=365  # 365 days TTL
            )
            logger.info(f"Save summary completed: {summary_success}")
        elif summary and len(summary) > 10:  # Only save non-empty summaries
            summary_categories = list(categories)
            summary_categories.append("knowledge_summary")  # Use a different category name
            summary_categories.append("ai_synthesis")  # Add ai_synthesis to summary instead
            logger.info(f"Saving standalone summary (legacy format) for quick retrieval")
            summary_success = await self.support.background_save_knowledge(
                input_text=f"AI: {conversational_response}",
                title=summary,
                user_id=user_id,
                bank_name=bank_name,
                thread_id=thread_id,
                topic=priority_topic_name or "user_teaching",
                categories=summary_categories,
                ttl_days=365  # 365 days TTL
            )
            logger.info(f"Save legacy summary completed: {summary_success}")
        else:
            logger.info("No valid summary available, skipping standalone summary save")

    async def should_save_knowledge_with_similarity_gate(self, response: Dict[str, Any], similarity_score: float, message: str) -> Dict[str, Any]:
        """
        Determine if knowledge should be saved based on similarity score and conversation flow.
        CRITICAL: Teaching intent is REQUIRED for all knowledge saving to prevent pollution.
        """
        has_teaching_intent = response.get("metadata", {}).get("has_teaching_intent", False)
        is_priority_topic = response.get("metadata", {}).get("is_priority_topic", False)
        
        # High similarity threshold for automatic saving
        HIGH_SIMILARITY_THRESHOLD = 0.70
        MEDIUM_SIMILARITY_THRESHOLD = 0.35
        
        logger.info(f"Evaluating knowledge saving: similarity={similarity_score}, teaching_intent={has_teaching_intent}, priority_topic={is_priority_topic}")
        
        # CRITICAL: No teaching intent = NO SAVING (prevents knowledge base pollution)
        if not has_teaching_intent:
            logger.info(f"❌ No teaching intent detected - not saving knowledge (similarity={similarity_score:.2f})")
            return {
                "should_save": False,
                "reason": "no_teaching_intent",
                "confidence": "high",
                "encourage_context": similarity_score < MEDIUM_SIMILARITY_THRESHOLD
            }
        
        # From here on, teaching intent is confirmed - evaluate based on similarity
        
        # Case 1: High similarity + teaching intent - always save
        if similarity_score >= HIGH_SIMILARITY_THRESHOLD:
            logger.info(f"✅ High similarity ({similarity_score:.2f}) + teaching intent - saving knowledge")
            return {
                "should_save": True,
                "reason": "high_similarity_teaching",
                "confidence": "high"
            }
        
        # Case 2: Medium similarity + teaching intent - need clarification first
        if MEDIUM_SIMILARITY_THRESHOLD <= similarity_score < HIGH_SIMILARITY_THRESHOLD:
            logger.info(f"⚠️ Medium similarity ({similarity_score:.2f}) + teaching intent - requesting clarification")
            return {
                "should_save": False,
                "reason": "medium_similarity_needs_clarification",
                "confidence": "medium",
                "clarification_needed": True
            }
        
        # Case 3: Low similarity + teaching intent + substantial content - new knowledge
        if similarity_score < MEDIUM_SIMILARITY_THRESHOLD and len(message.split()) > 15:
            logger.info(f"✅ Low similarity ({similarity_score:.2f}) + teaching intent + substantial content - saving as new knowledge")
            return {
                "should_save": True,
                "reason": "new_knowledge_teaching",
                "confidence": "medium",
                "is_new_knowledge": True
            }
        
        # Case 4: Low similarity + teaching intent but insufficient content
        if similarity_score < MEDIUM_SIMILARITY_THRESHOLD:
            logger.info(f"❌ Low similarity ({similarity_score:.2f}) + teaching intent but insufficient content - encouraging more detail")
            return {
                "should_save": False,
                "reason": "teaching_intent_insufficient_content",
                "confidence": "medium",
                "encourage_context": True
            }
        
        # Default case - don't save
        logger.info(f"❌ Default case - not saving knowledge (similarity={similarity_score:.2f})")
        return {
            "should_save": False,
            "reason": "default_no_save",
            "confidence": "low"
        }

    async def enhance_response_for_knowledge_quality(self, response: Dict[str, Any], similarity_score: float, 
                                                   save_decision: Dict[str, Any], message: str) -> Dict[str, Any]:
        """
        Enhance the AI response to naturally guide conversation toward higher quality knowledge.
        """
        if not save_decision.get("clarification_needed") and not save_decision.get("encourage_context"):
            return response  # No enhancement needed
        
        original_message = response.get("message", "")
        
        # Case 1: Medium similarity - ask for clarification to reach high similarity
        if save_decision.get("clarification_needed"):
            clarification_prompt = f"""
            The user provided teaching content with medium confidence. Enhance the response to naturally ask for clarification that would improve understanding.
            
            Original response: {original_message}
            
            Add 1-2 specific follow-up questions that would help clarify:
            - Specific context or scenarios
            - Concrete examples
            - Step-by-step details
            - When/where this applies
            
            Make it conversational and genuinely curious, not interrogative.
            Keep the original response and add the clarification questions naturally.
            
            Enhanced response:
            """
            
            try:
                enhanced_response = await LLM.ainvoke(clarification_prompt)
                enhanced_content = enhanced_response.content.strip()
                
                # Update the response
                response["message"] = enhanced_content
                response["metadata"]["enhanced_for_clarification"] = True
                logger.info("Enhanced response to request clarification for better knowledge quality")
                
            except Exception as e:
                logger.error(f"Failed to enhance response for clarification: {str(e)}")
        
        # Case 2: Low similarity - encourage more context
        elif save_decision.get("encourage_context"):
            context_prompt = f"""
            The user's message had low similarity to existing knowledge. Enhance the response to naturally encourage them to provide more context.
            
            Original response: {original_message}
            User message: {message}
            
            Add a natural request for more context such as:
            - "Could you tell me more about..."
            - "What specific situation are you dealing with?"
            - "Can you share an example of..."
            - "What's your experience been with..."
            
            Make it genuinely helpful and curious, not pushy.
            Keep the original response and add the context request naturally.
            
            Enhanced response:
            """
            
            try:
                enhanced_response = await LLM.ainvoke(context_prompt)
                enhanced_content = enhanced_response.content.strip()
                
                # Update the response
                response["message"] = enhanced_content
                response["metadata"]["enhanced_for_context"] = True
                logger.info("Enhanced response to encourage more context for better knowledge quality")
                
            except Exception as e:
                logger.error(f"Failed to enhance response for context: {str(e)}")
        
        return response

    async def read_human_input(self, message: str, conversation_context: str, user_id: str, thread_id: Optional[str] = None) -> AsyncGenerator[Union[str, Dict], None]:
        """Streaming version of process_incoming_message that yields chunks as they come."""
        logger.info(f"Processing streaming message from user {user_id}")
        try:
            if not message.strip():
                logger.error("Empty message")
                yield {"status": "error", "message": "Empty message provided", "complete": True}
                return
            
            # Step 1: Get suggested knowledge queries from previous responses
            suggested_queries = []
            if conversation_context:
                ai_messages = re.findall(r'AI: (.*?)(?:\n\n|$)', conversation_context, re.DOTALL)
                if ai_messages:
                    last_ai_message = ai_messages[-1]
                    query_section = re.search(r'<knowledge_queries>(.*?)</knowledge_queries>', last_ai_message, re.DOTALL)
                    if query_section:
                        try:
                            queries_data = json.loads(query_section.group(1).strip())
                            suggested_queries = queries_data if isinstance(queries_data, list) else queries_data.get("queries", [])
                            logger.info(f"Extracted {len(suggested_queries)} suggested knowledge queries")
                        except json.JSONDecodeError:
                            logger.warning("Failed to parse query section as JSON")

            # Step 2: Search for relevant knowledge using iterative exploration
            logger.info(f"Searching for knowledge based on message using iterative exploration...")
            analysis_knowledge = await self.knowledge_explorer.explore(message, conversation_context, user_id, thread_id)
            
            # Step 3: Enrich with suggested queries if we don't have sufficient results
            if suggested_queries and len(analysis_knowledge.get("query_results", [])) < 3:
                logger.info(f"Searching for additional knowledge using {len(suggested_queries)} suggested queries")
                primary_similarity = analysis_knowledge.get("similarity", 0.0)
                primary_knowledge = analysis_knowledge.get("knowledge_context", "")
                primary_queries = analysis_knowledge.get("queries", [])
                primary_query_results = analysis_knowledge.get("query_results", [])
                
                for query in suggested_queries:
                    if query not in primary_queries:  # Avoid duplicate queries
                        query_knowledge = await self.support.search_knowledge(query, conversation_context, user_id, thread_id)
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
            
            # Step 5: Generate streaming response
            prior_data = analysis_knowledge.get("prior_data", {})
            
            # Stream the response as it comes from LLM
            final_response = None
            async for chunk in self._active_learning_streaming(message, conversation_context, analysis_knowledge, user_id, prior_data):
                if chunk.get("type") == "response_chunk":
                    # Yield streaming chunks immediately
                    yield chunk
                elif chunk.get("type") == "response_complete":
                    # Store final response for post-processing
                    final_response = chunk
                    yield chunk
                elif chunk.get("type") == "error":
                    yield chunk
                    return
            
            # Step 6: Enhanced knowledge saving with similarity gating (after streaming is complete)
            if final_response and final_response.get("status") == "success":
                # Get teaching intent and priority topic info from LLM evaluation
                metadata = final_response.get("metadata", {})
                has_teaching_intent = metadata.get("has_teaching_intent", False)
                is_priority_topic = metadata.get("is_priority_topic", False)
                should_save_knowledge = metadata.get("should_save_knowledge", False)
                priority_topic_name = metadata.get("priority_topic_name", "")
                intent_type = metadata.get("intent_type", "unknown")
                
                # Get similarity score from analysis
                similarity_score = analysis_knowledge.get("similarity", 0.0) if analysis_knowledge else 0.0
                
                # Use new similarity-based gating system
                save_decision = await self.should_save_knowledge_with_similarity_gate(final_response, similarity_score, message)
                
                logger.info(f"Knowledge saving decision: {save_decision}")
                logger.info(f"LLM evaluation: intent={intent_type}, teaching_intent={has_teaching_intent}, priority_topic={is_priority_topic}, similarity={similarity_score:.2f}")
                
                # Update metadata based on similarity gating decision
                if save_decision.get("should_save", False):
                    # Only update knowledge saving flags, NOT intent classification
                    final_response["metadata"]["should_save_knowledge"] = True
                    final_response["metadata"]["similarity_gating_reason"] = save_decision.get("reason", "")
                    final_response["metadata"]["similarity_gating_confidence"] = save_decision.get("confidence", "")
                    
                    # Only override teaching intent if LLM actually detected teaching intent
                    # Don't force teaching intent just because of high similarity
                    original_teaching_intent = final_response["metadata"].get("has_teaching_intent", False)
                    if original_teaching_intent:
                        final_response["metadata"]["response_strategy"] = "TEACHING_INTENT"
                        logger.info(f"Confirmed teaching intent with similarity gating - will save knowledge")
                    else:
                        logger.info(f"High similarity ({similarity_score:.2f}) - saving knowledge but preserving original intent classification")
                    
                    # Run knowledge saving in background to not block the response
                    if original_teaching_intent:
                        # Handle as teaching intent
                        self._create_background_task(
                            self.handle_teaching_intent(message, final_response, user_id, thread_id, priority_topic_name)
                        )
                    else:
                        # Save as regular high-quality conversation without teaching intent processing
                        self._create_background_task(
                            self._save_high_quality_conversation(message, final_response, user_id, thread_id)
                        )

        except Exception as e:
            logger.error(f"Error processing streaming message: {str(e)}")
            import traceback
            logger.error(f"Stack trace: {traceback.format_exc()}")
            yield {"status": "error", "message": f"Error: {str(e)}", "complete": True}

    async def _active_learning_streaming(self, message: Union[str, List], conversation_context: str = "", analysis_knowledge: Dict = None, user_id: str = "unknown", prior_data: Dict = None) -> AsyncGenerator[Union[str, Dict], None]:
        """Streaming version of active learning method that yields chunks as they come from LLM."""
        logger.info("Starting streaming active learning approach")
        
        try:
            # Step 1: Setup and validation
            temporal_context = self.support.setup_temporal_context()
            message_str = self.support.validate_and_normalize_message(message)
            
            # Step 2: Extract and organize data
            analysis_data = self.support.extract_analysis_data(analysis_knowledge)
            prior_data_extracted = self.support.extract_prior_data(prior_data)
            prior_messages = self.support.extract_prior_messages(conversation_context)
            
            knowledge_context = analysis_data["knowledge_context"]
            similarity_score = analysis_data["similarity_score"]
            queries = analysis_data["queries"]
            query_results = analysis_data["query_results"]
            prior_topic = prior_data_extracted["prior_topic"]
            prior_knowledge = prior_data_extracted["prior_knowledge"]
            
            logger.info(f"Using similarity score: {similarity_score}")
            
            # Step 3: Detect conversation flow and message characteristics
            flow_result = await self.support.detect_conversation_flow(
                message=message_str,
                prior_messages=prior_messages,
                conversation_context=conversation_context
            )
            
            flow_type = flow_result.get("flow_type", "NEW_TOPIC")
            flow_confidence = flow_result.get("confidence", 0.5)
            logger.info(f"Active learning conversation flow: {flow_type} (confidence: {flow_confidence})")
            
            message_characteristics = self.support.detect_message_characteristics(message_str)
            knowledge_relevance = self.support.check_knowledge_relevance(analysis_knowledge)
            
            # Step 4: Determine response strategy
            knowledge_response_sections = []
            strategy_result = self.support.determine_response_strategy(
                flow_type=flow_type,
                flow_confidence=flow_confidence,
                message_characteristics=message_characteristics,
                knowledge_relevance=knowledge_relevance,
                similarity_score=similarity_score,
                prior_knowledge=prior_knowledge,
                queries=queries,
                query_results=query_results,
                knowledge_response_sections=knowledge_response_sections
            )
            
            response_strategy = strategy_result["strategy"]
            strategy_instructions = strategy_result["instructions"]
            
            # Update knowledge context and similarity score from strategy
            if "knowledge_context" in strategy_result and strategy_result["knowledge_context"]:
                knowledge_context = strategy_result["knowledge_context"]
            if "similarity_score" in strategy_result:
                similarity_score = strategy_result["similarity_score"]
            
            # Handle fallback knowledge sections if needed
            if not knowledge_context and queries and query_results:
                fallback_context = self.support.build_knowledge_fallback_sections(queries, query_results)
                if fallback_context:
                    knowledge_context = fallback_context
            
            logger.info(f"Knowledge context: {knowledge_context}")
            
            # Step 5: Build LLM prompt
            prompt = self.support.build_llm_prompt(
                message_str=message_str,
                conversation_context=conversation_context,
                temporal_context=temporal_context,
                knowledge_context=knowledge_context,
                response_strategy=response_strategy,
                strategy_instructions=strategy_instructions,
                core_prior_topic=prior_topic,
                user_id=user_id
            )
            
            # Step 6: Stream LLM response and process it
            from langchain_openai import ChatOpenAI
            StreamLLM = ChatOpenAI(model="gpt-4o", streaming=True, temperature=0.01)
            
            content_buffer = ""
            evaluation_started = False
            logger.info("Starting LLM streaming response")
            
            async for chunk in StreamLLM.astream(prompt):
                chunk_content = chunk.content if hasattr(chunk, 'content') else str(chunk)
                if chunk_content:
                    content_buffer += chunk_content
                    
                    # Check if we're entering an evaluation section
                    if '<evaluation>' in content_buffer and not evaluation_started:
                        evaluation_started = True
                        # Extract content before evaluation and yield it
                        pre_evaluation = content_buffer.split('<evaluation>')[0]
                        remaining_content = pre_evaluation[len(content_buffer) - len(chunk_content):]
                        if remaining_content:
                            yield {
                                "type": "response_chunk",
                                "content": remaining_content,
                                "complete": False
                            }
                        continue
                    
                    # Skip chunks if we're in evaluation section
                    if evaluation_started and '</evaluation>' not in content_buffer:
                        continue
                    
                    # If evaluation section ended, reset flag and continue
                    if evaluation_started and '</evaluation>' in content_buffer:
                        evaluation_started = False
                        continue
                    
                    # Only yield chunks if we're not in evaluation section
                    if not evaluation_started:
                        yield {
                            "type": "response_chunk",
                            "content": chunk_content,
                            "complete": False
                        }
            
            logger.info(f"LLM streaming completed, total content length: {len(content_buffer)}")
            
            # Step 7: Process the complete response
            content = content_buffer.strip()
            
            # Extract structured sections and metadata
            structured_sections = self.support.extract_structured_sections(content)
            content, tool_calls, evaluation = self.support.extract_tool_calls_and_evaluation(content)
            
            # Step 8: Handle teaching intent regeneration if needed
            if evaluation.get("has_teaching_intent", False) and response_strategy != "TEACHING_INTENT":
                content, response_strategy = await self._handle_teaching_intent_regeneration(
                    message_str, content, response_strategy
                )
            
            # Step 9: Extract user-facing content
            user_facing_content = self._extract_user_facing_content(content, response_strategy, structured_sections)
            
            # Step 10: Handle empty response fallbacks
            user_facing_content = self.support.handle_empty_response_fallbacks(
                user_facing_content, response_strategy, message_str
            )
            
            # Step 11: Yield final complete response with metadata
            yield {
                "type": "response_complete",
                "status": "success",
                "message": user_facing_content,
                "complete": True,
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "user_id": user_id,
                    "additional_knowledge_found": bool(knowledge_context),
                    "similarity_score": similarity_score,
                    "response_strategy": response_strategy,
                    "core_prior_topic": prior_topic,
                    "tool_calls": tool_calls,
                    "has_teaching_intent": evaluation.get("has_teaching_intent", False),
                    "is_priority_topic": evaluation.get("is_priority_topic", False),
                    "priority_topic_name": evaluation.get("priority_topic_name", ""),
                    "should_save_knowledge": evaluation.get("should_save_knowledge", False),
                    "full_structured_response": content
                }
            }
            
        except ValueError as e:
            logger.error(f"Validation error in streaming active learning: {str(e)}")
            yield {"type": "error", "status": "error", "message": "Empty message provided", "complete": True}
        except Exception as e:
            logger.error(f"Error in streaming active learning: {str(e)}")
            yield {
                "type": "error",
                "status": "error", 
                "message": "Tôi xin lỗi, nhưng tôi gặp lỗi khi xử lý yêu cầu của bạn. Vui lòng thử lại.",
                "complete": True
            }

    async def _save_high_quality_conversation(self, message: str, response: Dict[str, Any], user_id: str, thread_id: Optional[str]) -> None:
        """Save high-quality conversation that doesn't have teaching intent but has high similarity."""
        try:
            logger.info("Saving high-quality conversation without teaching intent")
            
            # Extract response content
            response_content = response.get("message", "")
            similarity_score = response.get("metadata", {}).get("similarity_score", 0.0)
            
            # Create combined knowledge entry
            combined_knowledge = f"User: {message}\n\nAI: {response_content}"
            
            # Determine categories
            categories = ["general", "high_quality_conversation"]
            
            # Add similarity-based category
            if similarity_score >= 0.70:
                categories.append("high_similarity")
            
            # Save to knowledge base
            await self._save_combined_knowledge(
                combined_knowledge=combined_knowledge,
                user_id=user_id,
                bank_name="conversation",
                thread_id=thread_id,
                priority_topic_name="general_conversation",
                categories=categories,
                response=response
            )
            
            logger.info(f"Successfully saved high-quality conversation (similarity: {similarity_score:.3f})")
            
        except Exception as e:
            logger.error(f"Failed to save high-quality conversation: {str(e)}")

