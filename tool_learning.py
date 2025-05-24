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
from utilities import logger,EMBEDDINGS

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
            logger.info(f"LLM response raw: {response}")
            
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
                        if "SUMMARY:" not in response["message"]:
                            topic_extract = message[:50] + ("..." if len(message) > 50 else "")
                            response["message"] += f"\n\nSUMMARY: {topic_extract} là một chiến lược quan trọng khi tương tác với khách hàng."
                            logger.info("Added fallback SUMMARY section to existing response")
                
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
                        # Save combined knowledge
                        try:
                            logger.info(f"Saving combined knowledge to {bank_name} bank: '{combined_knowledge[:100]}...'")
                            save_result = await self._background_save_knowledge(
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
                    else:
                        logger.info(f"Skipping combined knowledge save since structured format is available")

                    # If we have synthesized content, save an additional entry with just the AI response
                    # This helps with future retrievals by isolating the clean synthesized knowledge
                    if response.get("metadata", {}).get("response_strategy") == "TEACHING_INTENT":
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
                            synthesis_result = await self._background_save_knowledge(
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
                            if knowledge_summary:
                                # Define categories for summary - but DON'T add "summary" to avoid filtering
                                summary_categories = list(categories)
                                summary_categories.append("knowledge_summary")  # Use a different category name
                                summary_categories.append("ai_synthesis")  # Add ai_synthesis to summary instead
                                logger.info(f"Saving standalone summary for quick retrieval")
                                summary_success = await self._background_save_knowledge(
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
                                summary_success = await self._background_save_knowledge(
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
            flow_result = await self._detect_conversation_flow(
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
                    top_k=10,
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
                
                # Xử lý tất cả các kết quả từ một query, không chỉ kết quả đầu tiên
                for result_item in results:
                    knowledge_content = result_item["raw"]
                    
                    # Extract just the User portion if this is a combined knowledge entry
                    if knowledge_content.startswith("User:") and "\n\nAI:" in knowledge_content:
                        user_part = re.search(r'User:(.*?)(?=\n\nAI:)', knowledge_content, re.DOTALL)
                        if user_part:
                            knowledge_content = user_part.group(1).strip()
                            logger.info(f"Extracted User portion from combined knowledge")
                    
                    # Evaluate context relevance between query and retrieved knowledge
                    context_relevance = await self._evaluate_context_relevance(primary_query, knowledge_content)
                    
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
                # Đơn giản hóa cách xây dựng combined_knowledge_context
                # Lấy top 5 vectors có liên quan nhất (hoặc ít hơn nếu không đủ)
                # Sort by similarity to present most relevant first
                sorted_results = sorted(zip(best_results, highest_similarities, knowledge_contexts), 
                                      key=lambda pair: pair[1], reverse=True)
                
                # Đơn giản hóa hoàn toàn quá trình xử lý kết quả
                # Không cần cơ chế chống trùng lặp phức tạp
                
                # Format response sections
                knowledge_response_sections = []
                knowledge_response_sections.append("KNOWLEDGE RESULTS:")
                
                # Log số lượng kết quả
                result_count = min(len(sorted_results), 5)  # Tối đa 5 kết quả
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
        
        # Extract prior messages from conversation context
        prior_messages = []
        if conversation_context:
            user_messages = re.findall(r'User: (.*?)(?:\n\n|$)', conversation_context, re.DOTALL)
            if user_messages and len(user_messages) > 1:
                prior_messages = user_messages[:-1]  # All except current message
        
        # Use the new conversation flow detection instead of the old pattern matching
        flow_result = await self._detect_conversation_flow(
            message=message_str,
            prior_messages=prior_messages,
            conversation_context=conversation_context
        )
        
        flow_type = flow_result.get("flow_type", "NEW_TOPIC")
        flow_confidence = flow_result.get("confidence", 0.5)
        
        is_confirmation = flow_type == "CONFIRMATION"
        is_follow_up = flow_type in ["FOLLOW_UP", "CONFIRMATION"]
        is_practice_request = flow_type == "PRACTICE_REQUEST"
        is_closing = flow_type == "CLOSING"
        
        logger.info(f"Active learning conversation flow: {flow_type} (confidence: {flow_confidence})")
        
        core_prior_topic = prior_topic
        
        # Knowledge handling strategy based on queries and similarity
        knowledge_response_sections = []
        
        # Enhanced closing message detection with conversation flow
        closing_phrases = [
            "thế thôi", "hẹn gặp lại", "tạm biệt", "chào nhé", "goodbye", "bye", "cảm ơn nhé", 
            "cám ơn nhé", "đủ rồi", "vậy là đủ", "hôm nay vậy là đủ", "hẹn lần sau"
        ]
        
        # Use both pattern matching and LLM detection for closing
        is_closing_message = is_closing or any(phrase in message_str.lower() for phrase in closing_phrases)
        
        # Check context relevance of best knowledge if available
        best_context_relevance = 0.0
        has_low_relevance_knowledge = False
        
        if analysis_knowledge and "query_results" in analysis_knowledge:
            query_results = analysis_knowledge.get("query_results", [])
            if query_results and isinstance(query_results[0], dict) and "context_relevance" in query_results[0]:
                best_context_relevance = query_results[0].get("context_relevance", 0.0)
                has_low_relevance_knowledge = best_context_relevance < 0.3
                logger.info(f"Best knowledge context relevance: {best_context_relevance}")
        
        # Check for teaching intent in the message
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
        elif is_practice_request and prior_knowledge:
            # Special handling for practice requests
            response_strategy = "PRACTICE_REQUEST"
            strategy_instructions = (
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
                "ACTIVELY USE these techniques in your response format, not just talk about them. For example: "
                "- If knowledge mentions using 'em' to refer to yourself, use that pronoun in your response "
                "- If it suggests addressing users as 'anh/chị', use that form of address "
                "- If it recommends specific phrases or compliments, incorporate them naturally "
                "- If it suggests question techniques, use those exact techniques at the end of your response"
            )
            # For practice requests, consider all retrieved knowledge
            knowledge_context = prior_knowledge
            # Increase similarity score for practice requests to ensure we use prior knowledge
            similarity_score = max(similarity_score, 0.8)
            logger.info(f"Detected practice request, boosting similarity score to {similarity_score} and using response strategy: PRACTICE_REQUEST")
        elif has_low_relevance_knowledge and similarity_score > 0.3:
            # Handle case where we have knowledge but it's not very relevant to the current query
            response_strategy = "LOW_RELEVANCE_KNOWLEDGE"
            strategy_instructions = (
                "You have knowledge with low relevance to the current query. "
                "PRIORITIZE the user's current message over the retrieved knowledge. "
                "ONLY reference the knowledge if it genuinely helps answer the query. "
                "If the knowledge is off-topic, IGNORE it completely and focus on the user's message. "
                "Be clear and direct in addressing what the user is actually asking about. "
                "Generate a response primarily based on the user's current message and intent."
                
                "However, if the knowledge contains ANY communication techniques or relationship-building approaches, "
                "incorporate those techniques into HOW you construct your response, even if the topic is different."
            )
            
            # Add context information to knowledge_context to alert the LLM
            knowledge_context = f"LOW RELEVANCE KNOWLEDGE WARNING: The retrieved knowledge has low relevance " \
                f"(score: {best_context_relevance:.2f}) to the current query. Prioritize the user's message.\n\n" \
                f"{knowledge_context}"
            
            logger.info(f"Using LOW_RELEVANCE_KNOWLEDGE strategy due to relevance score: {best_context_relevance:.2f}")
        elif queries and query_results:
            # Simplified response strategy that preserves multi-vector output
            # Just use the knowledge_context that was already built in _search_knowledge
            if knowledge_context:
                logger.info(f"Using pre-built knowledge context with multiple results")
                # Don't modify knowledge_context - preserve it with all items
                # Debug log to confirm we're not filtering knowledge
                logger.info(f"Knowledge context length: {len(knowledge_context)}")
                # Use existing knowledge_context without re-filtering
            else:
                # Fallback if knowledge_context is empty but we have results
                high_confidence = []
                medium_confidence = []
                low_confidence = []
                
                # Phần fallback khi knowledge_context trống
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

        logger.info(f"Knowledge context: {knowledge_context}")
        
        # Initial response strategy determination
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
        # If this is just a name or greeting, treat it as a greeting
        elif (is_vn_greeting or contains_vn_name) and len(message_str.split()) <= 3:
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
        # If this appears to be a teaching intent message (longer messages without questions, or containing teaching markers)
        elif has_teaching_markers or (len(message_str.split()) > 20 and "?" not in message_str):
            response_strategy = "TEACHING_INTENT"
            strategy_instructions = (
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
            )
            logger.info(f"Detected teaching intent message, using TEACHING_INTENT strategy")
        else:
            response_strategy = "RELEVANT_KNOWLEDGE"
            strategy_instructions = (
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
        try:
            response = await LLM.ainvoke(prompt)
            logger.info(f"LLM response generated with similarity score: {similarity_score}")
            
            content = response.content.strip()
            tool_calls = []
            evaluation = {"has_teaching_intent": False, "is_priority_topic": False, "priority_topic_name": "", "should_save_knowledge": False, "intent_type": "query", "name_addressed": False, "ai_referenced": False}
            
            # First, check if we have structured sections
            user_response = ""
            knowledge_synthesis = ""
            knowledge_summary = ""
            
            # Extract structured sections
            user_response_match = re.search(r'<user_response>(.*?)</user_response>', content, re.DOTALL)
            if user_response_match:
                user_response = user_response_match.group(1).strip()
                logger.info(f"Found user_response section in initial response")
            
            synthesis_match = re.search(r'<knowledge_synthesis>(.*?)</knowledge_synthesis>', content, re.DOTALL)
            if synthesis_match:
                knowledge_synthesis = synthesis_match.group(1).strip()
                logger.info(f"Found knowledge_synthesis section in initial response")
            
            summary_match = re.search(r'<knowledge_summary>(.*?)</knowledge_summary>', content, re.DOTALL)
            if summary_match:
                knowledge_summary = summary_match.group(1).strip()
                logger.info(f"Found knowledge_summary section in initial response")
            
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
                        
                        # Override response_strategy based on LLM evaluation
                        if evaluation.get("has_teaching_intent", False) == True:
                            original_strategy = response_strategy
                            response_strategy = "TEACHING_INTENT"
                            logger.info(f"LLM detected teaching intent, changing response_strategy from {original_strategy} to TEACHING_INTENT")
                            
                            # Update strategy_instructions for teaching intent
                            strategy_instructions = (
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
                                
                                "CRITICAL LANGUAGE INSTRUCTION: ALWAYS respond in EXACTLY the SAME LANGUAGE as the user's message. "
                                "- If the user wrote in Vietnamese, respond entirely in Vietnamese "
                                "- If the user wrote in English, respond entirely in English "
                                "- Do not mix languages in your response "
                                
                                "This structured approach helps create high-quality, reusable knowledge while maintaining good user experience."
                            )
                            
                            # Generate a new response with teaching intent instructions if needed
                            if original_strategy != "TEACHING_INTENT":
                                logger.info("Regenerating response with TEACHING_INTENT instructions")
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
                                    teaching_content = teaching_response.content.strip()
                                    
                                    # Extract user response section for what the user sees
                                    user_response_match = re.search(r'<user_response>(.*?)</user_response>', teaching_content, re.DOTALL)
                                    if user_response_match:
                                        user_response = user_response_match.group(1).strip()
                                        # Update the response with just the user-facing content
                                        content = teaching_content  # Keep full content for knowledge processing
                                        
                                        # Create a new dict for response instead of modifying the AIMessage
                                        response = {
                                            "status": "success",
                                            "message": teaching_content,  # Store full structured content for knowledge saving
                                            "metadata": {
                                                "response_strategy": "TEACHING_INTENT",
                                                "has_teaching_intent": True
                                            }
                                        }
                                        
                                        # For the user-facing response, extract just the user part
                                        message_content = user_response
                                        logger.info("Successfully regenerated response with structured TEACHING_INTENT format")
                                        logger.info(f"Extracted user-facing content: {message_content[:100]}...")
                                    else:
                                        # Check if there's a SUMMARY section, add one if missing
                                        if "SUMMARY:" not in teaching_content:
                                            # Add default summary at the end if missing
                                            teaching_content += f"\n\nSUMMARY: This knowledge involves {message_str[:30]}... and requires further development."
                                        
                                        # Replace the original content
                                        content = teaching_content
                                        response["message"] = teaching_content
                                        logger.info("Successfully regenerated response with TEACHING_INTENT format")
                                except Exception as e:
                                    logger.error(f"Failed to regenerate teaching response: {str(e)}")
                                    # Keep original content if regeneration fails
                    except json.JSONDecodeError:
                        logger.warning("Failed to parse evaluation")
            
            # Extract just the user-facing response from structured sections if present
            user_facing_content = content
            if response_strategy == "TEACHING_INTENT":
                # Extract user response (this is what should be sent to the user)
                user_response_match = re.search(r'<user_response>(.*?)</user_response>', content, re.DOTALL)
                if user_response_match:
                    user_facing_content = user_response_match.group(1).strip()
                    logger.info(f"Extracted user-facing content from structured response")
                else:
                    # Remove knowledge_synthesis and knowledge_summary sections if they exist
                    user_facing_content = re.sub(r'<knowledge_synthesis>.*?</knowledge_synthesis>', '', content, flags=re.DOTALL).strip()
                    user_facing_content = re.sub(r'<knowledge_summary>.*?</knowledge_summary>', '', user_facing_content, flags=re.DOTALL).strip()
                    logger.info(f"Cleaned non-user sections from response")
            
            if content.startswith('{') and '"message"' in content:
                try:
                    parsed_json = json.loads(content)
                    if isinstance(parsed_json, dict) and "message" in parsed_json:
                        content = parsed_json["message"]
                        logger.info("Extracted message from JSON response")
                except Exception as json_error:
                    logger.warning(f"Failed to parse JSON response: {json_error}")
            
            # Ensure closing messages get a response even if empty
            if response_strategy == "CLOSING" and (not user_facing_content or user_facing_content.isspace()):
                # Default closing message if the LLM didn't provide one
                if "vietnamese" in message_str.lower() or any(vn_word in message_str.lower() for vn_word in ["tạm biệt", "cảm ơn", "hẹn gặp", "thế thôi"]):
                    user_facing_content = "Vâng, cảm ơn bạn đã trao đổi. Hẹn gặp lại bạn lần sau nhé!"
                else:
                    user_facing_content = "Thank you for the conversation. Have a great day and I'm here if you need anything else!"
                logger.info("Added default closing response for empty LLM response")
            
            # Ensure unclear or short queries also get a helpful response when content is empty
            elif (not user_facing_content or user_facing_content.isspace()):
                # Check if message is short (1-2 words) or unclear
                is_short_message = len(message_str.strip().split()) <= 2
                
                # Default response for short/unclear messages
                if "vietnamese" in message_str.lower() or any(vn_word in message_str.lower() for vn_word in ["anh", "chị", "bạn", "cô", "ông", "xin", "vui lòng"]):
                    user_facing_content = f"Xin lỗi, tôi không hiểu rõ câu hỏi '{message_str}'. Bạn có thể chia sẻ thêm thông tin hoặc đặt câu hỏi cụ thể hơn được không?"
                else:
                    user_facing_content = f"I'm sorry, I didn't fully understand your message '{message_str}'. Could you please provide more details or ask a more specific question?"
                
                logger.info(f"Added default response for empty LLM response to short/unclear query: '{message_str}'")
            
            return {
                "status": "success",
                "message": user_facing_content,
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
                    "should_save_knowledge": evaluation.get("should_save_knowledge", False),
                    "full_structured_response": content  # Store the full structured response for knowledge saving
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

    async def _background_save_knowledge(self, input_text: str, title: str, user_id: str, bank_name: str, 
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

    async def _evaluate_context_relevance(self, user_input: str, retrieved_knowledge: str) -> float:
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

    async def _detect_conversation_flow(self, message: str, prior_messages: List[str], conversation_context: str) -> Dict[str, Any]:
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