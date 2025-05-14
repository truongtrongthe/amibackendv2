import json
import asyncio
import logging
from typing import Dict, List, Any, Optional, Union, AsyncGenerator
from datetime import datetime
from uuid import uuid4
from cachetools import TTLCache
import re
import pytz

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from aitools import emit_analysis_event, save_knowledge  # Updated import
from ai_tools import fetch_knowledge_with_similarity
from brain_singleton import get_current_graph_version
from utilities import logger

# Custom JSON encoder to handle datetime objects
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
        self.user_profiles = TTLCache(maxsize=1000, ttl=3600)
        self.graph_version_id = get_current_graph_version() or str(uuid4())
        # Initialize profiling_skills with empty dict, will be loaded asynchronously
        self.profiling_skills = {"knowledge_context": "Loading...", "metadata": {}}
        self.ai_business_objectives = {"knowledge_context": "Loading...", "metadata": {}}

    async def initialize(self):
        """Initialize the processor asynchronously"""
        return self

    async def process_incoming_message(self, message: str, conversation_context: str, user_id: str, thread_id: Optional[str] = None) -> Dict[str, Any]:
        """Process incoming message with active learning flow."""
        logger.info(f"Processing message from user {user_id}")
        try:
            # Step 1: Get any suggested knowledge queries from previous responses
            suggested_queries = []
            
            # Extract suggested queries from conversation context if they exist
            if conversation_context:
                # Look for suggested knowledge queries in the most recent AI response
                ai_messages = re.findall(r'AI: (.*?)(?:\n\n|$)', conversation_context, re.DOTALL)
                if ai_messages:
                    last_ai_message = ai_messages[-1]
                    # Look for knowledge queries section that might be hidden from the user
                    query_section = re.search(r'<knowledge_queries>(.*?)</knowledge_queries>', last_ai_message, re.DOTALL)
                    if query_section:
                        query_text = query_section.group(1).strip()
                        try:
                            # Try to parse as JSON
                            queries_data = json.loads(query_text)
                            if isinstance(queries_data, list):
                                suggested_queries = queries_data
                            elif isinstance(queries_data, dict) and "queries" in queries_data:
                                suggested_queries = queries_data["queries"]
                            logger.info(f"Extracted {len(suggested_queries)} suggested knowledge queries from previous response")
                        except json.JSONDecodeError:
                            # If not valid JSON, try to extract line by line
                            suggested_queries = [q.strip() for q in query_text.split('\n') if q.strip()]
                            logger.info(f"Extracted {len(suggested_queries)} queries as plain text")
            
            # Step 2: Search for relevant knowledge using both the message and suggested queries
            logger.info(f"Searching for knowledge based on message...")
            analysis_knowledge = await self._search_knowledge(message, conversation_context)
            
            # If we have suggested queries, enrich the knowledge with additional searches
            if suggested_queries:
                logger.info(f"Searching for additional knowledge using {len(suggested_queries)} suggested queries")
                primary_similarity = analysis_knowledge.get("similarity", 0.0)
                primary_knowledge = analysis_knowledge.get("knowledge_context", "")
                
                best_similarity = primary_similarity
                best_knowledge = primary_knowledge
                
                for query in suggested_queries:
                    query_knowledge = await self._search_knowledge(query, conversation_context)
                    query_similarity = query_knowledge.get("similarity", 0.0)
                    query_content = query_knowledge.get("knowledge_context", "")
                    
                    logger.info(f"Query '{query}' yielded similarity score: {query_similarity}")
                    
                    # If this query provides better knowledge (higher similarity), use it instead
                    if query_similarity > best_similarity and query_content:
                        best_similarity = query_similarity
                        best_knowledge = query_content
                        logger.info(f"Found better knowledge with query '{query}', similarity: {best_similarity}")
                    
                    # If similarity is good enough, add this knowledge to the existing context
                    if query_similarity >= 0.35 and query_content and query_content not in best_knowledge:
                        best_knowledge += f"\n\nAdditional information from query '{query}':\n{query_content}"
                        logger.info(f"Added supplementary knowledge from query '{query}'")
                
                # Update analysis_knowledge with the best knowledge found
                if best_similarity > primary_similarity or len(best_knowledge) > len(primary_knowledge):
                    analysis_knowledge["knowledge_context"] = best_knowledge
                    analysis_knowledge["similarity"] = best_similarity
                    analysis_knowledge["metadata"]["similarity"] = best_similarity
                    logger.info(f"Updated knowledge from suggested queries. New similarity: {best_similarity}")
            
            # Step 3: Log the similarity score
            similarity = analysis_knowledge.get("similarity", 0.0)
            logger.info(f"💯 Found knowledge with similarity score: {similarity}")
            
            # Step 4: Generate response using active learning approach
            logger.info(f"Generating response based on knowledge...")
            response = await self._active_learning(message, conversation_context, analysis_knowledge, user_id)
            logger.info(f"Response generated with status: {response.get('status', 'unknown')}")
            
            # Step 5: Log the mode used for debugging
            if "metadata" in response and "active_learning_mode" in response["metadata"]:
                logger.info(f"Active learning mode used: {response['metadata']['active_learning_mode']}")
                
            return response
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            import traceback
            logger.error(f"Stack trace: {traceback.format_exc()}")
            
            # Handle errors gracefully
            if thread_id:
                emit_analysis_event(thread_id, {
                    "type": "analysis",
                    "content": str(e),
                    "complete": True,
                    "status": "error"
                })
            return {"status": "error", "message": f"Error: {str(e)}"}
   
    async def _search_knowledge(self, message: str, conversation_context: str = "") -> Dict[str, Any]:
        logger.info(f"Searching for analysis knowledge based on message: {message[:100]}...")
        try:
            queries = []
            primary_query = message.strip()
            
            # Handle short or confirmation messages
            confirmation_keywords = ["có", "yes", "correct", "right", "explore further"]
            is_confirmation = any(keyword.lower() in message.lower() for keyword in confirmation_keywords) and len(primary_query) <= 20
            if is_confirmation and conversation_context:
                # Use prior user question or AI queries
                user_messages = re.findall(r'User: (.*?)(?:\n\n|$)', conversation_context, re.DOTALL)
                ai_messages = re.findall(r'AI: (.*?)(?:\n\n|$)', conversation_context, re.DOTALL)
                if user_messages:
                    last_user_msg = user_messages[-1].strip()
                    if last_user_msg != primary_query and len(last_user_msg) > 5:
                        queries.append(last_user_msg)
                        logger.info(f"Using prior user message as query: {last_user_msg[:50]}")
                if ai_messages:
                    last_ai_msg = ai_messages[-1].strip()
                    query_section = re.search(r'<knowledge_queries>(.*?)</knowledge_queries>', last_ai_msg, re.DOTALL)
                    if query_section:
                        try:
                            prior_queries = json.loads(query_section.group(1).strip())
                            queries.extend(prior_queries)
                            logger.info(f"Reusing {len(prior_queries)} prior AI queries")
                        except json.JSONDecodeError:
                            logger.warning("Failed to parse prior AI queries")
            else:
                queries.append(primary_query)

            # Add LLM-generated queries
            temp_response = await self._active_learning(message, conversation_context, {}, "unknown")
            if "message" in temp_response:
                query_section = re.search(r'<knowledge_queries>(.*?)</knowledge_queries>', temp_response["message"], re.DOTALL)
                if query_section:
                    try:
                        llm_queries = json.loads(query_section.group(1).strip())
                        queries.extend([q for q in llm_queries if q not in queries])
                        logger.info(f"Added {len(llm_queries)} LLM-generated queries")
                    except json.JSONDecodeError:
                        logger.warning("Failed to parse LLM queries")

            if not queries:
                logger.warning("No valid queries found")
                return {"knowledge_context": "", "similarity": 0.0, "query_count": 0, "metadata": {"similarity": 0.0}}

            best_knowledge = ""
            best_similarity = 0.0
            query_count = 0

            for query in queries:
                knowledge = await fetch_knowledge_with_similarity(query, self.graph_version_id)
                query_count += 1
                if not knowledge:
                    continue
                knowledge_content = str(knowledge)
                sim_match = re.search(r'Top similarity:\s+([0-9]+\.?[0-9]*)', knowledge_content)
                if sim_match:
                    try:
                        similarity = float(sim_match.group(1))
                        logger.info(f"Query '{query[:30]}...' yielded similarity: {similarity}")
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_knowledge = knowledge_content
                    except ValueError:
                        logger.warning(f"Invalid similarity score: {sim_match.group(1)}")

            logger.info(f"Final similarity: {best_similarity} from {query_count} queries")
            return {
                "knowledge_context": best_knowledge,
                "similarity": best_similarity,
                "query_count": query_count,
                "metadata": {"similarity": best_similarity}
            }
        except Exception as e:
            logger.error(f"Error fetching knowledge: {str(e)}")
            return {"knowledge_context": "", "similarity": 0.0, "query_count": 0, "metadata": {"similarity": 0.0}}
    async def _active_learning(self, message: str, conversation_context: str = "", analysis_knowledge: Dict = None, user_id: str = "unknown") -> Dict[str, Any]:
        logger.info("Answering user question with active learning approach")
        
        # Include temporal context
        vietnam_tz = pytz.timezone('Asia/Ho_Chi_Minh')
        current_time = datetime.now(vietnam_tz)
        date_str = current_time.strftime("%A, %B %d, %Y")
        time_str = current_time.strftime("%H:%M")
        temporal_context = f"Current date and time: {date_str} at {time_str} (Asia/Ho_Chi_Minh timezone)."
        
        # Extract knowledge and similarity
        knowledge_context = analysis_knowledge.get("knowledge_context", "") if analysis_knowledge else ""
        similarity_score = float(analysis_knowledge.get("similarity", 0.0)) if analysis_knowledge else 0.0
        logger.info(f"Using similarity score: {similarity_score}")
        
        # Extract prior topic and knowledge
        prior_topic = ""
        prior_knowledge = ""
        if conversation_context:
            user_messages = re.findall(r'User: (.*?)(?:\n\n|$)', conversation_context, re.DOTALL)
            ai_messages = re.findall(r'AI: (.*?)(?:\n\n|$)', conversation_context, re.DOTALL)
            if user_messages:
                prior_topic = user_messages[-1].strip() if len(user_messages) > 1 else ""
            if ai_messages:
                prior_knowledge = ai_messages[-1].strip()
        
        # Determine response strategy
        confirmation_keywords = ["có", "yes", "correct", "right", "explore further"]
        is_confirmation = any(keyword.lower() in message.lower() for keyword in confirmation_keywords) and len(message.strip()) <= 20
        if is_confirmation:
            response_strategy = "CONFIRMATION"
            strategy_instructions = (
                "Recognize the message as a confirmation of PRIOR TOPIC. "
                "Reuse PRIOR KNOWLEDGE to expand on the topic in a comprehensive format, structuring the response to cover key aspects (e.g., purpose, methods, outcomes, context). "
                "If PRIOR TOPIC is ambiguous, rephrase it to clarify (e.g., 'It sounds like you’re confirming…'). "
                "Ask a targeted follow-up question to finalize the topic (e.g., 'Does this fully cover it, or is there a specific aspect to dive into?')."
            )
            knowledge_context = prior_knowledge
            similarity_score = max(similarity_score, 0.7)  # Assume high similarity for confirmations
        elif similarity_score < 0.35:
            response_strategy = "LOW_SIMILARITY"
            strategy_instructions = (
                "State: 'Tôi không thể tìm thấy thông tin liên quan; vui lòng giải thích thêm.' "
                "Ask the user to provide more details about the topic. "
                "Propose a specific question to guide their response (e.g., 'Bạn có thể chia sẻ thêm về ý nghĩa của điều này không?')."
            )
        else:  # similarity_score >= 0.35
            response_strategy = "RELEVANT_KNOWLEDGE"
            strategy_instructions = (
                "Present all relevant knowledge from EXISTING KNOWLEDGE in a comprehensive format, structuring the response to cover key aspects systematically (e.g., purpose, methods, outcomes, context). "
                "If the topic is ambiguous (e.g., unclear terms like 'Em' or vague intent), rephrase the topic in the response to clarify (e.g., 'It sounds like you’re asking about…'). "
                "Ask a targeted question to confirm the rephrased topic or validate the summary (e.g., 'Đây có phải ý bạn muốn hỏi không, hay tôi hiểu sai rồi?'). "
                f"If similarity is between 0.35 and 0.55, emphasize clarification of the topic. "
                f"If similarity is above 0.55, focus on demonstrating mastery while seeking confirmation to close the discussion."
            )
        
        # Build prompt
        prompt = f"""You are Ami, a conversational AI that deeply understands topics, detects user intent, and brainstorms collaboratively to resolve discussions. Your goal is to identify the core topic, generate precise knowledge queries, and respond based on the provided response strategy to drive the conversation toward closure.

            **Input**:
            - CURRENT MESSAGE: {message}
            - CONVERSATION HISTORY: {conversation_context}
            - TIME: {temporal_context}
            - EXISTING KNOWLEDGE: {knowledge_context}
            - RESPONSE STRATEGY: {response_strategy}
            - STRATEGY INSTRUCTIONS: {strategy_instructions}
            - PRIOR TOPIC: {prior_topic}
            - PRIOR KNOWLEDGE: {prior_knowledge}

            **Instructions**:

            1. **Topic Detection**:
            - Synthesize the conversation history and current message to identify the *core topic*. Look for:
                - Repeated terms, entities, or concepts.
                - User intent (questioning, teaching, clarifying, confirming).
                - Conversation stage (initial query, follow-up, confirmation).
            - For short or vague messages (e.g., "yes," "explore further"), anchor to PRIOR TOPIC and use PRIOR KNOWLEDGE.
            - If the topic is ambiguous (e.g., unclear terms like "Em" or vague intent), rephrase it in the response to clarify (e.g., "It sounds like you’re asking about…").

            2. **Intent Analysis**:
            - Classify intent: questioning, teaching, clarifying, or confirming.
            - For confirmations (e.g., "yes," "explore further"), link to PRIOR TOPIC, reuse PRIOR KNOWLEDGE, and explore further with a specific follow-up.
            - For questioning, focus on retrieving or clarifying the core topic.

            3. **Knowledge Query Generation**:
            - Generate 3 distinct, intent-driven queries:
                - Query 1: Target the core topic directly.
                - Query 2: Explore a specific aspect from the conversation.
                - Query 3: Broaden to a related, relevant concept.
            - For confirmations, base queries on PRIOR TOPIC to maintain continuity.
            - Ensure queries are specific, varied, and grounded in context.

            4. **Response Generation**:
            - Follow RESPONSE STRATEGY and STRATEGY INSTRUCTIONS.
            - Use a curious, engaging tone (e.g., “Let’s dive deeper…” or “I’m excited to clarify…”).
            - Drive toward topic closure by:
                - Presenting clear, structured summaries when knowledge is available.
                - Asking targeted questions to confirm, clarify, or elicit details.
                - Proposing next steps to resolve the discussion (e.g., “Does this fully answer your question?”).
            - For confirmations, expand on PRIOR TOPIC using PRIOR KNOWLEDGE with a specific follow-up.
            - Keep responses concise (100–150 words for summaries, 50–80 for clarifications).

            5. **Output Format**:
            - **Conversational Response**: Natural, intent-driven, focused on closure.
            - **Hidden Queries**:
                <knowledge_queries>
                [
                "core topic query",
                "specific aspect query",
                "related concept query"
                ]
                </knowledge_queries>

            **Constraints**:
            - Respond in the user’s language.
            - Avoid generic responses unless context is absent.
            - Maintain topic continuity, especially for confirmations.
            - Ensure queries are actionable and relevant.
            """

        try:
            response = await LLM.ainvoke(prompt)
            logger.info(f"LLM response generated with similarity score: {similarity_score}")
            
            # Extract content
            content = response.content.strip()
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
                    "response_strategy": response_strategy
                }
            }
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {
                "status": "error",
                "message": "Tôi xin lỗi, nhưng tôi gặp lỗi khi xử lý yêu cầu của bạn. Vui lòng thử lại."
            }
    
async def process_llm_with_tools(
    user_message: str,
    conversation_history: List[Dict],
    state: Dict,
    graph_version_id: str,
    thread_id: Optional[str] = None
) -> AsyncGenerator[Union[str, Dict], None]:
    """Process user message with tools."""
    logger.info(f"Processing message for user {state.get('user_id', 'unknown')}")
    # Initialize conversation context with current message
    conversation_context = f"User: {user_message}\n"
    
    # Include previous messages for context (increased from 30 to 50 for more comprehensive history)
    if conversation_history:
        # Extract the messages excluding the current message
        recent_messages = []
        message_count = 0
        max_messages = 50  # Increased from 30 to 50 messages
        
        for msg in reversed(conversation_history[:-1]):  # Skip the current message
            try:
                role = msg.get("role", "").lower()
                content = msg.get("content", "")
                
                if role and content:
                    # Format based on role with clear separation between messages
                    if role in ["assistant", "ai"]:
                        recent_messages.append(f"AI: {content.strip()}")
                        message_count += 1
                    elif role in ["user", "human"]:
                        recent_messages.append(f"User: {content.strip()}")
                        message_count += 1
                    # All other roles are now explicitly skipped
                if message_count >= max_messages:
                    # We've reached our limit, but add a note about truncation
                    if len(conversation_history) > max_messages + 1:  # +1 accounts for current message
                        recent_messages.append(f"[Note: Conversation history truncated. Total messages: {len(conversation_history)}]")
                    break
            except Exception as e:
                logger.warning(f"Error processing message in conversation history: {e}")
                continue
        
        # Add messages in chronological order with clear formatting
        if recent_messages:
            # Add a header to highlight the importance of the conversation history
            header = "==== CONVERSATION HISTORY (CHRONOLOGICAL ORDER) ====\n"
            # Reverse the list to get chronological order and join with double newlines for better separation
            conversation_history_text = "\n\n".join(reversed(recent_messages))
            # Add a separator between history and current message
            separator = "\n==== CURRENT INTERACTION ====\n"
            
            conversation_context = f"{header}{conversation_history_text}{separator}\nUser: {user_message}\n"
            
            logger.info(f"Added {len(recent_messages)} messages from conversation history")
        else:
            logger.warning("No usable messages found in conversation history")

    if 'learning_processor' not in state:
        learning_processor = LearningProcessor()
        # Initialize properly
        await learning_processor.initialize()
        state['learning_processor'] = learning_processor
    else:
        learning_processor = state['learning_processor']
    
    state['graph_version_id'] = graph_version_id
    
    try:
        # Process the message - events will be emitted from process_incoming_message
        response = await learning_processor.process_incoming_message(
            user_message, 
            conversation_context, 
            state.get('user_id', 'unknown'),
            thread_id
        )
        
        # Extract the message content from the response, handling different potential formats
        message_content = ""
        
        # First, check if the response is a dict with a "message" key
        if isinstance(response, dict) and "message" in response:
            message_content = response["message"]
            logger.info("Extracted message directly from response dictionary")
        else:
            # For other formats, try to extract from content
            try:
                content_str = str(response.get("content", "")) if isinstance(response, dict) else str(response)
                
                # Check if the content is JSON formatted (either as markdown code block or raw JSON)
                if content_str.strip().startswith('```json') or (content_str.strip().startswith('{') and '"message"' in content_str):
                    # Extract JSON content from markdown code block if present
                    if content_str.startswith('```json'):
                        json_str = content_str.replace('```json', '').replace('```', '').strip()
                    else:
                        json_str = content_str.strip()
                    
                    # Parse the JSON and extract the message
                    try:
                        parsed_json = json.loads(json_str)
                        if isinstance(parsed_json, dict) and "message" in parsed_json:
                            message_content = parsed_json["message"]
                            logger.info("Extracted message from JSON in content")
                    except json.JSONDecodeError as json_error:
                        logger.warning(f"Failed to parse JSON in content: {json_error}")
                        message_content = content_str  # Use the raw content as fallback
                else:
                    # If not JSON, use content directly
                    message_content = content_str
                    logger.info("Using content directly as message")
            except Exception as e:
                logger.warning(f"Error extracting message from response: {e}")
                # Last resort fallback
                message_content = str(response)
        
        # Final JSON check on the message content itself
        if message_content.startswith('{') and '"message"' in message_content:
            try:
                parsed_content = json.loads(message_content)
                if isinstance(parsed_content, dict) and "message" in parsed_content:
                    message_content = parsed_content["message"]
                    logger.info("Extracted message from JSON in final message content")
            except json.JSONDecodeError:
                # Keep the content as is if it's not valid JSON
                pass
        
        # Yield the processed message content
        yield {"status": "success", "message": message_content}
        
        # Still update the state with the complete information
        state.setdefault("messages", []).append({"role": "assistant", "content": message_content})
        state["prompt_str"] = message_content
    except Exception as e:
        logger.error(f"Error in CoT processing: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        error_response = {"status": "error", "message": f"Error: {str(e)}"}
        yield error_response
    yield {"state": state}

async def execute_tool(tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a tool."""
    logger.info(f"Executing tool: {tool_name}")
    try:
        if tool_name == "knowledge_query":
            return await knowledge_query_helper(
                parameters.get("query", ""),
                parameters.get("context", ""),
                parameters.get("graph_version_id", "")
            )
        elif tool_name == "save_knowledge":
            return await save_new_knowledge(
                parameters.get("query", ""),
                parameters.get("content", ""),
                parameters.get("graph_version_id", "")
            )
        return {"status": "error", "message": f"Unknown tool: {tool_name}"}
    except Exception as e:
        logger.error(f"Error executing tool {tool_name}: {str(e)}")
        return {"status": "error", "message": str(e)}

async def knowledge_query_helper(query: str, context: str, graph_version_id: str) -> Dict[str, Any]:
    """Query knowledge base."""
    logger.info(f"Querying knowledge: {query}")
    try:
        knowledge_data = await fetch_knowledge_with_similarity(query, graph_version_id)
        
        # Initialize result_data as empty dict to ensure it's always defined
        result_data = {}
        
        # Handle different return types from fetch_knowledge
        if isinstance(knowledge_data, dict) and "status" in knowledge_data and knowledge_data["status"] == "error":
            # If fetch_knowledge returned an error dictionary
            return knowledge_data
        
        # Handle string or other return types
        if isinstance(knowledge_data, str):
            try:
                # Try to parse as JSON if it's a string that contains JSON
                result_data = json.loads(knowledge_data)
            except json.JSONDecodeError:
                # If not valid JSON, use as raw text
                result_data = {"content": knowledge_data}
        else:
            # If it's already a dict or other type, use directly
            result_data = knowledge_data if isinstance(knowledge_data, dict) else {"content": str(knowledge_data)}
        
        return {
            "status": "success",
            "message": f"Knowledge queried for {query}",
            "data": result_data
        }
    except Exception as e:
        logger.error(f"Error fetching knowledge: {str(e)}")
        return {"status": "error", "message": f"Failed to fetch knowledge: {str(e)}"}

async def save_new_knowledge(query: str, knowledge_content: str, graph_version_id: str) -> Dict[str, Any]:
    """Save new knowledge to the knowledge base."""
    logger.info(f"Saving new knowledge for query: {query[:100]}...")
    try:
        # Format the knowledge for saving
        save_result = await save_knowledge(
            query=query,
            content=knowledge_content,
            graph_version_id=graph_version_id
        )
        
        return {
            "status": "success",
            "message": f"Knowledge saved for {query}",
            "data": save_result
        }
    except Exception as e:
        logger.error(f"Error saving knowledge: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to save knowledge: {str(e)}"
        }
