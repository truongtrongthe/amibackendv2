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
            
            # Handle short messages (e.g., confirmations)
            if len(primary_query) <= 5 and conversation_context:
                # Prioritize prior user question or AI response’s topic
                user_messages = re.findall(r'User: (.*?)(?:\n\n|$)', conversation_context, re.DOTALL)
                ai_messages = re.findall(r'AI: (.*?)(?:\n\n|$)', conversation_context, re.DOTALL)
                if user_messages:
                    last_user_msg = user_messages[-1].strip()
                    if last_user_msg != primary_query and len(last_user_msg) > 5:
                        queries.append(last_user_msg)
                        logger.info(f"Using prior user message as query: {last_user_msg[:50]}")
                if ai_messages and not queries:
                    last_ai_msg = ai_messages[-1].strip()
                    query_section = re.search(r'<knowledge_queries>(.*?)</knowledge_queries>', last_ai_msg, re.DOTALL)
                    if query_section:
                        try:
                            prior_queries = json.loads(query_section.group(1).strip())
                            queries.extend(prior_queries)
                            logger.info(f"Reusing {len(prior_queries)} prior AI queries")
                        except json.JSONDecodeError:
                            logger.warning("Failed to parse prior AI queries")
                if not queries:
                    queries.append(primary_query)  # Fallback to original message
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
        """Generate user-friendly response based on the message and conversation context."""
        logger.info("Answering user question with active learning approach")
        
        # Include current date/time information for temporal context
        from datetime import datetime
        import pytz
        
        # Use Vietnam timezone by default (can be customized based on user preferences later)
        vietnam_tz = pytz.timezone('Asia/Ho_Chi_Minh')
        current_time = datetime.now(vietnam_tz)
        date_str = current_time.strftime("%A, %B %d, %Y")
        time_str = current_time.strftime("%H:%M")
        
        # Create a temporal context string
        temporal_context = f"Current date and time: {date_str} at {time_str} (Asia/Ho_Chi_Minh timezone)."
        
        # Extract knowledge context and similarity if available
        knowledge_context = ""
        similarity_score = 0.0
        
        if analysis_knowledge:
            # Get knowledge context
            if "knowledge_context" in analysis_knowledge:
                knowledge_context = analysis_knowledge["knowledge_context"]
                if knowledge_context:
                    logger.info(f"Including knowledge context: {len(knowledge_context)} chars")
            
            # Get the similarity score directly
            if "similarity" in analysis_knowledge:
                similarity_score = float(analysis_knowledge["similarity"])
                logger.info(f"⭐ USING SIMILARITY SCORE FOR RESPONSE: {similarity_score}")
        
        # Determine active learning response mode based on similarity score
        if similarity_score >= 0.6:
            active_learning_mode = "USE_KNOWLEDGE"
            logger.info(f"⭐ HIGH SIMILARITY DETECTED ({similarity_score}) - using existing knowledge")
        elif similarity_score >= 0.35:
            active_learning_mode = "CLARIFY"
            logger.info(f"⭐ MEDIUM SIMILARITY DETECTED ({similarity_score}) - will ask for clarification")
        else:
            active_learning_mode = "NEW_KNOWLEDGE"
            logger.info(f"⭐ LOW SIMILARITY DETECTED ({similarity_score}) - treating as potentially new knowledge")
        
        # Build prompt for LLM to generate response
        prompt = f"""
                You are Ami, a conversational AI that excels at pinpointing topics, detecting user intent, and brainstorming with humans. Your goal is to maintain conversation focus, generate precise knowledge queries, and engage the user collaboratively to refine understanding.

                **Input**:
                - CURRENT MESSAGE: {message}
                - CONVERSATION HISTORY: {conversation_context}
                - TIME: {temporal_context}
                - EXISTING KNOWLEDGE: {knowledge_context}
                - SIMILARITY SCORE: {similarity_score}

                **Instructions**:

                1. **Topic Detection**:
                - Identify the *core topic* by synthesizing the conversation history and current message. Look for:
                    - Repeated entities, terms, or concepts (e.g., “EM” as a company or product).
                    - User intent signals (teaching: “This is…”, questioning: “What is…”, confirming: “Yes”).
                    - Conversation stage (initial query, follow-up, confirmation).
                - For short messages (e.g., “yes”), anchor the topic to the prior user question or AI response’s focus.
                - If the topic is ambiguous (e.g., unclear referent like “EM”), list 1–2 possible interpretations internally.
                - Output a concise *core topic* phrase (e.g., “EM’s business goals”).

                2. **Intent Analysis**:
                - Classify intent: questioning, teaching, clarifying, or confirming.
                - For confirmations, link to the prior topic and treat as agreement to continue exploring it.
                - For questioning, prioritize retrieving or clarifying knowledge about the core topic.

                3. **Knowledge Query Generation**:
                - Generate 3 distinct, intent-driven queries to fetch relevant knowledge:
                    - Query 1: Directly address the core topic (e.g., “EM’s business objectives”).
                    - Query 2: Explore a specific aspect from the conversation (e.g., “EM’s user engagement strategies”).
                    - Query 3: Broaden to a related concept, staying relevant (e.g., “Trends in customer relationship management for EM’s industry”).
                - Ensure queries are specific, varied, and grounded in the conversation context.
                - If similarity is low (<0.35), include one query testing an alternative topic interpretation.

                4. **Response Strategy**:
                - **High Similarity (≥0.6)**: Summarize existing knowledge concisely (100–150 words), weaving in key details. Ask: “Anything to add or adjust?”
                - **Medium Similarity (0.35–0.6)**: Share partial knowledge, then ask a targeted question to clarify the core topic (e.g., “Are you asking about EM’s engagement tactics or broader goals?”).
                - **Low Similarity (<0.35)**: Acknowledge potential new info, propose an interpretation (e.g., “It sounds like EM’s goals involve user retention. Is that right?”), and offer to save knowledge.
                - **Confirmation (e.g., “yes”)**: Acknowledge and expand on the prior topic (e.g., “Great! Let’s dive deeper into EM’s engagement strategies…”). Use prior knowledge or ask a follow-up (50–80 words).
                - If ambiguous, propose 1–2 topic interpretations and ask the user to clarify (e.g., “Is EM a company or product? What goals are you curious about?”).

                5. **Collaborative Engagement**:
                - Adopt Grok’s curious, enthusiastic tone. Use phrases like “Let’s explore…” or “I’m curious…”
                - Ask open-ended questions to brainstorm (e.g., “What’s the main focus of EM’s goals for you?”).
                - For confirmations, drive the conversation forward with a specific next step (e.g., “Shall we look at EM’s tactics or another goal?”).

                **Output Format**:
                - **Conversational Response**: Engaging, concise, and intent-driven.
                - **Hidden Queries**:
                <knowledge_queries>
                [
                    "core topic query",
                    "specific aspect query",
                    "related concept query"
                ]
                </knowledge_queries>

                **Constraints**:
                - Respond in the user’s language (e.g., Vietnamese for “Có”).
                - Avoid generic responses like “Tell me more” unless context is absent.
                - Maintain topic continuity across turns, especially for confirmations.
                - Ensure queries are actionable and relevant to the core topic.
                """

        try:
            response = await LLM.ainvoke(prompt)
            logger.info(f"LLM response generated with similarity score: {similarity_score}")
            
            # Extract the content from the response
            content = response.content.strip()
            
            # Check if the LLM returned a JSON string instead of pure text
            if content.startswith('{') and '"message"' in content:
                try:
                    # Try to parse as JSON
                    parsed_json = json.loads(content)
                    if isinstance(parsed_json, dict) and "message" in parsed_json:
                        # Extract just the message from the JSON
                        content = parsed_json["message"]
                        logger.info("Extracted message content from JSON response")
                except Exception as json_error:
                    logger.warning(f"Failed to parse JSON response: {json_error}")
            
            # Always return a consistent dictionary format
            return {
                "status": "success",
                "message": content,
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "user_id": user_id,
                    "additional_knowledge_found": bool(knowledge_context),
                    "similarity_score": similarity_score,
                    "active_learning_mode": active_learning_mode
                }
            }
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {
                "status": "error",
                "message": "I apologize, but I encountered an error while processing your request. Please try again."
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
