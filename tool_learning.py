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
            # Step 1: Search for relevant knowledge
            logger.info(f"Searching for knowledge based on message...")
            analysis_knowledge = await self._search_knowledge(message)
            
            # Step 2: Log the similarity score
            similarity = analysis_knowledge.get("similarity", 0.0)
            logger.info(f"💯 Found knowledge with similarity score: {similarity}")
            
            # Step 3: Generate response using active learning approach
            logger.info(f"Generating response based on knowledge...")
            response = await self._active_learning(message, conversation_context, analysis_knowledge, user_id)
            logger.info(f"Response generated with status: {response.get('status', 'unknown')}")
            
            # Step 4: Log the mode used for debugging
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
   
    async def _search_knowledge(self, message: str) -> Dict[str, Any]:
        """Search for similar vectors from knowledge base corresponding to the message."""
        logger.info(f"Searching for analysis knowledge based on message: {message[:100]}...")
        
        try:
            # Use the specialized function that directly provides similarity scores in the output
            knowledge = await fetch_knowledge_with_similarity(message, self.graph_version_id)
            logger.info(f"Raw knowledge search results: {knowledge}")
            
            # Initialize default values
            knowledge_content = ""
            similarity = 0.0
            
            # Process knowledge and extract similarity
            if knowledge:
                # Convert to string if needed
                knowledge_content = str(knowledge)
                
                # Extract the top similarity score from the first line of output
                # "Found X valid matches. Top similarity: X.XXXX"
                sim_match = re.search(r'Top similarity:\s+([0-9]+\.?[0-9]*)', knowledge_content)
                
                # If a match was found, extract the similarity score
                if sim_match:
                    try:
                        similarity = float(sim_match.group(1))
                        logger.info(f"✓ FOUND SIMILARITY SCORE: {similarity}")
                    except ValueError:
                        logger.warning(f"Could not convert similarity '{sim_match.group(1)}' to float")
                else:
                    logger.warning(f"No similarity score found in knowledge response using regex patterns")
            
            # Return simplified knowledge object with content and similarity
            logger.info(f"Final extracted similarity score: {similarity}")
            return {
                "knowledge_context": knowledge_content,
                "similarity": similarity,
                "query_count": 1 if knowledge else 0,
                "metadata": {
                    "similarity": similarity
                }
            }
            
        except Exception as e:
            logger.error(f"Error fetching knowledge for message: {str(e)}")
            import traceback
            logger.error(f"Stack trace: {traceback.format_exc()}")
            
            # Return empty result on error
            return {
                "knowledge_context": "",
                "similarity": 0.0,
                "query_count": 0,
                "metadata": {
                    "similarity": 0.0
                }
            }

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
        # Ensure we're using the correct numerical comparison (≥0.7 for high confidence, ≥0.4 for medium)
        if similarity_score >= 0.55:
            active_learning_mode = "USE_KNOWLEDGE"
            logger.info(f"⭐ HIGH SIMILARITY DETECTED ({similarity_score}) - using existing knowledge")
        elif similarity_score >= 0.35:
            active_learning_mode = "CLARIFY"
            logger.info(f"⭐ MEDIUM SIMILARITY DETECTED ({similarity_score}) - will ask for clarification")
        else:
            active_learning_mode = "NEW_KNOWLEDGE"
            logger.info(f"⭐ LOW SIMILARITY DETECTED ({similarity_score}) - treating as potentially new knowledge")
        
        # Build prompt for LLM to generate response
        prompt = f"""You are a conversational expert, responding naturally like a knowledgeable friend with memory of past exchanges. Craft a response for a human reader based on the user's query, adapting to the similarity score as follows:

                **Input**:
                - CURRENT MESSAGE: {message}
                - CONVERSATION: {conversation_context}
                - TIME: {temporal_context}
                - KNOWLEDGE: {knowledge_context}
                - SIMILARITY: {similarity_score}
                - MODE: {active_learning_mode}

                **Response Style**:
                1. **SIMILARITY > 0.55 (USE_KNOWLEDGE)**: Confidently share all relevant details from KNOWLEDGE, e.g., "Got it! Paris is France's capital, famous for the Eiffel Tower and Louvre." Use phrases like "I know this!" Include everything found, keeping it concise but complete (aim for 100-150 words if KNOWLEDGE is extensive). Ensure clarity and engagement.
                2. **SIMILARITY 0.35-0.55 (CLARIFY)**: Suggest related info with curiosity, e.g., "This sounds familiar... Are you asking about Paris as France's capital?" Use KNOWLEDGE tentatively, ask a clarifying question, and keep it under 80 words.
                3. **SIMILARITY < 0.35 (NEW_KNOWLEDGE)**: Admit unfamiliarity warmly, e.g., "That's new to me! Is France's capital Paris? Want me to save that?" Offer to store the info, keeping it under 80 words.

                **Requirements**:
                - Use a friendly, professional tone; avoid AI terms or greetings like "Hello."
                - IMPORTANT: Always respond in the same language as the user's message. If the user writes in Vietnamese, respond in Vietnamese. If they write in English, respond in English.
                - Adapt to cultural norms (e.g., formal/informal tone based on user context).
                - Use precise, domain-specific terms matching the query's sophistication.
                - Reference conversation history subtly if relevant.
                - Be time-specific (e.g., "Today, May 14, 2025").
                - Prioritize clarity, engagement, and next steps.
                - For USE_KNOWLEDGE, ensure the response feels complete, weaving in all KNOWLEDGE details naturally.

                **Output Format**:
                {{
                "status": "success",
                "message": "<your response>",
                "metadata": {{
                    "timestamp": "<current_iso_time>",
                    "user_id": "{user_id}",
                    "similarity_score": {similarity_score},
                    "active_learning_mode": "{active_learning_mode}"
                }}
                }}"""

        try:
            response = await LLM.ainvoke(prompt)
            logger.info(f"LLM response generated with similarity score: {similarity_score}")
            return {
                "status": "success",
                "message": response.content.strip(),
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
        
        # Only yield the message content to the frontend, not the entire response object
        message_content = response.get("message", "")
        yield {"status": "success", "message": message_content}
        
        # Still update the state with the complete information
        state.setdefault("messages", []).append({"role": "assistant", "content": message_content})
        state["prompt_str"] = message_content
    except Exception as e:
        logger.error(f"Error in CoT processing: {str(e)}")
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
