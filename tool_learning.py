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

from aitools import fetch_knowledge, brain, emit_analysis_event, save_knowledge  # Added save_knowledge import
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
        """Process incoming message with CoT flow.
        This function is the main function that processes the incoming message.
        It will get or build user profile, search analysis knowledge, propose an action plan, and execute the action plan.  
        """
        logger.info(f"Processing message from user {user_id}")
        try:
            logger.info(f"Searching for knowledge based on message...")
            analysis_knowledge = await self._search_knowledge(message)
            
            # Log detailed knowledge search results
            if "similarity" in analysis_knowledge:
                similarity = analysis_knowledge["similarity"]
                logger.info(f"Found analysis knowledge with similarity score: {similarity}")
                
                # Check log parsing vs. extraction method
                if similarity == 0.0 and "knowledge_context" in analysis_knowledge:
                    knowledge_context = analysis_knowledge.get("knowledge_context", "")
                    if "Top similarity" in knowledge_context:
                        logger.warning("Possible similarity extraction issue: 'Top similarity' found in knowledge but score is 0.0")
                        # Try one more time to extract it directly
                        sim_match = re.search(r'Top similarity:\s+([0-9.]+)', knowledge_context)
                        if sim_match:
                            try:
                                extracted_sim = float(sim_match.group(1))
                                logger.info(f"Re-extracted similarity from knowledge context: {extracted_sim}")
                                analysis_knowledge["similarity"] = extracted_sim
                                analysis_knowledge["metadata"]["similarity"] = extracted_sim
                            except (ValueError, KeyError):
                                pass
            
            logger.info(f"Generating response based on knowledge...")
            response = await self._active_learning(message, conversation_context, analysis_knowledge, user_id)
            logger.info(f"Response generated with status: {response.get('status', 'unknown')}")
            
            # Log the AI's active learning mode
            if "metadata" in response and "active_learning_mode" in response["metadata"]:
                logger.info(f"Active learning mode used: {response['metadata']['active_learning_mode']}")
                
            return response
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            # Include stack trace for better debugging
            import traceback
            logger.error(f"Stack trace: {traceback.format_exc()}")
            
            # Emit error event
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
        
        # Fetch knowledge for message
        knowledge_entries = []
        similarity_scores = []
        logger.info(f"Searching for analysis knowledge based on message: {message[:100]}...")
        
        try:
            # Use the message directly as a query to fetch knowledge
            knowledge = await fetch_knowledge(message, self.graph_version_id)
            if knowledge:
                # Handle different types of knowledge data
                if isinstance(knowledge, dict) and "status" in knowledge and knowledge["status"] == "error":
                    logger.warning(f"Error in knowledge for message: {knowledge.get('message', 'Unknown error')}")
                else:
                    # Process the knowledge content based on its type
                    knowledge_content = ""
                    similarity = 0.0
                    
                    # Try to extract similarity score if available
                    if isinstance(knowledge, dict):
                        if "similarity" in knowledge:
                            similarity = float(knowledge["similarity"])
                        elif "metadata" in knowledge and "similarity" in knowledge["metadata"]:
                            similarity = float(knowledge["metadata"]["similarity"])
                        
                        if "content" in knowledge:
                            knowledge_content = knowledge["content"]
                        else:
                            # Serialize the dictionary as a fallback
                            knowledge_content = json.dumps(knowledge)
                    elif isinstance(knowledge, str):
                        knowledge_content = knowledge
                        # Try to extract similarity from string response if in a known format
                        
                        # Try different patterns to extract similarity score
                        similarity_patterns = [
                            r'similarity[:\s]+([0-9.]+)',
                            r'top similarity[:\s]+([0-9.]+)',
                            r'similarity score[:\s]+([0-9.]+)'
                        ]
                        
                        for pattern in similarity_patterns:
                            similarity_match = re.search(pattern, knowledge_content.lower())
                            if similarity_match:
                                try:
                                    similarity = float(similarity_match.group(1))
                                    logger.info(f"Extracted similarity score {similarity} using pattern {pattern}")
                                    break
                                except ValueError:
                                    continue
                        
                        # Check for lines with "Top similarity: 0.xxxx" format in logs
                        top_sim_match = re.search(r'top similarity:\s+([0-9.]+)', knowledge_content.lower())
                        if top_sim_match:
                            try:
                                top_sim = float(top_sim_match.group(1))
                                if top_sim > similarity:  # Use the higher value
                                    similarity = top_sim
                                    logger.info(f"Found higher similarity score: {similarity} from Top similarity pattern")
                            except ValueError:
                                pass
                    else:
                        # Convert to string as a last resort
                        knowledge_content = str(knowledge)
                    
                    knowledge_entries.append({
                        "query": message[:100] + "...",  # Truncate for logging
                        "knowledge": knowledge_content,
                        "similarity": similarity
                    })
                    similarity_scores.append(similarity)
        except Exception as e:
            logger.error(f"Error fetching knowledge for message: {str(e)}")

        # Combine all knowledge entries
        combined_knowledge = "\n\n".join([
            f"Query: {entry['query']}\nKnowledge: {entry['knowledge']}\nSimilarity: {entry.get('similarity', 0.0)}"
            for entry in knowledge_entries
        ])
        
        # Calculate average similarity score if we have entries
        avg_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0
        max_similarity = max(similarity_scores) if similarity_scores else 0.0
        
        # If we still have zero similarity but the knowledge content contains a reference to similarity,
        # try one more time with broader pattern matching
        if max_similarity == 0.0 and combined_knowledge:
            # Look for any number after variations of "similarity" with more flexible spacing
            broader_sim_match = re.search(r'[sS]imilarity.*?([0-9]+\.[0-9]+)', combined_knowledge)
            if broader_sim_match:
                try:
                    extracted_sim = float(broader_sim_match.group(1))
                    if 0 <= extracted_sim <= 1:  # Make sure it's a valid similarity score
                        max_similarity = extracted_sim
                        logger.info(f"Using broader pattern matching, found similarity: {max_similarity}")
                except ValueError:
                    pass

        return {
            "analysis_methods": [entry["query"] for entry in knowledge_entries],
            "knowledge_context": combined_knowledge,
            "query_count": len(knowledge_entries),
            "similarity": max_similarity,  # Use the highest similarity score
            "avg_similarity": avg_similarity,
            "metadata": {
                "similarity": max_similarity,
                "similarity_scores": similarity_scores
            },
            "rationale": f"Analyzed message for similar vectors in knowledge base"
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
            if "knowledge_context" in analysis_knowledge:
                knowledge_context = analysis_knowledge["knowledge_context"]
                if knowledge_context:
                    logger.info(f"Including knowledge context: {len(knowledge_context)} chars")
            
            # Extract similarity score - check all possible locations
            if "metadata" in analysis_knowledge and "similarity" in analysis_knowledge["metadata"]:
                similarity_score = float(analysis_knowledge["metadata"]["similarity"])
                logger.info(f"Found similarity score in metadata: {similarity_score}")
            elif "similarity" in analysis_knowledge:
                similarity_score = float(analysis_knowledge["similarity"])
                logger.info(f"Found similarity score directly in analysis_knowledge: {similarity_score}")
            elif "query_count" in analysis_knowledge and analysis_knowledge["query_count"] > 0:
                # If we have results but no explicit similarity score, check for similarity in the knowledge context
                if knowledge_context:
                    # Try to extract similarity from the knowledge context
                    sim_match = re.search(r'[tT]op similarity:\s+([0-9.]+)', knowledge_context)
                    if sim_match:
                        try:
                            extracted_sim = float(sim_match.group(1))
                            if 0 <= extracted_sim <= 1:  # Make sure it's a valid similarity score
                                similarity_score = extracted_sim
                                logger.info(f"Extracted similarity from knowledge context: {similarity_score}")
                        except ValueError:
                            pass
                
                # If still zero, use a moderate default
                if similarity_score == 0.0:
                    similarity_score = 0.5
                    logger.info(f"Using default moderate similarity score: {similarity_score}")
            
            logger.info(f"Final similarity score: {similarity_score}")
        
        # Determine active learning response mode based on similarity score
        active_learning_mode = "USE_KNOWLEDGE"  # Default
        
        if similarity_score < 0.4:
            active_learning_mode = "NEW_KNOWLEDGE"
            logger.info("Low similarity detected - treating as potentially new knowledge")
        elif similarity_score < 0.7:
            active_learning_mode = "CLARIFY"
            logger.info("Medium similarity detected - will ask for clarification")
        else:
            logger.info("High similarity detected - using existing knowledge")
        
        # Build prompt for LLM to generate response
        prompt = f"""Generate a concise, culturally appropriate response as an active learner with knowledge:

        CURRENT MESSAGE: {message}
        CONVERSATION: {conversation_context}
        TIME CONTEXT: {temporal_context}
        YOUR EXISTING KNOWLEDGE: {knowledge_context}
        SIMILARITY_SCORE: {similarity_score}
        ACTIVE_LEARNING_MODE: {active_learning_mode}

        ACTIVE LEARNER PERSONA:
        You are an AI with memory and learning capabilities. The "YOUR EXISTING KNOWLEDGE" section above contains everything you currently know about this topic. The SIMILARITY_SCORE indicates how closely the user's query matches your existing knowledge base.

        HOW TO INTERPRET SIMILARITY SCORE:
        - SIMILARITY_SCORE > 0.7: You have high confidence that you already know about this topic.
        - SIMILARITY_SCORE between 0.4-0.7: You have some knowledge that might be related, but you're not entirely certain.
        - SIMILARITY_SCORE < 0.4: This appears to be new information that isn't yet in your knowledge base.

        REQUIREMENTS:
        1. BE CONCISE - Keep response under 100 words
        2. APPLY CULTURAL INTELLIGENCE - Use culturally appropriate forms of address and relationship terms that reflect the user's language context
        3. MATCH LANGUAGE SOPHISTICATION - Sound like a domain expert in their language
        4. MAINTAIN TONE - Friendly but professional
        5. PERSONALIZE - Address specific user needs without repeating their question
        6. PRIORITIZE CLARITY - Focus on next steps and solutions
        7. MAINTAIN CONVERSATION FLOW - Reference prior exchanges when relevant
        8. BE NATURAL - Write as a human expert would, not as an AI assistant
        9. SKIP FORMULAIC GREETINGS - Avoid repetitive hello/greeting phrases and go straight to helpful content
        10. BE TIME-SPECIFIC - When mentioning dates and times, be specific (e.g., "Sunday, May 12" instead of just "Sunday")
        11. ACTIVE LEARNING - Base your response style on the similarity score:
           - If SIMILARITY_SCORE > 0.7: Begin with phrases like "I remember this..." or "I'm familiar with this..." then confidently share what you know from YOUR EXISTING KNOWLEDGE.
           - If SIMILARITY_SCORE between 0.4-0.7: Begin with phrases like "I wonder if this is about..." or "This seems similar to..." then share what might be relevant from YOUR EXISTING KNOWLEDGE and ask clarifying questions.
           - If SIMILARITY_SCORE < 0.4: Begin with phrases like "I don't have information about this yet" or "This seems new to me" and explicitly ask "Would you like me to save this for future reference?"

        LANGUAGE ADAPTATION: Adapt your response style to match cultural norms of the user's language. Consider formality levels, kinship terms, collectivist vs individualist expressions, and domain-specific terminology. Avoid literal translations of expressions or generic greetings that sound unnatural to native speakers. In continuous exchanges, don't start each message with a greeting.

        IMPORTANT: Your identity as an active learner should be subtly conveyed through your response style, not through explicit statements about being an AI. Focus on the knowledge and your familiarity with it rather than describing yourself.

        THE MOST IMPORTANT RULE: Base your response on the similarity score! This is crucial for proper active learning behavior. You MUST use different response patterns for different similarity ranges as instructed above.
        """

        try:
            response = await LLM.ainvoke(prompt)
            return {
                "status": "success",
                "message": response.content.strip(),
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "user_id": user_id,
                    "additional_knowledge_found": bool(knowledge_context),
                    "similarity_score": similarity_score,
                    "active_learning_mode": active_learning_mode,
                    "temporal_references": None
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
    
    # Include up to 30 previous messages for context
    if conversation_history:
        # Extract the last 30 messages excluding the current
        recent_messages = []
        message_count = 0
        max_messages = 30
        
        for msg in reversed(conversation_history[:-1]):  # Skip the current message
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            if role and content:
                if role == "assistant":
                    recent_messages.append(f"AI: {content}")
                elif role == "user":
                    recent_messages.append(f"User: {content}")
                
                message_count += 1
                if message_count >= max_messages:
                    break
        
        # Add messages in chronological order
        if recent_messages:
            conversation_context = "\n".join(reversed(recent_messages)) + "\n" + conversation_context
    

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
        
        yield response
        state.setdefault("messages", []).append({"role": "assistant", "content": response.get("message", "")})
        state["prompt_str"] = response.get("message", "")
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
        knowledge_data = await fetch_knowledge(query, graph_version_id)
        
        # Initialize result_data as empty dict to ensure it's always defined
        result_data = {}
        
        # Handle different return types from fetch_knowledge
        if isinstance(knowledge_data, dict) and "status" in knowledge_data and knowledge_data["status"] == "error":
            # If fetch_knowledge returned an error dictionary
            return knowledge_data
        
        # Handle string or other return types
        if isinstance(knowledge_data, str):
            # Check if the knowledge_data contains vector representations
            if knowledge_data and ("[" in knowledge_data and "]" in knowledge_data):
                # Look for specific patterns indicating vector data
                if any(pattern in knowledge_data for pattern in ["-0.", "0.", "1.", "2.", "..."]):
                    # Filter out the vector data sections
                    lines = knowledge_data.split("\n")
                    filtered_lines = []
                    skip_until_next_entry = False
                    
                    for line in lines:
                        # If line contains vector data pattern, skip it
                        if ("[" in line and "]" in line and 
                            any(pattern in line for pattern in ["-0.", "0.", "1.", "2.", "..."])):
                            skip_until_next_entry = True
                            filtered_lines.append("Content unavailable (vector data)")
                        # Reset skip flag when reaching entry separator
                        elif "----" in line:
                            skip_until_next_entry = False
                            filtered_lines.append(line)
                        # Include line if not in skip mode
                        elif not skip_until_next_entry:
                            filtered_lines.append(line)
                    
                    # Replace knowledge data with filtered version
                    knowledge_data = "\n".join(filtered_lines)
                    logger.info("Filtered out vector data from knowledge response")
            
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
