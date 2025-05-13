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

from aitools import fetch_knowledge, brain, emit_analysis_event  # Assuming these are available
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
            
            
            logger.info(f"I am searching knowledge follow the queries ...")
            analysis_knowledge = await self._search_knowledge(message)
            logger.info(f"I found analysis knowledge: {analysis_knowledge}")
            response = await self._active_learning(message, conversation_context)
            logger.info(f"Now I have response: {response}")
            return response
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
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
        """Search for analysis methods based on user profile's analysis queries."""
        
        # Fetch knowledge for each analysis query
        knowledge_entries = []
        logger.info(f"Searching for analysis knowledge: {user_profile.analysis_queries}")
        for query in user_profile.analysis_queries:
            try:
                # Use the query tool to fetch specific knowledge
                knowledge = await fetch_knowledge(query, self.graph_version_id)
                if knowledge:
                    # Handle different types of knowledge data
                    if isinstance(knowledge, dict) and "status" in knowledge and knowledge["status"] == "error":
                        logger.warning(f"Error in knowledge for query '{query}': {knowledge.get('message', 'Unknown error')}")
                        continue
                    
                    # Process the knowledge content based on its type
                    knowledge_content = ""
                    if isinstance(knowledge, str):
                        knowledge_content = knowledge
                    elif isinstance(knowledge, dict):
                        if "content" in knowledge:
                            knowledge_content = knowledge["content"]
                        else:
                            # Serialize the dictionary as a fallback
                            knowledge_content = json.dumps(knowledge)
                    else:
                        # Convert to string as a last resort
                        knowledge_content = str(knowledge)
                    
                    knowledge_entries.append({
                        "query": query,
                        "knowledge": knowledge_content
                    })
            except Exception as e:
                logger.error(f"Error fetching knowledge for query '{query}': {str(e)}")

        # Combine all knowledge entries
        combined_knowledge = "\n\n".join([
            f"Query: {entry['query']}\nKnowledge: {entry['knowledge']}"
            for entry in knowledge_entries
        ])

        return {
            "analysis_methods": [entry["query"] for entry in knowledge_entries],
            "knowledge_context": combined_knowledge,
            "query_count": len(knowledge_entries),
            "rationale": f"Analyzed {len(knowledge_entries)} queries for user profile"
        }

    async def _active_learning(self, message: str, conversation_context: str = "") -> Dict[str, Any]:
        """Generate user-friendly response based on the action plan."""
        logger.info("Answering user question")
        

        # Get the action plan details
        action_plan = user_profile.action_plan
        analysis_summary = action_plan.get("analysis_summary", {})
        important_notes = action_plan.get("important_notes", {})
        next_actions = action_plan.get("next_actions", [])
        knowledge_queries = action_plan.get("knowledge_queries", [])
        
        # Extract resolved temporal references if available
        temporal_references = {}
        if "temporal_references" in user_profile.other_aspects:
            temporal_references = user_profile.other_aspects.get("temporal_references", {})
            logger.info(f"Including temporal references in response generation: {json.dumps(temporal_references, indent=2)}")
        
        # Fetch additional knowledge if queries are provided
        additional_knowledge = ""
        if knowledge_queries:
            logger.info(f"Fetching additional knowledge for {len(knowledge_queries)} queries")
            try:
                knowledge_results = []
                for query_info in knowledge_queries[:3]:  # Limit to top 3 queries
                    try:
                        # Extract just the query string from the query info dictionary
                        query = query_info.get('query', '') if isinstance(query_info, dict) else query_info
                        if query:
                            knowledge = await fetch_knowledge(query, self.graph_version_id)
                            if knowledge:
                                knowledge_results.append(f"Query: {query}\nResult: {knowledge}")
                    except Exception as e:
                        logger.error(f"Error fetching knowledge for query '{query}': {str(e)}")
                
                additional_knowledge = "\n\n".join(knowledge_results)
                logger.info(f"Fetched additional knowledge: {len(additional_knowledge)} chars")
            except Exception as e:
                logger.error(f"Error fetching additional knowledge: {str(e)}")

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
        
        # Format temporal references for inclusion in the prompt if available
        temporal_info = ""
        if temporal_references:
            original_mentions = temporal_references.get("original_mentions", [])
            resolved_dates = temporal_references.get("resolved_dates", [])
            resolved_times = temporal_references.get("resolved_times", [])
            
            temporal_info = "TEMPORAL REFERENCES:\n"
            if original_mentions and resolved_dates:
                for i, mention in enumerate(original_mentions):
                    resolved_date = resolved_dates[i] if i < len(resolved_dates) else "unknown"
                    resolved_time = resolved_times[i] if resolved_times and i < len(resolved_times) else ""
                    
                    if resolved_time:
                        temporal_info += f"When user mentioned '{mention}', they meant: {resolved_date} at {resolved_time}\n"
                    else:
                        temporal_info += f"When user mentioned '{mention}', they meant: {resolved_date}\n"
        
        # Build prompt for LLM to generate response
        prompt = f"""Generate a concise, culturally appropriate response to this user based on:

        CURRENT MESSAGE: {message}
        CONVERSATION: {conversation_context}
        TIME CONTEXT: {temporal_context}
        {temporal_info if temporal_info else ""}
        ANALYSIS: {json.dumps(analysis_summary, indent=2) if analysis_summary else "N/A"}
        PLAN: {json.dumps(next_actions, indent=2) if next_actions else "N/A"}
        CONTEXT: Class={user_profile.classification}, Skills={', '.join(user_profile.skills)}, Needs={', '.join(user_profile.requirements)}
        INFO: {additional_knowledge}

        EXECUTION STRATEGY:
        1. STRICT ACTION SEQUENCE - Execute actions in EXACT order as specified in the PLAN
        2. NO SKIPPING - Do not skip any actions in the sequence
        3. NO ADDITIONS - Do not add actions that are not in the PLAN
        4. USE AVAILABLE INFORMATION - Prioritize factual information from knowledge sources over assumptions
        5. APPLY DOMAIN EXPERTISE - For any next_action, include specific details identified from the available sources
        6. CONNECT DOTS - Link insights from different sources to provide comprehensive execution of the plan
        7. INCORPORATE TIME CONTEXT - Use the resolved temporal references when discussing dates and times

        REQUIREMENTS:
        1. BE CONCISE - Keep response under 100 words
        2. APPLY CULTURAL INTELLIGENCE - Use culturally appropriate forms of address and relationship terms that reflect the user's language context
        3. MATCH LANGUAGE SOPHISTICATION - Sound like a domain expert in their language
        4. MAINTAIN TONE - Friendly but professional
        5. PERSONALIZE - Address specific user needs without repeating their question
        6. INCORPORATE KNOWLEDGE - Use additional info where relevant
        7. PRIORITIZE CLARITY - Focus on next steps and solutions
        8. MAINTAIN CONVERSATION FLOW - Reference prior exchanges when relevant
        9. BE NATURAL - Write as a human expert would, not as an AI assistant
        10. SKIP FORMULAIC GREETINGS - Avoid repetitive hello/greeting phrases and go straight to helpful content
        11. BE TIME-SPECIFIC - When mentioning dates and times, be specific (e.g., "Sunday, May 12" instead of just "Sunday")

        ACTION EXECUTION RULES:
        1. Execute each action in the PLAN in sequence


        LANGUAGE ADAPTATION: Adapt your response style to match cultural norms of the user's language. Consider formality levels, kinship terms, collectivist vs individualist expressions, and domain-specific terminology. Avoid literal translations of expressions or generic greetings that sound unnatural to native speakers. In continuous exchanges, don't start each message with a greeting.
        """

        try:
            response = await LLM.ainvoke(prompt)
            return {
                "status": "success",
                "message": response.content.strip(),
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "user_id": user_profile.user_id,
                    "knowledge_queries_used": knowledge_queries,
                    "additional_knowledge_found": bool(additional_knowledge),
                    "temporal_references": temporal_references if temporal_references else None
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
        return {"status": "error", "message": f"Unknown tool: {tool_name}"}
    except Exception as e:
        logger.error(f"Error executing tool {tool_name}: {str(e)}")
        return {"status": "error", "message": str(e)}

async def knowledge_query_helper(query: str, context: str, graph_version_id: str) -> Dict[str, Any]:
    """Query knowledge base."""
    logger.info(f"Querying knowledge: {query}")
    try:
        knowledge_data = await fetch_knowledge(query, graph_version_id)
        
        # Handle different return types from fetch_knowledge
        if isinstance(knowledge_data, dict) and "status" in knowledge_data and knowledge_data["status"] == "error":
            # If fetch_knowledge returned an error dictionary
            return knowledge_data
        
        # Handle string or other return types
        result_data = {}
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
