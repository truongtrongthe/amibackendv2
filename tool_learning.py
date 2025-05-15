import json
import asyncio
import logging
from typing import Dict, List, Any, Optional, Union, AsyncGenerator
from datetime import datetime
from uuid import uuid4
import re
import pytz

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from ai_tools import fetch_knowledge_with_similarity
from pccontroller import save_knowledge
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

    async def initialize(self):
        """Initialize the processor asynchronously."""
        logger.info("Initializing LearningProcessor")
        return self

    async def process_incoming_message(self, message: str, conversation_context: str, user_id: str, thread_id: Optional[str] = None) -> Dict[str, Any]:
        """Process incoming message with active learning flow."""
        logger.info(f"Processing message from user {user_id}")
        try:
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
                            suggested_queries = []

            # Step 2: Search for relevant knowledge
            logger.info(f"Searching for knowledge based on message...")
            analysis_knowledge = await self._search_knowledge(message, conversation_context)
            
            # Step 3: Enrich with suggested queries
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
                    if query_similarity > best_similarity and query_content:
                        best_similarity = query_similarity
                        best_knowledge = query_content
                        logger.info(f"Found better knowledge with query '{query}', similarity: {best_similarity}")
                    if query_similarity >= 0.35 and query_content and query_content not in best_knowledge:
                        best_knowledge += f"\n\nAdditional information from query '{query}':\n{query_content}"
                        logger.info(f"Added supplementary knowledge from query '{query}'")
                
                if best_similarity > primary_similarity:
                    analysis_knowledge["knowledge_context"] = best_knowledge
                    analysis_knowledge["similarity"] = best_similarity
                    analysis_knowledge["metadata"]["similarity"] = best_similarity
                    logger.info(f"Updated knowledge from suggested queries. New similarity: {best_similarity}")
            
            # Step 4: Log similarity score
            similarity = analysis_knowledge.get("similarity", 0.0)
            logger.info(f"💯 Found knowledge with similarity score: {similarity}")
            
            # Step 5: Generate response
            logger.info(f"Generating response based on knowledge...")
            prior_data = analysis_knowledge.get("prior_data", {})
            response = await self._active_learning(message, conversation_context, analysis_knowledge, user_id, prior_data)
            logger.info(f"Response generated with status: {response.get('status', 'unknown')}")
            
            if "metadata" in response and "active_learning_mode" in response["metadata"]:
                logger.info(f"Active learning mode used: {response['metadata']['active_learning_mode']}")
                
            return response
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            import traceback
            logger.error(f"Stack trace: {traceback.format_exc()}")
            
   
    async def _search_knowledge(self, message: str, conversation_context: str = "") -> Dict[str, Any]:
        logger.info(f"Searching for analysis knowledge based on message: {message[:100]}...")
        try:
            queries = []
            primary_query = message.strip()
            prior_topic = ""
            prior_knowledge = ""

            # Extract prior topic and knowledge
            if conversation_context:
                user_messages = re.findall(r'User: (.*?)(?:\n\n|$)', conversation_context, re.DOTALL)
                ai_messages = re.findall(r'AI: (.*?)(?:\n\n|$)', conversation_context, re.DOTALL)
                logger.info(f"Found {len(user_messages)} user messages in context")
                if user_messages:
                    prior_topic = user_messages[-2].strip() if len(user_messages) > 1 else user_messages[0].strip()
                    logger.info(f"Extracted prior topic: {prior_topic[:50]}")
                if ai_messages:
                    prior_knowledge = ai_messages[-1].strip()

            # Handle short or confirmation messages
            confirmation_keywords = ["có", "yes", "correct", "right", "explore further"]
            is_confirmation = any(keyword.lower() in message.lower() for keyword in confirmation_keywords) and len(primary_query) <= 20
            if is_confirmation and prior_topic:
                queries.append(prior_topic)
                logger.info(f"Confirmation detected, reusing prior topic: {prior_topic[:50]}")
                if prior_knowledge:
                    query_section = re.search(r'<knowledge_queries>(.*?)</knowledge_queries>', prior_knowledge, re.DOTALL)
                    if query_section:
                        try:
                            prior_queries = json.loads(query_section.group(1).strip())
                            queries.extend(prior_queries)
                            logger.info(f"Reusing {len(prior_queries)} prior AI queries")
                        except json.JSONDecodeError:
                            logger.warning("Failed to parse prior AI queries")
                primary_query = f"{primary_query} about {prior_topic}"
                queries.append(primary_query)
                logger.info(f"Enriched confirmation query: {primary_query[:50]}")
            else:
                queries.append(primary_query)
                if prior_topic and prior_topic != primary_query:
                    queries.append(prior_topic)
                    logger.info(f"Added prior topic to queries: {prior_topic[:50]}")

            # Add LLM-generated queries
            temp_response = await self._active_learning(message, conversation_context, {}, "unknown", {})

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
                        logger.warning("Failed to parse LLM queries")
                        temp_response = await self._active_learning(message, conversation_context, {}, "unknown", {})
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

            # Remove duplicates and low-value queries
            queries = list(dict.fromkeys(queries))
            queries = [q for q in queries if len(q.strip()) > 5]
            if not queries:
                logger.warning("No valid queries found")
                return {
                    "knowledge_context": prior_knowledge if is_confirmation else "",
                    "similarity": 0.7 if is_confirmation else 0.0,
                    "query_count": 0,
                    "prior_data": {"topic": prior_topic, "knowledge": prior_knowledge},
                    "metadata": {"similarity": 0.7 if is_confirmation else 0.0}
                }

            best_knowledge = prior_knowledge if is_confirmation else ""
            best_similarity = 0.7 if is_confirmation else 0.0
            query_count = 0
            vector_count = 4  # Placeholder; ideally fetch actual size

            # Dynamic threshold
            base_threshold = 0.35 if vector_count <= 10 else 0.4
            logger.info(f"Using similarity threshold: {base_threshold} for {vector_count} vectors")

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
                        weighted_similarity = similarity if similarity >= base_threshold else similarity * 0.8
                        logger.info(f"Query '{query[:30]}...' yielded similarity: {similarity}, weighted: {weighted_similarity}")
                        if weighted_similarity > best_similarity:
                            best_similarity = weighted_similarity
                            best_knowledge = knowledge_content
                    except ValueError:
                        logger.warning(f"Invalid similarity score: {sim_match.group(1)}")

            # Vibe score boost for frequent topics
            vibe_score = 1.0
            if any(term in primary_query.lower() or (prior_topic and term in prior_topic.lower()) 
                   for term in ["mục tiêu", "goals", "active learning"]):
                vibe_score = 1.1
                best_similarity *= vibe_score
                logger.info(f"Applied vibe score {vibe_score} for frequent topic")

            logger.info(f"Final similarity: {best_similarity} from {query_count} queries")
            return {
                "knowledge_context": best_knowledge,
                "similarity": best_similarity,
                "query_count": query_count,
                "prior_data": {"topic": prior_topic, "knowledge": prior_knowledge},
                "metadata": {"similarity": best_similarity, "vibe_score": vibe_score}
            }
        except Exception as e:
            logger.error(f"Error fetching knowledge: {str(e)}")
            return {
                "knowledge_context": prior_knowledge if is_confirmation else "",
                "similarity": 0.7 if is_confirmation else 0.0,
                "query_count": 0,
                "prior_data": {"topic": prior_topic, "knowledge": prior_knowledge},
                "metadata": {"similarity": 0.7 if is_confirmation else 0.0}
            }

    async def _active_learning(self, message: str, conversation_context: str = "", analysis_knowledge: Dict = None, user_id: str = "unknown", prior_data: Dict = None) -> Dict[str, Any]:
        logger.info("Answering user question with active learning approach")
        
        vietnam_tz = pytz.timezone('Asia/Ho_Chi_Minh')
        current_time = datetime.now(vietnam_tz)
        date_str = current_time.strftime("%A, %B %d, %Y")
        time_str = current_time.strftime("%H:%M")
        temporal_context = f"Current date and time: {date_str} at {time_str} (Asia/Ho_Chi_Minh timezone)."
        
        knowledge_context = analysis_knowledge.get("knowledge_context", "") if analysis_knowledge else ""
        similarity_score = float(analysis_knowledge.get("similarity", 0.0)) if analysis_knowledge else 0.0
        logger.info(f"Using similarity score: {similarity_score}")
        
        prior_topic = prior_data.get("topic", "") if prior_data else ""
        prior_knowledge = prior_data.get("knowledge", "") if prior_data else ""
        
        # Expanded confirmation keywords
        confirmation_keywords = ["có", "yes", "correct", "right", "explore further", "đúng rồi", "nhóm này"]
        is_confirmation = any(keyword.lower() in message.lower() for keyword in confirmation_keywords)
        # Detect follow-ups or confirmations
        is_follow_up = is_confirmation or re.search(r'\b(nhóm này|this group|vậy thì sao)\b', message.lower(), re.IGNORECASE) or message.lower().strip() in prior_topic.lower()
        
        # Extract core topic for better continuity
        core_prior_topic = prior_topic
        if prior_topic:
            # Extract specific concept (e.g., "phân nhóm khách hàng")
            match = re.search(r'(phân nhóm khách hàng|customer segmentation|phân loại người dùng)', prior_topic, re.IGNORECASE)
            if match:
                core_prior_topic = match.group(1)
            elif any(term in prior_topic.lower() for term in ["phân nhóm", "segmentation", "nhóm khách hàng"]):
                core_prior_topic = "phân nhóm khách hàng"
            logger.info(f"Core prior topic extracted: {core_prior_topic}")

        if is_follow_up:
            response_strategy = "FOLLOW_UP"
            # Enhanced: Deepen prior topic with specific details and actionable follow-ups
            strategy_instructions = (
                "Recognize the message as a follow-up or confirmation of PRIOR TOPIC, referring to a specific concept or group from PRIOR KNOWLEDGE (e.g., customer segmentation methods). "
                "Use PRIOR KNOWLEDGE to deepen the discussion, leveraging specific details (e.g., named groups like 'Nhóm Rối Loạn Cương Dương') and offering actionable insights (e.g., implementation methods, support strategies). "
                "Structure the response with key aspects (e.g., purpose, methods, outcomes). "
                "If PRIOR TOPIC is ambiguous, rephrase it (e.g., 'It sounds like you’re confirming customer segmentation…'). "
                "Ask a targeted follow-up to advance the discussion (e.g., 'Which group do you want to focus on, like Nhóm Rối Loạn Cương Dương?')."
            )
            knowledge_context = prior_knowledge
            similarity_score = max(similarity_score, 0.7)
        elif similarity_score < 0.35:
            response_strategy = "LOW_SIMILARITY"
            strategy_instructions = (
                "State: 'Tôi không thể tìm thấy thông tin liên quan; vui lòng giải thích thêm.' "
                "Ask for more details about the topic. "
                "Propose a specific question (e.g., 'Bạn có thể chia sẻ thêm về ý nghĩa của điều này không?')."
            )
        else:
            response_strategy = "RELEVANT_KNOWLEDGE"
            # Enhanced: Prioritize continuity for related topics
            strategy_instructions = (
                "Present all relevant knowledge from EXISTING KNOWLEDGE in a comprehensive format, structuring the response to cover key aspects (e.g., purpose, methods, outcomes, context). "
                "If the message likely continues PRIOR TOPIC (e.g., contains keywords like 'phân nhóm' or 'khách hàng'), prioritize deepening that topic with specific details and actionable insights. "
                "If the topic is ambiguous, rephrase it (e.g., 'It sounds like you’re asking about…'). "
                "Ask a targeted question to confirm the topic or validate the summary (e.g., 'Đây có phải ý bạn muốn hỏi không?'). "
                f"If similarity is 0.35–0.55, emphasize clarification. "
                f"If above 0.55, focus on mastery and closure."
            )
        
        prompt = f"""You are Ami, a conversational AI that understands topics deeply and drives discussions toward closure.

            **Input**:
            - CURRENT MESSAGE: {message}
            - CONVERSATION HISTORY: {conversation_context}
            - TIME: {temporal_context}
            - EXISTING KNOWLEDGE: {knowledge_context}
            - RESPONSE STRATEGY: {response_strategy}
            - STRATEGY INSTRUCTIONS: {strategy_instructions}
            - PRIOR TOPIC: {core_prior_topic}
            - PRIOR KNOWLEDGE: {prior_knowledge}

            **Instructions**:
            1. **Topic Detection**: Identify the core topic from CONVERSATION HISTORY and CURRENT MESSAGE. For follow-ups or confirmations, anchor to PRIOR TOPIC and PRIOR KNOWLEDGE. Rephrase ambiguous terms (e.g., 'It sounds like you’re asking about…').
            2. **Intent Analysis**: Classify intent (questioning, confirming, follow-up, etc.). For follow-ups, expand PRIOR TOPIC with actionable insights based on PRIOR KNOWLEDGE’s specific details.
            3. **Knowledge Queries**: Generate 3 specific queries: core topic, specific aspect, related concept. Ensure queries are JSON-parsable and relevant to PRIOR TOPIC.
            4. **Response**: Follow RESPONSE STRATEGY with a curious tone. For follow-ups, deepen discussion with practical details and ask a targeted follow-up. Keep responses concise (100–150 words).
            5. **Output**:
                - Conversational Response
                - <knowledge_queries>["query1", "query2", "query3"]</knowledge_queries>

            **Constraints**:
            - Use user’s language (Vietnamese if applicable).
            - Ensure topic continuity for follow-ups and confirmations.
            - Ensure knowledge queries are specific and JSON-parsable.
            """
        try:
            response = await LLM.ainvoke(prompt)
            logger.info(f"LLM response generated with similarity score: {similarity_score}")
            
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
    conversation_context = ""
    if conversation_history:
        recent_messages = []
        message_count = 0
        max_messages = 50
        
        for msg in reversed(conversation_history):
            try:
                role = msg.get("role", "").lower()
                content = msg.get("content", "")
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
        else:
            logger.warning("No usable messages found in conversation history")
    
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
            state.get('user_id', 'unknown'),
            thread_id
        )
        
        message_content = response["message"] if isinstance(response, dict) and "message" in response else str(response)
        # Strip <knowledge_queries> for frontend
        message_content = re.split(r'<knowledge_queries>', message_content)[0].strip()
        logger.info("Stripped knowledge_queries from message for frontend")
        
        yield {"status": "success", "message": message_content}
        state.setdefault("messages", []).append({"role": "assistant", "content": message_content})
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
                parameters.get("graph_version_id", ""),
                parameters.get("user_id", ""),
                parameters.get("thread_id", ""),
                parameters.get("topic", ""),
                parameters.get("categories", [])
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
        if isinstance(knowledge_data, dict) and "status" in knowledge_data and knowledge_data["status"] == "error":
            return knowledge_data
        return {
            "status": "success",
            "message": f"Knowledge queried for {query}",
            "data": knowledge_data if isinstance(knowledge_data, dict) else {"content": str(knowledge_data)}
        }
    except Exception as e:
        logger.error(f"Error fetching knowledge: {str(e)}")
        return {"status": "error", "message": f"Failed to fetch knowledge: {str(e)}"}

async def save_new_knowledge(query: str, knowledge_content: str, graph_version_id: str, user_id: str, thread_id: str, topic: str, categories: List[str]) -> Dict[str, Any]:
    """Save new knowledge to the knowledge base."""
    logger.info(f"Saving new knowledge for query: {query[:100]}...")
    try:
        save_result = await save_knowledge(
            query=query,
            content=knowledge_content,
            graph_version_id=graph_version_id
        )
        await save_knowledge(
            input=query,
            user_id="learner",
            bank_name="health",
            thread_id="learning_thread_1747282472163",
            topic="phân nhóm khách hàng",
            categories=["health_segmentation"]
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