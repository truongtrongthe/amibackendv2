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
            
            if "metadata" in response and "response_strategy" in response["metadata"]:
                logger.info(f"Active learning mode used: {response['metadata']['response_strategy']}")
            
            # Step 6: Save knowledge for relevant responses
            if response.get("status") == "success" and response["metadata"]["similarity_score"] >= 0.35:
                topic = response["metadata"].get("core_prior_topic", "unknown")
                categories = ["health_segmentation"] if any(term in message.lower() or topic.lower() for term in ["rối loạn cương dương", "xuất tinh sớm", "phân nhóm khách hàng"]) else ["general"]
                message_content = response["message"]
                conversational_response = re.split(r'<knowledge_queries>', message_content)[0].strip()
                query_section = re.search(r'<knowledge_queries>(.*?)</knowledge_queries>', message_content, re.DOTALL)
                knowledge_queries = json.loads(query_section.group(1).strip()) if query_section else []

                await save_knowledge(
                    input=message,
                    user_id=user_id,
                    bank_name="health" if "health_segmentation" in categories else "default",
                    thread_id=thread_id,
                    topic=topic,
                    categories=categories
                )
                await save_knowledge(
                    input=conversational_response,
                    user_id=user_id,
                    bank_name="health" if "health_segmentation" in categories else "default",
                    thread_id=thread_id,
                    topic=topic,
                    categories=categories + ["response"]
                )
                for query in knowledge_queries:
                    await save_knowledge(
                        input=query,
                        user_id=user_id,
                        bank_name="health" if "health_segmentation" in categories else "default",
                        thread_id=thread_id,
                        topic=topic,
                        categories=categories + ["query"]
                    )
                logger.info(f"Saved knowledge: user input, response, and {len(knowledge_queries)} queries for topic '{topic}'")

            return response
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            import traceback
            logger.error(f"Stack trace: {traceback.format_exc()}")
            return {"status": "error", "message": f"Error: {str(e)}"}
   
    async def _search_knowledge(self, message: str, conversation_context: str = "") -> Dict[str, Any]:
        logger.info(f"Searching for analysis knowledge based on message: {message[:100]}...")
        try:
            queries = []
            primary_query = message.strip()
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
            is_confirmation = any(keyword.lower() in message.lower() for keyword in confirmation_keywords)
            is_follow_up = is_confirmation or re.search(r'\b(nhóm này|this group|vậy thì sao)\b', message.lower(), re.IGNORECASE) or message.lower().strip() in prior_topic.lower()
            
            if is_follow_up and prior_topic:
                queries.append(prior_topic)
                logger.info(f"Follow-up detected, reusing prior topic: {prior_topic[:50]}")
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
                    logger.info(f"Enriched follow-up query: {primary_query[:50]}")
            else:
                queries.append(primary_query)
                if prior_topic and prior_topic != primary_query:
                    queries.append(prior_topic)
                    logger.info(f"Added prior topic to queries: {prior_topic[:50]}")

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
                        logger.warning("Failed to parse LLM queries, retrying once")
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

            queries = list(dict.fromkeys(queries))
            queries = [q for q in queries if len(q.strip()) > 5]
            if not queries:
                logger.warning("No valid queries found")
                return {
                    "knowledge_context": prior_knowledge if is_follow_up else "",
                    "similarity": 0.7 if is_follow_up else 0.0,
                    "query_count": 0,
                    "prior_data": {"topic": prior_topic, "knowledge": prior_knowledge},
                    "metadata": {"similarity": 0.7 if is_follow_up else 0.0}
                }

            best_knowledge = prior_knowledge if is_follow_up else ""
            best_similarity = 0.7 if is_follow_up else 0.0
            query_count = 0

            categories = ["health_segmentation"] if any(term in message.lower() or prior_topic.lower() for term in ["rối loạn cương dương", "xuất tinh sớm", "phân nhóm khách hàng"]) else ["general"]
            bank_name = "health" if "health_segmentation" in categories else "default"
            
            for query in queries:
                results = await query_knowledge(
                    query=query,
                    bank_name=bank_name,
                    user_id=user_id,
                    thread_id=thread_id,
                    topic=prior_topic or "unknown",
                    top_k=5,
                    min_similarity=0.3
                )
                query_count += 1
                if not results:
                    continue
                top_result = results[0]
                similarity = top_result["score"]
                knowledge_content = top_result["raw"]
                logger.info(f"Query '{query[:30]}...' yielded similarity: {similarity}")
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_knowledge = knowledge_content
                    logger.info(f"Updated best knowledge with similarity: {best_similarity}")

            vibe_score = 1.0
            if any(term in primary_query.lower() or (prior_topic and term in prior_topic.lower()) 
                   for term in ["mục tiêu", "goals", "active learning", "phân nhóm"]):
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
                "knowledge_context": prior_knowledge if is_follow_up else "",
                "similarity": 0.7 if is_follow_up else 0.0,
                "query_count": 0,
                "prior_data": {"topic": prior_topic, "knowledge": prior_knowledge},
                "metadata": {"similarity": 0.7 if is_follow_up else 0.0}
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
        
        confirmation_keywords = ["có", "yes", "correct", "right", "explore further", "đúng rồi", "nhóm này"]
        is_confirmation = any(keyword.lower() in message.lower() for keyword in confirmation_keywords)
        is_follow_up = is_confirmation or re.search(r'\b(nhóm này|this group|vậy thì sao)\b', message.lower(), re.IGNORECASE) or message.lower().strip() in prior_topic.lower()
        
        core_prior_topic = prior_topic
        if prior_topic:
            match = re.search(r'(phân nhóm khách hàng|customer segmentation|phân loại người dùng)', prior_topic, re.IGNORECASE)
            if match:
                core_prior_topic = match.group(1)
            elif any(term in prior_topic.lower() for term in ["phân nhóm", "segmentation", "nhóm khách hàng"]):
                core_prior_topic = "phân nhóm khách hàng"
            logger.info(f"Core prior topic extracted: {core_prior_topic}")

        if is_follow_up:
            response_strategy = "FOLLOW_UP"
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
            strategy_instructions = (
                "Present all relevant knowledge from EXISTING KNOWLEDGE in a comprehensive format, structuring the response to cover key aspects (e.g., purpose, methods, outcomes, context). "
                "If the message likely continues PRIOR TOPIC (e.g., contains keywords like 'phân nhóm' or 'khách hàng'), prioritize deepening that topic with specific details and actionable insights. "
                "If the topic is ambiguous, rephrase it (e.g., 'It sounds like you’re asking about…'). "
                "Ask a targeted question to confirm the topic or validate the summary (e.g., 'Đây có phải ý bạn muốn hỏi không?'). "
                f"If similarity is 0.35–0.55, emphasize clarification. "
                f"If above 0.55, focus on mastery and closure."
            )
        
        prompt = f"""You are Ami, a conversational AI that understands topics deeply and drives discussions toward closure. You have access to tools for querying and saving knowledge.

            **Input**:
            - CURRENT MESSAGE: {message}
            - CONVERSATION HISTORY: {conversation_context}
            - TIME: {temporal_context}
            - EXISTING KNOWLEDGE: {knowledge_context}
            - RESPONSE STRATEGY: {response_strategy}
            - STRATEGY INSTRUCTIONS: {strategy_instructions}
            - PRIOR TOPIC: {core_prior_topic}
            - PRIOR KNOWLEDGE: {prior_knowledge}
            - USER ID: {user_id}

            **Tools Available**:
            - knowledge_query: Query the knowledge base.
              Parameters: query (str, required), context (str), user_id (str, required), thread_id (str), topic (str), top_k (int, default 5), min_similarity (float, default 0.3)
            - save_knowledge: Save knowledge to the database.
              Parameters: query (str), content (str), user_id (str, required), thread_id (str), topic (str), categories (list of str, default ["general"])

            **Instructions**:
            1. **Topic Detection**: Identify the core topic from CONVERSATION HISTORY and CURRENT MESSAGE. For follow-ups or confirmations, anchor to PRIOR TOPIC and PRIOR KNOWLEDGE. Rephrase ambiguous terms (e.g., 'It sounds like you’re asking about…').
            2. **Intent Analysis**: Classify intent (questioning, confirming, follow-up, etc.). For follow-ups, expand PRIOR TOPIC with actionable insights based on PRIOR KNOWLEDGE’s specific details.
            3. **Tool Usage**:
               - Use knowledge_query if additional information is needed (e.g., for LOW_SIMILARITY or complex queries).
               - Use save_knowledge to store important insights (e.g., after confirmations or high-relevance responses).
               - Include tool calls in the output if needed.
            4. **Knowledge Queries**: Generate 3 specific queries: core topic, specific aspect, related concept. Ensure queries are JSON-parsable and relevant to PRIOR TOPIC.
            5. **Response**: Follow RESPONSE STRATEGY with a curious tone. For follow-ups, deepen discussion with practical details and ask a targeted follow-up. Keep responses concise (100–150 words).
            6. **Output**:
                - Conversational Response
                - <knowledge_queries>["query1", "query2", "query3"]</knowledge_queries>
                - <tool_calls>[{{"name": "tool_name", "parameters": {{...}}}}]</tool_calls> (if tools are needed, JSON-parsable)

            **Constraints**:
            - Use user’s language (Vietnamese if applicable).
            - Ensure topic continuity for follow-ups and confirmations.
            - Ensure knowledge queries and tool calls are JSON-parsable.
            - Include user_id in all tool calls.
            """
        try:
            response = await LLM.ainvoke(prompt)
            logger.info(f"LLM response generated with similarity score: {similarity_score}")
            
            content = response.content.strip()
            tool_calls = []
            if "<tool_calls>" in content:
                tool_section = re.search(r'<tool_calls>(.*?)</tool_calls>', content, re.DOTALL)
                if tool_section:
                    try:
                        tool_calls = json.loads(tool_section.group(1).strip())
                        content = re.sub(r'<tool_calls>.*?</tool_calls>', '', content, flags=re.DOTALL).strip()
                        logger.info(f"Extracted {len(tool_calls)} tool calls")
                    except json.JSONDecodeError:
                        logger.warning("Failed to parse tool calls")

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
                    "tool_calls": tool_calls
                }
            }
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {
                "status": "error",
                "message": "Tôi xin lỗi, nhưng tôi gặp lỗi khi xử lý yêu cầu của bạn. Vui lòng thử lại."
            }

async def process_llm_with_tools(
        self,
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
        yield {"state": state}

async def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool with enhanced parameter validation and metadata enrichment."""
        logger.info(f"Executing tool: {tool_name} with parameters: {parameters}")
        try:
            user_id = parameters.get("user_id", "")
            if not user_id:
                return {"status": "error", "message": "Missing required parameter: user_id"}

            if tool_name == "knowledge_query":
                query = parameters.get("query", "")
                if not query:
                    return {"status": "error", "message": "Missing required parameter: query"}
                context = parameters.get("context", "")
                thread_id = parameters.get("thread_id", None)
                topic = parameters.get("topic", None)
                
                categories = ["health_segmentation"] if any(term in query.lower() or term in context.lower() for term in ["rối loạn cương dương", "xuất tinh sớm", "phân nhóm khách hàng"]) else ["general"]
                bank_name = "health" if "health_segmentation" in categories else "default"

                results = await query_knowledge(
                    query=query,
                    bank_name=bank_name,
                    user_id=user_id,
                    thread_id=thread_id,
                    topic=topic,
                    top_k=parameters.get("top_k", 5),
                    min_similarity=parameters.get("min_similarity", 0.3)
                )
                return {
                    "status": "success",
                    "message": f"Queried knowledge for '{query}'",
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
                    categories = ["health_segmentation"] if any(term in input_text.lower() for term in ["rối loạn cương dương", "xuất tinh sớm", "phân nhóm khách hàng"]) else ["general"]
                bank_name = "health" if "health_segmentation" in categories else "default"

                success = await save_knowledge(
                    input=input_text,
                    user_id=user_id,
                    bank_name=bank_name,
                    thread_id=thread_id,
                    topic=topic,
                    categories=categories
                )
                return {
                    "status": "success" if success else "error",
                    "message": f"{'Saved' if success else 'Failed to save'} knowledge for '{input_text[:50]}...'"
                }

            return {"status": "error", "message": f"Unknown tool: {tool_name}"}
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {str(e)}")
            return {"status": "error", "message": f"Tool execution failed: {str(e)}"}