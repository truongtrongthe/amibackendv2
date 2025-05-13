import json
import asyncio
import logging
import logging.handlers
import re
from queue import Queue
from typing import Dict, List, Any, Optional, Union, AsyncGenerator
from datetime import datetime
from uuid import uuid4
from cachetools import TTLCache
import pytz
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from asyncio import Semaphore

# Assuming these are available
from aitools import fetch_knowledge, brain, emit_analysis_event
from brain_singleton import get_current_graph_version
from utilities import logger
from tool_helpers import ensure_brain_loaded
from brain_singleton import get_brain_sync, get_brain

# Setup asynchronous logging
logger_queue = logging.handlers.QueueHandler(Queue())
logger = logging.getLogger(__name__)
logger.addHandler(logger_queue)
logger.setLevel(logging.INFO)

# Compile regex patterns
NEWLINE_PATTERN = re.compile(r'\n{3,}')
SPACE_PATTERN = re.compile(r' +')

# Custom JSON encoder
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

def safe_parse_json(data: str) -> Union[Dict, str]:
    """Safely parse JSON, returning original string on failure."""
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        logger.warning(f"Invalid JSON: {data[:50]}...")
        return data

def serialize_model(model: BaseModel) -> Dict[str, Any]:
    """Serialize Pydantic model to JSON-serializable dict."""
    return json.loads(json.dumps(model.model_dump(), cls=DateTimeEncoder))

# Centralized temporal context
def get_temporal_context(timezone: str = 'Asia/Ho_Chi_Minh') -> str:
    tz = pytz.timezone(timezone)
    current_time = datetime.now(tz)
    return current_time.strftime("Current date and time: %A, %B %d, %Y at %H:%M (%Z timezone).")

# Prompt templates
class PromptTemplates:
    USER_PORTRAIT = """
    Build a user profile from:
    KNOWLEDGE: {knowledge_context}
    USER: {message}
    CONTEXT: {conversation_context}
    TIME CONTEXT: {temporal_context}
    BUSINESS OBJECTIVES: {business_objectives}
    
    Analyze the message using classification techniques from the knowledge context.
    Match your language style to the user's message.
    
    LANGUAGE REQUIREMENTS:
    1. STRICTLY use the SAME LANGUAGE as the user's message
    2. If the message is in Vietnamese, respond COMPLETELY in Vietnamese
    3. If the message is in English, respond COMPLETELY in English
    4. DO NOT mix languages in your response
    5. Maintain consistent language throughout all sections
    6. Use natural expressions and terminology from the user's language
    7. Follow cultural communication patterns of the user's language
    
    Key focus areas:
    1. Apply classification techniques from knowledge context
    2. Identify behavioral patterns and classification signals
    3. Uncover hidden needs beyond stated requirements
    4. Determine information gaps for better service
    5. Plan next engagement steps
    6. Pay attention to BUSINESS OBJECTIVES, set a business objectives for this user
    {temporal_instruction}
    
    For ANALYSIS_QUERIES:
    - Create 3-5 natural language queries for Pinecone vector search
    - Include domain terms, user type, and specific needs
    - Mix general approaches with specific details
    - Add 1-2 queries in user's native language if appropriate
    - Format as document titles/summaries
    
    For BUSINESS_GOAL:
    - Pay attention to BUSINESS OBJECTIVES, set a business objectives for this user
    - Business objectives should be specific, measurable, achievable, relevant
    
    RESPOND WITH JSON ONLY:
    {{
        "classification": "string",
        "skills": ["identified skills"],
        "requirements": ["specific needs"],
        "analysis_queries": ["vector search queries"],
        "business_goal": "string",
        "other_aspects": {{
            "behavioral_patterns": "description",
            "hidden_needs": "analysis",
            "required_info": ["missing information"],
            "next_steps": ["engagement steps"],
            "classification_criteria": "rationale"
            {temporal_references}
        }},
        "portrait_paragraph": "Comprehensive profile narrative."
    }}
    """
    
    ACTION_PLAN = """
    Create a comprehensive action plan based on:
    
    USER PROFILE NARRATIVE:
    {user_story}
    
    ANALYSIS KNOWLEDGE:
    {knowledge_context}
    
    ANALYSIS QUERIES USED:
    {analysis_queries}
    
    TIME CONTEXT:
    {temporal_context}
    {temporal_info}
    
    LANGUAGE REQUIREMENTS:
    1. STRICTLY use the SAME LANGUAGE as the user's message
    2. If the message is in Vietnamese, respond COMPLETELY in Vietnamese
    3. If the message is in English, respond COMPLETELY in English
    4. DO NOT mix languages in your response
    5. Maintain consistent language throughout all sections
    6. Use natural expressions and terminology from the user's language
    7. Follow cultural communication patterns of the user's language
    
    YOUR GOAL:
    The business goal identified in the USER PROFILE NARRATIVE is YOUR goal. Every action must contribute to this goal.
    
    PART 1: CREATE ACTION PLAN
    - Incorporate "Next best steps" from USER PROFILE NARRATIVE
    - Convert steps into specific, actionable items
    - Ensure actions align with the business goal
    - Include reasoning and expected outcomes
    - Address temporal references if detected
    
    PART 2: IDENTIFY KNOWLEDGE GAPS
    - Create 3-5 specific queries for additional information
    - Prioritize queries that improve action execution
    
    RESPOND WITH JSON ONLY:
    {{
        "next_actions": [
            {{
                "action": "specific action",
                "priority": "high|medium|low",
                "reasoning": "why needed",
                "expected_outcome": "what it achieves",
                "source": "profile|context",
                "business_goal_alignment": "how it helps goal"
            }}
        ],
        "knowledge_queries": [
            {{
                "query": "specific query",
                "purpose": "action it supports",
                "expected_knowledge": "expected info"
            }}
        ],
        "analysis_summary": {{
            "key_findings": ["findings"],
            "user_needs": ["needs"],
            "potential_challenges": ["challenges"],
            "narrative": "summary paragraph"
        }},
        "important_notes": {{
            "immediate_concerns": ["concerns"],
            "long_term_considerations": ["considerations"],
            "additional_context_needed": ["info needed"],
            "narrative": "considerations paragraph"
        }}
    }}
    """

# Pydantic model
class UserProfileModel(BaseModel):
    user_id: str
    skills: List[str] = Field(default_factory=list)
    requirements: List[str] = Field(default_factory=list)
    classification: str = "unknown"
    business_goal: str = ""
    other_aspects: Dict[str, Any] = Field(default_factory=dict)
    analysis_summary: Dict[str, Any] = Field(default_factory=lambda: {
        "key_findings": [], "user_needs": [], "potential_challenges": [], "narrative": "No analysis available."
    })
    action_plan: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)
    analysis_queries: List[str] = Field(default_factory=list)

# Custom exception
class KnowledgeFetchError(Exception):
    pass

class CoTProcessor:
    def __init__(self):
        self.user_profiles = TTLCache(maxsize=1000, ttl=3600)
        self.graph_version_id = get_current_graph_version() or str(uuid4())
        self.profiling_skills = {"knowledge_context": "Loading...", "metadata": {}}
        self.ai_business_objectives = {"knowledge_context": "Loading...", "metadata": {}}
        self.llm = ChatOpenAI(model="gpt-4o", streaming=False, temperature=0.01)
        self.llm_semaphore = Semaphore(10)  # Limit concurrent LLM calls

    async def initialize(self):
        """Initialize processor asynchronously."""
        self.profiling_skills = await self.load_knowledge_base([
            "how to classify user", "how to build user profile", "what does user profile has",
            "how to analyze user profile", "Cách phân nhóm người dùng", "Cách xây dựng hồ sơ người dùng",
            "Cách phân tích hồ sơ người dùng", "Cần có những gì trong hồ sơ người dùng"
        ])
        self.ai_business_objectives = await self.load_knowledge_base(["Mục tiêu cuộc thảo luận với khách hàng"])
        return self

    async def load_knowledge_base(self, queries: List[str]) -> Dict[str, Any]:
        """Load knowledge base for given queries."""
        logger.info(f"Loading knowledge for {len(queries)} queries")
        batch_size = 5
        batches = [queries[i:i+batch_size] for i in range(0, len(queries), batch_size)]
        results = []
        for batch in batches:
            batch_results = await asyncio.gather(
                *[fetch_knowledge(query, self.graph_version_id) for query in batch],
                return_exceptions=True
            )
            results.extend(batch_results)
        
        knowledge = {}
        for query, result in zip(queries, results):
            if isinstance(result, Exception):
                logger.error(f"Error fetching knowledge for {query}: {str(result)}")
                knowledge[query] = {}
            else:
                knowledge[query] = safe_parse_json(result) if isinstance(result, str) else result
        return self._process_knowledge(knowledge)

    def _process_knowledge(self, knowledge: Dict[str, Any]) -> Dict[str, Any]:
        """Structure fetched knowledge."""
        knowledge_context = []
        for query, entry in knowledge.items():
            content = entry.get("content", entry) if isinstance(entry, dict) else entry
            if isinstance(content, str):
                cleaned = NEWLINE_PATTERN.sub('\n\n', content)
                cleaned = SPACE_PATTERN.sub(' ', cleaned).strip()
                if cleaned:
                    knowledge_context.append(cleaned)
        
        full_knowledge = "\n\n".join(knowledge_context) if knowledge_context else "No knowledge available."
        return {
            "knowledge_context": full_knowledge,
            "metadata": {
                "source_queries": list(knowledge.keys()),
                "entry_count": len(knowledge_context),
                "timestamp": datetime.now().isoformat()
            }
        }

    async def process_incoming_message(self, message: str, conversation_context: str, user_id: str, thread_id: Optional[str] = None) -> Dict[str, Any]:
        """Process incoming message with CoT flow."""
        logger.info(f"Processing message from user {user_id}")
        try:
            if self.profiling_skills["knowledge_context"] == "Loading...":
                self.profiling_skills = await self.load_knowledge_base([])  # Reload if needed
            
            if thread_id:
                emit_event("analysis", thread_id, {
                    "type": "analysis", "content": "Starting analysis", "complete": False, "status": "starting"
                })

            user_profile = await self._get_or_build_user_profile(user_id, message, conversation_context)
            
            if thread_id:
                emit_event("analysis", thread_id, {
                    "type": "analysis", "content": "User profile analysis complete", "complete": False,
                    "status": "profile_complete", "user_profile": serialize_model(user_profile)
                })

            analysis_knowledge = await self._search_analysis_knowledge(user_profile.classification, user_profile, message)
            
            if thread_id:
                emit_event("analysis", thread_id, {
                    "type": "analysis", "content": "Analysis knowledge found.", "complete": False,
                    "status": "knowledge_found", "analysis_knowledge": analysis_knowledge
                })

            action_plan = await self._decide_action_plan_with_llm(user_profile, analysis_knowledge)
            user_profile.action_plan = action_plan
            user_profile.analysis_summary = action_plan.get("analysis_summary", "Analysis completed.")
            
            if thread_id:
                emit_event("analysis", thread_id, {
                    "type": "analysis", "content": "Action plan created", "complete": False,
                    "status": "action_plan_created", "action_plan": action_plan
                })

            response = await self._execute_action_plan(user_profile, message, conversation_context)
            self.user_profiles[user_id] = user_profile

            if isinstance(response, str):
                response = {"status": "success", "message": response, "user_profile": serialize_model(user_profile)}
            else:
                response.setdefault("status", "success")
                response.setdefault("user_profile", serialize_model(user_profile))
            return response
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            if thread_id:
                emit_event("analysis", thread_id, {
                    "type": "analysis", "content": str(e), "complete": True, "status": "error"
                })
            return {"status": "error", "message": f"Error: {str(e)}"}

    async def _get_or_build_user_profile(self, user_id: str, message: str, conversation_context: str) -> UserProfileModel:
        """Get cached profile or build new one."""
        if user_id in self.user_profiles:
            profile = self.user_profiles[user_id]
            if (datetime.now() - profile.timestamp).total_seconds() < self.user_profiles.ttl:
                logger.info(f"Using cached profile for user {user_id}")
                return profile
        return await self._build_user_portrait_with_llm(user_id, message, conversation_context)

    async def _build_user_portrait_with_llm(self, user_id: str, message: str, conversation_context: str) -> UserProfileModel:
        """Build user profile using LLM."""
        knowledge_context = self.profiling_skills.get("knowledge_context", "")
        business_objectives = self.ai_business_objectives.get("knowledge_context", "")
        temporal_context = get_temporal_context()
        
        temporal_keywords = [
            "today", "tomorrow", "yesterday", "next week", "last week", "this week",
            "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
            "morning", "afternoon", "evening", "night", "ngày mai", "hôm nay", "hôm qua",
            "tuần tới", "tuần sau", "tuần trước", "thứ hai", "thứ ba", "thứ tư", "thứ năm",
            "thứ sáu", "thứ bảy", "chủ nhật"
        ]
        needs_temporal_resolution = any(keyword in message.lower() for keyword in temporal_keywords)
        temporal_instruction = """
        7. Resolve any temporal references (like 'Sunday', 'tomorrow', 'next week') to specific dates
        """ if needs_temporal_resolution else ""
        temporal_references = """
        "temporal_references": {
            "original_mentions": ["array of temporal expressions"],
            "resolved_dates": ["array of resolved dates in YYYY-MM-DD"],
            "resolved_times": ["array of resolved times in HH:MM if applicable"]
        },
        """ if needs_temporal_resolution else ""
        
        async with self.llm_semaphore:
            prompt = PromptTemplates.USER_PORTRAIT.format(
                knowledge_context=knowledge_context,
                message=message,
                conversation_context=conversation_context,
                temporal_context=temporal_context,
                business_objectives=business_objectives,
                temporal_instruction=temporal_instruction,
                temporal_references=temporal_references
            )
            response = await self.llm.ainvoke(prompt)
        
        content = response.content.strip()
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            content = json_match.group(0)
        
        try:
            llm_response = json.loads(content)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse LLM response: {content[:100]}...")
            llm_response = {
                "classification": "unknown", "skills": [], "requirements": [], "analysis_queries": [],
                "business_goal": "Keep user engaged", "other_aspects": {"error": "Failed to parse LLM response"},
                "portrait_paragraph": "Unable to generate portrait."
            }
        
        user_profile = UserProfileModel(
            user_id=user_id,
            skills=llm_response.get("skills", []),
            requirements=llm_response.get("requirements", []),
            classification=llm_response.get("classification", "unknown"),
            analysis_queries=llm_response.get("analysis_queries", []),
            business_goal=llm_response.get("business_goal", ""),
            other_aspects={
                **llm_response.get("other_aspects", {}),
                "portrait_paragraph": llm_response.get("portrait_paragraph", "")
            }
        )
        self.user_profiles[user_id] = user_profile
        return user_profile

    async def _search_analysis_knowledge(self, classification: str, user_profile: Optional[UserProfileModel], message: str) -> Dict[str, Any]:
        """Search for analysis methods."""
        if not user_profile or not user_profile.analysis_queries:
            return {
                "analysis_methods": [], "focus_areas": [], "additional_context_needed": [],
                "rationale": "No analysis queries available"
            }
        
        knowledge_entries = []
        for query in user_profile.analysis_queries:
            try:
                knowledge = await fetch_knowledge(query, self.graph_version_id)
                if isinstance(knowledge, dict) and knowledge.get("status") == "error":
                    continue
                content = knowledge.get("content", knowledge) if isinstance(knowledge, dict) else knowledge
                knowledge_entries.append({"query": query, "knowledge": content})
            except Exception as e:
                logger.error(f"Error fetching knowledge for '{query}': {str(e)}")
        
        combined_knowledge = "\n\n".join([f"Query: {e['query']}\nKnowledge: {e['knowledge']}" for e in knowledge_entries])
        return {
            "analysis_methods": [e["query"] for e in knowledge_entries],
            "knowledge_context": combined_knowledge,
            "query_count": len(knowledge_entries),
            "rationale": f"Analyzed {len(knowledge_entries)} queries"
        }

    async def _user_story(self, user_profile: UserProfileModel) -> str:
        """Transform user profile into a narrative."""
        classification = user_profile.classification
        skills = ', '.join(user_profile.skills) or 'None'
        requirements = ', '.join(user_profile.requirements) or 'None'
        portrait = user_profile.other_aspects.get("portrait_paragraph", "")
        behavioral = user_profile.other_aspects.get("behavioral_patterns", "")
        hidden_needs = user_profile.other_aspects.get("hidden_needs", "")
        criteria = user_profile.other_aspects.get("classification_criteria", "")
        required_info = ', '.join(user_profile.other_aspects.get("required_info", [])) or "None"
        next_steps = ', '.join(user_profile.other_aspects.get("next_steps", [])) or "None"
        business_goal = user_profile.business_goal
        return f"""This user is {classification} who {criteria}. {portrait} Their behavior shows {behavioral}, 
        suggesting {hidden_needs}. With {skills} skills, they're looking to {requirements}. 
        Missing information: {required_info}. Next steps: {next_steps}. Your goal: {business_goal}"""

    async def _decide_action_plan_with_llm(self, user_profile: UserProfileModel, analysis_knowledge: Dict[str, Any]) -> Dict[str, Any]:
        """Create action plan using LLM."""
        user_story = await self._user_story(user_profile)
        temporal_info = ""
        if "temporal_references" in user_profile.other_aspects:
            temporal_references = user_profile.other_aspects.get("temporal_references", {})
            for i, mention in enumerate(temporal_references.get("original_mentions", [])):
                date = temporal_references.get("resolved_dates", [])[i] if i < len(temporal_references.get("resolved_dates", [])) else "unknown"
                time = temporal_references.get("resolved_times", [])[i] if i < len(temporal_references.get("resolved_times", [])) else ""
                temporal_info += f"When user mentioned '{mention}', they meant: {date}{' at ' + time if time else ''}\n"
        
        async with self.llm_semaphore:
            prompt = PromptTemplates.ACTION_PLAN.format(
                user_story=user_story,
                knowledge_context=analysis_knowledge.get('knowledge_context', ''),
                analysis_queries=', '.join(analysis_knowledge.get('analysis_methods', [])),
                temporal_context=get_temporal_context(),
                temporal_info=temporal_info
            )
            response = await self.llm.ainvoke(prompt)
        
        content = response.content.strip()
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            content = json_match.group(0)
        
        try:
            action_plan = json.loads(content)
            action_plan["metadata"] = {
                "timestamp": datetime.now().isoformat(),
                "analysis_query_count": len(analysis_knowledge.get('analysis_methods', [])),
                "user_classification": user_profile.classification
            }
            return action_plan
        except json.JSONDecodeError:
            logger.error(f"Failed to parse action plan: {content[:100]}...")
            return {
                "next_actions": [], "knowledge_queries": [], "analysis_summary": {"narrative": "Unable to generate."},
                "important_notes": {"narrative": "Unable briestag (xai): Unable to generate."}, "metadata": {"timestamp": datetime.now().isoformat()}
            }

    async def _execute_action_plan(self, user_profile: UserProfileModel, message: str, conversation_context: str) -> Dict[str, Any]:
        """Generate user-friendly response."""
        if not user_profile.action_plan:
            return {"status": "error", "message": "No action plan available"}
        
        action_plan = user_profile.action_plan
        analysis_summary = action_plan.get("analysis_summary", {})
        next_actions = action_plan.get("next_actions", [])
        knowledge_queries = action_plan.get("knowledge_queries", [])
        
        additional_knowledge = ""
        for query_info in knowledge_queries[:3]:
            try:
                query = query_info.get('query', '') if isinstance(query_info, dict) else query_info
                if query:
                    knowledge = await fetch_knowledge(query, self.graph_version_id)
                    additional_knowledge += f"Query: {query}\nResult: {knowledge}\n\n"
            except Exception as e:
                logger.error(f"Error fetching knowledge for '{query}': {str(e)}")
        
        temporal_info = ""
        if "temporal_references" in user_profile.other_aspects:
            temporal_references = user_profile.other_aspects.get("temporal_references", {})
            for i, mention in enumerate(temporal_references.get("original_mentions", [])):
                date = temporal_references.get("resolved_dates", [])[i] if i < len(temporal_references.get("resolved_dates", [])) else "unknown"
                time = temporal_references.get("resolved_times", [])[i] if i < len(temporal_references.get("resolved_times", [])) else ""
                temporal_info += f"When user mentioned '{mention}', they meant: {date}{' at ' + time if time else ''}\n"
        
        prompt = f"""Generate a concise, culturally appropriate response based on:
        CURRENT MESSAGE: {message}
        CONVERSATION: {conversation_context}
        TIME CONTEXT: {get_temporal_context()}
        {temporal_info}
        ANALYSIS: {json.dumps(analysis_summary, indent=2)}
        PLAN: {json.dumps(next_actions, indent=2)}
        CONTEXT: Class={user_profile.classification}, Skills={', '.join(user_profile.skills)}, Needs={', '.join(user_profile.requirements)}
        INFO: {additional_knowledge}
        
        EXECUTION STRATEGY:
        1. Execute actions in EXACT order as specified
        2. Do not skip or add actions
        3. Use factual information from knowledge sources
        4. Apply domain expertise
        5. Link insights from sources
        6. Be time-specific with dates/times
        
        REQUIREMENTS:
        1. Keep response under 100 words
        2. Use culturally appropriate language
        3. Match domain expert tone
        4. Personalize without repeating question
        5. Prioritize clarity and next steps
        6. Avoid formulaic greetings
        7. Be specific with dates (e.g., "Sunday, May 12")
        """
        
        async with self.llm_semaphore:
            response = await self.llm.ainvoke(prompt)
            return {
                "status": "success",
                "message": response.content.strip(),
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "user_id": user_profile.user_id,
                    "knowledge_queries_used": knowledge_queries,
                    "additional_knowledge_found": bool(additional_knowledge)
                }
            }

async def fetch_knowledge(query: str, graph_version_id: str = "", state: Optional[Dict] = None) -> Dict:
    """Fetch knowledge from brain."""
    try:
        if not graph_version_id:
            raise KnowledgeFetchError("No graph version ID provided")
        
        if await ensure_brain_loaded(graph_version_id):
            brain = await get_brain(graph_version_id)
            if not brain:
                raise KnowledgeFetchError("Failed to get brain instance")
            
            results = await brain.get_similar_vectors_by_text(query, top_k=5, threshold=0.35)
            knowledge_entries = []
            
            for vector_id, vector, metadata, similarity in results:
                if any(entry.get("id") == vector_id for entry in knowledge_entries):
                    continue
                
                raw_text = metadata.get("raw", "") if isinstance(metadata, dict) else str(metadata)
                if not raw_text and isinstance(metadata, dict):
                    raw_text = next((metadata.get(k) for k in ["content", "text", "data"] if metadata.get(k)), json.dumps(metadata))
                
                knowledge_entries.append({
                    "id": vector_id,
                    "query": query,
                    "raw": raw_text,
                    "similarity": float(similarity)
                })
            
            sorted_entries = sorted(knowledge_entries, key=lambda x: x.get("similarity", 0), reverse=True)[:5]
            knowledge_context = prepare_knowledge(sorted_entries, query, is_profiling=True)
            return {"status": "success", "data": knowledge_context}
    except Exception as e:
        logger.error(f"Error in fetch_knowledge: {str(e)}")
        return {"status": "error", "message": str(e)}

def prepare_knowledge(knowledge_entries: List[Dict[str, Any]], query: str, is_profiling: bool = False) -> str:
    """Prepare knowledge entries for LLM."""
    similarity_threshold = 0.2 if is_profiling else 0.4
    ranked_entries = []
    
    for entry in knowledge_entries:
        raw_text = entry.get("raw", "")
        if is_vector_data(raw_text):
            raw_text = "Content unavailable (vector data)"
        
        if not raw_text:
            title = entry.get("title", "Untitled")
            description = entry.get("description", "")
            content = entry.get("content", "")
            if is_vector_data(title):
                title = "Untitled"
            if is_vector_data(description):
                description = ""
            if is_vector_data(content):
                content = ""
            raw_text = f"{title}\n{description}\n{content}".strip() or f"No content for query: {query}"
        
        relevance_score = entry.get("similarity", 0.0) * 0.7 + (1.0 if query.lower() in raw_text.lower() else 0.0) * 0.3
        if relevance_score >= similarity_threshold:
            ranked_entries.append((entry, relevance_score, raw_text))
    
    ranked_entries.sort(key=lambda x: x[1], reverse=True)
    top_entries = ranked_entries[:5]
    
    if not top_entries:
        return "No relevant knowledge found."
    
    formatted_output = [f"KNOWLEDGE ENTRY {i}:\nTitle: {e[0].get('title', 'Untitled')}\n\n{e[2]}\n----\n" for i, e in enumerate(top_entries, 1)]
    return "\n".join(formatted_output)

def is_vector_data(text: str) -> bool:
    """Check if text is vector data."""
    return isinstance(text, str) and text.startswith("[") and "]" in text and (
        "..." in text or any(x in text for x in ["-0.", "0.", "1.", "2."]) or "_tfl-user_" in text
    )

def emit_event(event_type: str, thread_id: str, data: Dict[str, Any]) -> bool:
    """Emit an event to all clients in a thread room."""
    try:
        from socketio_manager import emit_event as socket_emit
        return socket_emit(event_type, thread_id, data)
    except ImportError:
        logger.error(f"socketio_manager not available for {event_type} emission")
        return False
    except Exception as e:
        logger.error(f"Error emitting {event_type}: {str(e)}")
        return False

async def process_llm_with_tools(
    user_message: str,
    conversation_history: List[Dict],
    state: Dict,
    graph_version_id: str,
    thread_id: Optional[str] = None
) -> AsyncGenerator[Union[str, Dict], None]:
    """Process user message with tools."""
    conversation_context = f"User: {user_message}\n"
    if conversation_history:
        # Convert to list before reversing and slicing
        recent_messages = []
        history_slice = list(reversed(conversation_history[:-1]))[:30]
        for msg in history_slice:
            role, content = msg.get("role", ""), msg.get("content", "")
            if role and content:
                recent_messages.append(f"{role.capitalize()}: {content}")
        if recent_messages:
            conversation_context = "\n".join(reversed(recent_messages)) + "\n" + conversation_context
    
    if 'cot_processor' not in state:
        state['cot_processor'] = CoTProcessor()
        await state['cot_processor'].initialize()
    
    state['graph_version_id'] = graph_version_id
    response = await state['cot_processor'].process_incoming_message(
        user_message, conversation_context, state.get('user_id', 'unknown'), thread_id
    )
    
    yield response
    state.setdefault("messages", []).append({"role": "assistant", "content": response.get("message", "")})
    state["prompt_str"] = response.get("message", "")
    yield {"state": state}

async def execute_tool(tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a tool."""
    if tool_name == "knowledge_query":
        return await knowledge_query_helper(
            parameters.get("query", ""),
            parameters.get("context", ""),
            parameters.get("graph_version_id", "")
        )
    return {"status": "error", "message": f"Unknown tool: {tool_name}"}

async def knowledge_query_helper(query: str, context: str, graph_version_id: str) -> Dict[str, Any]:
    """Query knowledge base."""
    try:
        knowledge_data = await fetch_knowledge(query, graph_version_id)
        if isinstance(knowledge_data, dict) and knowledge_data.get("status") == "error":
            return knowledge_data
        
        result_data = safe_parse_json(knowledge_data) if isinstance(knowledge_data, str) else knowledge_data
        return {"status": "success", "message": f"Knowledge queried for {query}", "data": result_data}
    except Exception as e:
        logger.error(f"Error fetching knowledge: {str(e)}")
        return {"status": "error", "message": f"Failed to fetch knowledge: {str(e)}"}