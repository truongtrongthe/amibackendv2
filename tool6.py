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

# Pydantic model for UserProfile
class UserProfileModel(BaseModel):
    user_id: str
    skills: List[str] = Field(default_factory=list)
    requirements: List[str] = Field(default_factory=list)
    classification: str = "unknown"
    business_goal: str = ""
    other_aspects: Dict[str, Any] = Field(default_factory=dict)
    analysis_summary: Dict[str, Any] = Field(default_factory=lambda: {
        "key_findings": [],
        "user_needs": [],
        "potential_challenges": [],
        "narrative": "No analysis available."
    })
    action_plan: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)
    analysis_queries: List[str] = Field(default_factory=list)

class CoTProcessor:
    def __init__(self):
        self.user_profiles = TTLCache(maxsize=1000, ttl=3600)
        self.graph_version_id = get_current_graph_version() or str(uuid4())
        # Initialize profiling_skills with empty dict, will be loaded asynchronously
        self.profiling_skills = {"knowledge_context": "Loading...", "metadata": {}}
        self.ai_business_objectives = {"knowledge_context": "Loading...", "metadata": {}}

    async def initialize(self):
        """Initialize the processor asynchronously"""
        self.profiling_skills = await self._load_profiling_skills()
        self.ai_business_objectives = await self._load_business_objectives_awareness()
        return self

    async def _load_profiling_skills(self) -> Dict[str, Any]:
        """Load knowledge base concurrently using fetch_knowledge."""
        logger.info("Loading knowledge base")
        queries = [
            "how to classify user",
            "how to build user profile",
            "what does user profile has",
            "how to analyze user profile",
            "Cách phân nhóm người dùng",
            "Cách xây dựng hồ sơ người dùng",
            "Cách phânuyantich hồ sơ người dùng",
            "Cần có những gì trong hồ sơ người dùng",
        ]
        
        try:
            results = await asyncio.gather(
                *[fetch_knowledge(query, self.graph_version_id) for query in queries],
                return_exceptions=True
            )
            profiling_skills = {}
            for query, result in zip(queries, results):
                if isinstance(result, Exception):
                    logger.error(f"Error fetching knowledge for query {query}: {str(result)}")
                    profiling_skills[query] = {}
                else:
                    try:
                        profiling_skills[query] = json.loads(result) if isinstance(result, str) else result
                    except json.JSONDecodeError:
                        profiling_skills[query] = {"content": result}
            return self._process_profiling_skills(profiling_skills)
        except Exception as e:
            logger.error(f"Error loading profiling skills: {str(e)}")
            return {"knowledge_context": "No knowledge available.", "metadata": {"error": str(e)}}
        
    async def _load_business_objectives_awareness(self) -> Dict[str, Any]:
        """Load knowledge base concurrently using fetch_knowledge."""
        logger.info("Loading knowledge base")
        queries = [
            "Mục tiêu cuộc thảo luận với khách hàng"
        ]
        
        try:
            results = await asyncio.gather(
                *[fetch_knowledge(query, self.graph_version_id) for query in queries],
                return_exceptions=True
            )
            business_objectives = {}
            for query, result in zip(queries, results):
                if isinstance(result, Exception):
                    logger.error(f"Error fetching knowledge for query {query}: {str(result)}")
                    business_objectives[query] = {}
                else:
                    try:
                        business_objectives[query] = json.loads(result) if isinstance(result, str) else result
                    except json.JSONDecodeError:
                        business_objectives[query] = {"content": result}
            return self._process_business_objectives_knowledge(business_objectives)
        except Exception as e:
            logger.error(f"Error loading profiling skills: {str(e)}")
            return {"knowledge_context": "No knowledge available.", "metadata": {"error": str(e)}}

    def _process_profiling_skills(self, profiling_skills: Dict[str, Any]) -> Dict[str, Any]:
        """Structure the fetched knowledge."""
        # Extract and combine all knowledge entries
        knowledge_context = []
        
        for query, knowledge in profiling_skills.items():
            if isinstance(knowledge, dict):
                # If it's already a dict, use it directly
                if "content" in knowledge:
                    knowledge_context.append(knowledge["content"])
            elif isinstance(knowledge, str):
                # If it's a string, try to parse it
                try:
                    parsed = json.loads(knowledge)
                    if isinstance(parsed, dict) and "content" in parsed:
                        knowledge_context.append(parsed["content"])
                except json.JSONDecodeError:
                    # If not JSON, use the string directly
                    knowledge_context.append(knowledge)

        # Clean up the knowledge context
        cleaned_context = []
        for entry in knowledge_context:
            # Remove excessive newlines and spaces
            cleaned = re.sub(r'\n{3,}', '\n\n', entry)
            cleaned = re.sub(r' +', ' ', cleaned)
            cleaned = cleaned.strip()
            if cleaned:
                cleaned_context.append(cleaned)

        # Combine all knowledge into a single context
        full_knowledge = "\n\n".join(cleaned_context) if cleaned_context else "No knowledge available."
        
        return {
            "knowledge_context": full_knowledge,
            "metadata": {
                "source_queries": list(profiling_skills.keys()),
                "entry_count": len(cleaned_context),
                "timestamp": datetime.now().isoformat()
            }
        }
    
    def _process_business_objectives_knowledge(self, business_objectives: Dict[str, Any]) -> Dict[str, Any]:
        """Structure the fetched knowledge."""
        # Extract and combine all knowledge entries
        knowledge_context = []
        
        for query, knowledge in business_objectives.items():
            if isinstance(knowledge, dict):
                # If it's already a dict, use it directly
                if "content" in knowledge:
                    knowledge_context.append(knowledge["content"])
            elif isinstance(knowledge, str):
                # If it's a string, try to parse it
                try:
                    parsed = json.loads(knowledge)
                    if isinstance(parsed, dict) and "content" in parsed:
                        knowledge_context.append(parsed["content"])
                except json.JSONDecodeError:
                    # If not JSON, use the string directly
                    knowledge_context.append(knowledge)

        # Clean up the knowledge context
        cleaned_context = []
        for entry in knowledge_context:
            # Remove excessive newlines and spaces
            cleaned = re.sub(r'\n{3,}', '\n\n', entry)
            cleaned = re.sub(r' +', ' ', cleaned)
            cleaned = cleaned.strip()
            if cleaned:
                cleaned_context.append(cleaned)

        # Combine all knowledge into a single context
        full_knowledge = "\n\n".join(cleaned_context) if cleaned_context else "No knowledge available."
        
        return {
            "knowledge_context": full_knowledge,
            "metadata": {
                "source_queries": list(business_objectives.keys()),
                "entry_count": len(cleaned_context),
                "timestamp": datetime.now().isoformat()
            }
        }

    async def process_incoming_message(self, message: str, conversation_context: str, user_id: str, thread_id: Optional[str] = None) -> Dict[str, Any]:
        """Process incoming message with CoT flow.
        This function is the main function that processes the incoming message.
        It will get or build user profile, search analysis knowledge, propose an action plan, and execute the action plan.  
        """
        logger.info(f"Processing message from user {user_id}")
        try:
            # Ensure profiling skills are loaded
            if "knowledge_context" in self.profiling_skills and self.profiling_skills["knowledge_context"] == "Loading...":
                logger.info("Profiling skills not loaded yet, loading now...")
                self.profiling_skills = await self._load_profiling_skills()
            
            # Emit initial analysis event
            if thread_id:
                emit_analysis_event(thread_id, {
                    "type": "analysis",
                    "content": "Starting analysis",
                    "complete": False,
                    "status": "starting"
                })

            logger.info(f"Now I am processing message from user {user_id}")
            user_profile = await self._get_or_build_user_profile(user_id, message, conversation_context)
            
            # Emit user profile analysis event
            if thread_id:
                profile_data = serialize_model(user_profile)
                logger.info(f"Emitting profile_complete event with data: {json.dumps(profile_data, indent=2)}")
                emit_analysis_event(thread_id, {
                    "type": "analysis",
                    "content": "User profile analysis complete",
                    "complete": False,
                    "status": "profile_complete",
                    "user_profile": profile_data
                })
            
            logger.info("I've Built User Profile:")
            logger.info(f"Classification: {user_profile.classification}")
            logger.info(f"Portrait: {user_profile.other_aspects.get('portrait_paragraph', 'Not available')}")
            logger.info(f"Why I classified like this: {user_profile.other_aspects.get('classification_criteria', 'Not available')}")
            logger.info(f"I propose these queries to analyze user: {user_profile.analysis_queries}")
            
            # Emit knowledge search event
            if thread_id:
                emit_analysis_event(thread_id, {
                    "type": "analysis",
                    "content": "Searching for analysis knowledge...",
                    "complete": False,
                    "status": "searching_knowledge"
                })
            
            logger.info(f"Now I am searching knowledge follow the queries ...")
            analysis_knowledge = await self._search_analysis_knowledge(user_profile.classification, user_profile, message)
            logger.info(f"I found analysis knowledge: {analysis_knowledge}")
            
            # Emit knowledge found event
            if thread_id:
                emit_analysis_event(thread_id, {
                    "type": "analysis",
                    "content": "Analysis knowledge found.Analyzing it and proposing an action plan.",
                    "complete": False,
                    "status": "knowledge_found",
                    "analysis_knowledge": analysis_knowledge
                })
            
            logger.info(f"Now I am proposing an action plan for user {user_id} ...")
            action_plan = await self._decide_action_plan_with_llm(user_profile, analysis_knowledge)
            user_profile.action_plan = action_plan
            user_profile.analysis_summary = action_plan.get("analysis_summary", "Analysis completed.")
            
            # Emit action plan event
            if thread_id:
                emit_analysis_event(thread_id, {
                    "type": "analysis",
                    "content": "Action plan created",
                    "complete": False,
                    "status": "action_plan_created",
                    "action_plan": action_plan
                })
            
            # Log detailed analysis results
            logger.info("Here is the analysis results:")
            analysis_summary = action_plan.get("analysis_summary", {})
            logger.info(f"Key Findings: {analysis_summary.get('key_findings', [])}")
            logger.info(f"User Needs: {analysis_summary.get('user_needs', [])}")
            logger.info(f"Analysis Narrative: {analysis_summary.get('narrative', 'No narrative available')}")
            
            # Log action plan details
            next_actions = action_plan.get("next_actions", [])
            logger.info("Here is the action plan:")
            for i, action in enumerate(next_actions, 1):
                logger.info(f"Action {i}:")
                logger.info(f"  - What: {action.get('action', 'No action specified')}")
                logger.info(f"  - Priority: {action.get('priority', 'Not specified')}")
                logger.info(f"  - Why: {action.get('reasoning', 'No reasoning provided')}")
                logger.info(f"  - Expected: {action.get('expected_outcome', 'No expected outcome specified')}")
            
            logger.info(f"Now I am executing the action plan for user {user_id} ...")
            response = await self._execute_action_plan(user_profile, message, conversation_context)
            logger.info(f"Now I have response: {response}")
            self.user_profiles[user_id] = user_profile
            logger.info(f"I have updated user profile!")

            

            if isinstance(response, str):
                response = {"status": "success", "message": response, "user_profile": serialize_model(user_profile)}
            else:
                response.setdefault("status", "success")
                response.setdefault("user_profile", serialize_model(user_profile))
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

    async def _get_or_build_user_profile(self, user_id: str, message: str, conversation_context: str) -> UserProfileModel:
        """Get cached profile or build new one."""
        if user_id in self.user_profiles:
            profile = self.user_profiles[user_id]
            if (datetime.now() - profile.timestamp).total_seconds() < self.user_profiles.ttl:
                logger.info(f"Using cached profile for user {user_id}")
                return profile
        return await self._build_user_portrait_with_llm(user_id, message,conversation_context=conversation_context)

    async def _build_user_portrait_with_llm(self, user_id: str, message: str, conversation_context: str) -> UserProfileModel:
        """Build user profile using LLM."""
        
        # Get the knowledge context
        knowledge_context = self.profiling_skills.get("knowledge_context", "")
        business_objectives = self.ai_business_objectives.get("knowledge_context", "")
        
        # Add temporal context for resolving relative time references
        vietnam_tz = pytz.timezone('Asia/Ho_Chi_Minh')
        current_time = datetime.now(vietnam_tz)
        date_str = current_time.strftime("%A, %B %d, %Y")
        time_str = current_time.strftime("%H:%M")
        
        # Create a temporal context string
        temporal_context = f"Current date and time: {date_str} at {time_str} (Asia/Ho_Chi_Minh timezone)."
        
        # Check if message contains temporal references that need resolution
        temporal_keywords = [
            "today", "tomorrow", "yesterday", 
            "next week", "last week", "this week",
            "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
            "morning", "afternoon", "evening", "night",
            "ngày mai", "hôm nay", "hôm qua", "tuần tới", "tuần sau", "tuần trước", 
            "thứ hai", "thứ ba", "thứ tư", "thứ năm", "thứ sáu", "thứ bảy", "chủ nhật"
        ]
        
        needs_temporal_resolution = any(keyword in message.lower() for keyword in temporal_keywords)
        
        # Add temporal resolution instruction if needed
        temporal_resolution_instruction = ""
        if needs_temporal_resolution:
            temporal_resolution_instruction = """
            "temporal_references": {
                "original_mentions": ["array of temporal expressions found in message"],
                "resolved_dates": ["array of resolved absolute dates in YYYY-MM-DD format"],
                "resolved_times": ["array of resolved times in HH:MM format if applicable"]
            },
            """
            logger.info(f"Detected temporal references in message. Adding temporal resolution instructions.")
        
        # Log conversation context length to help with debugging
        conversation_lines = conversation_context.count('\n') if conversation_context else 0
        logger.info(f"Building user portrait with {conversation_lines} lines of conversation context")
        
        prompt = f"""Build a comprehensive user profile by THOROUGHLY ANALYZING the ENTIRE conversation history:

                ===== CRITICAL LANGUAGE INSTRUCTION =====
                YOU MUST RESPOND ENTIRELY IN THE SAME LANGUAGE AS THE USER'S MESSAGE:
                - If the user wrote in Vietnamese → RESPOND 100% IN VIETNAMESE 
                - If the user wrote in English → RESPOND 100% IN ENGLISH
                - DO NOT mix languages in your response
                - EVERY part of your JSON output must be in the user's language
                - This includes ALL fields, descriptions, queries, and analysis
                ============================================

                KNOWLEDGE: {knowledge_context}
                CONVERSATION HISTORY: {conversation_context}
                CURRENT MESSAGE: {message}
                TIME CONTEXT: {temporal_context}
                BUSINESS OBJECTIVES: {business_objectives}

                DEEP CONVERSATION ANALYSIS INSTRUCTIONS:
                1. THOROUGHLY EXAMINE the COMPLETE conversation history - NOT just the current message
                2. Identify patterns, topics, and key information across ALL messages
                3. Track changes in user's tone, needs, and requirements throughout the conversation
                4. Look for implicit information and context clues across the ENTIRE conversation
                5. Pay special attention to consistency/inconsistency in user statements over time
                6. Consider how earlier messages provide context for later statements
                7. Look for previous questions that may not have been fully answered

                Analyze the conversation using classification techniques from the knowledge context.
                
                CONVERSATION STAGE AWARENESS:
                Identify the current stage (first contact, information gathering, problem identification, etc.) and adapt your approach accordingly. Consider whether the conversation is progressing naturally or needs redirection.

                KEY FOCUS AREAS:
                • Apply classification techniques from knowledge context
                • Identify behavioral patterns and hidden needs beyond stated requirements
                • Determine information gaps and plan appropriate next steps
                • Analyze alignment with business objectives
                • Identify recurring themes or topics throughout the conversation
                • Track any evolution in the user's needs or priorities across messages
                {f"• Resolve temporal references (like 'Sunday', 'tomorrow', 'next week') to specific dates" if needs_temporal_resolution else ""}

                BUSINESS_GOAL SELECTION (CRITICAL):
                1. CAREFULLY EXAMINE the KNOWLEDGE and BUSINESS OBJECTIVES sections to identify ALL available business objectives
                2. SELECT EXACTLY ONE objective that best matches this user and conversation stage
                3. Your selection MUST be based on explicit objectives found in the knowledge context
                4. Format the selected objective as a clear, SMART goal statement 
                5. In your output, bold the business_goal
                6. PROVIDE A DETAILED JUSTIFICATION explaining WHY this objective is the most appropriate choice

                GENERATE 3-5 ANALYSIS_QUERIES:
                [OBJECTIVE-SPECIFIC QUERIES - generate your own based on context]
                - Generate a query about the definition of the selected business objective
                - Generate a query about implementation strategies for the selected objective

                [USER-SPECIFIC QUERIES - generate your own based on context]
                - Generate a query to analyze the user with their identified classification
                - Generate a query about solutions for the specific user need/problem identified
                - Optionally add one more context-specific query if needed

                RESPOND WITH JSON (following the language instruction at the top of this prompt):
                {{
                    "classification": "string (from knowledge context)",
                    "skills": ["identified skills/capabilities"],
                    "requirements": ["specific needs/requirements"],
                    "analysis_queries": ["3-5 precise queries as instructed above"],
                    "business_goal": "**SELECTED OBJECTIVE: clearly formatted SMART goal statement**",
                    "other_aspects": {{
                        "conversation_stage": "identified stage of the conversation",
                        "behavioral_patterns": "observed behavior description",
                        "hidden_needs": "underlying needs analysis",
                        "required_info": ["missing information needed from the user"],
                        "next_steps": ["engagement steps appropriate for conversation stage"],
                        "classification_criteria": "classification rationale",
                        "conversation_history_insights": "Key insights extracted from analyzing the COMPLETE conversation history",
                        "business_objective_justification": "DETAILED explanation of why this business objective is most appropriate"
                        {temporal_resolution_instruction if needs_temporal_resolution else ""}
                    }},
                    "portrait_paragraph": "Comprehensive profile including classification, needs, behavior, and business objective alignment, incorporating insights from the ENTIRE conversation history."
                }}
                """
        
        try:
            response = await LLM.ainvoke(prompt)
            content = response.content.strip()
            logger.info(f"RAW PORTRAIT RESPONSE: {content}")
            
            # Try to extract JSON if there's any surrounding text
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                content = json_match.group(0)
            try:
                llm_response = json.loads(content)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM response as JSON: {str(e)}")
                logger.error(f"Raw content: {content}")
                # Create a default response
                llm_response = {
                    "classification": "unknown",
                    "skills": [],
                    "requirements": [],
                    "analysis_queries": [],
                    "business_goal": "Keep user stayed engaged",
                    "other_aspects": {
                        "error": "Failed to parse LLM response",
                        "raw_content": content[:100] + "..." if len(content) > 100 else content
                    },
                    "portrait_paragraph": "Unable to generate portrait at this time."
                }
            
            # Extract portrait_paragraph from the response
            portrait_paragraph = llm_response.get("portrait_paragraph", "")
            
            # Create user profile with all fields
            user_profile = UserProfileModel(
                user_id=user_id,
                skills=llm_response.get("skills", []),
                requirements=llm_response.get("requirements", []),
                classification=llm_response.get("classification", "unknown"),
                analysis_queries=llm_response.get("analysis_queries", []),
                business_goal=llm_response.get("business_goal", ""),
                other_aspects={
                    **llm_response.get("other_aspects", {}),
                    "portrait_paragraph": portrait_paragraph
                }
            )
            
            # Log the created profile for debugging
            logger.info(f"Created user profile with portrait: {portrait_paragraph}")
            
            # If we resolved temporal references, log them
            if needs_temporal_resolution and "temporal_references" in user_profile.other_aspects:
                temporal_refs = user_profile.other_aspects.get("temporal_references", {})
                logger.info(f"Resolved temporal references: {json.dumps(temporal_refs, indent=2)}")
            
            self.user_profiles[user_id] = user_profile
            return user_profile
        except Exception as e:
            logger.error(f"Error building user portrait: {str(e)}")
            return UserProfileModel(
                user_id=user_id,
                business_goal_query="How to identify business objectives for error case",
                other_aspects={
                    "error": str(e),
                    "portrait_paragraph": "Error occurred while generating portrait."
                }
            )

    async def _search_analysis_knowledge(self, classification: str, user_profile: Optional[UserProfileModel], message: str) -> Dict[str, Any]:
        """Search for analysis methods based on user profile's analysis queries."""
        if not user_profile or not user_profile.analysis_queries:
            logger.warning("No analysis queries available for user profile")
            return {
                "analysis_methods": [],
                "focus_areas": [],
                "additional_context_needed": [],
                "rationale": "No analysis queries available"
            }

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
                    logger.info(f"Knowledge found for Analysis: {knowledge_entries}")
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

    async def _user_story(self, user_profile: UserProfileModel) -> str:
        """Transform user profile into a compact narrative format."""
        # Extract key elements
        classification = user_profile.classification
        skills = ', '.join(user_profile.skills) if user_profile.skills else 'None'
        requirements = ', '.join(user_profile.requirements) if user_profile.requirements else 'None'
        
        # Extract other aspects
        portrait = user_profile.other_aspects.get("portrait_paragraph", "")
        behavioral = user_profile.other_aspects.get("behavioral_patterns", "")
        hidden_needs = user_profile.other_aspects.get("hidden_needs", "")
        criteria = user_profile.other_aspects.get("classification_criteria", "")
        required_info = ', '.join(user_profile.other_aspects.get("required_info", [])) or "None"
        next_steps = ', '.join(user_profile.other_aspects.get("next_steps", [])) or "None"
        business_goal = user_profile.business_goal
        
        # Extract conversation history insights if available
        conversation_insights = user_profile.other_aspects.get("conversation_history_insights", "")
        conversation_stage = user_profile.other_aspects.get("conversation_stage", "")
        
        # Construct compact narrative with conversation history insights
        return f"""This user is {classification} who {criteria}. {portrait} Their behavior shows {behavioral}, 
        suggesting {hidden_needs} as underlying needs. With {skills} skills, they're looking to {requirements}. 
        The conversation is currently in the {conversation_stage} stage. {conversation_insights if conversation_insights else ""}
        Missing information we still need: {required_info}. Next best steps: {next_steps}. AI business goal: {business_goal}"""

    async def _decide_action_plan_with_llm(self, user_profile: UserProfileModel, analysis_knowledge: Dict[str, Any]) -> Dict[str, Any]:
        """Create action plan using LLM based on user profile and analysis knowledge."""
        
        # Get the user story
        user_story = await self._user_story(user_profile)
        logger.info(f"USER STORY: {user_story}")
        
        # Extract temporal references if available
        temporal_info = ""
        if "temporal_references" in user_profile.other_aspects:
            temporal_references = user_profile.other_aspects.get("temporal_references", {})
            original_mentions = temporal_references.get("original_mentions", [])
            resolved_dates = temporal_references.get("resolved_dates", [])
            resolved_times = temporal_references.get("resolved_times", [])
            
            if original_mentions and resolved_dates:
                temporal_info = "TEMPORAL REFERENCES:\n"
                for i, mention in enumerate(original_mentions):
                    resolved_date = resolved_dates[i] if i < len(resolved_dates) else "unknown"
                    resolved_time = resolved_times[i] if resolved_times and i < len(resolved_times) else ""
                    
                    if resolved_time:
                        temporal_info += f"When user mentioned '{mention}', they meant: {resolved_date} at {resolved_time}\n"
                    else:
                        temporal_info += f"When user mentioned '{mention}', they meant: {resolved_date}\n"
                
                logger.info(f"Including temporal references in action planning: {json.dumps(temporal_references, indent=2)}")
        
        # Add current date/time information for temporal context
        from datetime import datetime
        import pytz
        
        # Use Vietnam timezone by default
        vietnam_tz = pytz.timezone('Asia/Ho_Chi_Minh')
        current_time = datetime.now(vietnam_tz)
        date_str = current_time.strftime("%A, %B %d, %Y")
        time_str = current_time.strftime("%H:%M")
        
        # Create a temporal context string
        temporal_context = f"Current date and time: {date_str} at {time_str} (Asia/Ho_Chi_Minh timezone)."
        
        prompt = f"""
        Create a comprehensive action plan based on the user's profile and the analysis:

        USER PROFILE NARRATIVE:
        {user_story}

        ANALYSIS KNOWLEDGE:
        {analysis_knowledge.get('knowledge_context', '')}

        ANALYSIS QUERIES USED:
        {', '.join(analysis_knowledge.get('analysis_methods', []))}

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
        The business goal identified in the USER PROFILE NARRATIVE is YOUR goal for this interaction. Every action you plan must contribute to achieving this goal.

        Your task has two parts:
        
        PART 1: CREATE ACTION PLAN
        First, create a detailed action plan that addresses the user's needs and requirements while ensuring it contributes to achieving YOUR goal identified in the USER PROFILE NARRATIVE.
        
        IMPORTANT: You MUST incorporate the "Next best steps" identified in the USER PROFILE NARRATIVE into your action plan. These steps were specifically identified for this user and should be prioritized in your planning.
        
        When creating the action plan:
        1. First, review the "Next best steps" from the user profile
        2. Consider how these steps contribute to achieving YOUR goal identified in the USER PROFILE NARRATIVE
        3. Convert these steps into specific, actionable items
        4. Maintain their priority and order as identified in the profile
        5. Add any additional actions needed to achieve YOUR goal
        6. Ensure all actions have clear reasoning and expected outcomes
        7. Write all narrative sections in the same language style as the user's message
        8. If temporal references were detected, ensure the action plan explicitly addresses timing requirements

        PART 2: IDENTIFY KNOWLEDGE GAPS
        After creating the action plan, identify 3-5 specific knowledge queries we should run to get additional information needed to execute the plan effectively.
        
        When identifying knowledge gaps:
        1. Review each action in your plan
        2. For each action, identify what additional knowledge would help finding how to execute.
        3. Create specific queries to gather this knowledge by adding "How to" or "What is" to the action
        4. Ensure queries are in the same language as the user's message

        IMPORTANT: You MUST respond with a valid JSON object only. Do not include any other text or explanation.
        The JSON must follow this exact structure:
        {{
            "next_actions": [
                {{
                    "action": "string (specific action to take)",
                    "priority": "high|medium|low",
                    "reasoning": "string (why this action is needed)",
                    "expected_outcome": "string (what this action should achieve)",
                    "source": "string (whether this action came from user profile next steps or was added based on current context)",
                    "business_goal_alignment": "string (how this action helps achieve YOUR goal identified in the user profile)"
                }}
            ],
            "knowledge_queries": [
                {{
                    "query": "string (specific query to get more information)",
                    "purpose": "string (which action this query supports)",
                    "expected_knowledge": "string (what information we expect to get)"
                }}
            ],
            "analysis_summary": {{
                "key_findings": ["array of key findings"],
                "user_needs": ["array of identified needs"],
                "potential_challenges": ["array of potential challenges"],
                "narrative": "A detailed paragraph summarizing the analysis, written in the user's language style"
            }},
            "important_notes": {{
                "immediate_concerns": ["array of immediate concerns"],
                "long_term_considerations": ["array of long-term considerations"],
                "additional_context_needed": ["array of additional information needed"],
                "narrative": "A paragraph explaining the important considerations, written in the user's language style"
            }}
        }}
        """
        
        try:
            response = await LLM.ainvoke(prompt)
            content = response.content.strip()
            #logger.info(f"RAW ACTION PLAN RESPONSE: {content}")
            
            # Try to extract JSON if there's any surrounding text
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                content = json_match.group(0)
            
            try:
                action_plan = json.loads(content)
                # Add metadata
                action_plan["metadata"] = {
                    "timestamp": datetime.now().isoformat(),
                    "analysis_query_count": len(analysis_knowledge.get('analysis_methods', [])),
                    "user_classification": user_profile.classification
                }
                
                # Add temporal information to metadata if available
                if "temporal_references" in user_profile.other_aspects:
                    action_plan["metadata"]["temporal_references"] = user_profile.other_aspects.get("temporal_references", {})
                
                return action_plan
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM action plan: {str(e)}")
                logger.error(f"Raw content: {content}")
                return {
                    "next_actions": [],
                    "knowledge_queries": [],
                    "analysis_summary": {
                        "key_findings": [],
                        "user_needs": [],
                        "potential_challenges": [],
                        "narrative": "Unable to generate analysis summary at this time."
                    },
                    "important_notes": {
                        "immediate_concerns": [],
                        "long_term_considerations": [],
                        "additional_context_needed": [],
                        "narrative": "Unable to generate important notes at this time."
                    },
                    "metadata": {
                        "error": "Failed to parse action plan",
                        "timestamp": datetime.now().isoformat()
                    }
                }
        except Exception as e:
            logger.error(f"Error in action plan: {str(e)}")
            return {
                "next_actions": [],
                "knowledge_queries": [],
                "analysis_summary": {
                    "key_findings": [],
                    "user_needs": [],
                    "potential_challenges": [],
                    "narrative": "Error occurred while generating analysis summary."
                },
                "important_notes": {
                    "immediate_concerns": [],
                    "long_term_considerations": [],
                    "additional_context_needed": [],
                    "narrative": "Error occurred while generating important notes."
                },
                "metadata": {
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
            }

    async def _execute_action_plan(self, user_profile: UserProfileModel, message: str, conversation_context: str = "") -> Dict[str, Any]:
        """Generate user-friendly response based on the action plan."""
        logger.info("Executing action plan")
        
        if not user_profile.action_plan:
            return {
                "status": "error",
                "message": "No action plan available"
            }

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
        prompt = f"""Generate a response to the user that STRICTLY FOLLOWS the action plan:

        CURRENT MESSAGE: {message}
        CONVERSATION: {conversation_context}
        TIME CONTEXT: {temporal_context}
        {temporal_info if temporal_info else ""}
        ANALYSIS: {json.dumps(analysis_summary, indent=2) if analysis_summary else "N/A"}
        ACTION PLAN: {json.dumps(next_actions, indent=2) if next_actions else "N/A"}
        USER CONTEXT: Classification={user_profile.classification}, Business Goal={user_profile.business_goal}
        ADDITIONAL INFO: {additional_knowledge}

        STRICT EXECUTION REQUIREMENTS:
        1. ONLY implement the actions listed in the ACTION PLAN - nothing more, nothing less
        2. DO NOT introduce any external services, specialists, websites, or social media not mentioned in the plan
        3. DO NOT mention or suggest Facebook, messengers, or any external contacts unless explicitly in the plan
        4. Follow the exact priority order of the actions as they appear in the plan
        5. Use the same language as the user (Vietnamese/English/etc.) throughout

        FORMAT GUIDELINES:
        • Be concise and professional
        • Match the user's language and cultural context
        • Use appropriate pronouns (e.g., in Vietnamese: Anh/Chị/Em) based on detected language
        • Skip formulaic greetings and focus on delivering the actions
        • Adapt to the specific user needs indicated in the analysis

        CRITICAL: Your response must ONLY contain actions explicitly specified in the ACTION PLAN. Do not suggest connecting with specialists, visiting websites, or using external services unless these are explicitly listed as actions in the plan.
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
    
    # Get or create a CoTProcessor instance
    if 'cot_processor' not in state:
        cot_processor = CoTProcessor()
        # Initialize properly
        await cot_processor.initialize()
        state['cot_processor'] = cot_processor
    else:
        cot_processor = state['cot_processor']
    
    state['graph_version_id'] = graph_version_id
    
    try:
        # Process the message - events will be emitted from process_incoming_message
        response = await cot_processor.process_incoming_message(
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
