import json
import asyncio
import logging
from typing import Dict, List, Any, Optional, Union, AsyncGenerator
from datetime import datetime
from uuid import uuid4
from cachetools import TTLCache
import re

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from aitools import fetch_knowledge, brain, emit_analysis_event  # Assuming these are available
from brain_singleton import get_current_graph_version

# Custom JSON encoder to handle datetime objects
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

def serialize_model(model: BaseModel) -> Dict[str, Any]:
    """Serialize a Pydantic model to a JSON-serializable dict."""
    return json.loads(json.dumps(model.model_dump(), cls=DateTimeEncoder))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize LLM
LLM = ChatOpenAI(model="gpt-4o", streaming=False, temperature=0.01)

# Pydantic model for UserProfile
class UserProfileModel(BaseModel):
    user_id: str
    skills: List[str] = Field(default_factory=list)
    requirements: List[str] = Field(default_factory=list)
    classification: str = "unknown"
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
        self.profiling_skills = asyncio.run(self._load_profiling_skills())

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

    async def process_incoming_message(self, message: str, conversation_context: str, user_id: str, thread_id: Optional[str] = None) -> Dict[str, Any]:
        """Process incoming message with CoT flow.
        This function is the main function that processes the incoming message.
        It will get or build user profile, search analysis knowledge, propose an action plan, and execute the action plan.  
        """
        logger.info(f"Processing message from user {user_id}")
        try:
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
        
        prompt = f"""Build a user profile from:
                KNOWLEDGE: {knowledge_context}
                USER: {message}
                CONTEXT: {conversation_context}

                Analyze the message using classification techniques from the knowledge context.
                Match your language style to the user's message.

                Key focus areas:
                1. Apply classification techniques from knowledge context
                2. Identify behavioral patterns and classification signals
                3. Uncover hidden needs beyond stated requirements
                4. Determine information gaps for better service
                5. Plan next engagement steps

                For ANALYSIS_QUERIES:
                - Create 5-7 natural language queries for Pinecone vector search
                - Include domain terms, user type, and specific needs
                - Mix general approaches with specific details
                - Add 1-2 queries in user's native language if appropriate
                - Format as document titles/summaries (e.g., "Best approach for price-sensitive maternity patients")

                RESPOND WITH JSON ONLY:
                {{
                    "classification": "string (from knowledge context)",
                    "skills": ["identified skills/capabilities"],
                    "requirements": ["specific needs/requirements"],
                    "analysis_queries": ["vector search queries for Pinecone"],
                    "other_aspects": {{
                        "behavioral_patterns": "observed behavior description",
                        "hidden_needs": "underlying needs analysis",
                        "required_info": ["missing information we need from/about the user"],
                        "next_steps": ["engagement steps"],
                        "classification_criteria": "classification rationale"
                    }},
                    "portrait_paragraph": "Comprehensive profile narrative including classification, needs, and behavior."
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
                other_aspects={
                    **llm_response.get("other_aspects", {}),
                    "portrait_paragraph": portrait_paragraph
                }
            )
            
            # Log the created profile for debugging
            logger.info(f"Created user profile with portrait: {portrait_paragraph}")
            
            self.user_profiles[user_id] = user_profile
            return user_profile
        except Exception as e:
            logger.error(f"Error building user portrait: {str(e)}")
            return UserProfileModel(
                user_id=user_id,
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
                    knowledge_entries.append({
                        "query": query,
                        "knowledge": knowledge
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
        
        # Construct compact narrative
        return f"""This user is {classification} who {criteria}. {portrait} Their behavior shows {behavioral}, 
        suggesting {hidden_needs} as underlying needs. With {skills} skills, they're looking to {requirements}. 
        Missing information we still need: {required_info}. Next best steps: {next_steps}"""

    async def _decide_action_plan_with_llm(self, user_profile: UserProfileModel, analysis_knowledge: Dict[str, Any]) -> Dict[str, Any]:
        """Create action plan using LLM based on user profile and analysis knowledge."""
        
        # Extract knowledge context from analysis
        knowledge_context = analysis_knowledge.get("knowledge_context", "")
        analysis_queries = analysis_knowledge.get("analysis_methods", [])
        
        # Generate user story narrative
        user_story = await self._user_story(user_profile)
        
        prompt = f"""
        Create a comprehensive action plan based on the user's profile and the analysis:

        USER PROFILE NARRATIVE:
        {user_story}

        ANALYSIS KNOWLEDGE:
        {knowledge_context}

        ANALYSIS QUERIES USED:
        {', '.join(analysis_queries)}

        Your task has two parts:
        
        PART 1: IDENTIFY KNOWLEDGE GAPS
        First, identify 3-5 specific knowledge queries we should run to get additional information before finalizing the action plan.
        
        Notice in the USER PROFILE NARRATIVE where it mentions "Missing information we still need: [...]" - these are specific information gaps about the user that we've identified as missing. Your knowledge queries should:
        1. Prioritize finding information that can fill these specific gaps about the user
        2. Focus on knowledge that would help us better understand or serve this user without having to ask them directly
        3. Include queries that would retrieve helpful context, explanations, or approaches for addressing users with these information gaps
        
        Your queries should address:
        - The user's specific situation or needs
        - Relevant products, services, or solutions
        - Best practices for this type of user
        - Potential objections or concerns
        
        PART 2: CREATE ACTION PLAN
        Based on this information, create a detailed action plan that addresses the user's needs and requirements.
        Focus on providing clear, actionable steps that will help the user achieve their goals.
        Each action should have a clear priority, reasoning, and expected outcome.
        Write all narrative sections in the same language style as the user's message.

        IMPORTANT: You MUST respond with a valid JSON object only. Do not include any other text or explanation.
        The JSON must follow this exact structure:
        {{
            "knowledge_queries": [
                "specific query 1 to get more information",
                "specific query 2 to get more information",
                "specific query 3 to get more information"
            ],
            "next_actions": [
                {{
                    "action": "string (specific action to take)",
                    "priority": "high|medium|low",
                    "reasoning": "string (why this action is needed)",
                    "expected_outcome": "string (what this action should achieve)"
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
                    "analysis_query_count": len(analysis_queries),
                    "user_classification": user_profile.classification
                }
                return action_plan
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM action plan: {str(e)}")
                logger.error(f"Raw content: {content}")
                return {
                    "knowledge_queries": [],
                    "next_actions": [],
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
                "knowledge_queries": [],
                "next_actions": [],
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
        
        # Fetch additional knowledge if queries are provided
        additional_knowledge = ""
        if knowledge_queries:
            logger.info(f"Fetching additional knowledge for {len(knowledge_queries)} queries")
            try:
                knowledge_results = []
                for query in knowledge_queries[:3]:  # Limit to top 3 queries
                    try:
                        knowledge = await fetch_knowledge(query, self.graph_version_id)
                        if knowledge:
                            knowledge_results.append(f"Query: {query}\nResult: {knowledge}")
                    except Exception as e:
                        logger.error(f"Error fetching knowledge for query '{query}': {str(e)}")
                
                additional_knowledge = "\n\n".join(knowledge_results)
                logger.info(f"Fetched additional knowledge: {len(additional_knowledge)} chars")
            except Exception as e:
                logger.error(f"Error fetching additional knowledge: {str(e)}")

        # Build prompt for LLM to generate response
        prompt = f"""Generate a concise, culturally appropriate response to this user based on:

        CURRENT MESSAGE: {message}
        CONVERSATION: {conversation_context}
        ANALYSIS: {json.dumps(analysis_summary, indent=2) if analysis_summary else "N/A"}
        PLAN: {json.dumps(next_actions, indent=2) if next_actions else "N/A"}
        CONTEXT: Class={user_profile.classification}, Skills={', '.join(user_profile.skills)}, Needs={', '.join(user_profile.requirements)}
        INFO: {additional_knowledge}

        EXECUTION STRATEGY:
        1. EXTRACT KEY INFORMATION - Thoroughly examine all sources (conversation, knowledge, analysis) for data needed to execute the plan
        2. IDENTIFY INFORMATION GAPS - If information needed for next_actions isn't available, acknowledge this fact
        3. USE AVAILABLE INFORMATION - Prioritize factual information from knowledge sources over assumptions
        4. APPLY DOMAIN EXPERTISE - For any next_action, include specific details identified from the available sources
        5. CONNECT DOTS - Link insights from different sources to provide comprehensive execution of the plan

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
                    "additional_knowledge_found": bool(additional_knowledge)
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
    conversation_context = "\n".join(
        [f"{'AI' if msg['role'] == 'assistant' else 'User'}: {msg['content']}"
         for msg in conversation_history[-15:-1] if msg.get("role") and msg.get("content")]
    ) + f"\nUser: {user_message}\n"
    cot_processor = state.get('cot_processor', CoTProcessor())
    state['cot_processor'] = cot_processor
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
        return {
            "status": "success",
            "message": f"Knowledge queried for {query}",
            "data": json.loads(knowledge_data) if isinstance(knowledge_data, str) else knowledge_data
        }
    except Exception as e:
        logger.error(f"Error fetching knowledge: {str(e)}")
        return {"status": "error", "message": f"Failed to fetch knowledge: {str(e)}"}