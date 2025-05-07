import json
import os
import re
import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime

import requests
from bs4 import BeautifulSoup

# Assuming tool_helpers.py is in the same directory or properly referenced
try:
    from tool_helpers import (
        run_terminal_cmd_helper,
        list_dir_helper,
        read_file_helper,
        edit_file_helper,
        grep_search_helper,
        codebase_search_helper,
        file_search_helper,
        delete_file_helper,
        web_search_helper,
        reapply_helper,
    )
except ImportError:
    print("tool_helpers.py not found. Ensure it is in the correct directory.")

# Configuration and logging setup
CONFIG = json.load(open("config.json", "r")) if os.path.exists("config.json") else {}
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class UserProfile:
    user_id: str
    skills: List[str] = field(default_factory=list)
    requirements: List[str] = field(default_factory=list)
    classification: str = "unknown"
    other_aspects: Dict[str, Any] = field(default_factory=dict)
    analysis_summary: str = ""
    action_plan: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    analysis_queries: List[str] = field(default_factory=list)

class CoTProcessor:
    def __init__(self):
        self.user_profiles: Dict[str, UserProfile] = {}
        self.knowledge_base = self._load_knowledge_base()
        self.profile_cache_timeout = 3600  # Cache timeout in seconds (1 hour)

    def _load_knowledge_base(self) -> Dict[str, Any]:
        """Load knowledge base for user profiling and analysis. Placeholder for actual implementation."""
        return {
            "classification_techniques": ["skill-based", "requirement-based", "behavioral"],
            "profile_building": ["skills", "requirements", "interaction_history"],
            "analysis_methods": ["pattern_matching", "trend_analysis", "predictive_modeling"],
            "action_strategies": {
                "beginner": {"next_actions": ["tutorial", "basic_support"], "notes": "Needs guidance."},
                "advanced": {"next_actions": ["advanced_tools", "custom_solutions"], "notes": "Can handle complex tasks."}
            }
        }

    async def process_incoming_message(self, message: str, user_id: str) -> Dict[str, Any]:
        """Process incoming message following the CoT flow with optimized LLM calls."""
        logger.info(f"Processing message from user {user_id}")
        
        # Step 1: Understand the user with an LLM call, check cache first
        logger.info("Step 1: Understanding the user with LLM call")
        user_profile = await self._get_or_build_user_profile(user_id, message)
        
        # Step 2: Analyze the user portrait with lightweight rule-based approach
        logger.info("Step 2: Analyzing user portrait with lightweight approach")
        analysis_knowledge = self._search_analysis_knowledge(user_profile.classification, user_profile, message)
        
        # Step 3: Decide action plan with a structured CoT prompt and LLM call
        logger.info("Step 3: Deciding action plan with structured CoT LLM call")
        action_plan = await self._decide_action_plan_with_llm(user_profile, analysis_knowledge)
        user_profile.action_plan = action_plan
        user_profile.analysis_summary = action_plan.get("analysis_summary", "Analysis completed via LLM.")
        
        # Step 4: Execute the action plan
        logger.info("Step 4: Executing action plan")
        response = await self._execute_action_plan(user_profile, message)
        
        self.user_profiles[user_id] = user_profile
        return response

    async def _get_or_build_user_profile(self, user_id: str, message: str) -> UserProfile:
        """Get user profile from cache if recent, otherwise build with LLM call."""
        if user_id in self.user_profiles:
            cached_profile = self.user_profiles[user_id]
            time_elapsed = (datetime.now() - cached_profile.timestamp).total_seconds()
            if time_elapsed < self.profile_cache_timeout:
                logger.info(f"Using cached profile for user {user_id}, age: {time_elapsed} seconds")
                return cached_profile
            else:
                logger.info(f"Cached profile for user {user_id} expired, rebuilding")
        
        # If no valid cache, build new profile
        knowledge = self._understand_user(user_id)
        return await self._build_user_portrait_with_llm(user_id, knowledge, message)

    def _understand_user(self, user_id: str) -> Dict[str, Any]:
        """Step 1.1 & 1.2: Gather knowledge to understand the user."""
        logger.info(f"Gathering knowledge to understand user {user_id}")
        return {
            "classification_techniques": self.knowledge_base.get("classification_techniques", []),
            "profile_building": self.knowledge_base.get("profile_building", []),
            "other_aspects": {"source": "knowledge_base"}
        }

    async def _build_user_portrait_with_llm(self, user_id: str, knowledge: Dict[str, Any], message: str) -> UserProfile:
        """Step 1.3: Build user portrait using an LLM call for dynamic profiling and generate knowledge-fetching queries."""
        logger.info(f"Building user portrait for {user_id} with LLM and generating knowledge queries")
        if user_id in self.user_profiles:
            return self.user_profiles[user_id]
        
        # Placeholder for actual LLM call logic with CoT prompt
        # In a real implementation, this would construct a detailed CoT prompt and call an LLM API
        prompt = self._build_cot_prompt_for_profiling_and_queries(user_id, message, knowledge)
        logger.info(f"CoT prompt for profiling and queries: {prompt[:100]}...")
        
        # Placeholder response from LLM
        classification = "beginner" if len(knowledge["classification_techniques"]) < 2 else "advanced"
        skills = ["basic_coding"]  # Would be determined by LLM
        requirements = ["support"]  # Would be determined by LLM
        analysis_queries = [
            f"Recall analysis methods for a {classification} user needing {requirements[0]}",
            "Prioritize quick analysis if urgency detected in message"
        ]  # Would be generated by LLM
        
        user_profile = UserProfile(
            user_id=user_id,
            skills=skills,
            requirements=requirements,
            classification=classification,
            other_aspects=knowledge.get("other_aspects", {}),
            analysis_queries=analysis_queries  # Store queries in profile for Step 2
        )
        return user_profile

    def _build_cot_prompt_for_profiling_and_queries(self, user_id: str, message: str, knowledge: Dict[str, Any]) -> str:
        """Build a structured CoT prompt for LLM to build user profile and generate knowledge-fetching queries."""
        return f"""
        You are an AI assistant tasked with building a user profile and proactively generating queries for knowledge fetching. Follow this step-by-step reasoning process:

        1. **Understand the User**: Review the following information:
        - User ID: {user_id}
        - Message: {message}
        - Available Knowledge: {knowledge.get('classification_techniques', [])}
        Analyze the user's message to determine their needs, skill level, and context. Summarize the user's profile with:
        - Classification (e.g., beginner, advanced)
        - Skills (list specific skills if evident)
        - Requirements (list specific needs or goals)

        2. **Generate Knowledge-Fetching Queries**: Based on the user profile, think ahead and generate specific queries to fetch relevant analysis knowledge for the next step. These queries should address the user's classification, requirements, and any discrepancies or unique needs in the message context. Output 2-3 targeted queries.

        Provide your response structured as follows:
        - User Profile:
        - Classification: [classification]
        - Skills: [list of skills]
        - Requirements: [list of needs]
        - Knowledge-Fetching Queries:
        - Query 1: [query text]
        - Query 2: [query text]
        - Query 3: [query text, if applicable]
        Ensure each step builds on the previous one for a coherent output.
        """

    def _search_analysis_knowledge(self, classification: str = "unknown", user_profile: Optional[UserProfile] = None, message: str = "") -> List[str]:
        """Step 2.1: Search for knowledge on how to analyze the profile, tailored by classification and context."""
        logger.info("Searching for analysis knowledge with progressive recall")
        analysis_methods = self.knowledge_base.get("analysis_methods", [])
        
        # First, check if user_profile has pre-generated queries from Step 1
        if user_profile and hasattr(user_profile, 'analysis_queries') and user_profile.analysis_queries:
            logger.info(f"Using pre-generated queries from Step 1 for user {user_profile.user_id}")
            filtered_methods = self._apply_analysis_queries(user_profile.analysis_queries, analysis_methods)
            if filtered_methods:
                logger.info(f"Progressive knowledge recall using LLM queries for classification {classification}: {filtered_methods}")
                return filtered_methods
            else:
                logger.info("Pre-generated queries did not yield results, falling back to rule-based approach")
        
        # Check if an LLM call is needed for dynamic query generation due to complexity
        if user_profile and message and self._requires_llm_for_analysis_query(user_profile, message):
            logger.info(f"Complex context detected for user {user_profile.user_id}, using LLM for dynamic query")
            return self._generate_dynamic_analysis_query_with_llm(user_profile, message, analysis_methods)
        else:
            logger.info("Using enhanced rule-based progressive recall for analysis knowledge")
            # Enhanced rule-based filtering based on classification and additional context
            if classification == "beginner":
                filtered_methods = [method for method in analysis_methods if method != "predictive_modeling"]
            elif classification == "advanced":
                filtered_methods = analysis_methods
            else:
                filtered_methods = analysis_methods
            
            # Further refine with user profile details if available
            if user_profile:
                # Adjust based on specific requirements
                if "immediate_support" in user_profile.requirements:
                    filtered_methods = [method for method in filtered_methods if method == "pattern_matching"] or filtered_methods
                # Adjust based on skills
                if "advanced_coding" in user_profile.skills and classification == "advanced":
                    filtered_methods = [method for method in filtered_methods if method in ["trend_analysis", "predictive_modeling"]] or filtered_methods
                # Additional rule: Check for multiple requirements indicating complexity
                if len(user_profile.requirements) > 1:
                    filtered_methods = [method for method in filtered_methods if method in ["trend_analysis", "pattern_matching"]] or filtered_methods
            
            # Enhanced context from message with additional keywords
            if message:
                message_lower = message.lower()
                if "urgent" in message_lower or "quick" in message_lower:
                    logger.info("Message context indicates urgency, prioritizing quick analysis methods")
                    filtered_methods = [method for method in filtered_methods if method == "pattern_matching"] or filtered_methods
                elif "detailed" in message_lower or "thorough" in message_lower:
                    logger.info("Message context indicates need for detailed analysis")
                    filtered_methods = [method for method in filtered_methods if method in ["trend_analysis", "predictive_modeling"]] or filtered_methods
            
            # Progressive recall: Log the refinement process
            logger.info(f"Progressive knowledge recall for classification {classification}: {filtered_methods}")
            return filtered_methods

    def _apply_analysis_queries(self, queries: List[str], analysis_methods: List[str]) -> List[str]:
        """Apply pre-generated queries to filter analysis methods from the knowledge base."""
        logger.info("Applying pre-generated queries to filter analysis methods")
        # Placeholder logic for applying queries
        # In a real implementation, parse queries and match against knowledge base
        filtered_methods = []
        for query in queries:
            if "quick analysis" in query.lower() or "urgency" in query.lower():
                filtered_methods.extend([method for method in analysis_methods if method == "pattern_matching"])
            elif "beginner" in query.lower():
                filtered_methods.extend([method for method in analysis_methods if method != "predictive_modeling"])
            elif "advanced" in query.lower():
                filtered_methods.extend([method for method in analysis_methods if method in ["trend_analysis", "predictive_modeling"]])
        # Deduplicate while preserving order
        seen = set()
        filtered_methods = [method for method in filtered_methods if not (method in seen or seen.add(method))]
        return filtered_methods if filtered_methods else analysis_methods

    def _requires_llm_for_analysis_query(self, user_profile: UserProfile, message: str) -> bool:
        """Determine if an LLM call is needed for dynamic query generation based on complexity."""
        # Placeholder logic for complexity check
        # Example conditions: multiple conflicting requirements, unusual message content
        conflicting_requirements = len(user_profile.requirements) > 2
        unusual_context = any(keyword in message.lower() for keyword in ["confused", "contradictory", "complex"])
        return conflicting_requirements or unusual_context

    def _generate_dynamic_analysis_query_with_llm(self, user_profile: UserProfile, message: str, analysis_methods: List[str]) -> List[str]:
        """Use LLM to generate a dynamic query for analysis knowledge recall."""
        logger.info(f"Generating dynamic analysis query with LLM for user {user_profile.user_id}")
        # Placeholder for actual LLM call
        # In a real implementation, construct a prompt and call LLM API
        prompt = f"""
        Generate a query to recall analysis knowledge for a user with the following profile:
        - Classification: {user_profile.classification}
        - Skills: {', '.join(user_profile.skills)}
        - Requirements: {', '.join(user_profile.requirements)}
        - Message Context: {message[:100]}...
        Select the most relevant analysis methods from the following list: {', '.join(analysis_methods)}
        Address any discrepancies or unique needs in the context.
        """
        logger.info(f"LLM prompt for dynamic query: {prompt[:100]}...")
        
        # Placeholder response: Select methods based on classification for now
        if user_profile.classification == "beginner":
            return [method for method in analysis_methods if method != "predictive_modeling"]
        return analysis_methods

    async def _decide_action_plan_with_llm(self, user_profile: UserProfile, analysis_knowledge: List[str]) -> Dict[str, Any]:
        """Step 3: Use a structured CoT prompt with LLM to refine analysis and create action plan."""
        logger.info(f"Deciding action plan for user {user_profile.user_id} with LLM")
        # Construct structured CoT prompt
        prompt = self._build_cot_prompt(user_profile, analysis_knowledge)
        # Placeholder for actual LLM call
        # In a real implementation, this would send the prompt to an LLM API
        logger.info(f"Structured CoT prompt for LLM: {prompt[:100]}...")  # Log snippet of prompt
        
        # Placeholder logic for action plan based on classification
        action_knowledge = self._search_action_knowledge(user_profile.classification)
        return {
            "next_actions": action_knowledge.get("next_actions", []),
            "important_notes": action_knowledge.get("notes", ""),
            "analysis_summary": f"Analysis refined for {user_profile.classification} user using {analysis_knowledge}."
        }

    def _build_cot_prompt(self, user_profile: UserProfile, analysis_knowledge: List[str]) -> str:
        """Build a structured CoT prompt for LLM to refine analysis and plan actions."""
        return f"""
        You are an AI assistant tasked with creating an action plan for a user based on their profile and analysis knowledge. Follow this step-by-step reasoning process to ensure a thorough and accurate plan:

        1. **Understand the User**: Review the following user profile:
        - User ID: {user_profile.user_id}
        - Classification: {user_profile.classification}
        - Skills: {', '.join(user_profile.skills)}
        - Requirements: {', '.join(user_profile.requirements)}
        - Other Aspects: {user_profile.other_aspects}
        Summarize the user's needs and context based on this profile.

        2. **Analyze the User**: Consider the following analysis methods relevant to this user:
        - Analysis Methods: {', '.join(analysis_knowledge)}
        Use these methods to analyze the user's profile. Provide a brief summary of insights or key points from this analysis.

        3. **Decide on an Action Plan**: Based on the user's classification and the analysis insights, decide on an appropriate action plan. Select strategies that best fit the user's needs. Output the plan in the following format:
        - Next Actions: [list of actions]
        - Important Notes: [any relevant notes or considerations]

        Please provide your response structured as above, ensuring each step builds on the previous one for a coherent and tailored action plan.
        """

    def _search_action_knowledge(self, classification: str) -> Dict[str, Any]:
        """Step 3.1: Search for knowledge on what to do with this type of user."""
        logger.info(f"Searching action knowledge for classification {classification}")
        return self.knowledge_base.get("action_strategies", {}).get(classification, {"next_actions": [], "notes": ""})

    async def _execute_action_plan(self, user_profile: UserProfile, message: str) -> Dict[str, Any]:
        """Step 4: Execute the action plan, with conditional LLM call for complex execution."""
        logger.info(f"Executing action plan for user {user_profile.user_id}")
        actions = user_profile.action_plan.get("next_actions", [])
        
        # Check if actions can be handled by predefined templates
        if actions and self._can_use_action_template(actions):
            logger.info(f"Using predefined action template for user {user_profile.user_id}")
            response = self._apply_action_template(actions, user_profile.user_id)
        elif self._requires_llm_for_execution(actions):
            logger.info(f"LLM call required for execution for user {user_profile.user_id}")
            # Placeholder for LLM call for execution
            # In a real implementation, construct a prompt and call LLM API
            response = {
                "status": "success",
                "message": f"Action plan executed for user {user_profile.user_id} with LLM guidance",
                "actions_taken": actions
            }
        else:
            logger.info(f"Direct execution without LLM for user {user_profile.user_id}")
            # Direct execution without LLM, e.g., via tool calls
            response = {
                "status": "success",
                "message": f"Action plan executed for user {user_profile.user_id} directly",
                "actions_taken": actions
            }
        # Placeholder for actual execution logic
        return response

    def _can_use_action_template(self, actions: List[str]) -> bool:
        """Determine if the action plan can be handled by a predefined template."""
        # Predefined actions that have templates
        template_actions = ["tutorial", "basic_support", "advanced_tools"]
        return all(action in template_actions for action in actions)

    def _apply_action_template(self, actions: List[str], user_id: str) -> Dict[str, Any]:
        """Apply a predefined template for common actions."""
        logger.info(f"Applying action template for actions: {actions}")
        # Placeholder for template-based responses
        template_responses = {
            "tutorial": "Providing tutorial content...",
            "basic_support": "Offering basic support guidance...",
            "advanced_tools": "Suggesting advanced tool usage..."
        }
        action_messages = [template_responses.get(action, f"Executing {action}") for action in actions]
        return {
            "status": "success",
            "message": f"Action plan executed for user {user_id} using templates: {', '.join(action_messages)}",
            "actions_taken": actions
        }

    def _requires_llm_for_execution(self, actions: List[str]) -> bool:
        """Determine if the action plan requires an LLM call for execution."""
        # Placeholder logic: Assume LLM is needed if actions include complex tasks
        complex_actions = ["custom_solutions", "tutorial"]  # Example list
        return any(action in complex_actions for action in actions)

# Adapted from tool.py's process_llm_with_tools
async def process_llm_with_tools(
    message: str,
    user_id: str,
    tools: List[Dict[str, Any]],
    cot_processor: Optional[CoTProcessor] = None
) -> Dict[str, Any]:
    """Process LLM response with tools and CoT flow if processor is provided."""
    logger.info(f"Processing message with tools for user {user_id}")
    
    if cot_processor:
        logger.info(f"Using CoTProcessor for user {user_id}")
        return await cot_processor.process_incoming_message(message, user_id)
    
    # Default processing if no CoT processor is provided
    response = {
        "status": "success",
        "message": "Processed without CoT flow",
        "data": {}
    }
    
    for tool in tools:
        tool_name = tool.get("name", "")
        if not tool_name:
            continue
        
        logger.info(f"Processing tool: {tool_name}")
        tool_result = await execute_tool(tool_name, tool.get("parameters", {}))
        response["data"][tool_name] = tool_result
    
    return response

async def execute_tool(tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the specified tool with given parameters."""
    logger.info(f"Executing tool: {tool_name}")
    try:
        if tool_name == "codebase_search":
            return await codebase_search_helper(parameters.get("query", ""), parameters.get("target_directories", []))
        elif tool_name == "read_file":
            return await read_file_helper(
                parameters.get("target_file", ""),
                parameters.get("should_read_entire_file", False),
                parameters.get("start_line_one_indexed", 1),
                parameters.get("end_line_one_indexed_inclusive", 200)
            )
        elif tool_name == "run_terminal_cmd":
            return await run_terminal_cmd_helper(parameters.get("command", ""), parameters.get("is_background", False))
        elif tool_name == "list_dir":
            return await list_dir_helper(parameters.get("relative_workspace_path", "."))
        elif tool_name == "grep_search":
            return await grep_search_helper(
                parameters.get("query", ""),
                parameters.get("case_sensitive", False),
                parameters.get("include_pattern", ""),
                parameters.get("exclude_pattern", "")
            )
        elif tool_name == "edit_file":
            return await edit_file_helper(
                parameters.get("target_file", ""),
                parameters.get("instructions", ""),
                parameters.get("code_edit", "")
            )
        elif tool_name == "file_search":
            return await file_search_helper(parameters.get("query", ""))
        elif tool_name == "delete_file":
            return await delete_file_helper(parameters.get("target_file", ""))
        elif tool_name == "reapply":
            return await reapply_helper(parameters.get("target_file", ""))
        elif tool_name == "web_search":
            return await web_search_helper(parameters.get("search_term", ""))
        else:
            return {"status": "error", "message": f"Unknown tool: {tool_name}"}
    except Exception as e:
        logger.error(f"Error executing tool {tool_name}: {str(e)}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    async def test_cot_flow():
        processor = CoTProcessor()
        user_id = "test_user_123"
        message = "Hello, I need help with coding."
        result = await processor.process_incoming_message(message, user_id)
        print(f"CoT Flow Result: {result}")
    
    asyncio.run(test_cot_flow()) 