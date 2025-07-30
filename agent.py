"""
Agent Orchestration Module - Runtime execution engine for built AI agents
Provides deep reasoning and tool orchestration for specialized agent tasks
"""

import os
import json
import re
import traceback
import logging
import asyncio
from typing import Dict, List, Any, Optional, Union, AsyncGenerator
from datetime import datetime
from dataclasses import dataclass

# Import shared tools and executors
from search_tool import SearchTool
from learning_tools import LearningToolsFactory

# Import provider-specific executors (shared with Ami)
from exec_anthropic import AnthropicExecutor
from exec_openai import OpenAIExecutor

logger = logging.getLogger(__name__)

# Configure agent-specific logging
agent_logger = logging.getLogger("agent_runtime")
agent_logger.setLevel(logging.INFO)
if not agent_logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('🤖 [AGENT] %(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
    handler.setFormatter(formatter)
    agent_logger.addHandler(handler)

# Configure detailed execution logging
execution_logger = logging.getLogger("agent_execution")
execution_logger.setLevel(logging.INFO)
if not execution_logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('🔍 [EXEC] %(asctime)s - %(message)s', datefmt='%H:%M:%S')
    handler.setFormatter(formatter)
    execution_logger.addHandler(handler)

# Import language detection (shared with Ami)
try:
    from language import detect_language_with_llm, LanguageDetector
    LANGUAGE_DETECTION_AVAILABLE = True
    logger.info("Language detection imported successfully from language.py")
except Exception as e:
    logger.warning(f"Failed to import language detection from language.py: {e}")
    LANGUAGE_DETECTION_AVAILABLE = False


# NEW: Multi-Step Planning Data Structures
@dataclass
class TaskComplexityAnalysis:
    """Analysis of task complexity for multi-step planning"""
    complexity_score: int  # 1-10 scale
    complexity_level: str  # "simple", "standard", "complex"
    task_type: str  # "information_retrieval", "problem_solving", etc.
    required_steps: int  # Number of execution steps needed
    estimated_duration: str  # "short", "medium", "long"
    key_challenges: List[str]  # Identified challenges
    recommended_approach: str  # Execution approach
    confidence: float  # 0.0-1.0 confidence in analysis


@dataclass
class ExecutionStep:
    """Single step in multi-step execution plan"""
    step_number: int
    name: str
    description: str
    action: str
    tools_needed: List[str]
    success_criteria: str
    deliverable: str
    dependencies: List[int]  # Step numbers this depends on
    estimated_time: str
    validation_checkpoints: List[str]


@dataclass
class MultiStepExecutionPlan:
    """Complete multi-step execution plan"""
    plan_id: str
    task_description: str
    complexity_analysis: TaskComplexityAnalysis
    execution_steps: List[ExecutionStep]
    quality_checkpoints: List[Dict[str, Any]]
    success_metrics: List[str]
    risk_mitigation: List[str]
    total_estimated_time: str
    created_at: datetime


@dataclass
class StepExecutionResult:
    """Result of executing a single step"""
    step_number: int
    status: str  # "completed", "failed", "skipped"
    deliverable: Dict[str, Any]
    tools_used: List[str]
    execution_time: float
    success_criteria_met: bool
    validation_results: List[Dict[str, Any]]
    next_actions: List[str]
    error_details: Optional[str] = None


@dataclass
class AgentTask:
    """Single task in the agent's execution chain"""
    key: str
    task_query: str
    thought_description: str
    execution_template: str
    requires_tools: bool = True
    requires_reasoning: bool = True
    task_type: str = "general_execution"
    context_vars: Optional[Dict[str, Any]] = None
    priority: int = 1


@dataclass
class AgentExecutionPlan:
    """Complete plan for agent task execution"""
    initial_assessment: str
    tasks: List[AgentTask]
    total_tasks: int
    execution_focus: str
    
    @classmethod
    def from_agent_analysis(cls, agent_response: str) -> 'AgentExecutionPlan':
        """Parse agent LLM response into structured execution plan"""
        try:
            import json
            import re
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', agent_response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                
                tasks = []
                for i, task_data in enumerate(data.get("tasks", []), 1):
                    task = AgentTask(
                        key=task_data.get("key", f"task_{i}"),
                        task_query=task_data.get("task_query", ""),
                        thought_description=task_data.get("thought_description", ""),
                        execution_template=task_data.get("execution_template", ""),
                        requires_tools=task_data.get("requires_tools", True),
                        requires_reasoning=task_data.get("requires_reasoning", True),
                        task_type=task_data.get("task_type", "general_execution"),
                        context_vars=task_data.get("context_vars", {}),
                        priority=i
                    )
                    tasks.append(task)
                
                return cls(
                    initial_assessment=data.get("initial_assessment", "Starting task execution..."),
                    tasks=tasks,
                    total_tasks=len(tasks),
                    execution_focus=data.get("execution_focus", "task_completion")
                )
            else:
                # Fallback plan if parsing fails
                return cls._create_fallback_plan()
                
        except Exception as e:
            logger.error(f"Failed to parse agent execution plan: {e}")
            return cls._create_fallback_plan()
    
    @classmethod
    def _create_fallback_plan(cls) -> 'AgentExecutionPlan':
        """Create a basic fallback execution plan"""
        fallback_task = AgentTask(
            key="general_task",
            task_query="complete the requested task using available knowledge",
            thought_description="Let me process this request using my specialized knowledge...",
            execution_template="Processing {task} with {approach}...",
            task_type="general_execution",
            priority=1
        )
        
        return cls(
            initial_assessment="I need to analyze this request and execute it using my specialized capabilities...",
            tasks=[fallback_task],
            total_tasks=1,
            execution_focus="task_completion"
        )


@dataclass
class AgentExecutionRequest:
    """Request model for agent execution (different from Ami's ToolExecutionRequest)"""
    llm_provider: str  # 'anthropic' or 'openai'
    user_request: str  # The actual task to perform
    agent_id: str  # Specific agent instance ID
    agent_type: str  # Type of agent (e.g., "sales_agent", "support_agent", "analyst_agent")
    system_prompt: Optional[str] = None
    model: Optional[str] = None
    model_params: Optional[Dict[str, Any]] = None
    org_id: Optional[str] = "default"
    user_id: Optional[str] = "anonymous"
    
    # Agent operational mode
    agent_mode: Optional[str] = "execute"  # "collaborate" or "execute"
    
    # Agent-specific parameters (different focus from Ami)
    enable_deep_reasoning: Optional[bool] = True  # Deep reasoning enabled by default for agents
    reasoning_depth: Optional[str] = "standard"  # "light", "standard", "deep"
    task_focus: Optional[str] = "execution"  # "execution", "analysis", "communication"
    
    # Shared tool parameters
    enable_tools: Optional[bool] = True
    force_tools: Optional[bool] = False
    tools_whitelist: Optional[List[str]] = None
    
    # Agent knowledge context
    specialized_knowledge_domains: Optional[List[str]] = None  # Agent's specialization areas
    conversation_history: Optional[List[Dict[str, Any]]] = None
    max_history_messages: Optional[int] = 15  # Agents focus on recent context
    max_history_tokens: Optional[int] = 4000


@dataclass
class AgentExecutionResponse:
    """Response model for agent execution"""
    success: bool
    result: str
    agent_id: str
    agent_type: str
    execution_time: float
    tasks_completed: int
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None



    



class AgentOrchestrator:
    """Agent orchestration engine with dynamic configuration loading"""
    
    def __init__(self):
        """Initialize the agent orchestrator"""
        self.available_tools = self._initialize_shared_tools()
        
        # Use same provider executors as Ami (shared infrastructure)
        self.anthropic_executor = AnthropicExecutor(self)
        self.openai_executor = OpenAIExecutor(self)
        
        # Dynamic agent configuration cache
        self.agent_config_cache = {}
        self.cache_ttl = 300  # 5 minutes cache TTL
        
        # NEW: Multi-step execution tracking
        self.active_execution_plans = {}  # Track active execution plans
        self.execution_history = {}  # Track completed executions
        
        # Initialize language detection (shared with Ami)
        if LANGUAGE_DETECTION_AVAILABLE:
            self.language_detector = LanguageDetector()
        else:
            self.language_detector = None
            
        agent_logger.info("Agent Orchestrator initialized with multi-step planning support")
    
    def _initialize_shared_tools(self) -> Dict[str, Any]:
        """Initialize tools shared with Ami"""
        tools = {}
        
        # Initialize search tools factory (shared)
        try:
            from search_tool import create_search_tool
            tools["search_factory"] = create_search_tool
            logger.info("Search tools factory initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize search tools factory: {e}")
        
        # Initialize context tool (shared)
        try:
            from context_tool import ContextTool
            tools["context"] = ContextTool()
            logger.info("Context tool initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize context tool: {e}")
        
        # Initialize brain vector tool (shared - agents access same knowledge base)
        try:
            from brain_vector_tool import BrainVectorTool
            tools["brain_vector"] = BrainVectorTool()
            logger.info("Brain vector tool initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize brain vector tool: {e}")
        
        # Initialize learning tools factory (shared)
        try:
            from learning_tools import LearningToolsFactory
            tools["learning_factory"] = LearningToolsFactory
            logger.info("Learning tools factory initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize learning tools factory: {e}")
    
        # Add file access tools (local and Google Drive)
        try:
            from file_access_tool import FileAccessTool
            file_access = FileAccessTool()
            tools["file_access"] = file_access
            logger.info("File access tools (local + Google Drive) initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize file access tools: {e}")
        
        # Add business logic tools
        try:
            from business_logic_tool import BusinessLogicTool
            business_logic = BusinessLogicTool()
            tools["business_logic"] = business_logic
            logger.info("Business logic tools initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize business logic tools: {e}")
        
        return tools
    
    def _get_search_tool(self, llm_provider: str, model: str = None) -> Any:
        """Get the appropriate search tool based on LLM provider (shared with Ami)"""
        if "search_factory" in self.available_tools:
            return self.available_tools["search_factory"](llm_provider, model)
        else:
            logger.warning("Search factory not available, falling back to SERPAPI")
            try:
                from search_tool import SearchTool
                return SearchTool()
            except Exception as e:
                logger.error(f"Failed to create fallback search tool: {e}")
                return None
    
    async def load_agent_config(self, agent_id: str, org_id: str) -> Dict[str, Any]:
        """
        Load agent configuration from database with caching
        
        Args:
            agent_id: Agent UUID or name
            org_id: Organization ID
            
        Returns:
            Agent configuration dictionary
        """
        cache_key = f"{org_id}:{agent_id}"
        
        # Check cache first
        if cache_key in self.agent_config_cache:
            cached_config, cached_time = self.agent_config_cache[cache_key]
            if (datetime.now() - cached_time).seconds < self.cache_ttl:
                agent_logger.info(f"Using cached config for agent {agent_id}")
                return cached_config
        
        try:
            from orgdb import get_agent, get_agents
            
            # Try to get agent by ID first
            agent = get_agent(agent_id)
            
            # If not found by ID, try to find by name within the organization
            if not agent:
                agents = get_agents(org_id, status="active")
                agent = next((a for a in agents if a.name.lower() == agent_id.lower()), None)
            
            if not agent:
                raise ValueError(f"Agent '{agent_id}' not found in organization {org_id}")
            
            if agent.org_id != org_id:
                raise ValueError(f"Agent '{agent_id}' not accessible for organization {org_id}")
            
            if agent.status != "active":
                raise ValueError(f"Agent '{agent_id}' is not active (status: {agent.status})")
            
            # Build configuration dictionary
            config = {
                "id": agent.id,
                "agent_id": agent.agent_id,
                "name": agent.name,
                "description": agent.description,
                "system_prompt": agent.system_prompt,
                "tools_list": agent.tools_list,
                "knowledge_list": agent.knowledge_list,
                "created_by": agent.created_by,
                "created_date": agent.created_date,
                "updated_date": agent.updated_date
            }
            
            # Cache the configuration
            self.agent_config_cache[cache_key] = (config, datetime.now())
            
            agent_logger.info(f"Loaded config for agent: {agent.name} (ID: {agent.id})")
            return config
            
        except Exception as e:
            agent_logger.error(f"Failed to load agent config for {agent_id}: {str(e)}")
            raise Exception(f"Failed to load agent configuration: {str(e)}")
    
    async def resolve_agent_identifier(self, agent_identifier: str, org_id: str) -> str:
        """
        Resolve agent identifier (ID, name, or description) to agent ID
        
        Args:
            agent_identifier: Agent ID, name, or description
            org_id: Organization ID
            
        Returns:
            Resolved agent ID
        """
        try:
            from orgdb import get_agent, get_agents
            
            # Try direct ID lookup first
            agent = get_agent(agent_identifier)
            if agent and agent.org_id == org_id and agent.status == "active":
                return agent.id
            
            # Try name-based lookup
            agents = get_agents(org_id, status="active")
            
            # Exact name match
            exact_match = next((a for a in agents if a.name.lower() == agent_identifier.lower()), None)
            if exact_match:
                return exact_match.id
            
            # Partial name match
            partial_match = next((a for a in agents if agent_identifier.lower() in a.name.lower()), None)
            if partial_match:
                return partial_match.id
            
            # Description-based match
            desc_match = next((a for a in agents if agent_identifier.lower() in a.description.lower()), None)
            if desc_match:
                return desc_match.id
            
            raise ValueError(f"No agent found matching '{agent_identifier}'")
            
        except Exception as e:
            agent_logger.error(f"Agent resolution failed for '{agent_identifier}': {str(e)}")
            raise
    
    def _build_collaborate_prompt(self, agent_config: Dict[str, Any], user_request: str) -> str:
        """
        Build system prompt for COLLABORATE mode - interactive, discussion-focused
        
        Args:
            agent_config: Loaded agent configuration
            user_request: User's request for context
            
        Returns:
            Collaborate mode system prompt
        """
        try:
            system_prompt_data = agent_config.get("system_prompt", {})
            
            # Extract prompt components
            base_instruction = system_prompt_data.get("base_instruction", "")
            agent_type = system_prompt_data.get("agent_type", "general")
            language = system_prompt_data.get("language", "english")
            specialization = system_prompt_data.get("specialization", [])
            personality = system_prompt_data.get("personality", {})
            
            # Build collaborate-focused prompt
            collaborate_prompt = f"""You are {agent_config['name']}, operating in COLLABORATE mode.

{base_instruction}

AGENT PROFILE:
- Type: {agent_type.replace('_', ' ').title()} Agent
- Specialization: {', '.join(specialization) if specialization else 'General assistance'}
- Language: {language.title()}
- Personality: {personality.get('tone', 'professional')}, {personality.get('style', 'helpful')}, {personality.get('approach', 'solution-oriented')}

COLLABORATION APPROACH:
You are designed to work WITH the user in an interactive, collaborative manner:

🤝 COLLABORATION PRINCIPLES:
- Ask clarifying questions to better understand their needs
- Discuss options and alternatives before proceeding
- Explain your reasoning and approach
- Seek feedback and confirmation before taking major steps
- Be conversational and engage in back-and-forth dialogue
- Offer suggestions and recommendations, not just direct answers
- Help users think through problems step by step

AVAILABLE TOOLS:
You have access to these tools: {', '.join(agent_config.get('tools_list', []))}
Use tools to gather information and explore options, but DISCUSS findings with the user before drawing conclusions.

KNOWLEDGE ACCESS:
You can access these knowledge domains: {', '.join(agent_config.get('knowledge_list', [])) if agent_config.get('knowledge_list') else 'General knowledge base'}

INTERACTION STYLE:
- Start by understanding their current situation and goals
- Ask "What if..." and "Have you considered..." questions  
- Present multiple options when possible
- Explain trade-offs and implications
- Use phrases like "Let's explore...", "What do you think about...", "Would it help if..."
- Encourage the user to share their thoughts and preferences

CURRENT COLLABORATION:
The user has started this collaboration: "{user_request[:200]}{'...' if len(user_request) > 200 else ''}"

Let's work together to explore this thoroughly and find the best approach!"""

            agent_logger.info(f"Built collaborate mode prompt for {agent_config['name']} ({len(collaborate_prompt)} chars)")
            return collaborate_prompt
            
        except Exception as e:
            agent_logger.error(f"Failed to build collaborate prompt: {e}")
            return f"You are {agent_config.get('name', 'AI Agent')} in collaborative mode. Work with the user to explore their request: {user_request}"
    
    def _build_execute_prompt(self, agent_config: Dict[str, Any], user_request: str) -> str:
        """
        Build system prompt for EXECUTE mode - task-focused, efficient completion
        
        Args:
            agent_config: Loaded agent configuration
            user_request: User's request for context
            
        Returns:
            Execute mode system prompt
        """
        try:
            system_prompt_data = agent_config.get("system_prompt", {})
            
            # Extract prompt components
            base_instruction = system_prompt_data.get("base_instruction", "")
            agent_type = system_prompt_data.get("agent_type", "general")
            language = system_prompt_data.get("language", "english")
            specialization = system_prompt_data.get("specialization", [])
            personality = system_prompt_data.get("personality", {})
            
            # Build execution-focused prompt
            execute_prompt = f"""You are {agent_config['name']}, operating in EXECUTE mode.

{base_instruction}

AGENT PROFILE:
- Type: {agent_type.replace('_', ' ').title()} Agent
- Specialization: {', '.join(specialization) if specialization else 'General assistance'}
- Language: {language.title()}
- Personality: {personality.get('tone', 'professional')}, {personality.get('style', 'helpful')}, {personality.get('approach', 'solution-oriented')}

EXECUTION APPROACH:
You are designed to efficiently complete tasks and provide comprehensive results:

⚡ EXECUTION PRINCIPLES:
- Focus on completing the requested task efficiently
- Use tools directly when needed without extensive explanation
- Provide thorough, actionable results
- Minimize unnecessary back-and-forth questions
- Be direct and solution-focused
- Deliver comprehensive outputs that address the full request
- Take initiative to gather needed information

AVAILABLE TOOLS:
You have access to these tools: {', '.join(agent_config.get('tools_list', []))}
Use them efficiently to gather information, analyze data, and complete tasks.

KNOWLEDGE ACCESS:
You can access these knowledge domains: {', '.join(agent_config.get('knowledge_list', [])) if agent_config.get('knowledge_list') else 'General knowledge base'}

SPECIAL INSTRUCTIONS:
- PRIORITY: If the user provides a Google Drive link (docs.google.com), ALWAYS use the read_gdrive_link_docx or read_gdrive_link_pdf tool FIRST
- SPECIFICALLY: When you see URLs like "https://docs.google.com/document/d/..." or "https://drive.google.com/file/d/...", use read_gdrive_link_docx or read_gdrive_link_pdf immediately
- CRITICAL: When calling read_gdrive_link_docx or read_gdrive_link_pdf, you MUST extract the full URL from the user's request and pass it as the drive_link parameter
- FORCE: You MUST provide the drive_link parameter when calling these functions. The parameter cannot be empty.

- FOLDER READING: If the user asks to read or analyze an entire Google Drive folder, use the read_gdrive_folder tool
- PERMISSION ISSUES: If you get "File not found" or "Access denied" errors, use check_file_access tool to diagnose the issue

- If the user asks for analysis of a document, ALWAYS read the document content before providing any analysis
- Use the analyze_document or process_with_knowledge tool for business document analysis when appropriate
- CRITICAL: When calling process_with_knowledge, you MUST include user_id and org_id parameters
- Respond in {language} unless the user specifically requests another language
- DO NOT use search tools when a Google Drive link is provided - read the document directly

CURRENT TASK:
Execute this request efficiently: "{user_request[:200]}{'...' if len(user_request) > 200 else ''}"

Focus on delivering comprehensive, actionable results."""

            agent_logger.info(f"Built execute mode prompt for {agent_config['name']} ({len(execute_prompt)} chars)")
            return execute_prompt
            
        except Exception as e:
            agent_logger.error(f"Failed to build execute prompt: {e}")
            return f"You are {agent_config.get('name', 'AI Agent')} in execution mode. Complete this task: {user_request}"

    def _build_dynamic_system_prompt(self, agent_config: Dict[str, Any], user_request: str, agent_mode: str = "execute") -> str:
        """
        Build dynamic system prompt from agent configuration with mode support
        
        Args:
            agent_config: Loaded agent configuration
            user_request: User's request for context
            agent_mode: "collaborate" or "execute" mode
            
        Returns:
            Mode-specific system prompt string
        """
        try:
            if agent_mode.lower() == "collaborate":
                return self._build_collaborate_prompt(agent_config, user_request)
            else:
                return self._build_execute_prompt(agent_config, user_request)
                
        except Exception as e:
            agent_logger.error(f"Failed to build dynamic system prompt: {e}")
            # Fallback to basic prompt
            return f"You are {agent_config.get('name', 'AI Agent')}, a specialized AI assistant. {agent_config.get('description', 'I help with various tasks.')} Use your available tools to provide helpful assistance."
    
    def _determine_dynamic_tools(self, agent_config: Dict[str, Any], user_request: str) -> tuple[List[str], bool]:
        """
        Determine tools and force_tools setting based on agent config and request
        
        Args:
            agent_config: Loaded agent configuration
            user_request: User's request
            
        Returns:
            Tuple of (tools_whitelist, force_tools)
        """
        tools_list = agent_config.get("tools_list", [])
        
        # Check if this is a Google Drive request
        user_request_lower = user_request.lower()
        is_gdrive_request = (
            'docs.google.com' in user_request_lower or 
            'drive.google.com' in user_request_lower or
            'google drive' in user_request_lower or
            'gdrive' in user_request_lower or
            any(keyword in user_request_lower for keyword in ['folder', 'folders', 'directory', 'documents in', 'files in'])
        )
        
        # For Google Drive requests, ensure file_access tools are available
        if is_gdrive_request:
            if 'file_access' not in tools_list:
                tools_list = tools_list + ['file_access']
            if 'business_logic' not in tools_list:
                tools_list = tools_list + ['business_logic']
            force_tools = True
            agent_logger.info(f"Google Drive request detected - enhanced tools: {tools_list}")
        else:
            force_tools = False
        
        return tools_list, force_tools
    
    async def _load_knowledge_context(self, knowledge_list: List[str], user_request: str, user_id: str, org_id: str) -> str:
        """
        Load relevant knowledge context based on agent's knowledge list
        
        Args:
            knowledge_list: List of knowledge domains the agent has access to
            user_request: User's request for context
            user_id: User ID
            org_id: Organization ID
            
        Returns:
            Knowledge context string
        """
        if not knowledge_list:
            return ""
        
        try:
            # Use brain vector tool to search for relevant knowledge
            if "brain_vector" in self.available_tools:
                knowledge_context = ""
                for knowledge_domain in knowledge_list[:3]:  # Limit to first 3 domains to avoid token limit
                    try:
                        search_query = f"{user_request} {knowledge_domain}"
                        knowledge_results = await asyncio.to_thread(
                            self.available_tools["brain_vector"].query_knowledge,
                            user_id=user_id,
                            org_id=org_id,
                            query=search_query,
                            limit=5
                        )
                        
                        if knowledge_results:
                            knowledge_context += f"\n--- {knowledge_domain.replace('_', ' ').title()} Knowledge ---\n"
                            for result in knowledge_results[:2]:  # Top 2 results per domain
                                content = result.get('content', result.get('raw', ''))[:300]  # Truncate to avoid token limit
                                knowledge_context += f"• {content}...\n"
                    except Exception as e:
                        agent_logger.warning(f"Failed to load knowledge for domain {knowledge_domain}: {e}")
                        continue
                
                return knowledge_context
        except Exception as e:
            agent_logger.warning(f"Knowledge context loading failed: {e}")
        
        return ""
    
    async def _analyze_agent_task(self, request: AgentExecutionRequest) -> tuple[Dict[str, Any], list[str]]:
        """
        Analyze agent task for deep reasoning execution (different from Ami's teaching analysis)
        
        Args:
            request: The agent execution request
            
        Returns:
            Tuple of (task analysis, thinking steps)
        """
        
        # Create agent-specific analysis prompt
        analysis_prompt = f"""
        You are a specialized AI agent analyzing a task for execution. Think through this step-by-step.

        Task Request: "{request.user_request}"
        Agent Type: {request.agent_type}
        Agent ID: {request.agent_id}
        Specialized Domains: {request.specialized_knowledge_domains or ["general"]}
        
        Conversation History: {request.conversation_history[-3:] if request.conversation_history else "None"}

        Please provide your analysis in this EXACT format:

        TASK_UNDERSTANDING:
        [Clear understanding of what needs to be accomplished and the context]

        EXECUTION_THINKING:
        1. Analyzing the task requirements and my specialized capabilities...
        2. Identifying the key steps needed to complete this successfully...
        3. Determining which tools and knowledge areas I need to access...
        4. Planning my approach for optimal task completion...
        5. Considering potential challenges and how to address them...

        EXECUTION_ANALYSIS:
        {{
            "task_type": "information_retrieval|problem_solving|communication|analysis|automation",
            "complexity": "low|medium|high",
            "confidence": 0.85,
            "required_tools": ["search", "context", "brain_vector"],
            "knowledge_domains": ["domain1", "domain2"],
            "execution_approach": "Detailed description of how I'll complete this task",
            "estimated_effort": "low|medium|high"
        }}

        Task Classifications:
        - "information_retrieval": Need to find and synthesize information
        - "problem_solving": Need to analyze and solve a specific problem
        - "communication": Need to craft messages or communications
        - "analysis": Need to analyze data or situations
        - "automation": Need to perform automated tasks or workflows
        """
        
        try:
            # Use the same provider and model as the main request for task analysis
            if request.llm_provider.lower() == "anthropic":
                analysis_response = await self.anthropic_executor._analyze_with_anthropic(analysis_prompt, request.model)
            else:
                analysis_response = await self.openai_executor._analyze_with_openai(analysis_prompt, request.model)
            
            # Parse the response to extract components
            thinking_steps = []
            analysis_data = {}
            understanding_text = ""
            
            import re
            # Extract understanding section
            understanding_match = re.search(r'TASK_UNDERSTANDING:(.*?)EXECUTION_THINKING:', analysis_response, re.DOTALL)
            if understanding_match:
                understanding_text = understanding_match.group(1).strip()
                understanding_text = re.sub(r'\[.*?\]', '', understanding_text).strip()
            
            # Extract execution thinking section
            thinking_match = re.search(r'EXECUTION_THINKING:(.*?)EXECUTION_ANALYSIS:', analysis_response, re.DOTALL)
            if thinking_match:
                thinking_text = thinking_match.group(1).strip()
                steps = re.findall(r'\d+\.\s*(.*?)(?=\d+\.|$)', thinking_text, re.DOTALL)
                thinking_steps = [step.strip() for step in steps if step.strip()]
            
            # Extract JSON section
            json_match = re.search(r'\{.*\}', analysis_response, re.DOTALL)
            if json_match:
                try:
                    import json
                    analysis_data = json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse JSON from analysis response")
            
            # Create task analysis object
            task_analysis = {
                "task_type": analysis_data.get("task_type", "general"),
                "complexity": analysis_data.get("complexity", "medium"),
                "confidence": analysis_data.get("confidence", 0.5),
                "required_tools": analysis_data.get("required_tools", ["search"]),
                "knowledge_domains": analysis_data.get("knowledge_domains", ["general"]),
                "execution_approach": analysis_data.get("execution_approach", "Standard task execution approach"),
                "estimated_effort": analysis_data.get("estimated_effort", "medium"),
                "understanding": understanding_text,
                "detailed_thinking_steps": thinking_steps
            }
            
            return task_analysis, thinking_steps
            
        except Exception as e:
            logger.error(f"Agent task analysis failed: {e}")
            # Return default analysis and thinking steps
            default_thinking = [
                "Analyzing the task to understand what needs to be accomplished...",
                "Since detailed analysis failed, I'll use my general capabilities to assist...",
                "This appears to be a task I can handle with my available tools and knowledge...",
                "I'll proceed with a standard execution approach to complete this request..."
            ]
            
            return {
                "task_type": "general",
                "complexity": "medium",
                "confidence": 0.5,
                "required_tools": ["search"],
                "knowledge_domains": ["general"],
                "execution_approach": "Standard execution using available tools and knowledge",
                "estimated_effort": "medium",
                "understanding": f"Task: '{request.user_request[:100]}...' - Analysis failed, using defaults.",
                "detailed_thinking_steps": default_thinking
            }, default_thinking

    async def _generate_agent_thoughts(self, request: AgentExecutionRequest, task_analysis: Dict[str, Any], thinking_steps: List[str]) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Generate agent-style thoughts during task execution (different focus from Ami's teaching thoughts)
        
        Args:
            request: The agent execution request
            task_analysis: The task analysis results
            thinking_steps: Raw thinking steps from LLM analysis
            
        Yields:
            Dict containing agent thought messages focused on task execution
        """
        
        # 1. FIRST: Task understanding (agent-focused)
        understanding = task_analysis.get("understanding", "")
        if understanding:
            yield {
                "type": "thinking",
                "content": f"🎯 Task Analysis: {understanding}",
                "provider": request.llm_provider,
                "thought_type": "task_understanding",
                "agent_id": request.agent_id,
                "agent_type": request.agent_type,
                "step": 1,
                "timestamp": datetime.now().isoformat()
            }
        else:
            yield {
                "type": "thinking",
                "content": f"🎯 Processing task: \"{request.user_request[:80]}{'...' if len(request.user_request) > 80 else ''}\" with my specialized capabilities.",
                "provider": request.llm_provider,
                "thought_type": "task_understanding",
                "agent_id": request.agent_id,
                "agent_type": request.agent_type,
                "step": 1,
                "timestamp": datetime.now().isoformat()
            }
        
        await asyncio.sleep(0.4)
        
        # 2. SECOND: Execution approach (different from Ami's intent analysis)
        yield {
            "type": "thinking",
            "content": f"⚡ Execution Strategy: {task_analysis.get('execution_approach', 'Standard approach')} | Complexity: {task_analysis.get('complexity', 'medium')} | Confidence: {task_analysis.get('confidence', 0.5):.0%}",
            "provider": request.llm_provider,
            "thought_type": "execution_strategy",
            "agent_id": request.agent_id,
            "agent_type": request.agent_type,
            "step": 2,
            "timestamp": datetime.now().isoformat()
        }
        
        await asyncio.sleep(0.3)
        
        # 3. THIRD: Detailed execution thinking steps
        for i, step in enumerate(thinking_steps, start=3):
            yield {
                "type": "thinking",
                "content": f"🧠 {step}",
                "provider": request.llm_provider,
                "thought_type": "execution_analysis",
                "agent_id": request.agent_id,
                "agent_type": request.agent_type,
                "step": i,
                "timestamp": datetime.now().isoformat()
            }
            await asyncio.sleep(0.3)
        
        # 4. FOURTH: Tool activation (agent-focused)
        current_step = len(thinking_steps) + 3
        required_tools = task_analysis.get("required_tools", [])
        
        if required_tools and len(required_tools) > 0:
            tools_text = ", ".join([f"**{tool}**" for tool in required_tools])
            yield {
                "type": "thinking",
                "content": f"🛠️ Activating specialized tools: {tools_text} to complete this task efficiently...",
                "provider": request.llm_provider,
                "thought_type": "tool_activation",
                "agent_id": request.agent_id,
                "agent_type": request.agent_type,
                "step": current_step,
                "timestamp": datetime.now().isoformat()
            }
            await asyncio.sleep(0.4)
            current_step += 1
        
        # 5. FINALLY: Execution readiness
        yield {
            "type": "thinking",
            "content": "🚀 Beginning task execution with deep reasoning approach...",
            "provider": request.llm_provider,
            "thought_type": "execution_start",
            "agent_id": request.agent_id,
            "agent_type": request.agent_type,
            "step": current_step,
            "timestamp": datetime.now().isoformat()
        }

    async def _read_agent_specialized_knowledge(self, request: AgentExecutionRequest, task_context: str) -> List[Dict]:
        """Read agent's specialized knowledge from brain vectors"""
        
        brain_vectors = []
        try:
            if "brain_vector" in self.available_tools:
                # Query with agent's specialized domains and task context
                search_context = f"{task_context} {' '.join(request.specialized_knowledge_domains or [])}"
                brain_vectors = await asyncio.to_thread(
                    self.available_tools["brain_vector"].query_knowledge,
                    user_id=request.user_id,
                    org_id=request.org_id,
                    query=search_context,
                    limit=30  # Agents focus on most relevant knowledge
                )
            
        except Exception as e:
            logger.error(f"Failed to read agent specialized knowledge: {e}")
            brain_vectors = []
            
        return brain_vectors

    async def execute_agent_task_async(self, request: AgentExecutionRequest) -> AgentExecutionResponse:
        """
        Execute agent task asynchronously (different from Ami's execute_tool_async)
        
        Args:
            request: AgentExecutionRequest containing task parameters
            
        Returns:
            AgentExecutionResponse with execution results
        """
        start_time = datetime.now()
        
        try:
            # Validate provider
            if request.llm_provider.lower() not in ["anthropic", "openai"]:
                raise ValueError(f"Unsupported LLM provider: {request.llm_provider}")
            
            # Execute task using provider-specific executor (shared with Ami)
            if request.llm_provider.lower() == "anthropic":
                result = await self.anthropic_executor.execute(self._convert_to_tool_request(request))
            else:
                result = await self.openai_executor.execute(self._convert_to_tool_request(request))
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return AgentExecutionResponse(
                success=True,
                result=result,
                agent_id=request.agent_id,
                agent_type=request.agent_type,
                execution_time=execution_time,
                tasks_completed=1,
                metadata={
                    "org_id": request.org_id,
                    "user_id": request.user_id,
                    "tools_used": list(self.available_tools.keys()),
                    "specialized_domains": request.specialized_knowledge_domains
                }
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"Agent task execution failed: {str(e)}"
            traceback.print_exc()
            
            return AgentExecutionResponse(
                success=False,
                result="",
                agent_id=request.agent_id,
                agent_type=request.agent_type,
                execution_time=execution_time,
                tasks_completed=0,
                error=error_msg
            )

    async def execute_agent_task_stream(self, request: AgentExecutionRequest) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream agent task execution with dynamic configuration loading and multi-step planning
        
        Args:
            request: AgentExecutionRequest containing task parameters
            
        Yields:
            Dict containing streaming response data with dynamic agent behavior and multi-step execution
        """
        start_time = datetime.now()
        execution_id = f"{request.agent_id}-{start_time.strftime('%H%M%S')}"
        
        # Enhanced logging - execution start
        execution_logger.info(f"[{execution_id}] === DYNAMIC AGENT EXECUTION START ===")
        execution_logger.info(f"[{execution_id}] Agent Identifier: {request.agent_id} ({request.agent_type})")
        execution_logger.info(f"[{execution_id}] Provider: {request.llm_provider}")
        execution_logger.info(f"[{execution_id}] Model: {request.model or 'default'}")
        execution_logger.info(f"[{execution_id}] User Request: '{request.user_request[:100]}{'...' if len(request.user_request) > 100 else ''}'")
        execution_logger.info(f"[{execution_id}] Org ID: {request.org_id}, User ID: {request.user_id}")
        
        try:
            # Validate provider
            if request.llm_provider.lower() not in ["anthropic", "openai"]:
                execution_logger.error(f"[{execution_id}] VALIDATION FAILED - Unsupported provider: {request.llm_provider}")
                yield {
                    "type": "error",
                    "content": f"Unsupported LLM provider: {request.llm_provider}",
                    "provider": request.llm_provider,
                    "agent_id": request.agent_id,
                    "success": False
                }
                return
            
            execution_logger.info(f"[{execution_id}] VALIDATION PASSED - Provider {request.llm_provider} supported")
            
            # NEW: Load dynamic agent configuration
            execution_logger.info(f"[{execution_id}] LOADING AGENT CONFIG - Resolving agent identifier...")
            yield {
                "type": "status",
                "content": f"🔍 Loading configuration for {request.agent_id}...",
                "provider": request.llm_provider,
                "agent_id": request.agent_id,
                "status": "loading_config"
            }
            
            try:
                # Resolve agent identifier to actual agent ID
                resolved_agent_id = await self.resolve_agent_identifier(request.agent_id, request.org_id)
                execution_logger.info(f"[{execution_id}] AGENT RESOLVED - {request.agent_id} -> {resolved_agent_id}")
                
                # Load agent configuration from database
                agent_config = await self.load_agent_config(resolved_agent_id, request.org_id)
                execution_logger.info(f"[{execution_id}] CONFIG LOADED - Agent: {agent_config['name']}")
                execution_logger.info(f"[{execution_id}] CONFIG DETAILS - Tools: {agent_config.get('tools_list', [])}")
                execution_logger.info(f"[{execution_id}] CONFIG DETAILS - Knowledge: {agent_config.get('knowledge_list', [])}")
                
                # Update request with resolved agent ID and loaded config
                request.agent_id = resolved_agent_id
                # Store agent config for use in conversion
                self._current_agent_config = agent_config
                
                # NEW: Store current execution context for skill discovery
                self._current_user_id = request.user_id
                self._current_org_id = request.org_id
                agent_config["org_id"] = request.org_id  # Ensure org_id is in config
                
                yield {
                    "type": "status", 
                    "content": f"✅ Loaded {agent_config['name']} - ready to assist!",
                    "provider": request.llm_provider,
                    "agent_id": resolved_agent_id,
                    "agent_name": agent_config['name'],  
                    "status": "config_loaded"
                }
                
            except Exception as e:
                execution_logger.error(f"[{execution_id}] CONFIG LOADING FAILED - {str(e)}")
                yield {
                    "type": "error",
                    "content": f"❌ Failed to load agent configuration: {str(e)}",
                    "provider": request.llm_provider,
                    "agent_id": request.agent_id,
                    "success": False
                }
                return
            
            # NEW: Analyze task complexity for multi-step planning
            execution_logger.info(f"[{execution_id}] COMPLEXITY ANALYSIS - Analyzing task complexity...")
            yield {
                "type": "status",
                "content": "🧠 Analyzing task complexity...",
                "provider": request.llm_provider,
                "agent_id": request.agent_id,
                "status": "complexity_analysis"
            }
            
            try:
                complexity_analysis = await self.analyze_task_complexity(
                    request.user_request, agent_config, request.llm_provider, request.model
                )
                
                execution_logger.info(f"[{execution_id}] COMPLEXITY ANALYSIS - Score: {complexity_analysis.complexity_score}/10 ({complexity_analysis.complexity_level})")
                execution_logger.info(f"[{execution_id}] COMPLEXITY ANALYSIS - Required Steps: {complexity_analysis.required_steps}")
                execution_logger.info(f"[{execution_id}] COMPLEXITY ANALYSIS - Task Type: {complexity_analysis.task_type}")
                
                yield {
                    "type": "analysis",
                    "content": f"📊 Task Complexity: {complexity_analysis.complexity_score}/10 ({complexity_analysis.complexity_level}) - {complexity_analysis.required_steps} steps planned",
                    "provider": request.llm_provider,
                    "agent_id": request.agent_id,
                    "complexity_analysis": {
                        "score": complexity_analysis.complexity_score,
                        "level": complexity_analysis.complexity_level,
                        "steps": complexity_analysis.required_steps,
                        "task_type": complexity_analysis.task_type,
                        "confidence": complexity_analysis.confidence
                    }
                }
                
                # NEW: Multi-step execution for complex tasks
                system_prompt_data = agent_config.get("system_prompt", {})
                execution_capabilities = system_prompt_data.get("execution_capabilities", {})
                multi_step_threshold = execution_capabilities.get("multi_step_threshold", 4)
                supports_multi_step = execution_capabilities.get("supports_multi_step", True)
                
                if (complexity_analysis.complexity_score >= multi_step_threshold and 
                    supports_multi_step and 
                    complexity_analysis.complexity_level in ["standard", "complex"]):
                    
                    execution_logger.info(f"[{execution_id}] MULTI-STEP EXECUTION - Complexity {complexity_analysis.complexity_score} >= threshold {multi_step_threshold}")
                    
                    # Generate execution plan
                    yield {
                        "type": "status",
                        "content": f"📋 Generating {complexity_analysis.required_steps}-step execution plan...",
                        "provider": request.llm_provider,
                        "agent_id": request.agent_id,
                        "status": "planning"
                    }
                    
                    execution_plan = await self.generate_execution_plan(
                        complexity_analysis, request.user_request, agent_config, 
                        request.llm_provider, request.model
                    )
                    
                    # Store active execution plan
                    self.active_execution_plans[execution_id] = execution_plan
                    
                    yield {
                        "type": "plan",
                        "content": f"✅ Execution plan generated: {len(execution_plan.execution_steps)} steps, estimated time: {execution_plan.total_estimated_time}",
                        "provider": request.llm_provider,
                        "agent_id": request.agent_id,
                        "execution_plan": {
                            "plan_id": execution_plan.plan_id,
                            "total_steps": len(execution_plan.execution_steps),
                            "estimated_time": execution_plan.total_estimated_time,
                            "steps": [
                                {
                                    "step": step.step_number,
                                    "name": step.name,
                                    "estimated_time": step.estimated_time
                                } for step in execution_plan.execution_steps
                            ]
                        }
                    }
                    
                    # Execute multi-step plan
                    execution_logger.info(f"[{execution_id}] MULTI-STEP EXECUTION - Starting step-by-step execution")
                    
                    async for step_result in self._execute_multi_step_plan(execution_plan, request, agent_config):
                        yield step_result
                    
                    # Multi-step execution complete
                    execution_time = (datetime.now() - start_time).total_seconds()
                    execution_logger.info(f"[{execution_id}] MULTI-STEP EXECUTION COMPLETE - {execution_time:.2f}s")
                    
                    yield {
                        "type": "complete",
                        "content": f"✅ Multi-step execution completed successfully ({len(execution_plan.execution_steps)} steps)",
                        "provider": request.llm_provider,
                        "agent_id": request.agent_id,
                        "agent_type": request.agent_type,
                        "execution_time": execution_time,
                        "execution_mode": "multi_step",
                        "success": True
                    }
                    
                    return
                    
                else:
                    execution_logger.info(f"[{execution_id}] SINGLE-STEP EXECUTION - Complexity {complexity_analysis.complexity_score} < threshold {multi_step_threshold} or multi-step disabled")
                    yield {
                        "type": "status",
                        "content": f"⚡ Executing with single-step approach (complexity: {complexity_analysis.complexity_level})",
                        "provider": request.llm_provider,
                        "agent_id": request.agent_id,
                        "status": "single_step_execution"
                    }
                
            except Exception as e:
                execution_logger.warning(f"[{execution_id}] COMPLEXITY ANALYSIS FAILED - {str(e)}, proceeding with standard execution")
                yield {
                    "type": "status",
                    "content": "⚠️ Complexity analysis failed, proceeding with standard execution",
                    "provider": request.llm_provider,
                    "agent_id": request.agent_id,
                    "status": "fallback_execution"
                }
            
            # Convert to tool request with dynamic configuration (existing logic)
            execution_logger.info(f"[{execution_id}] TOOL CONVERSION - Converting with dynamic config")
            tool_request = self._convert_to_tool_request_dynamic(request, agent_config)
            execution_logger.info(f"[{execution_id}] DYNAMIC CONVERSION COMPLETE - Tools: {tool_request.tools_whitelist}")
            
            # Log dynamic system prompt (truncated for readability)
            system_prompt_preview = tool_request.system_prompt[:200] + "..." if len(tool_request.system_prompt) > 200 else tool_request.system_prompt
            execution_logger.info(f"[{execution_id}] DYNAMIC SYSTEM PROMPT: {system_prompt_preview}")
            
            # Execute with dynamic configuration (existing logic)
            execution_logger.info(f"[{execution_id}] LLM EXECUTION START - Provider: {request.llm_provider}")
            
            chunk_count = 0
            tool_calls_detected = 0
            response_chunks = 0
            
            if request.llm_provider.lower() == "anthropic":
                async for chunk in self.anthropic_executor.execute_stream(tool_request):
                    chunk_count += 1
                    
                    # Log different chunk types
                    chunk_type = chunk.get("type", "unknown")
                    if chunk_type == "tool_execution":
                        tool_calls_detected += 1
                        tool_name = chunk.get("tool_name", "unknown")
                        execution_logger.info(f"[{execution_id}] TOOL CALL #{tool_calls_detected} - {tool_name}: {chunk.get('content', '')}")
                    elif chunk_type == "response_chunk":
                        response_chunks += 1
                        if response_chunks <= 5:  # Log first 5 response chunks
                            execution_logger.info(f"[{execution_id}] RESPONSE CHUNK #{response_chunks}: '{chunk.get('content', '')[:50]}{'...' if len(chunk.get('content', '')) > 50 else ''}'")
                    elif chunk_type in ["thinking", "analysis", "status"]:
                        execution_logger.info(f"[{execution_id}] {chunk_type.upper()}: {chunk.get('content', '')[:100]}{'...' if len(chunk.get('content', '')) > 100 else ''}")
                    
                    # Add agent context to chunks
                    chunk["agent_id"] = request.agent_id
                    chunk["agent_type"] = request.agent_type
                    yield chunk
            else:
                async for chunk in self.openai_executor.execute_stream(tool_request):
                    chunk_count += 1
                    
                    # Log different chunk types
                    chunk_type = chunk.get("type", "unknown")
                    if chunk_type == "tool_execution":
                        tool_calls_detected += 1
                        tool_name = chunk.get("tool_name", "unknown")
                        execution_logger.info(f"[{execution_id}] TOOL CALL #{tool_calls_detected} - {tool_name}: {chunk.get('content', '')}")
                    elif chunk_type == "response_chunk":
                        response_chunks += 1
                        if response_chunks <= 5:  # Log first 5 response chunks
                            execution_logger.info(f"[{execution_id}] RESPONSE CHUNK #{response_chunks}: '{chunk.get('content', '')[:50]}{'...' if len(chunk.get('content', '')) > 50 else ''}'")
                    elif chunk_type in ["thinking", "analysis", "status"]:
                        execution_logger.info(f"[{execution_id}] {chunk_type.upper()}: {chunk.get('content', '')[:100]}{'...' if len(chunk.get('content', '')) > 100 else ''}")
                    
                    # Add agent context to chunks
                    chunk["agent_id"] = request.agent_id
                    chunk["agent_type"] = request.agent_type
                    yield chunk
            
            # Simple completion status
            execution_time = (datetime.now() - start_time).total_seconds()
            execution_logger.info(f"[{execution_id}] LLM EXECUTION COMPLETE - Processed {chunk_count} chunks")
            execution_logger.info(f"[{execution_id}] EXECUTION SUMMARY:")
            execution_logger.info(f"[{execution_id}] - Total chunks: {chunk_count}")
            execution_logger.info(f"[{execution_id}] - Tool calls: {tool_calls_detected}")
            execution_logger.info(f"[{execution_id}] - Response chunks: {response_chunks}")
            execution_logger.info(f"[{execution_id}] - Execution time: {execution_time:.2f}s")
            execution_logger.info(f"[{execution_id}] === AGENT EXECUTION SUCCESS ===")
            
            yield {
                "type": "complete",
                "content": f"✅ {request.agent_id} completed successfully",
                "provider": request.llm_provider,
                "agent_id": request.agent_id,
                "agent_type": request.agent_type,
                "execution_time": execution_time,
                "execution_mode": "single_step",
                "success": True
            }
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"Agent execution failed: {str(e)}"
            
            execution_logger.error(f"[{execution_id}] === AGENT EXECUTION FAILED ===")
            execution_logger.error(f"[{execution_id}] Error: {error_msg}")
            execution_logger.error(f"[{execution_id}] Execution time: {execution_time:.2f}s")
            execution_logger.error(f"[{execution_id}] Exception details: {str(e)}")
            import traceback
            execution_logger.error(f"[{execution_id}] Traceback: {traceback.format_exc()}")
            
            yield {
                "type": "error",
                "content": error_msg,
                "provider": request.llm_provider,
                "agent_id": request.agent_id,
                "agent_type": request.agent_type,
                "execution_time": execution_time,
                "success": False
            }



    def _convert_to_tool_request_dynamic(self, agent_request: AgentExecutionRequest, agent_config: Dict[str, Any]):
        """Convert AgentExecutionRequest to ToolExecutionRequest with dynamic configuration"""
        from exec_tool import ToolExecutionRequest
        
        # Build dynamic system prompt from agent configuration with mode support
        dynamic_system_prompt = self._build_dynamic_system_prompt(agent_config, agent_request.user_request, agent_request.agent_mode)
        
        # Determine dynamic tool settings
        tools_whitelist, force_tools = self._determine_dynamic_tools(agent_config, agent_request.user_request)
        
        execution_logger.info(f"[{agent_request.agent_id}] DYNAMIC CONFIG APPLIED:")
        execution_logger.info(f"[{agent_request.agent_id}] - Agent Name: {agent_config['name']}")
        execution_logger.info(f"[{agent_request.agent_id}] - Tools: {tools_whitelist}")
        execution_logger.info(f"[{agent_request.agent_id}] - Force Tools: {force_tools}")
        execution_logger.info(f"[{agent_request.agent_id}] - Knowledge Domains: {agent_config.get('knowledge_list', [])}")
        
        return ToolExecutionRequest(
            llm_provider=agent_request.llm_provider,
            user_query=agent_request.user_request,
            system_prompt=dynamic_system_prompt,
            model=agent_request.model,
            model_params=agent_request.model_params,
            org_id=agent_request.org_id,
            user_id=agent_request.user_id,
            enable_tools=agent_request.enable_tools,
            force_tools=force_tools,
            tools_whitelist=tools_whitelist,
            conversation_history=agent_request.conversation_history,
            max_history_messages=agent_request.max_history_messages,
            max_history_tokens=agent_request.max_history_tokens,
            # Simplified settings for clean agent execution
            enable_deep_reasoning=False,  
            reasoning_depth="light",      
            enable_intent_classification=False,  
            enable_request_analysis=False,      
            cursor_mode=False
        )
    
    def _convert_to_tool_request(self, agent_request: AgentExecutionRequest):
        """Legacy method - kept for backward compatibility"""
        # If agent config is available, use dynamic conversion
        if hasattr(self, '_current_agent_config') and self._current_agent_config:
            return self._convert_to_tool_request_dynamic(agent_request, self._current_agent_config)
        
        # Fallback to original implementation for backward compatibility
        from exec_tool import ToolExecutionRequest
        
        # Create basic agent system prompt
        agent_system_prompt = f"""You are {agent_request.agent_id}, a specialized {agent_request.agent_type}.

Your specialization areas: {', '.join(agent_request.specialized_knowledge_domains or ['general'])}

IMPORTANT: You have access to tools when needed. Use them naturally as part of your reasoning process to provide helpful, accurate responses.

SPECIAL INSTRUCTIONS:
- PRIORITY: If the user provides a Google Drive link (docs.google.com), ALWAYS use the read_gdrive_link_docx or read_gdrive_link_pdf tool FIRST to read the document content
- SPECIFICALLY: When you see URLs like "https://docs.google.com/document/d/..." or "https://drive.google.com/file/d/...", use read_gdrive_link_docx or read_gdrive_link_pdf immediately
- CRITICAL: When calling read_gdrive_link_docx or read_gdrive_link_pdf, you MUST extract the full URL from the user's request and pass it as the drive_link parameter
- EXAMPLE: If user says "analyse this: https://docs.google.com/document/d/ABC123/edit", call read_gdrive_link_docx with drive_link="https://docs.google.com/document/d/ABC123/edit"
- FORCE: You MUST provide the drive_link parameter when calling these functions. The parameter cannot be empty.

- FOLDER READING: If the user asks to read or analyze an entire Google Drive folder, use the read_gdrive_folder tool
- FOLDER EXAMPLES: 
  * "read all documents in the Reports folder" → use read_gdrive_folder(folder_name="Reports")
  * "analyze all PDFs in the Sales folder" → use read_gdrive_folder(folder_name="Sales", file_types=["pdf"])
  * "read all files from folder ID 1ABC123..." → use read_gdrive_folder(folder_id="1ABC123...")
- FOLDER BENEFITS: read_gdrive_folder reads all supported files (DOCX, PDF) and returns combined content for comprehensive analysis
- TOKEN LIMITS: If folder content is too large, use max_chars parameter to limit content size and avoid token limit errors
- EXAMPLE: read_gdrive_folder(folder_name="Large Reports", max_chars=30000) for smaller content
- PERMISSION ISSUES: If you get "File not found" or "Access denied" errors, use check_file_access tool to diagnose the issue
- EXAMPLE: check_file_access(drive_link="https://docs.google.com/document/d/ABC123/edit") to check file permissions

- If the user asks for analysis of a document, ALWAYS read the document content before providing any analysis
- Use the analyze_document or process_with_knowledge tool for business document analysis when appropriate
- CRITICAL: When calling process_with_knowledge, you MUST include user_id="{agent_request.user_id}" and org_id="{agent_request.org_id}" parameters
- EXAMPLE: process_with_knowledge(document_content="...", knowledge_query="how to analyze business plans", user_id="{agent_request.user_id}", org_id="{agent_request.org_id}")
- Respond in English unless the user specifically requests another language
- DO NOT use search tools when a Google Drive link is provided - read the document directly

Task Focus: {agent_request.task_focus}"""
        
        # Check if this is a Google Drive document analysis request
        user_request_lower = agent_request.user_request.lower()
        is_gdrive_request = (
            'docs.google.com' in user_request_lower or 
            'drive.google.com' in user_request_lower or
            'google drive' in user_request_lower or
            'gdrive' in user_request_lower or
            any(keyword in user_request_lower for keyword in ['folder', 'folders', 'directory', 'documents in', 'files in'])
        )
        
        # If it's a Google Drive request, whitelist only the relevant tools and force tool usage
        tools_whitelist = None
        force_tools = False
        if is_gdrive_request:
            tools_whitelist = ['file_access', 'business_logic']
            force_tools = True  # Force the LLM to use tools
            execution_logger.info(f"[{agent_request.agent_id}] DETECTED Google Drive request - restricting tools to: {tools_whitelist} and forcing tool usage")
        
        return ToolExecutionRequest(
            llm_provider=agent_request.llm_provider,
            user_query=agent_request.user_request,
            system_prompt=agent_system_prompt,
            model=agent_request.model,
            model_params=agent_request.model_params,
            org_id=agent_request.org_id,
            user_id=agent_request.user_id,
            enable_tools=agent_request.enable_tools,
            force_tools=force_tools,  # Use our force_tools setting for Google Drive requests
            tools_whitelist=tools_whitelist,  # Use our whitelist for Google Drive requests
            conversation_history=agent_request.conversation_history,
            max_history_messages=agent_request.max_history_messages,
            max_history_tokens=agent_request.max_history_tokens,
            # Simplified settings - let LLM handle reasoning
            enable_deep_reasoning=False,  
            reasoning_depth="light",      
            enable_intent_classification=False,  
            enable_request_analysis=False,      
            cursor_mode=False  # Disable complex cursor mode for clean flow
        )



    async def _detect_language_and_create_prompt(self, request, user_query: str, base_prompt: str) -> str:
        """
        Detect the language of user query and create a language-aware system prompt
        (Simplified version for agents - avoids circular dependency)
        
        Args:
            request: The tool execution request containing LLM provider and model
            user_query: The user's input query
            base_prompt: The base system prompt to enhance
            
        Returns:
            Language-aware system prompt
        """
        if not LANGUAGE_DETECTION_AVAILABLE:
            logger.warning("Language detection not available, using default prompt")
            return base_prompt
        
        try:
            # For agents, we'll use a simpler language detection approach
            # to avoid circular dependency with executors
            
            # Simple language detection based on common patterns
            vietnamese_patterns = ['em', 'anh', 'chị', 'ạ', 'ơi', 'được', 'làm', 'gì', 'như', 'thế', 'nào']
            french_patterns = ['je', 'tu', 'il', 'elle', 'nous', 'vous', 'ils', 'elles', 'le', 'la', 'les']
            spanish_patterns = ['yo', 'tú', 'él', 'ella', 'nosotros', 'vosotros', 'ellos', 'ellas', 'el', 'la', 'los', 'las']
            
            user_query_lower = user_query.lower()
            
            # Check for Vietnamese
            vietnamese_matches = sum(1 for pattern in vietnamese_patterns if pattern in user_query_lower)
            if vietnamese_matches >= 2:
                enhanced_prompt = f"""{base_prompt}

IMPORTANT LANGUAGE INSTRUCTION:
The user is communicating in Vietnamese. Please respond in Vietnamese naturally and fluently.

Remember to:
1. Respond in Vietnamese naturally and fluently
2. Use appropriate Vietnamese cultural context and expressions
3. Maintain the same tone and formality level as the user
4. If you need to clarify something, ask in Vietnamese
"""
                logger.info("Enhanced prompt with Vietnamese language guidance")
                return enhanced_prompt
            
            # Check for French (but be more careful about URLs and technical terms)
            french_matches = sum(1 for pattern in french_patterns if pattern in user_query_lower)
            
            # Don't trigger French mode if the query contains URLs or technical terms
            has_url = 'http' in user_query_lower or 'docs.google.com' in user_query_lower
            has_technical_terms = any(term in user_query_lower for term in ['analyse', 'analyze', 'document', 'file', 'link', 'url'])
            
            if french_matches >= 2 and not has_url and not has_technical_terms:
                enhanced_prompt = f"""{base_prompt}

IMPORTANT LANGUAGE INSTRUCTION:
The user is communicating in French. Please respond in French naturally and fluently.

Remember to:
1. Respond in French naturally and fluently
2. Use appropriate French cultural context and expressions
3. Maintain the same tone and formality level as the user
4. If you need to clarify something, ask in French
"""
                logger.info("Enhanced prompt with French language guidance")
                return enhanced_prompt
            
            # Check for Spanish
            spanish_matches = sum(1 for pattern in spanish_patterns if pattern in user_query_lower)
            if spanish_matches >= 2:
                enhanced_prompt = f"""{base_prompt}

IMPORTANT LANGUAGE INSTRUCTION:
The user is communicating in Spanish. Please respond in Spanish naturally and fluently.

Remember to:
1. Respond in Spanish naturally and fluently
2. Use appropriate Spanish cultural context and expressions
3. Maintain the same tone and formality level as the user
4. If you need to clarify something, ask in Spanish
"""
                logger.info("Enhanced prompt with Spanish language guidance")
                return enhanced_prompt
            
            # Default to English
            logger.info("Using base prompt (English or undetected language)")
            return base_prompt
        
        except Exception as e:
            logger.error(f"Language detection error: {e}")
            # Return base prompt if language detection fails
            return base_prompt

    async def _generate_response_thoughts(self, provider: str, has_tool_results: bool = False) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Generate thoughts about response generation (simplified for agents)
        
        Args:
            provider: LLM provider name
            has_tool_results: Whether tool results are available
            
        Yields:
            Dict containing response generation thoughts
        """
        
        if has_tool_results:
            yield {
                "type": "thinking",
                "content": "✅ Task analysis complete! Now I'll synthesize everything into a comprehensive response...",
                "provider": provider,
                "thought_type": "response_generation",
                "timestamp": datetime.now().isoformat()
            }
        else:
            yield {
                "type": "thinking",
                "content": "💡 I'll use my specialized knowledge to provide a helpful response...",
                "provider": provider,
                "thought_type": "response_generation",
                "timestamp": datetime.now().isoformat()
            }
        
        await asyncio.sleep(0.3)
        
        yield {
            "type": "thinking",
            "content": "✍️ Generating agent response now...",
            "provider": provider,
            "thought_type": "response_generation",
            "timestamp": datetime.now().isoformat()
        }

    def get_available_tools(self) -> List[str]:
        """Get list of available tools (shared with Ami)"""
        return list(self.available_tools.keys())

    # NEW: Multi-Step Planning Methods
    async def analyze_task_complexity(self, user_request: str, agent_config: Dict[str, Any], 
                                        llm_provider: str, model: str = None) -> TaskComplexityAnalysis:
        """
        Analyze task complexity to determine if multi-step planning is needed
        WITH skill discovery and knowledge-aware enhancement
        
        Args:
            user_request: The user's task request
            agent_config: Agent configuration with capabilities
            llm_provider: LLM provider for analysis
            model: Optional model name
            
        Returns:
            TaskComplexityAnalysis with complexity scoring and recommendations enhanced by discovered skills
        """
        
        # Get agent's execution capabilities
        system_prompt_data = agent_config.get("system_prompt", {})
        execution_capabilities = system_prompt_data.get("execution_capabilities", {})
        max_complexity = execution_capabilities.get("max_complexity_score", 10)
        complexity_thresholds = execution_capabilities.get("complexity_thresholds", {
            "simple": {"min": 1, "max": 3, "steps": 3},
            "standard": {"min": 4, "max": 7, "steps": 5},
            "complex": {"min": 8, "max": 10, "steps": 7}
        })
        
        # NEW: Discover relevant skills from agent's knowledge base
        discovered_skills = {}
        try:
            # Extract user_id and org_id from current context or agent config
            user_id = getattr(self, '_current_user_id', 'anonymous')
            org_id = agent_config.get("org_id", "default")
            
            discovered_skills = await self.discover_agent_skills(
                user_request, agent_config, user_id, org_id
            )
            
            # Log discovered skills
            if discovered_skills.get("skills") or discovered_skills.get("methodologies"):
                agent_logger.info(f"Discovered {len(discovered_skills.get('skills', []))} skills, {len(discovered_skills.get('methodologies', []))} methodologies")
                
        except Exception as e:
            agent_logger.warning(f"Skill discovery failed during complexity analysis: {e}")
            discovered_skills = {"skills": [], "methodologies": [], "experience": [], "capabilities_enhancement": {}}
        
        # Create complexity analysis prompt enhanced with discovered skills
        skills_context = ""
        if discovered_skills.get("skills") or discovered_skills.get("methodologies") or discovered_skills.get("frameworks"):
            skills_context = f"""
DISCOVERED AGENT SKILLS & EXPERIENCE:
- Skills: {', '.join(discovered_skills.get('skills', [])[:3])}
- Methodologies: {', '.join(discovered_skills.get('methodologies', [])[:3])}
- Frameworks: {', '.join(discovered_skills.get('frameworks', [])[:3])}
- Best Practices: {', '.join(discovered_skills.get('best_practices', [])[:2])}

CAPABILITY ENHANCEMENT:
- Complexity Boost: +{discovered_skills.get('capabilities_enhancement', {}).get('complexity_boost', 0)} points
- Additional Capabilities: {len(discovered_skills.get('capabilities_enhancement', {}).get('additional_steps', []))} specialized steps available
"""
        
        analysis_prompt = f"""
        You are an expert task complexity analyzer with access to the agent's specific knowledge and skills. 
        Analyze this task request and determine its complexity level, considering the agent's discovered capabilities.

        AGENT CAPABILITIES:
        - Agent Name: {agent_config.get('name', 'Unknown')}
        - Agent Type: {system_prompt_data.get('agent_type', 'general')}
        - Available Tools: {', '.join(agent_config.get('tools_list', []))}
        - Knowledge Domains: {', '.join(agent_config.get('knowledge_list', []))}
        - Max Complexity Score: {max_complexity}
        - Domain Specialization: {system_prompt_data.get('domain_specialization', ['general'])}

        {skills_context}

        TASK REQUEST: "{user_request}"

        COMPLEXITY SCORING CRITERIA (Enhanced with Skills):
        1-3 (Simple): Single tool usage, direct information retrieval, straightforward Q&A
        4-7 (Standard): Multiple tool coordination, analysis tasks, document processing
        8-10 (Complex): Multi-step reasoning, cross-domain analysis, complex problem solving
        
        SKILL-BASED ADJUSTMENTS:
        - If agent has relevant methodologies/frameworks: +1-2 complexity points (can handle more complex tasks)
        - If agent has specific experience: More sophisticated execution steps
        - If agent has best practices: Enhanced quality standards

        Consider the agent's discovered skills when scoring. An agent with relevant experience can handle 
        more complex approaches to tasks that might otherwise be simpler.

        Provide your analysis in this EXACT JSON format:
        {{
            "complexity_score": 6,
            "complexity_level": "standard",
            "task_type": "information_retrieval|problem_solving|analysis|communication|automation",
            "required_steps": 5,
            "estimated_duration": "short|medium|long",
            "key_challenges": ["challenge1", "challenge2"],
            "recommended_approach": "Detailed description of recommended execution approach incorporating discovered skills",
            "confidence": 0.85,
            "skill_integration": "How discovered skills enhance the execution approach",
            "reasoning": "Explanation of complexity scoring including skill-based adjustments"
        }}
        """
        
        try:
            # Get complexity analysis from LLM
            if llm_provider.lower() == "anthropic":
                analysis_response = await self.anthropic_executor._analyze_with_anthropic(analysis_prompt, model)
            else:
                analysis_response = await self.openai_executor._analyze_with_openai(analysis_prompt, model)
            
            # Parse JSON response
            import json
            import re
            json_match = re.search(r'\{.*\}', analysis_response, re.DOTALL)
            if json_match:
                analysis_data = json.loads(json_match.group(0))
                
                # Apply skill-based complexity boost
                base_complexity_score = analysis_data.get("complexity_score", 5)
                skill_boost = discovered_skills.get("capabilities_enhancement", {}).get("complexity_boost", 0)
                complexity_score = min(base_complexity_score + skill_boost, max_complexity)
                
                # Determine complexity level based on thresholds (with skill enhancement)
                complexity_level = "standard"
                required_steps = 5
                
                for level, threshold in complexity_thresholds.items():
                    if threshold["min"] <= complexity_score <= threshold["max"]:
                        complexity_level = level
                        required_steps = threshold["steps"]
                        break
                
                # Add skill-based additional steps
                additional_steps = len(discovered_skills.get("capabilities_enhancement", {}).get("additional_steps", []))
                if additional_steps > 0:
                    required_steps += additional_steps
                
                # Enhanced recommended approach
                base_approach = analysis_data.get("recommended_approach", "Standard execution approach")
                if discovered_skills.get("methodologies"):
                    base_approach += f" Enhanced with discovered methodologies: {', '.join(discovered_skills['methodologies'][:2])}"
                
                return TaskComplexityAnalysis(
                    complexity_score=complexity_score,
                    complexity_level=complexity_level,
                    task_type=analysis_data.get("task_type", "general"),
                    required_steps=required_steps,
                    estimated_duration=analysis_data.get("estimated_duration", "medium"),
                    key_challenges=analysis_data.get("key_challenges", []),
                    recommended_approach=base_approach,
                    confidence=analysis_data.get("confidence", 0.7)
                )
            else:
                # Fallback to heuristic-based analysis with skill enhancement
                return self._heuristic_complexity_analysis_with_skills(user_request, agent_config, discovered_skills)
                
        except Exception as e:
            agent_logger.warning(f"LLM complexity analysis failed: {e}, using heuristic analysis")
            return self._heuristic_complexity_analysis_with_skills(user_request, agent_config, discovered_skills)
    
    def _heuristic_complexity_analysis_with_skills(self, user_request: str, agent_config: Dict[str, Any], 
                                                 discovered_skills: Dict[str, Any]) -> TaskComplexityAnalysis:
        """Fallback heuristic-based complexity analysis enhanced with discovered skills"""
        
        request_lower = user_request.lower()
        complexity_indicators = {
            'simple': ['what', 'who', 'when', 'where', 'define', 'explain', 'list'],
            'standard': ['analyze', 'compare', 'evaluate', 'process', 'review', 'optimize'],
            'complex': ['integrate', 'coordinate', 'transform', 'redesign', 'implement', 'multiple', 'comprehensive']
        }
        
        # Count indicators
        simple_count = sum(1 for indicator in complexity_indicators['simple'] if indicator in request_lower)
        standard_count = sum(1 for indicator in complexity_indicators['standard'] if indicator in request_lower)
        complex_count = sum(1 for indicator in complexity_indicators['complex'] if indicator in request_lower)
        
        # Determine base complexity
        if complex_count >= 2 or len(user_request) > 200:
            complexity_score = 8
            complexity_level = "complex"
            required_steps = 7
        elif standard_count >= 1 or simple_count == 0:
            complexity_score = 5
            complexity_level = "standard"
            required_steps = 5
        else:
            complexity_score = 3
            complexity_level = "simple"
            required_steps = 3
        
        # Apply skill-based enhancements
        skill_boost = discovered_skills.get("capabilities_enhancement", {}).get("complexity_boost", 0)
        complexity_score = min(complexity_score + skill_boost, 10)
        
        # Add skill-based steps
        additional_steps = len(discovered_skills.get("capabilities_enhancement", {}).get("additional_steps", []))
        required_steps += additional_steps
        
        # Update complexity level based on enhanced score
        if complexity_score >= 8:
            complexity_level = "complex"
        elif complexity_score >= 4:
            complexity_level = "standard"
        
        # Enhanced approach description
        approach = "Standard execution with complexity-appropriate steps"
        if discovered_skills.get("methodologies"):
            approach += f" enhanced with {len(discovered_skills['methodologies'])} discovered methodologies"
        
        return TaskComplexityAnalysis(
            complexity_score=complexity_score,
            complexity_level=complexity_level,
            task_type="general",
            required_steps=required_steps,
            estimated_duration="medium",
            key_challenges=["Heuristic analysis - enhanced with discovered skills"],
            recommended_approach=approach,
            confidence=0.7
        )
    
    async def generate_execution_plan(self, complexity_analysis: TaskComplexityAnalysis, 
                                    user_request: str, agent_config: Dict[str, Any],
                                    llm_provider: str, model: str = None) -> MultiStepExecutionPlan:
        """
        Generate detailed multi-step execution plan based on complexity analysis
        WITH discovered skills and knowledge integration
        
        Args:
            complexity_analysis: Task complexity analysis
            user_request: Original user request
            agent_config: Agent configuration
            llm_provider: LLM provider
            model: Optional model name
            
        Returns:
            MultiStepExecutionPlan with detailed execution steps enhanced by agent's discovered skills
        """
        
        plan_id = f"plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # NEW: Re-discover skills for detailed planning (with caching potential)
        discovered_skills = {}
        try:
            user_id = getattr(self, '_current_user_id', 'anonymous')
            org_id = agent_config.get("org_id", "default")
            
            discovered_skills = await self.discover_agent_skills(
                user_request, agent_config, user_id, org_id
            )
        except Exception as e:
            agent_logger.warning(f"Skill discovery failed during planning: {e}")
            discovered_skills = {"skills": [], "methodologies": [], "experience": [], "capabilities_enhancement": {}}
        
        # Create execution plan generation prompt enhanced with skills
        system_prompt_data = agent_config.get("system_prompt", {})
        available_tools = agent_config.get("tools_list", [])
        knowledge_domains = agent_config.get("knowledge_list", [])
        
        # Build skill-aware planning context
        skills_planning_context = ""
        if discovered_skills.get("skills") or discovered_skills.get("methodologies"):
            skills_planning_context = f"""
AGENT'S DISCOVERED SKILLS & METHODOLOGIES TO INTEGRATE:

SPECIFIC SKILLS:
{chr(10).join(f"- {skill}" for skill in discovered_skills.get('skills', [])[:5])}

METHODOLOGIES & FRAMEWORKS:
{chr(10).join(f"- {method}" for method in discovered_skills.get('methodologies', [])[:5])}

ORGANIZATIONAL EXPERIENCE:
{chr(10).join(f"- {exp}" for exp in discovered_skills.get('experience', [])[:3])}

AVAILABLE FRAMEWORKS:
{chr(10).join(f"- {framework}" for framework in discovered_skills.get('frameworks', [])[:3])}

BEST PRACTICES TO APPLY:
{chr(10).join(f"- {practice}" for practice in discovered_skills.get('best_practices', [])[:3])}

SKILL-BASED ENHANCEMENTS:
- Additional specialized steps: {len(discovered_skills.get('capabilities_enhancement', {}).get('additional_steps', []))}
- Enhanced quality standards: {len(discovered_skills.get('capabilities_enhancement', {}).get('quality_standards', []))}
"""
        
        planning_prompt = f"""
        You are an expert execution planner for AI agents with access to the agent's specific skills and organizational knowledge.
        Create a detailed multi-step execution plan that leverages the agent's discovered capabilities.

        AGENT CONTEXT:
        - Agent: {agent_config.get('name')} ({system_prompt_data.get('agent_type', 'general')})
        - Available Tools: {', '.join(available_tools)}
        - Knowledge Domains: {', '.join(knowledge_domains)}
        - Domain Specialization: {system_prompt_data.get('domain_specialization', ['general'])}

        {skills_planning_context}

        TASK REQUEST: "{user_request}"

        COMPLEXITY ANALYSIS:
        - Score: {complexity_analysis.complexity_score}/10 ({complexity_analysis.complexity_level})
        - Required Steps: {complexity_analysis.required_steps}
        - Task Type: {complexity_analysis.task_type}
        - Key Challenges: {', '.join(complexity_analysis.key_challenges)}
        - Recommended Approach: {complexity_analysis.recommended_approach}

        SKILL-AWARE EXECUTION FRAMEWORK:
        Create {complexity_analysis.required_steps} detailed execution steps following this enhanced pattern:
        1. Context Analysis & Knowledge Activation (activate relevant agent knowledge)
        2. Information Gathering with Skill Application (use discovered methodologies)
        3. Analysis & Processing with Best Practices (apply organizational experience)
        4. Framework-Based Solution Generation (use discovered frameworks)
        5. Experience-Validated Synthesis (validate against organizational lessons learned)
        (+ additional specialized steps based on discovered capabilities)

        IMPORTANT: Incorporate the agent's discovered skills and methodologies into specific steps.
        For example:
        - If agent has "Lean methodology" → include "Apply Lean principles to eliminate waste"
        - If agent has "Risk assessment experience" → include "Conduct risk assessment using organizational experience"
        - If agent has "Quality frameworks" → include "Apply quality validation frameworks"

        Provide your plan in this EXACT JSON format:
        {{
            "execution_steps": [
                {{
                    "step_number": 1,
                    "name": "Context Analysis & Knowledge Activation",
                    "description": "Detailed description incorporating agent's skills",
                    "action": "Specific action that leverages discovered capabilities",
                    "tools_needed": ["tool1", "tool2"],
                    "success_criteria": "How to measure success including skill application",
                    "deliverable": "What this step produces enhanced by agent knowledge",
                    "dependencies": [],
                    "estimated_time": "short|medium|long",
                    "validation_checkpoints": ["checkpoint1", "checkpoint2"],
                    "skills_applied": ["skill1", "methodology1"]
                }}
            ],
            "quality_checkpoints": [
                {{
                    "checkpoint_name": "Knowledge-Based Quality Check",
                    "trigger_after_step": 2,
                    "validation_criteria": "Validate using agent's experience and best practices",
                    "success_threshold": "Success criteria enhanced by discovered knowledge"
                }}
            ],
            "success_metrics": ["metric1", "metric2"],
            "risk_mitigation": ["risk1_mitigation", "risk2_mitigation"],
            "total_estimated_time": "short|medium|long",
            "skills_integration_summary": "How agent's skills enhance the execution plan"
        }}
        """
        
        try:
            # Generate execution plan
            if llm_provider.lower() == "anthropic":
                plan_response = await self.anthropic_executor._analyze_with_anthropic(planning_prompt, model)
            else:
                plan_response = await self.openai_executor._analyze_with_openai(planning_prompt, model)
            
            # Parse JSON response
            import json
            import re
            json_match = re.search(r'\{.*\}', plan_response, re.DOTALL)
            if json_match:
                plan_data = json.loads(json_match.group(0))
                
                # Create ExecutionStep objects with skill information
                execution_steps = []
                for step_data in plan_data.get("execution_steps", []):
                    step = ExecutionStep(
                        step_number=step_data.get("step_number", 1),
                        name=step_data.get("name", "Execution Step"),
                        description=step_data.get("description", ""),
                        action=step_data.get("action", ""),
                        tools_needed=step_data.get("tools_needed", []),
                        success_criteria=step_data.get("success_criteria", ""),
                        deliverable=step_data.get("deliverable", ""),
                        dependencies=step_data.get("dependencies", []),
                        estimated_time=step_data.get("estimated_time", "medium"),
                        validation_checkpoints=step_data.get("validation_checkpoints", [])
                    )
                    execution_steps.append(step)
                
                # Add skill-based additional steps if any
                additional_steps = discovered_skills.get("capabilities_enhancement", {}).get("additional_steps", [])
                for i, additional_step in enumerate(additional_steps):
                    step_number = len(execution_steps) + 1
                    step = ExecutionStep(
                        step_number=step_number,
                        name=additional_step["name"],
                        description=additional_step["description"],
                        action=f"Apply discovered capabilities: {additional_step['description']}",
                        tools_needed=additional_step.get("tools_needed", ["brain_vector"]),
                        success_criteria="Successfully integrate agent's specialized knowledge",
                        deliverable=f"Enhanced output using agent's {additional_step['name'].lower()} capabilities",
                        dependencies=[step_number - 1] if execution_steps else [],
                        estimated_time="medium",
                        validation_checkpoints=["Knowledge integration verified", "Quality standards met"]
                    )
                    execution_steps.append(step)
                
                # Enhance quality checkpoints with skill-based validation
                quality_checkpoints = plan_data.get("quality_checkpoints", [])
                if discovered_skills.get("experience"):
                    quality_checkpoints.append({
                        "checkpoint_name": "Experience-Based Validation",
                        "trigger_after_step": len(execution_steps) - 1,
                        "validation_criteria": "Validate against organizational experience and lessons learned",
                        "success_threshold": "Meets historical success patterns and quality standards"
                    })
                
                # Enhance success metrics with skill-specific measures
                success_metrics = plan_data.get("success_metrics", [])
                if discovered_skills.get("methodologies"):
                    success_metrics.append("Methodology application effectiveness")
                if discovered_skills.get("best_practices"):
                    success_metrics.append("Best practices integration completeness")
                
                return MultiStepExecutionPlan(
                    plan_id=plan_id,
                    task_description=user_request,
                    complexity_analysis=complexity_analysis,
                    execution_steps=execution_steps,
                    quality_checkpoints=quality_checkpoints,
                    success_metrics=success_metrics,
                    risk_mitigation=plan_data.get("risk_mitigation", []),
                    total_estimated_time=plan_data.get("total_estimated_time", "medium"),
                    created_at=datetime.now()
                )
            else:
                # Fallback to skill-enhanced default plan
                return self._create_skill_enhanced_execution_plan(plan_id, user_request, complexity_analysis, discovered_skills)
                
        except Exception as e:
            agent_logger.warning(f"LLM execution planning failed: {e}, using skill-enhanced default plan")
            return self._create_skill_enhanced_execution_plan(plan_id, user_request, complexity_analysis, discovered_skills)
    
    def _create_skill_enhanced_execution_plan(self, plan_id: str, user_request: str, 
                                            complexity_analysis: TaskComplexityAnalysis,
                                            discovered_skills: Dict[str, Any]) -> MultiStepExecutionPlan:
        """Create a default execution plan enhanced with discovered skills"""
        
        # Create basic execution steps based on complexity
        base_steps = [
            ExecutionStep(
                step_number=1,
                name="Context Analysis & Knowledge Activation",
                description="Analyze the request and activate relevant agent knowledge and skills",
                action="Understand task requirements and identify applicable methodologies from knowledge base",
                tools_needed=["context", "brain_vector"],
                success_criteria="Clear understanding of task objectives with relevant skills identified",
                deliverable="Task analysis with activated agent knowledge context",
                dependencies=[],
                estimated_time="short",
                validation_checkpoints=["Requirements clear", "Relevant skills identified", "Context established"]
            ),
            ExecutionStep(
                step_number=2,
                name="Information Gathering with Skill Application",
                description="Gather necessary information using agent's specialized approaches",
                action="Use available tools and apply discovered methodologies to collect relevant information",
                tools_needed=["search_factory", "brain_vector"],
                success_criteria="Sufficient information collected using best practices",
                deliverable="Gathered information enhanced by agent's specialized knowledge",
                dependencies=[1],
                estimated_time="medium",
                validation_checkpoints=["Information quality validated", "Methodology application verified", "Completeness check"]
            ),
            ExecutionStep(
                step_number=3,
                name="Analysis & Processing with Best Practices",
                description="Analyze gathered information using organizational best practices and experience",
                action="Process and analyze collected information applying discovered frameworks",
                tools_needed=["business_logic", "brain_vector"],
                success_criteria="Analysis completed using agent's specialized knowledge",
                deliverable="Processed analysis results enhanced by organizational experience",
                dependencies=[2],
                estimated_time="medium",
                validation_checkpoints=["Analysis accuracy verified", "Best practices applied", "Results validity confirmed"]
            )
        ]
        
        # Add skill-specific steps based on discovered capabilities
        if discovered_skills.get("methodologies"):
            methodology_step = ExecutionStep(
                step_number=len(base_steps) + 1,
                name="Methodology Application",
                description=f"Apply discovered methodologies: {', '.join(discovered_skills['methodologies'][:2])}",
                action="Integrate organizational methodologies and frameworks into solution approach",
                tools_needed=["brain_vector", "business_logic"],
                success_criteria="Methodologies successfully applied to enhance solution quality",
                deliverable="Solution approach enhanced by organizational methodologies",
                dependencies=[len(base_steps)],
                estimated_time="medium",
                validation_checkpoints=["Methodology integration verified", "Framework application confirmed"]
            )
            base_steps.append(methodology_step)
        
        # Add additional steps for higher complexity
        if complexity_analysis.complexity_level in ["standard", "complex"]:
            solution_step = ExecutionStep(
                step_number=len(base_steps) + 1,
                name="Framework-Based Solution Generation",
                description="Generate solutions using agent's discovered frameworks and experience",
                action="Create actionable solutions leveraging organizational knowledge and best practices",
                tools_needed=["business_logic", "brain_vector"],
                success_criteria="Solutions are practical, actionable, and incorporate agent's specialized knowledge",
                deliverable="Generated solutions enhanced by organizational frameworks",
                dependencies=[len(base_steps)],
                estimated_time="medium",
                validation_checkpoints=["Solution feasibility confirmed", "Framework integration verified", "Quality standards met"]
            )
            base_steps.append(solution_step)
        
        if complexity_analysis.complexity_level == "complex":
            validation_step = ExecutionStep(
                step_number=len(base_steps) + 1,
                name="Experience-Based Integration & Validation",
                description="Integrate solutions and validate using organizational experience",
                action="Ensure all components work together effectively using lessons learned",
                tools_needed=["business_logic", "context", "brain_vector"],
                success_criteria="Integrated solution validated against organizational experience",
                deliverable="Validated integrated solution enhanced by historical success patterns",
                dependencies=[len(base_steps)],
                estimated_time="medium",
                validation_checkpoints=["Integration success verified", "Experience validation complete", "Quality assurance passed"]
            )
            base_steps.append(validation_step)
        
        # Final synthesis step enhanced with skills
        final_step_number = len(base_steps) + 1
        synthesis_step = ExecutionStep(
            step_number=final_step_number,
            name="Knowledge-Enhanced Response Synthesis",
            description="Synthesize all results into final response using agent's complete knowledge",
            action="Create comprehensive final response incorporating all discovered capabilities",
            tools_needed=[],
            success_criteria="Complete, coherent response that demonstrates agent's specialized knowledge",
            deliverable="Final comprehensive response enhanced by agent's full knowledge base",
            dependencies=[final_step_number - 1],
            estimated_time="short",
            validation_checkpoints=["Response completeness verified", "Knowledge integration confirmed", "Quality standards exceeded"]
        )
        base_steps.append(synthesis_step)
        
        # Create enhanced quality checkpoints
        quality_checkpoints = [
            {
                "checkpoint_name": "Knowledge Integration Quality Check",
                "trigger_after_step": len(base_steps) // 2,
                "validation_criteria": "Verify effective integration of agent's knowledge and skills",
                "success_threshold": "All discovered capabilities are being effectively applied"
            }
        ]
        
        # Add experience-based checkpoint if available
        if discovered_skills.get("experience"):
            quality_checkpoints.append({
                "checkpoint_name": "Experience-Based Validation",
                "trigger_after_step": len(base_steps) - 1,
                "validation_criteria": "Validate outputs against organizational experience and lessons learned",
                "success_threshold": "Meets or exceeds historical success patterns"
            })
        
        # Enhanced success metrics
        success_metrics = ["Task completion", "Quality standards met", "User satisfaction"]
        if discovered_skills.get("methodologies"):
            success_metrics.append("Methodology application effectiveness")
        if discovered_skills.get("best_practices"):
            success_metrics.append("Best practices integration success")
        if discovered_skills.get("frameworks"):
            success_metrics.append("Framework utilization quality")
        
        # Enhanced risk mitigation
        risk_mitigation = ["Fallback to simpler approach if needed", "Error handling at each step"]
        if discovered_skills.get("experience"):
            risk_mitigation.append("Apply lessons learned to avoid historical pitfalls")
        
        return MultiStepExecutionPlan(
            plan_id=plan_id,
            task_description=user_request,
            complexity_analysis=complexity_analysis,
            execution_steps=base_steps,
            quality_checkpoints=quality_checkpoints,
            success_metrics=success_metrics,
            risk_mitigation=risk_mitigation,
            total_estimated_time=complexity_analysis.estimated_duration,
            created_at=datetime.now()
        )

    async def _execute_multi_step_plan(self, execution_plan: MultiStepExecutionPlan, 
                                     request: AgentExecutionRequest, 
                                     agent_config: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Execute multi-step plan with step-by-step progress tracking
        
        Args:
            execution_plan: The multi-step execution plan
            request: Original execution request
            agent_config: Agent configuration
            
        Yields:
            Dict containing step execution results and progress updates
        """
        
        step_results = {}
        
        for step in execution_plan.execution_steps:
            step_start_time = datetime.now()
            
            # Yield step start notification
            yield {
                "type": "step_start",
                "content": f"🔄 Step {step.step_number}/{len(execution_plan.execution_steps)}: {step.name}",
                "provider": request.llm_provider,
                "agent_id": request.agent_id,
                "step_info": {
                    "step_number": step.step_number,
                    "name": step.name,
                    "description": step.description,
                    "estimated_time": step.estimated_time,
                    "tools_needed": step.tools_needed
                }
            }
            
            try:
                # Check dependencies
                for dep_step in step.dependencies:
                    if dep_step not in step_results or step_results[dep_step].status != "completed":
                        yield {
                            "type": "error",
                            "content": f"❌ Step {step.step_number} dependency not met: Step {dep_step} must complete first",
                            "provider": request.llm_provider,
                            "agent_id": request.agent_id,
                            "step_number": step.step_number
                        }
                        return
                
                # Build step-specific prompt
                step_prompt = self._build_step_execution_prompt(
                    step, execution_plan, step_results, agent_config, request.user_request
                )
                
                # Execute step using tool request
                step_tool_request = self._create_step_tool_request(
                    request, agent_config, step_prompt, step.tools_needed
                )
                
                # Execute and collect results
                step_response_chunks = []
                tool_calls_made = []
                
                if request.llm_provider.lower() == "anthropic":
                    async for chunk in self.anthropic_executor.execute_stream(step_tool_request):
                        # Track tool calls
                        if chunk.get("type") == "tool_execution":
                            tool_calls_made.append(chunk.get("tool_name"))
                        
                        # Forward chunk with step context
                        chunk["step_number"] = step.step_number
                        chunk["step_name"] = step.name
                        yield chunk
                        
                        # Collect response chunks
                        if chunk.get("type") == "response_chunk":
                            step_response_chunks.append(chunk.get("content", ""))
                else:
                    async for chunk in self.openai_executor.execute_stream(step_tool_request):
                        # Track tool calls
                        if chunk.get("type") == "tool_execution":
                            tool_calls_made.append(chunk.get("tool_name"))
                        
                        # Forward chunk with step context
                        chunk["step_number"] = step.step_number
                        chunk["step_name"] = step.name
                        yield chunk
                        
                        # Collect response chunks
                        if chunk.get("type") == "response_chunk":
                            step_response_chunks.append(chunk.get("content", ""))
                
                # Combine response
                step_response = "".join(step_response_chunks)
                step_execution_time = (datetime.now() - step_start_time).total_seconds()
                
                # Create step result
                step_result = StepExecutionResult(
                    step_number=step.step_number,
                    status="completed",
                    deliverable={
                        "response": step_response,
                        "summary": f"Step {step.step_number} completed: {step.name}"
                    },
                    tools_used=list(set(tool_calls_made)),
                    execution_time=step_execution_time,
                    success_criteria_met=True,  # TODO: Add validation logic
                    validation_results=[],
                    next_actions=[]
                )
                
                step_results[step.step_number] = step_result
                
                # Yield step completion
                yield {
                    "type": "step_complete",
                    "content": f"✅ Step {step.step_number} completed: {step.name} ({step_execution_time:.1f}s)",
                    "provider": request.llm_provider,
                    "agent_id": request.agent_id,
                    "step_result": {
                        "step_number": step.step_number,
                        "name": step.name,
                        "status": "completed",
                        "execution_time": step_execution_time,
                        "tools_used": step_result.tools_used,
                        "success_criteria_met": step_result.success_criteria_met
                    }
                }
                
                # Check for quality checkpoints
                for checkpoint in execution_plan.quality_checkpoints:
                    if checkpoint.get("trigger_after_step") == step.step_number:
                        yield {
                            "type": "checkpoint",
                            "content": f"🔍 Quality Checkpoint: {checkpoint['checkpoint_name']}",
                            "provider": request.llm_provider,
                            "agent_id": request.agent_id,
                            "checkpoint": checkpoint
                        }
                
            except Exception as e:
                step_execution_time = (datetime.now() - step_start_time).total_seconds()
                
                # Create failed step result
                step_result = StepExecutionResult(
                    step_number=step.step_number,
                    status="failed",
                    deliverable={"error": str(e)},
                    tools_used=[],
                    execution_time=step_execution_time,
                    success_criteria_met=False,
                    validation_results=[],
                    next_actions=["Retry step", "Skip to next step", "Abort execution"],
                    error_details=str(e)
                )
                
                step_results[step.step_number] = step_result
                
                yield {
                    "type": "step_error",
                    "content": f"❌ Step {step.step_number} failed: {step.name} - {str(e)}",
                    "provider": request.llm_provider,
                    "agent_id": request.agent_id,
                    "step_result": {
                        "step_number": step.step_number,
                        "name": step.name,
                        "status": "failed",
                        "error": str(e),
                        "execution_time": step_execution_time
                    }
                }
                
                # For now, continue with next step (could add retry logic here)
                continue
        
        # Multi-step execution summary
        completed_steps = sum(1 for result in step_results.values() if result.status == "completed")
        failed_steps = sum(1 for result in step_results.values() if result.status == "failed")
        total_execution_time = sum(result.execution_time for result in step_results.values())
        
        yield {
            "type": "execution_summary",
            "content": f"📋 Multi-step execution summary: {completed_steps}/{len(execution_plan.execution_steps)} steps completed, {failed_steps} failed, {total_execution_time:.1f}s total",
            "provider": request.llm_provider,
            "agent_id": request.agent_id,
            "summary": {
                "total_steps": len(execution_plan.execution_steps),
                "completed_steps": completed_steps,
                "failed_steps": failed_steps,
                "total_execution_time": total_execution_time,
                "success_rate": completed_steps / len(execution_plan.execution_steps) if execution_plan.execution_steps else 0
            }
        }
    
    def _build_step_execution_prompt(self, step: ExecutionStep, execution_plan: MultiStepExecutionPlan,
                                   step_results: Dict[int, StepExecutionResult], agent_config: Dict[str, Any],
                                   original_request: str) -> str:
        """Build step-specific execution prompt"""
        
        # Get previous step context
        previous_context = ""
        if step_results:
            previous_context = "\nPREVIOUS STEPS COMPLETED:\n"
            for step_num, result in step_results.items():
                if result.status == "completed":
                    previous_context += f"- Step {step_num}: {result.deliverable.get('summary', 'Completed')}\n"
        
        step_prompt = f"""
You are executing Step {step.step_number} of a {len(execution_plan.execution_steps)}-step plan.

ORIGINAL REQUEST: "{original_request}"

CURRENT STEP:
- Step Number: {step.step_number}/{len(execution_plan.execution_steps)}
- Name: {step.name}
- Description: {step.description}
- Action: {step.action}
- Success Criteria: {step.success_criteria}
- Expected Deliverable: {step.deliverable}

AVAILABLE TOOLS FOR THIS STEP: {', '.join(step.tools_needed)}

{previous_context}

EXECUTION INSTRUCTIONS:
1. Focus specifically on completing this step's objectives
2. Use the available tools as needed to accomplish the step action
3. Ensure your output meets the success criteria
4. Provide the deliverable as described

Complete this step now:
"""
        
        return step_prompt
    
    def _create_step_tool_request(self, request: AgentExecutionRequest, agent_config: Dict[str, Any], 
                                step_prompt: str, step_tools: List[str]):
        """Create tool request for a specific step"""
        from exec_tool import ToolExecutionRequest
        
        # Determine tools for this step
        tools_whitelist = step_tools if step_tools else None
        force_tools = bool(step_tools)  # Force tools if specific tools are needed
        
        return ToolExecutionRequest(
            llm_provider=request.llm_provider,
            user_query=request.user_request,
            system_prompt=step_prompt,
            model=request.model,
            model_params=request.model_params,
            org_id=request.org_id,
            user_id=request.user_id,
            enable_tools=True,
            force_tools=force_tools,
            tools_whitelist=tools_whitelist,
            conversation_history=None,  # Don't pass history for individual steps
            max_history_messages=0,
            max_history_tokens=0,
            # Simplified settings for step execution
            enable_deep_reasoning=False,
            reasoning_depth="light",
            enable_intent_classification=False,
            enable_request_analysis=False,
            cursor_mode=False
        )

    # NEW: Enhanced Knowledge Discovery for Skill-Aware Execution
    async def discover_agent_skills(self, user_request: str, agent_config: Dict[str, Any], 
                                  user_id: str, org_id: str) -> Dict[str, Any]:
        """
        Discover relevant agent skills and experience from knowledge base for the current task
        
        Args:
            user_request: The user's task request
            agent_config: Agent configuration
            user_id: User ID  
            org_id: Organization ID
            
        Returns:
            Dict containing discovered skills, methodologies, and experience relevant to the task
        """
        
        knowledge_list = agent_config.get("knowledge_list", [])
        if not knowledge_list or "brain_vector" not in self.available_tools:
            return {"skills": [], "methodologies": [], "experience": [], "capabilities_enhancement": {}}
        
        discovered_skills = {
            "skills": [],
            "methodologies": [], 
            "experience": [],
            "frameworks": [],
            "best_practices": [],
            "capabilities_enhancement": {}
        }
        
        try:
            # Query for specific skill types related to the task
            skill_queries = [
                f"{user_request} methodology framework approach",
                f"{user_request} best practices experience lessons learned", 
                f"{user_request} skills expertise capabilities",
                f"{user_request} process workflow procedures",
                f"{user_request} tools techniques methods"
            ]
            
            for query in skill_queries:
                try:
                    # Search across all knowledge domains
                    knowledge_results = await asyncio.to_thread(
                        self.available_tools["brain_vector"].query_knowledge,
                        user_id=user_id,
                        org_id=org_id,
                        query=query,
                        limit=10  # More results for skill discovery
                    )
                    
                    if knowledge_results:
                        for result in knowledge_results:
                            content = result.get('content', result.get('raw', ''))
                            
                            # Extract different types of knowledge
                            self._extract_skills_from_content(content, discovered_skills)
                            
                except Exception as e:
                    agent_logger.warning(f"Failed to discover skills with query '{query}': {e}")
                    continue
            
            # Analyze discovered skills for capability enhancement
            discovered_skills["capabilities_enhancement"] = self._analyze_skill_based_capabilities(
                discovered_skills, user_request
            )
            
            agent_logger.info(f"Discovered {len(discovered_skills['skills'])} skills, {len(discovered_skills['methodologies'])} methodologies for task")
            
            return discovered_skills
            
        except Exception as e:
            agent_logger.warning(f"Skill discovery failed: {e}")
            return {"skills": [], "methodologies": [], "experience": [], "capabilities_enhancement": {}}
    
    def _extract_skills_from_content(self, content: str, discovered_skills: Dict[str, List]):
        """Extract skills, methodologies, and experience from knowledge content"""
        
        content_lower = content.lower()
        
        # Skill indicators
        skill_indicators = [
            'expertise in', 'skilled at', 'proficient in', 'experienced with', 'specializes in',
            'competent in', 'adept at', 'capable of', 'trained in', 'knowledgeable about'
        ]
        
        # Methodology indicators  
        methodology_indicators = [
            'methodology', 'framework', 'approach', 'process', 'procedure', 'workflow',
            'system', 'method', 'technique', 'strategy', 'protocol'
        ]
        
        # Experience indicators
        experience_indicators = [
            'lessons learned', 'best practices', 'experience shows', 'proven approach',
            'successful implementation', 'case study', 'practical experience', 'field experience'
        ]
        
        # Framework indicators
        framework_indicators = [
            'lean', 'agile', 'six sigma', 'kaizen', 'scrum', 'kanban', 'design thinking',
            'waterfall', 'devops', 'continuous improvement', 'quality management'
        ]
        
        # Extract and categorize content
        sentences = content.split('.')
        
        for sentence in sentences[:5]:  # Limit to first 5 sentences to avoid noise
            sentence_lower = sentence.lower().strip()
            
            if any(indicator in sentence_lower for indicator in skill_indicators):
                if len(sentence) < 200:  # Avoid overly long extractions
                    discovered_skills["skills"].append(sentence.strip())
            
            elif any(indicator in sentence_lower for indicator in methodology_indicators):
                if len(sentence) < 200:
                    discovered_skills["methodologies"].append(sentence.strip())
                    
            elif any(indicator in sentence_lower for indicator in experience_indicators):
                if len(sentence) < 200:
                    discovered_skills["experience"].append(sentence.strip())
                    
            elif any(framework in sentence_lower for framework in framework_indicators):
                if len(sentence) < 200:
                    discovered_skills["frameworks"].append(sentence.strip())
        
        # Extract best practices
        if 'best practice' in content_lower or 'recommended approach' in content_lower:
            practice_sentences = [s.strip() for s in sentences if 'best practice' in s.lower() or 'recommended' in s.lower()]
            discovered_skills["best_practices"].extend(practice_sentences[:3])
    
    def _analyze_skill_based_capabilities(self, discovered_skills: Dict[str, List], user_request: str) -> Dict[str, Any]:
        """Analyze discovered skills to enhance agent capabilities"""
        
        capabilities_enhancement = {
            "complexity_boost": 0,
            "additional_steps": [],
            "specialized_approaches": [],
            "quality_standards": [],
            "risk_mitigation": []
        }
        
        # Boost complexity handling if agent has relevant methodologies
        methodology_count = len(discovered_skills.get("methodologies", []))
        framework_count = len(discovered_skills.get("frameworks", []))
        
        if methodology_count >= 2 or framework_count >= 1:
            capabilities_enhancement["complexity_boost"] = min(2, methodology_count)  # Up to +2 complexity points
        
        # Add specialized execution steps based on discovered skills
        if discovered_skills.get("frameworks"):
            capabilities_enhancement["additional_steps"].append({
                "name": "Framework Application",
                "description": "Apply relevant frameworks and methodologies from knowledge base",
                "tools_needed": ["brain_vector", "business_logic"]
            })
        
        if discovered_skills.get("best_practices"):
            capabilities_enhancement["additional_steps"].append({
                "name": "Best Practices Integration", 
                "description": "Integrate organizational best practices and lessons learned",
                "tools_needed": ["brain_vector"]
            })
        
        # Enhance quality standards based on experience
        if discovered_skills.get("experience"):
            capabilities_enhancement["quality_standards"].extend([
                "Apply lessons learned from previous implementations",
                "Validate against organizational experience",
                "Consider historical success patterns"
            ])
        
        return capabilities_enhancement


# Convenience functions for agent execution
def create_agent_request(
    llm_provider: str,
    user_request: str,
    agent_id: str,
    agent_type: str,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    model_params: Optional[Dict[str, Any]] = None,
    org_id: str = "default",
    user_id: str = "anonymous",
    agent_mode: str = "execute",
    enable_deep_reasoning: bool = True,
    reasoning_depth: str = "standard",
    task_focus: str = "execution",
    enable_tools: bool = True,
    specialized_knowledge_domains: Optional[List[str]] = None,
    conversation_history: Optional[List[Dict[str, Any]]] = None
) -> AgentExecutionRequest:
    """
    Create an agent execution request
    
    Args:
        llm_provider: 'anthropic' or 'openai'
        user_request: The task to execute
        agent_id: Specific agent instance ID
        agent_type: Type of agent (e.g., "sales_agent", "support_agent")
        system_prompt: Optional custom system prompt
        model: Optional custom model name
        model_params: Optional model parameters
        org_id: Organization ID
        user_id: User ID
        enable_deep_reasoning: Enable deep reasoning (default True for agents)
        reasoning_depth: Depth of reasoning
        task_focus: Focus area for task execution
        enable_tools: Whether to enable tools
        specialized_knowledge_domains: Agent's specialization areas
        conversation_history: Previous conversation messages
        
    Returns:
        AgentExecutionRequest object
    """
    return AgentExecutionRequest(
        llm_provider=llm_provider,
        user_request=user_request,
        agent_id=agent_id,
        agent_type=agent_type,
        agent_mode=agent_mode,
        system_prompt=system_prompt,
        model=model,
        model_params=model_params,
        org_id=org_id,
        user_id=user_id,
        enable_deep_reasoning=enable_deep_reasoning,
        reasoning_depth=reasoning_depth,
        task_focus=task_focus,
        enable_tools=enable_tools,
        specialized_knowledge_domains=specialized_knowledge_domains,
        conversation_history=conversation_history
    )


async def execute_agent_async(
    llm_provider: str,
    user_request: str,
    agent_id: str,
    agent_type: str,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    org_id: str = "default",
    user_id: str = "anonymous",
    agent_mode: str = "execute",
    specialized_knowledge_domains: Optional[List[str]] = None
) -> AgentExecutionResponse:
    """
    Execute agent task asynchronously
    
    Args:
        llm_provider: 'anthropic' or 'openai'
        user_request: The task to execute
        agent_id: Specific agent instance ID
        agent_type: Type of agent
        system_prompt: Optional custom system prompt
        model: Optional custom model name
        org_id: Organization ID
        user_id: User ID
        specialized_knowledge_domains: Agent's specialization areas
        
    Returns:
        AgentExecutionResponse with results
    """
    orchestrator = AgentOrchestrator()
    request = create_agent_request(
        llm_provider, user_request, agent_id, agent_type, system_prompt, model, 
        None, org_id, user_id, agent_mode, True, "standard", "execution", True, 
        specialized_knowledge_domains, None
    )
    return await orchestrator.execute_agent_task_async(request)


async def execute_agent_stream(
    llm_provider: str,
    user_request: str,
    agent_id: str,
    agent_type: str,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    org_id: str = "default",
    user_id: str = "anonymous",
    agent_mode: str = "execute",
    enable_deep_reasoning: bool = True,  # Deep reasoning enabled by default for agents
    reasoning_depth: str = "standard",
    task_focus: str = "execution",
    specialized_knowledge_domains: Optional[List[str]] = None,
    conversation_history: Optional[List[Dict[str, Any]]] = None
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Stream agent task execution with simplified flow
    
    Args:
        llm_provider: 'anthropic' or 'openai'
        user_request: The task to execute
        agent_id: Specific agent instance ID
        agent_type: Type of agent
        system_prompt: Optional custom system prompt (not used in simplified flow)
        model: Optional custom model name
        org_id: Organization ID
        user_id: User ID
        enable_deep_reasoning: Enable deep reasoning (default True for agents)
        reasoning_depth: Depth of reasoning (standard, deep, comprehensive)
        task_focus: Focus area for task execution
        specialized_knowledge_domains: Agent's specialization areas
        conversation_history: Previous conversation messages
        
    Yields:
        Dict containing streaming response data with simplified flow
    """
    orchestrator = AgentOrchestrator()
    request = AgentExecutionRequest(
        llm_provider=llm_provider,
        user_request=user_request,
        agent_id=agent_id,
        agent_type=agent_type,
        agent_mode=agent_mode,
        system_prompt=system_prompt,  # Will be overridden by simplified flow
        model=model,
        org_id=org_id,
        user_id=user_id,
        enable_deep_reasoning=enable_deep_reasoning,  # Use passed parameter (default True)
        reasoning_depth=reasoning_depth,              # Use passed parameter (default "standard")
        task_focus=task_focus,
        enable_tools=True,            # Always enable tools
        specialized_knowledge_domains=specialized_knowledge_domains,
        conversation_history=conversation_history
    )
    async for chunk in orchestrator.execute_agent_task_stream(request):
        yield chunk 

