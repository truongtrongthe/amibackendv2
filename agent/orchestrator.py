"""
Agent Orchestrator - Main Coordination Module
=============================================

Coordinates all agent components for streamlined execution.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, AsyncGenerator
from datetime import datetime

from .models import AgentExecutionRequest, AgentExecutionResponse
from .complexity_analyzer import TaskComplexityAnalyzer
from .execution_planner import ExecutionPlanner
from .step_executor import StepExecutor
from .prompt_builder import PromptBuilder
from .tool_manager import ToolManager

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


class AgentOrchestrator:
    """Main agent orchestration engine with dynamic configuration loading"""
    
    def __init__(self):
        """Initialize the agent orchestrator with modular components"""
        self.available_tools = self._initialize_shared_tools()
        
        # Use same provider executors as Ami (shared infrastructure)
        from exec_anthropic import AnthropicExecutor
        from exec_openai import OpenAIExecutor
        self.anthropic_executor = AnthropicExecutor(self)
        self.openai_executor = OpenAIExecutor(self)
        
        # Initialize modular components
        self.complexity_analyzer = TaskComplexityAnalyzer(
            self.available_tools, self.anthropic_executor, self.openai_executor
        )
        self.execution_planner = ExecutionPlanner(
            self.available_tools, self.anthropic_executor, self.openai_executor
        )
        self.step_executor = StepExecutor(
            self.available_tools, self.anthropic_executor, self.openai_executor
        )
        self.prompt_builder = PromptBuilder()
        self.tool_manager = ToolManager(self.available_tools)
        
        # Dynamic agent configuration cache
        self.agent_config_cache = {}
        self.cache_ttl = 300  # 5 minutes cache TTL
        
        # Multi-step execution tracking
        self.active_execution_plans = {}  # Track active execution plans
        self.execution_history = {}  # Track completed executions
        
        # Initialize language detection (shared with Ami)
        try:
            from language import LanguageDetector
            self.language_detector = LanguageDetector()
        except Exception as e:
            logger.warning(f"Failed to import language detection: {e}")
            self.language_detector = None
            
        agent_logger.info("Agent Orchestrator initialized with modular architecture")
    
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
        """Load agent configuration from database with caching"""
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
        """Resolve agent identifier (ID, name, or description) to agent ID"""
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
    
    async def execute_agent_task_stream(self, request: AgentExecutionRequest) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Execute an agent task with streaming response
        
        This is the main entry point for agent execution that provides
        real-time streaming of the agent's work process.
        """
        start_time = datetime.now()
        execution_id = f"{request.agent_id}-{start_time.strftime('%H%M%S')}"
        
        execution_logger.info(f"[{execution_id}] === AGENT EXECUTION START ===")
        execution_logger.info(f"[{execution_id}] Agent: {request.agent_id} ({request.agent_type})")
        execution_logger.info(f"[{execution_id}] Request: '{request.user_request[:100]}{'...' if len(request.user_request) > 100 else ''}'")
        
        try:
            # 1. Load agent configuration
            async for status_update in self._load_agent_configuration(request, execution_id):
                yield status_update
            
            # 2. Analyze task complexity and determine execution approach
            async for analysis_update in self._analyze_and_plan_execution(request, execution_id):
                yield analysis_update
            
            # 3. Execute based on complexity analysis
            async for execution_update in self._execute_task(request, execution_id):
                yield execution_update
            
            # 4. Completion
            execution_time = (datetime.now() - start_time).total_seconds()
            execution_logger.info(f"[{execution_id}] === EXECUTION COMPLETE ({execution_time:.2f}s) ===")
             
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"Agent execution failed: {str(e)}"
             
            execution_logger.error(f"[{execution_id}] === EXECUTION FAILED ===")
            execution_logger.error(f"[{execution_id}] Error: {error_msg}")
             
            yield {
                "type": "error",
                "content": error_msg,
                "provider": request.llm_provider,
                "agent_id": request.agent_id,
                "execution_time": execution_time,
                "success": False
            }

    async def execute_agent_task_async(self, request: AgentExecutionRequest) -> AgentExecutionResponse:
        """
        Execute an agent task asynchronously and return the complete result
        
        This method collects all streaming chunks and returns a single response,
        useful for non-streaming API endpoints.
        """
        try:
            start_time = datetime.now()
            result_chunks = []
            final_response = ""
            tasks_completed = 0
            metadata = {}
            
            # Execute the task via streaming and collect results
            async for chunk in self.execute_agent_task_stream(request):
                result_chunks.append(chunk)
                
                # Extract response content
                if chunk.get("type") == "response_chunk":
                    final_response += chunk.get("content", "")
                elif chunk.get("type") == "final_response":
                    final_response = chunk.get("content", final_response)
                elif chunk.get("type") == "step_complete":
                    tasks_completed += 1
                elif chunk.get("type") == "execution_summary":
                    metadata = chunk.get("summary", {})
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return AgentExecutionResponse(
                success=True,
                result=final_response if final_response else "Task completed successfully",
                agent_id=request.agent_id,
                agent_type=request.agent_type,
                execution_time=execution_time,
                tasks_completed=max(tasks_completed, 1),  # At least 1 task completed
                error=None,
                metadata={
                    "total_chunks": len(result_chunks),
                    "execution_summary": metadata,
                    "completion_time": datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Agent async execution failed: {e}")
            
            return AgentExecutionResponse(
                success=False,
                result="",
                agent_id=request.agent_id,
                agent_type=request.agent_type,
                execution_time=execution_time,
                tasks_completed=0,
                error=str(e),
                metadata={"error_time": datetime.now().isoformat()}
            )
    
    async def _load_agent_configuration(self, request: AgentExecutionRequest, execution_id: str):
        """Load and validate agent configuration"""
        yield {
            "type": "status",
            "content": f"🔍 Loading configuration for {request.agent_id}...",
            "status": "loading_config"
        }
        
        resolved_agent_id = await self.resolve_agent_identifier(request.agent_id, request.org_id)
        agent_config = await self.load_agent_config(resolved_agent_id, request.org_id)
        
        # Store configuration for other components
        request.agent_id = resolved_agent_id
        self._current_agent_config = agent_config
        self._current_user_id = request.user_id
        self._current_org_id = request.org_id
        agent_config["org_id"] = request.org_id
        
        yield {
            "type": "status",
            "content": f"✅ Loaded {agent_config['name']} - ready to assist!",
            "agent_name": agent_config['name'],
            "status": "config_loaded"
        }
    
    async def _analyze_and_plan_execution(self, request: AgentExecutionRequest, execution_id: str):
        """Analyze task complexity and create execution plan if needed"""
        yield {
            "type": "status",
            "content": "🧠 Analyzing task complexity...",
            "status": "complexity_analysis"  
        }
        
        complexity_analysis = await self.complexity_analyzer.analyze_task_complexity(
            request.user_request, self._current_agent_config, request.llm_provider,
            self._current_user_id, self._current_org_id, request.model
        )
        
        yield {
            "type": "analysis",
            "content": f"📊 Task Complexity: {complexity_analysis.complexity_score}/10 ({complexity_analysis.complexity_level}) - {complexity_analysis.required_steps} steps planned",
            "complexity_analysis": {
                "score": complexity_analysis.complexity_score,
                "level": complexity_analysis.complexity_level,
                "steps": complexity_analysis.required_steps,
                "task_type": complexity_analysis.task_type,
                "confidence": complexity_analysis.confidence
            }
        }
        
        # Store complexity analysis for execution phase
        self._current_complexity_analysis = complexity_analysis
    
    async def _execute_task(self, request: AgentExecutionRequest, execution_id: str):
        """Execute task using appropriate approach (single-step or multi-step)"""
        complexity_analysis = self._current_complexity_analysis
        agent_config = self._current_agent_config
        
        # Determine execution approach
        system_prompt_data = agent_config.get("system_prompt", {})
        execution_capabilities = system_prompt_data.get("execution_capabilities", {})
        multi_step_threshold = execution_capabilities.get("multi_step_threshold", 4)
        supports_multi_step = execution_capabilities.get("supports_multi_step", True)
        
        if (complexity_analysis.complexity_score >= multi_step_threshold and 
            supports_multi_step and 
            complexity_analysis.complexity_level in ["standard", "complex"]):
            
            # Multi-step execution
            async for step_update in self._execute_multi_step(request, complexity_analysis, execution_id):
                yield step_update
        else:
            # Single-step execution
            async for single_update in self._execute_single_step(request, execution_id):
                yield single_update
    
    async def _execute_multi_step(self, request: AgentExecutionRequest, complexity_analysis, execution_id: str):
        """Execute using multi-step approach"""
        yield {
            "type": "status",
            "content": f"📋 Generating {complexity_analysis.required_steps}-step execution plan...",
            "status": "planning"
        }
        
        execution_plan = await self.execution_planner.generate_execution_plan(
            complexity_analysis, request.user_request, self._current_agent_config,
            request.llm_provider, self._current_user_id, self._current_org_id, request.model
        )
        
        self.active_execution_plans[execution_id] = execution_plan
        
        yield {
            "type": "plan",
            "content": f"✅ Execution plan generated: {len(execution_plan.execution_steps)} steps",
            "execution_plan": {
                "total_steps": len(execution_plan.execution_steps),
                "estimated_time": execution_plan.total_estimated_time
            }
        }
        
        # Execute step by step
        async for step_result in self.step_executor.execute_multi_step_plan(
            execution_plan, request, self._current_agent_config):
            yield step_result
        
        yield {
            "type": "complete",
            "content": f"✅ Multi-step execution completed successfully ({len(execution_plan.execution_steps)} steps)",
            "execution_mode": "multi_step",
            "success": True
        }
    
    async def _execute_single_step(self, request: AgentExecutionRequest, execution_id: str):
        """Execute using single-step approach"""
        yield {
            "type": "status",
            "content": f"⚡ Executing with single-step approach",
            "status": "single_step_execution"
        }
        
        # Use existing single-step execution logic
        tool_request = self._convert_to_tool_request_dynamic(request, self._current_agent_config)
        
        if request.llm_provider.lower() == "anthropic":
            executor = self.anthropic_executor
        else:
            executor = self.openai_executor
        
        async for chunk in executor.execute_stream(tool_request):
            yield chunk
        
        yield {
            "type": "complete",
            "content": f"✅ Single-step execution completed successfully",
            "execution_mode": "single_step",
            "success": True
        }
    
    def _convert_to_tool_request_dynamic(self, agent_request: AgentExecutionRequest, agent_config: Dict[str, Any]):
        """Convert AgentExecutionRequest to ToolExecutionRequest with dynamic configuration"""
        from exec_tool import ToolExecutionRequest
        
        # Build dynamic system prompt from agent configuration with mode support
        dynamic_system_prompt = self.prompt_builder.build_dynamic_system_prompt(
            agent_config, agent_request.user_request, agent_request.agent_mode
        )
        
        # Determine dynamic tool settings
        tools_whitelist, force_tools = self.tool_manager.determine_dynamic_tools(
            agent_config, agent_request.user_request
        )
        
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
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tools (shared with Ami)"""
        return list(self.available_tools.keys())

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
        # Use the prompt builder for language detection
        return await self.prompt_builder._detect_language_and_create_prompt(request, user_query, base_prompt)

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