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
    formatter = logging.Formatter('🤖 [AGENT] %(asctime)s - %(message)s', datefmt='%H:%M:%S')
    handler.setFormatter(formatter)
    agent_logger.addHandler(handler)

# Import language detection (shared with Ami)
try:
    from language import detect_language_with_llm, LanguageDetector
    LANGUAGE_DETECTION_AVAILABLE = True
    logger.info("Language detection imported successfully from language.py")
except Exception as e:
    logger.warning(f"Failed to import language detection from language.py: {e}")
    LANGUAGE_DETECTION_AVAILABLE = False


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
    """Agent orchestration engine - different from Ami's ExecutiveTool"""
    
    def __init__(self):
        """Initialize the agent orchestrator"""
        self.available_tools = self._initialize_shared_tools()
        
        # Use same provider executors as Ami (shared infrastructure)
        self.anthropic_executor = AnthropicExecutor(self)
        self.openai_executor = OpenAIExecutor(self)
        
        # Agent-specific system prompts (different from Ami's teaching/building role)
        self.agent_system_prompts = {
            "anthropic": "You are a specialized AI agent with deep reasoning capabilities. Your role is to execute tasks efficiently using your specialized knowledge and available tools.",
            "openai": "You are a specialized AI agent with deep reasoning capabilities. Your role is to execute tasks efficiently using your specialized knowledge and available tools."
        }
        
        # Initialize language detection (shared with Ami)
        if LANGUAGE_DETECTION_AVAILABLE:
            self.language_detector = LanguageDetector()
        else:
            self.language_detector = None
    
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
        Stream agent task execution (different from Ami's execute_tool_stream)
        
        Args:
            request: AgentExecutionRequest containing task parameters
            
        Yields:
            Dict containing streaming response data focused on task execution
        """
        start_time = datetime.now()
        
        try:
            # Validate provider
            if request.llm_provider.lower() not in ["anthropic", "openai"]:
                yield {
                    "type": "error",
                    "content": f"Unsupported LLM provider: {request.llm_provider}",
                    "provider": request.llm_provider,
                    "agent_id": request.agent_id,
                    "success": False
                }
                return
            
            # Yield initial processing status
            yield {
                "type": "status",
                "content": f"Agent {request.agent_id} ({request.agent_type}) starting task execution...",
                "provider": request.llm_provider,
                "agent_id": request.agent_id,
                "agent_type": request.agent_type,
                "status": "processing"
            }
            
            # Analyze the task (different from Ami's request analysis)
            task_analysis, thinking_steps = await self._analyze_agent_task(request)
            
            # Generate agent-focused thoughts
            async for thought in self._generate_agent_thoughts(request, task_analysis, thinking_steps):
                yield thought
            
            # Execute task with streaming using provider-specific executor (shared infrastructure)
            tool_request = self._convert_to_tool_request(request)
            if request.llm_provider.lower() == "anthropic":
                async for chunk in self.anthropic_executor.execute_stream(tool_request):
                    # Add agent context to chunks
                    chunk["agent_id"] = request.agent_id
                    chunk["agent_type"] = request.agent_type
                    yield chunk
            else:
                async for chunk in self.openai_executor.execute_stream(tool_request):
                    # Add agent context to chunks
                    chunk["agent_id"] = request.agent_id
                    chunk["agent_type"] = request.agent_type
                    yield chunk
            
            # Yield completion status
            execution_time = (datetime.now() - start_time).total_seconds()
            yield {
                "type": "complete",
                "content": f"Agent {request.agent_id} task execution completed successfully",
                "provider": request.llm_provider,
                "agent_id": request.agent_id,
                "agent_type": request.agent_type,
                "execution_time": execution_time,
                "success": True,
                "metadata": {
                    "org_id": request.org_id,
                    "user_id": request.user_id,
                    "tools_used": list(self.available_tools.keys()),
                    "specialized_domains": request.specialized_knowledge_domains
                }
            }
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"Agent task execution failed: {str(e)}"
            traceback.print_exc()
            
            yield {
                "type": "error",
                "content": error_msg,
                "provider": request.llm_provider,
                "agent_id": request.agent_id,
                "agent_type": request.agent_type,
                "execution_time": execution_time,
                "success": False
            }

    def _convert_to_tool_request(self, agent_request: AgentExecutionRequest):
        """Convert AgentExecutionRequest to ToolExecutionRequest for shared executors"""
        from exec_tool import ToolExecutionRequest
        
        # Create agent-specific system prompt
        agent_system_prompt = f"""You are {agent_request.agent_id}, a specialized {agent_request.agent_type} with deep reasoning capabilities.

Your specialization areas: {', '.join(agent_request.specialized_knowledge_domains or ['general'])}

Your role is to execute tasks efficiently using your specialized knowledge and available tools. You should:
1. Use deep reasoning to understand and approach tasks
2. Leverage your specialized knowledge domains
3. Use available tools when needed to gather additional information
4. Provide thorough, actionable responses focused on task completion

Task Focus: {agent_request.task_focus}
Reasoning Depth: {agent_request.reasoning_depth}"""
        
        return ToolExecutionRequest(
            llm_provider=agent_request.llm_provider,
            user_query=agent_request.user_request,
            system_prompt=agent_system_prompt,
            model=agent_request.model,
            model_params=agent_request.model_params,
            org_id=agent_request.org_id,
            user_id=agent_request.user_id,
            enable_tools=agent_request.enable_tools,
            force_tools=agent_request.force_tools,
            tools_whitelist=agent_request.tools_whitelist,
            conversation_history=agent_request.conversation_history,
            max_history_messages=agent_request.max_history_messages,
            max_history_tokens=agent_request.max_history_tokens,
            # Disable complex features for agents to avoid missing method calls
            enable_deep_reasoning=False,  # Agents use simpler reasoning
            reasoning_depth="light",      # Keep it simple
            enable_intent_classification=False,  # Skip complex intent analysis
            enable_request_analysis=False,      # Skip request analysis
            cursor_mode=False                   # Disable cursor mode to avoid complex method calls
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
            
            # Check for French
            french_matches = sum(1 for pattern in french_patterns if pattern in user_query_lower)
            if french_matches >= 2:
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
        None, org_id, user_id, True, "standard", "execution", True, 
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
    enable_deep_reasoning: bool = True,
    reasoning_depth: str = "standard",
    task_focus: str = "execution",
    specialized_knowledge_domains: Optional[List[str]] = None,
    conversation_history: Optional[List[Dict[str, Any]]] = None
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Stream agent task execution
    
    Args:
        llm_provider: 'anthropic' or 'openai'
        user_request: The task to execute
        agent_id: Specific agent instance ID
        agent_type: Type of agent
        system_prompt: Optional custom system prompt
        model: Optional custom model name
        org_id: Organization ID
        user_id: User ID
        enable_deep_reasoning: Enable deep reasoning
        reasoning_depth: Depth of reasoning
        task_focus: Focus area for task execution
        specialized_knowledge_domains: Agent's specialization areas
        conversation_history: Previous conversation messages
        
    Yields:
        Dict containing streaming response data
    """
    orchestrator = AgentOrchestrator()
    request = AgentExecutionRequest(
        llm_provider=llm_provider,
        user_request=user_request,
        agent_id=agent_id,
        agent_type=agent_type,
        system_prompt=system_prompt,
        model=model,
        org_id=org_id,
        user_id=user_id,
        enable_deep_reasoning=enable_deep_reasoning,
        reasoning_depth=reasoning_depth,
        task_focus=task_focus,
        specialized_knowledge_domains=specialized_knowledge_domains,
        conversation_history=conversation_history
    )
    async for chunk in orchestrator.execute_agent_task_stream(request):
        yield chunk 