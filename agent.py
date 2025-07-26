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
        Stream agent task execution with simplified flow
        
        Args:
            request: AgentExecutionRequest containing task parameters
            
        Yields:
            Dict containing streaming response data with simplified flow
        """
        start_time = datetime.now()
        execution_id = f"{request.agent_id}-{start_time.strftime('%H%M%S')}"
        
        # Enhanced logging - execution start
        execution_logger.info(f"[{execution_id}] === AGENT EXECUTION START ===")
        execution_logger.info(f"[{execution_id}] Agent: {request.agent_id} ({request.agent_type})")
        execution_logger.info(f"[{execution_id}] Provider: {request.llm_provider}")
        execution_logger.info(f"[{execution_id}] Model: {request.model or 'default'}")
        execution_logger.info(f"[{execution_id}] User Request: '{request.user_request[:100]}{'...' if len(request.user_request) > 100 else ''}'")
        execution_logger.info(f"[{execution_id}] Specialized Domains: {request.specialized_knowledge_domains}")
        execution_logger.info(f"[{execution_id}] Tools Enabled: {request.enable_tools}")
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
            
            # Simple status - no redundant analysis
            execution_logger.info(f"[{execution_id}] STREAMING STATUS - Initial processing message")
            yield {
                "type": "status",
                "content": f"🤖 {request.agent_id} processing your request...",
                "provider": request.llm_provider,
                "agent_id": request.agent_id,
                "agent_type": request.agent_type,
                "status": "processing"
            }
            
            # Convert to tool request with simplified approach
            execution_logger.info(f"[{execution_id}] TOOL CONVERSION - Converting agent request to tool request")
            tool_request = self._convert_to_tool_request(request)
            execution_logger.info(f"[{execution_id}] TOOL CONVERSION COMPLETE - Available tools: {list(self.available_tools.keys())}")
            
            # Log system prompt (truncated for readability)
            system_prompt_preview = tool_request.system_prompt[:200] + "..." if len(tool_request.system_prompt) > 200 else tool_request.system_prompt
            execution_logger.info(f"[{execution_id}] SYSTEM PROMPT: {system_prompt_preview}")
            
            # Execute directly with LLM - let LLM decide tools and handle everything
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



    def _convert_to_tool_request(self, agent_request: AgentExecutionRequest):
        """Convert AgentExecutionRequest to ToolExecutionRequest with simplified approach"""
        from exec_tool import ToolExecutionRequest
        
        # Create focused agent system prompt
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

