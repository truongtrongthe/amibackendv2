"""
Agent System Models and Data Classes
====================================

Contains all dataclasses, enums, and type definitions used throughout the agent system.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime


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
    """Request model for agent execution"""
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
    
    # Agent-specific parameters
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


# Multi-Step Planning Data Structures
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
class DiscoveredSkills:
    """Container for agent's discovered skills and capabilities"""
    skills: List[str]
    methodologies: List[str]
    experience: List[str]
    frameworks: List[str]
    best_practices: List[str]
    capabilities_enhancement: Dict[str, Any] 