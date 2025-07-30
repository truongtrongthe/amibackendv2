"""
Agent System - Modular Architecture
===================================

A refactored, maintainable agent system with clear separation of concerns.

Main Components:
- AgentOrchestrator: Main coordination engine
- TaskComplexityAnalyzer: Analyzes task complexity  
- ExecutionPlanner: Creates multi-step execution plans
- StepExecutor: Executes step-by-step plans
- SkillDiscoveryEngine: Discovers agent skills from knowledge base
- PromptBuilder: Builds dynamic prompts
- ToolManager: Manages tool selection

Usage:
    from agent import AgentOrchestrator, create_agent_request, execute_agent_stream
    
    orchestrator = AgentOrchestrator()
    request = create_agent_request(...)
    async for chunk in orchestrator.execute_agent_task_stream(request):
        print(chunk)
"""

# Main components
from .orchestrator import AgentOrchestrator
from .models import (
    AgentExecutionRequest, 
    AgentExecutionResponse,
    TaskComplexityAnalysis,
    ExecutionStep,
    MultiStepExecutionPlan,
    StepExecutionResult,
    DiscoveredSkills
)

# Individual components (for advanced usage)
from .complexity_analyzer import TaskComplexityAnalyzer
from .skill_discovery import SkillDiscoveryEngine

# Convenience functions (backwards compatibility)
def create_agent_request(
    llm_provider: str,
    user_request: str,
    agent_id: str,
    agent_type: str,
    system_prompt: str = None,
    model: str = None,
    model_params: dict = None,
    org_id: str = "default",
    user_id: str = "anonymous",
    agent_mode: str = "execute",
    enable_deep_reasoning: bool = True,
    reasoning_depth: str = "standard",
    task_focus: str = "execution",
    enable_tools: bool = True,
    specialized_knowledge_domains: list = None,
    conversation_history: list = None
) -> AgentExecutionRequest:
    """
    Create an agent execution request (backwards compatible)
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


async def execute_agent_stream(
    llm_provider: str,
    user_request: str,
    agent_id: str,
    agent_type: str,
    system_prompt: str = None,
    model: str = None,
    org_id: str = "default",
    user_id: str = "anonymous",
    agent_mode: str = "execute",
    enable_deep_reasoning: bool = True,
    reasoning_depth: str = "standard",
    task_focus: str = "execution",
    specialized_knowledge_domains: list = None,
    conversation_history: list = None
):
    """
    Stream agent task execution (backwards compatible)
    """
    orchestrator = AgentOrchestrator()
    request = AgentExecutionRequest(
        llm_provider=llm_provider,
        user_request=user_request,
        agent_id=agent_id,
        agent_type=agent_type,
        agent_mode=agent_mode,
        system_prompt=system_prompt,
        model=model,
        org_id=org_id,
        user_id=user_id,
        enable_deep_reasoning=enable_deep_reasoning,
        reasoning_depth=reasoning_depth,
        task_focus=task_focus,
        enable_tools=True,
        specialized_knowledge_domains=specialized_knowledge_domains,
        conversation_history=conversation_history
    )
    async for chunk in orchestrator.execute_agent_task_stream(request):
        yield chunk


async def execute_agent_async(
    llm_provider: str,
    user_request: str,
    agent_id: str,
    agent_type: str,
    system_prompt: str = None,
    model: str = None,
    org_id: str = "default",
    user_id: str = "anonymous",
    agent_mode: str = "execute",
    specialized_knowledge_domains: list = None
) -> AgentExecutionResponse:
    """
    Execute agent task asynchronously and return complete result (backwards compatible)
    
    Args:
        llm_provider: 'anthropic' or 'openai'
        user_request: The task to execute
        agent_id: Specific agent instance ID
        agent_type: Type of agent
        system_prompt: Optional custom system prompt
        model: Optional custom model name
        org_id: Organization ID
        user_id: User ID
        agent_mode: Agent operational mode ('execute' or 'collaborate')
        specialized_knowledge_domains: Agent's specialization areas
        
    Returns:
        AgentExecutionResponse with results
    """
    orchestrator = AgentOrchestrator()
    request = AgentExecutionRequest(
        llm_provider=llm_provider,
        user_request=user_request,
        agent_id=agent_id,
        agent_type=agent_type,
        agent_mode=agent_mode,
        system_prompt=system_prompt,
        model=model,
        org_id=org_id,
        user_id=user_id,
        enable_deep_reasoning=True,
        reasoning_depth="standard",
        task_focus="execution",
        enable_tools=True,
        specialized_knowledge_domains=specialized_knowledge_domains,
        conversation_history=None
    )
    return await orchestrator.execute_agent_task_async(request)


# Export main classes
__all__ = [
    'AgentOrchestrator',
    'AgentExecutionRequest',
    'AgentExecutionResponse', 
    'TaskComplexityAnalysis',
    'ExecutionStep',
    'MultiStepExecutionPlan',
    'StepExecutionResult',
    'DiscoveredSkills',
    'TaskComplexityAnalyzer',
    'SkillDiscoveryEngine',
    'create_agent_request',
    'execute_agent_stream',
    'execute_agent_async'
] 