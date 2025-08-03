"""
Ami - The Agent Creator Module
==============================

A refactored, maintainable agent creation system with clear separation of concerns.
Following the same modular architecture pattern as the agent system.

Main Components:
- AmiOrchestrator: Main coordination engine for agent creation
- CollaborativeCreator: Handles iterative agent refinement (Chief Product Officer approach)
- DirectCreator: Handles simple, direct agent creation (legacy)
- AmiKnowledgeManager: Manages knowledge persistence to Pinecone
- Models: All data classes and type definitions

Usage:
    # NEW: Modular approach
    from ami import AmiOrchestrator, create_agent_request, collaborate_on_agent
    
    orchestrator = AmiOrchestrator()
    request = create_agent_request(...)
    result = await orchestrator.create_agent(request)
    
    # Legacy: Direct functions (backwards compatible)
    from ami import create_agent_simple, collaborate_on_agent_via_api
"""

# Main components
from .orchestrator import AmiOrchestrator, get_ami_orchestrator
from .models import (
    # Core data models
    ConversationState,
    AgentIdea,
    AgentSkeleton,
    CollaborativeAgentRequest,
    CollaborativeAgentResponse,
    AgentCreationRequest,
    AgentCreationResult,
    SimpleAgentConfig,
    
    # API models
    CreateAgentAPIRequest,
    CreateAgentAPIResponse,
    CollaborativeAgentAPIRequest,
    CollaborativeAgentAPIResponse
)

# Individual components (for advanced usage)
from .knowledge_manager import AmiKnowledgeManager
from .collaborative_creator import CollaborativeCreator
from .direct_creator import DirectCreator


# Convenience functions for backwards compatibility
def create_agent_request(
    user_request: str,
    org_id: str,
    user_id: str,
    llm_provider: str = "anthropic",
    model: str = None
) -> AgentCreationRequest:
    """
    Create an agent creation request (backwards compatible)
    """
    return AgentCreationRequest(
        user_request=user_request,
        org_id=org_id,
        user_id=user_id,
        llm_provider=llm_provider,
        model=model
    )


def create_collaborative_request(
    user_input: str,
    agent_id: str,
    blueprint_id: str,
    org_id: str,
    user_id: str,
    conversation_id: str = None,
    current_state: str = "refinement",
    llm_provider: str = "anthropic",
    model: str = None
) -> CollaborativeAgentRequest:
    """
    Create a collaborative agent request (updated for agent/blueprint context)
    """
    return CollaborativeAgentRequest(
        user_input=user_input,
        agent_id=agent_id,
        blueprint_id=blueprint_id,
        conversation_id=conversation_id,
        org_id=org_id,
        user_id=user_id,
        llm_provider=llm_provider,
        model=model,
        current_state=ConversationState(current_state)
    )


async def create_agent_simple(
    user_request: str, 
    org_id: str, 
    user_id: str, 
    provider: str = "anthropic"
) -> AgentCreationResult:
    """
    Simple function to create an agent (backwards compatible)
    
    Args:
        user_request: Description of what kind of agent is needed
        org_id: Organization ID
        user_id: User ID
        provider: LLM provider to use
        
    Returns:
        AgentCreationResult with creation status and details
    """
    orchestrator = get_ami_orchestrator()
    request = AgentCreationRequest(
        user_request=user_request,
        org_id=org_id,
        user_id=user_id,
        llm_provider=provider
    )
    return await orchestrator.create_agent(request)


async def collaborate_on_agent(
    user_input: str,
    org_id: str,
    user_id: str,
    conversation_id: str = None,
    current_state: str = "initial_idea",
    llm_provider: str = "anthropic"
) -> CollaborativeAgentResponse:
    """
    Collaborate on agent creation (new function)
    
    Args:
        user_input: Human input at any stage of conversation
        org_id: Organization ID
        user_id: User ID
        conversation_id: Existing conversation ID (optional)
        current_state: Current conversation state
        llm_provider: LLM provider to use
        
    Returns:
        CollaborativeAgentResponse with Ami's guidance
    """
    orchestrator = get_ami_orchestrator()
    request = CollaborativeAgentRequest(
        user_input=user_input,
        conversation_id=conversation_id,
        org_id=org_id,
        user_id=user_id,
        llm_provider=llm_provider,
        current_state=ConversationState(current_state)
    )
    return await orchestrator.collaborate_on_agent(request)


# FastAPI Integration Functions (backwards compatible)
async def create_agent_via_api(
    api_request: CreateAgentAPIRequest, 
    org_id: str, 
    user_id: str
) -> CreateAgentAPIResponse:
    """
    Create agent via API call (backwards compatible)
    """
    try:
        orchestrator = get_ami_orchestrator()
        
        creation_request = AgentCreationRequest(
            user_request=api_request.user_request,
            org_id=org_id,
            user_id=user_id,
            llm_provider=api_request.llm_provider,
            model=api_request.model
        )
        
        result = await orchestrator.create_agent(creation_request)
        
        return CreateAgentAPIResponse(
            success=result.success,
            agent_id=result.agent_id,
            agent_name=result.agent_name,
            message=result.message,
            error=result.error,
            agent_config=result.agent_config
        )
        
    except Exception as e:
        return CreateAgentAPIResponse(
            success=False,
            message="❌ Agent creation failed",
            error=str(e)
        )


async def collaborate_on_agent_via_api(
    api_request: CollaborativeAgentAPIRequest, 
    org_id: str, 
    user_id: str
) -> CollaborativeAgentAPIResponse:
    """
    Collaborative agent creation via API (backwards compatible)
    """
    try:
        orchestrator = get_ami_orchestrator()
        
        collaboration_request = CollaborativeAgentRequest(
            user_input=api_request.user_input,
            agent_id=api_request.agent_id,
            blueprint_id=api_request.blueprint_id,
            conversation_id=api_request.conversation_id,
            org_id=org_id,
            user_id=user_id,
            llm_provider=api_request.llm_provider,
            model=api_request.model,
            current_state=ConversationState(api_request.current_state)
        )
        
        result = await orchestrator.collaborate_on_agent(collaboration_request)
        
        return CollaborativeAgentAPIResponse(
            success=result.success,
            conversation_id=result.conversation_id,
            current_state=result.current_state.value,
            ami_message=result.ami_message,
            data=result.data,
            next_actions=result.next_actions,
            error=result.error
        )
        
    except Exception as e:
        return CollaborativeAgentAPIResponse(
            success=False,
            conversation_id=api_request.conversation_id or "unknown",
            current_state=api_request.current_state,
            ami_message="Something went wrong. Let me help you start fresh!",
            error=str(e)
        )


# Export main classes and functions
__all__ = [
    # Main orchestrator
    'AmiOrchestrator',
    'get_ami_orchestrator',
    
    # Core models
    'ConversationState',
    'AgentIdea',
    'AgentSkeleton',
    'CollaborativeAgentRequest',
    'CollaborativeAgentResponse',
    'AgentCreationRequest',
    'AgentCreationResult',
    'SimpleAgentConfig',
    
    # API models
    'CreateAgentAPIRequest',
    'CreateAgentAPIResponse',
    'CollaborativeAgentAPIRequest',
    'CollaborativeAgentAPIResponse',
    
    # Individual components
    'AmiKnowledgeManager',
    'CollaborativeCreator',
    'DirectCreator',
    
    # Convenience functions
    'create_agent_request',
    'create_collaborative_request',
    'create_agent_simple',
    'collaborate_on_agent',
    
    # API functions
    'create_agent_via_api',
    'collaborate_on_agent_via_api'
]


# Legacy compatibility - maintain old imports
# This ensures existing code using 'from ami import AmiAgentCreator' still works
AmiAgentCreator = get_ami_orchestrator  # Alias for backwards compatibility

# Legacy function aliases
async def create_agent_legacy(user_request: str, org_id: str, user_id: str, provider: str = "anthropic"):
    """Legacy compatibility function"""
    return await create_agent_simple(user_request, org_id, user_id, provider)