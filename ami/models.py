"""
Ami Agent Creator - Models and Data Classes
===========================================

Contains all dataclasses, enums, and type definitions used throughout the Ami system.
Following the same pattern as agent/models.py for consistency.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


class ConversationState(Enum):
    """States in the collaborative agent creation process"""
    INITIAL_IDEA = "initial_idea"           # Human provides initial idea
    UNDERSTANDING = "understanding"         # Ami analyzes and refines
    SKELETON_REVIEW = "skeleton_review"     # Human reviews skeleton plan
    REFINEMENT = "refinement"              # Ami refines based on feedback
    APPROVED = "approved"                  # Human approves, ready to build
    BUILDING = "building"                  # Ami is building the agent
    COMPLETED = "completed"                # Agent creation completed


@dataclass
class AgentIdea:
    """Initial human idea for an agent"""
    raw_idea: str                          # Original human input
    conversation_id: str                   # Unique conversation ID
    org_id: str
    user_id: str
    created_at: datetime
    state: ConversationState = ConversationState.INITIAL_IDEA


@dataclass
class AgentSkeleton:
    """Complete 7-part agent blueprint for human review"""
    conversation_id: str
    
    # Basic info
    agent_name: str
    agent_purpose: str
    target_users: str
    agent_type: str
    language: str
    
    # 1. Meet Me: Personal introduction
    meet_me: Dict[str, str]  # introduction, value_proposition
    
    # 2. What I Can Do: Tasks and personality
    what_i_do: Dict[str, Any]  # primary_tasks, personality, sample_conversation
    
    # 3. Where I Get My Smarts: Knowledge sources
    knowledge_sources: List[Dict[str, Any]]  # source, type, update_frequency, content_examples
    
    # 4. Apps I Team Up With: Integrations
    integrations: List[Dict[str, str]]  # app_name, trigger, action
    
    # 5. How I Keep You in the Loop: Monitoring
    monitoring: Dict[str, Any]  # reporting_method, metrics_tracked, fallback_response, escalation_method
    
    # 6. Try Me Out: Test scenarios
    test_scenarios: List[Dict[str, str]]  # question, expected_response
    
    # 7. How I Work: Workflow
    workflow_steps: List[str]
    visual_flow: str
    
    # Implementation metadata
    success_criteria: List[str]
    potential_challenges: List[str]
    
    # Conversation context
    created_at: datetime
    state: ConversationState = ConversationState.SKELETON_REVIEW


@dataclass
class CollaborativeAgentRequest:
    """Request for collaborative agent creation - supports both first call and refinement"""
    user_input: str                        # Human input at any stage
    agent_id: Optional[str] = None         # Optional for first call, required for refinement
    blueprint_id: Optional[str] = None     # Optional for first call, required for refinement
    conversation_id: Optional[str] = None  # Existing conversation ID
    org_id: str = ""
    user_id: str = ""
    llm_provider: str = "anthropic"
    model: Optional[str] = None
    current_state: ConversationState = ConversationState.INITIAL_IDEA
    # Conversation history support (following exec_tool.py pattern)
    conversation_history: Optional[List[Dict[str, Any]]] = None  # Previous messages from frontend
    max_history_messages: Optional[int] = 25  # Maximum number of history messages to include


@dataclass
class CollaborativeAgentResponse:
    """Response from collaborative agent creation"""
    success: bool
    conversation_id: str
    current_state: ConversationState
    ami_message: str                       # Ami's response to human
    agent_id: Optional[str] = None         # Include for frontend to use in follow-ups
    blueprint_id: Optional[str] = None     # Include for frontend to use in follow-ups
    data: Optional[Dict[str, Any]] = None  # State-specific data
    next_actions: List[str] = None         # What human can do next
    error: Optional[str] = None


# Legacy models for backward compatibility
@dataclass
class AgentCreationRequest:
    """Request model for direct agent creation (legacy)"""
    user_request: str           # "Create a sales agent for Vietnamese market"
    org_id: str
    user_id: str
    llm_provider: str = "anthropic"
    model: Optional[str] = None


@dataclass
class AgentCreationResult:
    """Response model for agent creation"""
    success: bool
    agent_id: Optional[str] = None
    agent_name: Optional[str] = None
    message: str = ""
    error: Optional[str] = None
    agent_config: Optional[Dict[str, Any]] = None


@dataclass
class SimpleAgentConfig:
    """Internal config model - keep simple"""
    name: str
    description: str
    agent_type: str             # "sales", "support", "analyst", "document_analysis", "general"
    tools_needed: List[str]
    language: str = "english"
    specialization: List[str] = None


# FastAPI Integration Models
class CreateAgentAPIRequest(BaseModel):
    """API request model for direct agent creation (legacy)"""
    user_request: str = Field(..., description="Description of what kind of agent is needed", min_length=10, max_length=1000)
    llm_provider: str = Field("anthropic", description="LLM provider to use for agent creation")
    model: Optional[str] = Field(None, description="Specific model to use (optional)")


class CollaborativeAgentAPIRequest(BaseModel):
    """API request model for collaborative agent creation - supports conversation-first flow"""
    user_input: str = Field(..., description="Human input at any stage of conversation", min_length=1, max_length=2000)
    agent_id: Optional[str] = Field(None, description="Agent ID to collaborate on (only for refinement)")
    blueprint_id: Optional[str] = Field(None, description="Blueprint ID to modify (only for refinement)")
    conversation_id: Optional[str] = Field(None, description="Chat conversation ID for context and message saving")
    current_state: Optional[str] = Field("initial_idea", description="Current conversation state")
    llm_provider: str = Field("anthropic", description="LLM provider to use")
    model: Optional[str] = Field(None, description="Specific model to use")
    # Conversation history support (following exec_tool.py pattern)
    conversation_history: Optional[List[Dict[str, Any]]] = Field(None, description="Previous conversation messages from frontend")
    max_history_messages: Optional[int] = Field(25, description="Maximum number of history messages to include")


class CollaborativeAgentAPIResponse(BaseModel):
    """API response model for collaborative agent creation"""
    success: bool
    conversation_id: str
    current_state: str
    ami_message: str
    agent_id: Optional[str] = None         # Include for frontend refinement
    blueprint_id: Optional[str] = None     # Include for frontend refinement
    data: Optional[Dict[str, Any]] = None
    next_actions: Optional[List[str]] = None
    error: Optional[str] = None


class CreateAgentAPIResponse(BaseModel):
    """API response model for agent creation"""
    success: bool
    agent_id: Optional[str] = None
    agent_name: Optional[str] = None
    message: str
    error: Optional[str] = None
    agent_config: Optional[Dict[str, Any]] = None