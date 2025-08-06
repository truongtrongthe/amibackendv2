#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from supabase import create_client, Client
from typing import Optional, Dict, Any, List
import os
from datetime import datetime, UTC
from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from utilities import logger
from orgdb import (
    create_agent,
    create_agent_with_blueprint,
    get_agents,
    get_agent,
    update_agent,
    delete_agent,
    search_agents,
    get_user_role_in_organization,
    get_organization,
    # Blueprint functions
    create_blueprint,
    get_blueprint,
    get_agent_blueprints,
    get_current_blueprint,
    activate_blueprint,
    get_agent_with_current_blueprint,
    # Compilation functions
    compile_blueprint,
    get_blueprint_compilation_status,
    get_compiled_blueprints_for_agent,
    # Model classes
    AgentBlueprint
)

# Initialize Supabase client
spb_url = os.getenv("SUPABASE_URL")
spb_key = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(spb_url, spb_key)

# Initialize security
security = HTTPBearer()

# Create router
router = APIRouter(prefix="/org-agents", tags=["organization-agents"])

# Request/Response Models for Agents
class CreateAgentRequest(BaseModel):
    name: str = Field(..., description="Agent name", min_length=1, max_length=255)
    description: Optional[str] = Field(None, description="Agent description")

class CreateAgentWithBlueprintRequest(BaseModel):
    name: str = Field(..., description="Agent name", min_length=1, max_length=255)
    description: Optional[str] = Field(None, description="Agent description")
    agent_blueprint: Dict[str, Any] = Field(..., description="Complete agent blueprint JSON")
    conversation_id: Optional[str] = Field(None, description="Conversation ID that created this agent")

class UpdateAgentRequest(BaseModel):
    name: Optional[str] = Field(None, description="Agent name", min_length=1, max_length=255)
    description: Optional[str] = Field(None, description="Agent description")
    status: Optional[str] = Field(None, description="Agent status", pattern="^(active|deactive|delete)$")

class SearchAgentsRequest(BaseModel):
    query: str = Field(..., description="Search query")
    limit: Optional[int] = Field(10, description="Maximum number of results", ge=1, le=100)

class AgentResponse(BaseModel):
    id: str
    agent_id: int
    org_id: str
    name: str
    description: Optional[str] = None
    status: str
    created_by: str
    created_date: datetime
    updated_date: datetime
    current_blueprint_id: Optional[str] = None

# Request/Response Models for Blueprints
class CreateBlueprintRequest(BaseModel):
    agent_blueprint: Dict[str, Any] = Field(..., description="Complete agent blueprint JSON")
    conversation_id: Optional[str] = Field(None, description="Conversation ID that created this blueprint")

class AgentBlueprintResponse(BaseModel):
    id: str
    agent_id: str
    version: int
    agent_blueprint: Dict[str, Any]
    created_date: datetime
    created_by: str
    conversation_id: Optional[str] = None
    compiled_system_prompt: Optional[str] = None
    compiled_at: Optional[datetime] = None
    compiled_by: Optional[str] = None
    compilation_status: str = "draft"
    # Todo-related fields
    implementation_todos: Optional[List[Dict[str, Any]]] = None
    todos_completion_status: str = "not_generated"
    todos_generated_at: Optional[datetime] = None
    todos_generated_by: Optional[str] = None
    todos_completed_at: Optional[datetime] = None
    todos_completed_by: Optional[str] = None

class AgentBuildState(BaseModel):
    current_step: int  # 1-8
    step_name: str
    step_description: str
    is_completed: bool
    next_actions: List[str]
    completion_percentage: float

class AgentWithBlueprintResponse(BaseModel):
    agent: AgentResponse
    blueprint: Optional[AgentBlueprintResponse] = None
    build_state: Optional[AgentBuildState] = None

class CreateDraftAgentRequest(BaseModel):
    name: str = Field(..., description="Agent name", min_length=3, max_length=100)
    initial_idea: str = Field(..., description="Initial agent idea or purpose", min_length=10, max_length=500)
    language: str = Field("english", description="Primary language for the agent")
    agent_type: str = Field("support", description="Type of agent (support, sales, etc.)")

class CreateDraftAgentResponse(BaseModel):
    success: bool
    agent_id: str
    blueprint_id: str
    message: str
    next_actions: List[str]

# Request/Response Models for Compilation
class CompilationStatusResponse(BaseModel):
    status: str
    compiled_at: Optional[datetime] = None
    compiled_by: Optional[str] = None

class CompilationResultResponse(BaseModel):
    blueprint: AgentBlueprintResponse
    compilation_status: str
    compiled_system_prompt: Optional[str] = None
    message: str

# Request/Response Models for Todos
class UpdateTodoStatusRequest(BaseModel):
    todo_id: str = Field(..., description="Todo ID to update")
    new_status: str = Field(..., description="New status: pending, in_progress, completed, cancelled")

class CollectTodoInputsRequest(BaseModel):
    collected_inputs: Dict[str, Any] = Field(..., description="Collected inputs for the todo")

class ValidateTodoInputsRequest(BaseModel):
    provided_inputs: Dict[str, Any] = Field(..., description="Inputs to validate")

class TodoInputValidationResponse(BaseModel):
    valid: bool
    errors: List[str]
    todo_id: str

class TodoStatistics(BaseModel):
    total: int
    completed: int
    in_progress: int
    pending: int
    completion_percentage: float

class BlueprintTodosResponse(BaseModel):
    blueprint_id: str
    todos: List[Dict[str, Any]]
    completion_status: str
    statistics: TodoStatistics
    generated_at: Optional[datetime] = None
    generated_by: Optional[str] = None

class GenerateTodosResponse(BaseModel):
    blueprint: AgentBlueprintResponse
    todos_generated: int
    message: str

# Helper function to convert blueprint to response
def blueprint_to_response(blueprint: AgentBlueprint) -> AgentBlueprintResponse:
    """Convert AgentBlueprint to AgentBlueprintResponse"""
    return AgentBlueprintResponse(
        id=blueprint.id,
        agent_id=blueprint.agent_id,
        version=blueprint.version,
        agent_blueprint=blueprint.agent_blueprint,
        created_date=blueprint.created_date,
        created_by=blueprint.created_by,
        conversation_id=blueprint.conversation_id,
        compiled_system_prompt=blueprint.compiled_system_prompt,
        compiled_at=blueprint.compiled_at,
        compiled_by=blueprint.compiled_by,
        compilation_status=blueprint.compilation_status,
        # Todo-related fields
        implementation_todos=blueprint.implementation_todos,
        todos_completion_status=blueprint.todos_completion_status,
        todos_generated_at=blueprint.todos_generated_at,
        todos_generated_by=blueprint.todos_generated_by,
        todos_completed_at=blueprint.todos_completed_at,
        todos_completed_by=blueprint.todos_completed_by
    )

# Helper function to get current user (imported from login.py)
def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current user from JWT token"""
    try:
        from login import verify_token, JWT_SECRET
        payload = verify_token(credentials.credentials, JWT_SECRET)
        
        if payload.get("type") != "access":
            raise HTTPException(status_code=401, detail="Invalid token type")
        
        user_id = payload.get("user_id")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token payload")
        
        # Get user from database
        from login import get_user_by_id
        user = get_user_by_id(user_id)
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        
        return user
    except Exception as e:
        logger.error(f"Error getting current user: {str(e)}")
        raise HTTPException(status_code=401, detail="Invalid token")

# Agent Management Endpoints

@router.post("/create-draft", response_model=CreateDraftAgentResponse)
async def create_draft_agent_endpoint(
    request: CreateDraftAgentRequest,
    current_user: dict = Depends(get_current_user)
):
    """Create a draft agent with initial blueprint for AMI collaboration"""
    try:
        # Get user's organization
        from organization import get_my_organization
        org_response = await get_my_organization(current_user)
        
        # Create initial blueprint structure based on user's idea
        initial_blueprint = {
            "agent_name": request.name,
            "agent_purpose": request.initial_idea,
            "target_users": "To be defined during collaboration",
            "agent_type": request.agent_type,
            "language": request.language,
            "meet_me": {
                "introduction": f"I'm {request.name}, ready to be configured!",
                "value_proposition": "I'll be customized based on your needs."
            },
            "what_i_do": {
                "primary_tasks": [
                    {
                        "task": "Initial Task",
                        "description": "To be defined during collaboration with AMI"
                    }
                ],
                "personality": {
                    "tone": "professional",
                    "style": "helpful",
                    "analogy": "like a helpful assistant"
                },
                "sample_conversation": {
                    "user_question": "To be defined",
                    "agent_response": "To be defined"
                }
            },
            "knowledge_sources": [],
            "integrations": [],
            "monitoring": {
                "reporting_method": "To be defined",
                "metrics_tracked": [],
                "fallback_response": "I need more information to help you properly.",
                "escalation_method": "To be defined"
            },
            "test_scenarios": [],
            "workflow_steps": ["To be defined during collaboration"],
            "visual_flow": "To be defined",
            "success_criteria": ["To be defined"],
            "potential_challenges": ["To be defined"]
        }
        
        # Create agent with initial blueprint
        from orgdb import create_agent_with_blueprint
        agent, blueprint = create_agent_with_blueprint(
            org_id=org_response.id,
            created_by=current_user["id"],
            name=request.name,
            blueprint_data=initial_blueprint,
            description=request.initial_idea,
            conversation_id=None  # Will be set when AMI collaboration starts
        )
        
        logger.info(f"Created draft agent: {agent.name} (ID: {agent.id}, Blueprint: {blueprint.id})")
        
        return CreateDraftAgentResponse(
            success=True,
            agent_id=agent.id,
            blueprint_id=blueprint.id,
            message=f"Draft agent '{request.name}' created successfully! Ready for AMI collaboration.",
            next_actions=[
                "Start collaborating with AMI using this agent_id and blueprint_id",
                "Refine the agent's purpose and capabilities",
                "Configure integrations and workflows"
            ]
        )
        
    except Exception as e:
        logger.error(f"Error creating draft agent: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create draft agent: {str(e)}")

@router.post("/", response_model=AgentResponse)
async def create_agent_endpoint(request: CreateAgentRequest, current_user: dict = Depends(get_current_user)):
    """Create a new agent (without blueprint) for the current user's organization"""
    try:
        # Get user's organization
        from organization import get_my_organization
        org_response = await get_my_organization(current_user)
        org_id = org_response.id
        
        # Check if user has permission to create agents (owner, admin, or member)
        user_role = get_user_role_in_organization(current_user["id"], org_id)
        if not user_role:
            raise HTTPException(status_code=403, detail="You must be a member of an organization to create agents")
        
        # Create agent
        agent = create_agent(
            org_id=org_id,
            created_by=current_user["id"],
            name=request.name,
            description=request.description
        )
        
        return AgentResponse(
            id=agent.id,
            agent_id=agent.agent_id,
            org_id=agent.org_id,
            name=agent.name,
            description=agent.description,
            status=agent.status,
            created_by=agent.created_by,
            created_date=agent.created_date,
            updated_date=agent.updated_date,
            current_blueprint_id=agent.current_blueprint_id
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating agent: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create agent")

@router.post("/with-blueprint", response_model=AgentWithBlueprintResponse)
async def create_agent_with_blueprint_endpoint(
    request: CreateAgentWithBlueprintRequest, 
    current_user: dict = Depends(get_current_user)
):
    """Create a new agent with its initial blueprint"""
    try:
        # Get user's organization
        from organization import get_my_organization
        org_response = await get_my_organization(current_user)
        org_id = org_response.id
        
        # Check if user has permission to create agents
        user_role = get_user_role_in_organization(current_user["id"], org_id)
        if not user_role:
            raise HTTPException(status_code=403, detail="You must be a member of an organization to create agents")
        
        # Create agent with blueprint
        agent, blueprint = create_agent_with_blueprint(
            org_id=org_id,
            created_by=current_user["id"],
            name=request.name,
            blueprint_data=request.agent_blueprint,
            description=request.description,
            conversation_id=request.conversation_id
        )
        
        return AgentWithBlueprintResponse(
            agent=AgentResponse(
                id=agent.id,
                agent_id=agent.agent_id,
                org_id=agent.org_id,
                name=agent.name,
                description=agent.description,
                status=agent.status,
                created_by=agent.created_by,
                created_date=agent.created_date,
                updated_date=agent.updated_date,
                current_blueprint_id=agent.current_blueprint_id
            ),
            blueprint=blueprint_to_response(blueprint)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating agent with blueprint: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create agent with blueprint")

@router.get("/", response_model=List[AgentResponse])
@router.get("", response_model=List[AgentResponse])
async def get_agents_endpoint(
    status: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """Get all agents for the current user's organization"""
    try:
        # Get user's organization
        from organization import get_my_organization
        org_response = await get_my_organization(current_user)
        org_id = org_response.id
        
        # Check if user has permission to view agents
        user_role = get_user_role_in_organization(current_user["id"], org_id)
        if not user_role:
            raise HTTPException(status_code=403, detail="You must be a member of an organization to view agents")
        
        # Validate status parameter
        if status and status not in ["active", "deactive", "delete"]:
            raise HTTPException(status_code=400, detail="Status must be 'active', 'deactive', or 'delete'")
        
        # Get agents
        agents = get_agents(org_id, status)
        
        # Convert to response models
        agent_responses = []
        for agent in agents:
            agent_responses.append(AgentResponse(
                id=agent.id,
                agent_id=agent.agent_id,
                org_id=agent.org_id,
                name=agent.name,
                description=agent.description,
                status=agent.status,
                created_by=agent.created_by,
                created_date=agent.created_date,
                updated_date=agent.updated_date,
                current_blueprint_id=agent.current_blueprint_id
            ))
        
        return agent_responses
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting agents: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get agents")

@router.get("/{agent_id}", response_model=AgentWithBlueprintResponse)
async def get_agent_endpoint(agent_id: str, current_user: dict = Depends(get_current_user)):
    """Get a specific agent by ID with its current blueprint"""
    try:
        # Get the agent with blueprint
        result = get_agent_with_current_blueprint(agent_id)
        if not result:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        agent, blueprint = result
        
        # Check if user has permission to view this agent
        user_role = get_user_role_in_organization(current_user["id"], agent.org_id)
        if not user_role:
            raise HTTPException(status_code=403, detail="You don't have permission to view this agent")
        
        agent_response = AgentResponse(
            id=agent.id,
            agent_id=agent.agent_id,
            org_id=agent.org_id,
            name=agent.name,
            description=agent.description,
            status=agent.status,
            created_by=agent.created_by,
            created_date=agent.created_date,
            updated_date=agent.updated_date,
            current_blueprint_id=agent.current_blueprint_id
        )
        
        blueprint_response = None
        if blueprint:
            blueprint_response = blueprint_to_response(blueprint)
        
        # Calculate current build state
        build_state = determine_agent_build_state(agent, blueprint)
        
        return AgentWithBlueprintResponse(
            agent=agent_response, 
            blueprint=blueprint_response,
            build_state=build_state
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting agent: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get agent")

@router.put("/{agent_id}", response_model=AgentResponse)
async def update_agent_endpoint(
    agent_id: str, 
    request: UpdateAgentRequest, 
    current_user: dict = Depends(get_current_user)
):
    """Update an agent's basic information (not blueprint)"""
    try:
        # Get the agent
        agent = get_agent(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Check if user has permission to update this agent
        user_role = get_user_role_in_organization(current_user["id"], agent.org_id)
        if not user_role:
            raise HTTPException(status_code=403, detail="You don't have permission to update this agent")
        
        # Only owners and admins can update agents
        if user_role not in ["owner", "admin"]:
            raise HTTPException(status_code=403, detail="Only organization owners and admins can update agents")
        
        # Update agent
        updated_agent = update_agent(
            agent_id=agent_id,
            name=request.name,
            description=request.description,
            status=request.status
        )
        
        if not updated_agent:
            raise HTTPException(status_code=500, detail="Failed to update agent")
        
        return AgentResponse(
            id=updated_agent.id,
            agent_id=updated_agent.agent_id,
            org_id=updated_agent.org_id,
            name=updated_agent.name,
            description=updated_agent.description,
            status=updated_agent.status,
            created_by=updated_agent.created_by,
            created_date=updated_agent.created_date,
            updated_date=updated_agent.updated_date,
            current_blueprint_id=updated_agent.current_blueprint_id
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating agent: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update agent")

@router.delete("/{agent_id}")
async def delete_agent_endpoint(agent_id: str, current_user: dict = Depends(get_current_user)):
    """Soft delete an agent by setting status to 'delete'"""
    try:
        # Get the agent
        agent = get_agent(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Check if user has permission to delete this agent
        user_role = get_user_role_in_organization(current_user["id"], agent.org_id)
        if not user_role:
            raise HTTPException(status_code=403, detail="You don't have permission to delete this agent")
        
        # Only owners and admins can delete agents
        if user_role not in ["owner", "admin"]:
            raise HTTPException(status_code=403, detail="Only organization owners and admins can delete agents")
        
        # Delete agent
        success = delete_agent(agent_id)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete agent")
        
        return {"message": "Agent deleted successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting agent: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete agent")

# Blueprint Management Endpoints

@router.post("/{agent_id}/blueprints", response_model=AgentBlueprintResponse)
async def create_blueprint_endpoint(
    agent_id: str,
    request: CreateBlueprintRequest,
    current_user: dict = Depends(get_current_user)
):
    """Create a new blueprint version for an agent"""
    try:
        # Get the agent
        agent = get_agent(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Check if user has permission to create blueprints for this agent
        user_role = get_user_role_in_organization(current_user["id"], agent.org_id)
        if not user_role:
            raise HTTPException(status_code=403, detail="You don't have permission to create blueprints for this agent")
        
        # Create blueprint
        blueprint = create_blueprint(
            agent_id=agent_id,
            blueprint_data=request.agent_blueprint,
            created_by=current_user["id"],
            conversation_id=request.conversation_id
        )
        
        return blueprint_to_response(blueprint)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating blueprint: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create blueprint")

@router.get("/{agent_id}/blueprints", response_model=List[AgentBlueprintResponse])
async def get_agent_blueprints_endpoint(
    agent_id: str,
    limit: int = 10,
    current_user: dict = Depends(get_current_user)
):
    """Get all blueprint versions for an agent"""
    try:
        # Get the agent
        agent = get_agent(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Check if user has permission to view blueprints for this agent
        user_role = get_user_role_in_organization(current_user["id"], agent.org_id)
        if not user_role:
            raise HTTPException(status_code=403, detail="You don't have permission to view blueprints for this agent")
        
        # Get blueprints
        blueprints = get_agent_blueprints(agent_id, limit)
        
        blueprint_responses = []
        for blueprint in blueprints:
            blueprint_responses.append(blueprint_to_response(blueprint))
        
        return blueprint_responses
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting agent blueprints: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get agent blueprints")

@router.get("/{agent_id}/blueprints/{blueprint_id}", response_model=AgentBlueprintResponse)
async def get_blueprint_endpoint(
    agent_id: str,
    blueprint_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get a specific blueprint by ID"""
    try:
        # Get the agent
        agent = get_agent(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Check if user has permission to view blueprints for this agent
        user_role = get_user_role_in_organization(current_user["id"], agent.org_id)
        if not user_role:
            raise HTTPException(status_code=403, detail="You don't have permission to view blueprints for this agent")
        
        # Get blueprint
        blueprint = get_blueprint(blueprint_id)
        if not blueprint or blueprint.agent_id != agent_id:
            raise HTTPException(status_code=404, detail="Blueprint not found")
        
        return blueprint_to_response(blueprint)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting blueprint: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get blueprint")

@router.post("/{agent_id}/blueprints/{blueprint_id}/activate")
async def activate_blueprint_endpoint(
    agent_id: str,
    blueprint_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Activate a blueprint version as the current blueprint for an agent"""
    try:
        # Get the agent
        agent = get_agent(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Check if user has permission to activate blueprints for this agent
        user_role = get_user_role_in_organization(current_user["id"], agent.org_id)
        if not user_role:
            raise HTTPException(status_code=403, detail="You don't have permission to activate blueprints for this agent")
        
        # Only owners and admins can activate blueprints
        if user_role not in ["owner", "admin"]:
            raise HTTPException(status_code=403, detail="Only organization owners and admins can activate blueprints")
        
        # Activate blueprint
        success = activate_blueprint(agent_id, blueprint_id)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to activate blueprint")
        
        return {"message": "Blueprint activated successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error activating blueprint: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to activate blueprint")

@router.post("/search", response_model=List[AgentResponse])
async def search_agents_endpoint(
    request: SearchAgentsRequest, 
    current_user: dict = Depends(get_current_user)
):
    """Search agents by name within the current user's organization"""
    try:
        # Get user's organization
        from organization import get_my_organization
        org_response = await get_my_organization(current_user)
        org_id = org_response.id
        
        # Check if user has permission to search agents
        user_role = get_user_role_in_organization(current_user["id"], org_id)
        if not user_role:
            raise HTTPException(status_code=403, detail="You must be a member of an organization to search agents")
        
        # Search agents
        agents = search_agents(org_id, request.query, request.limit)
        
        # Convert to response models
        agent_responses = []
        for agent in agents:
            agent_responses.append(AgentResponse(
                id=agent.id,
                agent_id=agent.agent_id,
                org_id=agent.org_id,
                name=agent.name,
                description=agent.description,
                status=agent.status,
                created_by=agent.created_by,
                created_date=agent.created_date,
                updated_date=agent.updated_date,
                current_blueprint_id=agent.current_blueprint_id
            ))
        
        return agent_responses
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching agents: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to search agents")

# Blueprint Compilation Endpoints

@router.post("/{agent_id}/blueprints/{blueprint_id}/compile", response_model=CompilationResultResponse)
async def compile_blueprint_endpoint(
    agent_id: str,
    blueprint_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Compile a blueprint into a system prompt"""
    try:
        # Get the agent
        agent = get_agent(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Check if user has permission to compile blueprints for this agent
        user_role = get_user_role_in_organization(current_user["id"], agent.org_id)
        if not user_role:
            raise HTTPException(status_code=403, detail="You don't have permission to compile blueprints for this agent")
        
        # Only owners and admins can compile blueprints
        if user_role not in ["owner", "admin"]:
            raise HTTPException(status_code=403, detail="Only organization owners and admins can compile blueprints")
        
        # Compile blueprint
        compiled_blueprint = compile_blueprint(blueprint_id, current_user["id"])
        if not compiled_blueprint:
            raise HTTPException(status_code=500, detail="Failed to compile blueprint")
        
        return CompilationResultResponse(
            blueprint=blueprint_to_response(compiled_blueprint),
            compilation_status=compiled_blueprint.compilation_status,
            compiled_system_prompt=compiled_blueprint.compiled_system_prompt,
            message="Blueprint compiled successfully"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error compiling blueprint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to compile blueprint: {str(e)}")

@router.get("/{agent_id}/blueprints/{blueprint_id}/compilation-status", response_model=CompilationStatusResponse)
async def get_compilation_status_endpoint(
    agent_id: str,
    blueprint_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get compilation status for a specific blueprint"""
    try:
        # Get the agent
        agent = get_agent(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Check if user has permission to view compilation status
        user_role = get_user_role_in_organization(current_user["id"], agent.org_id)
        if not user_role:
            raise HTTPException(status_code=403, detail="You don't have permission to view compilation status for this agent")
        
        # Get compilation status
        status_info = get_blueprint_compilation_status(blueprint_id)
        if not status_info:
            raise HTTPException(status_code=404, detail="Blueprint not found")
        
        return CompilationStatusResponse(
            status=status_info["status"],
            compiled_at=datetime.fromisoformat(status_info["compiled_at"].replace("Z", "+00:00")) if status_info.get("compiled_at") else None,
            compiled_by=status_info.get("compiled_by")
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting compilation status: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get compilation status")

@router.get("/{agent_id}/compiled-blueprints", response_model=List[AgentBlueprintResponse])
async def get_compiled_blueprints_endpoint(
    agent_id: str,
    limit: int = 10,
    current_user: dict = Depends(get_current_user)
):
    """Get all compiled blueprint versions for an agent"""
    try:
        # Get the agent
        agent = get_agent(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Check if user has permission to view compiled blueprints
        user_role = get_user_role_in_organization(current_user["id"], agent.org_id)
        if not user_role:
            raise HTTPException(status_code=403, detail="You don't have permission to view compiled blueprints for this agent")
        
        # Get compiled blueprints
        compiled_blueprints = get_compiled_blueprints_for_agent(agent_id, limit)
        
        blueprint_responses = []
        for blueprint in compiled_blueprints:
            blueprint_responses.append(blueprint_to_response(blueprint))
        
        return blueprint_responses
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting compiled blueprints: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get compiled blueprints")

# Status Management Endpoints (activate/deactivate shortcuts)

@router.post("/{agent_id}/activate")
async def activate_agent_endpoint(agent_id: str, current_user: dict = Depends(get_current_user)):
    """Activate a deactivated agent"""
    try:
        # Get the agent
        agent = get_agent(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Check if user has permission to activate this agent
        user_role = get_user_role_in_organization(current_user["id"], agent.org_id)
        if not user_role:
            raise HTTPException(status_code=403, detail="You don't have permission to activate this agent")
        
        # Only owners and admins can activate agents
        if user_role not in ["owner", "admin"]:
            raise HTTPException(status_code=403, detail="Only organization owners and admins can activate agents")
        
        # Update agent status to active
        updated_agent = update_agent(agent_id=agent_id, status="active")
        if not updated_agent:
            raise HTTPException(status_code=500, detail="Failed to activate agent")
        
        return {"message": "Agent activated successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error activating agent: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to activate agent")

@router.post("/{agent_id}/deactivate")
async def deactivate_agent_endpoint(agent_id: str, current_user: dict = Depends(get_current_user)):
    """Deactivate an active agent"""
    try:
        # Get the agent
        agent = get_agent(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Check if user has permission to deactivate this agent
        user_role = get_user_role_in_organization(current_user["id"], agent.org_id)
        if not user_role:
            raise HTTPException(status_code=403, detail="You don't have permission to deactivate this agent")
        
        # Only owners and admins can deactivate agents
        if user_role not in ["owner", "admin"]:
            raise HTTPException(status_code=403, detail="Only organization owners and admins can deactivate agents")
        
        # Update agent status to deactive
        updated_agent = update_agent(agent_id=agent_id, status="deactive")
        if not updated_agent:
            raise HTTPException(status_code=500, detail="Failed to deactivate agent")
        
        return {"message": "Agent deactivated successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deactivating agent: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to deactivate agent")

# Blueprint Implementation Todos Endpoints

@router.post("/{agent_id}/blueprints/{blueprint_id}/generate-todos", response_model=GenerateTodosResponse)
async def generate_todos_endpoint(
    agent_id: str,
    blueprint_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Generate implementation todos for a blueprint based on Ami's analysis"""
    try:
        # Get the agent
        agent = get_agent(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Check if user has permission to generate todos for this agent
        user_role = get_user_role_in_organization(current_user["id"], agent.org_id)
        if not user_role:
            raise HTTPException(status_code=403, detail="You don't have permission to generate todos for this agent")
        
        # Import the function
        from orgdb import generate_implementation_todos
        
        # Generate todos
        updated_blueprint = generate_implementation_todos(blueprint_id, current_user["id"])
        if not updated_blueprint:
            raise HTTPException(status_code=500, detail="Failed to generate todos")
        
        return GenerateTodosResponse(
            blueprint=blueprint_to_response(updated_blueprint),
            todos_generated=len(updated_blueprint.implementation_todos),
            message=f"Generated {len(updated_blueprint.implementation_todos)} implementation todos for the blueprint"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating todos: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate todos: {str(e)}")

@router.get("/{agent_id}/blueprints/{blueprint_id}/todos", response_model=BlueprintTodosResponse)
async def get_blueprint_todos_endpoint(
    agent_id: str,
    blueprint_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get todos for a blueprint with completion statistics"""
    try:
        # Get the agent
        agent = get_agent(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Check if user has permission to view todos for this agent
        user_role = get_user_role_in_organization(current_user["id"], agent.org_id)
        if not user_role:
            raise HTTPException(status_code=403, detail="You don't have permission to view todos for this agent")
        
        # Import the function
        from orgdb import get_blueprint_todos
        
        # Get todos
        todos_info = get_blueprint_todos(blueprint_id)
        if not todos_info:
            raise HTTPException(status_code=404, detail="Blueprint not found or no todos generated")
        
        return BlueprintTodosResponse(
            blueprint_id=todos_info["blueprint_id"],
            todos=todos_info["todos"],
            completion_status=todos_info["completion_status"],
            statistics=TodoStatistics(**todos_info["statistics"]),
            generated_at=todos_info["generated_at"],
            generated_by=todos_info["generated_by"]
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting blueprint todos: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get blueprint todos")

@router.put("/{agent_id}/blueprints/{blueprint_id}/todos/{todo_id}", response_model=dict)
async def update_todo_status_endpoint(
    agent_id: str,
    blueprint_id: str,
    todo_id: str,
    request: UpdateTodoStatusRequest,
    current_user: dict = Depends(get_current_user)
):
    """Update the status of a specific todo"""
    try:
        # Get the agent
        agent = get_agent(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Check if user has permission to update todos for this agent
        user_role = get_user_role_in_organization(current_user["id"], agent.org_id)
        if not user_role:
            raise HTTPException(status_code=403, detail="You don't have permission to update todos for this agent")
        
        # Validate status
        valid_statuses = ["pending", "in_progress", "completed", "cancelled"]
        if request.new_status not in valid_statuses:
            raise HTTPException(status_code=400, detail=f"Invalid status. Must be one of: {valid_statuses}")
        
        # Import the function
        from orgdb import update_todo_status, check_todos_completion_and_update_status
        
        # Update todo status (without collected inputs for this endpoint)
        success = update_todo_status(blueprint_id, request.todo_id, request.new_status, current_user["id"])
        if not success:
            raise HTTPException(status_code=404, detail="Todo not found or update failed")
        
        # Check if all todos are completed and update blueprint status
        all_completed = check_todos_completion_and_update_status(blueprint_id)
        
        message = f"Todo status updated to '{request.new_status}'"
        if all_completed:
            message += ". All todos completed! Blueprint is now ready for compilation."
        
        return {
            "message": message,
            "todo_id": request.todo_id,
            "new_status": request.new_status,
            "all_todos_completed": all_completed
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating todo status: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update todo status")

@router.post("/{agent_id}/blueprints/{blueprint_id}/todos/{todo_id}/validate-inputs", response_model=TodoInputValidationResponse)
async def validate_todo_inputs_endpoint(
    agent_id: str,
    blueprint_id: str,
    todo_id: str,
    request: ValidateTodoInputsRequest,
    current_user: dict = Depends(get_current_user)
):
    """Validate inputs for a specific todo before collection"""
    try:
        # Get the agent
        agent = get_agent(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Check if user has permission to validate inputs for this agent
        user_role = get_user_role_in_organization(current_user["id"], agent.org_id)
        if not user_role:
            raise HTTPException(status_code=403, detail="You don't have permission to validate inputs for this agent")
        
        # Import the function
        from orgdb import validate_todo_inputs
        
        # Validate inputs
        validation_result = validate_todo_inputs(blueprint_id, todo_id, request.provided_inputs)
        
        return TodoInputValidationResponse(
            valid=validation_result["valid"],
            errors=validation_result["errors"],
            todo_id=todo_id
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error validating todo inputs: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to validate todo inputs")

@router.post("/{agent_id}/blueprints/{blueprint_id}/todos/{todo_id}/collect-inputs", response_model=dict)
async def collect_todo_inputs_endpoint(
    agent_id: str,
    blueprint_id: str,
    todo_id: str,
    request: CollectTodoInputsRequest,
    current_user: dict = Depends(get_current_user)
):
    """Collect and store inputs for a specific todo"""
    try:
        # Get the agent
        agent = get_agent(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Check if user has permission to collect inputs for this agent
        user_role = get_user_role_in_organization(current_user["id"], agent.org_id)
        if not user_role:
            raise HTTPException(status_code=403, detail="You don't have permission to collect inputs for this agent")
        
        # Import the functions
        from orgdb import validate_todo_inputs, update_todo_status, check_todos_completion_and_update_status
        
        # Validate inputs first
        validation_result = validate_todo_inputs(blueprint_id, todo_id, request.collected_inputs)
        if not validation_result["valid"]:
            raise HTTPException(status_code=400, detail=f"Input validation failed: {', '.join(validation_result['errors'])}")
        
        # Store inputs and mark todo as completed
        success = update_todo_status(
            blueprint_id=blueprint_id,
            todo_id=todo_id,
            new_status="completed",
            updated_by=current_user["id"],
            collected_inputs=request.collected_inputs
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Todo not found or update failed")
        
        # Check if all todos are completed and update blueprint status
        all_completed = check_todos_completion_and_update_status(blueprint_id)
        
        message = f"Inputs collected and todo completed successfully"
        if all_completed:
            message += ". All todos completed! Blueprint is now ready for compilation."
        
        return {
            "message": message,
            "todo_id": todo_id,
            "status": "completed",
            "inputs_collected": len(request.collected_inputs),
            "all_todos_completed": all_completed
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error collecting todo inputs: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to collect todo inputs")

@router.get("/{agent_id}/blueprints/{blueprint_id}/collected-inputs", response_model=dict)
async def get_collected_inputs_endpoint(
    agent_id: str,
    blueprint_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get all collected inputs from completed todos for compilation preview"""
    try:
        # Get the agent
        agent = get_agent(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Check if user has permission to view collected inputs for this agent
        user_role = get_user_role_in_organization(current_user["id"], agent.org_id)
        if not user_role:
            raise HTTPException(status_code=403, detail="You don't have permission to view collected inputs for this agent")
        
        # Import the function
        from orgdb import get_all_collected_inputs
        
        # Get all collected inputs
        collected_inputs = get_all_collected_inputs(blueprint_id)
        
        return {
            "blueprint_id": blueprint_id,
            "collected_inputs": collected_inputs,
            "message": "All collected inputs retrieved successfully"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting collected inputs: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get collected inputs")

def determine_agent_build_state(agent, blueprint) -> AgentBuildState:
    """
    Determine the current build state of an agent based on its status and blueprint
    
    Returns build state information including current step, completion status, and next actions
    """
    if not blueprint:
        # Agent exists but no blueprint - shouldn't happen in normal flow
        return AgentBuildState(
            current_step=1,
            step_name="Blueprint Creation", 
            step_description="Agent blueprint needs to be created",
            is_completed=False,
            next_actions=["Create agent blueprint"],
            completion_percentage=0.0
        )
    
    # Import here to avoid circular imports
    from orgdb import get_blueprint_todos
    
    # Step 4: Agent created with blueprint
    if blueprint.todos_completion_status == "not_generated":
        return AgentBuildState(
            current_step=4,
            step_name="Agent Created",
            step_description="Agent and blueprint created, todos not yet generated",
            is_completed=True,
            next_actions=["Generate implementation todos"],
            completion_percentage=50.0
        )
    
    # Step 5: Todo completion phase
    if blueprint.todos_completion_status in ["generated", "in_progress"]:
        # Get todo statistics
        todos_info = get_blueprint_todos(blueprint.id)
        completion_pct = 50.0  # Base for having todos generated
        
        if todos_info:
            stats = todos_info["statistics"]
            todo_progress = stats["completion_percentage"]
            completion_pct = 50.0 + (todo_progress * 0.25)  # 50-75% range for todos
        
        return AgentBuildState(
            current_step=5,
            step_name="Implementation Setup",
            step_description="Complete setup requirements and configurations",
            is_completed=False,
            next_actions=["Complete remaining todos", "Configure integrations", "Provide required credentials"],
            completion_percentage=completion_pct
        )
    
    # Step 6-7: Compilation phase
    if blueprint.todos_completion_status == "completed":
        if blueprint.compilation_status == "ready_for_compilation":
            return AgentBuildState(
                current_step=6,
                step_name="Ready for Compilation",
                step_description="All setup complete, ready to compile agent",
                is_completed=False,
                next_actions=["Compile blueprint to create production-ready agent"],
                completion_percentage=75.0
            )
        elif blueprint.compilation_status == "compiled":
            # Step 8: Production ready
            return AgentBuildState(
                current_step=8,
                step_name="Production Ready",
                step_description="Agent is fully configured and ready for use",
                is_completed=True,
                next_actions=["Use agent", "Monitor performance", "Create another agent"],
                completion_percentage=100.0
            )
        elif blueprint.compilation_status == "failed":
            return AgentBuildState(
                current_step=7,
                step_name="Compilation Failed", 
                step_description="Blueprint compilation encountered errors",
                is_completed=False,
                next_actions=["Review compilation errors", "Fix configuration issues", "Retry compilation"],
                completion_percentage=80.0
            )
    
    # Default fallback
    return AgentBuildState(
        current_step=4,
        step_name="Agent Created",
        step_description="Agent created, determining next steps",
        is_completed=True,
        next_actions=["Review agent status"],
        completion_percentage=50.0
    )