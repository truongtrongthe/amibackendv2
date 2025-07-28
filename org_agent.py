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
    get_agents,
    get_agent,
    update_agent,
    delete_agent,
    search_agents,
    get_user_role_in_organization,
    get_organization
)

# Initialize Supabase client
spb_url = os.getenv("SUPABASE_URL")
spb_key = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(spb_url, spb_key)

# Initialize security
security = HTTPBearer()

# Create router
router = APIRouter(prefix="/org-agents", tags=["organization-agents"])

# Request/Response Models
class CreateAgentRequest(BaseModel):
    name: str = Field(..., description="Agent name", min_length=1, max_length=255)
    description: Optional[str] = Field(None, description="Agent description")
    system_prompt: Optional[Dict[str, Any]] = Field(default_factory=dict, description="System prompt as JSON")
    tools_list: Optional[List[str]] = Field(default_factory=list, description="List of tool names/IDs")
    knowledge_list: Optional[List[str]] = Field(default_factory=list, description="List of knowledge base names/IDs")

class UpdateAgentRequest(BaseModel):
    name: Optional[str] = Field(None, description="Agent name", min_length=1, max_length=255)
    description: Optional[str] = Field(None, description="Agent description")
    system_prompt: Optional[Dict[str, Any]] = Field(None, description="System prompt as JSON")
    tools_list: Optional[List[str]] = Field(None, description="List of tool names/IDs")
    knowledge_list: Optional[List[str]] = Field(None, description="List of knowledge base names/IDs")
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
    system_prompt: Dict[str, Any]
    tools_list: List[str]
    knowledge_list: List[str]
    status: str
    created_by: str
    created_date: datetime
    updated_date: datetime

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

# Organization Agent Management Endpoints

@router.post("/", response_model=AgentResponse)
async def create_agent_endpoint(request: CreateAgentRequest, current_user: dict = Depends(get_current_user)):
    """Create a new agent for the current user's organization"""
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
            description=request.description,
            system_prompt=request.system_prompt,
            tools_list=request.tools_list,
            knowledge_list=request.knowledge_list
        )
        
        return AgentResponse(
            id=agent.id,
            agent_id=agent.agent_id,
            org_id=agent.org_id,
            name=agent.name,
            description=agent.description,
            system_prompt=agent.system_prompt,
            tools_list=agent.tools_list,
            knowledge_list=agent.knowledge_list,
            status=agent.status,
            created_by=agent.created_by,
            created_date=agent.created_date,
            updated_date=agent.updated_date
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating agent: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create agent")

@router.get("/", response_model=List[AgentResponse])
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
                system_prompt=agent.system_prompt,
                tools_list=agent.tools_list,
                knowledge_list=agent.knowledge_list,
                status=agent.status,
                created_by=agent.created_by,
                created_date=agent.created_date,
                updated_date=agent.updated_date
            ))
        
        return agent_responses
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting agents: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get agents")

@router.get("/{agent_id}", response_model=AgentResponse)
async def get_agent_endpoint(agent_id: str, current_user: dict = Depends(get_current_user)):
    """Get a specific agent by ID"""
    try:
        # Get the agent
        agent = get_agent(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Check if user has permission to view this agent
        user_role = get_user_role_in_organization(current_user["id"], agent.org_id)
        if not user_role:
            raise HTTPException(status_code=403, detail="You don't have permission to view this agent")
        
        return AgentResponse(
            id=agent.id,
            agent_id=agent.agent_id,
            org_id=agent.org_id,
            name=agent.name,
            description=agent.description,
            system_prompt=agent.system_prompt,
            tools_list=agent.tools_list,
            knowledge_list=agent.knowledge_list,
            status=agent.status,
            created_by=agent.created_by,
            created_date=agent.created_date,
            updated_date=agent.updated_date
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
    """Update an agent's information"""
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
            system_prompt=request.system_prompt,
            tools_list=request.tools_list,
            knowledge_list=request.knowledge_list,
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
            system_prompt=updated_agent.system_prompt,
            tools_list=updated_agent.tools_list,
            knowledge_list=updated_agent.knowledge_list,
            status=updated_agent.status,
            created_by=updated_agent.created_by,
            created_date=updated_agent.created_date,
            updated_date=updated_agent.updated_date
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
                system_prompt=agent.system_prompt,
                tools_list=agent.tools_list,
                knowledge_list=agent.knowledge_list,
                status=agent.status,
                created_by=agent.created_by,
                created_date=agent.created_date,
                updated_date=agent.updated_date
            ))
        
        return agent_responses
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching agents: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to search agents")

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