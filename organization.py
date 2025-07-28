#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from supabase import create_client, Client
from typing import Optional, Dict, Any
import os
from datetime import datetime, UTC
from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from utilities import logger
from orgdb import (
    create_organization, 
    update_organization,
    find_organization_by_name, 
    search_organizations,
    add_user_to_organization,
    get_user_organization,
    get_user_role_in_organization,
    get_user_owned_organizations,
    get_user_organizations,
    get_organization_members,
    remove_user_from_organization,
    Organization
)

# Initialize Supabase client
spb_url = os.getenv("SUPABASE_URL")
spb_key = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(spb_url, spb_key)

# Initialize security
security = HTTPBearer()

# Create router
router = APIRouter(prefix="/organizations", tags=["organizations"])

# Request/Response Models
class CreateOrganizationRequest(BaseModel):
    name: str
    description: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None

class JoinOrganizationRequest(BaseModel):
    organizationId: str

class UpdateOrganizationRequest(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None

class SearchOrganizationsRequest(BaseModel):
    query: str
    limit: Optional[int] = 10

class LeaveOrganizationRequest(BaseModel):
    organizationId: str

class OrganizationResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    userRole: Optional[str] = None
    createdDate: datetime

class OrganizationMemberResponse(BaseModel):
    user_id: str
    email: str
    name: str
    phone: Optional[str] = None
    avatar: Optional[str] = None
    provider: str = "email"
    email_verified: bool = False
    role: str
    joined_at: str

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

# Organization Management Endpoints

@router.post("/", response_model=OrganizationResponse)
async def create_organization_endpoint(request: CreateOrganizationRequest, current_user: dict = Depends(get_current_user)):
    """Create a new organization"""
    try:
        # Check if user is already an owner of any organization
        owned_orgs = get_user_owned_organizations(current_user["id"])
        if owned_orgs:
            raise HTTPException(status_code=400, detail="You can only be owner of one organization. You already own an organization.")
        
        # Create organization
        org = create_organization(
            name=request.name,
            description=request.description,
            email=request.email,
            phone=request.phone,
            address=request.address
        )
        
        # Add user as owner
        success = add_user_to_organization(current_user["id"], org.id, "owner")
        if not success:
            raise HTTPException(status_code=500, detail="Failed to add user as organization owner")
        
        # Update user's org_id (set as primary organization)
        supabase.table("users").update({
            "org_id": org.id,
            "updated_at": datetime.now(UTC).isoformat()
        }).eq("id", current_user["id"]).execute()
        
        return OrganizationResponse(
            id=org.id,
            name=org.name,
            description=org.description,
            email=org.email,
            phone=org.phone,
            address=org.address,
            userRole="owner",
            createdDate=org.created_date
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating organization: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create organization")

@router.post("/update", response_model=OrganizationResponse)
async def update_organization_endpoint(request: UpdateOrganizationRequest, current_user: dict = Depends(get_current_user)):
    """Update an existing organization's information"""
    try:
        # Check if user belongs to the organization they're trying to update
        user_org = get_user_organization(current_user["id"])
        if not user_org or user_org.id != request.id:
            raise HTTPException(status_code=403, detail="You can only update your own organization")
        
        # Check if user has permission to update organization (owner or admin)
        user_role = get_user_role_in_organization(current_user["id"], request.id)
        if user_role not in ["owner", "admin"]:
            raise HTTPException(status_code=403, detail="Only organization owners and admins can update organization information")
        
        # Update organization
        updated_org = update_organization(
            id=request.id,
            name=request.name,
            description=request.description,
            email=request.email,
            phone=request.phone,
            address=request.address
        )
        
        return OrganizationResponse(
            id=updated_org.id,
            name=updated_org.name,
            description=updated_org.description,
            email=updated_org.email,
            phone=updated_org.phone,
            address=updated_org.address,
            userRole=user_role,
            createdDate=updated_org.created_date
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating organization: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update organization")

@router.post("/search")
async def search_organizations_endpoint(request: SearchOrganizationsRequest, current_user: dict = Depends(get_current_user)):
    """Search for organizations"""
    try:
        organizations = search_organizations(request.query, request.limit)
        
        results = []
        for org in organizations:
            # Get user's role if they're a member
            user_role = get_user_role_in_organization(current_user["id"], org.id)
            
            results.append(OrganizationResponse(
                id=org.id,
                name=org.name,
                description=org.description,
                email=org.email,
                phone=org.phone,
                address=org.address,
                userRole=user_role,
                createdDate=org.created_date
            ))
        
        return {"organizations": results}
    
    except Exception as e:
        logger.error(f"Error searching organizations: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to search organizations")

@router.post("/join")
async def join_organization_endpoint(request: JoinOrganizationRequest, current_user: dict = Depends(get_current_user)):
    """Join an existing organization as a member"""
    try:
        # Check if user is already a member of this specific organization
        existing_role = get_user_role_in_organization(current_user["id"], request.organizationId)
        if existing_role:
            # User is already a member - return success response instead of error
            return {
                "message": f"You are already a {existing_role} of this organization",
                "status": "already_member",
                "role": existing_role,
                "organization_id": request.organizationId
            }
        
        # Add user to organization as member
        success = add_user_to_organization(current_user["id"], request.organizationId, "member")
        if not success:
            raise HTTPException(status_code=500, detail="Failed to join organization")
        
        # Update user's org_id only if they don't have one (for backwards compatibility)
        # This maintains the primary organization concept for legacy features
        from login import get_user_by_id
        user = get_user_by_id(current_user["id"])
        if not user.get("org_id"):
            supabase.table("users").update({
                "org_id": request.organizationId,
                "updated_at": datetime.now(UTC).isoformat()
            }).eq("id", current_user["id"]).execute()
        
        return {
            "message": "Successfully joined organization as member",
            "status": "joined",
            "role": "member",
            "organization_id": request.organizationId
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error joining organization: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to join organization")

@router.get("/my", response_model=OrganizationResponse)
async def get_my_organization(current_user: dict = Depends(get_current_user)):
    """Get current user's primary organization (for backwards compatibility)"""
    try:
        org = get_user_organization(current_user["id"])
        if not org:
            raise HTTPException(status_code=404, detail="You are not a member of any organization")
        
        user_role = get_user_role_in_organization(current_user["id"], org.id)
        
        return OrganizationResponse(
            id=org.id,
            name=org.name,
            description=org.description,
            email=org.email,
            phone=org.phone,
            address=org.address,
            userRole=user_role,
            createdDate=org.created_date
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user organization: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get organization")

@router.get("/all")
async def get_all_my_organizations(current_user: dict = Depends(get_current_user)):
    """Get all organizations the user belongs to"""
    try:
        user_orgs = get_user_organizations(current_user["id"])
        
        if not user_orgs:
            return {"organizations": [], "message": "You are not a member of any organizations"}
        
        organizations = []
        for org, role in user_orgs:
            organizations.append(OrganizationResponse(
                id=org.id,
                name=org.name,
                description=org.description,
                email=org.email,
                phone=org.phone,
                address=org.address,
                userRole=role,
                createdDate=org.created_date
            ))
        
        return {"organizations": organizations}
    
    except Exception as e:
        logger.error(f"Error getting user organizations: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get organizations")

@router.post("/leave")
async def leave_organization_endpoint(request: LeaveOrganizationRequest, current_user: dict = Depends(get_current_user)):
    """Leave a specific organization"""
    try:
        # Check if user is a member of this organization
        user_role = get_user_role_in_organization(current_user["id"], request.organizationId)
        if not user_role:
            raise HTTPException(status_code=404, detail="You are not a member of this organization")
        
        # Check if user is the owner
        if user_role == "owner":
            raise HTTPException(status_code=400, detail="Organization owners cannot leave. Transfer ownership first.")
        
        # Remove user from organization
        success = remove_user_from_organization(current_user["id"], request.organizationId)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to leave organization")
        
        # Update user's org_id if this was their primary organization
        from login import get_user_by_id
        user = get_user_by_id(current_user["id"])
        if user.get("org_id") == request.organizationId:
            # Find another organization to set as primary, or set to None
            remaining_orgs = get_user_organizations(current_user["id"])
            new_primary_org_id = None
            
            # Prefer an owned organization as the new primary
            for org, role in remaining_orgs:
                if role == "owner":
                    new_primary_org_id = org.id
                    break
            
            # If no owned organization, use the first available
            if not new_primary_org_id and remaining_orgs:
                new_primary_org_id = remaining_orgs[0][0].id
            
            supabase.table("users").update({
                "org_id": new_primary_org_id,
                "updated_at": datetime.now(UTC).isoformat()
            }).eq("id", current_user["id"]).execute()
        
        return {"message": "Successfully left organization"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error leaving organization: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to leave organization")

@router.get("/{org_id}/members")
async def get_organization_members_endpoint(org_id: str, current_user: dict = Depends(get_current_user)):
    """Get all members and their roles for a specific organization"""
    try:
        # Check if user has permission to view organization members
        user_role = get_user_role_in_organization(current_user["id"], org_id)
        if not user_role:
            raise HTTPException(status_code=403, detail="You are not a member of this organization")
        
        # Only owners and admins can view all members
        if user_role not in ["owner", "admin"]:
            raise HTTPException(status_code=403, detail="Only organization owners and admins can view member list")
        
        # Get organization members
        members = get_organization_members(org_id)
        
        # Convert to response models
        member_responses = []
        for member in members:
            member_responses.append(OrganizationMemberResponse(
                user_id=member["user_id"],
                email=member["email"],
                name=member["name"],
                phone=member.get("phone"),
                avatar=member.get("avatar"),
                provider=member.get("provider", "email"),
                email_verified=member.get("email_verified", False),
                role=member["role"],
                joined_at=member["joined_at"]
            ))
        
        return {
            "organization_id": org_id,
            "members": member_responses,
            "total_members": len(member_responses)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting organization members: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get organization members")

@router.get("/members/by-name/{org_name}")
async def get_organization_members_by_name_endpoint(org_name: str, current_user: dict = Depends(get_current_user)):
    """Get all members and their roles for an organization by name"""
    try:
        # Find organization by name
        org = find_organization_by_name(org_name)
        if not org:
            raise HTTPException(status_code=404, detail="Organization not found")
        
        # Check if user has permission to view organization members
        user_role = get_user_role_in_organization(current_user["id"], org.id)
        if not user_role:
            raise HTTPException(status_code=403, detail="You are not a member of this organization")
        
        # Only owners and admins can view all members
        if user_role not in ["owner", "admin"]:
            raise HTTPException(status_code=403, detail="Only organization owners and admins can view member list")
        
        # Get organization members
        members = get_organization_members(org.id)
        
        # Convert to response models
        member_responses = []
        for member in members:
            member_responses.append(OrganizationMemberResponse(
                user_id=member["user_id"],
                email=member["email"],
                name=member["name"],
                phone=member.get("phone"),
                avatar=member.get("avatar"),
                provider=member.get("provider", "email"),
                email_verified=member.get("email_verified", False),
                role=member["role"],
                joined_at=member["joined_at"]
            ))
        
        return {
            "organization_id": org.id,
            "organization_name": org.name,
            "members": member_responses,
            "total_members": len(member_responses)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting organization members by name: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get organization members") 