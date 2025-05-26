#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from uuid import UUID, uuid4
from datetime import datetime
import json
import os
import traceback

# Import dependencies
from braingraph import (
    create_brain_graph, get_brain_graph,
    get_brain_graph_versions,
    update_brain_graph_version_status, BrainGraphVersion,
    add_brains_to_version, remove_brains_from_version
)
from supabase import create_client, Client
from utilities import logger

# Supabase initialization
spb_url = os.getenv("SUPABASE_URL", "https://example.supabase.co")
spb_key = os.getenv("SUPABASE_KEY", "your-supabase-key")

# Add proper error handling for Supabase initialization
try:
    supabase: Client = create_client(spb_url, spb_key)
    logger.info("Supabase client initialized successfully in braingraph_routes.py")
except Exception as e:
    logger.error(f"Failed to initialize Supabase client in braingraph_routes.py: {e}")

# Create FastAPI router
router = APIRouter()

# Define request/response models
class CreateGraphVersionRequest(BaseModel):
    graph_id: str = Field(..., description="UUID of the brain graph")
    brain_ids: List[str] = Field(default=[], description="List of brain UUIDs to include in this version")

class VersionStatusRequest(BaseModel):
    version_id: str = Field(..., description="UUID of the version to update")

class UpdateVersionBrainsRequest(BaseModel):
    version_id: str = Field(..., description="UUID of the version to update")
    action: str = Field(..., description="Action to perform: 'add' or 'remove'")
    brain_ids: List[str] = Field(..., description="List of brain UUIDs to add or remove")

# Helper function to handle OPTIONS requests
def handle_options():
    """Common OPTIONS handler for all endpoints."""
    return JSONResponse(
        content={},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
            "Access-Control-Max-Age": "86400"
        }
    )

@router.post('/create-graph-version')
async def create_graph_version(request: CreateGraphVersionRequest):
    """Create a new version for a brain graph"""
    # Validate UUID format
    try:
        UUID(request.graph_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid graph_id format - must be a valid UUID")
        
    # Validate brain_ids format
    for brain_id in request.brain_ids:
        try:
            UUID(brain_id)
        except ValueError:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid brain_id format - {brain_id} must be a valid UUID"
            )
    
    try:
        # Start a transaction
        response = supabase.rpc('next_version_number', {'graph_uuid': request.graph_id}).execute()
        if not response.data:
            raise HTTPException(status_code=500, detail="Failed to generate version number")
            
        version_number = response.data
        
        # Update the placeholder version with the actual brain_ids
        update_response = supabase.table("brain_graph_version")\
            .update({"brain_ids": request.brain_ids})\
            .eq("graph_id", request.graph_id)\
            .eq("version_number", version_number)\
            .execute()
            
        if not update_response.data:
            raise HTTPException(status_code=500, detail="Failed to update version with brain IDs")
            
        version_data = update_response.data[0]
        version = BrainGraphVersion(
            id=version_data["id"],
            graph_id=version_data["graph_id"],
            version_number=version_data["version_number"],
            brain_ids=version_data["brain_ids"],
            released_date=datetime.fromisoformat(version_data["released_date"].replace("Z", "+00:00")),
            status=version_data["status"]
        )
        
        return {
            "message": "Graph version created successfully",
            "version": {
                "id": version.id,
                "graph_id": version.graph_id,
                "version_number": version.version_number,
                "brain_ids": version.brain_ids,
                "status": version.status,
                "released_date": version.released_date.isoformat()
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating graph version: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@router.options('/create-graph-version')
async def create_graph_version_options():
    return handle_options()

@router.post('/release-graph-version')
async def release_graph_version(request: VersionStatusRequest):
    """Release/publish a graph version"""
    try:
        # Validate UUID format
        try:
            UUID(request.version_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid version_id format - must be a valid UUID")
            
        version = update_brain_graph_version_status(request.version_id, "published")
        return {
            "message": "Graph version published successfully",
            "version": {
                "id": version.id,
                "graph_id": version.graph_id,
                "version_number": version.version_number,
                "brain_ids": version.brain_ids,
                "status": version.status,
                "released_date": version.released_date.isoformat()
            }
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error releasing graph version: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@router.options('/release-graph-version')
async def release_graph_version_options():
    return handle_options()

@router.post('/revoke-graph-version')
async def revoke_graph_version(request: VersionStatusRequest):
    """Revoke/unpublish a graph version"""
    try:
        # Validate UUID format
        try:
            UUID(request.version_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid version_id format - must be a valid UUID")
            
        version = update_brain_graph_version_status(request.version_id, "training")
        return {
            "message": "Graph version de-published successfully",
            "version": {
                "id": version.id,
                "graph_id": version.graph_id,
                "version_number": version.version_number,
                "brain_ids": version.brain_ids,
                "status": version.status,
                "revoked_date": version.released_date.isoformat()
            }
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error revoking graph version: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@router.options('/revoke-graph-version')
async def revoke_graph_version_options():
    return handle_options()

@router.post('/update-graph-version-brains')
async def update_graph_version_brains(request: UpdateVersionBrainsRequest):
    """Add or remove brains from a graph version"""
    # Validate action
    if request.action not in ["add", "remove"]:
        raise HTTPException(status_code=400, detail="action must be either 'add' or 'remove'")
    
    # Validate UUID format
    try:
        UUID(request.version_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid version_id format - must be a valid UUID")
        
    # Validate brain_ids format
    for brain_id in request.brain_ids:
        try:
            UUID(brain_id)
        except ValueError:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid brain_id format - {brain_id} must be a valid UUID"
            )
    
    try:
        # Check version status before attempting modification
        response = supabase.table("brain_graph_version")\
            .select("status")\
            .eq("id", request.version_id)\
            .execute()
            
        if not response.data:
            raise HTTPException(status_code=404, detail="Version not found")
            
        status = response.data[0].get("status")
        if status == "published":
            raise HTTPException(status_code=400, detail="Cannot modify a published version")
            
        if request.action == "add":
            version = add_brains_to_version(request.version_id, request.brain_ids)
        else:
            version = remove_brains_from_version(request.version_id, request.brain_ids)
            
        return {
            "message": f"Brain IDs {request.action}ed successfully",
            "version": {
                "id": version.id,
                "graph_id": version.graph_id,
                "version_number": version.version_number,
                "brain_ids": version.brain_ids,
                "status": version.status,
                "released_date": version.released_date.isoformat()
            }
        }
    except HTTPException:
        raise
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error updating graph version brains: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@router.options('/update-graph-version-brains')
async def update_graph_version_brains_options():
    return handle_options()

@router.get('/get-version-brains')
async def get_version_brains(version_id: str):
    """Get brains for a specific version"""
    if not version_id:
        raise HTTPException(status_code=400, detail="version_id parameter is required")
    
    try:
        UUID(version_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid version_id format - must be a valid UUID")
    
    try:
        response = supabase.table("brain_graph_version")\
            .select("brain_ids")\
            .eq("id", version_id)\
            .execute()
        
        if not response.data:
            raise HTTPException(status_code=404, detail=f"No version found with id {version_id}")
        
        brain_ids = response.data[0]["brain_ids"]
        
        # Get brain details for each UUID
        brains = []
        if brain_ids:
            brain_response = supabase.table("brain")\
                .select("*")\
                .in_("id", brain_ids)\
                .execute()
            
            if brain_response.data:
                brains = brain_response.data
        
        return {
            "version_id": version_id,
            "brains": brains
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting version brains: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@router.options('/get-version-brains')
async def get_version_brains_options():
    return handle_options()