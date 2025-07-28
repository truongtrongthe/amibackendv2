#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from fastapi import APIRouter, HTTPException, Request, Depends, BackgroundTasks, Response
from fastapi.responses import JSONResponse, StreamingResponse
from typing import Dict, List, Optional, Any, AsyncGenerator
from pydantic import BaseModel, Field
from uuid import UUID, uuid4
from datetime import datetime
import json
import time
import os
import traceback
import asyncio

# Import run_async_in_thread from the utils module
from async_utils import run_async_in_thread

# Keep track of recent webhook requests to detect duplicates
from collections import deque
recent_requests = deque(maxlen=1000)

# Import dependencies

from orgdb import get_brains, get_brain_details, update_brain, create_brain
from contactconvo import ConversationManager
# Import the router from chatwoot instead of individual functions
from chatwoot import router as chatwoot_router
from braingraph import (
    create_brain_graph, get_brain_graph,
    get_brain_graph_versions,
    BrainGraphVersion
)
from supabase import create_client, Client
from utilities import logger


# Load inbox mapping for Chatwoot
INBOX_MAPPING = {}
try:
    with open('inbox_mapping.json', 'r') as f:
        mapping_data = json.load(f)
        # Create a lookup dictionary by inbox_id
        for inbox in mapping_data.get('inboxes', []):
            INBOX_MAPPING[inbox['inbox_id']] = {
                'organization_id': inbox['organization_id'],
                'facebook_page_id': inbox['facebook_page_id'],
                'page_name': inbox['page_name']
            }
    logger.info(f"Loaded {len(INBOX_MAPPING)} inbox mappings")
except Exception as e:
    logger.error(f"Failed to load inbox_mapping.json: {e}")
    logger.warning("Chatwoot webhook will operate without inbox validation")

# Supabase initialization
spb_url = os.getenv("SUPABASE_URL", "https://example.supabase.co")
spb_key = os.getenv("SUPABASE_KEY", "your-supabase-key")

# Add proper error handling for Supabase initialization
try:
    supabase: Client = create_client(spb_url, spb_key)
    logger.info("Supabase client initialized successfully in fastapi_routes.py")
except Exception as e:
    logger.error(f"Failed to initialize Supabase client in fastapi_routes.py: {e}")

# Create ConversationManager instance
convo_mgr = ConversationManager()

# Create FastAPI router
router = APIRouter()

# Define request/response models
class PilotRequest(BaseModel):
    user_input: str
    user_id: str = "thefusionlab"
    thread_id: str = "pilot_thread"

class BrainGraphRequest(BaseModel):
    org_id: str
    name: str
    description: Optional[str] = None

class ActivateBrainRequest(BaseModel):
    graph_version_id: str

class ChatwootWebhookData(BaseModel):
    event: str
    inbox: Optional[Dict[str, Any]] = None
    id: Optional[str] = None

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

# Define endpoints
@router.post('/pilot')
async def pilot(request: PilotRequest, background_tasks: BackgroundTasks):
    """
    Handle pilot requests with thread-based async processing
    """
    logger.info("Pilot API called!")
    
    # Define the async process function
    async def process_stream():
        """Process the stream and return all items."""
        try:
            # Import directly here to ensure fresh imports
            # from ami import convo_stream  # Archived
            
            # Get the stream - Archived functionality
            # stream = convo_stream(
            #     user_input=request.user_input, 
            #     user_id=request.user_id, 
            #     thread_id=request.thread_id, 
            #     mode="pilot"
            # )
            
            # Process and yield each result - Archived functionality
            # async for item in stream:
            #     if isinstance(item, str) and item.startswith("data: "):
            #         # Already formatted for SSE
            #         yield item + "\n"
            #     elif isinstance(item, dict):
            #         # Format JSON for SSE
            #         yield f"data: {json.dumps(item)}\n\n"
            #     else:
            #         # For string responses without SSE format
            #         yield f"data: {json.dumps({'message': item})}\n\n"
            
            # Return archived message
            yield f"data: {json.dumps({'message': 'This endpoint has been archived. Please use the new tool endpoints instead.'})}\n\n"
                    
        except Exception as e:
            error_msg = f"Error processing stream: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            yield f"data: {json.dumps({'error': error_msg})}\n\n"
    
    # Return a streaming response
    return StreamingResponse(
        process_stream(),
        media_type="text/event-stream",
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache",
            "Access-Control-Allow-Origin": "*"
        }
    )

@router.options('/pilot')
async def pilot_options():
    return handle_options()

@router.get('/brains')
async def brains(org_id: str):
    """Get all brains for an organization"""
    if not org_id:
        raise HTTPException(status_code=400, detail="org_id parameter is required")
    
    try:
        brains_list = get_brains(org_id)
        if not brains_list:
            return {"message": "No brains found for this organization", "brains": []}
        
        # Convert Brain objects to dictionaries for JSON serialization
        brains_data = [{
            "id": brain.id,
            "brain_id": brain.brain_id,
            "org_id": brain.org_id,
            "name": brain.name,
            "status": brain.status,
            "bank_name": brain.bank_name,
            "summary": brain.summary,
            "created_date": brain.created_date
        } for brain in brains_list]
        
        return {"brains": brains_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.options('/brains')
async def brains_options():
    return handle_options()

from contact import ContactManager
cm = ContactManager()

@router.get('/contacts')
async def get_all_contacts(organization_id: str):
    """Get all contacts for an organization"""
    if not organization_id:
        raise HTTPException(status_code=400, detail="organization_id parameter is required")
    
    try:
        contacts = cm.get_contacts(organization_id)
        if not contacts:
            return {"message": "No contacts found", "contacts": []}
        
        return {"contacts": contacts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.options('/contacts')
async def contacts_options():
    return handle_options()

@router.get('/brain-details')
async def brain_details(brain_id: str):
    """Get details for a specific brain"""
    if not brain_id:
        raise HTTPException(status_code=400, detail="brain_id parameter is required")
    
    try:
        brain = get_brain_details(brain_id)  # Pass the UUID string directly
        if not brain:
            raise HTTPException(status_code=404, detail=f"No brain found with id {brain_id}")
        
        brain_data = {
            "id": brain.id,
            "brain_id": brain.brain_id,
            "org_id": brain.org_id,
            "name": brain.name,
            "status": brain.status,
            "bank_name": brain.bank_name,
            "summary": brain.summary,
            "created_date": brain.created_date
        }
        return {"brain": brain_data}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.options('/brain-details')
async def brain_details_options():
    return handle_options()

async def activate_brain_with_version(graph_version_id):
    """
    Activate the brain by setting its status to published in the database.
    
    Args:
        graph_version_id: The graph version ID to activate
        
    Returns:
        dict: A dictionary with activation results
    """
    try:
        # Update the brain_graph_version status to published
        response = supabase.table("brain_graph_version")\
            .update({"status": "published"})\
            .eq("id", graph_version_id)\
            .execute()
        
        if response.data:
            logger.info(f"Successfully activated brain graph version: {graph_version_id}")
            return {
                "success": True,
                "graph_version_id": graph_version_id,
                "loaded": True,
                "vector_count": 0  # Not applicable for simple activation
            }
        else:
            logger.error(f"No brain graph version found with id: {graph_version_id}")
            return {
                "success": False,
                "error": f"No brain graph version found with id: {graph_version_id}"
            }
            
    except Exception as e:
        logger.error(f"Error activating brain graph version {graph_version_id}: {str(e)}")
        return {
            "success": False,
            "error": f"Database error: {str(e)}"
        }

@router.post('/activate-brain')
async def activate_brain(request: ActivateBrainRequest):
    """
    Activate the brain with a specific graph version ID.
    This endpoint will load vectors for the specified graph version.
    """
    start_time = datetime.now()
    logger.info(f"[SESSION_TRACE] === BEGIN ACTIVATE BRAIN request at {start_time.isoformat()} ===")
    
    if not request.graph_version_id:
        raise HTTPException(status_code=400, detail="graph_version_id is required")
    
    try:
        # Call the centralized activate_brain_with_version function
        result = await activate_brain_with_version(request.graph_version_id)
        
        # Log completion
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()
        logger.info(f"[SESSION_TRACE] === END ACTIVATE BRAIN request - total time: {elapsed:.2f}s ===")
        
        if result["success"]:
            return {
                "message": "Brain activated successfully", 
                "graph_version_id": result["graph_version_id"],
                "loaded": result["loaded"],
                "elapsed_seconds": elapsed,
                "worker_id": "",
                "vector_count": result["vector_count"]
            }
        else:
            raise HTTPException(status_code=500, detail=result["error"])
            
    except Exception as e:
        # Handle any errors
        error_msg = f"Error in activate_brain: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_msg)

@router.options('/activate-brain')
async def activate_brain_options():
    return handle_options()

VERIFY_TOKEN = os.getenv("CALLBACK_V_TOKEN")

@router.get("/webhook")
async def verify_webhook(
    hub_mode: Optional[str] = None,
    hub_verify_token: Optional[str] = None,
    hub_challenge: Optional[str] = None,
    org_id: Optional[str] = None
):
    """Verify webhook for Facebook"""
    if hub_mode == "subscribe" and verify_webhook_token(hub_verify_token, org_id):
        logger.info(f"✅ Webhook verified by Facebook for organization: {org_id or 'default'}")
        return hub_challenge
    else:
        logger.info(f"❌ Webhook verification failed. Token: {hub_verify_token}, org_id: {org_id}")
        raise HTTPException(status_code=403, detail="Forbidden")

# Include the chatwoot router
router.include_router(chatwoot_router)

@router.post('/create-brain-graph')
async def create_brain_graph_endpoint(request_data: BrainGraphRequest):
    """Create a new brain graph"""
    try:
        brain_graph = create_brain_graph(
            request_data.org_id,
            request_data.name,
            request_data.description
        )
        
        return {
            "message": "Brain graph created successfully",
            "brain_graph": {
                "id": brain_graph.id,
                "org_id": brain_graph.org_id,
                "name": brain_graph.name,
                "description": brain_graph.description,
                "created_date": brain_graph.created_date.isoformat()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.options('/create-brain-graph')
async def create_brain_graph_options():
    return handle_options()

@router.get('/get-org-brain-graph')
async def get_org_brain_graph(org_id: str):
    """Get brain graph for an organization"""
    if not org_id:
        raise HTTPException(status_code=400, detail="org_id parameter is required")
    
    try:
        # Validate UUID format
        try:
            UUID(org_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid org_id format - must be a valid UUID")
            
        # Get the brain graph ID for this org
        response = supabase.table("brain_graph")\
            .select("id")\
            .eq("org_id", org_id)\
            .execute()
        
        if not response.data:
            return {
                "brain_graph": None,
                "message": "No brain graph exists for this organization",
                "org_id": org_id
            }
        
        graph_id = response.data[0]["id"]
        brain_graph = get_brain_graph(graph_id)
        
        if not brain_graph:
            raise HTTPException(status_code=404, detail="Brain graph not found")
        
        # Get the latest version
        versions = get_brain_graph_versions(graph_id)
        latest_version = None
        if versions:
            latest_version = {
                "id": versions[0].id,
                "version_number": versions[0].version_number,
                "brain_ids": versions[0].brain_ids,
                "status": versions[0].status,
                "released_date": versions[0].released_date.isoformat()
            }
        
        return {
            "brain_graph": {
                "id": brain_graph.id,
                "org_id": brain_graph.org_id,
                "name": brain_graph.name,
                "description": brain_graph.description,
                "created_date": brain_graph.created_date.isoformat(),
                "latest_version": latest_version
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.options('/get-org-brain-graph')
async def get_org_brain_graph_options():
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
        raise HTTPException(status_code=500, detail=str(e))

@router.options('/get-version-brains')
async def get_version_brains_options():
    return handle_options()

# Helper to verify webhook tokens
def verify_webhook_token(token, org_id):
    # Implementation depends on your verification logic
    return token == VERIFY_TOKEN 

# Add the new endpoint
@router.post("/tool/knowledge-approval")
async def handle_knowledge_approval(request: Request):
    """Handle human approval of knowledge pieces and return copilot-style summary"""
    try:
        # Parse request body
        body = await request.json()
        approved_knowledge_ids = body.get("approved_knowledge_ids", [])
        all_knowledge_pieces = body.get("all_knowledge_pieces", [])
        original_request = body.get("original_request", {})
        
        # Debug logging
        logger.info(f"Knowledge approval request body keys: {list(body.keys())}")
        logger.info(f"Approved knowledge IDs: {approved_knowledge_ids}")
        logger.info(f"All knowledge pieces count: {len(all_knowledge_pieces)}")
        logger.info(f"Original request keys: {list(original_request.keys()) if original_request else 'None'}")
        logger.info(f"Original request org_id: {original_request.get('org_id', 'NOT_FOUND')}")
        logger.info(f"Body content: {json.dumps(body, indent=2)[:500]}...")
        
        # Import ExecutiveTool
        from exec_tool import ExecutiveTool, ToolExecutionRequest
        
        # Create executive tool and request
        executive_tool = ExecutiveTool()
        
        # Try to get org_id from multiple possible sources
        org_id = (
            original_request.get("org_id") or 
            body.get("org_id") or 
            "5376f68e-ff74-43fe-a72c-c1be9c6ae652"  # Fallback to your known org_id
        )
        
        user_id = (
            original_request.get("user_id") or 
            body.get("user_id") or 
            "user_GjswTZUz8Fhv2v2X0c8OyA"  # Fallback to your known user_id
        )
        
        logger.info(f"Using org_id: {org_id}, user_id: {user_id}")
        
        # Safely construct ToolExecutionRequest with required fields
        tool_request = ToolExecutionRequest(
            llm_provider=original_request.get("llm_provider", "openai"),
            user_query=original_request.get("user_query", ""),
            system_prompt=original_request.get("system_prompt"),
            model=original_request.get("model"),
            model_params=original_request.get("model_params"),
            tools_config=original_request.get("tools_config"),
            org_id=org_id,
            user_id=user_id,
            enable_tools=original_request.get("enable_tools", True),
            force_tools=original_request.get("force_tools", False),
            tools_whitelist=original_request.get("tools_whitelist"),
            conversation_history=original_request.get("conversation_history"),
            max_history_messages=original_request.get("max_history_messages", 25),
            max_history_tokens=original_request.get("max_history_tokens", 6000)
        )
        
        # Return streaming response with copilot summary
        return StreamingResponse(
            executive_tool.handle_knowledge_approval(
                approved_knowledge_ids,
                all_knowledge_pieces, 
                tool_request
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                "Access-Control-Allow-Headers": "*",
            }
        )
        
    except Exception as e:
        logger.error(f"Knowledge approval endpoint error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Knowledge approval failed: {str(e)}"}
        ) 

