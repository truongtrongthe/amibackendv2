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

from braindb import get_brains, get_brain_details, update_brain, create_brain, get_organization, create_organization, update_organization
from contactconvo import ConversationManager
from chatwoot import handle_message_created, handle_message_updated, handle_conversation_created
from braingraph import (
    create_brain_graph, get_brain_graph,
    get_brain_graph_versions,
    BrainGraphVersion
)
from supabase import create_client, Client
from utilities import logger
from enrich_profile import ProfileEnricher

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
            from ami import convo_stream
            
            # Get the stream
            stream = convo_stream(
                user_input=request.user_input, 
                user_id=request.user_id, 
                thread_id=request.thread_id, 
                mode="pilot"
            )
            
            # Process and yield each result
            async for item in stream:
                if isinstance(item, str) and item.startswith("data: "):
                    # Already formatted for SSE
                    yield item + "\n"
                elif isinstance(item, dict):
                    # Format JSON for SSE
                    yield f"data: {json.dumps(item)}\n\n"
                else:
                    # For string responses without SSE format
                    yield f"data: {json.dumps({'message': item})}\n\n"
                    
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

@router.get('/get-org-detail/{orgid}')
async def get_org_detail(orgid: str):
    """Get details for a specific organization"""
    if not orgid:
        raise HTTPException(status_code=400, detail="orgid is required")
    
    try:
        org = get_organization(orgid)
        if not org:
            raise HTTPException(status_code=404, detail=f"No organization found with id {orgid}")
        
        org_data = {
            "id": org.id,
            "org_id": org.org_id,
            "name": org.name,
            "description": org.description,
            "email": org.email,
            "phone": org.phone,
            "address": org.address,
            "created_date": org.created_date.isoformat()
        }
        return {"organization": org_data}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.options('/get-org-detail/{orgid}')
async def get_org_detail_options(orgid: str):
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

@router.post('/webhook/chatwoot')
async def chatwoot_webhook(data: dict, request: Request):
    """Handle Chatwoot webhook events"""
    logger.info(f"→ RECEIVED WEBHOOK - Path: {request.url.path}, Query Params: {request.query_params}")
    logger.info(f"→ HEADERS: {dict(request.headers)}")
    
    try:
        # Log parsed JSON data
        logger.info(f"→ PARSED JSON: {json.dumps(data)[:1000]}...")  # Log first 1000 chars
        event = data.get('event')
        logger.info(f"→ EVENT TYPE: {event}")
        
        # Get organization_id from query parameters or headers
        organization_id = request.query_params.get('organization_id')
        logger.info(f"→ QUERY PARAM organization_id: {organization_id}")
        
        if not organization_id:
            organization_id = request.headers.get('x-organization-id')
            logger.info(f"→ HEADER X-Organization-Id: {organization_id}")
        
        # Extract inbox information
        inbox = data.get('inbox', {})
        inbox_id = str(inbox.get('id', ''))
        facebook_page_id = inbox.get('channel', {}).get('facebook_page_id', 'N/A')
        
        logger.info(f"→ INBOX DETAILS - ID: {inbox_id}, Facebook Page ID: {facebook_page_id}")
        
        # Create a unique ID for this webhook to detect duplicates
        request_id = f"{event}_{data.get('id', '')}_{inbox_id}_{datetime.now().timestamp()}"
        
        # Check for duplicates
        if request_id in recent_requests:
            logger.info(f"→ DUPLICATE detected! Ignoring: {request_id}")
            return {"status": "success", "message": "Duplicate request ignored"}
        
        recent_requests.append(request_id)
        logger.info(f"→ Added to recent_requests: {request_id}")
        
        # Log basic information
        logger.info(
            f"→ WEBHOOK SUMMARY - Event: {event}, Inbox: {inbox_id}, "
            f"Page: {facebook_page_id}, Organization: {organization_id or 'Not provided'}"
        )
        
        # Validate against inbox mapping if available
        if INBOX_MAPPING and inbox_id:
            logger.info(f"→ CHECKING inbox mapping for inbox_id: {inbox_id}")
            inbox_config = INBOX_MAPPING.get(inbox_id)
            if inbox_config:
                logger.info(f"→ FOUND inbox config: {inbox_config}")
                expected_organization_id = inbox_config['organization_id']
                expected_facebook_page_id = inbox_config['facebook_page_id']
                
                # If organization_id was not provided, use the one from mapping
                if not organization_id:
                    organization_id = expected_organization_id
                    logger.info(f"→ USING organization_id {organization_id} from inbox mapping")
                
                # If provided, validate that it matches what's expected
                elif organization_id != expected_organization_id:
                    logger.warning(
                        f"→ ORGANIZATION MISMATCH: provided={organization_id}, "
                        f"expected={expected_organization_id} for inbox {inbox_id}"
                    )
                    raise HTTPException(
                        status_code=400, 
                        detail="Organization ID does not match inbox configuration"
                    )
                
                # Validate Facebook page ID if available
                if facebook_page_id != 'N/A' and facebook_page_id != expected_facebook_page_id:
                    logger.warning(
                        f"→ FACEBOOK PAGE MISMATCH: actual={facebook_page_id}, "
                        f"expected={expected_facebook_page_id} for inbox {inbox_id}"
                    )
            else:
                logger.warning(f"→ NO MAPPING found for inbox_id: {inbox_id}")
        
        # Handle different event types
        if event == "message_created":
            logger.info(f"→ PROCESSING message_created event for organization: {organization_id or 'None'}")
            await handle_message_created(data, organization_id)
        elif event == "message_updated":
            logger.info(f"→ PROCESSING message_updated event for organization: {organization_id or 'None'}")
            await handle_message_updated(data, organization_id)
        elif event == "conversation_created":
            logger.info(f"→ PROCESSING conversation_created event for organization: {organization_id or 'None'}")
            await handle_conversation_created(data, organization_id)
        else:
            logger.info(f"→ UNHANDLED event type: {event}")
        
        logger.info("→ WEBHOOK PROCESSING SUCCESSFUL - Returning 200 OK")
        return {
            "status": "success", 
            "message": f"Processed {event} event", 
            "organization_id": organization_id,
            "inbox_id": inbox_id
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"→ CRITICAL ERROR processing webhook: {str(e)}")
        logger.error(f"→ STACK TRACE: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@router.options('/webhook/chatwoot')
async def chatwoot_webhook_options():
    return handle_options()

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
            raise HTTPException(status_code=404, detail="No brain graph exists for this organization")
        
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