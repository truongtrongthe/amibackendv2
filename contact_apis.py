#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from uuid import UUID, uuid4
from datetime import datetime
import json
import os
import traceback

# Import dependencies
from contact import ContactManager
from contactconvo import ConversationManager
from contact_analysis import ContactAnalyzer
from supabase import create_client, Client
from utilities import logger

# Supabase initialization
spb_url = os.getenv("SUPABASE_URL", "https://example.supabase.co")
spb_key = os.getenv("SUPABASE_KEY", "your-supabase-key")

# Add proper error handling for Supabase initialization
try:
    supabase: Client = create_client(spb_url, spb_key)
    logger.info("Supabase client initialized successfully in contact_apis.py")
except Exception as e:
    logger.error(f"Failed to initialize Supabase client in contact_apis.py: {e}")

# Create FastAPI router
router = APIRouter()

# Initialize managers
cm = ContactManager()
convo_mgr = ConversationManager()
contact_analyzer = ContactAnalyzer()

# Define request/response models
class UpdateContactRequest(BaseModel):
    id: int = Field(..., description="Contact ID to update")
    organization_id: Optional[str] = Field(None, description="Organization ID")
    type: Optional[str] = Field(None, description="Contact type")
    first_name: Optional[str] = Field(None, description="First name")
    last_name: Optional[str] = Field(None, description="Last name")
    email: Optional[str] = Field(None, description="Email address")
    phone: Optional[str] = Field(None, description="Phone number")
    facebook_id: Optional[str] = Field(None, description="Facebook ID")
    profile_picture_url: Optional[str] = Field(None, description="Profile picture URL")

class CreateContactRequest(BaseModel):
    organization_id: str = Field(..., description="Organization ID")
    type: str = Field(..., description="Contact type")
    first_name: str = Field(..., description="First name")
    last_name: str = Field(..., description="Last name")
    email: Optional[str] = Field(None, description="Email address")
    phone: Optional[str] = Field(None, description="Phone number")
    facebook_id: Optional[str] = Field(None, description="Facebook ID")
    profile_picture_url: Optional[str] = Field(None, description="Profile picture URL")

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

@router.get('/contact-details')
async def contact_details(
    contact_id: int = Query(..., description="Contact ID to retrieve"),
    organization_id: Optional[str] = Query(None, description="Organization ID")
):
    """Get details for a specific contact"""
    try:
        contact = cm.get_contact_details(contact_id, organization_id)
        if not contact:
            raise HTTPException(status_code=404, detail=f"No contact found with contact_id {contact_id}")
        
        contact_data = {
            "id": contact["id"],
            "uuid": contact["uuid"],
            "organization_id": contact.get("organization_id"),
            "type": contact["type"],
            "first_name": contact["first_name"],
            "last_name": contact["last_name"],
            "email": contact["email"],
            "phone": contact["phone"],
            "facebook_id": contact.get("facebook_id"),
            "profile_picture_url": contact.get("profile_picture_url"),
            "created_at": contact["created_at"],
            "profile": contact.get("profiles", None)
        }
        return {"contact": contact_data}
    except Exception as e:
        logger.error(f"Error getting contact details: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@router.options('/contact-details')
async def contact_details_options():
    return handle_options()

@router.post('/update-contact')
async def update_contact_endpoint(request: UpdateContactRequest):
    """Update an existing contact"""
    try:
        update_data = {}
        if request.type:
            update_data["type"] = request.type
        if request.first_name:
            update_data["first_name"] = request.first_name
        if request.last_name:
            update_data["last_name"] = request.last_name
        if request.email is not None:
            update_data["email"] = request.email
        if request.phone is not None:
            update_data["phone"] = request.phone
        if request.facebook_id is not None:
            update_data["facebook_id"] = request.facebook_id
        if request.profile_picture_url is not None:
            update_data["profile_picture_url"] = request.profile_picture_url
        
        updated_contact = cm.update_contact(request.id, request.organization_id, **update_data)
        if not updated_contact:
            raise HTTPException(status_code=404, detail=f"No contact found with id {request.id}")
        
        contact_data = {
            "id": updated_contact["id"],
            "uuid": updated_contact["uuid"],
            "organization_id": updated_contact.get("organization_id"),
            "type": updated_contact["type"],
            "first_name": updated_contact["first_name"],
            "last_name": updated_contact["last_name"],
            "email": updated_contact["email"],
            "phone": updated_contact["phone"],
            "facebook_id": updated_contact.get("facebook_id"),
            "profile_picture_url": updated_contact.get("profile_picture_url"),
            "created_at": updated_contact["created_at"]
        }
        return {"message": "Contact updated successfully", "contact": contact_data}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error updating contact: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@router.options('/update-contact')
async def update_contact_options():
    return handle_options()

@router.post('/create-contact')
async def create_contact_endpoint(request: CreateContactRequest):
    """Create a new contact"""
    try:
        new_contact = cm.create_contact(
            request.organization_id,
            request.type, 
            request.first_name, 
            request.last_name, 
            request.email, 
            request.phone, 
            request.facebook_id, 
            request.profile_picture_url
        )
        contact_data = {
            "id": new_contact["id"],
            "uuid": new_contact["uuid"],
            "organization_id": new_contact.get("organization_id"),
            "type": new_contact["type"],
            "first_name": new_contact["first_name"],
            "last_name": new_contact["last_name"],
            "email": new_contact["email"],
            "phone": new_contact["phone"],
            "facebook_id": new_contact.get("facebook_id"),
            "profile_picture_url": new_contact.get("profile_picture_url"),
            "created_at": new_contact["created_at"]
        }
        return {"message": "Contact created successfully", "contact": contact_data}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error creating contact: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@router.options('/create-contact')
async def create_contact_options():
    return handle_options()

@router.get('/contacts')
async def get_all_contacts(organization_id: Optional[str] = Query(None)):
    """Get all contacts, optionally filtered by organization"""
    try:
        contacts = cm.get_contacts(organization_id)
        if not contacts:
            return {"message": "No contacts found", "contacts": []}
        
        return {"contacts": contacts}
    except Exception as e:
        logger.error(f"Error getting contacts: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@router.options('/contacts')
async def contacts_options():
    return handle_options()



