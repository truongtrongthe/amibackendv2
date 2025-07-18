#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Chat API Routes for AMI Backend

This module provides FastAPI endpoints for the chat system including:
- Creating and managing chat sessions
- Sending and retrieving messages
- Chat statistics and management

Built by: The Fusion Lab
Date: January 2025
"""

import asyncio
import traceback
from typing import Dict, List, Optional, Any
from datetime import datetime
from uuid import UUID
import re

from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from chat import (
    ChatManager, MessageManager, 
    create_chat_with_first_message, get_chat_statistics,
    # Import all Pydantic models
    CreateChatRequest, UpdateChatRequest, CreateMessageRequest, UpdateMessageRequest,
    UpdateMessageMetadataRequest, PostThoughtsRequest, CreateChatWithMessageRequest, GetChatsRequest, GetMessagesRequest,
    ChatResponse, MessageResponse, ChatWithMessagesResponse, ChatListResponse,
    MessageListResponse, ChatStatsResponse, CreateChatWithMessageResponse,
    StandardResponse, ChatRole, ChatStatus, MessageType, ChatType
)
from utilities import logger

# Create FastAPI router
router = APIRouter(prefix="/api/chats", tags=["chats"])

# Initialize managers
chat_manager = ChatManager()
message_manager = MessageManager()

# UUID validation function
def validate_uuid(uuid_string: str, field_name: str = "ID") -> None:
    """Validate UUID format and provide helpful error messages"""
    if not uuid_string:
        raise HTTPException(status_code=400, detail=f"{field_name} is required")
    
    # Check if it's a valid UUID format
    uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
    if not re.match(uuid_pattern, uuid_string.lower()):
        if uuid_string.startswith('user_'):
            user_id = uuid_string
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid {field_name}: '{uuid_string}' appears to be a user ID, not a message ID. Please use a valid message UUID. To find message IDs for this user, call: GET /api/chats/users/{user_id}/recent-messages"
            )
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid {field_name} format: '{uuid_string}'. Expected UUID format (e.g., 123e4567-e89b-12d3-a456-426614174000)"
            )

# Dependency for error handling
async def handle_exceptions(func):
    """Decorator for handling exceptions in API endpoints"""
    try:
        return await func()
    except Exception as e:
        logger.error(f"API Error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# Chat endpoints
@router.post("/", response_model=StandardResponse)
async def create_chat(request: CreateChatRequest):
    """Create a new chat session"""
    try:
        chat = chat_manager.create_chat(
            user_id=request.user_id,
            title=request.title,
            org_id=request.org_id,
            chat_type=request.chat_type.value,
            metadata=request.metadata
        )
        
        if chat:
            return StandardResponse(
                success=True,
                message="Chat created successfully",
                data=chat
            )
        else:
            raise HTTPException(status_code=400, detail="Failed to create chat")
            
    except Exception as e:
        logger.error(f"Error creating chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{chat_id}", response_model=StandardResponse)
async def get_chat(chat_id: str, user_id: str = Query(None)):
    """Get a specific chat by ID"""
    try:
        chat = chat_manager.get_chat(chat_id, user_id)
        
        if chat:
            return StandardResponse(
                success=True,
                message="Chat retrieved successfully",
                data=chat
            )
        else:
            raise HTTPException(status_code=404, detail="Chat not found")
            
    except Exception as e:
        logger.error(f"Error getting chat {chat_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{chat_id}/with-messages", response_model=StandardResponse)
async def get_chat_with_messages(chat_id: str, user_id: str = Query(None)):
    """Get a chat with all its messages"""
    try:
        chat_with_messages = chat_manager.get_chat_with_messages(chat_id, user_id)
        
        if chat_with_messages:
            return StandardResponse(
                success=True,
                message="Chat with messages retrieved successfully",
                data=chat_with_messages
            )
        else:
            raise HTTPException(status_code=404, detail="Chat not found")
            
    except Exception as e:
        logger.error(f"Error getting chat with messages {chat_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/", response_model=StandardResponse)
async def get_user_chats(
    user_id: str = Query(..., description="User ID"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of chats"),
    offset: int = Query(0, ge=0, description="Number of chats to skip"),
    status: Optional[ChatStatus] = Query(None, description="Status filter"),
    org_id: Optional[str] = Query(None, description="Organization filter")
):
    """Get all chats for a user"""
    try:
        chats = chat_manager.get_user_chats(
            user_id=user_id,
            limit=limit,
            offset=offset,
            status=status.value if status else None,
            org_id=org_id
        )
        
        return StandardResponse(
            success=True,
            message=f"Retrieved {len(chats)} chats",
            data={
                "chats": chats,
                "total": len(chats),
                "limit": limit,
                "offset": offset
            }
        )
        
    except Exception as e:
        logger.error(f"Error getting chats for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/{chat_id}", response_model=StandardResponse)
async def update_chat(chat_id: str, request: UpdateChatRequest):
    """Update a chat"""
    try:
        # Validate chat_id if provided in request body matches URL path
        if request.chat_id and request.chat_id != chat_id:
            raise HTTPException(status_code=400, detail="Chat ID in request body does not match URL path")
        
        updated_chat = chat_manager.update_chat(
            chat_id=chat_id,
            user_id=request.user_id,
            title=request.title,
            status=request.status.value if request.status else None,
            metadata=request.metadata
        )
        
        if updated_chat:
            return StandardResponse(
                success=True,
                message="Chat updated successfully",
                data=updated_chat
            )
        else:
            raise HTTPException(status_code=404, detail="Chat not found or not updated")
            
    except Exception as e:
        logger.error(f"Error updating chat {chat_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# OPTIONS requests are handled automatically by CORS middleware in main.py

@router.delete("/{chat_id}", response_model=StandardResponse)
async def delete_chat(chat_id: str, user_id: str = Query(None)):
    """Delete a chat and all its messages"""
    try:
        deleted = chat_manager.delete_chat(chat_id, user_id)
        
        if deleted:
            return StandardResponse(
                success=True,
                message="Chat deleted successfully",
                data={"chat_id": chat_id}
            )
        else:
            raise HTTPException(status_code=404, detail="Chat not found or not deleted")
            
    except Exception as e:
        logger.error(f"Error deleting chat {chat_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Message endpoints
@router.post("/{chat_id}/messages", response_model=StandardResponse)
async def create_message(chat_id: str, request: CreateMessageRequest):
    """Create a new message in a chat"""
    try:
        # Override chat_id from URL
        request.chat_id = chat_id
        
        message = message_manager.create_message(
            chat_id=request.chat_id,
            user_id=request.user_id,
            role=request.role.value,
            content=request.content,
            message_type=request.message_type.value,
            metadata=request.metadata,
            thread_id=request.thread_id,
            parent_message_id=request.parent_message_id,
            token_count=request.token_count
        )
        
        if message:
            return StandardResponse(
                success=True,
                message="Message created successfully",
                data=message
            )
        else:
            raise HTTPException(status_code=400, detail="Failed to create message")
            
    except Exception as e:
        logger.error(f"Error creating message in chat {chat_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{chat_id}/messages", response_model=StandardResponse)
async def get_chat_messages(
    chat_id: str,
    limit: int = Query(100, ge=1, le=200, description="Maximum number of messages"),
    offset: int = Query(0, ge=0, description="Number of messages to skip"),
    role: Optional[ChatRole] = Query(None, description="Role filter"),
    since: Optional[datetime] = Query(None, description="Messages after this time")
):
    """Get messages for a specific chat"""
    try:
        messages = message_manager.get_chat_messages(
            chat_id=chat_id,
            limit=limit,
            offset=offset,
            role=role.value if role else None,
            since=since
        )
        
        return StandardResponse(
            success=True,
            message=f"Retrieved {len(messages)} messages",
            data={
                "messages": messages,
                "total": len(messages),
                "limit": limit,
                "offset": offset
            }
        )
        
    except Exception as e:
        logger.error(f"Error getting messages for chat {chat_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.options("/messages/{message_id}")
async def message_options(message_id: str):
    """Handle OPTIONS request for message operations - no UUID validation needed
    
    This must come BEFORE the GET handler to ensure OPTIONS requests 
    don't trigger UUID validation from the GET endpoint.
    """
    logger.info(f"OPTIONS request received for message ID: {message_id}")
    return JSONResponse(
        content={"message": "OK"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, PUT, PATCH, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        }
    )

@router.get("/messages/{message_id}", response_model=StandardResponse)
async def get_message(message_id: str):
    """Get a specific message by ID"""
    try:
        # Validate message ID format
        validate_uuid(message_id, "Message ID")
        
        message = message_manager.get_message(message_id)
        
        if message:
            return StandardResponse(
                success=True,
                message="Message retrieved successfully",
                data=message
            )
        else:
            raise HTTPException(status_code=404, detail="Message not found")
            
    except HTTPException:
        # Re-raise HTTPException as-is
        raise
    except Exception as e:
        logger.error(f"Error getting message {message_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/messages/{message_id}", response_model=StandardResponse)
async def update_message(message_id: str, request: UpdateMessageRequest):
    """Update a message"""
    try:
        updated_message = message_manager.update_message(
            message_id=message_id,
            content=request.content,
            metadata=request.metadata,
            token_count=request.token_count
        )
        
        if updated_message:
            return StandardResponse(
                success=True,
                message="Message updated successfully",
                data=updated_message
            )
        else:
            raise HTTPException(status_code=404, detail="Message not found or not updated")
            
    except Exception as e:
        logger.error(f"Error updating message {message_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.patch("/messages/{message_id}", response_model=StandardResponse)
async def update_message_metadata(message_id: str, request: UpdateMessageMetadataRequest):
    """Update message metadata for storing thoughts and other supplementary data"""
    try:
        # Validate message ID format
        validate_uuid(message_id, "Message ID")
        
        # First, get the existing message to retrieve current metadata
        existing_message = message_manager.get_message(message_id)
        
        if not existing_message:
            raise HTTPException(status_code=404, detail="Message not found")
        
        # Merge new metadata with existing metadata
        existing_metadata = existing_message.get("metadata", {}) or {}
        merged_metadata = {**existing_metadata, **request.metadata}
        
        # Update the message with merged metadata
        updated_message = message_manager.update_message(
            message_id=message_id,
            metadata=merged_metadata
        )
        
        if updated_message:
            return StandardResponse(
                success=True,
                message="Message metadata updated successfully",
                data={
                    "id": message_id,
                    "metadata": updated_message.get("metadata", {})
                }
            )
        else:
            raise HTTPException(status_code=404, detail="Message not found or not updated")
            
    except HTTPException:
        # Re-raise HTTPException as-is
        raise
    except Exception as e:
        logger.error(f"Error updating message metadata {message_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update message metadata: {str(e)}")

@router.delete("/messages/{message_id}", response_model=StandardResponse)
async def delete_message(message_id: str):
    """Delete a message"""
    try:
        deleted = message_manager.delete_message(message_id)
        
        if deleted:
            return StandardResponse(
                success=True,
                message="Message deleted successfully",
                data={"message_id": message_id}
            )
        else:
            raise HTTPException(status_code=404, detail="Message not found or not deleted")
            
    except Exception as e:
        logger.error(f"Error deleting message {message_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/messages/{message_id}/thoughts", response_model=StandardResponse)
async def get_message_thoughts(message_id: str):
    """Get thoughts data from message metadata"""
    try:
        # Validate message ID format
        validate_uuid(message_id, "Message ID")
        
        # Get the message
        message = message_manager.get_message(message_id)
        
        if not message:
            raise HTTPException(status_code=404, detail="Message not found")
        
        # Extract thoughts from metadata
        metadata = message.get("metadata", {}) or {}
        thoughts = metadata.get("thoughts", [])
        
        return StandardResponse(
            success=True,
            message="Thoughts retrieved successfully",
            data={
                "message_id": message_id,
                "thoughts": thoughts,
                "total_thoughts": len(thoughts)
            }
        )
        
    except HTTPException:
        # Re-raise HTTPException as-is
        raise
    except Exception as e:
        logger.error(f"Error getting thoughts for message {message_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve thoughts: {str(e)}")

@router.post("/messages/{message_id}/thoughts", response_model=StandardResponse)
async def save_message_thoughts(message_id: str, request: PostThoughtsRequest):
    """Save thoughts data to message metadata"""
    try:
        # Validate message ID format
        validate_uuid(message_id, "Message ID")
        
        # Get the existing message to retrieve current metadata
        existing_message = message_manager.get_message(message_id)
        
        if not existing_message:
            raise HTTPException(status_code=404, detail="Message not found")
        
        # Merge new thoughts with existing metadata
        existing_metadata = existing_message.get("metadata", {}) or {}
        merged_metadata = {**existing_metadata, "thoughts": request.thoughts}
        
        # Update the message with merged metadata
        updated_message = message_manager.update_message(
            message_id=message_id,
            metadata=merged_metadata
        )
        
        if updated_message:
            return StandardResponse(
                success=True,
                message="Thoughts saved successfully",
                data={
                    "message_id": message_id,
                    "thoughts": request.thoughts,
                    "total_thoughts": len(request.thoughts)
                }
            )
        else:
            raise HTTPException(status_code=404, detail="Message not found or not updated")
            
    except HTTPException:
        # Re-raise HTTPException as-is
        raise
    except Exception as e:
        logger.error(f"Error saving thoughts for message {message_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save thoughts: {str(e)}")

@router.get("/debug/message-id/{message_id}")
async def debug_message_id(message_id: str):
    """Debug endpoint to help identify message ID issues"""
    try:
        # Check if it's a valid UUID
        uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
        is_valid_uuid = re.match(uuid_pattern, message_id.lower()) is not None
        
        # Check if it looks like a user ID
        looks_like_user_id = message_id.startswith('user_')
        
        # Try to get the message
        message_exists = False
        message_data = None
        try:
            message_data = message_manager.get_message(message_id)
            message_exists = message_data is not None
        except Exception as e:
            message_exists = False
        
        return {
            "message_id": message_id,
            "is_valid_uuid": is_valid_uuid,
            "looks_like_user_id": looks_like_user_id,
            "message_exists": message_exists,
            "message_data": message_data,
            "suggestions": {
                "if_user_id": "Use a valid message UUID instead of user ID",
                "if_invalid_uuid": "Use format: 123e4567-e89b-12d3-a456-426614174000",
                "if_not_found": "Check if message exists in database"
            }
        }
    except Exception as e:
        return {
            "message_id": message_id,
            "error": str(e),
            "debug_info": "Exception occurred during debug"
        }

@router.get("/users/{user_id}/recent-messages")
async def get_user_recent_messages(user_id: str, limit: int = Query(10, ge=1, le=50)):
    """Get recent messages for a user - helpful for finding message IDs to attach thoughts to"""
    try:
        messages = message_manager.get_user_messages(
            user_id=user_id,
            limit=limit,
            offset=0
        )
        
        # Format the response to be helpful for frontend
        formatted_messages = []
        for msg in messages:
            formatted_messages.append({
                "message_id": msg.get("id"),
                "chat_id": msg.get("chat_id"),
                "role": msg.get("role"),
                "content": msg.get("content", "")[:100] + "..." if len(msg.get("content", "")) > 100 else msg.get("content", ""),
                "created_at": msg.get("created_at"),
                "has_thoughts": bool(msg.get("metadata", {}).get("thoughts"))
            })
        
        return StandardResponse(
            success=True,
            message=f"Retrieved {len(formatted_messages)} recent messages for user {user_id}",
            data={
                "user_id": user_id,
                "messages": formatted_messages,
                "total": len(formatted_messages),
                "note": "Use the 'message_id' field to save/retrieve thoughts for each message"
            }
        )
        
    except Exception as e:
        logger.error(f"Error getting recent messages for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Utility endpoints
@router.post("/create-with-message", response_model=StandardResponse)
async def create_chat_with_message(request: CreateChatWithMessageRequest):
    """Create a new chat with an initial message"""
    try:
        result = create_chat_with_first_message(
            user_id=request.user_id,
            content=request.content,
            title=request.title,
            org_id=request.org_id,
            chat_type=request.chat_type.value,
            thread_id=request.thread_id
        )
        
        if result:
            return StandardResponse(
                success=True,
                message="Chat created with first message successfully",
                data=result
            )
        else:
            raise HTTPException(status_code=400, detail="Failed to create chat with message")
            
    except Exception as e:
        logger.error(f"Error creating chat with message: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/users/{user_id}/messages", response_model=StandardResponse)
async def get_user_messages(
    user_id: str,
    limit: int = Query(100, ge=1, le=200, description="Maximum number of messages"),
    offset: int = Query(0, ge=0, description="Number of messages to skip"),
    chat_id: Optional[str] = Query(None, description="Chat ID filter")
):
    """Get messages for a specific user"""
    try:
        messages = message_manager.get_user_messages(
            user_id=user_id,
            limit=limit,
            offset=offset,
            chat_id=chat_id
        )
        
        return StandardResponse(
            success=True,
            message=f"Retrieved {len(messages)} messages",
            data={
                "messages": messages,
                "total": len(messages),
                "limit": limit,
                "offset": offset
            }
        )
        
    except Exception as e:
        logger.error(f"Error getting messages for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/statistics", response_model=StandardResponse)
async def get_chat_stats(
    user_id: Optional[str] = Query(None, description="User ID filter"),
    org_id: Optional[str] = Query(None, description="Organization ID filter")
):
    """Get chat statistics"""
    try:
        stats = get_chat_statistics(user_id=user_id, org_id=org_id)
        
        return StandardResponse(
            success=True,
            message="Chat statistics retrieved successfully",
            data=stats
        )
        
    except Exception as e:
        logger.error(f"Error getting chat statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@router.get("/health", response_model=StandardResponse)
async def health_check():
    """Health check endpoint for the chat service"""
    try:
        return StandardResponse(
            success=True,
            message="Chat service is healthy",
            data={"status": "healthy", "timestamp": datetime.now().isoformat()}
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 