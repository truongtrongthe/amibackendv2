#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Chat System Module for AMI Backend

This module provides functionality for managing chat sessions and messages
similar to Cursor's chat interface. It includes:

- ChatManager: Handles chat session CRUD operations
- MessageManager: Handles message CRUD operations within chats
- Utility functions for chat operations

Built by: The Fusion Lab
Date: January 2025
"""

import os
import json
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from supabase import create_client, Client
from dotenv import load_dotenv
from utilities import logger
from pydantic import BaseModel, Field
from enum import Enum

# Load environment variables
load_dotenv()

# Initialize Supabase client
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")

if not supabase_url or not supabase_key:
    raise ValueError("SUPABASE_URL and SUPABASE_KEY environment variables must be set")

supabase: Client = create_client(supabase_url, supabase_key)

class ChatManager:
    """Manager class for handling chat operations"""
    
    def __init__(self):
        self.chats_table = "chats"
        self.messages_table = "messages"
    
    def create_chat(
        self, 
        user_id: str, 
        title: str = None, 
        org_id: str = None,
        chat_type: str = "conversation",
        metadata: Dict = None
    ) -> Dict:
        """
        Create a new chat session
        
        Args:
            user_id: ID of the user creating the chat
            title: Optional title for the chat
            org_id: Optional organization ID
            chat_type: Type of chat (conversation, support, etc.)
            metadata: Additional metadata for the chat
            
        Returns:
            The created chat record
        """
        try:
            chat_data = {
                "user_id": user_id,
                "title": title or f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                "org_id": org_id,
                "chat_type": chat_type,
                "status": "active",
                "metadata": metadata or {},
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
            
            response = supabase.table(self.chats_table).insert(chat_data).execute()
            
            if response.data:
                logger.info(f"Created chat {response.data[0]['id']} for user {user_id}")
                return response.data[0]
            else:
                logger.error(f"Failed to create chat for user {user_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error creating chat: {str(e)}")
            raise
    
    def get_chat(self, chat_id: str, user_id: str = None) -> Dict:
        """
        Get a specific chat by ID
        
        Args:
            chat_id: ID of the chat to retrieve
            user_id: Optional user ID for access control
            
        Returns:
            The chat record or None if not found
        """
        try:
            query = supabase.table(self.chats_table).select("*").eq("id", chat_id)
            
            if user_id:
                query = query.eq("user_id", user_id)
            
            response = query.execute()
            
            if response.data:
                return response.data[0]
            else:
                logger.warning(f"Chat {chat_id} not found")
                return None
                
        except Exception as e:
            logger.error(f"Error getting chat {chat_id}: {str(e)}")
            raise
    
    def get_user_chats(
        self, 
        user_id: str, 
        limit: int = 50, 
        offset: int = 0,
        status: str = None,
        org_id: str = None
    ) -> List[Dict]:
        """
        Get all chats for a user
        
        Args:
            user_id: ID of the user
            limit: Maximum number of chats to return
            offset: Number of chats to skip
            status: Optional status filter
            org_id: Optional organization filter
            
        Returns:
            List of chat records
        """
        try:
            query = (supabase.table(self.chats_table)
                    .select("*")
                    .eq("user_id", user_id)
                    .order("updated_at", desc=True)
                    .limit(limit)
                    .offset(offset))
            
            if status:
                query = query.eq("status", status)
            
            if org_id:
                query = query.eq("org_id", org_id)
            
            response = query.execute()
            
            logger.info(f"Retrieved {len(response.data)} chats for user {user_id}")
            return response.data
            
        except Exception as e:
            logger.error(f"Error getting chats for user {user_id}: {str(e)}")
            raise
    
    def update_chat(
        self, 
        chat_id: str, 
        user_id: str = None,
        title: str = None,
        status: str = None,
        metadata: Dict = None
    ) -> Dict:
        """
        Update a chat
        
        Args:
            chat_id: ID of the chat to update
            user_id: Optional user ID for access control
            title: New title for the chat
            status: New status for the chat
            metadata: New metadata for the chat
            
        Returns:
            The updated chat record
        """
        try:
            update_data = {
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
            
            if title is not None:
                update_data["title"] = title
            if status is not None:
                update_data["status"] = status
            if metadata is not None:
                update_data["metadata"] = metadata
            
            query = supabase.table(self.chats_table).update(update_data).eq("id", chat_id)
            
            if user_id:
                query = query.eq("user_id", user_id)
            
            response = query.execute()
            
            if response.data:
                logger.info(f"Updated chat {chat_id}")
                return response.data[0]
            else:
                logger.warning(f"Chat {chat_id} not found or not updated")
                return None
                
        except Exception as e:
            logger.error(f"Error updating chat {chat_id}: {str(e)}")
            raise
    
    def delete_chat(self, chat_id: str, user_id: str = None) -> bool:
        """
        Delete a chat and all its messages
        
        Args:
            chat_id: ID of the chat to delete
            user_id: Optional user ID for access control
            
        Returns:
            True if deleted successfully
        """
        try:
            query = supabase.table(self.chats_table).delete().eq("id", chat_id)
            
            if user_id:
                query = query.eq("user_id", user_id)
            
            response = query.execute()
            
            if response.data:
                logger.info(f"Deleted chat {chat_id}")
                return True
            else:
                logger.warning(f"Chat {chat_id} not found or not deleted")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting chat {chat_id}: {str(e)}")
            raise
    
    def get_chat_with_messages(self, chat_id: str, user_id: str = None) -> Dict:
        """
        Get a chat with all its messages
        
        Args:
            chat_id: ID of the chat
            user_id: Optional user ID for access control
            
        Returns:
            Chat data with messages array
        """
        try:
            # Get the chat
            chat = self.get_chat(chat_id, user_id)
            if not chat:
                return None
            
            # Get messages for this chat
            message_manager = MessageManager()
            messages = message_manager.get_chat_messages(chat_id)
            
            # Combine chat and messages
            chat["messages"] = messages
            
            return chat
            
        except Exception as e:
            logger.error(f"Error getting chat with messages {chat_id}: {str(e)}")
            raise

class MessageManager:
    """Manager class for handling message operations"""
    
    def __init__(self):
        self.messages_table = "messages"
    
    def create_message(
        self,
        chat_id: str,
        user_id: str,
        role: str,
        content: str,
        message_type: str = "text",
        metadata: Dict = None,
        thread_id: str = None,
        parent_message_id: str = None,
        token_count: int = 0
    ) -> Dict:
        """
        Create a new message in a chat
        
        Args:
            chat_id: ID of the chat
            user_id: ID of the user creating the message
            role: Role of the message sender (user, assistant, system)
            content: Content of the message
            message_type: Type of message (text, image, etc.)
            metadata: Additional metadata
            thread_id: Optional thread ID for conversation tracking
            parent_message_id: Optional parent message ID for replies
            token_count: Number of tokens in the message
            
        Returns:
            The created message record
        """
        try:
            message_data = {
                "chat_id": chat_id,
                "user_id": user_id,
                "role": role,
                "content": content,
                "message_type": message_type,
                "metadata": metadata or {},
                "thread_id": thread_id,
                "parent_message_id": parent_message_id,
                "token_count": token_count,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
            
            response = supabase.table(self.messages_table).insert(message_data).execute()
            
            if response.data:
                logger.info(f"Created message {response.data[0]['id']} in chat {chat_id}")
                return response.data[0]
            else:
                logger.error(f"Failed to create message in chat {chat_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error creating message: {str(e)}")
            raise
    
    def get_message(self, message_id: str) -> Dict:
        """
        Get a specific message by ID
        
        Args:
            message_id: ID of the message to retrieve
            
        Returns:
            The message record or None if not found
        """
        try:
            response = supabase.table(self.messages_table).select("*").eq("id", message_id).execute()
            
            if response.data:
                return response.data[0]
            else:
                logger.warning(f"Message {message_id} not found")
                return None
                
        except Exception as e:
            logger.error(f"Error getting message {message_id}: {str(e)}")
            raise
    
    def get_chat_messages(
        self,
        chat_id: str,
        limit: int = 100,
        offset: int = 0,
        role: str = None,
        since: datetime = None
    ) -> List[Dict]:
        """
        Get messages for a specific chat
        
        Args:
            chat_id: ID of the chat
            limit: Maximum number of messages to return
            offset: Number of messages to skip
            role: Optional role filter
            since: Optional datetime filter for messages after this time
            
        Returns:
            List of message records
        """
        try:
            query = (supabase.table(self.messages_table)
                    .select("*")
                    .eq("chat_id", chat_id)
                    .order("created_at", desc=False)
                    .limit(limit)
                    .offset(offset))
            
            if role:
                query = query.eq("role", role)
            
            if since:
                query = query.gte("created_at", since.isoformat())
            
            response = query.execute()
            
            logger.info(f"Retrieved {len(response.data)} messages for chat {chat_id}")
            return response.data
            
        except Exception as e:
            logger.error(f"Error getting messages for chat {chat_id}: {str(e)}")
            raise
    
    def update_message(
        self,
        message_id: str,
        content: str = None,
        metadata: Dict = None,
        token_count: int = None
    ) -> Dict:
        """
        Update a message
        
        Args:
            message_id: ID of the message to update
            content: New content for the message
            metadata: New metadata for the message
            token_count: New token count for the message
            
        Returns:
            The updated message record
        """
        try:
            update_data = {
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
            
            if content is not None:
                update_data["content"] = content
            if metadata is not None:
                update_data["metadata"] = metadata
            if token_count is not None:
                update_data["token_count"] = token_count
            
            response = supabase.table(self.messages_table).update(update_data).eq("id", message_id).execute()
            
            if response.data:
                logger.info(f"Updated message {message_id}")
                return response.data[0]
            else:
                logger.warning(f"Message {message_id} not found or not updated")
                return None
                
        except Exception as e:
            logger.error(f"Error updating message {message_id}: {str(e)}")
            raise
    
    def delete_message(self, message_id: str) -> bool:
        """
        Delete a message
        
        Args:
            message_id: ID of the message to delete
            
        Returns:
            True if deleted successfully
        """
        try:
            response = supabase.table(self.messages_table).delete().eq("id", message_id).execute()
            
            if response.data:
                logger.info(f"Deleted message {message_id}")
                return True
            else:
                logger.warning(f"Message {message_id} not found or not deleted")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting message {message_id}: {str(e)}")
            raise
    
    def get_user_messages(
        self,
        user_id: str,
        limit: int = 100,
        offset: int = 0,
        chat_id: str = None
    ) -> List[Dict]:
        """
        Get messages for a specific user
        
        Args:
            user_id: ID of the user
            limit: Maximum number of messages to return
            offset: Number of messages to skip
            chat_id: Optional chat ID filter
            
        Returns:
            List of message records
        """
        try:
            query = (supabase.table(self.messages_table)
                    .select("*")
                    .eq("user_id", user_id)
                    .order("created_at", desc=True)
                    .limit(limit)
                    .offset(offset))
            
            if chat_id:
                query = query.eq("chat_id", chat_id)
            
            response = query.execute()
            
            logger.info(f"Retrieved {len(response.data)} messages for user {user_id}")
            return response.data
            
        except Exception as e:
            logger.error(f"Error getting messages for user {user_id}: {str(e)}")
            raise

# Utility functions
def create_chat_with_first_message(
    user_id: str,
    content: str,
    title: str = None,
    org_id: str = None,
    chat_type: str = "conversation",
    thread_id: str = None
) -> Dict:
    """
    Create a new chat with an initial user message
    
    Args:
        user_id: ID of the user
        content: Content of the first message
        title: Optional title for the chat
        org_id: Optional organization ID
        chat_type: Type of chat
        thread_id: Optional thread ID
        
    Returns:
        Dictionary containing chat and message data
    """
    try:
        chat_manager = ChatManager()
        message_manager = MessageManager()
        
        # Create the chat
        chat = chat_manager.create_chat(
            user_id=user_id,
            title=title,
            org_id=org_id,
            chat_type=chat_type
        )
        
        if not chat:
            raise Exception("Failed to create chat")
        
        # Create the first message
        message = message_manager.create_message(
            chat_id=chat["id"],
            user_id=user_id,
            role="user",
            content=content,
            thread_id=thread_id
        )
        
        if not message:
            raise Exception("Failed to create first message")
        
        return {
            "chat": chat,
            "message": message
        }
        
    except Exception as e:
        logger.error(f"Error creating chat with first message: {str(e)}")
        raise

def get_chat_statistics(user_id: str = None, org_id: str = None) -> Dict:
    """
    Get statistics about chats and messages
    
    Args:
        user_id: Optional user ID filter
        org_id: Optional organization ID filter
        
    Returns:
        Dictionary with chat statistics
    """
    try:
        # Build query filters
        chat_filters = []
        message_filters = []
        
        if user_id:
            chat_filters.append(f"user_id.eq.{user_id}")
        if org_id:
            chat_filters.append(f"org_id.eq.{org_id}")
        
        # Get chat counts
        chat_query = supabase.table("chats").select("id", count="exact")
        for filter_condition in chat_filters:
            field, operator, value = filter_condition.split(".")
            chat_query = getattr(chat_query, operator)(field, value)
        
        chat_response = chat_query.execute()
        total_chats = chat_response.count
        
        # Get message counts
        message_query = supabase.table("messages").select("id", count="exact")
        if user_id:
            message_query = message_query.eq("user_id", user_id)
        
        message_response = message_query.execute()
        total_messages = message_response.count
        
        # Get active chats
        active_chat_query = supabase.table("chats").select("id", count="exact").eq("status", "active")
        for filter_condition in chat_filters:
            field, operator, value = filter_condition.split(".")
            active_chat_query = getattr(active_chat_query, operator)(field, value)
        
        active_chat_response = active_chat_query.execute()
        active_chats = active_chat_response.count
        
        return {
            "total_chats": total_chats,
            "active_chats": active_chats,
            "total_messages": total_messages,
            "average_messages_per_chat": total_messages / total_chats if total_chats > 0 else 0
        }
        
    except Exception as e:
        logger.error(f"Error getting chat statistics: {str(e)}")
        raise

# Pydantic Models for API requests and responses
class ChatRole(str, Enum):
    """Enumeration of message roles"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class ChatStatus(str, Enum):
    """Enumeration of chat statuses"""
    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"

class MessageType(str, Enum):
    """Enumeration of message types"""
    TEXT = "text"
    IMAGE = "image"
    FILE = "file"
    SYSTEM = "system"

class ChatType(str, Enum):
    """Enumeration of chat types"""
    CONVERSATION = "conversation"
    SUPPORT = "support"
    TRAINING = "training"
    PILOT = "pilot"

# Request Models
class CreateChatRequest(BaseModel):
    """Request model for creating a new chat"""
    user_id: str = Field(..., description="ID of the user creating the chat")
    title: Optional[str] = Field(None, description="Optional title for the chat")
    org_id: Optional[str] = Field(None, description="Optional organization ID")
    chat_type: ChatType = Field(ChatType.CONVERSATION, description="Type of chat")
    metadata: Optional[Dict] = Field(None, description="Additional metadata for the chat")

class UpdateChatRequest(BaseModel):
    """Request model for updating a chat"""
    chat_id: str = Field(..., description="ID of the chat to update")
    user_id: Optional[str] = Field(None, description="User ID for access control")
    title: Optional[str] = Field(None, description="New title for the chat")
    status: Optional[ChatStatus] = Field(None, description="New status for the chat")
    metadata: Optional[Dict] = Field(None, description="New metadata for the chat")

class CreateMessageRequest(BaseModel):
    """Request model for creating a new message"""
    chat_id: str = Field(..., description="ID of the chat")
    user_id: str = Field(..., description="ID of the user creating the message")
    role: ChatRole = Field(..., description="Role of the message sender")
    content: str = Field(..., description="Content of the message")
    message_type: MessageType = Field(MessageType.TEXT, description="Type of message")
    metadata: Optional[Dict] = Field(None, description="Additional metadata")
    thread_id: Optional[str] = Field(None, description="Optional thread ID for conversation tracking")
    parent_message_id: Optional[str] = Field(None, description="Optional parent message ID for replies")
    token_count: int = Field(0, description="Number of tokens in the message")

class UpdateMessageRequest(BaseModel):
    """Request model for updating a message"""
    message_id: str = Field(..., description="ID of the message to update")
    content: Optional[str] = Field(None, description="New content for the message")
    metadata: Optional[Dict] = Field(None, description="New metadata for the message")
    token_count: Optional[int] = Field(None, description="New token count for the message")

class CreateChatWithMessageRequest(BaseModel):
    """Request model for creating a chat with an initial message"""
    user_id: str = Field(..., description="ID of the user")
    content: str = Field(..., description="Content of the first message")
    title: Optional[str] = Field(None, description="Optional title for the chat")
    org_id: Optional[str] = Field(None, description="Optional organization ID")
    chat_type: ChatType = Field(ChatType.CONVERSATION, description="Type of chat")
    thread_id: Optional[str] = Field(None, description="Optional thread ID")

class GetChatsRequest(BaseModel):
    """Request model for getting user chats"""
    user_id: str = Field(..., description="ID of the user")
    limit: int = Field(50, description="Maximum number of chats to return", ge=1, le=100)
    offset: int = Field(0, description="Number of chats to skip", ge=0)
    status: Optional[ChatStatus] = Field(None, description="Optional status filter")
    org_id: Optional[str] = Field(None, description="Optional organization filter")

class GetMessagesRequest(BaseModel):
    """Request model for getting chat messages"""
    chat_id: str = Field(..., description="ID of the chat")
    limit: int = Field(100, description="Maximum number of messages to return", ge=1, le=200)
    offset: int = Field(0, description="Number of messages to skip", ge=0)
    role: Optional[ChatRole] = Field(None, description="Optional role filter")
    since: Optional[datetime] = Field(None, description="Optional datetime filter for messages after this time")

# Response Models
class ChatResponse(BaseModel):
    """Response model for chat data"""
    id: str = Field(..., description="Chat ID")
    user_id: str = Field(..., description="User ID")
    org_id: Optional[str] = Field(None, description="Organization ID")
    title: str = Field(..., description="Chat title")
    status: str = Field(..., description="Chat status")
    chat_type: str = Field(..., description="Chat type")
    metadata: Dict = Field(..., description="Chat metadata")
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")

class MessageResponse(BaseModel):
    """Response model for message data"""
    id: str = Field(..., description="Message ID")
    chat_id: str = Field(..., description="Chat ID")
    user_id: str = Field(..., description="User ID")
    role: str = Field(..., description="Message role")
    content: str = Field(..., description="Message content")
    message_type: str = Field(..., description="Message type")
    metadata: Dict = Field(..., description="Message metadata")
    thread_id: Optional[str] = Field(None, description="Thread ID")
    parent_message_id: Optional[str] = Field(None, description="Parent message ID")
    token_count: int = Field(..., description="Token count")
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")

class ChatWithMessagesResponse(BaseModel):
    """Response model for chat with messages"""
    id: str = Field(..., description="Chat ID")
    user_id: str = Field(..., description="User ID")
    org_id: Optional[str] = Field(None, description="Organization ID")
    title: str = Field(..., description="Chat title")
    status: str = Field(..., description="Chat status")
    chat_type: str = Field(..., description="Chat type")
    metadata: Dict = Field(..., description="Chat metadata")
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")
    messages: List[MessageResponse] = Field(..., description="List of messages in the chat")

class ChatListResponse(BaseModel):
    """Response model for list of chats"""
    chats: List[ChatResponse] = Field(..., description="List of chats")
    total: int = Field(..., description="Total number of chats")
    limit: int = Field(..., description="Limit used in the query")
    offset: int = Field(..., description="Offset used in the query")

class MessageListResponse(BaseModel):
    """Response model for list of messages"""
    messages: List[MessageResponse] = Field(..., description="List of messages")
    total: int = Field(..., description="Total number of messages")
    limit: int = Field(..., description="Limit used in the query")
    offset: int = Field(..., description="Offset used in the query")

class ChatStatsResponse(BaseModel):
    """Response model for chat statistics"""
    total_chats: int = Field(..., description="Total number of chats")
    active_chats: int = Field(..., description="Number of active chats")
    total_messages: int = Field(..., description="Total number of messages")
    average_messages_per_chat: float = Field(..., description="Average messages per chat")

class CreateChatWithMessageResponse(BaseModel):
    """Response model for creating a chat with first message"""
    chat: ChatResponse = Field(..., description="Created chat data")
    message: MessageResponse = Field(..., description="Created message data")

class StandardResponse(BaseModel):
    """Standard API response model"""
    success: bool = Field(..., description="Whether the operation was successful")
    message: str = Field(..., description="Response message")
    data: Optional[Any] = Field(None, description="Response data")
    error: Optional[str] = Field(None, description="Error message if any") 