#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Google Drive API Routes

This module provides FastAPI endpoints for Google Drive integration including:
- Listing folder contents
- Searching folders
- Ingesting folder contents for knowledge management
"""

import asyncio
import json
import uuid
import os
import requests
from typing import Dict, List, Optional, Any
from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from google_drive_service import GoogleDriveService, ContentExtractor
from utilities import logger
from login import get_current_user
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Depends

# Create FastAPI router
router = APIRouter()

# Initialize security
security = HTTPBearer()

# Define request/response models
class GoogleDriveAuthRequest(BaseModel):
    access_token: str = Field(..., description="Google OAuth access token")
    refresh_token: Optional[str] = Field(None, description="Google OAuth refresh token")

class ListFolderRequest(BaseModel):
    access_token: str = Field(..., description="Google OAuth access token")
    refresh_token: Optional[str] = Field(None, description="Google OAuth refresh token")
    folder_id: str = Field(default="root", description="Google Drive folder ID ('root' for root folder)")

class SearchFoldersRequest(BaseModel):
    access_token: str = Field(..., description="Google OAuth access token")
    refresh_token: Optional[str] = Field(None, description="Google OAuth refresh token")
    query: str = Field(..., description="Search query for folders")

class IngestFolderRequest(BaseModel):
    access_token: str = Field(..., description="Google OAuth access token")
    refresh_token: Optional[str] = Field(None, description="Google OAuth refresh token")
    brainGraphId: str = Field(..., description="Brain graph UUID")
    folderId: str = Field(..., description="Google Drive folder ID")
    folderName: str = Field(..., description="Folder name")
    fileTypes: List[str] = Field(..., description="List of MIME types to process")
    recursive: bool = Field(default=False, description="Whether to process subfolders recursively")

class DriveFileResponse(BaseModel):
    id: str
    name: str
    mimeType: str
    size: Optional[str] = None
    modifiedTime: Optional[str] = None
    webViewLink: Optional[str] = None
    parents: Optional[List[str]] = None
    isFolder: bool = False
    thumbnailLink: Optional[str] = None

class DriveSubfolderResponse(BaseModel):
    id: str
    name: str
    webViewLink: Optional[str] = None
    files: List[DriveFileResponse] = []
    subfolders: List['DriveSubfolderResponse'] = []

class DriveFolderResponse(BaseModel):
    id: str
    name: str
    webViewLink: Optional[str] = None
    files: List[DriveFileResponse] = []
    subfolders: List[DriveSubfolderResponse] = []

class ListFolderResponse(BaseModel):
    folder: DriveFolderResponse

class SearchFoldersResponse(BaseModel):
    folders: List[DriveFileResponse]

class IngestFolderResponse(BaseModel):
    success: bool
    message: str
    jobId: str
    filesProcessed: int = 0

# Helper function to convert DriveFile to DriveFileResponse
def convert_drive_file_to_response(drive_file) -> DriveFileResponse:
    """Convert DriveFile dataclass to DriveFileResponse"""
    return DriveFileResponse(
        id=drive_file.id,
        name=drive_file.name,
        mimeType=drive_file.mimeType,
        size=drive_file.size,
        modifiedTime=drive_file.modifiedTime,
        webViewLink=drive_file.webViewLink,
        parents=drive_file.parents,
        isFolder=drive_file.isFolder,
        thumbnailLink=drive_file.thumbnailLink
    )

# Helper function to convert DriveFolder to DriveSubfolderResponse
def convert_drive_folder_to_subfolder_response(drive_folder) -> DriveSubfolderResponse:
    """Convert DriveFolder dataclass to DriveSubfolderResponse"""
    files = [convert_drive_file_to_response(file) for file in drive_folder.files]
    subfolders = [convert_drive_folder_to_subfolder_response(subfolder) for subfolder in drive_folder.subfolders]
    
    return DriveSubfolderResponse(
        id=drive_folder.id,
        name=drive_folder.name,
        webViewLink=drive_folder.webViewLink,
        files=files,
        subfolders=subfolders
    )

# Helper function to convert DriveFolder to DriveFolderResponse
def convert_drive_folder_to_response(drive_folder) -> DriveFolderResponse:
    """Convert DriveFolder dataclass to DriveFolderResponse"""
    files = [convert_drive_file_to_response(file) for file in drive_folder.files]
    subfolders = [convert_drive_folder_to_subfolder_response(subfolder) for subfolder in drive_folder.subfolders]
    
    return DriveFolderResponse(
        id=drive_folder.id,
        name=drive_folder.name,
        webViewLink=drive_folder.webViewLink,
        files=files,
        subfolders=subfolders
    )

# Helper function to create GoogleDriveService from user credentials
def create_drive_service_from_tokens(access_token: str, refresh_token: Optional[str] = None) -> GoogleDriveService:
    """
    Create GoogleDriveService instance from OAuth tokens
    
    Args:
        access_token: Google OAuth access token
        refresh_token: Google OAuth refresh token (optional)
        
    Returns:
        GoogleDriveService instance
    """
    # Get client credentials from environment
    client_id = os.getenv("GOOGLE_CLIENT_ID")
    client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
    
    if not client_id or not client_secret:
        raise HTTPException(
            status_code=500, 
            detail="Google OAuth client credentials not configured"
        )
    
    return GoogleDriveService(
        access_token=access_token,
        refresh_token=refresh_token or "",
        client_id=client_id,
        client_secret=client_secret
    )

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

@router.post('/google-drive/list-folder')
async def list_folder(request: ListFolderRequest):
    """
    List contents of a Google Drive folder using user's OAuth tokens
    
    Args:
        request: ListFolderRequest with access tokens and folder_id
        
    Returns:
        ListFolderResponse with folder contents
    """
    try:
        logger.info(f"Listing Google Drive folder: {request.folder_id}")
        
        # Initialize Google Drive service with user's tokens
        drive_service = create_drive_service_from_tokens(
            access_token=request.access_token,
            refresh_token=request.refresh_token
        )
        
        # List folder contents
        folder_contents = drive_service.list_folder_contents(request.folder_id)
        
        # Convert to response format
        response_data = convert_drive_folder_to_response(folder_contents)
        
        return ListFolderResponse(folder=response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing folder contents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.options('/google-drive/list-folder')
async def list_folder_options():
    return handle_options()

@router.post('/google-drive/search-folders')
async def search_folders(request: SearchFoldersRequest):
    """
    Search for folders in Google Drive using user's OAuth tokens
    
    Args:
        request: SearchFoldersRequest with access tokens and query
        
    Returns:
        SearchFoldersResponse with matching folders
    """
    try:
        logger.info(f"Searching Google Drive folders: {request.query}")
        
        # Initialize Google Drive service with user's tokens
        drive_service = create_drive_service_from_tokens(
            access_token=request.access_token,
            refresh_token=request.refresh_token
        )
        
        # Search folders
        folders = drive_service.search_folders(request.query)
        
        # Convert to response format
        response_folders = [convert_drive_file_to_response(folder) for folder in folders]
        
        return SearchFoldersResponse(folders=response_folders)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching folders: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.options('/google-drive/search-folders')
async def search_folders_options():
    return handle_options()

@router.post('/google-drive/ingest-folder')
async def ingest_folder(
    request: IngestFolderRequest, 
    background_tasks: BackgroundTasks
):
    """
    Ingest Google Drive folder contents for knowledge management using user's OAuth tokens
    
    Args:
        request: IngestFolderRequest with folder details and access tokens
        background_tasks: FastAPI background tasks
        
    Returns:
        IngestFolderResponse with job details
    """
    try:
        logger.info(f"Starting Google Drive folder ingestion: {request.folderName}")
        
        # Initialize Google Drive service with user's tokens
        drive_service = create_drive_service_from_tokens(
            access_token=request.access_token,
            refresh_token=request.refresh_token
        )
        content_extractor = ContentExtractor(drive_service)
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Start background ingestion job
        background_tasks.add_task(
            process_folder_ingestion,
            job_id=job_id,
            drive_service=drive_service,
            content_extractor=content_extractor,
            request=request,
            user_id="anonymous"  # Since we removed user authentication
        )
        
        return IngestFolderResponse(
            success=True,
            message="Folder ingestion started",
            jobId=job_id,
            filesProcessed=0
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting folder ingestion: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.options('/google-drive/ingest-folder')
async def ingest_folder_options():
    return handle_options()

# Background task functions
async def process_folder_ingestion(
    job_id: str,
    drive_service: GoogleDriveService,
    content_extractor: ContentExtractor,
    request: IngestFolderRequest,
    user_id: str
):
    """
    Background task to process folder ingestion
    
    Args:
        job_id: Unique job identifier
        drive_service: GoogleDriveService instance
        content_extractor: ContentExtractor instance
        request: IngestFolderRequest with processing details
        user_id: ID of the user who initiated the ingestion
    """
    try:
        logger.info(f"Starting folder ingestion job {job_id} for user {user_id}")
        
        # Get all files in the folder
        files = drive_service.get_files_in_folder(
            folder_id=request.folderId,
            file_types=request.fileTypes,
            recursive=request.recursive
        )
        
        files_processed = 0
        
        for file in files:
            try:
                # Extract content from file
                content = content_extractor.extract_content(file.id, file.mimeType)
                
                if content:
                    # TODO: Process content with knowledge management system
                    # This would typically involve:
                    # 1. Chunking the content
                    # 2. Creating embeddings
                    # 3. Storing in vector database
                    # 4. Updating brain graph
                    
                    logger.info(f"Job {job_id}: Processed file: {file.name} ({len(content)} characters)")
                    files_processed += 1
                else:
                    logger.warning(f"Job {job_id}: Could not extract content from file: {file.name}")
                    
            except Exception as e:
                logger.error(f"Job {job_id}: Error processing file {file.name}: {e}")
                continue
        
        logger.info(f"Completed folder ingestion job {job_id} for user {user_id}. Processed {files_processed} files.")
        
        # TODO: Update job status in database
        # This would typically involve storing job status and results
        
    except Exception as e:
        logger.error(f"Error in folder ingestion job {job_id} for user {user_id}: {e}")
        # TODO: Update job status as failed in database

# Additional helper endpoints
@router.post('/google-drive/test-connection')
async def test_connection(request: GoogleDriveAuthRequest):
    """
    Test Google Drive connection with user's OAuth tokens
    
    Args:
        request: GoogleDriveAuthRequest with access tokens
        
    Returns:
        Connection status and user info
    """
    try:
        logger.info("Testing Google Drive connection")
        
        # Initialize Google Drive service with user's tokens
        drive_service = create_drive_service_from_tokens(
            access_token=request.access_token,
            refresh_token=request.refresh_token
        )
        
        # Test connection by getting user info
        user_info = drive_service.get_user_info()
        
        return {
            "connected": True,
            "user_email": user_info.get('emailAddress', 'Unknown'),
            "user_name": user_info.get('displayName', 'Unknown'),
            "storage_quota": user_info.get('storageQuota', {}),
            "message": "Google Drive connection successful"
        }
        
    except Exception as e:
        logger.error(f"Error testing Google Drive connection: {e}")
        return {
            "connected": False,
            "error": str(e),
            "message": "Google Drive connection failed"
        }

@router.options('/google-drive/test-connection')
async def test_connection_options():
    return handle_options()

@router.get('/google-drive/auth-config')
async def get_auth_config():
    """
    Get Google OAuth configuration for Drive access
    
    Returns:
        OAuth configuration for frontend
    """
    try:
        client_id = os.getenv("GOOGLE_CLIENT_ID")
        if not client_id:
            raise HTTPException(
                status_code=500,
                detail="Google OAuth client ID not configured"
            )
        
        # Define the required scopes for Google Drive
        scopes = [
            "https://www.googleapis.com/auth/drive.readonly",
            "https://www.googleapis.com/auth/userinfo.email",
            "https://www.googleapis.com/auth/userinfo.profile"
        ]
        
        return {
            "client_id": client_id,
            "scopes": scopes,
            "redirect_uri": os.getenv('GOOGLE_OAUTH_REDIRECT_URI', 'http://localhost:5173/google-oauth-callback.html'),
            "response_type": "code",
            "access_type": "offline",
            "prompt": "consent"
        }
        
    except Exception as e:
        logger.error(f"Error getting auth config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.options('/google-drive/auth-config')
async def get_auth_config_options():
    return handle_options()

@router.post('/google-drive/exchange-code')
async def exchange_code_for_tokens(request: dict):
    """
    Exchange authorization code for Google Drive tokens
    
    Request body:
    {
        "code": "authorization_code_from_google",
        "state": "optional_state_parameter"
    }
    
    Response:
    {
        "access_token": "...",
        "refresh_token": "...",
        "expires_in": 3600,
        "token_type": "Bearer"
    }
    """
    try:
        code = request.get("code")
        if not code:
            raise HTTPException(status_code=400, detail="Authorization code is required")
        
        # Get client credentials from environment
        client_id = os.getenv("GOOGLE_CLIENT_ID")
        client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
        redirect_uri = os.getenv('GOOGLE_OAUTH_REDIRECT_URI', 'http://localhost:5173/google-oauth-callback.html')
        
        if not client_id or not client_secret:
            raise HTTPException(
                status_code=500,
                detail="Google OAuth client credentials not configured"
            )
        
        # Exchange code for tokens
        token_url = "https://oauth2.googleapis.com/token"
        token_data = {
            "client_id": client_id,
            "client_secret": client_secret,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": redirect_uri
        }
        
        response = requests.post(token_url, data=token_data)
        
        if response.status_code == 200:
            tokens = response.json()
            logger.info("Successfully exchanged authorization code for tokens")
            return {
                "access_token": tokens.get("access_token"),
                "refresh_token": tokens.get("refresh_token"),
                "expires_in": tokens.get("expires_in", 3600),
                "token_type": tokens.get("token_type", "Bearer"),
                "scope": tokens.get("scope")
            }
        else:
            error_data = response.json()
            logger.error(f"Token exchange failed: {error_data}")
            raise HTTPException(
                status_code=400,
                detail=f"Token exchange failed: {error_data.get('error_description', 'Unknown error')}"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exchanging authorization code: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.options('/google-drive/exchange-code')
async def exchange_code_options():
    return handle_options()

# Update the DriveSubfolderResponse model to handle recursive references
DriveSubfolderResponse.model_rebuild() 