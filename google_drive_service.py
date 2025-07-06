"""
Google Drive Integration Service

This module provides functionality to interact with Google Drive API for folder browsing,
file searching, and content extraction for knowledge management integration.
"""

import os
import io
import json
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from dataclasses import dataclass

from google.oauth2.credentials import Credentials
from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

# Optional imports for content extraction
try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Google Drive API scope
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

@dataclass
class DriveFile:
    """Represents a Google Drive file or folder"""
    id: str
    name: str
    mimeType: str
    size: Optional[str] = None
    modifiedTime: Optional[str] = None
    webViewLink: Optional[str] = None
    parents: Optional[List[str]] = None
    isFolder: bool = False
    thumbnailLink: Optional[str] = None

@dataclass
class DriveFolder:
    """Represents a Google Drive folder with its contents"""
    id: str
    name: str
    webViewLink: Optional[str] = None
    files: List[DriveFile] = None
    subfolders: List['DriveFolder'] = None

    def __post_init__(self):
        if self.files is None:
            self.files = []
        if self.subfolders is None:
            self.subfolders = []

class GoogleDriveService:
    """Service class for interacting with Google Drive API"""
    
    def __init__(self, access_token: str, refresh_token: str, client_id: str, client_secret: str):
        """
        Initialize Google Drive service with OAuth credentials
        
        Args:
            access_token: OAuth access token
            refresh_token: OAuth refresh token
            client_id: Google OAuth client ID
            client_secret: Google OAuth client secret
        """
        self.credentials = Credentials(
            token=access_token,
            refresh_token=refresh_token,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=client_id,
            client_secret=client_secret,
            scopes=SCOPES
        )
        self.service = None
        self._initialize_service()
    
    def _initialize_service(self):
        """Initialize the Google Drive service"""
        try:
            # Refresh token if expired
            if self.credentials.expired and self.credentials.refresh_token:
                self.credentials.refresh(Request())
            
            self.service = build('drive', 'v3', credentials=self.credentials)
            logger.info("Google Drive service initialized successfully")
        except RefreshError as e:
            logger.error(f"Failed to refresh credentials: {e}")
            raise Exception("Invalid or expired credentials")
        except Exception as e:
            logger.error(f"Failed to initialize Google Drive service: {e}")
            raise
    
    def list_folder_contents(self, folder_id: str = 'root', recursive: bool = False) -> DriveFolder:
        """
        List contents of a Google Drive folder
        
        Args:
            folder_id: ID of the folder to list ('root' for root folder)
            recursive: Whether to recursively list subfolders
            
        Returns:
            DriveFolder object with files and subfolders
        """
        try:
            # Get folder metadata
            folder_metadata = {}
            if folder_id != 'root':
                folder_metadata = self.service.files().get(
                    fileId=folder_id,
                    fields="id,name,webViewLink"
                ).execute()
            
            # List files in the folder
            query = f"'{folder_id}' in parents and trashed=false"
            results = self.service.files().list(
                q=query,
                fields="files(id,name,mimeType,size,modifiedTime,webViewLink,parents,thumbnailLink)",
                orderBy="name"
            ).execute()
            
            files = results.get('files', [])
            
            # Separate files and folders
            regular_files = []
            subfolders = []
            
            for file in files:
                drive_file = DriveFile(
                    id=file['id'],
                    name=file['name'],
                    mimeType=file['mimeType'],
                    size=file.get('size'),
                    modifiedTime=file.get('modifiedTime'),
                    webViewLink=file.get('webViewLink'),
                    parents=file.get('parents', []),
                    thumbnailLink=file.get('thumbnailLink'),
                    isFolder=file['mimeType'] == 'application/vnd.google-apps.folder'
                )
                
                if drive_file.isFolder:
                    # Recursively get subfolder contents if requested
                    if recursive:
                        subfolder_contents = self.list_folder_contents(drive_file.id, recursive=True)
                        subfolders.append(subfolder_contents)
                    else:
                        subfolders.append(DriveFolder(
                            id=drive_file.id,
                            name=drive_file.name,
                            webViewLink=drive_file.webViewLink
                        ))
                else:
                    regular_files.append(drive_file)
            
            # Create and return DriveFolder
            folder_name = folder_metadata.get('name', 'Root') if folder_id != 'root' else 'Root'
            folder_webview = folder_metadata.get('webViewLink') if folder_id != 'root' else None
            
            return DriveFolder(
                id=folder_id,
                name=folder_name,
                webViewLink=folder_webview,
                files=regular_files,
                subfolders=subfolders
            )
            
        except HttpError as error:
            logger.error(f'Google Drive API error: {error}')
            raise Exception(f"Failed to list folder contents: {error}")
        except Exception as e:
            logger.error(f'Error listing folder contents: {e}')
            raise
    
    def search_folders(self, query: str) -> List[DriveFile]:
        """
        Search for folders in Google Drive
        
        Args:
            query: Search query string
            
        Returns:
            List of DriveFile objects representing folders
        """
        try:
            # Search for folders only
            search_query = f"mimeType='application/vnd.google-apps.folder' and name contains '{query}' and trashed=false"
            results = self.service.files().list(
                q=search_query,
                fields="files(id,name,mimeType,modifiedTime,webViewLink,parents)",
                orderBy="name"
            ).execute()
            
            folders = []
            for file in results.get('files', []):
                drive_file = DriveFile(
                    id=file['id'],
                    name=file['name'],
                    mimeType=file['mimeType'],
                    modifiedTime=file.get('modifiedTime'),
                    webViewLink=file.get('webViewLink'),
                    parents=file.get('parents', []),
                    isFolder=True
                )
                folders.append(drive_file)
            
            return folders
            
        except HttpError as error:
            logger.error(f'Google Drive API error: {error}')
            raise Exception(f"Failed to search folders: {error}")
        except Exception as e:
            logger.error(f'Error searching folders: {e}')
            raise
    
    def get_files_in_folder(self, folder_id: str, file_types: List[str] = None, recursive: bool = False) -> List[DriveFile]:
        """
        Get all files in a folder (optionally filtered by file types)
        
        Args:
            folder_id: ID of the folder
            file_types: List of MIME types to filter by
            recursive: Whether to search subfolders recursively
            
        Returns:
            List of DriveFile objects
        """
        try:
            all_files = []
            
            # Build query
            query = f"'{folder_id}' in parents and trashed=false"
            
            if file_types:
                mime_conditions = " or ".join([f"mimeType='{mime}'" for mime in file_types])
                query += f" and ({mime_conditions})"
            
            # Get files
            results = self.service.files().list(
                q=query,
                fields="files(id,name,mimeType,size,modifiedTime,webViewLink,parents,thumbnailLink)",
                orderBy="name"
            ).execute()
            
            files = results.get('files', [])
            
            for file in files:
                drive_file = DriveFile(
                    id=file['id'],
                    name=file['name'],
                    mimeType=file['mimeType'],
                    size=file.get('size'),
                    modifiedTime=file.get('modifiedTime'),
                    webViewLink=file.get('webViewLink'),
                    parents=file.get('parents', []),
                    thumbnailLink=file.get('thumbnailLink'),
                    isFolder=file['mimeType'] == 'application/vnd.google-apps.folder'
                )
                
                if drive_file.isFolder and recursive:
                    # Recursively get files from subfolders
                    subfolder_files = self.get_files_in_folder(drive_file.id, file_types, recursive)
                    all_files.extend(subfolder_files)
                elif not drive_file.isFolder:
                    all_files.append(drive_file)
            
            return all_files
            
        except HttpError as error:
            logger.error(f'Google Drive API error: {error}')
            raise Exception(f"Failed to get files in folder: {error}")
        except Exception as e:
            logger.error(f'Error getting files in folder: {e}')
            raise
    
    def get_user_info(self) -> Dict[str, Any]:
        """
        Get Google Drive user information
        
        Returns:
            Dictionary containing user information
        """
        try:
            user_info = self.service.about().get(fields="user,storageQuota").execute()
            return user_info
        except HttpError as error:
            logger.error(f'Google Drive API error getting user info: {error}')
            raise Exception(f"Failed to get user info: {error}")
        except Exception as e:
            logger.error(f'Error getting user info: {e}')
            raise

class ContentExtractor:
    """Service class for extracting content from Google Drive files"""
    
    def __init__(self, drive_service: GoogleDriveService):
        """
        Initialize content extractor with Google Drive service
        
        Args:
            drive_service: GoogleDriveService instance
        """
        self.drive_service = drive_service
    
    def extract_content(self, file_id: str, mime_type: str) -> Optional[str]:
        """
        Extract text content from a Google Drive file
        
        Args:
            file_id: ID of the file to extract content from
            mime_type: MIME type of the file
            
        Returns:
            Extracted text content or None if extraction failed
        """
        try:
            if mime_type == 'application/vnd.google-apps.document':
                return self._extract_google_doc(file_id)
            elif mime_type == 'application/vnd.google-apps.spreadsheet':
                return self._extract_google_sheet(file_id)
            elif mime_type == 'application/vnd.google-apps.presentation':
                return self._extract_google_slides(file_id)
            elif mime_type == 'application/pdf':
                return self._extract_pdf(file_id)
            elif mime_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                return self._extract_docx(file_id)
            elif mime_type == 'text/plain':
                return self._extract_text_file(file_id)
            else:
                logger.warning(f"Unsupported file type: {mime_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error extracting content from file {file_id}: {e}")
            return None
    
    def _extract_google_doc(self, file_id: str) -> Optional[str]:
        """Extract content from Google Docs"""
        try:
            request = self.drive_service.service.files().export_media(
                fileId=file_id,
                mimeType='text/plain'
            )
            content = request.execute()
            return content.decode('utf-8')
        except Exception as e:
            logger.error(f"Error extracting Google Doc content: {e}")
            return None
    
    def _extract_google_sheet(self, file_id: str) -> Optional[str]:
        """Extract content from Google Sheets"""
        try:
            request = self.drive_service.service.files().export_media(
                fileId=file_id,
                mimeType='text/csv'
            )
            content = request.execute()
            return content.decode('utf-8')
        except Exception as e:
            logger.error(f"Error extracting Google Sheet content: {e}")
            return None
    
    def _extract_google_slides(self, file_id: str) -> Optional[str]:
        """Extract content from Google Slides"""
        try:
            request = self.drive_service.service.files().export_media(
                fileId=file_id,
                mimeType='text/plain'
            )
            content = request.execute()
            return content.decode('utf-8')
        except Exception as e:
            logger.error(f"Error extracting Google Slides content: {e}")
            return None
    
    def _extract_pdf(self, file_id: str) -> Optional[str]:
        """Extract content from PDF files"""
        if not PDF_AVAILABLE:
            logger.warning("PyPDF2 not available, cannot extract PDF content")
            return None
        
        try:
            request = self.drive_service.service.files().get_media(fileId=file_id)
            file_io = io.BytesIO()
            downloader = MediaIoBaseDownload(file_io, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
            
            file_io.seek(0)
            pdf_reader = PyPDF2.PdfReader(file_io)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        except Exception as e:
            logger.error(f"Error extracting PDF content: {e}")
            return None
    
    def _extract_docx(self, file_id: str) -> Optional[str]:
        """Extract content from DOCX files"""
        if not DOCX_AVAILABLE:
            logger.warning("python-docx not available, cannot extract DOCX content")
            return None
        
        try:
            request = self.drive_service.service.files().get_media(fileId=file_id)
            file_io = io.BytesIO()
            downloader = MediaIoBaseDownload(file_io, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
            
            file_io.seek(0)
            doc = Document(file_io)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            logger.error(f"Error extracting DOCX content: {e}")
            return None
    
    def _extract_text_file(self, file_id: str) -> Optional[str]:
        """Extract content from plain text files"""
        try:
            request = self.drive_service.service.files().get_media(fileId=file_id)
            content = request.execute()
            return content.decode('utf-8')
        except Exception as e:
            logger.error(f"Error extracting text file content: {e}")
            return None

# Note: create_drive_service_from_integration function removed as we now use direct user authentication 