"""
Google Calendar Integration Module

This module provides functionality to create and manage Google Calendar events,
specifically for creating appointments between a host account and user email accounts.
"""

import os
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google.auth.exceptions import RefreshError
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Google Calendar API scopes
SCOPES = [
    'https://www.googleapis.com/auth/calendar',
    'https://www.googleapis.com/auth/calendar.events'
]

# Environment variables
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8080/callback")

@dataclass
class CalendarEvent:
    """Data class for calendar event information"""
    title: str
    description: str
    start_time: datetime
    end_time: datetime
    attendee_emails: List[str]
    location: Optional[str] = None
    timezone: str = 'UTC'
    
    def to_google_event(self) -> Dict[str, Any]:
        """Convert to Google Calendar API event format"""
        return {
            'summary': self.title,
            'description': self.description,
            'location': self.location,
            'start': {
                'dateTime': self.start_time.isoformat(),
                'timeZone': self.timezone,
            },
            'end': {
                'dateTime': self.end_time.isoformat(),
                'timeZone': self.timezone,
            },
            'attendees': [{'email': email} for email in self.attendee_emails],
            'reminders': {
                'useDefault': False,
                'overrides': [
                    {'method': 'email', 'minutes': 24 * 60},  # 24 hours before
                    {'method': 'popup', 'minutes': 15},       # 15 minutes before
                ],
            },
            'conferenceData': {
                'createRequest': {
                    'requestId': f"meeting-{datetime.now().timestamp()}",
                    'conferenceSolutionKey': {'type': 'hangoutsMeet'}
                }
            }
        }

class GoogleCalendarManager:
    """Google Calendar API Manager"""
    
    def __init__(self, credentials_path: str = "credentials.json", token_path: str = "token.json"):
        """
        Initialize Google Calendar Manager
        
        Args:
            credentials_path: Path to Google OAuth2 credentials file
            token_path: Path to store access tokens
        """
        self.credentials_path = credentials_path
        self.token_path = token_path
        self.service = None
        self.credentials = None
        
    def authenticate(self, host_email: str = None) -> bool:
        """
        Authenticate with Google Calendar API
        
        Args:
            host_email: Email of the host account (optional)
            
        Returns:
            bool: True if authentication successful, False otherwise
        """
        try:
            # Try to load existing credentials
            if os.path.exists(self.token_path):
                self.credentials = Credentials.from_authorized_user_file(self.token_path, SCOPES)
            
            # If there are no valid credentials, initiate OAuth flow
            if not self.credentials or not self.credentials.valid:
                if self.credentials and self.credentials.expired and self.credentials.refresh_token:
                    try:
                        self.credentials.refresh(Request())
                        logger.info("Credentials refreshed successfully")
                    except RefreshError:
                        logger.warning("Failed to refresh credentials, need to re-authenticate")
                        self.credentials = None
                
                if not self.credentials:
                    if not os.path.exists(self.credentials_path):
                        logger.error(f"Credentials file not found: {self.credentials_path}")
                        return False
                    
                    flow = InstalledAppFlow.from_client_secrets_file(self.credentials_path, SCOPES)
                    self.credentials = flow.run_local_server(port=8080)
                    logger.info("New credentials obtained")
                
                # Save credentials for future use
                with open(self.token_path, 'w') as token_file:
                    token_file.write(self.credentials.to_json())
                    logger.info(f"Credentials saved to {self.token_path}")
            
            # Build the Calendar API service
            self.service = build('calendar', 'v3', credentials=self.credentials)
            logger.info("Google Calendar service initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Authentication failed: {str(e)}")
            return False
    
    def authenticate_with_service_account(self, service_account_path: str) -> bool:
        """
        Authenticate using service account credentials
        
        Args:
            service_account_path: Path to service account JSON file
            
        Returns:
            bool: True if authentication successful, False otherwise
        """
        try:
            from google.oauth2 import service_account
            
            credentials = service_account.Credentials.from_service_account_file(
                service_account_path, scopes=SCOPES
            )
            self.service = build('calendar', 'v3', credentials=credentials)
            logger.info("Service account authentication successful")
            return True
            
        except Exception as e:
            logger.error(f"Service account authentication failed: {str(e)}")
            return False
    
    def create_calendar(self, calendar_name: str, description: str = "", timezone: str = "UTC") -> Optional[str]:
        """
        Create a new calendar
        
        Args:
            calendar_name: Name of the calendar
            description: Calendar description
            timezone: Calendar timezone
            
        Returns:
            str: Calendar ID if successful, None otherwise
        """
        try:
            calendar_body = {
                'summary': calendar_name,
                'description': description,
                'timeZone': timezone
            }
            
            created_calendar = self.service.calendars().insert(body=calendar_body).execute()
            calendar_id = created_calendar.get('id')
            
            logger.info(f"Calendar created successfully: {calendar_name} (ID: {calendar_id})")
            return calendar_id
            
        except HttpError as e:
            logger.error(f"Failed to create calendar: {str(e)}")
            return None
    
    def create_appointment(self, 
                          host_email: str,
                          user_email: str,
                          title: str,
                          description: str,
                          start_time: datetime,
                          duration_minutes: int = 60,
                          timezone: str = "UTC",
                          location: str = None,
                          calendar_id: str = "primary") -> Optional[str]:
        """
        Create an appointment between host and user
        
        Args:
            host_email: Email of the host account
            user_email: Email of the user
            title: Event title
            description: Event description
            start_time: Start time of the appointment
            duration_minutes: Duration in minutes
            timezone: Timezone for the event
            location: Location of the meeting
            calendar_id: Calendar ID to create event in
            
        Returns:
            str: Event ID if successful, None otherwise
        """
        try:
            end_time = start_time + timedelta(minutes=duration_minutes)
            
            # Create CalendarEvent object
            event = CalendarEvent(
                title=title,
                description=description,
                start_time=start_time,
                end_time=end_time,
                attendee_emails=[host_email, user_email],
                location=location,
                timezone=timezone
            )
            
            # Convert to Google Calendar format
            event_body = event.to_google_event()
            
            # Create the event
            created_event = self.service.events().insert(
                calendarId=calendar_id,
                body=event_body,
                conferenceDataVersion=1,  # Enable Google Meet integration
                sendUpdates='all'  # Send invitations to all attendees
            ).execute()
            
            event_id = created_event.get('id')
            event_link = created_event.get('htmlLink')
            
            logger.info(f"Appointment created successfully: {title}")
            logger.info(f"Event ID: {event_id}")
            logger.info(f"Event Link: {event_link}")
            
            return event_id
            
        except HttpError as e:
            logger.error(f"Failed to create appointment: {str(e)}")
            return None
    
    def list_calendars(self) -> List[Dict[str, Any]]:
        """
        List all calendars accessible by the authenticated user
        
        Returns:
            List of calendar information
        """
        try:
            calendars_result = self.service.calendarList().list().execute()
            calendars = calendars_result.get('items', [])
            
            calendar_info = []
            for calendar in calendars:
                calendar_info.append({
                    'id': calendar.get('id'),
                    'name': calendar.get('summary'),
                    'description': calendar.get('description', ''),
                    'timezone': calendar.get('timeZone'),
                    'access_role': calendar.get('accessRole')
                })
            
            logger.info(f"Found {len(calendar_info)} calendars")
            return calendar_info
            
        except HttpError as e:
            logger.error(f"Failed to list calendars: {str(e)}")
            return []
    
    def get_upcoming_events(self, calendar_id: str = "primary", max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Get upcoming events from a calendar
        
        Args:
            calendar_id: Calendar ID to fetch events from
            max_results: Maximum number of events to return
            
        Returns:
            List of event information
        """
        try:
            now = datetime.utcnow().isoformat() + 'Z'  # 'Z' indicates UTC time
            
            events_result = self.service.events().list(
                calendarId=calendar_id,
                timeMin=now,
                maxResults=max_results,
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            
            events = events_result.get('items', [])
            
            event_info = []
            for event in events:
                start = event['start'].get('dateTime', event['start'].get('date'))
                event_info.append({
                    'id': event.get('id'),
                    'title': event.get('summary'),
                    'start': start,
                    'description': event.get('description', ''),
                    'location': event.get('location', ''),
                    'attendees': event.get('attendees', [])
                })
            
            logger.info(f"Found {len(event_info)} upcoming events")
            return event_info
            
        except HttpError as e:
            logger.error(f"Failed to get events: {str(e)}")
            return []
    
    def delete_event(self, event_id: str, calendar_id: str = "primary") -> bool:
        """
        Delete a calendar event
        
        Args:
            event_id: ID of the event to delete
            calendar_id: Calendar ID containing the event
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.service.events().delete(
                calendarId=calendar_id,
                eventId=event_id,
                sendUpdates='all'
            ).execute()
            
            logger.info(f"Event deleted successfully: {event_id}")
            return True
            
        except HttpError as e:
            logger.error(f"Failed to delete event: {str(e)}")
            return False
    
    def update_event(self, 
                    event_id: str,
                    title: str = None,
                    description: str = None,
                    start_time: datetime = None,
                    duration_minutes: int = None,
                    calendar_id: str = "primary") -> bool:
        """
        Update an existing calendar event
        
        Args:
            event_id: ID of the event to update
            title: New title (optional)
            description: New description (optional)
            start_time: New start time (optional)
            duration_minutes: New duration in minutes (optional)
            calendar_id: Calendar ID containing the event
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Get the existing event
            event = self.service.events().get(
                calendarId=calendar_id,
                eventId=event_id
            ).execute()
            
            # Update fields if provided
            if title:
                event['summary'] = title
            if description:
                event['description'] = description
            if start_time:
                event['start']['dateTime'] = start_time.isoformat()
                if duration_minutes:
                    end_time = start_time + timedelta(minutes=duration_minutes)
                    event['end']['dateTime'] = end_time.isoformat()
            
            # Update the event
            updated_event = self.service.events().update(
                calendarId=calendar_id,
                eventId=event_id,
                body=event,
                sendUpdates='all'
            ).execute()
            
            logger.info(f"Event updated successfully: {event_id}")
            return True
            
        except HttpError as e:
            logger.error(f"Failed to update event: {str(e)}")
            return False

# Helper functions for easy usage
def create_quick_appointment(host_email: str,
                           user_email: str,
                           title: str,
                           start_time: datetime,
                           duration_minutes: int = 60,
                           description: str = "",
                           location: str = None) -> Optional[str]:
    """
    Quick function to create an appointment
    
    Args:
        host_email: Email of the host account
        user_email: Email of the user
        title: Event title
        start_time: Start time of the appointment
        duration_minutes: Duration in minutes
        description: Event description
        location: Location of the meeting
        
    Returns:
        str: Event ID if successful, None otherwise
    """
    calendar_manager = GoogleCalendarManager()
    
    if not calendar_manager.authenticate():
        logger.error("Failed to authenticate with Google Calendar")
        return None
    
    return calendar_manager.create_appointment(
        host_email=host_email,
        user_email=user_email,
        title=title,
        description=description,
        start_time=start_time,
        duration_minutes=duration_minutes,
        location=location
    )

def setup_google_calendar_credentials():
    """
    Helper function to guide users through setting up Google Calendar credentials
    """
    instructions = """
    To use Google Calendar integration, you need to:
    
    1. Go to the Google Cloud Console (https://console.cloud.google.com/)
    2. Create a new project or select an existing one
    3. Enable the Google Calendar API
    4. Create credentials (OAuth 2.0 Client ID)
    5. Download the credentials file as 'credentials.json'
    6. Place the file in your project directory
    
    Environment Variables needed:
    - GOOGLE_CLIENT_ID: Your Google OAuth client ID
    - GOOGLE_CLIENT_SECRET: Your Google OAuth client secret
    - GOOGLE_REDIRECT_URI: Redirect URI (default: http://localhost:8080/callback)
    
    For service account authentication (recommended for server applications):
    1. Create a service account in Google Cloud Console
    2. Download the service account key file
    3. Use authenticate_with_service_account() method
    """
    
    print(instructions)
    return instructions

# Example usage
if __name__ == "__main__":
    # Example of how to use the Google Calendar integration
    
    # Initialize the calendar manager
    calendar_manager = GoogleCalendarManager()
    
    # Authenticate (this will open a browser for first-time auth)
    if calendar_manager.authenticate():
        print("Authentication successful!")
        
        # List available calendars
        calendars = calendar_manager.list_calendars()
        print(f"Available calendars: {len(calendars)}")
        for cal in calendars:
            print(f"  - {cal['name']} ({cal['id']})")
        
        # Create a sample appointment
        start_time = datetime.now() + timedelta(hours=1)
        event_id = calendar_manager.create_appointment(
            host_email="host@example.com",
            user_email="user@example.com",
            title="Sample Meeting",
            description="This is a test meeting created by the Google Calendar integration",
            start_time=start_time,
            duration_minutes=30,
            location="Virtual Meeting"
        )
        
        if event_id:
            print(f"Event created successfully: {event_id}")
        else:
            print("Failed to create event")
    
    else:
        print("Authentication failed!")
        setup_google_calendar_credentials() 