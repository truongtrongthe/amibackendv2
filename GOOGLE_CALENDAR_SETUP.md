# Google Calendar Integration Setup Guide

This guide will help you set up and use the Google Calendar integration in your application.

## 🚀 Quick Start

The `googlecalendar.py` module provides functionality to:
- Create calendar appointments between host and user accounts
- Manage Google Calendar events 
- Create new calendars
- List and manage existing calendars
- Automatically send meeting invitations with Google Meet links

## 📋 Prerequisites

1. **Google Cloud Project**: You need a Google Cloud project with Calendar API enabled
2. **OAuth 2.0 Credentials**: For user authentication
3. **Python Dependencies**: Install required packages

## 🔧 Setup Instructions

### Step 1: Google Cloud Console Setup

1. **Create/Select Project**
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project or select existing one

2. **Enable Google Calendar API**
   ```bash
   # Navigate to APIs & Services > Library
   # Search for "Google Calendar API" and enable it
   ```

3. **Create OAuth 2.0 Credentials**
   - Go to APIs & Services > Credentials
   - Click "Create Credentials" > "OAuth 2.0 Client ID"
   - Choose "Desktop Application" 
   - Download the credentials file as `credentials.json`

4. **Configure OAuth Consent Screen**
   - Go to APIs & Services > OAuth consent screen
   - Fill in required information
   - Add your email to test users (for development)

### Step 2: Environment Variables

Add these environment variables to your `.env` file:

```env
# Google Calendar Configuration
GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret
GOOGLE_REDIRECT_URI=http://localhost:8080/callback
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

The following packages will be installed:
- `google-api-python-client`
- `google-auth-httplib2` 
- `google-auth-oauthlib`

### Step 4: Place Credentials File

- Download the `credentials.json` file from Google Cloud Console
- Place it in your project root directory
- The file should be named exactly `credentials.json`

## 📖 Usage Examples

### Basic Usage - Quick Appointment

```python
from datetime import datetime, timedelta
from googlecalendar import create_quick_appointment

# Create a quick appointment
event_id = create_quick_appointment(
    host_email="host@company.com",
    user_email="client@external.com", 
    title="Business Meeting",
    start_time=datetime.now() + timedelta(hours=2),
    duration_minutes=60,
    description="Discussion about project requirements",
    location="Virtual Meeting"
)

if event_id:
    print(f"Appointment created! Event ID: {event_id}")
```

### Advanced Usage - Full Calendar Manager

```python
from googlecalendar import GoogleCalendarManager
from datetime import datetime, timedelta

# Initialize calendar manager
calendar_manager = GoogleCalendarManager()

# Authenticate (will open browser for first-time setup)
if calendar_manager.authenticate():
    
    # Create a new calendar
    calendar_id = calendar_manager.create_calendar(
        calendar_name="Client Meetings",
        description="Calendar for client appointments"
    )
    
    # Create appointment
    event_id = calendar_manager.create_appointment(
        host_email="sales@company.com",
        user_email="prospect@client.com",
        title="Sales Demo",
        description="Product demonstration for potential client",
        start_time=datetime.now() + timedelta(days=1, hours=14),
        duration_minutes=45,
        location="Conference Room A"
    )
    
    # List upcoming events
    events = calendar_manager.get_upcoming_events()
    for event in events:
        print(f"Upcoming: {event['title']} at {event['start']}")
```

### Service Account Authentication (Recommended for Production)

```python
from googlecalendar import GoogleCalendarManager

calendar_manager = GoogleCalendarManager()

# Use service account for server-to-server authentication
if calendar_manager.authenticate_with_service_account("service-account-key.json"):
    # Create appointments without user interaction
    event_id = calendar_manager.create_appointment(
        host_email="automated@company.com",
        user_email="customer@client.com",
        title="Automated Booking",
        description="System-generated appointment",
        start_time=datetime.now() + timedelta(hours=1),
        duration_minutes=30
    )
```

## 🧪 Testing

Run the example script to test your setup:

```bash
# Show setup instructions
python calendar_example.py setup

# Run full demo
python calendar_example.py demo

# Create sample appointments
python calendar_example.py samples
```

## 🔐 Authentication Methods

### 1. OAuth 2.0 (User Authentication)
- **Best for**: Applications where users manage their own calendars
- **Setup**: Download `credentials.json` from Google Cloud Console
- **Flow**: Opens browser for user consent on first run

### 2. Service Account (Server Authentication)
- **Best for**: Server applications, automated booking systems
- **Setup**: Create service account in Google Cloud Console
- **Flow**: No user interaction required

## 📁 File Structure

After setup, you'll have these files:

```
your_project/
├── googlecalendar.py          # Main integration module
├── calendar_example.py        # Usage examples
├── credentials.json           # OAuth credentials (user auth)
├── service-account-key.json   # Service account key (server auth)
├── token.json                # Stored access tokens
└── requirements.txt          # Updated with new dependencies
```

## ⚡ Key Features

### 1. Calendar Creation
```python
calendar_id = calendar_manager.create_calendar(
    calendar_name="Team Meetings", 
    description="Weekly team sync meetings",
    timezone="America/New_York"
)
```

### 2. Appointment Management
- **Create**: Book appointments with automatic invitations
- **Update**: Modify existing appointments  
- **Delete**: Cancel appointments with notifications
- **List**: View upcoming events

### 3. Automatic Features
- **Google Meet Integration**: Automatically adds video conferencing
- **Email Reminders**: 24-hour and 15-minute reminders
- **Invitation Sending**: Automatic invites to all attendees
- **Timezone Handling**: Proper timezone conversion

### 4. Error Handling
- **Token Refresh**: Automatic refresh of expired tokens
- **API Errors**: Graceful handling of API failures
- **Logging**: Comprehensive logging for debugging

## 🚨 Security Considerations

1. **Credentials Protection**
   - Never commit `credentials.json` or `token.json` to version control
   - Add them to `.gitignore`
   - Use environment variables for production

2. **Scope Limitation**
   - Only request necessary Calendar scopes
   - Regularly review and rotate credentials

3. **Service Account Security**
   - Limit service account permissions
   - Use domain-wide delegation carefully
   - Store service account keys securely

## 🔍 Troubleshooting

### Common Issues

1. **"Credentials file not found"**
   - Ensure `credentials.json` is in project root
   - Check file permissions

2. **"Authentication failed"**
   - Verify OAuth consent screen configuration
   - Check if your email is added to test users

3. **"Calendar API not enabled"**
   - Enable Google Calendar API in Google Cloud Console

4. **"Permission denied"**
   - Ensure proper OAuth scopes are requested
   - Check service account permissions

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Your calendar code here
```

## 🌟 Production Deployment

For production deployments:

1. **Use Service Accounts**: Avoid user OAuth for automated systems
2. **Environment Variables**: Store all credentials as environment variables
3. **Error Monitoring**: Implement proper error tracking
4. **Rate Limiting**: Respect Google API rate limits
5. **Backup Strategy**: Have fallback mechanisms for API failures

## 📞 Support

If you need help with setup:

1. Check the [Google Calendar API documentation](https://developers.google.com/calendar)
2. Review the example code in `calendar_example.py`
3. Enable debug logging to see detailed error messages
4. Verify your Google Cloud project configuration

## 🎯 Use Cases

This integration is perfect for:

- **Appointment Booking Systems**: Customer-business appointments
- **CRM Integration**: Automatic meeting scheduling
- **Event Management**: Conference and meeting organization
- **Automated Scheduling**: System-generated appointments
- **Multi-user Calendars**: Team and resource scheduling

---

**Happy Scheduling! 📅✨** 