"""
Example usage of Google Calendar Integration

This file demonstrates how to use the googlecalendar.py module to create
appointments between a host account and user accounts.
"""

from datetime import datetime, timedelta
from googlecalendar import GoogleCalendarManager, create_quick_appointment, setup_google_calendar_credentials

def demo_calendar_functionality():
    """Demo function showing how to use Google Calendar integration"""
    
    print("=== Google Calendar Integration Demo ===\n")
    
    # Example 1: Quick appointment creation
    print("1. Creating a quick appointment...")
    
    # Define appointment details
    host_email = "host@example.com"  # Replace with actual host email
    user_email = "user@example.com"  # Replace with actual user email
    title = "Business Meeting"
    description = "Discussing project requirements and timeline"
    start_time = datetime.now() + timedelta(hours=2)  # 2 hours from now
    duration = 60  # 60 minutes
    location = "Virtual Meeting Room"
    
    # Create appointment using the quick function
    event_id = create_quick_appointment(
        host_email=host_email,
        user_email=user_email,
        title=title,
        start_time=start_time,
        duration_minutes=duration,
        description=description,
        location=location
    )
    
    if event_id:
        print(f"✅ Quick appointment created successfully!")
        print(f"   Event ID: {event_id}")
        print(f"   Title: {title}")
        print(f"   Start: {start_time}")
        print(f"   Duration: {duration} minutes")
        print(f"   Attendees: {host_email}, {user_email}")
    else:
        print("❌ Failed to create quick appointment")
    
    print("\n" + "="*50 + "\n")
    
    # Example 2: Using GoogleCalendarManager directly
    print("2. Using GoogleCalendarManager for advanced operations...")
    
    # Initialize calendar manager
    calendar_manager = GoogleCalendarManager()
    
    # Authenticate
    if calendar_manager.authenticate():
        print("✅ Authentication successful!")
        
        # List available calendars
        print("\n📅 Available calendars:")
        calendars = calendar_manager.list_calendars()
        for i, cal in enumerate(calendars[:5], 1):  # Show first 5 calendars
            print(f"   {i}. {cal['name']} ({cal['access_role']})")
        
        # Create a new calendar
        print("\n🆕 Creating new calendar...")
        new_calendar_id = calendar_manager.create_calendar(
            calendar_name="Meeting Room Bookings",
            description="Calendar for booking meeting rooms",
            timezone="America/New_York"
        )
        
        if new_calendar_id:
            print(f"✅ New calendar created: {new_calendar_id}")
        
        # Create appointment in specific calendar
        print("\n📝 Creating appointment...")
        appointment_start = datetime.now() + timedelta(days=1, hours=14)  # Tomorrow at 2 PM
        
        event_id = calendar_manager.create_appointment(
            host_email=host_email,
            user_email=user_email,
            title="Product Demo",
            description="Demonstrating new features to potential client",
            start_time=appointment_start,
            duration_minutes=45,
            timezone="America/New_York",
            location="Conference Room A",
            calendar_id="primary"  # Use primary calendar
        )
        
        if event_id:
            print(f"✅ Appointment created successfully!")
            print(f"   Event ID: {event_id}")
            print(f"   Scheduled for: {appointment_start}")
        
        # Get upcoming events
        print("\n📋 Upcoming events:")
        upcoming = calendar_manager.get_upcoming_events(max_results=5)
        for i, event in enumerate(upcoming, 1):
            print(f"   {i}. {event['title']} - {event['start']}")
        
    else:
        print("❌ Authentication failed!")
        print("Please make sure you have set up Google Calendar credentials.")
        print("Run setup_google_calendar_credentials() for instructions.")

def setup_credentials_guide():
    """Guide users through setting up credentials"""
    print("=== Google Calendar Setup Guide ===\n")
    setup_google_calendar_credentials()

def create_sample_appointments():
    """Create several sample appointments for testing"""
    print("=== Creating Sample Appointments ===\n")
    
    calendar_manager = GoogleCalendarManager()
    
    if not calendar_manager.authenticate():
        print("❌ Authentication failed. Please set up credentials first.")
        return
    
    # Sample appointments data
    appointments = [
        {
            "host_email": "manager@company.com",
            "user_email": "client1@external.com",
            "title": "Sales Call - Q1 Review",
            "description": "Quarterly business review with key client",
            "start_time": datetime.now() + timedelta(days=1, hours=10),
            "duration_minutes": 60,
            "location": "Virtual Meeting"
        },
        {
            "host_email": "hr@company.com", 
            "user_email": "candidate@email.com",
            "title": "Job Interview - Software Engineer",
            "description": "Technical interview for senior software engineer position",
            "start_time": datetime.now() + timedelta(days=2, hours=14),
            "duration_minutes": 90,
            "location": "Office Conference Room B"
        },
        {
            "host_email": "support@company.com",
            "user_email": "customer@client.com", 
            "title": "Support Session - System Training",
            "description": "Training session for new system features",
            "start_time": datetime.now() + timedelta(days=3, hours=16),
            "duration_minutes": 45,
            "location": "Online Training Room"
        }
    ]
    
    created_events = []
    
    for i, appointment in enumerate(appointments, 1):
        print(f"Creating appointment {i}/3: {appointment['title']}")
        
        event_id = calendar_manager.create_appointment(**appointment)
        
        if event_id:
            created_events.append(event_id)
            print(f"✅ Created successfully (ID: {event_id})")
        else:
            print(f"❌ Failed to create appointment")
        
        print()
    
    print(f"Summary: {len(created_events)}/{len(appointments)} appointments created successfully")
    return created_events

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "setup":
            setup_credentials_guide()
        elif command == "demo":
            demo_calendar_functionality()
        elif command == "samples":
            create_sample_appointments()
        else:
            print("Available commands:")
            print("  python calendar_example.py setup    - Show setup instructions")
            print("  python calendar_example.py demo     - Run full demo")
            print("  python calendar_example.py samples  - Create sample appointments")
    else:
        # Run demo by default
        demo_calendar_functionality() 