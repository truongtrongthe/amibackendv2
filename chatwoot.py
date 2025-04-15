from flask import Flask, request, jsonify
import json
from typing import Dict, Any
from datetime import datetime, timezone
from collections import deque
import requests
import os
from dotenv import load_dotenv
from contact import ContactManager
from contactconvo import ConversationManager

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Initialize managers
contact_manager = ContactManager()
conversation_manager = ConversationManager()

# Chatwoot API configuration
CHATWOOT_API_TOKEN = os.getenv('CHATWOOT_API_TOKEN')
CHATWOOT_BASE_URL = os.getenv('CHATWOOT_BASE_URL', 'https://app.chatwoot.com')
CHATWOOT_ACCOUNT_ID = os.getenv('CHATWOOT_ACCOUNT_ID')

# Keep track of recent webhook requests to detect duplicates
# Store the last 100 request IDs
recent_requests = deque(maxlen=100)

def send_message(conversation_id: int, message: str, attachment_url: str = None):
    """
    Send a message to a conversation in Chatwoot
    """
    print(f"\n=== Sending Message ===")
    print(f"Conversation ID: {conversation_id}")
    print(f"Message: {message}")
    print(f"Attachment URL: {attachment_url}")
    
    headers = {
        'api_access_token': CHATWOOT_API_TOKEN,
        'Content-Type': 'application/json'
    }
    
    data = {
        'content': message,
        'message_type': 'outgoing',
        'private': False,
        'content_type': 'text',
        'content_attributes': {}
    }
    
    if attachment_url:
        print("Downloading attachment...")
        try:
            # Download the image
            response = requests.get(attachment_url)
            if response.status_code != 200:
                print(f"Failed to download image: {response.status_code}")
                return False
                
            # Save the image temporarily
            temp_file = 'temp_image.jpg'
            with open(temp_file, 'wb') as f:
                f.write(response.content)
            
            print("Preparing multipart form data...")
            # Prepare multipart form data
            files = {
                'attachments[]': ('image.jpg', open(temp_file, 'rb'), 'image/jpeg')
            }
            data['content_type'] = 'image'
            
            # Send the request with multipart form data
            url = f"{CHATWOOT_BASE_URL}/api/v1/accounts/{CHATWOOT_ACCOUNT_ID}/conversations/{conversation_id}/messages"
            print(f"Sending request to: {url}")
            
            response = requests.post(
                url,
                headers={'api_access_token': CHATWOOT_API_TOKEN},
                data={'content': message},
                files=files
            )
            
            # Clean up the temporary file
            os.remove(temp_file)
            
        except Exception as e:
            print(f"Error handling attachment: {str(e)}")
            return False
            
    else:
        # Send regular text message
        url = f"{CHATWOOT_BASE_URL}/api/v1/accounts/{CHATWOOT_ACCOUNT_ID}/conversations/{conversation_id}/messages"
        print(f"Sending request to: {url}")
        print(f"Request data: {json.dumps(data, indent=2)}")
        
        response = requests.post(
            url,
            headers=headers,
            json=data
        )
    
    print(f"Response Status: {response.status_code}")
    print(f"Response Text: {response.text}")
    
    if response.status_code == 200:
        response_data = response.json()
        print(f"Message sent successfully. Message ID: {response_data.get('id')}")
        print("=== End Message Sending ===\n")
        return True
    else:
        print(f"Failed to send message. Status: {response.status_code}")
        print(f"Response: {response.text}")
        print("=== End Message Sending ===\n")
        return False

def log_raw_data(data: Dict[str, Any], event_type: str):
    """
    Log raw data for debugging
    """
    print(f"\n=== Raw Data for {event_type} ===")
    print(json.dumps(data, indent=2, default=str))
    print("===============================\n")


def handle_conversation_event(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Base handler for conversation-related events.
    Extracts common conversation context and Facebook ID.
    
    Args:
        data: The webhook event data
        
    Returns:
        Dictionary containing conversation context
    """
    conversation = data.get('conversation', {})
    contact_inbox = conversation.get('contact_inbox', {})
    facebook_id = contact_inbox.get('source_id')
    
    # Handle last_activity_at timestamp
    last_activity_at = conversation.get('last_activity_at')
    if last_activity_at:
        if isinstance(last_activity_at, (int, float)):
            # Convert Unix timestamp to UTC datetime
            last_activity_at = datetime.fromtimestamp(last_activity_at, timezone.utc)
        else:
            # Parse ISO format string to UTC
            last_activity_at = datetime.fromisoformat(last_activity_at.replace('Z', '+00:00'))
    
    # Get contact name from multiple possible locations
    contact_name = None
    # Try from conversation meta first
    meta = conversation.get('meta', {})
    if meta and meta.get('sender'):
        contact_name = meta['sender'].get('name')
    
    # If not found, try from contact data
    if not contact_name:
        contact = data.get('contact', {})
        contact_name = contact.get('name')
    
    # If still not found, try from sender data
    if not contact_name:
        sender = data.get('sender', {})
        contact_name = sender.get('name')
    
    # Default to Unknown if still not found
    contact_name = contact_name or 'Unknown'
    
    return {
        'conversation_id': conversation.get('id'),
        'facebook_id': facebook_id,
        'status': conversation.get('status'),
        'channel': conversation.get('channel'),
        'unread_count': conversation.get('unread_count', 0),
        'last_activity_at': last_activity_at,
        'contact_name': contact_name
    }

def handle_contact_creation(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle contact creation or update based on webhook data.
    
    Args:
        data: The webhook event data
        
    Returns:
        Dictionary containing contact information
    """
    contact = data.get('contact', {})
    contact_inbox = data.get('conversation', {}).get('contact_inbox', {})
    facebook_id = contact_inbox.get('source_id')
    
    if not facebook_id:
        return None
    
    try:
        # First try to get contact by Facebook ID
        existing_contact = contact_manager.get_contact_by_facebook_id(facebook_id)
        if existing_contact:
            print(f"Found existing contact with Facebook ID {facebook_id}")
            return existing_contact
            
        # If not found, create new contact
        print("\n👤 New Contact Created:")
        
        # Get name from multiple possible sources
        name = contact.get('name')
        if not name:
            # Try to get from conversation meta
            meta = data.get('conversation', {}).get('meta', {})
            if meta and meta.get('sender'):
                name = meta['sender'].get('name')
        
        # Split name into parts
        name_parts = name.split() if name else []
        first_name = name_parts[0] if name_parts else 'Facebook'
        last_name = ' '.join(name_parts[1:]) if len(name_parts) > 1 else 'User'
        
        # Create new contact
        new_contact = contact_manager.create_contact(
            type='customer',
            first_name=first_name,
            last_name=last_name,
            email=contact.get('email'),
            phone=contact.get('phone_number'),
            facebook_id=facebook_id,
            profile_picture_url=contact.get('avatar_url')
        )
        
        if new_contact:
            print(f"Name: {first_name} {last_name}")
            print(f"Facebook ID: {facebook_id}")
            print(f"Contact ID: {new_contact['id']}")
            return new_contact
        else:
            print("Failed to create new contact")
            return None
            
    except Exception as e:
        print(f"Error in contact management: {str(e)}")
        # Try one more time to get the contact in case it was created by another process
        try:
            existing_contact = contact_manager.get_contact_by_facebook_id(facebook_id)
            if existing_contact:
                print(f"Found existing contact after error: {facebook_id}")
                return existing_contact
        except:
            pass
        return None

def handle_conversation_management(data: Dict[str, Any], contact_id: int) -> Dict[str, Any]:
    """
    Handle conversation creation and message management.
    
    Args:
        data: The webhook event data
        contact_id: The ID of the contact
        
    Returns:
        Dictionary containing conversation information
    """
    conversation = data.get('conversation', {})
    conversation_id = conversation.get('id')
    platform_conversation_id = f"chatwoot_{conversation_id}"
    
    # Get or create conversation
    conversation_data = conversation_manager.get_or_create_conversation_by_platform_id(
        contact_id=contact_id,
        platform='facebook',
        platform_conversation_id=platform_conversation_id
    )
    
    # Map Chatwoot status to our valid statuses
    status_mapping = {
        'open': 'active',
        'resolved': 'completed',
        'pending': 'active',
        'snoozed': 'active'
    }
    
    # Update conversation status if needed
    chatwoot_status = conversation.get('status')
    if chatwoot_status:
        mapped_status = status_mapping.get(chatwoot_status, 'active')
        conversation_manager.update_conversation(
            conversation_id=conversation_data['id'],
            status=mapped_status
        )
    
    return conversation_data

def handle_message_created(data: Dict[str, Any]):
    """
    Handle message created event within conversation context
    """
    try:
        # Get conversation context
        context = handle_conversation_event(data)
        contact_data = handle_contact_creation(data)
        if not contact_data:
            return
        
        conversation_data = handle_conversation_management(data, contact_data['id'])
        
        # Get message details
        content = data.get('content', '')
        message_type = data.get('message_type')
        is_incoming = message_type == 'incoming'
        status = 'received' if is_incoming else 'sent'
        
        # Log initial message
        print(f"\n📨 New Message:")
        print(f"{'From' if is_incoming else 'To'}: {context['contact_name']}")
        print(f"Content: {content}")
        print(f"Type: {'Incoming' if is_incoming else 'Outgoing'}")
        print(f"Status: {status}")
        print(f"Conversation ID: {context['conversation_id']}")
        if context['last_activity_at']:
            print(f"Time: {context['last_activity_at'].strftime('%Y-%m-%d %H:%M:%S UTC')}")
        
        # Add message to conversation
        message = {
            'sender_type': 'agent' if message_type == 'outgoing' else 'contact',
            'content': content,
            'platform': 'ami',
            'platform_message_id': data.get('id'),
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'direction': 'incoming' if is_incoming else 'outgoing',
            'status': status
        }
        
        if is_incoming:
            message.update({
                'source': 'facebook',
                'raw_data': {
                    'facebook_id': context['facebook_id'],
                    'contact_name': context['contact_name']
                }
            })
        
        conversation_manager.add_message(conversation_data['id'], message)
    
    except Exception as e:
        print(f"Error in handle_message_created: {str(e)}")
        raise

def handle_message_updated(data: Dict[str, Any]):
    """
    Handle message updated event within conversation context
    """
    try:
        # Get conversation context
        context = handle_conversation_event(data)
        
        # Get existing contact by Facebook ID instead of creating a new one
        facebook_id = context.get('facebook_id')
        if not facebook_id:
            return
            
        contact_data = contact_manager.get_contact_by_facebook_id(facebook_id)
        if not contact_data:
            return
        
        conversation_data = handle_conversation_management(data, contact_data['id'])
        
        # Get message details
        content = data.get('content', '')
        message_type = data.get('message_type')
        is_incoming = message_type == 'incoming'
        status = data.get('status', 'received' if is_incoming else 'sent')
        
        # Only log status changes
        if status != 'sent':
            print(f"\n📨 Message Update:")
            print(f"{'From' if is_incoming else 'To'}: {context['contact_name']}")
            print(f"Content: {content}")
            print(f"Type: {'Incoming' if is_incoming else 'Outgoing'}")
            print(f"New Status: {status}")
            print(f"Conversation ID: {context['conversation_id']}")
            if context['last_activity_at']:
                print(f"Time: {context['last_activity_at'].strftime('%Y-%m-%d %H:%M:%S UTC')}")
            
            # Show changes if any
            changed_attributes = data.get('changed_attributes', [])
            if changed_attributes:
                print("Changes:")
                for change in changed_attributes:
                    for attr, values in change.items():
                        print(f"  - {attr}: {values.get('previous_value')} -> {values.get('current_value')}")
    except Exception as e:
        print(f"Error in handle_message_updated: {str(e)}")
        raise

def handle_conversation_created(data: Dict[str, Any]):
    """
    Handle conversation created event
    """
    log_raw_data(data, "Conversation Created")
    
    print("\n=== Conversation Created ===")
    conversation = data.get('conversation', {})
    print(f"Conversation ID: {conversation.get('id')}")
    print(f"Status: {conversation.get('status')}")
    
    contact = data.get('contact', {})
    print(f"Contact: {contact.get('name')} (ID: {contact.get('id')})")
    print(f"Email: {contact.get('email')}")
    print(f"Phone: {contact.get('phone_number')}")
    
    # Get Facebook ID from additional_attributes
    additional_attrs = contact.get('additional_attributes', {})
    facebook_id = additional_attrs.get('id')
    if facebook_id:
        print(f"Facebook ID: {facebook_id}")
        # Send a message with contact's profile picture
        profile_pic = contact.get('avatar_url')
        if profile_pic:
            send_message(
                conversation.get('id'),
                f"New conversation started with Facebook ID: {facebook_id}",
                profile_pic
            )
    print("=====================\n")

@app.route('/webhook/chatwoot', methods=['POST'])
def chatwoot_webhook():
    """
    Handle incoming webhooks from Chatwoot
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "No JSON data received"}), 400
            
        event = data.get('event')
        request_id = f"{event}_{data.get('id', '')}_{datetime.now().timestamp()}"
        
        # Check for duplicates
        if request_id in recent_requests:
            return jsonify({"status": "success", "message": "Duplicate request ignored"})
        
        recent_requests.append(request_id)
        
        # Handle events
        if event == "message_created":
            handle_message_created(data)
        elif event == "message_updated":
            handle_message_updated(data)
        elif event == "conversation_created":
            handle_conversation_created(data)
        
        return jsonify({"status": "success"})
    
    except Exception as e:
        print(f"Error processing webhook: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

# Suppress Flask access logs
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True) 