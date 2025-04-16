import requests
import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import requests
from supabase import create_client, Client
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("facebook_messenger")

# Load environment variables
load_dotenv()

# Initialize Supabase client
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

VERIFY_TOKEN = os.getenv("CALLBACK_V_TOKEN")
PAGE_ACCESS_TOKEN = os.getenv("FBAPP_TOKEN")

def get_sender_text(body):
    message_data = {}
    try:
        messaging = body["entry"][0]["messaging"][0]
        sender_id = messaging["sender"]["id"]
        message_data["senderID"] = sender_id
    except Exception:
        message_data["senderID"] = "UNKNOWN"

    try:
        message_data["GET_STARTED"] = messaging.get("postback", {}).get("title", "NO-DATA")
    except Exception:
        message_data["GET_STARTED"] = "NO-DATA"

    try:
        message_data["messageText"] = messaging.get("message", {}).get("text", "NO-DATA")
    except Exception:
        message_data["messageText"] = "NO-DATA"

    try:
        message_data["URL"] = messaging.get("message", {}).get("attachments", [])[0]["payload"]["url"]
    except Exception:
        message_data["URL"] = "NO-DATA"

    return message_data


def send_message(sender_id, message_data, access_token=None):
    """
    Send a message to Facebook Messenger.
    Only sends messages to test user ID 29495554333369135.
    For all other users, just logs without sending.
    
    Args:
        sender_id: Facebook user ID to send message to
        message_data: Message data to send in Facebook format
        access_token: Facebook page access token (optional - falls back to env var)
        
    Returns:
        1 for success, 0 for failure
    """
    # Use provided access token or fall back to environment variable
    token = access_token or PAGE_ACCESS_TOKEN
    if not token:
        logger.error("No Facebook access token provided and FBAPP_TOKEN environment variable not set")
        return 0
        
    url = f"https://graph.facebook.com/v22.0/me/messages?access_token={token}"
    headers = {"Content-Type": "application/json"}

    payload = {
        "recipient": {"id": sender_id},
        "messaging_type": "RESPONSE",
        "message": message_data
    }

    try:
        # Check if this is our test user ID
        if sender_id == "29495554333369135":
            logger.info(f"Sending message to Facebook user {sender_id}: {json.dumps(message_data, indent=2)}")
            # Actually send the message to Facebook
            response = requests.post(url, json=payload, headers=headers)
            
            if response.status_code == 200:
                logger.info(f"Message sent successfully to {sender_id}")
                return 1
            else:
                logger.error(f"Error sending Facebook message: {response.text}")
                return 0
        else:
            # For all other users, log that we're skipping them in test mode
            logger.info(f"[TEST MODE] Skipping message send to non-test user {sender_id}")
            return 1  # Still return success so conversation history is maintained
    except Exception as e:
        logger.error(f"Exception during Facebook message sending: {str(e)}")
        return 0


def format_message(fulfillment):
    if fulfillment.get("message") == "text":
        return {
            "text": fulfillment["text"]["text"][0]
        }
    elif fulfillment.get("message") == "quickReplies":
        quick_replies = fulfillment["quickReplies"]["quickReplies"]
        title = fulfillment["quickReplies"]["title"]

        formatted = {
            "text": title,
            "quick_replies": [
                {
                    "content_type": "text",
                    "title": qr,
                    "payload": "<POSTBACK_PAYLOAD>"
                } for qr in quick_replies
            ]
        }
        return formatted
    else:
        return {"text": "Unsupported message type"}

def parse_fb_message(webhook_event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse a Facebook webhook event into a standardized message format
    for storage in our conversations database.
    
    Args:
        webhook_event: The Facebook webhook event
        
    Returns:
        A structured message object for storage
    """
    messaging = webhook_event.get("entry", [{}])[0].get("messaging", [{}])[0]
    
    # Check if this is an echo message (sent by the page)
    is_echo = messaging.get("message", {}).get("is_echo", False)
    
    # Parse basic message information
    sender_id = messaging.get("sender", {}).get("id")
    recipient_id = messaging.get("recipient", {}).get("id")
    timestamp = messaging.get("timestamp")
    
    # For echo messages, the sender is actually the page and recipient is the user
    if is_echo:
        # Swap sender and recipient for echo messages
        temp_id = sender_id
        sender_id = recipient_id  # The actual user is the recipient in echoes
        recipient_id = temp_id    # The page is the sender in echoes
    
    # Convert timestamp to ISO format if it's a Unix timestamp
    if timestamp and isinstance(timestamp, int):
        timestamp = datetime.fromtimestamp(timestamp/1000.0).isoformat()
    
    # Initialize message object
    message_obj = {
        "id": f"fb_{timestamp}_{recipient_id}" if timestamp else f"fb_{datetime.utcnow().isoformat()}_{recipient_id}",
        "platform_msg_id": messaging.get("message", {}).get("mid"),
        "sender_id": "system" if is_echo else sender_id,  # For echoes, sender is system
        "sender_type": "system" if is_echo else "contact",  # System for page messages, contact for user messages
        "recipient_id": sender_id if is_echo else recipient_id,  # For echoes, recipient is the user
        "timestamp": timestamp or datetime.utcnow().isoformat(),
        "platform": "facebook",
        "status": "sent" if is_echo else "received"  # Echoes are outgoing messages
    }
    
    # Check if it's a text message
    if "message" in messaging and "text" in messaging["message"]:
        message_obj["content"] = messaging["message"]["text"]
        message_obj["content_type"] = "text"
    
    # Check for attachments (images, videos, etc)
    elif "message" in messaging and "attachments" in messaging["message"]:
        attachments = messaging["message"]["attachments"]
        message_obj["attachments"] = []
        
        # Get the first attachment type as the primary content type
        first_attachment = attachments[0]
        message_obj["content_type"] = first_attachment.get("type", "attachment")
        
        # If it's an image, use the URL as content for preview purposes
        if first_attachment.get("type") == "image":
            message_obj["content"] = f"[Image] {first_attachment.get('payload', {}).get('url', '')}"
        elif first_attachment.get("type") == "video":
            message_obj["content"] = f"[Video] {first_attachment.get('payload', {}).get('url', '')}"
        elif first_attachment.get("type") == "audio":
            message_obj["content"] = f"[Audio] {first_attachment.get('payload', {}).get('url', '')}"
        elif first_attachment.get("type") == "file":
            message_obj["content"] = f"[File] {first_attachment.get('payload', {}).get('url', '')}"
        else:
            message_obj["content"] = f"[{first_attachment.get('type', 'Attachment')}]"
        
        # Process all attachments
        for attachment in attachments:
            attachment_obj = {
                "type": attachment.get("type", "unknown"),
                "url": attachment.get("payload", {}).get("url", "")
            }
            message_obj["attachments"].append(attachment_obj)
    
    # Check for postbacks (button clicks)
    elif "postback" in messaging:
        message_obj["content"] = messaging["postback"].get("title", "")
        message_obj["content_type"] = "postback"
        message_obj["metadata"] = {
            "facebook": {
                "payload": messaging["postback"].get("payload", "")
            }
        }
    
    # Unknown message type
    else:
        message_obj["content"] = "Unsupported message type"
        message_obj["content_type"] = "unknown"
    
    # Add full Facebook data for reference
    message_obj["metadata"] = {
        "facebook": messaging,
        "is_echo": is_echo
    }
    
    return message_obj

def save_fb_message_to_conversation(contact_id: int, platform_conversation_id: str, 
                                   fb_message: Dict[str, Any], convo_manager) -> Dict[str, Any]:
    """
    Save a Facebook message to the conversation system
    
    Args:
        contact_id: ID of the contact in our system
        platform_conversation_id: Facebook's conversation ID 
        fb_message: Parsed Facebook message object
        convo_manager: Instance of ConversationManager
        
    Returns:
        The updated conversation
    """
    try:
        return convo_manager.add_fb_message(
            contact_id=contact_id,
            platform_conversation_id=platform_conversation_id,
            fb_message=fb_message,
            create_if_missing=True
        )
    except Exception as e:
        print(f"❌ Error saving Facebook message to conversation: {str(e)}")
        return None

def get_facebook_integration_for_org(org_id: str) -> Optional[Dict[str, Any]]:
    """
    Get the active Facebook integration for an organization
    
    Args:
        org_id: Organization ID to look up
        
    Returns:
        Dictionary with integration details or None if not found
    """
    try:
        response = supabase.table("organization_integrations")\
            .select("*")\
            .eq("org_id", org_id)\
            .eq("integration_type", "facebook")\
            .eq("is_active", True)\
            .execute()
            
        if not response.data:
            logger.warning(f"No active Facebook integration found for organization {org_id}")
            return None
            
        integration = response.data[0]
        
        # Parse config if it's stored as a string
        if "config" in integration and isinstance(integration["config"], str):
            try:
                integration["config"] = json.loads(integration["config"])
            except json.JSONDecodeError:
                integration["config"] = {}
                
        return integration
    except Exception as e:
        logger.error(f"Error getting Facebook integration for org {org_id}: {str(e)}")
        return None

def verify_webhook_token(token: str, org_id: str = None) -> bool:
    """
    Verify if the provided token matches the webhook verification token for the organization.
    
    Args:
        token: Token sent by Facebook in the verification request
        org_id: Organization ID to find the matching integration
        
    Returns:
        True if token is valid, False otherwise
    """
    # If org_id is provided, check the organization's webhook_verify_token
    if org_id:
        integration = get_facebook_integration_for_org(org_id)
        if integration:
            if integration.get("webhook_verify_token") and integration.get("webhook_verify_token") == token:
                logger.info(f"Verified webhook using organization ({org_id}) webhook_verify_token")
                return True
    
    # Default fallback to environment variable
    if token == VERIFY_TOKEN:
        logger.info("Verified webhook using default VERIFY_TOKEN environment variable")
        return True
        
    logger.warning(f"Webhook verification failed for token: {token}")
    return False

def get_page_access_token(org_id: str = None) -> Optional[str]:
    """
    Get the Facebook page access token for an organization.
    
    Args:
        org_id: Organization ID to find the matching integration
        
    Returns:
        Page access token or None if not found
    """
    # If org_id is provided, get the organization's access token
    if org_id:
        integration = get_facebook_integration_for_org(org_id)
        if integration and integration.get("access_token"):
            token = integration.get("access_token")
            logger.info(f"Using organization-specific Facebook access token for org {org_id}")
            return token
    
    # Default fallback to environment variable
    if PAGE_ACCESS_TOKEN:
        logger.info("Using default PAGE_ACCESS_TOKEN environment variable")
        return PAGE_ACCESS_TOKEN
        
    logger.error("No Facebook access token found for the organization or in environment variables")
    return None

def send_and_save_message(recipient_id: str, message_data: Dict[str, Any], 
                        convo_manager, contact_id: int = None, org_id: str = None) -> Tuple[int, Optional[Dict[str, Any]]]:
    """
    Send a message to Facebook and save it to our conversation system.
    Automatically creates contact if needed.
    
    Args:
        recipient_id: Facebook user ID to send message to
        message_data: Message data to send (Facebook format)
        convo_manager: Instance of ConversationManager
        contact_id: Optional ID of the contact (if already known)
        org_id: Optional organization ID to find the appropriate Facebook integration
        
    Returns:
        Tuple of (success_status, updated_conversation)
    """
    # If contact_id is not provided, try to find or create the contact
    if not contact_id:
        contact = get_or_create_contact_by_facebook_id(recipient_id, org_id)
        if not contact:
            logger.error(f"Failed to get or create contact for Facebook user {recipient_id}")
            return 0, None
        contact_id = contact["id"]
    
    # Get the appropriate access token if org_id is provided
    access_token = get_page_access_token(org_id)
    
    # First send the message to Facebook
    success = send_message(recipient_id, message_data, access_token)
    
    if success:
        # Create a message object for our database
        timestamp = datetime.utcnow().isoformat()
        message_obj = {
            "id": f"fb_sent_{timestamp}_{recipient_id}",
            "sender_id": "system",  # Or could be the actual user ID if available
            "sender_type": "system",
            "recipient_id": recipient_id,
            "content": message_data.get("text", str(message_data)),
            "content_type": "text" if "text" in message_data else "complex",
            "timestamp": timestamp,
            "platform": "facebook",
            "status": "sent",
            "metadata": {
                "facebook": message_data,
                "is_echo": True
            }
        }
        
        # Add attachments if present
        if "attachment" in message_data:
            attachment_data = message_data["attachment"]
            attachment_type = attachment_data.get("type")
            
            # Add attachment array
            message_obj["attachments"] = [{
                "type": attachment_type,
                "url": attachment_data.get("payload", {}).get("url", "")
            }]
            
            # Set content type based on attachment
            message_obj["content_type"] = attachment_type
            
            # Set content preview
            if attachment_type == "image":
                message_obj["content"] = "[Image sent]"
            elif attachment_type == "video":
                message_obj["content"] = "[Video sent]"
            elif attachment_type == "audio":
                message_obj["content"] = "[Audio sent]"
            elif attachment_type == "file":
                message_obj["content"] = "[File sent]"
            elif attachment_type == "template":
                message_obj["content"] = "[Template sent]"
        
        # Save to conversation
        logger.info(f"Saving outgoing message to conversation for contact {contact_id}")
        conversation = save_fb_message_to_conversation(
            contact_id=contact_id,
            platform_conversation_id=recipient_id,
            fb_message=message_obj,
            convo_manager=convo_manager
        )
        
        return success, conversation
    
    return success, None

def get_or_create_contact_by_facebook_id(sender_id: str, organization_id: str = None) -> Optional[Dict[str, Any]]:
    """
    Get a contact by Facebook ID or create a new one if not found.
    
    Args:
        sender_id: Facebook user ID
        organization_id: Organization ID to associate the contact with
        
    Returns:
        Contact dict or None if creation failed
    """
    # Try to find existing contact
    query = supabase.table("contacts").select("*").eq("facebook_id", sender_id)
    
    # Add organization filter if provided
    if organization_id:
        query = query.eq("organization_id", organization_id)
        
    contact_response = query.execute()
    contact = contact_response.data[0] if contact_response.data else None
    
    if contact:
        logger.info(f"Found existing contact for Facebook ID {sender_id}: {contact['id']}")
        return contact
    
    # No contact found, create a new one
    try:
        # First try a minimal request with just name and picture
        minimal_fields = [
            "id",
            "first_name",
            "last_name", 
            "picture.type(large)"
        ]
        
        logger.info(f"Making minimal Facebook data request for user {sender_id}")
        user_url = f"https://graph.facebook.com/v18.0/{sender_id}?fields={','.join(minimal_fields)}&access_token={PAGE_ACCESS_TOKEN}"
        user_response = requests.get(user_url)
        
        # Log response status
        logger.info(f"Facebook API minimal request status: {user_response.status_code}")
        
        # If the minimal request fails, create a basic contact record
        if user_response.status_code != 200:
            logger.warning(f"Facebook API minimal request failed: {user_response.text}")
            # Create a basic contact with default values
            contact_data = {
                "organization_id": organization_id,
                "type": "customer",
                "first_name": "Facebook",
                "last_name": "User",
                "facebook_id": sender_id,
                "created_at": datetime.utcnow().isoformat()
            }
            
            logger.info(f"Creating basic contact with data: {json.dumps(contact_data, indent=2)}")
            contact_response = supabase.table("contacts").insert(contact_data).execute()
            contact = contact_response.data[0] if contact_response.data else None
            
            if contact:
                logger.info(f"Created basic contact with ID: {contact['id']}")
                return contact
            else:
                logger.error("Failed to create basic contact")
                return None
        
        # If minimal request was successful, extract the data
        user_data = user_response.json()
        
        # Extract basic info
        first_name = user_data.get("first_name", "Facebook")
        last_name = user_data.get("last_name", "User")
        
        # Get profile picture URL 
        profile_pic_url = None
        if "picture" in user_data and "data" in user_data["picture"]:
            profile_pic_url = user_data["picture"]["data"].get("url")
            logger.info(f"Found profile picture URL: {profile_pic_url}")
        
        # Use direct URL as fallback
        if not profile_pic_url:
            profile_pic_url = f"https://graph.facebook.com/{sender_id}/picture?type=large"
        
        # Create contact with basic info
        contact_data = {
            "organization_id": organization_id,
            "type": "customer",
            "first_name": first_name,
            "last_name": last_name,
            "facebook_id": sender_id,
            "profile_picture_url": profile_pic_url,
            "created_at": datetime.utcnow().isoformat()
        }
        
        # Create the contact
        logger.info(f"Creating contact with data: {json.dumps(contact_data, indent=2)}")
        contact_response = supabase.table("contacts").insert(contact_data).execute()
        contact = contact_response.data[0] if contact_response.data else None
        
        if not contact:
            logger.error("Failed to create contact in database")
            return None
            
        logger.info(f"Successfully created contact with ID: {contact['id']}")
        
        # Try to get additional fields in a separate request (if it fails, we still have a basic contact)
        try:
            # Try to get more detailed user profile info
            additional_fields = [
                "email", 
                "gender", 
                "locale", 
                "timezone", 
                "about", 
                "birthday", 
                "location", 
                "website",
                "education",
                "work"        
            ]
            
            logger.info(f"Requesting additional Facebook data for user {sender_id}")
            additional_url = f"https://graph.facebook.com/v18.0/{sender_id}?fields={','.join(additional_fields)}&access_token={PAGE_ACCESS_TOKEN}"
            additional_response = requests.get(additional_url)
            
            # If additional request succeeds, enhance the profile
            if additional_response.status_code == 200:
                additional_data = additional_response.json()
                
                # Update the contact with email if available
                email = additional_data.get("email")
                if email:
                    logger.info(f"Updating contact with email: {email}")
                    supabase.table("contacts").update({"email": email}).eq("id", contact["id"]).execute()
                
                # Create profile with additional data if available
                if contact.get("id"):
                    # Create profile summary
                    profile_summary_parts = []
                    if additional_data.get("about"):
                        profile_summary_parts.append(additional_data.get("about"))
                    
                    location_info = ""
                    if additional_data.get("location") and additional_data["location"].get("name"):
                        location_info = f" from {additional_data['location'].get('name')}"
                    
                    if not profile_summary_parts:
                        profile_summary_parts.append(f"Facebook user {first_name} {last_name}{location_info}.")
                    
                    # Create general info
                    general_info = {"source": "Facebook Messenger"}
                    
                    for field in ["gender", "locale", "timezone", "birthday"]:
                        if additional_data.get(field):
                            general_info[field] = additional_data.get(field)
                    
                    if additional_data.get("location") and additional_data["location"].get("name"):
                        general_info["location"] = additional_data["location"].get("name")
                    
                    # Process work and education if available
                    work_history = additional_data.get("work", [])
                    education_history = additional_data.get("education", [])
                    
                    # Extract work history
                    if work_history:
                        work_info = []
                        for work_item in work_history:
                            work_entry = {}
                            if "employer" in work_item and "name" in work_item["employer"]:
                                work_entry["employer"] = work_item["employer"]["name"]
                            if "position" in work_item and "name" in work_item["position"]:
                                work_entry["position"] = work_item["position"]["name"]
                            if work_entry:
                                work_info.append(work_entry)
                        
                        if work_info:
                            general_info["work_history"] = work_info
                    
                    # Extract education
                    if education_history:
                        education_info = []
                        for edu_item in education_history:
                            edu_entry = {}
                            if "school" in edu_item and "name" in edu_item["school"]:
                                edu_entry["school"] = edu_item["school"]["name"]
                            if "type" in edu_item:
                                edu_entry["type"] = edu_item["type"]
                            if edu_entry:
                                education_info.append(edu_entry)
                        
                        if education_info:
                            general_info["education_history"] = education_info
                    
                    # Social media URLs
                    social_media_urls = [{"platform": "facebook", "url": f"https://facebook.com/{sender_id}"}]
                    
                    if additional_data.get("website"):
                        website_url = additional_data.get("website")
                        if isinstance(website_url, str):
                            social_media_urls.append({"platform": "website", "url": website_url})
                    
                    # Create profile
                    profile_data = {
                        "contact_id": contact["id"],
                        "profile_summary": "\n".join(profile_summary_parts),
                        "general_info": json.dumps(general_info),
                        "personality": f"{first_name} is a Facebook Messenger user.",
                        "social_media_urls": social_media_urls,
                        "best_goals": [{"goal": "Get information or assistance", "importance": "High"}],
                        "updated_at": datetime.utcnow().isoformat()
                    }
                    
                    logger.info(f"Creating profile with additional data for contact {contact['id']}")
                    supabase.table("profiles").insert(profile_data).execute()
            else:
                # Create a basic profile even if additional data request fails
                create_basic_profile(contact, first_name, last_name)
                
        except Exception as profile_err:
            logger.error(f"Error creating enhanced profile: {str(profile_err)}")
            # Still create a basic profile
            create_basic_profile(contact, first_name, last_name)
        
        return contact
        
    except Exception as e:
        logger.error(f"Error creating contact from Facebook: {str(e)}", exc_info=True)
        return None

def create_basic_profile(contact: Dict[str, Any], first_name: str, last_name: str) -> bool:
    """
    Create a basic profile for a contact with minimal information.
    
    Args:
        contact: The contact dictionary
        first_name: Contact's first name
        last_name: Contact's last name
        
    Returns:
        Success status
    """
    try:
        if not contact or not contact.get("id"):
            return False
            
        profile_data = {
            "contact_id": contact["id"],
            "profile_summary": f"Facebook user {first_name} {last_name}.",
            "general_info": json.dumps({"source": "Facebook Messenger"}),
            "personality": f"{first_name} is a Facebook Messenger user.",
            "social_media_urls": [{"platform": "facebook", "url": f"https://facebook.com/{contact['facebook_id']}"}],
            "best_goals": [{"goal": "Get information or assistance", "importance": "High"}],
            "updated_at": datetime.utcnow().isoformat()
        }
        
        logger.info(f"Creating basic profile for contact {contact['id']}")
        profile_response = supabase.table("profiles").insert(profile_data).execute()
        return True
    except Exception as e:
        logger.error(f"Error creating basic profile: {str(e)}")
        return False

def process_facebook_webhook(data: Dict[str, Any], convo_manager, org_id: str = None) -> bool:
    """
    Process an incoming Facebook webhook event - extract message data,
    find or create contact, and save the message to the database.
    Skip echo messages that we already tracked from our outgoing messages.
    
    Args:
        data: The webhook event data
        convo_manager: Instance of ConversationManager
        org_id: Optional organization ID for determining which integration to use
        
    Returns:
        Success status as boolean
    """
    try:
        # Parse message into standardized format first
        fb_message = parse_fb_message(data)
        
        # Check if this is an echo (message sent by our page)
        is_echo = fb_message.get("metadata", {}).get("is_echo", False)
        
        # Get the original message data (for backward compatibility)
        message_data = get_sender_text(data)
        sender_id = message_data.get("senderID", "UNKNOWN")
        
        # For echo messages (sent by page):
        # - recipient_id is the user who received the message
        # - For non-echo messages, user_id is the sender (who sent message to page)
        user_id = fb_message.get("recipient_id") if is_echo else sender_id
        
        logger.info(f"Processing {'echo' if is_echo else 'regular'} message for user ID: {user_id}")
        
        # Skip echo messages - we already save outgoing messages when we send them
        if is_echo:
            logger.info(f"Skipping echo message as we already saved it when sending")
            return True
        
        # Get or create contact
        contact = get_or_create_contact_by_facebook_id(user_id, org_id)
        
        if not contact:
            logger.error(f"Could not process message: No contact found or created for ID {user_id}")
            return False
        
        # Save the message to the conversation
        # Use user_id as the platform_conversation_id
        # This ensures all messages with a user are in the same conversation thread
        conversation = save_fb_message_to_conversation(
            contact_id=contact["id"],
            platform_conversation_id=user_id,
            fb_message=fb_message,
            convo_manager=convo_manager
        )
        
        logger.info(f"✅ Saved Facebook message to conversation for contact {contact['id']}")
        
        # Return success
        return True
    except Exception as e:
        logger.error(f"❌ Error processing Facebook webhook: {str(e)}", exc_info=True)
        return False

def send_text_to_facebook_user(user_id: str, text_message: str, convo_manager, contact_id: int = None, org_id: str = None) -> bool:
    """
    Simplified helper to send a text message to a Facebook user and save it in conversations.
    
    Args:
        user_id: Facebook user ID to send to
        text_message: Plain text message to send
        convo_manager: ConversationManager instance
        contact_id: Optional contact ID if already known
        org_id: Optional organization ID for determining which integration to use
    
    Returns:
        Boolean success status
    """
    try:
        # Create Facebook message format
        message_data = {"text": text_message}
        
        # Send the message
        success, conversation = send_and_save_message(
            recipient_id=user_id,
            message_data=message_data,
            convo_manager=convo_manager,
            contact_id=contact_id,
            org_id=org_id
        )
        
        return success == 1
    except Exception as e:
        logger.error(f"Error sending text to Facebook user {user_id}: {str(e)}")
        return False

def send_image_to_facebook_user(user_id: str, image_url: str, convo_manager, contact_id: int = None, org_id: str = None) -> bool:
    """
    Simplified helper to send an image to a Facebook user and save it in conversations.
    
    Args:
        user_id: Facebook user ID to send to
        image_url: URL of the image to send
        convo_manager: ConversationManager instance
        contact_id: Optional contact ID if already known
        org_id: Optional organization ID for determining which integration to use
    
    Returns:
        Boolean success status
    """
    try:
        # Create Facebook message format for image
        message_data = {
            "attachment": {
                "type": "image",
                "payload": {
                    "url": image_url,
                    "is_reusable": True
                }
            }
        }
        
        # Send the message
        success, conversation = send_and_save_message(
            recipient_id=user_id,
            message_data=message_data,
            convo_manager=convo_manager,
            contact_id=contact_id,
            org_id=org_id
        )
        
        return success == 1
    except Exception as e:
        logger.error(f"Error sending image to Facebook user {user_id}: {str(e)}")
        return False
