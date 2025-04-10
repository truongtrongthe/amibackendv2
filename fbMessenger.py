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


def send_message(sender_id, message_data):
    url = f"https://graph.facebook.com/v22.0/me/messages?access_token={PAGE_ACCESS_TOKEN}"
    headers = {"Content-Type": "application/json"}

    payload = {
        "recipient": {"id": sender_id},
        "messaging_type": "RESPONSE",
        "message": message_data
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            return 1
        else:
            print("❌ Error sending message:", response.text)
            return 0
    except requests.exceptions.RequestException as e:
        print("❌ Exception during sending:", str(e))
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
    
    # Parse basic message information
    sender_id = messaging.get("sender", {}).get("id")
    recipient_id = messaging.get("recipient", {}).get("id")
    timestamp = messaging.get("timestamp")
    
    # Convert timestamp to ISO format if it's a Unix timestamp
    if timestamp and isinstance(timestamp, int):
        timestamp = datetime.fromtimestamp(timestamp/1000.0).isoformat()
    
    # Initialize message object
    message_obj = {
        "id": f"fb_{timestamp}_{sender_id}" if timestamp else f"fb_{datetime.utcnow().isoformat()}_{sender_id}",
        "platform_msg_id": messaging.get("message", {}).get("mid"),
        "sender_id": sender_id,
        "sender_type": "contact",  # Assuming messages from Facebook are from contacts
        "recipient_id": recipient_id,
        "timestamp": timestamp or datetime.utcnow().isoformat(),
        "platform": "facebook",
        "status": "received"
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
        "facebook": messaging
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

def send_and_save_message(contact_id: int, platform_conversation_id: str, 
                        sender_id: str, message_data: Dict[str, Any], 
                        convo_manager) -> Tuple[int, Optional[Dict[str, Any]]]:
    """
    Send a message to Facebook and save it to our conversation system
    
    Args:
        contact_id: ID of the contact in our system
        platform_conversation_id: Facebook's conversation ID
        sender_id: Facebook user ID to send message to
        message_data: Message data to send (Facebook format)
        convo_manager: Instance of ConversationManager
        
    Returns:
        Tuple of (success_status, updated_conversation)
    """
    # First send the message to Facebook
    success = send_message(sender_id, message_data)
    
    if success:
        # Create a message object for our database
        timestamp = datetime.utcnow().isoformat()
        message_obj = {
            "id": f"fb_sent_{timestamp}",
            "sender_id": "system",  # Or could be the actual user ID if available
            "sender_type": "system",
            "recipient_id": sender_id,
            "content": message_data.get("text", str(message_data)),
            "content_type": "text" if "text" in message_data else "complex",
            "timestamp": timestamp,
            "platform": "facebook",
            "status": "sent",
            "metadata": {
                "facebook": message_data
            }
        }
        
        # Save to conversation
        conversation = save_fb_message_to_conversation(
            contact_id=contact_id,
            platform_conversation_id=platform_conversation_id,
            fb_message=message_obj,
            convo_manager=convo_manager
        )
        
        return success, conversation
    
    return success, None

def get_or_create_contact_by_facebook_id(sender_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a contact by Facebook ID or create a new one if not found.
    
    Args:
        sender_id: Facebook user ID
        
    Returns:
        Contact dict or None if creation failed
    """
    # Try to find existing contact
    contact_response = supabase.table("contacts").select("*").eq("facebook_id", sender_id).execute()
    contact = contact_response.data[0] if contact_response.data else None
    
    if contact:
        logger.info(f"Found existing contact for Facebook ID {sender_id}: {contact['id']}")
        return contact
    
    # No contact found, create a new one
    try:
        # Validated Facebook Graph API fields
        # Reference: https://developers.facebook.com/docs/graph-api/reference/user/
        fields = [
            "id",
            "first_name",
            "last_name", 
            "picture.type(large)",  # Use picture.type(large) instead of profile_pic
            "email", 
            "gender", 
            "locale", 
            "timezone", 
            "about", 
            "birthday", 
            "location", 
            "website",
            "education",  # Education history
            "work"        # Work history
        ]
        
        logger.info(f"Requesting Facebook data for user {sender_id} with fields: {', '.join(fields)}")
        user_url = f"https://graph.facebook.com/v18.0/{sender_id}?fields={','.join(fields)}&access_token={PAGE_ACCESS_TOKEN}"
        user_response = requests.get(user_url)
        
        # Log raw response for debugging
        logger.info(f"Facebook API status code: {user_response.status_code}")
        
        # Check for error response
        if user_response.status_code != 200:
            logger.error(f"Facebook API error: {user_response.text}")
            raise Exception(f"Facebook API returned status {user_response.status_code}")
            
        user_data = user_response.json()
        
        # Log sanitized response (removing access token)
        sanitized_response = json.dumps(user_data, indent=2)
        logger.info(f"Facebook user data received: {sanitized_response}")
        
        # Extract basic contact info
        first_name = user_data.get("first_name", "Facebook")
        last_name = user_data.get("last_name", "User")
        
        # Get profile picture URL properly from Facebook
        # The field picture.type(large) returns a nested object
        profile_pic_url = None
        if "picture" in user_data and "data" in user_data["picture"]:
            profile_pic_url = user_data["picture"]["data"].get("url")
            logger.info(f"Found profile picture URL: {profile_pic_url}")
        
        # Fallback to direct URL if not found
        if not profile_pic_url:
            logger.info("No profile picture found in response, using direct URL")
            profile_pic_url = f"https://graph.facebook.com/{sender_id}/picture?type=large&access_token={PAGE_ACCESS_TOKEN}"
        
        # Get email if available (requires specific permissions)
        email = user_data.get("email")
        if email:
            logger.info(f"Email found for user: {email}")
        else:
            logger.info("No email available for user")
        
        # Extract work and education
        work_history = user_data.get("work", [])
        education_history = user_data.get("education", [])
        
        if work_history:
            logger.info(f"Work history found: {len(work_history)} entries")
        
        if education_history:
            logger.info(f"Education history found: {len(education_history)} entries")
        
        # Create the contact
        contact_data = {
            "type": "customer",
            "first_name": first_name,
            "last_name": last_name,
            "email": email,
            "facebook_id": sender_id,
            "profile_picture_url": profile_pic_url,
            "created_at": datetime.utcnow().isoformat()
        }
        
        logger.info(f"Creating new contact with data: {json.dumps(contact_data, indent=2)}")
        contact_response = supabase.table("contacts").insert(contact_data).execute()
        contact = contact_response.data[0] if contact_response.data else None
        
        if not contact:
            logger.error("Failed to create contact in database")
            return None
            
        logger.info(f"Successfully created contact with ID: {contact['id']}")
        
        # If contact was created and profile info exists, create a profile with additional info
        if contact and contact.get("id"):
            # Collect additional profile data from Facebook
            profile_summary_parts = []
            if user_data.get("about"):
                profile_summary_parts.append(user_data.get("about"))
                
            location_info = ""
            if user_data.get("location") and user_data["location"].get("name"):
                location_info = f" from {user_data['location'].get('name')}"
            
            if not profile_summary_parts:
                # Create a default summary if none exists
                profile_summary_parts.append(f"Facebook user {first_name} {last_name}{location_info}.")
            
            # Basic profile info as JSON
            general_info = {}
            if user_data.get("gender"):
                general_info["gender"] = user_data.get("gender")
            if user_data.get("locale"):
                general_info["locale"] = user_data.get("locale")
            if user_data.get("timezone"):
                general_info["timezone"] = user_data.get("timezone")
            if user_data.get("birthday"):
                general_info["birthday"] = user_data.get("birthday")
            if user_data.get("location") and user_data["location"].get("name"):
                general_info["location"] = user_data["location"].get("name")
            
            # Process work history
            work_info = []
            for work_item in work_history:
                work_entry = {}
                if "employer" in work_item and "name" in work_item["employer"]:
                    work_entry["employer"] = work_item["employer"]["name"]
                if "position" in work_item and "name" in work_item["position"]:
                    work_entry["position"] = work_item["position"]["name"]
                if "start_date" in work_item:
                    work_entry["start_date"] = work_item["start_date"]
                if "end_date" in work_item:
                    work_entry["end_date"] = work_item["end_date"]
                if work_entry:
                    work_info.append(work_entry)
            
            if work_info:
                general_info["work_history"] = work_info
                
            # Process education history
            education_info = []
            for edu_item in education_history:
                edu_entry = {}
                if "school" in edu_item and "name" in edu_item["school"]:
                    edu_entry["school"] = edu_item["school"]["name"]
                if "type" in edu_item:
                    edu_entry["type"] = edu_item["type"]
                if "year" in edu_item and "name" in edu_item["year"]:
                    edu_entry["year"] = edu_item["year"]["name"]
                if "concentration" in edu_item:
                    concentrations = []
                    for concentration in edu_item["concentration"]:
                        if "name" in concentration:
                            concentrations.append(concentration["name"])
                    if concentrations:
                        edu_entry["concentration"] = concentrations
                if edu_entry:
                    education_info.append(edu_entry)
            
            if education_info:
                general_info["education_history"] = education_info
            
            # Add Facebook as source
            general_info["source"] = "Facebook Messenger"
            
            # Social media info
            social_media_urls = []
            if user_data.get("website"):
                website_url = user_data.get("website")
                # Ensure website URL is a string
                if isinstance(website_url, str):
                    social_media_urls.append({"platform": "website", "url": website_url})
                elif isinstance(website_url, list) and website_url:
                    # If it's a list (sometimes Facebook returns multiple websites)
                    social_media_urls.append({"platform": "website", "url": website_url[0]})
            
            # Always add Facebook profile as a social media link
            facebook_url = f"https://facebook.com/{sender_id}"
            social_media_urls.append({
                "platform": "facebook", 
                "url": facebook_url
            })
            
            # Log social media URLs for debugging
            logger.info(f"Social media URLs: {json.dumps(social_media_urls)}")
            
            # Create some default personality traits based on available info
            personality_traits = f"{first_name} is a Facebook Messenger user who initiated contact with our platform."
            
            # Add some default goals - this is placeholder content
            # In a real system, these would be determined through conversation analysis
            best_goals = [
                {
                    "goal": "Get information or assistance",
                    "deadline": "Ongoing",
                    "importance": "High"
                }
            ]
            
            # Create profile in Supabase
            try:
                profile_data = {
                    "contact_id": contact["id"],
                    "profile_summary": "\n".join(profile_summary_parts),
                    "general_info": json.dumps(general_info),
                    "personality": personality_traits,
                    "social_media_urls": social_media_urls,
                    "best_goals": best_goals,
                    "linkedin_url": None,
                    "updated_at": datetime.utcnow().isoformat()
                }
                
                logger.info(f"Creating profile for contact {contact['id']} with data: {json.dumps(profile_data, indent=2)}")
                profile_response = supabase.table("profiles").insert(profile_data).execute()
                logger.info(f"✅ Created profile for Facebook user {sender_id}")
            except Exception as profile_err:
                logger.error(f"Error creating profile for Facebook user: {str(profile_err)}")
        
        return contact
        
    except Exception as e:
        logger.error(f"Error creating contact from Facebook: {str(e)}", exc_info=True)
        return None

def process_facebook_webhook(data: Dict[str, Any], convo_manager) -> bool:
    """
    Process an incoming Facebook webhook event - extract message data,
    find or create contact, and save the message to the database.
    
    Args:
        data: The webhook event data
        convo_manager: Instance of ConversationManager
        
    Returns:
        Success status as boolean
    """
    try:
        # Get message data
        message_data = get_sender_text(data)
        sender_id = message_data["senderID"]
        
        # Parse the message into our standardized format
        fb_message = parse_fb_message(data)
        
        # Get or create contact
        contact = get_or_create_contact_by_facebook_id(sender_id)
        
        if not contact:
            print("❌ Could not process message: No contact found or created")
            return False
        
        # Save the message to the conversation
        conversation = save_fb_message_to_conversation(
            contact_id=contact["id"],
            platform_conversation_id=sender_id,  # Using Facebook sender_id as conversation ID
            fb_message=fb_message,
            convo_manager=convo_manager
        )
        
        print(f"✅ Saved Facebook message to conversation for contact {contact['id']}")
        
        # Return success
        return True
    except Exception as e:
        print(f"❌ Error processing Facebook webhook: {str(e)}")
        return False
