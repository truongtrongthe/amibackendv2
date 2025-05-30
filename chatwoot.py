from fastapi import APIRouter, Request, Response, Depends, Header, Query, HTTPException
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from collections import deque, defaultdict
import requests
import os
from dotenv import load_dotenv
from contact import ContactManager
from contactconvo import ConversationManager
import time
import multiprocessing
import queue
import threading
import logging
import asyncio
from functools import partial
import concurrent.futures

# Add multiprocessing freeze support
from multiprocessing import freeze_support

# Load environment variables
load_dotenv()

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Create an APIRouter instead of Flask app
router = APIRouter()

# Initialize managers
contact_manager = ContactManager()
conversation_manager = ConversationManager()

# Chatwoot API configuration
CHATWOOT_API_TOKEN = os.getenv('CHATWOOT_API_TOKEN')
CHATWOOT_BASE_URL = os.getenv('CHATWOOT_BASE_URL', 'https://app.chatwoot.com')
CHATWOOT_ACCOUNT_ID = os.getenv('CHATWOOT_ACCOUNT_ID')

# AI response configuration
ENABLE_AI_RESPONSES = os.getenv('ENABLE_AI_RESPONSES', 'false').lower() == 'true'
AI_RESPONSE_BLOCKLIST = os.getenv('AI_RESPONSE_BLOCKLIST', '').split(',')  # Comma-separated list of conversation IDs to exclude
AI_IGNORE_PREFIX = os.getenv('AI_IGNORE_PREFIX', '!').strip()  # Messages starting with this prefix won't trigger AI
AI_RESPONSE_DELAY = float(os.getenv('AI_RESPONSE_DELAY', '3.0'))  # Delay in seconds before sending response
DEFAULT_GRAPH_VERSION_ID = os.getenv('DEFAULT_GRAPH_VERSION_ID', '')  # Default graph version ID for knowledge retrieval

# Keep track of recent webhook requests to detect duplicates
recent_requests = deque(maxlen=100)
# Keep track of messages we've already responded to
processed_messages = deque(maxlen=200)

# Create global shared resources for the persistent AI process
ai_process = None
request_queue = None
response_queue = None
ai_process_running = False
process_lock = threading.Lock()

# Track last message time and pending messages by conversation
last_message_times = {}
pending_messages = defaultdict(list)
CONSECUTIVE_MESSAGE_WINDOW = 60.0  # seconds to wait for more messages

# Keep track of active tasks for each conversation
active_tasks = {}

def send_message(conversation_id: int, message: str, attachment_url: str = None) -> tuple:
    """
    Send a message to a conversation in Chatwoot.
    
    Args:
        conversation_id: The ID of the conversation to send to
        message: The message text to send
        attachment_url: Optional URL of an attachment to include
        
    Returns:
        tuple: (success_bool, response_data)
    """
    if not conversation_id:
        print("❌ Error: Missing conversation ID")
        return False, None
        
    # Allow empty messages if there's an attachment URL
    if (not message or not message.strip()) and not attachment_url:
        print("❌ Error: Empty message")
        return False, None
        
    if not CHATWOOT_API_TOKEN or not CHATWOOT_BASE_URL or not CHATWOOT_ACCOUNT_ID:
        print("❌ Error: Chatwoot API configuration is incomplete")
        return False, None
    
    print(f"\n📤 Sending message to conversation {conversation_id}")
    print(f"Message: {message}")
    if attachment_url:
        print(f"Attachment: {attachment_url}")
    
    headers = {
        'api_access_token': CHATWOOT_API_TOKEN,
        'Content-Type': 'application/json'
    }
    
    data = {
        'content': message,
        'message_type': 'outgoing',
        'private': 'false'  # Send as string "false" instead of boolean False
    }
    
    start_time = time.time()
    
    try:
        if attachment_url:
            print("📥 Downloading attachment...")
            try:
                # Download the image with timeout
                response = requests.get(attachment_url, timeout=10)
                if response.status_code != 200:
                    print(f"❌ Failed to download attachment: Status code {response.status_code}")
                    return False, None
                
                # Check file size (<= 40MB)
                file_size = len(response.content)
                max_size = 40 * 1024 * 1024  # 40MB
                if file_size > max_size:
                    print(f"❌ Attachment too large: {file_size} bytes (max 40MB). Sending as link instead.")
                    # Fallback: send the URL as a message
                    data = {
                        'content': attachment_url,
                        'message_type': 'outgoing',
                        'private': 'false'  # Send as string "false" instead of boolean False
                    }
                    url = f"{CHATWOOT_BASE_URL}/api/v1/accounts/{CHATWOOT_ACCOUNT_ID}/conversations/{conversation_id}/messages"
                    response = requests.post(
                        url,
                        headers=headers,
                        json=data,
                        timeout=10
                    )
                else:
                    # Save the image temporarily with a unique name
                    temp_filename = f'temp_attachment_{int(time.time())}.dat'
                    content_type = response.headers.get('Content-Type', 'application/octet-stream')
                    extension = get_file_extension(content_type)
                    
                    # Only allow supported image types
                    supported_types = ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp']
                    if extension and extension.lower() in supported_types:
                        temp_filename = f'temp_attachment_{int(time.time())}.{extension}'
                        with open(temp_filename, 'wb') as f:
                            f.write(response.content)
                        print(f"📋 Preparing multipart form data with attachment ({file_size} bytes)")
                        # Prepare multipart form data
                        files = {
                            'attachments[]': (os.path.basename(temp_filename), open(temp_filename, 'rb'), content_type)
                        }
                        # Send the request with multipart form data
                        url = f"{CHATWOOT_BASE_URL}/api/v1/accounts/{CHATWOOT_ACCOUNT_ID}/conversations/{conversation_id}/messages"
                        print(f"🔄 Sending request to: {url}")
                        response = requests.post(
                            url,
                            headers={'api_access_token': CHATWOOT_API_TOKEN},  # Content-Type is set automatically with files
                            data={
                                'content': message,
                                'message_type': 'outgoing',
                                'private': 'false'  # Send as string "false" instead of boolean False
                            },
                            files=files,
                            timeout=15
                        )
                        # Clean up the temporary file
                        try:
                            os.remove(temp_filename)
                            print(f"🗑️ Deleted temporary file {temp_filename}")
                        except Exception as e:
                            print(f"⚠️ Warning: Failed to delete temporary file: {str(e)}")
                    else:
                        print(f"❌ Unsupported file type: {content_type} ({extension}). Sending as link instead.")
                        # Fallback: send the URL as a message
                        data = {
                            'content': attachment_url,
                            'message_type': 'outgoing',
                            'private': 'false'  # Send as string "false" instead of boolean False
                        }
                        url = f"{CHATWOOT_BASE_URL}/api/v1/accounts/{CHATWOOT_ACCOUNT_ID}/conversations/{conversation_id}/messages"
                        response = requests.post(
                            url,
                            headers=headers,
                            json=data,
                            timeout=10
                        )
            except requests.RequestException as e:
                print(f"❌ Error handling attachment: {str(e)}")
                return False, None
            except Exception as e:
                print(f"❌ Error processing attachment: {str(e)}")
                return False, None
                
        else:
            # Send regular text message
            url = f"{CHATWOOT_BASE_URL}/api/v1/accounts/{CHATWOOT_ACCOUNT_ID}/conversations/{conversation_id}/messages"
            print(f"🔄 Sending request to: {url}")
            
            response = requests.post(
                url,
                headers=headers,
                json=data,
                timeout=10
            )
        
        elapsed_time = time.time() - start_time
        
        if response.status_code == 200:
            response_data = response.json()
            print(f"✅ Message sent successfully in {elapsed_time:.2f}s")
            print(f"Message ID: {response_data.get('id')}")
            return True, response_data
        else:
            print(f"❌ Failed to send message. Status: {response.status_code}")
            print(f"Response: {response.text}")
            return False, None
            
    except requests.RequestException as e:
        print(f"❌ Network error sending message: {str(e)}")
        return False, None
    except Exception as e:
        print(f"❌ Unexpected error sending message: {str(e)}")
        return False, None

def get_file_extension(content_type: str) -> str:
    """
    Get file extension from content type.
    
    Args:
        content_type: MIME type of the content
        
    Returns:
        String extension or empty string if unknown
    """
    type_map = {
        'image/jpeg': 'jpg',
        'image/jpg': 'jpg',
        'image/png': 'png',
        'image/gif': 'gif',
        'image/webp': 'webp',
        'image/svg+xml': 'svg',
        'video/mp4': 'mp4',
        'video/mpeg': 'mpeg',
        'video/quicktime': 'mov',
        'audio/mpeg': 'mp3',
        'audio/mp4': 'mp4',
        'audio/wav': 'wav',
        'application/pdf': 'pdf',
        'application/msword': 'doc',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx',
        'application/vnd.ms-excel': 'xls',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'xlsx',
        'text/plain': 'txt',
        'text/csv': 'csv'
    }
    
    return type_map.get(content_type.lower(), '')

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

def handle_contact_creation(data: Dict[str, Any], organization_id: str = None) -> Dict[str, Any]:
    """
    Create a contact record from webhook data if it doesn't exist.
    
    Args:
        data: The webhook event data
        organization_id: The organization ID to associate the contact with
        
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
        existing_contact = contact_manager.get_contact_by_facebook_id(facebook_id, organization_id)
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
            organization_id=organization_id,
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
            existing_contact = contact_manager.get_contact_by_facebook_id(facebook_id, organization_id)
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

def split_into_messages(text: str, max_sentences: int = 2) -> List[str]:
    """
    Split text into smaller chunks for more natural conversation flow.
    Handles languages like Vietnamese where sentence boundaries might not have spaces.
    Also prepare a special image for QR code if found the bank account number in the text
    
    Args:
        text: The text to split
        max_sentences: Maximum number of sentences per chunk
        
    Returns:
        List of text chunks
    """
    import re
    
    # First, split by paragraphs (empty lines)
    paragraphs = re.split(r'\n\s*\n', text)
    
    result = []
    
    # Check if text contains specific keywords and add a special URL item
        
    for paragraph in paragraphs:
        if not paragraph.strip():
            continue
            
        # Enhanced sentence detection - handle periods followed by capital letters or spaces
        # This works for English and many other languages including Vietnamese
        sentence_pattern = r'(?<=[.!?])(?=\s*[A-ZÀ-Ỹ0-9]|\s|$)'
        sentences = re.split(sentence_pattern, paragraph.strip())
        
        # Clean up any empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Log what we found
        if sentences:
            print(f"Split paragraph into {len(sentences)} sentences")
            if len(sentences) > 3:
                print(f"First 3 sentences: {sentences[:3]}")
        
        # Group sentences into chunks of max_sentences
        for i in range(0, len(sentences), max_sentences):
            chunk = sentences[i:i+max_sentences]
            result.append(' '.join(chunk))
    if "Techcombank" in text or "6021111999" in text:
        print("Found bank information, adding QR code URL")
        result.append("@https://drive.google.com/uc?export=download&id=1tzbJNfwulNE7KK1hgeW3kCGx8RO0hs-o")

    # If we ended up with nothing, use fallback chunking
    if not result:
        print("No sentences detected, using fallback chunking")
        # Split into chunks of about 150 characters, breaking at spaces
        chunks = []
        current_pos = 0
        text_length = len(text)
        
        while current_pos < text_length:
            end_pos = min(current_pos + 150, text_length)
            
            # Try to find a space to break at
            if end_pos < text_length:
                space_pos = text.rfind(' ', current_pos, end_pos + 20)  # Look up to 20 chars ahead
                if space_pos > current_pos:
                    end_pos = space_pos + 1
            
            chunks.append(text[current_pos:end_pos].strip())
            current_pos = end_pos
        
        return chunks
        
    print(f"Created {len(result)} message chunks")
    return result

# Function to run in the separate process - moved outside to make it picklable
def ai_process_function(request_queue, response_queue):
    """
    Long-running function that handles AI requests in a separate process.
    This way we only initialize the AI system once.
    
    Args:
        request_queue: Queue to receive requests from the main process
        response_queue: Queue to send responses back to the main process
    """
    try:
        # Import here to ensure clean environment in subprocess
        from ami import convo_stream
        import json
        import time
        import os
        
        print("🚀 Starting persistent AI process")
        
        # IMPORTANT: Pre-set environment flags to optimize personality loading
        os.environ['SKIP_PERSONALITY_LOAD'] = 'true'
        os.environ['USE_DEFAULT_PERSONALITY'] = 'true'
        
        print("✅ AI process initialized and ready to handle requests")
        
        # Add a heartbeat mechanism to detect process health
        last_heartbeat = time.time()
        HEARTBEAT_INTERVAL = 60  # 60 seconds
        
        # Process requests until shutdown
        while True:
            try:
                # Get request from queue with timeout
                request = request_queue.get(timeout=300)  # 5-minute timeout
                
                # Check for shutdown signal
                if request == "SHUTDOWN":
                    print("🛑 Received shutdown signal, closing AI process")
                    break
                
                # Check for heartbeat request
                if request == "HEARTBEAT":
                    response_queue.put("ALIVE")
                    last_heartbeat = time.time()
                    continue
                
                # Update heartbeat
                last_heartbeat = time.time()
                
                # Process the request
                message, user_id, thread_id, graph_version_id = request
                print(f"📩 Processing request: message='{message[:50]}...', user={user_id}, thread={thread_id}")
                
                start_time = time.time()
                
                # Process the request with a timeout
                MAX_PROCESSING_TIME = 300  # 120 seconds maximum processing time (increased from 30s)
                response_chunks = []
                
                try:
                    # Use a separate thread with timeout for processing
                    import threading
                    import queue as py_queue
                    
                    result_queue = py_queue.Queue()
                    
                    def process_with_timeout():
                        try:
                            chunks = []
                            for chunk in convo_stream(
                                user_input=message,
                                user_id=user_id,
                                thread_id=thread_id,
                                graph_version_id=graph_version_id,
                                mode="mc"
                            ):
                                # Process the chunk
                                if chunk.startswith('data: '):
                                    try:
                                        data_json = json.loads(chunk[6:])
                                        if 'message' in data_json:
                                            chunks.append(data_json['message'])
                                    except Exception as e:
                                        print(f"Error processing chunk: {str(e)}")
                            result_queue.put(chunks)
                        except Exception as e:
                            result_queue.put(f"Error: {str(e)}")
                    
                    # Start processing thread
                    thread = threading.Thread(target=process_with_timeout)
                    thread.daemon = True
                    thread.start()
                    
                    # Wait for result with timeout
                    thread.join(MAX_PROCESSING_TIME)
                    
                    if thread.is_alive():
                        # Thread is still running after timeout
                        print(f"⚠️ AI processing timeout after {MAX_PROCESSING_TIME} seconds")
                        response_queue.put("Error: AI processing timeout")
                        continue
                    
                    # Get result
                    result = result_queue.get(block=False)
                    if isinstance(result, list):
                        response_chunks = result
                    else:
                        print(f"Error in AI processing: {result}")
                        response_queue.put(result)
                        continue
                    
                except Exception as e:
                    print(f"Error during AI processing: {str(e)}")
                    response_queue.put(f"Error: {str(e)}")
                    continue
                
                # Combine into a single response
                full_response = ' '.join(response_chunks)
                processing_time = time.time() - start_time
                
                print(f"✅ Request processed in {processing_time:.2f}s, response length: {len(full_response)} chars")
                
                # Send response back
                response_queue.put(full_response)
                
            except queue.Empty:
                # Check if we've been silent for too long
                if time.time() - last_heartbeat > HEARTBEAT_INTERVAL * 2:
                    print("⚠️ AI process appears to be stalled - no activity for too long")
                    break
                    
                print("⏰ AI process timeout - no requests received in 5 minutes")
                break
            except Exception as e:
                print(f"❌ Error in AI process: {str(e)}")
                import traceback
                traceback.print_exc()
                # Send error response
                response_queue.put(f"Error: {str(e)}")
        
        print("🏁 AI process shutting down")
    except Exception as e:
        print(f"💥 Critical error in AI process: {str(e)}")
        import traceback
        traceback.print_exc()

def start_ai_process():
    """
    Start the persistent AI process if it's not already running.
    """
    global ai_process, request_queue, response_queue, ai_process_running, process_lock
    
    with process_lock:
        if not ai_process_running:
            # Create the queues
            request_queue = multiprocessing.Queue()
            response_queue = multiprocessing.Queue()
            
            # Create and start the process
            ai_process = multiprocessing.Process(
                target=ai_process_function,
                args=(request_queue, response_queue)
            )
            ai_process.daemon = True  # Process will terminate when main process exits
            ai_process.start()
            
            ai_process_running = True
            print(f"🚀 Started persistent AI process with PID: {ai_process.pid}")
            
            # Give the process time to initialize
            time.sleep(1)
        else:
            print("🔄 AI process already running")

def stop_ai_process():
    """
    Stop the persistent AI process.
    """
    global ai_process, request_queue, response_queue, ai_process_running, process_lock
    
    with process_lock:
        if ai_process_running:
            # Send shutdown signal
            try:
                request_queue.put("SHUTDOWN")
                ai_process.join(timeout=5)
            except:
                pass
            
            # Force terminate if still running
            if ai_process.is_alive():
                ai_process.terminate()
                ai_process.join()
            
            ai_process_running = False
            print("🛑 Stopped persistent AI process")

async def generate_ai_response(message_text: str, user_id: str = None, thread_id: str = None, graph_version_id: str = None, organization_id: str = None):
    """
    Helper function to generate an AI response without needing Chatwoot integration.
    Uses direct processing to avoid startup overhead.
    
    Args:
        message_text: The user's message to respond to
        user_id: Optional user ID for the conversation
        thread_id: Optional thread ID for the conversation
        graph_version_id: Optional graph version ID for knowledge retrieval
        organization_id: Optional organization ID for usage tracking
        
    Returns:
        A list of message chunks for a more natural conversation flow
    """
    if not user_id:
        user_id = f"test_user_{int(datetime.now().timestamp())}"
    
    if not thread_id:
        thread_id = f"test_thread_{int(datetime.now().timestamp())}"
    
    # Prepare enhanced message with conversation history
    enhanced_message = message_text
    
    try:
        print(f"\n🔄 Starting AI response generation with thread_id: {thread_id}")
        
        try:
            from ami import convo_stream
            
            # Process the response directly with async/await and implement timeout handling
            loop = asyncio.get_event_loop()
            response_chunks = []
            
            # Use ThreadPoolExecutor for potentially blocking operations
            with concurrent.futures.ThreadPoolExecutor() as executor:
                try:
                    # Run in executor with a timeout
                    result = await loop.run_in_executor(
                        executor,
                        partial(process_ai_response, enhanced_message, user_id, thread_id, graph_version_id)
                    )
                    response_chunks = result
                except Exception as e:
                    print(f"Error processing AI response: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    return ["I'm sorry, I encountered an error while processing your request. The team has been notified."]
            
            # Combine all response chunks into a single text
            full_response = ' '.join(response_chunks)
            print(f"✅ AI response generation complete. Response length: {len(full_response)} characters")
            
            if not full_response.strip():
                print("Warning: Generated response is empty!")
                return ["I'm sorry, I couldn't generate a proper response. Could you try asking in a different way?"]
            
            # Split the full response into smaller message chunks
            message_chunks = split_into_messages(full_response, max_sentences=2)

            print(f"✂️ Split response into {len(message_chunks)} message chunks")
            
            # Track usage for the organization
            if organization_id:
                try:
                    from usage import OrganizationUsage
                    
                    # Initialize usage tracking for this organization
                    org_usage = OrganizationUsage(organization_id)
                    
                    # Track message usage - one count per message chunk
                    logger.info(f"Adding message count for org at Chatwoot.py: {organization_id}: {len(message_chunks)}")
                    org_usage.add_message(len(message_chunks))
                    
                    # Track reasoning usage - use a count of 1 for the reasoning operation
                    org_usage.add_reasoning(1)
                    
                    print(f"✓ Tracked usage for organization {organization_id}: {len(message_chunks)} messages, 1 reasoning")
                except Exception as e:
                    print(f"❌ Error tracking usage: {str(e)}")
                    import traceback
                    traceback.print_exc()
            
            return message_chunks
            
        except ImportError as e:
            print(f"Error importing necessary modules: {str(e)}")
            return ["I'm sorry, I'm having trouble accessing my AI capabilities right now."]
        except Exception as e:
            print(f"Error in AI processing: {str(e)}")
            import traceback
            traceback.print_exc()
            return ["I'm sorry, I encountered an error while processing your request. The team has been notified."]
        
    except Exception as e:
        print(f"❌ Critical error generating AI response: {str(e)}")
        import traceback
        traceback.print_exc()
        return ["I'm sorry, I encountered an error while processing your request. The team has been notified."]

# Add a new helper function for AI response processing
def process_ai_response(message, user_id, thread_id, graph_version_id):
    """Helper function to process AI response in a separate thread"""
    from ami import convo_stream
    import json
    import asyncio
    
    response_chunks = []
    # This will be run in a thread, so we need to run the async generator in a new event loop
    async def process_stream():
        async for chunk in convo_stream(
            user_input=message,
            user_id=user_id,
            thread_id=thread_id,
            graph_version_id=graph_version_id,
            mode="mc"
        ):
            # Process the chunk
            if chunk.startswith('data: '):
                try:
                    data_json = json.loads(chunk[6:])
                    if 'message' in data_json:
                        message_content = data_json['message']
                        response_chunks.append(message_content)
                        print(f"Received chunk: {message_content[:50]}...")
                except json.JSONDecodeError:
                    print(f"Error parsing chunk: {chunk}")
    
    # Create a new event loop and run the coroutine
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(process_stream())
    loop.close()
    
    return response_chunks

# Modify process_messages_with_batching function to include timeouts and better error handling
async def process_messages_with_batching(conv_id_str: str):
    """Process messages in batches of 2 with a delay between batches."""
    try:
        # Add a maximum queue size limit to prevent memory issues
        MAX_QUEUE_SIZE = 20  # Maximum pending messages to keep
        if len(pending_messages[conv_id_str]) > MAX_QUEUE_SIZE:
            print(f"⚠️ Queue size limit reached ({len(pending_messages[conv_id_str])} messages). Trimming to {MAX_QUEUE_SIZE}.")
            # Keep only the most recent messages
            pending_messages[conv_id_str] = pending_messages[conv_id_str][-MAX_QUEUE_SIZE:]
            
        # Initial delay to allow more messages to accumulate
        wait_seconds = CONSECUTIVE_MESSAGE_WINDOW
        print(f"⏱️ Waiting {wait_seconds} seconds to allow more messages to accumulate...")
        await asyncio.sleep(wait_seconds)  # Full CONSECUTIVE_MESSAGE_WINDOW seconds
        
        while pending_messages[conv_id_str]:
            # Get up to 10 messages from the queue
            batch = []
            while len(batch) < 10 and pending_messages[conv_id_str]:
                batch.append(pending_messages[conv_id_str].pop(0))
            
            if not batch:
                break
                
            print(f"Processing batch of {len(batch)} messages for conversation {conv_id_str}")
            
            # Combine messages in the batch with "User:" prefix for each message
            formatted_messages = []
            for m in batch:
                # Add "User:" prefix to each incoming message for AI context
                formatted_message = f"User: {m['content']}"
                formatted_messages.append(formatted_message)
            
            combined_content = "\n".join(formatted_messages)
            print(f"Combined {len(batch)} messages into one request: \n{combined_content}")
            
            # Use the most recent message's metadata
            latest = batch[-1]
            context = latest['context']
            conversation_data = latest['conversation_data']
            contact_data = latest['contact_data']
            organization_id = latest['organization_id']
            
            # Create thread_id and user_id for AI response
            db_conversation_id = conversation_data['id']
            thread_id = f"chatwoot_{db_conversation_id}"
            user_id = f"chatwoot_{contact_data['id']}"
            
            print(f"\n🤖 Generating AI response for message batch: '{combined_content}'")
            
            # Add a delay to make the response seem more natural
            if AI_RESPONSE_DELAY > 0:
                delay_seconds = AI_RESPONSE_DELAY
                print(f"⏱️ Waiting {delay_seconds} seconds before responding...")
                await asyncio.sleep(delay_seconds)
            
            # Retrieve conversation history
            conversation_history = get_conversation_history(db_conversation_id, max_messages=100)
            print(f"Conversation history: {conversation_history}")

            # Add conversation history to the combined content
            combined_content = f"{conversation_history}\n{combined_content}"

            # Try to find a graph version ID
            graph_version_id = get_graph_version_id(organization_id) or DEFAULT_GRAPH_VERSION_ID
            print(f"Using graph_version_id: {graph_version_id if graph_version_id else 'None'}")
            
            # Generate and send AI response with timeout
            response_time_start = time.time()
            try:
                # Set a timeout for AI response generation
                message_chunks = await asyncio.wait_for(
                    generate_ai_response(
                        combined_content, 
                        user_id, 
                        thread_id, 
                        graph_version_id,
                        organization_id
                    ),
                    timeout=300.0  # 120 second timeout for AI response (increased from 30s)
                )
                response_time = time.time() - response_time_start
                print(f"⏱️ AI response generated in {response_time:.2f} seconds")
                
                if message_chunks:
                    # Set a timeout for message sending
                    await asyncio.wait_for(
                        send_message_chunks(message_chunks, context, conversation_data),
                        timeout=60.0  # 60 second timeout for sending all chunks
                    )
                else:
                    print("❌ No AI response generated")
            except asyncio.TimeoutError:
                print(f"⚠️ Timeout exceeded while processing messages for conversation {conv_id_str}")
                # Send a fallback message to avoid leaving the user hanging
                fallback_message = ["I'm sorry, I'm taking too long to process your request. Please try again in a moment."]
                try:
                    await send_message_chunks(fallback_message, context, conversation_data)
                except Exception as e:
                    print(f"Failed to send fallback message: {str(e)}")
            
            # Wait before processing the next batch (if any)
            if pending_messages[conv_id_str]:
                print(f"⏱️ Waiting before processing next batch...")
                await asyncio.sleep(CONSECUTIVE_MESSAGE_WINDOW / 3)  # 20 seconds
                
    except Exception as e:
        print(f"Error in batch message processing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up task reference when done
        if conv_id_str in active_tasks:
            del active_tasks[conv_id_str]

def get_conversation_history(conversation_id: int, max_messages: int = 10) -> str:
    """
    Retrieve and format recent conversation history for AI context.
    
    Args:
        conversation_id: The conversation ID
        max_messages: Maximum number of recent messages to include
        
    Returns:
        Formatted conversation history string
    """
    conversation_history = ""

    print(f"Getting conversation history for conversation_id: {conversation_id} with max_messages: {max_messages}")
    
    try:
        conversation = conversation_manager.get_conversation(conversation_id)
        if conversation and "messages" in conversation.get("conversation_data", {}):
            # Get last N messages for context (or fewer if there aren't enough)
            messages = conversation["conversation_data"]["messages"]
            recent_messages = messages[-max_messages:] if len(messages) > max_messages else messages
            
            # Format as readable conversation history
            if recent_messages:
                conversation_history = "Previous conversation:\n"
                for msg in recent_messages:
                    sender = "User" if msg.get("sender_type") == "contact" else "Assistant"
                    conversation_history += f"{sender}: {msg.get('content', '')}\n"
                
                print(f"📚 Added {len(recent_messages)} messages of conversation history for context")
            else:
                print("📝 No recent messages found in conversation data")
        else:
            print("📝 No messages found in conversation data")
    except Exception as e:
        print(f"❌ Error retrieving conversation history: {str(e)}")
        
    return conversation_history

def get_graph_version_id(organization_id: str = None) -> str:
    """
    Get the appropriate graph version ID for the organization.
    
    Args:
        organization_id: The organization ID
        
    Returns:
        Graph version ID string or empty string if not found
    """
    # This is a placeholder - implement logic to retrieve the correct graph version
    # based on organization settings or defaults
    return DEFAULT_GRAPH_VERSION_ID

async def send_message_chunks(message_chunks: List[str], context: Dict[str, Any], conversation_data: Dict[str, Any]):
    """
    Send message chunks with appropriate delays between them.
    
    Args:
        message_chunks: List of message chunks to send
        context: Conversation context
        conversation_data: Conversation data for persistence
    """
    # Configure delay between messages
    chunk_delay = AI_RESPONSE_DELAY / 2 if AI_RESPONSE_DELAY > 0 else 1.0
    chunk_delay = min(max(chunk_delay, 1.0), 3.0)  # Ensure between 1-3 seconds
    print(f"📤 Sending {len(message_chunks)} message chunks with {chunk_delay:.1f}s delay between them")
    
    # Send each chunk as a separate message with a small delay
    for i, chunk in enumerate(message_chunks):
        print(f"📤 Sending chunk {i+1}/{len(message_chunks)}: {chunk}")

        success = False
        response_data = None
        
        # Check if the chunk starts with @ which indicates it's an attachment URL
        if chunk.startswith('@'):
            attachment_url = chunk[1:]  # Remove the @ prefix
            print(f"📎 Detected attachment URL: {attachment_url}")
            # Send as attachment with empty message
            success, response_data = send_message(context['conversation_id'], "", attachment_url)
        else:
            # Send as regular text message
            success, response_data = send_message(context['conversation_id'], chunk)
        
        if success:
            # Only store the message ourselves if we have the platform_message_id
            # Otherwise, we'll rely on the webhook to store it
            platform_message_id = response_data.get('id') if response_data else None
            
            if platform_message_id:
                # Store with the platform_message_id to avoid duplication
                ai_message = {
                    'sender_type': 'agent',
                    'content': chunk,
                    'platform': 'chatwoot',
                    'platform_message_id': platform_message_id,
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'direction': 'outgoing',
                    'status': 'sent'
                    # No longer adding chunk_index and total_chunks to reduce duplication
                }
                
                # Check if we already have this message stored (by platform_message_id)
                should_store = True
                try:
                    conversation = conversation_manager.get_conversation(conversation_data['id'])
                    if conversation and "messages" in conversation.get("conversation_data", {}):
                        messages = conversation["conversation_data"]["messages"]
                        for msg in messages:
                            if msg.get('platform_message_id') == platform_message_id:
                                should_store = False
                                print(f"✅ Message already stored with platform_message_id: {platform_message_id}")
                                break
                except Exception as e:
                    print(f"⚠️ Error checking existing messages: {str(e)}")
                
                if should_store:
                    print(f"📝 Storing message with platform_message_id: {platform_message_id}")
                    conversation_manager.add_message(conversation_data['id'], ai_message)
            else:
                print("⚠️ No platform_message_id received, relying on webhook for storage")
        else:
            print(f"❌ Failed to send message chunk {i+1}")
        
        # Add delay between messages (except for the last one)
        if i < len(message_chunks) - 1:
            print(f"⏱️ Waiting {chunk_delay:.1f}s before sending next chunk...")
            await asyncio.sleep(chunk_delay)  # Use asyncio.sleep instead of time.sleep
    
    print(f"✅ Sent all {len(message_chunks)} message chunks")

async def handle_message_updated(data: Dict[str, Any], organization_id: str = None):
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
            
        contact_data = contact_manager.get_contact_by_facebook_id(facebook_id, organization_id)
        if not contact_data:
            print(f"Contact not found for Facebook ID: {facebook_id}")
            return
            
        # Get message details
        message_id = data.get('id')
        content = data.get('content')
        
        # Log update
        print(f"📝 Message updated:")
        
        
        # Find the conversation by platform message ID if available
        conversation_id = None
        conversations = conversation_manager.get_conversations_by_contact(contact_data['id'])
        
        for convo in conversations:
            if "messages" in convo.get("conversation_data", {}):
                for msg in convo["conversation_data"]["messages"]:
                    if msg.get("platform_message_id") == message_id:
                        conversation_id = convo["id"]
                        break
                if conversation_id:
                    break
        
        if conversation_id:
            # Update the message in conversation record
            try:
                # Get conversation information
                conversation = conversation_manager.get_conversation(conversation_id)
                
                if conversation and "messages" in conversation.get("conversation_data", {}):
                    # Find the specific message
                    messages = conversation["conversation_data"]["messages"]
                    updated = False
                    
                    for i, msg in enumerate(messages):
                        if msg.get("platform_message_id") == message_id:
                            # Update the content in place
                            messages[i]["content"] = content
                            # Add update indicator but don't create a new message
                            messages[i]["updated_at"] = datetime.now(timezone.utc).isoformat()
                            updated = True
                            break
                            
                    if updated:
                        # Instead of trying to update with conversation_data parameter,
                        # create a new message record with the updated info
                        updated_message = {
                            'sender_type': messages[i].get('sender_type', 'unknown'),
                            'content': content,
                            'platform': 'chatwoot',
                            'platform_message_id': message_id,
                            'timestamp': datetime.now(timezone.utc).isoformat(),
                            'direction': messages[i].get('direction', 'outgoing'),
                            'status': 'updated',
                            'is_updated': True,
                            'original_id': message_id
                        }
                        
                        # Add the new message with updated content instead
                        # of trying to update the entire conversation
                        conversation_manager.add_message(conversation_id, updated_message)
                        
                    else:
                        print(f"⚠️ Message with ID {message_id} not found in conversation {conversation_id}")
                else:
                    print(f"⚠️ Could not find messages in conversation with ID {conversation_id}")
            except Exception as e:
                print(f"❌ Error updating message: {str(e)}")
                import traceback
                traceback.print_exc()
        else:
            print(f"⚠️ Could not find conversation with message ID {message_id}")
        
    except Exception as e:
        print(f"❌ Error in handle_message_updated: {str(e)}")
        import traceback
        traceback.print_exc()

async def handle_conversation_created(data: Dict[str, Any], organization_id: str = None):
    """
    Handle conversation created event
    """
    try:
        # Get conversation context
        context = handle_conversation_event(data)
        
        # Get or create contact
        contact_data = handle_contact_creation(data, organization_id)
        if not contact_data:
            return
            
        # Create a conversation
        conversation_data = handle_conversation_management(data, contact_data['id'])
        
        facebook_id = context.get('facebook_id')
        if facebook_id:
            print(
                f"New conversation started with Facebook ID: {facebook_id}",
                f"Contact ID: {contact_data['id']}",
                f"Conversation ID: {conversation_data['id']}" if conversation_data else ""
            )
        
    except Exception as e:
        print(f"Error handling conversation created: {str(e)}")
        raise

def create_chatwoot_webhook(inbox_id, webhook_url, subscriptions=None):
    """
    Create a webhook in Chatwoot for the given inbox_id and webhook_url.
    Checks for existing webhooks to avoid duplicates.
    
    Args:
        inbox_id (str): Chatwoot inbox ID
        webhook_url (str): Webhook URL with organization_id
        subscriptions (list): List of event subscriptions (default: conversation/message events)
        
    Returns: 
        Dict with webhook details or None if failed
    """
    if subscriptions is None:
        subscriptions = [
            "conversation_created",
            "message_created",
            "message_updated"
        ]

    # Validate inputs
    if not webhook_url or not webhook_url.startswith('http'):
        print(f"Error: Invalid webhook URL: {webhook_url}")
        return None
        
    if not CHATWOOT_API_TOKEN:
        print("Error: CHATWOOT_API_TOKEN is not configured")
        return None
        
    # Get or generate a webhook secret
    webhook_secret = os.getenv('WEBHOOK_SECRET')
    if not webhook_secret:
        import secrets
        webhook_secret = secrets.token_hex(16)
        print(f"Generated new webhook secret (consider saving this in your .env file)")

    headers = {
        "Content-Type": "application/json",
        "api_access_token": CHATWOOT_API_TOKEN
    }

    # Check for existing webhooks
    try:
        print(f"Checking for existing webhooks matching URL: {webhook_url}")
        response = requests.get(f"{CHATWOOT_BASE_URL}/api/v1/accounts/{CHATWOOT_ACCOUNT_ID}/integrations/hooks", headers=headers)
        response.raise_for_status()
        
        existing_webhooks = response.json()
        print(f"Found {len(existing_webhooks)} existing webhooks")
        
        for webhook in existing_webhooks:
            if webhook.get('url') == webhook_url:
                print(f"Webhook already exists for URL: {webhook_url} (ID: {webhook.get('id')})")
                return {"url": webhook_url, "id": webhook.get('id'), "status": "exists"}
                
    except requests.exceptions.HTTPError as e:
        print(f"Failed to check existing webhooks: {str(e)}")
        if hasattr(response, 'text'):
            print(f"Response: {response.text}")
        print("Proceeding with webhook creation anyway")
    except Exception as e:
        print(f"Error checking existing webhooks: {str(e)}")
        print("Proceeding with webhook creation anyway")

    # Create webhook
    payload = {
        "url": webhook_url,
        "subscriptions": subscriptions,
        "secret": webhook_secret
    }

    try:
        print(f"Creating webhook for URL: {webhook_url} with events: {subscriptions}")
        response = requests.post(
            f"{CHATWOOT_BASE_URL}/api/v1/accounts/{CHATWOOT_ACCOUNT_ID}/integrations/hooks",
            json=payload,
            headers=headers
        )
        response.raise_for_status()
        
        webhook_data = response.json()
        print(f"✅ Webhook created successfully - ID: {webhook_data.get('id')}")
        print(f"✅ Webhook will receive events: {', '.join(subscriptions)}")
        print(f"✅ Webhook secret: {webhook_secret}")
        return webhook_data

    except requests.exceptions.HTTPError as e:
        print(f"Failed to create webhook for inbox_id: {inbox_id}")
        print(f"Error: {str(e)}")
        print(f"Response: {response.text if hasattr(response, 'text') else 'No response text'}")
        return None
    except Exception as e:
        print(f"Unexpected error creating webhook for inbox_id {inbox_id}: {str(e)}")
        return None

def verify_webhook_signature(request):
    """
    Verify the webhook signature from Chatwoot.
    
    Args:
        request: The Flask request object
        
    Returns:
        bool: True if signature is valid or checking is disabled, False otherwise
    """
    # Get the secret from environment
    webhook_secret = os.getenv('WEBHOOK_SECRET')
    
    # If no secret is set, we can't verify (but we'll allow for development)
    if not webhook_secret:
        print("⚠️ Warning: No WEBHOOK_SECRET set, skipping signature verification")
        return True
        
    # Get the signature from headers
    signature = request.headers.get('X-Hub-Signature-256')
    if not signature:
        print("❌ No signature found in webhook request")
        return False
        
    # Compute the expected signature
    import hmac
    import hashlib
    
    # Get the raw request body
    payload = request.get_data()
    
    # Compute HMAC
    expected_signature = 'sha256=' + hmac.new(
        webhook_secret.encode('utf-8'),
        payload,
        hashlib.sha256
    ).hexdigest()
    
    # Secure comparison
    if not hmac.compare_digest(signature, expected_signature):
        print("❌ Invalid webhook signature")
        return False
        
    return True

async def handle_message_created(data: Dict[str, Any], organization_id: str = None):
    """
    Handle message created event within conversation context.
    Processes incoming messages and generates AI responses when appropriate.
    
    Args:
        data: The webhook event data
        organization_id: Optional organization ID for multi-tenant setups
    """
    start_time = time.time()
    try:
        # Extract message ID for duplicate detection
        message_id = data.get('id')
        
        # Skip if we've already processed this message
        if message_id and message_id in processed_messages:
            print(f"⏭️ Skipping duplicate message with ID: {message_id}")
            return
            
        # Add to processed messages
        if message_id:
            processed_messages.append(message_id)
        
        # Get conversation context
        context = handle_conversation_event(data)
        if not context or not context.get('conversation_id'):
            print("❌ Invalid conversation context, skipping message")
            return
            
        # Handle contact data
        contact_data = handle_contact_creation(data, organization_id)
        if not contact_data:
            print("❌ Could not create or retrieve contact, skipping message")
            return
        
        # Handle conversation data
        conversation_data = handle_conversation_management(data, contact_data['id'])
        if not conversation_data:
            print("❌ Could not create or retrieve conversation, skipping message")
            return
        
        # Get message details
        content = data.get('content', '')
        message_type = data.get('message_type')
        is_incoming = message_type == 'incoming'
        status = 'received' if is_incoming else 'sent'
        
        # Skip empty messages
        if not content or not content.strip():
            print("⏭️ Skipping empty message")
            return
        
        # Log initial message with clearer direction indication
        if is_incoming:
            print(f"\n📨 Incoming Message FROM {context['contact_name']}:")
        else:
            print(f"\n📤 Outgoing Message TO {context['contact_name']}:")
        
        if context.get('last_activity_at'):
            print(f"Time: {context['last_activity_at'].strftime('%Y-%m-%d %H:%M:%S UTC')}")
        
        # Add message to conversation - but only if it's incoming OR not from our own system
        # (outgoing messages from our AI system are already stored when we send them)
        sender = data.get('sender')
        sender_id = sender.get('id') if sender else None
        sender_type = sender.get('type') if sender else None
        
        # For outgoing messages, determine if this might be our own message by checking
        # if we've recently sent a similar message with this content
        is_likely_our_message = False
        if not is_incoming:
            try:
                conversation = conversation_manager.get_conversation(conversation_data['id'])
                if conversation and "messages" in conversation.get("conversation_data", {}):
                    messages = conversation["conversation_data"]["messages"]
                    
                    # Look for a recent message with same content (within last 30 seconds)
                    current_time = datetime.now(timezone.utc)
                    for msg in reversed(messages):  # Start from most recent
                        # Skip non-matching messages
                        if msg.get('content') != content or msg.get('direction') != 'outgoing':
                            continue
                            
                        # Check timestamp if available
                        if msg.get('timestamp'):
                            try:
                                msg_time = datetime.fromisoformat(msg['timestamp'].replace('Z', '+00:00'))
                                # If message is within last 30 seconds, it's likely a duplicate
                                time_diff = (current_time - msg_time).total_seconds()
                                if time_diff < 30:
                                    is_likely_our_message = True
                                    logger.info(f"Detected outgoing message that is likely our own (based on content match)")
                                    break
                            except (ValueError, TypeError):
                                # If timestamp parsing fails, continue checking
                                pass
            except Exception as e:
                logger.warning(f"Error while checking if message is our own: {str(e)}")
        
        # Only store incoming messages or outgoing messages that are not likely our own
        if is_incoming or (not is_likely_our_message and sender_type != 'bot'):
            message = {
                'sender_type': 'agent' if message_type == 'outgoing' else 'contact',
                'content': content,
                'platform': 'chatwoot',
                'platform_message_id': message_id,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'direction': 'incoming' if is_incoming else 'outgoing',
                'status': status
            }
            
            if is_incoming:
                message.update({
                    'source': context.get('channel', 'facebook'),
                    'raw_data': {
                        'facebook_id': context.get('facebook_id'),
                        'contact_name': context.get('contact_name')
                    }
                })
                
            logger.info(f"Adding message to conversation {conversation_data['id']}")
            conversation_manager.add_message(conversation_data['id'], message)
        else:
            logger.info(f"Skipping storage of our own outgoing message (already stored when sent)")
        
        # If this is an incoming message from a customer, check if we should generate an AI response
        if is_incoming and ENABLE_AI_RESPONSES:
            # Skip if conversation is in blocklist
            conv_id_str = str(context['conversation_id'])
            if conv_id_str in AI_RESPONSE_BLOCKLIST:
                print(f"⏭️ Skipping AI response for blocklisted conversation ID: {conv_id_str}")
                return
                
            # Skip if message starts with ignore prefix
            if AI_IGNORE_PREFIX and content.strip().startswith(AI_IGNORE_PREFIX):
                print(f"⏭️ Skipping AI response for message with ignore prefix: {AI_IGNORE_PREFIX}")
                return

            # Add to pending messages (without prefix - we'll add it during AI processing)
            pending_messages[conv_id_str].append({
                'content': content,
                'time': time.time(),
                'context': context,
                'conversation_data': conversation_data,
                'contact_data': contact_data,
                'organization_id': organization_id
            })
            
            # If no active task for this conversation, create one
            if conv_id_str not in active_tasks or active_tasks[conv_id_str].done():
                print(f"⏱️ Scheduling message processing task for conversation {conv_id_str}")
                active_tasks[conv_id_str] = asyncio.create_task(
                    process_messages_with_batching(conv_id_str)
                )
            else:
                print(f"⏱️ Task already scheduled for conversation {conv_id_str}, adding message to queue")
            
        elif is_incoming and not ENABLE_AI_RESPONSES:
            print("ℹ️ AI responses are disabled. Set ENABLE_AI_RESPONSES=true to enable.")
            
        print(f"✅ Message processing completed in {time.time() - start_time:.2f} seconds")
    
    except Exception as e:
        print(f"❌ Error in handle_message_created: {str(e)}")
        import traceback
        traceback.print_exc()

@router.post('/webhook/chatwoot')
async def chatwoot_webhook(
    request: Request,
    organization_id: Optional[str] = Query(None),
    x_organization_id: Optional[str] = Header(None, alias="X-Organization-Id")
):
    """
    Handle Chatwoot webhook events using FastAPI
    """
    try:
        # Get the body content
        body = await request.body()
        
        # Verify webhook signature (optional but recommended)
        if not await verify_webhook_signature_fastapi(request, body):
            logger.error("❌ Webhook signature verification failed")
            raise HTTPException(status_code=401, detail="Invalid signature")
            
        # Parse the JSON data
        data = await request.json()
        event_type = data.get('event')
        
        # Use organization_id from query param or header
        org_id = organization_id or x_organization_id
            
        # Log the webhook event
        logger.info(f"\n📩 Chatwoot Webhook - Event Type: {event_type} (Organization ID: {org_id or 'None'})")
        
        # Detailed logging to debug
        #log_raw_data(data, event_type)
        
        # Process events with appropriate async handling
        if event_type == 'message_created':
            await handle_message_created(data, org_id)
            return {"success": True, "message": "Message created event processed"}
            
        elif event_type == 'message_updated':
            await handle_message_updated(data, org_id)
            return {"success": True, "message": "Message updated event processed"}
            
        elif event_type == 'conversation_created':
            await handle_conversation_created(data, org_id)
            return {"success": True, "message": "Conversation created event processed"}
            
        else:
            logger.info(f"Unhandled event type: {event_type}")
            return {"success": True, "message": f"Event type {event_type} not processed"}
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"Error processing Chatwoot webhook: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# New function for FastAPI signature verification
async def verify_webhook_signature_fastapi(request: Request, body: bytes) -> bool:
    """
    Verify the webhook signature from Chatwoot for FastAPI.
    
    Args:
        request: The FastAPI request object
        body: Raw request body
        
    Returns:
        bool: True if signature is valid or checking is disabled, False otherwise
    """
    # Get the secret from environment
    webhook_secret = os.getenv('WEBHOOK_SECRET')
    
    # If no secret is set, we can't verify (but we'll allow for development)
    if not webhook_secret:
        logger.warning("⚠️ Warning: No WEBHOOK_SECRET set, skipping signature verification")
        return True
        
    # Get the signature from headers
    signature = request.headers.get('X-Hub-Signature-256')
    if not signature:
        logger.error("❌ No signature found in webhook request")
        return False
        
    # Compute the expected signature
    import hmac
    import hashlib
    
    # Compute HMAC
    expected_signature = 'sha256=' + hmac.new(
        webhook_secret.encode('utf-8'),
        body,
        hashlib.sha256
    ).hexdigest()
    
    # Secure comparison
    if not hmac.compare_digest(signature, expected_signature):
        logger.error("❌ Invalid webhook signature")
        return False
        
    return True

# Add a heartbeat function to monitor AI process health
def check_ai_process_health():
    """Check if the AI process is healthy and restart if necessary"""
    global ai_process, request_queue, response_queue, ai_process_running, process_lock
    
    with process_lock:
        if ai_process_running:
            try:
                # Send heartbeat request
                request_queue.put("HEARTBEAT", timeout=5)
                
                # Wait for response with timeout
                try:
                    response = response_queue.get(timeout=10)
                    if response == "ALIVE":
                        print("✅ AI process heartbeat check successful")
                        return True
                    else:
                        print(f"⚠️ AI process heartbeat returned unexpected response: {response}")
                except queue.Empty:
                    print("⚠️ AI process heartbeat timeout - no response received")
                
                # If we get here, process might be unhealthy - try to restart
                print("🔄 Restarting AI process due to failed heartbeat")
                stop_ai_process()
                time.sleep(1)
                start_ai_process()
                return False
                
            except Exception as e:
                print(f"⚠️ Error checking AI process health: {str(e)}")
                # Try to restart the process
                print("🔄 Restarting AI process due to error")
                stop_ai_process()
                time.sleep(1)
                start_ai_process()
                return False
        return False
