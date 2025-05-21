

# Then other imports
from flask import Flask, Response, request, jsonify, g, copy_current_request_context, current_app, Blueprint
import json
from uuid import UUID, uuid4
from datetime import datetime
import time

import asyncio
from typing import List, Optional, Dict, Any  # Added List and Optional imports
from collections import deque
import threading
import queue

# Import run_async_in_thread from the utils module
from async_utils import run_async_in_thread

# Keep track of recent webhook requests to detect duplicates
recent_requests = deque(maxlen=1000)

# Assuming these are in a module called 'data_fetch.py' - adjust as needed
from database import get_all_labels, get_raw_data_by_label, clean_text
#from docuhandler import process_document,summarize_document

from braindb import get_brains,get_brain_details,update_brain,create_brain,get_organization, create_organization, update_organization
from contactconvo import ConversationManager
from chatwoot import handle_message_created, handle_message_updated, handle_conversation_created
from braingraph import (
    create_brain_graph, get_brain_graph,
    add_brains_to_version, remove_brains_from_version, get_brain_graph_versions,
    update_brain_graph_version_status, BrainGraphVersion
)
from supabase import create_client, Client
import os
from utilities import logger
from enrich_profile import ProfileEnricher

# Add a dictionary to store locks for each thread_id
thread_locks = {}
thread_locks_lock = threading.RLock()
# Add a lock maintenance mechanism
thread_lock_last_used = {}
thread_lock_cleanup_interval = 300  # 5 minutes

# Load inbox mapping for Chatwoot
INBOX_MAPPING = {}
try:
    with open('inbox_mapping.json', 'r') as f:
        mapping_data = json.load(f)
        # Create a lookup dictionary by inbox_id
        for inbox in mapping_data.get('inboxes', []):
            INBOX_MAPPING[inbox['inbox_id']] = {
                'organization_id': inbox['organization_id'],
                'facebook_page_id': inbox['facebook_page_id'],
                'page_name': inbox['page_name']
            }
    logger.info(f"Loaded {len(INBOX_MAPPING)} inbox mappings")
except Exception as e:
    logger.error(f"Failed to load inbox_mapping.json: {e}")
    logger.warning("Chatwoot webhook will operate without inbox validation")

spb_url = os.getenv("SUPABASE_URL", "https://example.supabase.co")
spb_key = os.getenv("SUPABASE_KEY", "your-supabase-key")

# Add proper error handling for Supabase initialization
try:
    supabase: Client = create_client(spb_url, spb_key)
    logger.info("Supabase client initialized successfully in main.py")
except Exception as e:
    logger.error(f"Failed to initialize Supabase client in main.py: {e}")
    # Create a placeholder client for testing
# Create a Blueprint instead of using app directly
api_bp = Blueprint('api', __name__)

# Add numpy and json imports (needed elsewhere)
import numpy as np
import json

# Simple CORS configuration that was working before
CORS(api_bp, resources={r"/*": {"origins": "*"}})  # Enable CORS for all routes, all origins

cm = ContactManager()
convo_mgr = ConversationManager()
contact_analyzer = ContactAnalyzer()


def handle_options():
    """Common OPTIONS handler for all endpoints."""
    response = Response()
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    response.headers['Access-Control-Max-Age'] = '86400'
    return response, 200

def create_stream_response(gen):
    """
    Common response creator for streaming endpoints.
    Works with both regular generators and async generators.
    """
    # For async generators, we need a special handler that preserves Flask context
    if hasattr(gen, '__aiter__'):
        import asyncio
        
        # Get the current app and request context outside the wrapper
        app = current_app._get_current_object()
        
        # Define a wrapper that will consume the async generator while preserving Flask context
        @copy_current_request_context
        def async_generator_handler():
            # Create a new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # This function will run in the current thread and consume the async generator
            def run_async_generator():
                async def consume_async_generator():
                    try:
                        async for item in gen:
                            yield item
                    except Exception as e:
                        logger.error(f"Error consuming async generator: {str(e)}")
                        import traceback
                        logger.error(traceback.format_exc())
                
                # Create a list to store all generated items
                all_items = []
                
                # Run the async generator to completion and collect all items
                try:
                    coro = consume_async_generator().__aiter__().__anext__()
                    while True:
                        try:
                            item = loop.run_until_complete(coro)
                            all_items.append(item)
                            coro = consume_async_generator().__aiter__().__anext__()
                        except StopAsyncIteration:
                            break
                except Exception as e:
                    logger.error(f"Error in async generator execution: {str(e)}")
                    import traceback
                    logger.error(traceback.format_exc())
                finally:
                    loop.close()
                
                # Return all collected items
                return all_items
            
            # Collect all items first
            try:
                items = run_async_generator()
                
                # Now yield them one by one in the Flask context
                for item in items:
                    yield item
            except Exception as e:
                logger.error(f"Error in async generator handler: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
        
        # Use the context-preserving wrapper
        wrapped_gen = async_generator_handler()
    else:
        # Regular generator can be used directly
        wrapped_gen = gen
    
    # Use Flask's stream_with_context to ensure request context is maintained
    from flask import stream_with_context
    
    # Create the response
    response = Response(stream_with_context(wrapped_gen), mimetype='text/event-stream')
    response.headers['X-Accel-Buffering'] = 'no'
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

# Existing endpoints (pilot, training, havefun) remain unchanged
@api_bp.route('/pilot', methods=['POST', 'OPTIONS'])
def pilot():
    if request.method == 'OPTIONS':
        return handle_options()
    
    data = request.get_json() or {}
    user_input = data.get("user_input", "")
    user_id = data.get("user_id", "thefusionlab")
    thread_id = data.get("thread_id", "pilot_thread")
    
    logger.info("Pilot API called!")
    
    # Create a synchronous response streaming solution
    def generate_response():
        """Generate streaming response items synchronously."""
        try:
            # Import directly here to ensure fresh imports
            from ami import convo_stream
            
            # Define the async process function
            async def process_stream():
                """Process the stream and return all items."""
                outputs = []
                try:
                    # Get the stream
                    stream = convo_stream(
                        user_input=user_input, 
                        user_id=user_id, 
                        thread_id=thread_id, 
                        mode="pilot"
                    )
                    
                    # Process all the output
                    async for item in stream:
                        outputs.append(item)
                except Exception as e:
                    error_msg = f"Error processing stream: {str(e)}"
                    logger.error(error_msg)
                    import traceback
                    logger.error(traceback.format_exc())
                    outputs.append(f"data: {json.dumps({'error': error_msg})}\n\n")
                
                return outputs
            
            # Run the async function in a separate thread
            outputs = run_async_in_thread(process_stream)
            
            # Yield each output
            for item in outputs:
                yield item
                
        except Exception as outer_e:
            # Handle any errors in the outer function
            error_msg = f"Error in pilot response generator: {str(outer_e)}"
            logger.error(error_msg)
            import traceback
            logger.error(traceback.format_exc())
            yield f"data: {json.dumps({'error': error_msg})}\n\n"
    
    # Create a streaming response with our generator
    from flask import stream_with_context
    response = Response(stream_with_context(generate_response()), mimetype='text/event-stream')
    response.headers['X-Accel-Buffering'] = 'no'
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

@api_bp.route('/brains', methods=['GET', 'OPTIONS'])
def brains():
    if request.method == 'OPTIONS':
        return handle_options()
    
    org_id = request.args.get('org_id', '')
    if not org_id:
        return jsonify({"error": "org_id parameter is required"}), 400
    
    try:
        brains_list = get_brains(org_id)
        if not brains_list:
            return jsonify({"message": "No brains found for this organization", "brains": []}), 200
        
        # Convert Brain objects to dictionaries for JSON serialization
        brains_data = [{
            "id": brain.id,
            "brain_id": brain.brain_id,
            "org_id": brain.org_id,
            "name": brain.name,
            "status": brain.status,
            "bank_name": brain.bank_name,
            "summary": brain.summary,
            "created_date":brain.created_date
        } for brain in brains_list]
        
        return jsonify({"brains": brains_data}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api_bp.route('/brain-details', methods=['GET', 'OPTIONS'])
def brain_details():
    if request.method == 'OPTIONS':
        return handle_options()
    
    brain_id = request.args.get('brain_id', '')
    if not brain_id:
        return jsonify({"error": "brain_id parameter is required"}), 400
    
    try:
        brain = get_brain_details(brain_id)  # Pass the UUID string directly
        if not brain:
            return jsonify({"error": f"No brain found with id {brain_id}"}), 404
        
        brain_data = {
            "id": brain.id,
            "brain_id": brain.brain_id,
            "org_id": brain.org_id,
            "name": brain.name,
            "status": brain.status,
            "bank_name": brain.bank_name,
            "summary": brain.summary,
            "created_date":brain.created_date
        }
        return jsonify({"brain": brain_data}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# New endpoint: Get Organization Details
@api_bp.route('/get-org-detail/<orgid>', methods=['GET', 'OPTIONS'])
def get_org_detail(orgid):
    if request.method == 'OPTIONS':
        return handle_options()
    
    if not orgid:
        return jsonify({"error": "orgid is required"}), 400
    
    try:
        org = get_organization(orgid)
        if not org:
            return jsonify({"error": f"No organization found with id {orgid}"}), 404
        
        org_data = {
            "id": org.id,
            "org_id": org.org_id,
            "name": org.name,
            "description": org.description,
            "email": org.email,
            "phone": org.phone,
            "address": org.address,
            "created_date": org.created_date.isoformat()
        }
        return jsonify({"organization": org_data}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


import os
VERIFY_TOKEN = os.getenv("CALLBACK_V_TOKEN")

@api_bp.route("/webhook", methods=["GET"])
def verify_webhook():
    mode = request.args.get("hub.mode")
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")

    # When configuring in Facebook, you can add org_id as query parameter
    # to the callback URL. E.g., /webhook?org_id=123
    org_id = request.args.get("org_id")
    
    # Use the verification function
    if mode == "subscribe" and verify_webhook_token(token, org_id):
        print(f"✅ Webhook verified by Facebook for organization: {org_id or 'default'}")
        return challenge, 200
    else:
        print(f"❌ Webhook verification failed. Token: {token}, org_id: {org_id}")
        return "Forbidden", 403

@api_bp.route('/webhook/chatwoot', methods=['POST', 'OPTIONS'])
def chatwoot_webhook():
    """
    Handle Chatwoot webhook events and route based on inbox_id and organization_id
    """
    logger.info(f"→ RECEIVED WEBHOOK - Method: {request.method}, Path: {request.path}, Args: {request.args}")
    logger.info(f"→ HEADERS: {dict(request.headers)}")
    
    if request.method == 'OPTIONS':
        logger.info("→ Processing OPTIONS request")
        return handle_options()
        
    try:
        # Log raw request data first
        raw_data = request.get_data().decode('utf-8')
        logger.info(f"→ RAW REQUEST DATA: {raw_data[:1000]}")  # Log first 1000 chars to avoid huge logs
        
        data = request.get_json()
        if not data:
            logger.error("→ ERROR: No JSON data received in webhook")
            return jsonify({"status": "error", "message": "No JSON data received"}), 400
            
        logger.info(f"→ PARSED JSON: {json.dumps(data)[:1000]}...")  # Log first 1000 chars
        event = data.get('event')
        logger.info(f"→ EVENT TYPE: {event}")
        
        # Get organization_id from query parameters or headers
        organization_id = request.args.get('organization_id')
        logger.info(f"→ QUERY PARAM organization_id: {organization_id}")
        
        if not organization_id:
            organization_id = request.headers.get('X-Organization-Id')
            logger.info(f"→ HEADER X-Organization-Id: {organization_id}")
        
        # Extract inbox information
        inbox = data.get('inbox', {})
        inbox_id = str(inbox.get('id', ''))
        facebook_page_id = inbox.get('channel', {}).get('facebook_page_id', 'N/A')
        
        logger.info(f"→ INBOX DETAILS - ID: {inbox_id}, Facebook Page ID: {facebook_page_id}")
        
        # Create a unique ID for this webhook to detect duplicates
        request_id = f"{event}_{data.get('id', '')}_{inbox_id}_{datetime.now().timestamp()}"
        
        # Check for duplicates
        if request_id in recent_requests:
            logger.info(f"→ DUPLICATE detected! Ignoring: {request_id}")
            return jsonify({"status": "success", "message": "Duplicate request ignored"}), 200
        
        recent_requests.append(request_id)
        logger.info(f"→ Added to recent_requests: {request_id}")
        
        # Log basic information
        logger.info(
            f"→ WEBHOOK SUMMARY - Event: {event}, Inbox: {inbox_id}, "
            f"Page: {facebook_page_id}, Organization: {organization_id or 'Not provided'}"
        )
        
        # Validate against inbox mapping if available
        if INBOX_MAPPING and inbox_id:
            logger.info(f"→ CHECKING inbox mapping for inbox_id: {inbox_id}")
            inbox_config = INBOX_MAPPING.get(inbox_id)
            if inbox_config:
                logger.info(f"→ FOUND inbox config: {inbox_config}")
                expected_organization_id = inbox_config['organization_id']
                expected_facebook_page_id = inbox_config['facebook_page_id']
                
                # If organization_id was not provided, use the one from mapping
                if not organization_id:
                    organization_id = expected_organization_id
                    logger.info(f"→ USING organization_id {organization_id} from inbox mapping")
                
                # If provided, validate that it matches what's expected
                elif organization_id != expected_organization_id:
                    logger.warning(
                        f"→ ORGANIZATION MISMATCH: provided={organization_id}, "
                        f"expected={expected_organization_id} for inbox {inbox_id}"
                    )
                    return jsonify({
                        "status": "error",
                        "message": "Organization ID does not match inbox configuration"
                    }), 400
                
                # Validate Facebook page ID if available
                if facebook_page_id != 'N/A' and facebook_page_id != expected_facebook_page_id:
                    logger.warning(
                        f"→ FACEBOOK PAGE MISMATCH: actual={facebook_page_id}, "
                        f"expected={expected_facebook_page_id} for inbox {inbox_id}"
                    )
            else:
                logger.warning(f"→ NO MAPPING found for inbox_id: {inbox_id}")
                # Continue processing even without mapping, using provided organization_id
        
        # Handle different event types
        if event == "message_created":
            logger.info(f"→ PROCESSING message_created event for organization: {organization_id or 'None'}")
            handle_message_created(data, organization_id)
        elif event == "message_updated":
            logger.info(f"→ PROCESSING message_updated event for organization: {organization_id or 'None'}")
            handle_message_updated(data, organization_id)
        elif event == "conversation_created":
            logger.info(f"→ PROCESSING conversation_created event for organization: {organization_id or 'None'}")
            handle_conversation_created(data, organization_id)
        else:
            logger.info(f"→ UNHANDLED event type: {event}")
        
        logger.info("→ WEBHOOK PROCESSING SUCCESSFUL - Returning 200 OK")
        return jsonify({
            "status": "success", 
            "message": f"Processed {event} event", 
            "organization_id": organization_id,
            "inbox_id": inbox_id
        }), 200
    
    except Exception as e:
        logger.error(f"→ CRITICAL ERROR processing webhook: {str(e)}")
        import traceback
        trace = traceback.format_exc()
        logger.error(f"→ STACK TRACE: {trace}")
        return jsonify({"status": "error", "message": str(e)}), 500

    if request.method == 'OPTIONS':
        return handle_options()
    
    data = request.get_json() or {}
    org_id = data.get("org_id", "")
    name = data.get("name", "")
    description = data.get("description")
    
    if not org_id or not name:
        return jsonify({"error": "org_id and name are required"}), 400
    
    try:
        brain_graph = create_brain_graph(org_id, name, description)
        return jsonify({
            "message": "Brain graph created successfully",
            "brain_graph": {
                "id": brain_graph.id,
                "org_id": brain_graph.org_id,
                "name": brain_graph.name,
                "description": brain_graph.description,
                "created_date": brain_graph.created_date.isoformat()
            }
        }), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api_bp.route('/get-org-brain-graph', methods=['GET', 'OPTIONS'])
def get_org_brain_graph():
    if request.method == 'OPTIONS':
        return handle_options()
    
    org_id = request.args.get('org_id', '')
    if not org_id:
        return jsonify({"error": "org_id parameter is required"}), 400
    
    try:
        # Validate UUID format
        try:
            UUID(org_id)
        except ValueError:
            return jsonify({"error": "Invalid org_id format - must be a valid UUID"}), 400
            
        # Get the brain graph ID for this org
        response = supabase.table("brain_graph")\
            .select("id")\
            .eq("org_id", org_id)\
            .execute()
        
        if not response.data:
            return jsonify({"error": "No brain graph exists for this organization"}), 404
        
        graph_id = response.data[0]["id"]
        brain_graph = get_brain_graph(graph_id)
        
        if not brain_graph:
            return jsonify({"error": "Brain graph not found"}), 404
        
        # Get the latest version
        versions = get_brain_graph_versions(graph_id)
        latest_version = None
        if versions:
            latest_version = {
                "id": versions[0].id,
                "version_number": versions[0].version_number,
                "brain_ids": versions[0].brain_ids,
                "status": versions[0].status,
                "released_date": versions[0].released_date.isoformat()
            }
        
        return jsonify({
            "brain_graph": {
                "id": brain_graph.id,
                "org_id": brain_graph.org_id,
                "name": brain_graph.name,
                "description": brain_graph.description,
                "created_date": brain_graph.created_date.isoformat(),
                "latest_version": latest_version
            }
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route('/get-version-brains', methods=['GET', 'OPTIONS'])
def get_version_brains():
    if request.method == 'OPTIONS':
        return handle_options()
    
    version_id = request.args.get('version_id', '')
    if not version_id:
        return jsonify({"error": "version_id parameter is required"}), 400
    
    try:
        UUID(version_id)
    except ValueError:
        return jsonify({"error": "Invalid version_id format - must be a valid UUID"}), 400
    
    try:
        response = supabase.table("brain_graph_version")\
            .select("brain_ids")\
            .eq("id", version_id)\
            .execute()
        
        if not response.data:
            return jsonify({"error": f"No version found with id {version_id}"}), 404
        
        brain_ids = response.data[0]["brain_ids"]
        
        # Get brain details for each UUID
        brains = []
        if brain_ids:
            brain_response = supabase.table("brain")\
                .select("*")\
                .in_("id", brain_ids)\
                .execute()
            
            if brain_response.data:
                brains = brain_response.data
        
        return jsonify({
            "version_id": version_id,
            "brains": brains
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Function to initialize the blueprint with the app
def init_app(app):
    # Register app-specific configurations 
    app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'ami_secret_key')
    
    # Set up CORS for the app
    CORS(app, resources={r"/*": {"origins": "*"}})
    
    # Register the blueprint with the app
    app.register_blueprint(api_bp)
