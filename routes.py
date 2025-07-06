

# Then other imports
from flask import Flask, Response, request, jsonify, g, copy_current_request_context, current_app, Blueprint
import json
from uuid import UUID, uuid4
from datetime import datetime
import time

from flask_cors import CORS
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

from training_prep import process_document
from training_prep_new import  understand_document,save_document_insights
from braindb import get_brains,get_brain_details,update_brain,create_brain,get_organization, create_organization, update_organization
from aia import create_aia,get_all_aias,get_aia_detail,delete_aia,update_aia
from brainlog import get_brain_logs, get_brain_log_detail, BrainLog  # Assuming these are in brain_logs.py
from contact import ContactManager
from fbMessenger import get_sender_text, process_facebook_webhook,send_text_to_facebook_user, verify_webhook_token
from contactconvo import ConversationManager
from chatwoot import handle_message_created, handle_message_updated, handle_conversation_created
from contact_analysis import ContactAnalyzer
from braingraph import (
    create_brain_graph, get_brain_graph,
    add_brains_to_version, remove_brains_from_version, get_brain_graph_versions,
    update_brain_graph_version_status, BrainGraphVersion
)
from org_integrations import (
    create_integration, get_org_integrations, get_integration_by_id,
    update_integration, delete_integration, toggle_integration, OrganizationIntegration
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


@app.route('/process-document', methods=['POST', 'OPTIONS'])
def process_document_endpoint():
    if request.method == 'OPTIONS':
        return handle_options()

    if 'user_id' not in request.form or 'bank_name' not in request.form or 'reformatted_text' not in request.form:
        return jsonify({"error": "Missing user_id, bank_name, or reformatted_text"}), 400

    user_id = request.form['user_id']
    bank_name = request.form['bank_name']
    reformatted_text = request.form['reformatted_text']
    knowledge_elements = request.form['knowledge_elements'] # Get knowledge_elements from frontend
    mode = request.form.get('mode', 'default')  # Optional mode parameter

    # Debug logging
    logger.info(f"Process document request for bank_name={bank_name}")
    logger.info(f"Knowledge elements provided to MAIN: {len(knowledge_elements)} characters")
    if knowledge_elements and len(knowledge_elements) > 0:
        logger.info(f"First 100 chars of knowledge elements hit at MAIN: {knowledge_elements[:100]}...")
        logger.info(f"Knowledge elements contain 'KEY POINT': {'KEY POINT' in knowledge_elements}")
    logger.debug(f"First 100 chars of reformatted_text: {reformatted_text[:100]}...")
    if not reformatted_text.strip():
        return jsonify({"error": "Empty reformatted_text provided"}), 400

    # Use the thread-based approach instead of the global event loop
    try:
        # Run the async process_document function in a separate thread
        # We only pass the processed text and knowledge elements - no file to avoid reprocessing
        success = run_async_in_thread(
            process_document, 
            text=reformatted_text, 
            user_id=user_id, 
            mode=mode, 
            bank=bank_name,
            knowledge_elements=knowledge_elements  # Pass knowledge_elements to process_document
        )
        
        if success:
            return jsonify({
                "message": "Document processed successfully"
            }), 200
        else:
            return jsonify({
                "error": "Failed to process document"
            }), 500
    except Exception as e:
        logger.error(f"Error in process_document: {str(e)}")
        return jsonify({
            "error": f"Error processing document: {str(e)}"
        }), 500

@app.route('/understand-document', methods=['POST', 'OPTIONS'])
def understand_document_endpoint():
    if request.method == 'OPTIONS':
        return handle_options()

    if 'file' not in request.files:
        return jsonify({"success": False, "error": "Missing file"}), 400

    file = request.files['file']

    if not file.filename:
        return jsonify({"success": False, "error": "No file selected"}), 400
        
    # Determine file type from extension
    file_extension = file.filename.split('.')[-1].lower()
    if file_extension not in ['docx', 'pdf']:
        return jsonify({"success": False, "error": f"Unsupported file type: {file_extension}. Only DOCX and PDF files are supported."}), 400

    # Use the thread-based approach instead of the global event loop
    try:
        # Read file content into BytesIO
        file_content = file.read()
        if not file_content:
            logger.warning(f"Empty file content for {file.filename}")
            return jsonify({"success": False, "error": "Empty file content."}), 400
            
        # Create BytesIO object from file content
        from io import BytesIO
        file_bytes = BytesIO(file_content)
        
        # Log file info for debugging
        logger.info(f"Processing file '{file.filename}' ({len(file_content)} bytes) as {file_extension}")
        
        # Run the async understand_document function in a separate thread
        result = run_async_in_thread(
            understand_document, 
            input_source=file_bytes,  # Pass BytesIO object
            file_type=file_extension  # Pass detected file type
        )
        
        # Validate result structure
        if not isinstance(result, dict):
            logger.error(f"Invalid result type from understand_document: {type(result)}")
            return jsonify({
                "success": False,
                "error": "Document processing returned invalid data structure",
                "error_type": "ProcessingError"
            }), 500
        
        # Check for success
        if result.get("success", False):
            # Validate presence of document_insights
            if "document_insights" not in result:
                logger.warning("Missing document_insights in successful result")
                result["document_insights"] = {}
                
            # Log success statistics
            insights = result["document_insights"]
            logger.info(f"Document processed successfully: {insights.get('metadata', {}).get('sentence_count', 0)} sentences, "
                        f"{insights.get('metadata', {}).get('cluster_count', 0)} clusters")
            
            # Return the document insights with proper structure
            return jsonify(result), 200
        else:
            error_msg = result.get("error", "Failed to process document")
            error_type = result.get("error_type", "UnknownError")
            logger.error(f"Document processing failed: {error_type}: {error_msg}")
            return jsonify({
                "success": False,
                "error": error_msg,
                "error_type": error_type
            }), 500
    except Exception as e:
        logger.error(f"Error in understand_document: {type(e).__name__}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": f"Error processing document: {str(e)}",
            "error_type": type(e).__name__
        }), 500

@app.route('/save-document-insights', methods=['POST', 'OPTIONS'])
def save_document_insights_endpoint():
    if request.method == 'OPTIONS':
        return handle_options()

    # Check for data in JSON format or form data
    if request.is_json:
        # Process JSON request
        if 'document_insight' not in request.json:
            return jsonify({"success": False, "error": "Missing document_insight in request body"}), 400
            
        if 'user_id' not in request.json:
            return jsonify({"success": False, "error": "Missing user_id in request body"}), 400
            
        if 'bank_name' not in request.json:
            return jsonify({"success": False, "error": "Missing bank_name in request body"}), 400
            
        document_insight = request.json['document_insight']
        user_id = request.json['user_id']
        bank_name = request.json['bank_name']
        mode = request.json.get('mode', 'default')
    else:
        # Process form data
        if 'document_insight' not in request.form:
            return jsonify({"success": False, "error": "Missing document_insight in form data"}), 400
            
        if 'user_id' not in request.form:
            return jsonify({"success": False, "error": "Missing user_id in form data"}), 400
            
        if 'bank_name' not in request.form:
            return jsonify({"success": False, "error": "Missing bank_name in form data"}), 400
            
        document_insight = request.form['document_insight']
        user_id = request.form['user_id']
        bank_name = request.form['bank_name']
        mode = request.form.get('mode', 'default')

    # Ensure document_insight is a JSON string
    if isinstance(document_insight, dict):
        document_insight = json.dumps(document_insight)
    elif not isinstance(document_insight, str):
        return jsonify({
            "success": False, 
            "error": f"Invalid document_insight format. Expected JSON object or string, got {type(document_insight).__name__}"
        }), 400

    # Log request details
    logger.info(f"Saving document insights to bank '{bank_name}' for user '{user_id}'")
    
    # Use the thread-based approach to call the async function
    try:
        # Run the async save_document_insights function in a separate thread
        success = run_async_in_thread(
            save_document_insights, 
            document_insight=document_insight, 
            user_id=user_id, 
            mode=mode, 
            bank=bank_name
        )

        if success:
            logger.info(f"Document insights saved successfully to bank '{bank_name}'")
            return jsonify({
                "success": True,
                "message": "Document insights saved successfully"
            }), 200
        else:
            logger.error(f"Failed to save document insights to bank '{bank_name}'")
            return jsonify({
                "success": False, 
                "error": "Failed to save document insights"
            }), 500
    except Exception as e:
        logger.error(f"Error in save_document_insights endpoint: {type(e).__name__}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({
            "success": False, 
            "error": f"Error saving document insights: {str(e)}",
            "error_type": type(e).__name__
        }), 500



@api_bp.route('/labels', methods=['GET', 'OPTIONS'])
def get_labels():
    print(f"Received request: {request.method} {request.path}")
    if request.method == 'OPTIONS':
        return handle_options()
    
    bank_name = request.args.get('bank_name', '')
    if not bank_name:
        return jsonify({"error": "bank_name parameter is required"}), 400
    
    try:
        # Run the async get_all_labels function in a separate thread
        labels = run_async_in_thread(get_all_labels, lang="original", bank_name=bank_name)
        print(f"Labels from DB: {labels}")
        
        if not labels:
            return jsonify({"error": "No labels found"}), 404
        
        label_list = list(labels)
        return jsonify({"labels": label_list}), 200
    except Exception as e:
        logger.error(f"Error in get_labels: {str(e)}")
        return jsonify({"error": f"Error retrieving labels: {str(e)}"}), 500

@api_bp.route('/label-details', methods=['GET', 'OPTIONS'])
def get_label_details():
    if request.method == 'OPTIONS':
        return handle_options()
    
    label = request.args.get('label', '')
    if not label:
        return jsonify({"error": "Label parameter is required"}), 400
    bank_name = request.args.get('bank_name', '')
    if not bank_name:
        return jsonify({"error": "bank_name parameter is required"}), 400
    
    try:
        # Run the async get_raw_data_by_label function in a separate thread
        raw_data = run_async_in_thread(get_raw_data_by_label, label, lang="original", bank_name=bank_name)
        
        if not raw_data:
            return jsonify({"error": f"No data found for label '{label}'"}), 404
        
        cleaned_data = [clean_text(text) for text in raw_data]
        
        response_data = {
            "label": label,
            "total_entries": len(cleaned_data),
            "data": cleaned_data
        }
        return jsonify(response_data), 200
    except Exception as e:
        logger.error(f"Error in get_label_details: {str(e)}")
        return jsonify({"error": f"Error retrieving label details: {str(e)}"}), 500

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

@api_bp.route('/update-brain', methods=['POST', 'OPTIONS'])
def update_brain_endpoint():
    if request.method == 'OPTIONS':
        return handle_options()
    
    data = request.get_json() or {}
    id = data.get("id", "")
    new_name = data.get("name", "")
    new_status = data.get("status", "")
    
    if not id or not new_name or not new_status:
        return jsonify({"error": "id, name, and status are required"}), 400
    
    try:
        updated_brain = update_brain(id, new_name, new_status)
        brain_data = {
            "id": updated_brain.id,
            "brain_id": updated_brain.brain_id,
            "org_id": updated_brain.org_id,
            "name": updated_brain.name,
            "status": updated_brain.status,
            "bank_name": updated_brain.bank_name,
            "summary": updated_brain.summary
        }
        return jsonify({"message": "Brain updated successfully", "brain": brain_data}), 200
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api_bp.route('/create-brain', methods=['POST', 'OPTIONS'])
def create_brain_endpoint():
    if request.method == 'OPTIONS':
        return handle_options()
    
    data = request.get_json() or {}
    org_id = data.get("org_id", "")
    user_id = data.get("user_id", "")
    name = data.get("name", "")
    summary=data.get("summary","")
    
    if not org_id or not user_id or not name:
        return jsonify({"error": "org_id, user_id, and name are required"}), 400
    
    try:
        new_brain = create_brain(org_id, user_id, name,summary)
        brain_data = {
            "id": new_brain.id,
            "brain_id": new_brain.brain_id,
            "org_id": new_brain.org_id,
            "name": new_brain.name,
            "status": new_brain.status,
            "bank_name": new_brain.bank_name,
            "summary": new_brain.summary,
            "created_date": new_brain.created_date.isoformat()
        }
        return jsonify({"message": "Brain created successfully", "brain": brain_data}), 201
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
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

@api_bp.route('/aias', methods=['GET', 'OPTIONS'])
def aias():
    """
    Fetch all AIAs for a given organization.
    Query parameter: org_id (UUID)
    """
    if request.method == 'OPTIONS':
        return handle_options()
    
    org_id = request.args.get('org_id', '')
    if not org_id:
        return jsonify({"error": "org_id parameter is required"}), 400
    
    try:
        aias_list = get_all_aias(org_id)
        if not aias_list:
            return jsonify({"message": "No AIAs found for this organization", "aias": []}), 200
        
        # Convert AIA objects to dictionaries for JSON serialization
        aias_data = [{
            "id": aia.id,
            "aia_id": aia.aia_id,  # Adjust if no integer ID exists
            "org_id": aia.org_id,
            "task_type": aia.task_type,
            "name": aia.name,
            "brain_ids": aia.brain_ids,
            "delivery_method_ids": aia.delivery_method_ids,
            "created_date": aia.created_date.isoformat()
        } for aia in aias_list]
        
        return jsonify({"aias": aias_data}), 200
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api_bp.route('/aia-details', methods=['GET', 'OPTIONS'])
def aia_details():
    """
    Fetch details of a specific AIA by its UUID.
    Query parameter: aia_id (UUID)
    """
    if request.method == 'OPTIONS':
        return handle_options()
    
    aia_id = request.args.get('aia_id', '')
    if not aia_id:
        return jsonify({"error": "aia_id parameter is required"}), 400
    
    try:
        aia = get_aia_detail(aia_id)
        if not aia:
            return jsonify({"error": f"No AIA found with aia_id {aia_id}"}), 404
        
        aia_data = {
            "id": aia.id,
            "aia_id": aia.aia_id,  # Adjust if no integer ID exists
            "org_id": aia.org_id,
            "task_type": aia.task_type,
            "name": aia.name,
            "brain_ids": aia.brain_ids,
            "delivery_method_ids": aia.delivery_method_ids,
            "created_date": aia.created_date.isoformat()
        }
        return jsonify({"aia": aia_data}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api_bp.route('/create-aia', methods=['POST', 'OPTIONS'])
def create_aia_endpoint():
    """
    Create a new AIA.
    Body: {org_id, task_type, name (optional), brain_ids, delivery_method_ids}
    """
    if request.method == 'OPTIONS':
        return handle_options()
    
    data = request.get_json() or {}
    org_id = data.get("org_id", "")
    task_type = data.get("task_type", "")
    name = data.get("name")  # Optional
    brain_ids = data.get("brain_ids", [])  # Expecting a list
    delivery_method_ids = data.get("delivery_method_ids", [])  # Expecting a list
    
    if not org_id or not task_type or not isinstance(brain_ids, list) or not isinstance(delivery_method_ids, list):
        return jsonify({"error": "org_id, task_type, brain_ids (list), and delivery_method_ids (list) are required"}), 400
    
    try:
        new_aia = create_aia(org_id, task_type, name, brain_ids, delivery_method_ids)
        aia_data = {
            "id": new_aia.id,
            "aia_id": new_aia.aia_id,  # Adjust if no integer ID exists
            "org_id": new_aia.org_id,
            "task_type": new_aia.task_type,
            "name": new_aia.name,
            "brain_ids": new_aia.brain_ids,
            "delivery_method_ids": new_aia.delivery_method_ids,
            "created_date": new_aia.created_date.isoformat()
        }
        return jsonify({"message": "AIA created successfully", "aia": aia_data}), 201
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api_bp.route('/update-aia', methods=['POST', 'OPTIONS'])
def update_aia_endpoint():
    """
    Update an existing AIA.
    Body: {id, task_type (optional), name (optional), brain_ids (optional), delivery_method_ids (optional)}
    """
    if request.method == 'OPTIONS':
        return handle_options()
    
    data = request.get_json() or {}
    id = data.get("id", "")
    task_type = data.get("task_type")  # Optional
    name = data.get("name")  # Optional
    brain_ids = data.get("brain_ids")  # Optional list
    delivery_method_ids = data.get("delivery_method_ids")  # Optional list
    
    if not id:
        return jsonify({"error": "id is required"}), 400
    
    try:
        updated_aia = update_aia(id, task_type, name, brain_ids, delivery_method_ids)
        aia_data = {
            "id": updated_aia.id,
            "aia_id": updated_aia.aia_id,  # Adjust if no integer ID exists
            "org_id": updated_aia.org_id,
            "task_type": updated_aia.task_type,
            "name": updated_aia.name,
            "brain_ids": updated_aia.brain_ids,
            "delivery_method_ids": updated_aia.delivery_method_ids,
            "created_date": updated_aia.created_date.isoformat()
        }
        return jsonify({"message": "AIA updated successfully", "aia": aia_data}), 200
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api_bp.route('/delete-aia', methods=['POST', 'OPTIONS'])
def delete_aia_endpoint():
    """
    Delete an AIA by its UUID.
    Body: {id}
    """
    if request.method == 'OPTIONS':
        return handle_options()
    
    data = request.get_json() or {}
    id = data.get("id", "")
    
    if not id:
        return jsonify({"error": "id is required"}), 400
    
    try:
        delete_aia(id)
        return jsonify({"message": f"AIA with id {id} deleted successfully"}), 200
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api_bp.route('/brainlogs', methods=['GET', 'OPTIONS'])
def brainlogs():
    """
    Fetch all brain logs for a given brain.
    Query parameter: brain_id (UUID)
    """
    if request.method == 'OPTIONS':
        return handle_options()
    
    brain_id = request.args.get('brain_id', '')
    if not brain_id:
        return jsonify({"error": "brain_id parameter is required"}), 400
    
    try:
        logs_list: List[BrainLog] = get_brain_logs(brain_id)
        if not logs_list:
            return jsonify({"message": "No logs found for this brain", "logs": []}), 200
        
        # Convert BrainLog objects to dictionaries for JSON serialization
        logs_data = [{
            "entry_id": log.entry_id,
            "brain_id": log.brain_id,
            "entry_values": log.entry_values,
            "gap_analysis": log.gap_analysis,
            "created_date": log.created_date.isoformat()
        } for log in logs_list]
        
        return jsonify({"logs": logs_data}), 200
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api_bp.route('/brain-log-detail', methods=['GET', 'OPTIONS'])
def brain_log_detail():
    """
    Fetch details of a specific brain log by its entry_id.
    Query parameter: entry_id (integer)
    """
    if request.method == 'OPTIONS':
        return handle_options()
    
    entry_id = request.args.get('entry_id', '')
    if not entry_id:
        return jsonify({"error": "entry_id parameter is required"}), 400
    
    try:
        # Convert entry_id to int, since it's an integer in the database
        entry_id_int = int(entry_id)
        log: Optional[BrainLog] = get_brain_log_detail(entry_id_int)
        if not log:
            return jsonify({"error": f"No brain log found with entry_id {entry_id}"}), 404
        
        log_data = {
            "entry_id": log.entry_id,
            "brain_id": log.brain_id,
            "entry_values": log.entry_values,
            "gap_analysis": log.gap_analysis,
            "created_date": log.created_date.isoformat()
        }
        return jsonify({"log": log_data}), 200
    except ValueError as ve:
        return jsonify({"error": "entry_id must be a valid integer"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api_bp.route('/contact-details', methods=['GET', 'OPTIONS'])
def contact_details():
    if request.method == 'OPTIONS':
        return handle_options()
    
    contact_id = request.args.get('contact_id', '')
    organization_id = request.args.get('organization_id', None)
    
    if not contact_id:
        return jsonify({"error": "contact_id parameter is required"}), 400
    
    try:
        contact = cm.get_contact_details(int(contact_id), organization_id)  # Pass organization_id
        if not contact:
            return jsonify({"error": f"No contact found with contact_id {contact_id}"}), 404
        
        contact_data = {
            "id": contact["id"],
            "uuid": contact["uuid"],
            "organization_id": contact.get("organization_id"),
            "type": contact["type"],
            "first_name": contact["first_name"],
            "last_name": contact["last_name"],
            "email": contact["email"],
            "phone": contact["phone"],
            "facebook_id": contact.get("facebook_id"),
            "profile_picture_url": contact.get("profile_picture_url"),
            "created_at": contact["created_at"],
            "profile": contact.get("profiles", None)  # Include profile if exists
        }
        return jsonify({"contact": contact_data}), 200
    except ValueError:
        return jsonify({"error": "contact_id must be an integer"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api_bp.route('/update-contact', methods=['POST', 'OPTIONS'])
def update_contact_endpoint():
    if request.method == 'OPTIONS':
        return handle_options()
    
    data = request.get_json() or {}
    contact_id = data.get("id", "")
    organization_id = data.get("organization_id", None)
    type = data.get("type", "")
    first_name = data.get("first_name", "")
    last_name = data.get("last_name", "")
    email = data.get("email", None)
    phone = data.get("phone", None)
    facebook_id = data.get("facebook_id", None)
    profile_picture_url = data.get("profile_picture_url", None)
    
    if not contact_id:
        return jsonify({"error": "id is required"}), 400
    
    try:
        update_data = {}
        if type:
            update_data["type"] = type
        if first_name:
            update_data["first_name"] = first_name
        if last_name:
            update_data["last_name"] = last_name
        if email is not None:
            update_data["email"] = email
        if phone is not None:
            update_data["phone"] = phone
        if facebook_id is not None:
            update_data["facebook_id"] = facebook_id
        if profile_picture_url is not None:
            update_data["profile_picture_url"] = profile_picture_url
        
        updated_contact = cm.update_contact(int(contact_id), organization_id, **update_data)
        if not updated_contact:
            return jsonify({"error": f"No contact found with id {contact_id}"}), 404
        
        contact_data = {
            "id": updated_contact["id"],
            "uuid": updated_contact["uuid"],
            "organization_id": updated_contact.get("organization_id"),
            "type": updated_contact["type"],
            "first_name": updated_contact["first_name"],
            "last_name": updated_contact["last_name"],
            "email": updated_contact["email"],
            "phone": updated_contact["phone"],
            "facebook_id": updated_contact.get("facebook_id"),
            "profile_picture_url": updated_contact.get("profile_picture_url"),
            "created_at": updated_contact["created_at"]
        }
        return jsonify({"message": "Contact updated successfully", "contact": contact_data}), 200
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api_bp.route('/create-contact', methods=['POST', 'OPTIONS'])
def create_contact_endpoint():
    if request.method == 'OPTIONS':
        return handle_options()
    
    data = request.get_json() or {}
    organization_id = data.get("organization_id", "")
    type = data.get("type", "")
    first_name = data.get("first_name", "")
    last_name = data.get("last_name", "")
    email = data.get("email", None)
    phone = data.get("phone", None)
    facebook_id = data.get("facebook_id", None)
    profile_picture_url = data.get("profile_picture_url", None)
    
    if not organization_id:
        return jsonify({"error": "organization_id is required"}), 400
    
    if not type or not first_name or not last_name:
        return jsonify({"error": "type, first_name, and last_name are required"}), 400
    
    try:
        new_contact = cm.create_contact(
            organization_id,
            type, 
            first_name, 
            last_name, 
            email, 
            phone, 
            facebook_id, 
            profile_picture_url
        )
        contact_data = {
            "id": new_contact["id"],
            "uuid": new_contact["uuid"],
            "organization_id": new_contact.get("organization_id"),
            "type": new_contact["type"],
            "first_name": new_contact["first_name"],
            "last_name": new_contact["last_name"],
            "email": new_contact["email"],
            "phone": new_contact["phone"],
            "facebook_id": new_contact.get("facebook_id"),
            "profile_picture_url": new_contact.get("profile_picture_url"),
            "created_at": new_contact["created_at"]
        }
        return jsonify({"message": "Contact created successfully", "contact": contact_data}), 201
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api_bp.route('/profile-details', methods=['GET', 'OPTIONS'])
def profile_details():
    if request.method == 'OPTIONS':
        return handle_options()
    
    contact_id = request.args.get('contact_id', '')
    organization_id = request.args.get('organization_id', None)
    
    if not contact_id:
        return jsonify({"error": "contact_id parameter is required"}), 400
    
    try:
        contact = cm.get_contact_details(int(contact_id), organization_id)
        if not contact or not contact.get("profiles"):
            return jsonify({"error": f"No profile found for contact_id {contact_id}"}), 404
        
        profile = contact["profiles"]
        profile_data = {
            "id": profile["id"],
            "uuid": profile["uuid"],
            "contact_id": profile["contact_id"],
            "profile_summary": profile["profile_summary"],
            "general_info": profile["general_info"],
            "personality": profile["personality"],
            "hidden_desires": profile["hidden_desires"],
            "linkedin_url": profile["linkedin_url"],
            "social_media_urls": profile["social_media_urls"],
            "best_goals": profile["best_goals"],
            "updated_at": profile["updated_at"]
        }
        return jsonify({"profile": profile_data}), 200
    except ValueError:
        return jsonify({"error": "contact_id must be an integer"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api_bp.route('/update-profile', methods=['POST', 'OPTIONS'])
def update_profile_endpoint():
    if request.method == 'OPTIONS':
        return handle_options()
    
    data = request.get_json() or {}
    contact_id = data.get("contact_id", "")
    organization_id = data.get("organization_id", None)
    profile_summary = data.get("profile_summary", None)
    general_info = data.get("general_info", None)
    personality = data.get("personality", None)
    hidden_desires = data.get("hidden_desires", None)
    linkedin_url = data.get("linkedin_url", None)
    social_media_urls = data.get("social_media_urls", None)
    best_goals = data.get("best_goals", None)
    
    if not contact_id:
        return jsonify({"error": "contact_id is required"}), 400
    
    try:
        # First verify the contact exists and belongs to the organization
        if organization_id:
            contact = cm.get_contact_details(int(contact_id), organization_id)
            if not contact:
                return jsonify({"error": f"No contact found with id {contact_id} in this organization"}), 404
                
        update_data = {}
        if profile_summary is not None:
            update_data["profile_summary"] = profile_summary
        if general_info is not None:
            update_data["general_info"] = general_info
        if personality is not None:
            update_data["personality"] = personality
        if hidden_desires is not None:
            update_data["hidden_desires"] = hidden_desires
        if linkedin_url is not None:
            update_data["linkedin_url"] = linkedin_url
        if social_media_urls is not None:
            update_data["social_media_urls"] = social_media_urls
        if best_goals is not None:
            update_data["best_goals"] = best_goals
        
        updated_profile = cm.update_contact_profile(int(contact_id), **update_data)
        if not updated_profile:
            return jsonify({"error": f"No profile found for contact_id {contact_id}"}), 404
        
        profile_data = {
            "id": updated_profile["id"],
            "uuid": updated_profile["uuid"],
            "contact_id": updated_profile["contact_id"],
            "profile_summary": updated_profile["profile_summary"],
            "general_info": updated_profile["general_info"],
            "personality": updated_profile["personality"],
            "hidden_desires": updated_profile["hidden_desires"],
            "linkedin_url": updated_profile["linkedin_url"],
            "social_media_urls": updated_profile["social_media_urls"],
            "best_goals": updated_profile["best_goals"],
            "updated_at": updated_profile["updated_at"]
        }
        
        # After successful profile update, trigger sales signal analysis
        try:
            # Run in a separate thread to not block the response
            def run_analysis():
                analysis_result = contact_analyzer.analyze_and_store(
                    int(contact_id),
                    organization_id or contact.get("organization_id", ""),
                    updated_profile
                )
                logger.info(f"Updated sales analysis for contact {contact_id} with score {analysis_result['analysis'].get('sales_readiness_score', 0)}")
            
            # Start analysis in background thread
            import threading
            thread = threading.Thread(target=run_analysis)
            thread.daemon = True
            thread.start()
        except Exception as analysis_error:
            # Log the error but don't fail the profile update
            logger.error(f"Error updating sales analysis: {str(analysis_error)}")
        
        return jsonify({"message": "Profile updated successfully", "profile": profile_data}), 200
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api_bp.route('/create-profile', methods=['POST', 'OPTIONS'])
def create_profile_endpoint():
    if request.method == 'OPTIONS':
        return handle_options()
    
    data = request.get_json() or {}
    contact_id = data.get("contact_id", "")
    organization_id = data.get("organization_id", None)
    profile_summary = data.get("profile_summary", None)
    general_info = data.get("general_info", None)
    personality = data.get("personality", None)
    hidden_desires = data.get("hidden_desires", None)
    linkedin_url = data.get("linkedin_url", None)
    social_media_urls = data.get("social_media_urls", None)
    best_goals = data.get("best_goals", None)
    
    if not contact_id:
        return jsonify({"error": "contact_id is required"}), 400
    
    try:
        # First verify the contact exists and belongs to the organization
        if organization_id:
            contact = cm.get_contact_details(int(contact_id), organization_id)
            if not contact:
                return jsonify({"error": f"No contact found with id {contact_id} in this organization"}), 404
            
            # Check if profile already exists
            if contact.get("profiles"):
                # Profile already exists, return 409 Conflict with the existing profile
                profile = contact["profiles"]
                profile_data = {
                    "id": profile["id"],
                    "uuid": profile["uuid"],
                    "contact_id": profile["contact_id"],
                    "profile_summary": profile["profile_summary"],
                    "general_info": profile["general_info"],
                    "personality": profile["personality"],
                    "hidden_desires": profile["hidden_desires"],
                    "linkedin_url": profile["linkedin_url"],
                    "social_media_urls": profile["social_media_urls"],
                    "best_goals": profile["best_goals"],
                    "updated_at": profile["updated_at"]
                }
                return jsonify({"message": "Profile already exists", "profile": profile_data}), 409
                
        new_profile = cm.create_contact_profile(
            int(contact_id), profile_summary, general_info, personality,
            hidden_desires, linkedin_url, social_media_urls, best_goals
        )
        profile_data = {
            "id": new_profile["id"],
            "uuid": new_profile["uuid"],
            "contact_id": new_profile["contact_id"],
            "profile_summary": new_profile["profile_summary"],
            "general_info": new_profile["general_info"],
            "personality": new_profile["personality"],
            "hidden_desires": new_profile["hidden_desires"],
            "linkedin_url": new_profile["linkedin_url"],
            "social_media_urls": new_profile["social_media_urls"],
            "best_goals": new_profile["best_goals"],
            "updated_at": new_profile["updated_at"]
        }
        
        # After successful profile creation, trigger sales signal analysis
        try:
            # Run in a separate thread to not block the response
            def run_analysis():
                analysis_result = contact_analyzer.analyze_and_store(
                    int(contact_id),
                    organization_id or "",
                    new_profile
                )
                logger.info(f"Created initial sales analysis for contact {contact_id} with score {analysis_result['analysis'].get('sales_readiness_score', 0)}")
            
            # Start analysis in background thread
            import threading
            thread = threading.Thread(target=run_analysis)
            thread.daemon = True
            thread.start()
        except Exception as analysis_error:
            # Log the error but don't fail the profile creation
            logger.error(f"Error creating initial sales analysis: {str(analysis_error)}")
        
        return jsonify({"message": "Profile created successfully", "profile": profile_data}), 201
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        # Check for Supabase unique constraint violation
        if "duplicate key value violates unique constraint" in str(e) and "profiles_contact_id_key" in str(e):
            # Try to get the existing profile
            try:
                contact = cm.get_contact_details(int(contact_id), organization_id)
                if contact and contact.get("profiles"):
                    profile = contact["profiles"]
                    profile_data = {
                        "id": profile["id"],
                        "uuid": profile["uuid"],
                        "contact_id": profile["contact_id"],
                        "profile_summary": profile["profile_summary"],
                        "general_info": profile["general_info"],
                        "personality": profile["personality"],
                        "hidden_desires": profile["hidden_desires"],
                        "linkedin_url": profile["linkedin_url"],
                        "social_media_urls": profile["social_media_urls"],
                        "best_goals": profile["best_goals"],
                        "updated_at": profile["updated_at"]
                    }
                    return jsonify({"message": "Profile already exists", "profile": profile_data}), 409
            except Exception:
                pass
                
            return jsonify({"error": "A profile already exists for this contact"}), 409
        return jsonify({"error": str(e)}), 500

@api_bp.route('/contacts', methods=['GET', 'OPTIONS'])
def get_all_contacts():
    if request.method == 'OPTIONS':
        return handle_options()
    
    organization_id = request.args.get('organization_id', None)
    
    try:
        contacts = cm.get_contacts(organization_id)
        if not contacts:
            return jsonify({"message": "No contacts found", "contacts": []}), 200
        
        return jsonify({"contacts": contacts}), 200
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


@api_bp.route("/webhook", methods=["POST"])
def handle_message():
    data = request.json
    
    try:
        # First check if org_id is in query parameters (preferred approach)
        org_id = request.args.get("org_id")
        
        # If not in query params, try to determine it from the page ID in the webhook payload
        if not org_id:
            # Get the page/recipient ID from the webhook data
            page_id = None
            if data and "entry" in data and len(data["entry"]) > 0:
                if "messaging" in data["entry"][0] and len(data["entry"][0]["messaging"]) > 0:
                    # For message events
                    page_id = data["entry"][0]["messaging"][0].get("recipient", {}).get("id")
                elif "id" in data["entry"][0]:
                    # Some events provide page ID directly
                    page_id = data["entry"][0]["id"]
            
            if page_id:
                print(f"Looking up organization for page ID: {page_id}")
                # Look up the organization by page ID in integrations
                try:
                    # Query our integrations to find the org that owns this page
                    response = supabase.table("organization_integrations")\
                        .select("org_id, config")\
                        .eq("integration_type", "facebook")\
                        .eq("is_active", True)\
                        .execute()
                    
                    if response.data:
                        for integration in response.data:
                            # Parse config if it's a string
                            config = integration.get("config", {})
                            if isinstance(config, str):
                                try:
                                    config = json.loads(config)
                                except:
                                    config = {}
                            
                            # Check if this integration's page ID matches
                            if config.get("page_id") == page_id:
                                org_id = integration.get("org_id")
                                print(f"Found matching organization {org_id} for page {page_id}")
                                break
                    
                    if not org_id:
                        print(f"No organization found for page ID {page_id}")
                except Exception as lookup_err:
                    print(f"Error looking up organization for page ID {page_id}: {str(lookup_err)}")
            else:
                print("Could not extract page ID from webhook data")
            
        # Log the organization context
        if org_id:
            print(f"Processing Facebook webhook for organization: {org_id}")
        else:
            print("Processing Facebook webhook without organization context - using default credentials")
            
        # Check if this is a "message echo" event (messages sent by our page)
        entry = data.get("entry", [{}])[0]
        messaging = entry.get("messaging", [{}])[0]
        is_echo = messaging.get("message", {}).get("is_echo", False)
        
        if is_echo:
            print(f"Received message echo event (message sent by our page)")
        else:
            print(f"Received message from user")
            
        # Always process the webhook to track messages in our database
        success = process_facebook_webhook(data, convo_mgr, org_id)
        
        if not success:
            print("Warning: Failed to process Facebook webhook")
        
        # For non-echo messages (from actual users), you might want to generate a response
        if not is_echo:
            try:
                message_data = get_sender_text(data)
                sender_id = message_data["senderID"]
                message_text = message_data["messageText"]
                
                if message_text != "NO-DATA":
                    # Send responses ONLY to the specific test user ID
                    if sender_id == "29495554333369135":
                        response_text = "Thank you for your message!"
                        send_text_to_facebook_user(sender_id, response_text, convo_mgr, org_id=org_id)
            except Exception as e:
                print(f"Error processing user message for response: {str(e)}")
    except Exception as e:
        print(f"Error in webhook handler: {str(e)}")
    
    # Always return 200 to Facebook
    return "", 200

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

@api_bp.route('/activate-brain', methods=['POST', 'OPTIONS'])
def activate_brain():
    """
    Activate the brain with a specific graph version ID.
    This endpoint will load vectors for the specified graph version.
    """
    if request.method == 'OPTIONS':
        return handle_options()
    
    start_time = datetime.now()
    logger.info(f"[SESSION_TRACE] === BEGIN ACTIVATE BRAIN request at {start_time.isoformat()} ===")
    
    data = request.get_json() or {}
    graph_version_id = data.get("graph_version_id", "")
    
    if not graph_version_id:
        return jsonify({"error": "graph_version_id is required"}), 400
    
    try:
        # Call the centralized activate_brain_with_version function
        result = run_async_in_thread(activate_brain_with_version, graph_version_id)
        
        # Log completion
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()
        logger.info(f"[SESSION_TRACE] === END ACTIVATE BRAIN request - total time: {elapsed:.2f}s ===")
        
        if result["success"]:
            return jsonify({
                "message": "Brain activated successfully", 
                "graph_version_id": result["graph_version_id"],
                "loaded": result["loaded"],
                "elapsed_seconds": elapsed,
                "worker_id": "",
                "vector_count": result["vector_count"]
            }), 200
        else:
            return jsonify({"error": result["error"]}), 500
            
    except Exception as e:
        # Handle any errors
        error_msg = f"Error in activate_brain: {str(e)}"
        logger.error(error_msg)
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({"error": error_msg}), 500


@api_bp.route('/')
def home():
    return "Hello, It's me Ami!"

@api_bp.route('/ping', methods=['POST'])
def ping():
    return "Pong"



@api_bp.route('/contact-conversations', methods=['GET', 'OPTIONS'])
def get_contact_conversations():
    if request.method == 'OPTIONS':
        return handle_options()
    
    try:
        contact_id = request.args.get('contact_id')
        if not contact_id:
            return jsonify({"error": "contact_id parameter is required"}), 400
        
        # Add detailed logging
        print(f"Fetching conversations for contact_id: {contact_id}")
        
        contact_id = int(contact_id)
        
        # First check if contact exists
        contact = cm.get_contact_details(contact_id)
        if not contact:
            return jsonify({"error": f"No contact found with ID {contact_id}"}), 404
            
        # Get conversations
        conversations = []
        try:
            # Get recent conversations with pagination if specified
            if 'limit' in request.args:
                limit = int(request.args.get('limit', 10))
                offset = int(request.args.get('offset', 0))
                print(f"Using pagination with limit={limit}, offset={offset}")
                conversations = convo_mgr.get_recent_conversations(contact_id, limit, offset)
            else:
                conversations = convo_mgr.get_conversations_by_contact(contact_id)
        except Exception as convo_err:
            print(f"Error retrieving conversations: {str(convo_err)}")
            # Return empty list instead of error
            conversations = []
        
        print(f"Found {len(conversations)} conversations")
        
        # Ensure conversation_data is properly serialized JSON
        for convo in conversations:
            # Handle potential string JSON representation in conversation_data
            if isinstance(convo.get("conversation_data"), str):
                try:
                    convo["conversation_data"] = json.loads(convo["conversation_data"])
                except Exception as e:
                    print(f"Error parsing conversation_data as JSON: {str(e)}")
                    # Provide a default empty structure
                    convo["conversation_data"] = {"messages": []}
            
            # Ensure we have a messages array
            if isinstance(convo.get("conversation_data"), dict) and "messages" not in convo["conversation_data"]:
                convo["conversation_data"]["messages"] = []
        
        return jsonify({"conversations": conversations}), 200
    except ValueError as ve:
        print(f"Value error in contact-conversations: {str(ve)}")
        return jsonify({"error": "Invalid contact_id format"}), 400
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Server error in contact-conversations: {str(e)}")
        return jsonify({"error": str(e)}), 500

@api_bp.route('/conversation', methods=['GET', 'OPTIONS'])
def get_conversation_details():
    if request.method == 'OPTIONS':
        return handle_options()
    
    conversation_id = request.args.get('conversation_id')
    if not conversation_id:
        return jsonify({"error": "conversation_id parameter is required"}), 400
    
    try:
        conversation_id = int(conversation_id)
        conversation = convo_mgr.get_conversation(conversation_id)
        
        if not conversation:
            return jsonify({"error": f"No conversation found with ID {conversation_id}"}), 404
        
        # If read=true parameter is passed, mark conversation as read
        if request.args.get('read', '').lower() == 'true':
            convo_mgr.mark_conversation_as_read(conversation_id)
            # Get updated conversation after marking as read
            conversation = convo_mgr.get_conversation(conversation_id)
        
        return jsonify({"conversation": conversation}), 200
    except ValueError:
        return jsonify({"error": "Invalid conversation_id format"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api_bp.route('/create-organization', methods=['POST', 'OPTIONS'])
def create_organization_endpoint():
    """
    Create a new organization with optional contact information.
    """
    if request.method == 'OPTIONS':
        return handle_options()
    
    data = request.get_json() or {}
    name = data.get("name", "")
    description = data.get("description")
    email = data.get("email")
    phone = data.get("phone")
    address = data.get("address")
    
    if not name:
        return jsonify({"error": "name is required"}), 400
    
    try:
        new_org = create_organization(
            name=name,
            description=description,
            email=email,
            phone=phone,
            address=address
        )
        
        org_data = {
            "id": new_org.id,
            "org_id": new_org.org_id,
            "name": new_org.name,
            "description": new_org.description,
            "email": new_org.email,
            "phone": new_org.phone,
            "address": new_org.address,
            "created_date": new_org.created_date.isoformat()
        }
        
        return jsonify({
            "message": "Organization created successfully",
            "organization": org_data
        }), 201
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api_bp.route('/update-organization', methods=['POST', 'OPTIONS'])
def update_organization_endpoint():
    """
    Update an existing organization's information.
    """
    if request.method == 'OPTIONS':
        return handle_options()
    
    data = request.get_json() or {}
    org_id = data.get("id", "")
    name = data.get("name", "")
    description = data.get("description")
    email = data.get("email")
    phone = data.get("phone")
    address = data.get("address")
    
    if not org_id or not name:
        return jsonify({"error": "id and name are required"}), 400
    
    try:
        updated_org = update_organization(
            id=org_id,
            name=name,
            description=description,
            email=email,
            phone=phone,
            address=address
        )
        
        org_data = {
            "id": updated_org.id,
            "org_id": updated_org.org_id,
            "name": updated_org.name,
            "description": updated_org.description,
            "email": updated_org.email,
            "phone": updated_org.phone,
            "address": updated_org.address,
            "created_date": updated_org.created_date.isoformat()
        }
        
        return jsonify({
            "message": "Organization updated successfully",
            "organization": org_data
        }), 200
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api_bp.route('/create-brain-graph', methods=['POST', 'OPTIONS'])
def create_brain_graph_endpoint():
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

@api_bp.route('/create-graph-version', methods=['POST', 'OPTIONS'])
def create_graph_version():
    if request.method == 'OPTIONS':
        return handle_options()
    
    data = request.get_json() or {}
    graph_id = data.get("graph_id", "")
    brain_ids = data.get("brain_ids", [])
    
    # Validate UUID format
    try:
        UUID(graph_id)
    except ValueError:
        return jsonify({"error": "Invalid graph_id format - must be a valid UUID"}), 400
        
    # Validate brain_ids format
    for brain_id in brain_ids:
        try:
            UUID(brain_id)
        except ValueError:
            return jsonify({"error": f"Invalid brain_id format - {brain_id} must be a valid UUID"}), 400
    
    try:
        # Start a transaction
        response = supabase.rpc('next_version_number', {'graph_uuid': graph_id}).execute()
        if not response.data:
            return jsonify({"error": "Failed to generate version number"}), 500
            
        version_number = response.data
        
        # Update the placeholder version with the actual brain_ids
        update_response = supabase.table("brain_graph_version")\
            .update({"brain_ids": brain_ids})\
            .eq("graph_id", graph_id)\
            .eq("version_number", version_number)\
            .execute()
            
        if not update_response.data:
            return jsonify({"error": "Failed to update version with brain IDs"}), 500
            
        version_data = update_response.data[0]
        version = BrainGraphVersion(
            id=version_data["id"],
            graph_id=version_data["graph_id"],
            version_number=version_data["version_number"],
            brain_ids=version_data["brain_ids"],
            released_date=datetime.fromisoformat(version_data["released_date"].replace("Z", "+00:00")),
            status=version_data["status"]
        )
        
        return jsonify({
            "message": "Graph version created successfully",
            "version": {
                "id": version.id,
                "graph_id": version.graph_id,
                "version_number": version.version_number,
                "brain_ids": version.brain_ids,
                "status": version.status,
                "released_date": version.released_date.isoformat()
            }
        }), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api_bp.route('/release-graph-version', methods=['POST', 'OPTIONS'])
def release_graph_version():
    if request.method == 'OPTIONS':
        return handle_options()
    
    data = request.get_json() or {}
    version_id = data.get("version_id", "")
    
    if not version_id:
        return jsonify({"error": "version_id is required"}), 400
    
    try:
        # Validate UUID format
        try:
            UUID(version_id)
        except ValueError:
            return jsonify({"error": "Invalid version_id format - must be a valid UUID"}), 400
            
        version = update_brain_graph_version_status(version_id, "published")
        return jsonify({
            "message": "Graph version published successfully",
            "version": {
                "id": version.id,
                "graph_id": version.graph_id,
                "version_number": version.version_number,
                "brain_ids": version.brain_ids,
                "status": version.status,
                "released_date": version.released_date.isoformat()
            }
        }), 200
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api_bp.route('/revoke-graph-version', methods=['POST', 'OPTIONS'])
def revoke_graph_version():
    if request.method == 'OPTIONS':
        return handle_options()
    
    data = request.get_json() or {}
    version_id = data.get("version_id", "")
    
    if not version_id:
        return jsonify({"error": "version_id is required"}), 400
    
    try:
        # Validate UUID format
        try:
            UUID(version_id)
        except ValueError:
            return jsonify({"error": "Invalid version_id format - must be a valid UUID"}), 400
            
        version = update_brain_graph_version_status(version_id, "training")
        return jsonify({
            "message": "Graph version de-published successfully",
            "version": {
                "id": version.id,
                "graph_id": version.graph_id,
                "version_number": version.version_number,
                "brain_ids": version.brain_ids,
                "status": version.status,
                "revoked_date": version.released_date.isoformat()
            }
        }), 200
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api_bp.route('/update-graph-version-brains', methods=['POST', 'OPTIONS'])
def update_graph_version_brains():
    if request.method == 'OPTIONS':
        return handle_options()
    
    data = request.get_json() or {}
    version_id = data.get("version_id", "")
    action = data.get("action", "")  # "add" or "remove"
    brain_ids = data.get("brain_ids", [])
    
    if not version_id or not action or not brain_ids:
        return jsonify({"error": "version_id, action, and brain_ids are required"}), 400
    
    if action not in ["add", "remove"]:
        return jsonify({"error": "action must be either 'add' or 'remove'"}), 400
    
    try:
        # Validate UUIDs
        try:
            UUID(version_id)
            for brain_id in brain_ids:
                UUID(brain_id)
        except ValueError:
            return jsonify({"error": "Invalid UUID format"}), 400
            
        # Check version status before attempting modification
        response = supabase.table("brain_graph_version")\
            .select("status")\
            .eq("id", version_id)\
            .execute()
            
        if not response.data:
            return jsonify({"error": "Version not found"}), 404
            
        status = response.data[0].get("status")
        if status == "published":
            return jsonify({"error": "Cannot modify a published version"}), 400
            
        if action == "add":
            version = add_brains_to_version(version_id, brain_ids)
        else:
            version = remove_brains_from_version(version_id, brain_ids)
            
        return jsonify({
            "message": f"Brain IDs {action}ed successfully",
            "version": {
                "id": version.id,
                "graph_id": version.graph_id,
                "version_number": version.version_number,
                "brain_ids": version.brain_ids,
                "status": version.status,
                "released_date": version.released_date.isoformat()
            }
        }), 200
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
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

@api_bp.route('/organization-integrations', methods=['GET', 'OPTIONS'])
def get_organization_integrations():
    if request.method == 'OPTIONS':
        return handle_options()
    
    org_id = request.args.get('org_id', '')
    active_only = request.args.get('active_only', 'false').lower() == 'true'
    integration_type = request.args.get('integration_type', '')
    
    if not org_id:
        return jsonify({"error": "org_id parameter is required"}), 400
    
    try:
        integrations = get_org_integrations(org_id, active_only, integration_type if integration_type else None)
        
        # Convert to serializable format
        integrations_data = []
        for integration in integrations:
            integration_dict = {
                "id": str(integration.id),
                "org_id": str(integration.org_id),
                "integration_type": integration.integration_type,
                "name": integration.name,
                "is_active": integration.is_active,
                "api_base_url": integration.api_base_url,
                "webhook_url": integration.webhook_url,
                # Do not include sensitive information in the list endpoint
                "created_at": integration.created_at.isoformat() if integration.created_at else None,
                "updated_at": integration.updated_at.isoformat() if integration.updated_at else None
            }
            integrations_data.append(integration_dict)
        
        return jsonify({
            "integrations": integrations_data
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api_bp.route('/organization-integration/<integration_id>', methods=['GET', 'OPTIONS'])
def get_organization_integration(integration_id):
    if request.method == 'OPTIONS':
        return handle_options()
    
    try:
        integration = get_integration_by_id(integration_id)
        
        if not integration:
            return jsonify({"error": f"Integration with ID {integration_id} not found"}), 404
        
        # Convert to serializable format, including sensitive information for admin view
        integration_data = {
            "id": str(integration.id),
            "org_id": str(integration.org_id),
            "integration_type": integration.integration_type,
            "name": integration.name,
            "is_active": integration.is_active,
            "api_base_url": integration.api_base_url,
            "webhook_url": integration.webhook_url,
            "api_key": integration.api_key,
            "api_secret": "••••••" if integration.api_secret else None,  # Mask secret
            "access_token": "••••••" if integration.access_token else None,  # Mask token
            "refresh_token": "••••••" if integration.refresh_token else None,  # Mask refresh token
            "token_expires_at": integration.token_expires_at.isoformat() if integration.token_expires_at else None,
            "config": integration.config,
            "created_at": integration.created_at.isoformat() if integration.created_at else None,
            "updated_at": integration.updated_at.isoformat() if integration.updated_at else None
        }
        
        return jsonify({
            "integration": integration_data
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api_bp.route('/create-organization-integration', methods=['POST', 'OPTIONS'])
def create_organization_integration():
    if request.method == 'OPTIONS':
        return handle_options()
    
    data = request.get_json() or {}
    org_id = data.get("org_id", "")
    integration_type = data.get("integration_type", "")
    name = data.get("name", "")
    api_base_url = data.get("api_base_url")
    webhook_url = data.get("webhook_url")
    webhook_verify_token = data.get("webhook_verify_token")
    api_key = data.get("api_key")
    api_secret = data.get("api_secret")
    access_token = data.get("access_token")
    refresh_token = data.get("refresh_token")
    config = data.get("config")
    is_active = data.get("is_active", False)
    
    if not org_id or not integration_type or not name:
        return jsonify({"error": "org_id, integration_type, and name are required"}), 400
    
    try:
        # Parse dates if provided
        token_expires_at = None
        if data.get("token_expires_at"):
            try:
                token_expires_at = datetime.fromisoformat(data["token_expires_at"].replace("Z", "+00:00"))
            except ValueError:
                return jsonify({"error": "Invalid token_expires_at format. Use ISO 8601 format (YYYY-MM-DDTHH:MM:SS.sssZ)"}), 400
        
        # Get the base domain for webhook URL generation
        base_domain = os.getenv("API_BASE_URL")
        if not base_domain:
            # Try to derive from request
            host = request.headers.get('Host')
            scheme = request.headers.get('X-Forwarded-Proto', 'http')
            if host:
                base_domain = f"{scheme}://{host}"
            else:
                # Default fallback
                base_domain = "https://api.yourdomain.com"
        
        integration = create_integration(
            org_id=org_id,
            integration_type=integration_type,
            name=name,
            api_base_url=api_base_url,
            webhook_url=webhook_url,
            webhook_verify_token=webhook_verify_token,
            api_key=api_key,
            api_secret=api_secret,
            access_token=access_token,
            refresh_token=refresh_token,
            token_expires_at=token_expires_at,
            config=config,
            is_active=is_active,
            base_domain=base_domain  # Pass the base domain for webhook URL generation
        )
        
        # Convert to serializable format
        integration_data = {
            "id": str(integration.id),
            "org_id": str(integration.org_id),
            "integration_type": integration.integration_type,
            "name": integration.name,
            "is_active": integration.is_active,
            "api_base_url": integration.api_base_url,
            "webhook_url": integration.webhook_url,
            "webhook_verify_token": integration.webhook_verify_token,  # Include verify token for setup
            "api_key": integration.api_key,
            "api_secret": "••••••" if integration.api_secret else None,  # Mask secret
            "access_token": "••••••" if integration.access_token else None,  # Mask token
            "refresh_token": "••••••" if integration.refresh_token else None,  # Mask refresh token
            "token_expires_at": integration.token_expires_at.isoformat() if integration.token_expires_at else None,
            "config": integration.config,
            "created_at": integration.created_at.isoformat() if integration.created_at else None,
            "updated_at": integration.updated_at.isoformat() if integration.updated_at else None
        }
        
        return jsonify({
            "message": "Integration created successfully",
            "integration": integration_data
        }), 201
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api_bp.route('/update-organization-integration', methods=['POST', 'OPTIONS'])
def update_organization_integration():
    if request.method == 'OPTIONS':
        return handle_options()
    
    data = request.get_json() or {}
    integration_id = data.get("id", "")
    
    if not integration_id:
        return jsonify({"error": "id is required"}), 400
    
    # Fields that can be updated
    update_fields = {
        "name": data.get("name"),
        "api_base_url": data.get("api_base_url"),
        "webhook_url": data.get("webhook_url"),
        "api_key": data.get("api_key"),
        "api_secret": data.get("api_secret"),
        "access_token": data.get("access_token"),
        "refresh_token": data.get("refresh_token"),
        "config": data.get("config"),
        "is_active": data.get("is_active")
    }
    
    # Remove None values
    update_fields = {k: v for k, v in update_fields.items() if v is not None}
    
    # Parse token_expires_at if provided
    if data.get("token_expires_at"):
        try:
            update_fields["token_expires_at"] = datetime.fromisoformat(data["token_expires_at"].replace("Z", "+00:00"))
        except ValueError:
            return jsonify({"error": "Invalid token_expires_at format. Use ISO 8601 format (YYYY-MM-DDTHH:MM:SS.sssZ)"}), 400
    
    try:
        integration = update_integration(integration_id, **update_fields)
        
        if not integration:
            return jsonify({"error": f"Integration with ID {integration_id} not found or update failed"}), 404
        
        # Convert to serializable format
        integration_data = {
            "id": str(integration.id),
            "org_id": str(integration.org_id),
            "integration_type": integration.integration_type,
            "name": integration.name,
            "is_active": integration.is_active,
            "api_base_url": integration.api_base_url,
            "webhook_url": integration.webhook_url,
            "api_key": integration.api_key,
            "api_secret": "••••••" if integration.api_secret else None,  # Mask secret
            "access_token": "••••••" if integration.access_token else None,  # Mask token
            "refresh_token": "••••••" if integration.refresh_token else None,  # Mask refresh token
            "token_expires_at": integration.token_expires_at.isoformat() if integration.token_expires_at else None,
            "config": integration.config,
            "created_at": integration.created_at.isoformat() if integration.created_at else None,
            "updated_at": integration.updated_at.isoformat() if integration.updated_at else None
        }
        
        return jsonify({
            "message": "Integration updated successfully",
            "integration": integration_data
        }), 200
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api_bp.route('/delete-organization-integration', methods=['POST', 'OPTIONS'])
def delete_organization_integration():
    if request.method == 'OPTIONS':
        return handle_options()
    
    data = request.get_json() or {}
    integration_id = data.get("id", "")
    
    if not integration_id:
        return jsonify({"error": "id is required"}), 400
    
    try:
        success = delete_integration(integration_id)
        
        if not success:
            return jsonify({"error": f"Integration with ID {integration_id} not found or delete failed"}), 404
        
        return jsonify({
            "message": "Integration deleted successfully"
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api_bp.route('/toggle-organization-integration', methods=['POST', 'OPTIONS'])
def toggle_organization_integration():
    if request.method == 'OPTIONS':
        return handle_options()
    
    data = request.get_json() or {}
    integration_id = data.get("id", "")
    active = data.get("active", False)
    
    if not integration_id:
        return jsonify({"error": "id is required"}), 400
    
    try:
        integration = toggle_integration(integration_id, active)
        
        if not integration:
            return jsonify({"error": f"Integration with ID {integration_id} not found or update failed"}), 404
        
        # Convert to serializable format
        integration_data = {
            "id": str(integration.id),
            "org_id": str(integration.org_id),
            "integration_type": integration.integration_type,
            "name": integration.name,
            "is_active": integration.is_active,
            "updated_at": integration.updated_at.isoformat() if integration.updated_at else None
        }
        
        status = "activated" if active else "deactivated"
        return jsonify({
            "message": f"Integration {status} successfully",
            "integration": integration_data
        }), 200
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Add a new debugging endpoint to check active sessions for a specific thread
@api_bp.route('/debug/thread-sessions', methods=['GET'])
def debug_thread_sessions():
    """Debug endpoint to view all sessions for a specific thread"""
    thread_id = request.args.get('thread_id')
    if not thread_id:
        return jsonify({"error": "thread_id parameter is required"}), 400
    
    active_sessions = []
    with session_lock:
        for sid, session_data in ws_sessions.items():
            if session_data.get('thread_id') == thread_id:
                # Clean session data for display
                clean_data = {k: v for k, v in session_data.items() 
                             if not callable(v)}
                clean_data['session_id'] = sid
                active_sessions.append(clean_data)
    
    result = {
        "thread_id": thread_id,
        "active_session_count": len(active_sessions),
        "active_sessions": active_sessions
    }
    
    return jsonify(result), 200

@api_bp.route('/analyze-contact', methods=['GET', 'OPTIONS'])
def analyze_contact_endpoint():
    if request.method == 'OPTIONS':
        return handle_options()
    
    contact_id = request.args.get('contact_id', '')
    organization_id = request.args.get('organization_id', None)
    store_result = request.args.get('store', 'false').lower() == 'true'
    
    if not contact_id:
        return jsonify({"error": "contact_id parameter is required"}), 400
    
    try:
        # Get contact details with profile
        contact = cm.get_contact_details(int(contact_id), organization_id)
        if not contact:
            return jsonify({"error": f"No contact found with contact_id {contact_id}"}), 404
        
        # Check if contact has a profile
        if not contact.get("profiles"):
            return jsonify({"error": f"Contact with id {contact_id} does not have a profile"}), 404
        
        if store_result:
            # Analyze and store the results
            result = contact_analyzer.analyze_and_store(
                int(contact_id), 
                organization_id or contact.get("organization_id"), 
                contact["profiles"]
            )
            return jsonify({
                "contact_id": contact["id"],
                "contact_name": f"{contact['first_name']} {contact['last_name']}",
                "analysis": result["analysis"],
                "changes": result["changes"],
                "stored": result["stored"]
            }), 200
        else:
            # Just analyze without storing
            analysis = contact_analyzer.analyze_profile(contact["profiles"])
            return jsonify({
                "contact_id": contact["id"],
                "contact_name": f"{contact['first_name']} {contact['last_name']}",
                "analysis": analysis
            }), 200
    except ValueError:
        return jsonify({"error": "contact_id must be an integer"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api_bp.route('/get-analysis-history', methods=['GET', 'OPTIONS'])
def get_analysis_history_endpoint():
    if request.method == 'OPTIONS':
        return handle_options()
    
    contact_id = request.args.get('contact_id', '')
    limit = request.args.get('limit', '10')
    
    if not contact_id:
        return jsonify({"error": "contact_id parameter is required"}), 400
    
    try:
        history = contact_analyzer.get_analysis_history(int(contact_id), int(limit))
        return jsonify({"contact_id": contact_id, "history": history}), 200
    except ValueError:
        return jsonify({"error": "contact_id and limit must be integers"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api_bp.route('/get-hot-leads', methods=['GET', 'OPTIONS'])
def get_hot_leads_endpoint():
    if request.method == 'OPTIONS':
        return handle_options()
    
    organization_id = request.args.get('organization_id', None)
    min_score = request.args.get('min_score', '70')
    limit = request.args.get('limit', '20')
    
    if not organization_id:
        return jsonify({"error": "organization_id parameter is required"}), 400
    
    try:
        hot_leads = contact_analyzer.get_hot_leads(
            organization_id,
            int(min_score),
            int(limit)
        )
        
        return jsonify({
            "organization_id": organization_id,
            "hot_leads_count": len(hot_leads),
            "hot_leads": hot_leads
        }), 200
    except ValueError:
        return jsonify({"error": "min_score and limit must be integers"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api_bp.route('/get-sales-report', methods=['GET', 'OPTIONS'])
def get_sales_report():
    if request.method == 'OPTIONS':
        return handle_options()
    
    organization_id = request.args.get('organization_id', None)
    
    if not organization_id:
        return jsonify({"error": "organization_id parameter is required"}), 400
    
    try:
        # Get all contacts for the organization
        contacts = cm.get_contacts(organization_id)
        if not contacts:
            return jsonify({"message": "No contacts found for this organization", "report": {
                "total_contacts": 0,
                "total_with_profiles": 0,
                "lead_breakdown": {
                    "hot_leads": 0,
                    "warm_leads": 0,
                    "nurture_leads": 0,
                    "unqualified_leads": 0
                },
                "top_opportunities": [],
                "urgent_opportunities": []
            }}), 200
        
        # Generate the report
        report = contact_analyzer.generate_weekly_leads_report(contacts)
        
        return jsonify({"report": report}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api_bp.route('/run-bulk-analysis', methods=['POST', 'OPTIONS'])
def run_bulk_analysis():
    if request.method == 'OPTIONS':
        return handle_options()
    
    data = request.get_json() or {}
    organization_id = data.get("organization_id", "")
    
    if not organization_id:
        return jsonify({"error": "organization_id is required"}), 400
    
    try:
        # Get all contacts for the organization
        contacts = cm.get_contacts(organization_id)
        if not contacts:
            return jsonify({"message": "No contacts found for this organization", "processed": 0}), 200
        
        # Count contacts with profiles
        contacts_with_profiles = [c for c in contacts if c.get("profiles")]
        
        if not contacts_with_profiles:
            return jsonify({"message": "No contacts with profiles found for this organization", "processed": 0}), 200
        
        # Start background task
        def run_bulk_analysis():
            processed = 0
            for contact in contacts_with_profiles:
                try:
                    contact_analyzer.analyze_and_store(
                        contact["id"],
                        organization_id,
                        contact["profiles"]
                    )
                    processed += 1
                    logger.info(f"Processed {processed}/{len(contacts_with_profiles)} contacts in bulk analysis")
                except Exception as e:
                    logger.error(f"Error analyzing contact {contact['id']}: {str(e)}")
            
            logger.info(f"Bulk analysis completed. Processed {processed}/{len(contacts_with_profiles)} contacts")
        
        # Start analysis in background thread
        import threading
        thread = threading.Thread(target=run_bulk_analysis)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            "message": "Bulk analysis started", 
            "total_contacts": len(contacts),
            "contacts_with_profiles": len(contacts_with_profiles)
        }), 202  # 202 Accepted
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api_bp.route('/create-or-update-profile', methods=['POST', 'OPTIONS'])
def create_or_update_profile_endpoint():
    if request.method == 'OPTIONS':
        return handle_options()
    
    data = request.get_json() or {}
    contact_id = data.get("contact_id", "")
    organization_id = data.get("organization_id", None)
    profile_summary = data.get("profile_summary", None)
    general_info = data.get("general_info", None)
    personality = data.get("personality", None)
    hidden_desires = data.get("hidden_desires", None)
    linkedin_url = data.get("linkedin_url", None)
    social_media_urls = data.get("social_media_urls", None)
    best_goals = data.get("best_goals", None)
    
    if not contact_id:
        return jsonify({"error": "contact_id is required"}), 400
    
    try:
        # First verify the contact exists and belongs to the organization
        if organization_id:
            contact = cm.get_contact_details(int(contact_id), organization_id)
            if not contact:
                return jsonify({"error": f"No contact found with id {contact_id} in this organization"}), 404
                
        # Prepare update data
        profile_data = {}
        if profile_summary is not None:
            profile_data["profile_summary"] = profile_summary
        if general_info is not None:
            profile_data["general_info"] = general_info
        if personality is not None:
            profile_data["personality"] = personality
        if hidden_desires is not None:
            profile_data["hidden_desires"] = hidden_desires
        if linkedin_url is not None:
            profile_data["linkedin_url"] = linkedin_url
        if social_media_urls is not None:
            profile_data["social_media_urls"] = social_media_urls
        if best_goals is not None:
            profile_data["best_goals"] = best_goals
        
        # Create or update the profile
        updated_profile = cm.create_or_update_contact_profile(int(contact_id), **profile_data)
        if not updated_profile:
            return jsonify({"error": f"Failed to create or update profile for contact {contact_id}"}), 500
        
        # Format response
        profile_data = {
            "id": updated_profile["id"],
            "uuid": updated_profile["uuid"],
            "contact_id": updated_profile["contact_id"],
            "profile_summary": updated_profile["profile_summary"],
            "general_info": updated_profile["general_info"],
            "personality": updated_profile["personality"],
            "hidden_desires": updated_profile["hidden_desires"],
            "linkedin_url": updated_profile["linkedin_url"],
            "social_media_urls": updated_profile["social_media_urls"],
            "best_goals": updated_profile["best_goals"],
            "updated_at": updated_profile["updated_at"]
        }
        
        # After successful profile update, trigger sales signal analysis
        try:
            # Run in a separate thread to not block the response
            def run_analysis():
                analysis_result = contact_analyzer.analyze_and_store(
                    int(contact_id),
                    organization_id or contact.get("organization_id", ""),
                    updated_profile
                )
                logger.info(f"Updated sales analysis for contact {contact_id} with score {analysis_result['analysis'].get('sales_readiness_score', 0)}")
            
            # Start analysis in background thread
            import threading
            thread = threading.Thread(target=run_analysis)
            thread.daemon = True
            thread.start()
        except Exception as analysis_error:
            # Log the error but don't fail the profile update
            logger.error(f"Error updating sales analysis: {str(analysis_error)}")
        
        is_new = contact.get("profiles") is None
        status_code = 201 if is_new else 200
        message = "Profile created successfully" if is_new else "Profile updated successfully"
        
        return jsonify({"message": message, "profile": profile_data}), status_code
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api_bp.route('/batch-enrich-profiles', methods=['POST', 'OPTIONS'])
def batch_enrich_profiles_endpoint():
    if request.method == 'OPTIONS':
        return handle_options()
    
    data = request.get_json() or {}
    organization_id = data.get("organization_id", None)
    max_contacts = data.get("max_contacts", 100)
    
    if not organization_id:
        return jsonify({"error": "organization_id is required"}), 400
    
    try:
        # Start background task
        def run_batch_enrichment():
            try:
                enricher = ProfileEnricher()
                profiles = asyncio.run(enricher.batch_update_profiles(organization_id, max_contacts))
                logger.info(f"Batch profile enrichment completed. Updated {len(profiles)} profiles.")
            except Exception as e:
                logger.error(f"Error in batch profile enrichment: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
        
        # Start enrichment in background thread
        import threading
        thread = threading.Thread(target=run_batch_enrichment)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            "message": "Batch profile enrichment started", 
            "organization_id": organization_id,
            "max_contacts": max_contacts
        }), 202  # 202 Accepted
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api_bp.route('/batch-analyze-contacts', methods=['POST', 'OPTIONS'])
def batch_analyze_contacts_endpoint():
    if request.method == 'OPTIONS':
        return handle_options()
    
    data = request.get_json() or {}
    organization_id = data.get("organization_id", None)
    min_contacts = data.get("min_contacts", 5)
    max_contacts = data.get("max_contacts", 100)
    batch_size = data.get("batch_size", 10)
    delay_seconds = data.get("delay_seconds", 0.5)
    
    if not organization_id:
        return jsonify({"error": "organization_id is required"}), 400
    
    try:
        # Start background task
        def run_batch_analysis():
            try:
                analyzer = ContactAnalyzer()
                result = analyzer.batch_analyze_contacts(
                    organization_id=organization_id,
                    min_contacts=min_contacts,
                    max_contacts=max_contacts,
                    batch_size=batch_size,
                    delay_seconds=delay_seconds
                )
                logger.info(f"Batch contact analysis completed. Processed {result.get('processed', 0)} contacts.")
            except Exception as e:
                logger.error(f"Error in batch contact analysis: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
        
        # Start analysis in background thread
        import threading
        thread = threading.Thread(target=run_batch_analysis)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            "message": "Batch contact analysis started", 
            "organization_id": organization_id,
            "max_contacts": max_contacts
        }), 202  # 202 Accepted
        
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
