# main.py
# Purpose: Flask app with /pilot, /training, /havefun, /labels, and /label-details endpoints
# Date: March 23, 2025 (Updated April 01, 2025)

from flask import Flask, Response, request, jsonify
import json
from uuid import UUID
from datetime import datetime

from flask_cors import CORS
from ami import convo_stream  # Unified stream function from ami.py
import asyncio
from typing import List, Optional  # Added List and Optional imports
# Assuming these are in a module called 'data_fetch.py' - adjust as needed
from database import get_all_labels, get_raw_data_by_label, clean_text
from docuhandler import process_document,summarize_document
from braindb import get_brains,get_brain_details,update_brain,create_brain,get_organization, create_organization, update_organization
from aia import create_aia,get_all_aias,get_aia_detail,delete_aia,update_aia
from brainlog import get_brain_logs, get_brain_log_detail, BrainLog  # Assuming these are in brain_logs.py
from contact import ContactManager
from fbMessenger import get_sender_text, send_message, parse_fb_message, save_fb_message_to_conversation, process_facebook_webhook,send_text_to_facebook_user
from contactconvo import ConversationManager
from braingraph import (
    create_brain_graph, get_brain_graph, create_brain_graph_version,
    add_brains_to_version, remove_brains_from_version, get_brain_graph_versions,
    update_brain_graph_version_status, BrainGraphVersion
)
from org_integrations import (
    create_integration, get_org_integrations, get_integration_by_id,
    update_integration, delete_integration, toggle_integration, OrganizationIntegration
)
from supabase import create_client, Client
import os
spb_url = os.getenv("SUPABASE_URL")
spb_key = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(
    spb_url,
    spb_key
)


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Simple CORS configuration that was working before
CORS(app, resources={r"/*": {"origins": "*"}})  # Enable CORS for all routes, all origins

cm = ContactManager()
convo_mgr = ConversationManager()
# Single event loop for the app
loop = asyncio.get_event_loop()


def handle_options():
    """Common OPTIONS handler for all endpoints."""
    response = Response()
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    response.headers['Access-Control-Max-Age'] = '86400'
    return response, 200

def create_stream_response(gen):
    """Common response creator for streaming endpoints."""
    
    response = Response(gen, mimetype='text/event-stream')
    response.headers['X-Accel-Buffering'] = 'no'
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

# Existing endpoints (pilot, training, havefun) remain unchanged
@app.route('/pilot', methods=['POST', 'OPTIONS'])
def pilot():
    if request.method == 'OPTIONS':
        return handle_options()
    data = request.get_json() or {}
    user_input = data.get("user_input", "")
    user_id = data.get("user_id", "thefusionlab")
    thread_id = data.get("thread_id", "pilot_thread")
    print("Headers:", request.headers)
    print("Pilot API called!")
    async_gen = convo_stream(user_input=user_input, user_id=user_id, thread_id=thread_id, mode="pilot")
    return create_stream_response(async_gen)

@app.route('/training', methods=['POST', 'OPTIONS'])
def training():
    if request.method == 'OPTIONS':
        return handle_options()
    data = request.get_json() or {}
    user_input = data.get("user_input", "")
    user_id = data.get("user_id", "thefusionlab")
    thread_id = data.get("thread_id", "training_thread")
    print("Headers:", request.headers)
    print("Training API called!")
    async_gen = convo_stream(user_input=user_input, user_id=user_id, thread_id=thread_id, mode="training")
    return create_stream_response(async_gen)

@app.route('/havefun', methods=['POST', 'OPTIONS'])
def havefun():
    if request.method == 'OPTIONS':
        return handle_options()
    data = request.get_json() or {}
    user_input = data.get("user_input", "")
    user_id = data.get("user_id", "thefusionlab")
    thread_id = data.get("thread_id", "chat_thread")
    graph_version_id = data.get("graph_version_id", "")

    print("Headers:", request.headers)
    print("Fun API called!")
    print("graph_version_id=", graph_version_id)
    gen = convo_stream(
        user_input=user_input, 
        user_id=user_id, 
        thread_id=thread_id,
        graph_version_id=graph_version_id,
        mode="mc"
    )
    return create_stream_response(gen)

@app.route('/autopilot', methods=['POST', 'OPTIONS'])
def gopilot():
    if request.method == 'OPTIONS':
        return handle_options()
    data = request.get_json() or {}
    user_input = data.get("user_input", "")
    user_id = data.get("user_id", "thefusionlab")
    thread_id = data.get("thread_id", "chat_thread")
    
    print("Headers:", request.headers)
    print("Fun API called!")
    async_gen = convo_stream(user_input=user_input, user_id=user_id, thread_id=thread_id,bank_name=bank_name, mode="mc")
    return create_stream_response(async_gen)

@app.route('/labels', methods=['GET', 'OPTIONS'])
async def get_labels():
    print(f"Received request: {request.method} {request.path}")
    if request.method == 'OPTIONS':
        return handle_options()
    
    bank_name = request.args.get('bank_name', '')
    if not bank_name:
        return jsonify({"error": "bank_name parameter is required"}), 400
    
    labels = await get_all_labels(lang="original",bank_name=bank_name)
    print(f"Labels from DB: {labels}")
    if not labels:
        return jsonify({"error": "No labels found"}), 404
    
    label_list = list(labels)
    return jsonify({"labels": label_list}), 200

@app.route('/label-details', methods=['GET', 'OPTIONS'])
async def get_label_details():
    if request.method == 'OPTIONS':
        return handle_options()
    
    label = request.args.get('label', '')
    if not label:
        return jsonify({"error": "Label parameter is required"}), 400
    bank_name = request.args.get('bank_name', '')
    if not bank_name:
        return jsonify({"error": "bank_name parameter is required"}), 400
    
    raw_data = await get_raw_data_by_label(label, lang="original",bank_name=bank_name)
    if not raw_data:
        return jsonify({"error": f"No data found for label '{label}'"}), 404
    
    cleaned_data = [clean_text(text) for text in raw_data]
    
    response_data = {
        "label": label,
        "total_entries": len(cleaned_data),
        "data": cleaned_data
    }
    return jsonify(response_data), 200

@app.route('/summarize-document', methods=['POST', 'OPTIONS'])
def summary_document_endpoint():
    if request.method == 'OPTIONS':
        return handle_options()

    if 'file' not in request.files or 'user_id' not in request.form:
        return jsonify({"error": "Missing file or user_id"}), 400

    file = request.files['file']
    user_id = request.form['user_id']
    mode = request.form.get('mode', 'default')  # Optional mode parameter

    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    # Run the async process_document function synchronously using the event loop
    summary = loop.run_until_complete(summarize_document(file, user_id, mode))

    if summary:
        return jsonify({
            "summary": summary
        }), 200
    else:
        return jsonify({
            "error": "Failed to process document"
        }), 500

@app.route('/process-document', methods=['POST', 'OPTIONS'])
def process_document_endpoint():
    if request.method == 'OPTIONS':
        return handle_options()

    if 'file' not in request.files or 'user_id' not in request.form or 'bank_name' not in request.form:
        return jsonify({"error": "Missing file or user_id"}), 400

    file = request.files['file']
    user_id = request.form['user_id']
    bank_name=request.form['bank_name']
    mode = request.form.get('mode', 'default')  # Optional mode parameter
    

    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    # Run the async process_document function synchronously using the event loop
    success = loop.run_until_complete(process_document(file, user_id, mode,bank_name))

    if success:
        return jsonify({
            "message": "Document processed successfully"
        }), 200
    else:
        return jsonify({
            "error": "Failed to process document"
        }), 500


@app.route('/brains', methods=['GET', 'OPTIONS'])
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

@app.route('/brain-details', methods=['GET', 'OPTIONS'])
def brain_details():
    if request.method == 'OPTIONS':
        return handle_options()
    
    brain_id = request.args.get('brain_id', '')
    if not brain_id:
        return jsonify({"error": "brain_id parameter is required"}), 400
    
    try:
        brain = get_brain_details(int(brain_id))  # Convert to int since brain_id is an integer
        if not brain:
            return jsonify({"error": f"No brain found with brain_id {brain_id}"}), 404
        
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
    except ValueError:
        return jsonify({"error": "brain_id must be an integer"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/update-brain', methods=['POST', 'OPTIONS'])
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

@app.route('/create-brain', methods=['POST', 'OPTIONS'])
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
@app.route('/get-org-detail/<orgid>', methods=['GET', 'OPTIONS'])
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

@app.route('/aias', methods=['GET', 'OPTIONS'])
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

@app.route('/aia-details', methods=['GET', 'OPTIONS'])
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

@app.route('/create-aia', methods=['POST', 'OPTIONS'])
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

@app.route('/update-aia', methods=['POST', 'OPTIONS'])
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

@app.route('/delete-aia', methods=['POST', 'OPTIONS'])
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

@app.route('/brainlogs', methods=['GET', 'OPTIONS'])
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

@app.route('/brain-log-detail', methods=['GET', 'OPTIONS'])
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

@app.route('/contact-details', methods=['GET', 'OPTIONS'])
def contact_details():
    if request.method == 'OPTIONS':
        return handle_options()
    
    contact_id = request.args.get('contact_id', '')
    if not contact_id:
        return jsonify({"error": "contact_id parameter is required"}), 400
    
    try:
        contact = cm.get_contact_details(int(contact_id))  # Convert to int since id is an integer
        if not contact:
            return jsonify({"error": f"No contact found with contact_id {contact_id}"}), 404
        
        contact_data = {
            "id": contact["id"],
            "uuid": contact["uuid"],
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

@app.route('/update-contact', methods=['POST', 'OPTIONS'])
def update_contact_endpoint():
    if request.method == 'OPTIONS':
        return handle_options()
    
    data = request.get_json() or {}
    contact_id = data.get("id", "")
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
        
        updated_contact = cm.update_contact(int(contact_id), **update_data)
        if not updated_contact:
            return jsonify({"error": f"No contact found with id {contact_id}"}), 404
        
        contact_data = {
            "id": updated_contact["id"],
            "uuid": updated_contact["uuid"],
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

@app.route('/create-contact', methods=['POST', 'OPTIONS'])
def create_contact_endpoint():
    if request.method == 'OPTIONS':
        return handle_options()
    
    data = request.get_json() or {}
    type = data.get("type", "")
    first_name = data.get("first_name", "")
    last_name = data.get("last_name", "")
    email = data.get("email", None)
    phone = data.get("phone", None)
    facebook_id = data.get("facebook_id", None)
    profile_picture_url = data.get("profile_picture_url", None)
    
    if not type or not first_name or not last_name:
        return jsonify({"error": "type, first_name, and last_name are required"}), 400
    
    try:
        new_contact = cm.create_contact(type, first_name, last_name, email, phone, facebook_id, profile_picture_url)
        contact_data = {
            "id": new_contact["id"],
            "uuid": new_contact["uuid"],
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

@app.route('/profile-details', methods=['GET', 'OPTIONS'])
def profile_details():
    if request.method == 'OPTIONS':
        return handle_options()
    
    contact_id = request.args.get('contact_id', '')
    if not contact_id:
        return jsonify({"error": "contact_id parameter is required"}), 400
    
    try:
        contact = cm.get_contact_details(int(contact_id))
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

@app.route('/update-profile', methods=['POST', 'OPTIONS'])
def update_profile_endpoint():
    if request.method == 'OPTIONS':
        return handle_options()
    
    data = request.get_json() or {}
    contact_id = data.get("contact_id", "")
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
        return jsonify({"message": "Profile updated successfully", "profile": profile_data}), 200
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/create-profile', methods=['POST', 'OPTIONS'])
def create_profile_endpoint():
    if request.method == 'OPTIONS':
        return handle_options()
    
    data = request.get_json() or {}
    contact_id = data.get("contact_id", "")
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
        return jsonify({"message": "Profile created successfully", "profile": profile_data}), 201
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/contacts', methods=['GET', 'OPTIONS'])
def get_all_contacts():
    if request.method == 'OPTIONS':
        return handle_options()
    
    try:
        contacts = cm.get_contacts()
        if not contacts:
            return jsonify({"message": "No contacts found", "contacts": []}), 200
        
        return jsonify({"contacts": contacts}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

import os
VERIFY_TOKEN = os.getenv("CALLBACK_V_TOKEN")

@app.route("/webhook", methods=["GET"])
def verify_webhook():
    mode = request.args.get("hub.mode")
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")

    if mode == "subscribe" and token == VERIFY_TOKEN:
        print("✅ Webhook verified by Facebook.")
        return challenge, 200
    else:
        return "Forbidden", 403


@app.route("/webhook", methods=["POST"])
def handle_message():
    data = request.json
    
    try:
        # Check if this is a "message echo" event (messages sent by our page)
        entry = data.get("entry", [{}])[0]
        messaging = entry.get("messaging", [{}])[0]
        is_echo = messaging.get("message", {}).get("is_echo", False)
        
        if is_echo:
            print(f"Received message echo event (message sent by our page)")
        else:
            print(f"Received message from user")
            
        # Always process the webhook to track messages in our database
        success = process_facebook_webhook(data, convo_mgr)
        
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
                        send_text_to_facebook_user(sender_id, response_text, convo_mgr)
            except Exception as e:
                print(f"Error processing user message for response: {str(e)}")
    except Exception as e:
        print(f"Error in webhook handler: {str(e)}")
    
    # Always return 200 to Facebook
    return "", 200


@app.route('/')
def home():
    return "Hello, It's me Ami!"

@app.route('/ping', methods=['POST'])
def ping():
    return "Pong"

@app.route('/contact-conversations', methods=['GET', 'OPTIONS'])
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

@app.route('/conversation', methods=['GET', 'OPTIONS'])
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

@app.route('/create-organization', methods=['POST', 'OPTIONS'])
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

@app.route('/update-organization', methods=['POST', 'OPTIONS'])
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

@app.route('/create-brain-graph', methods=['POST', 'OPTIONS'])
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

@app.route('/get-org-brain-graph', methods=['GET', 'OPTIONS'])
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

@app.route('/create-graph-version', methods=['POST', 'OPTIONS'])
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

@app.route('/release-graph-version', methods=['POST', 'OPTIONS'])
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

@app.route('/revoke-graph-version', methods=['POST', 'OPTIONS'])
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

@app.route('/update-graph-version-brains', methods=['POST', 'OPTIONS'])
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

@app.route('/get-version-brains', methods=['GET', 'OPTIONS'])
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

@app.route('/organization-integrations', methods=['GET', 'OPTIONS'])
def get_organization_integrations():
    if request.method == 'OPTIONS':
        return handle_options()
    
    org_id = request.args.get('org_id', '')
    active_only = request.args.get('active_only', 'false').lower() == 'true'
    
    if not org_id:
        return jsonify({"error": "org_id parameter is required"}), 400
    
    try:
        integrations = get_org_integrations(org_id, active_only)
        
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

@app.route('/organization-integration/<integration_id>', methods=['GET', 'OPTIONS'])
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

@app.route('/create-organization-integration', methods=['POST', 'OPTIONS'])
def create_organization_integration():
    if request.method == 'OPTIONS':
        return handle_options()
    
    data = request.get_json() or {}
    org_id = data.get("org_id", "")
    integration_type = data.get("integration_type", "")
    name = data.get("name", "")
    api_base_url = data.get("api_base_url")
    webhook_url = data.get("webhook_url")
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
        
        integration = create_integration(
            org_id=org_id,
            integration_type=integration_type,
            name=name,
            api_base_url=api_base_url,
            webhook_url=webhook_url,
            api_key=api_key,
            api_secret=api_secret,
            access_token=access_token,
            refresh_token=refresh_token,
            token_expires_at=token_expires_at,
            config=config,
            is_active=is_active
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

@app.route('/update-organization-integration', methods=['POST', 'OPTIONS'])
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

@app.route('/delete-organization-integration', methods=['POST', 'OPTIONS'])
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

@app.route('/toggle-organization-integration', methods=['POST', 'OPTIONS'])
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

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5001, threaded=True)
    finally:
        if not loop.is_closed():
            pending = asyncio.all_tasks(loop)
            if pending:
                loop.run_until_complete(asyncio.gather(*pending))
            loop.close()