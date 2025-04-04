# main.py
# Purpose: Flask app with /pilot, /training, /havefun, /labels, and /label-details endpoints
# Date: March 23, 2025 (Updated April 01, 2025)

from flask import Flask, Response, request, jsonify
from flask_cors import CORS
from ami import convo_stream  # Unified stream function from ami.py
import asyncio
# Assuming these are in a module called 'data_fetch.py' - adjust as needed
from database import get_all_labels, get_raw_data_by_label, clean_text
from docuhandler import process_document,summarize_document
from braindb import get_brains,get_brain_details,update_brain,create_brain,get_organization
from aia import create_aia,get_all_aias,get_aia_detail,delete_aia

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Enable CORS for all routes, all origins

# Single event loop for the app
loop = asyncio.get_event_loop()

def async_to_sync_generator(async_gen):
    """Convert an async generator to a synchronous generator using the app's event loop."""
    while True:
        try:
            yield loop.run_until_complete(anext(async_gen))
        except StopAsyncIteration:
            break

def handle_options():
    """Common OPTIONS handler for all endpoints."""
    response = Response()
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    response.headers['Access-Control-Max-Age'] = '86400'
    return response, 200

def create_stream_response(async_gen):
    """Common response creator for streaming endpoints."""
    sync_gen = async_to_sync_generator(async_gen)
    response = Response(sync_gen, mimetype='text/event-stream')
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
    bank_name = data.get("bank_name","")

    print("Headers:", request.headers)
    print("Fun API called!")
    print("bankname=",bank_name)
    async_gen = convo_stream(user_input=user_input, user_id=user_id, thread_id=thread_id,bank_name=bank_name, mode="mc")
    return create_stream_response(async_gen)

@app.route('/autopilot', methods=['POST', 'OPTIONS'])
def gopilot():
    if request.method == 'OPTIONS':
        return handle_options()
    data = request.get_json() or {}
    user_input = data.get("user_input", "")
    user_id = data.get("user_id", "thefusionlab")
    thread_id = data.get("thread_id", "chat_thread")
    bank_name = data.get("bank_name","")

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
            "name": org.name
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

@app.route('/')
def home():
    return "Hello, It's me Ami!"

@app.route('/ping', methods=['POST'])
def ping():
    return "Pong"

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5001, threaded=True)
    finally:
        if not loop.is_closed():
            pending = asyncio.all_tasks(loop)
            if pending:
                loop.run_until_complete(asyncio.gather(*pending))
            loop.close()