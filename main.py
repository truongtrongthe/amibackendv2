import os   
from dotenv import load_dotenv
from flask import Flask, Response, stream_with_context, request, jsonify
from flask_cors import CORS  # Import CORS
from ami2 import ami_response
from knowledge import tobrain
from summarizer import summarize_text
from brain import ami_telling
from database import insert_knowledge_entry, get_knowledge_entries
app = Flask(__name__)

# Enable CORS for all routes and allow all origins
CORS(app)

# Load environment variables from .env file
load_dotenv()

@app.route('/chat', methods=['POST'])
def chat_response():
    data = request.get_json()
    prompt = data.get("prompt")

    return Response(
        stream_with_context(ami_response(prompt)),
        content_type='text/plain',
        headers={'X-Accel-Buffering': 'no'}  # Disable buffering for Nginx (if used)
    )

@app.route('/ami-spell', methods=['POST'])
def spell_response():
    data = request.get_json()
    prompt = data.get("prompt")
    return Response(
        stream_with_context((chunk.content for chunk in ami_telling(prompt))),  # Access content directly
        content_type='text/plain',
        headers={'X-Accel-Buffering': 'no'}  # Disable buffering for Nginx (if used)
    )

@app.route('/preview-knowledge', methods=['POST'])
def preview_response():
    data = request.get_json()
    rawknowledge = data.get("raw_knowledge")

    return Response(
        stream_with_context(summarize_text(rawknowledge)),
        content_type='text/plain',
        headers={'X-Accel-Buffering': 'no'}  # Disable buffering for Nginx (if used)
    )


@app.route('/save-knowledge', methods=['POST', 'OPTIONS'])
def save_response():
    if request.method == "OPTIONS":
        return Response(status=200)
    data = request.get_json()
    new_knowledge = data.get("new_knowledge")
    raw_content = data.get("raw_content")

    try:
        tobrain(new_knowledge, raw_content)
        insert_knowledge_entry(raw_content, new_knowledge)
        print("summary generated:", new_knowledge)
        return Response(
            "Memory saved successfully!",
            content_type='text/plain',
            headers={'X-Accel-Buffering': 'no'}  # Disable buffering for Nginx (if used)
        ), 200  # Return 200 if streaming is successful
    except Exception as e:
        # Handle the exception (log it, return an error response, etc.)
        return Response(str(e), status=500)  # Return 500 if there's an error

@app.route('/get-knowledge-entries', methods=['GET'])
def get_knowledge_response():
    """Retrieve all knowledge entries."""
    try:
        entries = get_knowledge_entries()  # Fetch all knowledge entries
        return jsonify(entries.data), 200  # Return the entries as JSON
    except Exception as e:
        return Response(str(e), status=500)  # Return 500 if there's an error

@app.route('/<path:path>', methods=['OPTIONS'])
def options_response1(path):
    return Response(status=200)

@app.after_request
def add_cors_headers(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS, GET')  # Ensure GET is included
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
    return response

@app.route('/')
def home():
    return "Hello, It's me Ami!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)