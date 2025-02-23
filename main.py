import os
from dotenv import load_dotenv
from flask import Flask, Response, stream_with_context, request, jsonify
from flask_cors import CORS  # Import CORS
from ami2 import ami_response
from knowledge import tobrain
from summarizer import summarize_text
from brain import ami_telling
app = Flask(__name__)

# Enable CORS for all routes and specify allowed origins
CORS(app, resources={r"/*": {"origins": ["http://localhost", "http://localhost:5173"], "methods": ["POST", "OPTIONS"], "allow_headers": "*"}})

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

@app.route('/save-knowledge', methods=['POST'])
def save_response():
    data = request.get_json()
    new_knowledge = data.get("new_knowledge")
    raw_content = data.get("raw_content")

    return Response(
        stream_with_context(tobrain(new_knowledge, raw_content)),
        content_type='text/plain',
        headers={'X-Accel-Buffering': 'no'}  # Disable buffering for Nginx (if used)
    )


@app.route('/save-knowledge', methods=['OPTIONS'])
def options_response():
    return Response(status=200)

@app.route('/')
def home():
    return "Hello, It's me Ami!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)