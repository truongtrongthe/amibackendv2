import os
from dotenv import load_dotenv
from flask import Flask, Response, stream_with_context, request, jsonify
from flask_cors import CORS  # Import CORS
from ami import generate_response
from ami2 import generate_response_v2
app = Flask(__name__)

# Enable CORS for all routes and specify allowed origins
CORS(app, resources={r"/stream": {"origins": "*", "methods": ["POST", "OPTIONS"], "allow_headers": "*"}})

# Load environment variables from .env file
load_dotenv()

@app.route('/stream', methods=['POST'])
def stream_response():
    data = request.get_json()
    prompt = data.get("prompt", "Tell me about Ami.")

    return Response(
        stream_with_context(generate_response(prompt)),
        content_type='text/plain',
        headers={'X-Accel-Buffering': 'no'}  # Disable buffering for Nginx (if used)
    )

@app.route('/chat', methods=['POST'])
def chat_response():
    data = request.get_json()
    prompt = data.get("prompt")

    return Response(
        stream_with_context(generate_response_v2(prompt)),
        content_type='text/plain',
        headers={'X-Accel-Buffering': 'no'}  # Disable buffering for Nginx (if used)
    )

@app.route('/stream', methods=['OPTIONS'])
def options_response():
    return Response(status=200)

@app.route('/')
def home():
    return "Hello, It's me Ami!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)