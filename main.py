from dotenv import load_dotenv
from flask import Flask, Response, stream_with_context, request, jsonify, make_response
from flask_cors import CORS  # Import CORS
from conversationflow import event_stream
from brain import ami_telling
from graph import g_app

app = Flask(__name__)

# Enable CORS for all routes and allow all origins

CORS(app)


@app.route('/ami-spell', methods=['POST'])
def spell_response():
    data = request.get_json()
    prompt = data.get("prompt")
    return Response(
        stream_with_context((chunk.content for chunk in ami_telling(prompt))),  # Access content directly
        content_type='text/plain',
        headers={'X-Accel-Buffering': 'no'}  # Disable buffering for Nginx (if used)
    )


@app.route('/ami-convo', methods=['POST'])
def ami_convo_response():
    data = request.get_json()
    prompt = data.get("prompt")
    resp = Response(event_stream(prompt), mimetype='text/event-stream')
    return resp



# Middleware to log headers for debugging

@app.after_request
def add_cors_headers(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS, GET')  # Ensure GET is included
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
    return response

@app.route('/')
def home():
    return "Hello, It's me Ami!"

@app.route('/ping', methods=['POST'])
def ping():
    return "Pong"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)