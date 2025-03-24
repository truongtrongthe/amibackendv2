# main.py
# Purpose: Flask app with /pilot for Copilot Mode and /learning for Teaching Mode
# Date: March 23, 2025

from flask import Flask, Response, request
from flask_cors import CORS
from ami import convo_stream  # Unified stream function from ami.py

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes, all origins

@app.route('/pilot', methods=['POST', 'OPTIONS'])
def ami_copilot():
    if request.method == 'OPTIONS':
        response = Response()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        response.headers['Access-Control-Max-Age'] = '86400'  # Cache preflight for 1 day
        return response, 200

    data = request.get_json() or {}
    user_input = data.get("user_input")
    user_id = data.get("user_id", "tfl")
    thread_id = data.get("thread_id", "copilot_thread")

    print("Headers:", request.headers)
    print("Copilot API called!")

    return Response(
        convo_stream(user_input, user_id, thread_id, mode="copilot"),  # Copilot Mode
        mimetype='text/event-stream',
        headers={
            'X-Accel-Buffering': 'no',
            'Access-Control-Allow-Origin': '*'
        }
    )

@app.route('/learning', methods=['POST', 'OPTIONS'])
def learn():
    if request.method == 'OPTIONS':
        response = Response()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        response.headers['Access-Control-Max-Age'] = '86400'
        return response, 200

    data = request.get_json() or {}
    user_input = data.get("user_input", "")
    user_id = data.get("user_id", "Teacher")  # Default to "Teacher" for teaching mode
    thread_id = data.get("thread_id", "global_thread")

    print("Headers:", request.headers)
    print("Learning API called!")

    return Response(
        convo_stream(user_input, user_id, thread_id, mode="teaching"),  # Teaching Mode
        mimetype='text/event-stream',
        headers={
            'X-Accel-Buffering': 'no',
            'Access-Control-Allow-Origin': '*'
        }
    )

@app.route('/')
def home():
    return "Hello, It's me Ami!"

@app.route('/ping', methods=['POST'])
def ping():
    return "Pong"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)