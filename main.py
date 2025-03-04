from flask import Flask, Response, request
from flask_cors import CORS  # Import CORS
from amilearn import event_stream
from copilot import pilot_stream
app = Flask(__name__)

# Enable CORS for all routes and allow all origins

CORS(app)


@app.route('/copilot', methods=['POST'])
def copilot():
    data = request.get_json()
    user_input = data.get("user_input")
    user_id ="tfl"
    user_id = data.get("user_id", "tfl")  # Allow client to specify, default to "tfl"
    thread_id = data.get("thread_id", "global_thread")  # Optional thread_id from client
    return Response(
        pilot_stream(user_input, user_id, thread_id), 
        mimetype='text/event-stream', 
        headers={'X-Accel-Buffering': 'no'}
        )  # Disable buffering for Nginx (if used))

@app.route('/ami-learn', methods=['POST'])
def ami_learn():
    data = request.get_json()
    user_input = data.get("user_input")
    user_id ="tfl"
    #user_id = data.get("user_id", "tfl")  # Allow client to specify, default to "tfl"
    thread_id = data.get("thread_id", "global_thread")  # Optional thread_id from client
    return Response(
        event_stream(user_input, user_id, thread_id), 
        mimetype='text/event-stream', 
        headers={'X-Accel-Buffering': 'no'}
        )  # Disable buffering for Nginx (if used))
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