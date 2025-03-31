# main.py
# Purpose: Flask app with /pilot, /training, and /havefun endpoints for different modes
# Date: March 23, 2025 (Updated March 31, 2025)

from flask import Flask, Response, request
from flask_cors import CORS
from ami import convo_stream  # Unified stream function from ami.py
import asyncio

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
    response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
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

    print("Headers:", request.headers)
    print("Fun API called!")

    async_gen = convo_stream(user_input=user_input, user_id=user_id, thread_id=thread_id, mode="funny")
    return create_stream_response(async_gen)

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
        # Ensure the loop is closed cleanly on shutdown
        if not loop.is_closed():
            pending = asyncio.all_tasks(loop)
            if pending:
                loop.run_until_complete(asyncio.gather(*pending))
            loop.close()