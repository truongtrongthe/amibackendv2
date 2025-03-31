# main.py
# Purpose: Flask app with /pilot, /training, and /havefun endpoints for different modes
# Date: March 23, 2025 (Updated March 30, 2025)

from flask import Flask, Response, request
from flask_cors import CORS
from ami import convo_stream  # Unified stream function from ami.py
import asyncio

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes, all origins

def async_to_sync_generator(async_gen):
    """Convert an async generator to a synchronous generator for Flask."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        while True:
            try:
                yield loop.run_until_complete(anext(async_gen))
            except StopAsyncIteration:
                break
    finally:
        loop.close()

@app.route('/pilot', methods=['POST', 'OPTIONS'])
def pilot():
    if request.method == 'OPTIONS':
        response = Response()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        response.headers['Access-Control-Max-Age'] = '86400'
        return response, 200

    data = request.get_json() or {}
    user_input = data.get("user_input", "")
    user_id = data.get("user_id", "thefusionlab")
    thread_id = data.get("thread_id", "pilot_thread")

    print("Headers:", request.headers)
    print("Pilot API called!")

    def generate():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        async_gen = convo_stream(user_input=user_input, user_id=user_id, thread_id=thread_id, mode="pilot")
        try:
            while True:
                next_value = loop.run_until_complete(anext(async_gen))
                yield next_value
        except StopAsyncIteration:
            pass
        finally:
            loop.close()

    response = Response(generate(), mimetype='text/event-stream')
    response.headers['X-Accel-Buffering'] = 'no'  # Disable buffering for Nginx
    response.headers['Cache-Control'] = 'no-cache'  # Prevent caching
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

@app.route('/training', methods=['POST', 'OPTIONS'])
def training():
    if request.method == 'OPTIONS':
        response = Response()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        response.headers['Access-Control-Max-Age'] = '86400'
        return response, 200

    data = request.get_json() or {}
    user_input = data.get("user_input", "")
    user_id = data.get("user_id", "thefusionlab")
    thread_id = data.get("thread_id", "training_thread")

    print("Headers:", request.headers)
    print("Training API called!")

    def generate():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        async_gen = convo_stream(user_input=user_input, user_id=user_id, thread_id=thread_id, mode="training")
        try:
            while True:
                next_value = loop.run_until_complete(anext(async_gen))
                yield next_value
        except StopAsyncIteration:
            pass
        finally:
            loop.close()

    response = Response(generate(), mimetype='text/event-stream')
    response.headers['X-Accel-Buffering'] = 'no'
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

@app.route('/havefun', methods=['POST', 'OPTIONS'])
def havefun():
    if request.method == 'OPTIONS':
        response = Response()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        response.headers['Access-Control-Max-Age'] = '86400'
        return response, 200

    data = request.get_json() or {}
    user_input = data.get("user_input", "")
    user_id = data.get("user_id", "thefusionlab")
    thread_id = data.get("thread_id", "chat_thread")

    print("Headers:", request.headers)
    print("Fun API called!")

    async_gen = convo_stream(user_input=user_input, user_id=user_id, thread_id=thread_id, mode="funny")
    sync_gen = async_to_sync_generator(async_gen)

    response = Response(sync_gen, mimetype='text/event-stream')
    response.headers['X-Accel-Buffering'] = 'no'
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

@app.route('/')
def home():
    return "Hello, It's me Ami!"

@app.route('/ping', methods=['POST'])
def ping():
    return "Pong"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, threaded=True)  # Use threaded mode for better handling