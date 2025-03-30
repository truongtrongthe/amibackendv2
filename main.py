# main.py
# Purpose: Flask app with /pilot for Copilot Mode and /learning for Training Mode
# Date: March 23, 2025

from flask import Flask, Response, request
from flask_cors import CORS
from ami import convo_stream  # Unified stream function from ami.py
import asyncio
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes, all origins

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
    user_id = data.get("user_id", "thefusionlab")  # Default user_id
    thread_id = data.get("thread_id", "pilot_thread")

    print("Headers:", request.headers)
    print("Pilot API called!")

    # Define a generator to handle the async convo_stream
    def generate():
        # Run the async generator synchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        async_gen = convo_stream(user_input=user_input, user_id=user_id, thread_id=thread_id,mode="pilot")
        
        try:
            while True:
                # Get the next value from the async generator
                next_value = loop.run_until_complete(anext(async_gen))
                yield next_value
        except StopAsyncIteration:
            # End of stream
            pass
        finally:
            loop.close()

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'X-Accel-Buffering': 'no',
            'Access-Control-Allow-Origin': '*'
        }
    )
@app.route('/training', methods=['POST', 'OPTIONS'])
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
    user_id = data.get("user_id", "thefusionlab")  # Default user_id
    thread_id = data.get("thread_id", "training_thread")

    print("Headers:", request.headers)
    print("Learning API called!")

    # Define a generator to handle the async convo_stream
    def generate():
        # Run the async generator synchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        async_gen = convo_stream(user_input=user_input, user_id=user_id, thread_id=thread_id)
        
        try:
            while True:
                # Get the next value from the async generator
                next_value = loop.run_until_complete(anext(async_gen))
                yield next_value
        except StopAsyncIteration:
            # End of stream
            pass
        finally:
            loop.close()

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'X-Accel-Buffering': 'no',
            'Access-Control-Allow-Origin': '*'
        }
    )

@app.route('/havefun', methods=['POST', 'OPTIONS'])
def funny():
    if request.method == 'OPTIONS':
        response = Response()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        response.headers['Access-Control-Max-Age'] = '86400'
        return response, 200

    data = request.get_json() or {}
    user_input = data.get("user_input", "")
    user_id = data.get("user_id", "thefusionlab")  # Default user_id
    thread_id = data.get("thread_id", "chat_thread")

    print("Headers:", request.headers)
    print("Fun API called!")

    # Define a generator to handle the async convo_stream
    def generate():
        # Run the async generator synchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        async_gen = convo_stream(user_input=user_input, user_id=user_id, thread_id=thread_id,mode="funny")
        
        try:
            while True:
                # Get the next value from the async generator
                next_value = loop.run_until_complete(anext(async_gen))
                yield next_value
        except StopAsyncIteration:
            # End of stream
            pass
        finally:
            loop.close()

    return Response(
        generate(),
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