from flask import Flask, Response, request
from flask_cors import CORS  # Import CORS
from amilearn import event_stream
from copilot import pilot_stream
from learning import learning_stream
app = Flask(__name__)

# Enable CORS for all routes and allow all origins

CORS(app)


@app.route('/copilot', methods=['POST', 'OPTIONS'])
def ami_copilot():
    if request.method == 'OPTIONS':
        response = Response()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        response.headers['Access-Control-Max-Age'] = '86400'  # Cache preflight for 1 day
        return response, 200  # Fast response for preflight
   
    data = request.get_json()
    user_input = data.get("user_input")
    user_id = data.get("user_id", "tfl")
    thread_id = data.get("thread_id", "copilot_thread")
    
    # Log headers to check for OPTIONS requests
    print("Headers:", request.headers)
    print("Copilot API called!")

    return Response(
        pilot_stream(user_input, user_id, thread_id), 
        mimetype='text/event-stream', 
        headers={'X-Accel-Buffering': 'no',
                 'Access-Control-Allow-Origin': '*'}
    )



@app.route('/ami-learn', methods=['POST','OPTIONS'])
def ami_learn():
    if request.method == 'OPTIONS':
        response = Response()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        response.headers['Access-Control-Max-Age'] = '86400'  # Cache preflight for 1 day
        return response, 200  # Fast response for preflight
    data = request.get_json()
    user_input = data.get("user_input")
    user_id ="tfl"
    thread_id = data.get("thread_id", "global_thread")  # Optional thread_id from client
    return Response(
        event_stream(user_input, user_id, thread_id), 
        mimetype='text/event-stream', 
        headers={'X-Accel-Buffering': 'no','Access-Control-Allow-Origin': '*'}
        )  # Disable buffering for Nginx (if used))


@app.route('/learning', methods=['POST','OPTIONS'])
def learn():
    if request.method == 'OPTIONS':
        response = Response()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        response.headers['Access-Control-Max-Age'] = '86400'  # Cache preflight for 1 day
        return response, 200  # Fast response for preflight
    data = request.get_json()
    user_input = data.get("user_input")
    user_id ="tfl"
    thread_id = data.get("thread_id", "global_thread")  # Optional thread_id from client
    return Response(
        learning_stream(user_input, user_id, thread_id), 
        mimetype='text/event-stream', 
        headers={'X-Accel-Buffering': 'no','Access-Control-Allow-Origin': '*'}
        )  # Disable buffering for Nginx (if used))


# Middleware to log headers for debugging
@app.route('/')
def home():
    return "Hello, It's me Ami!"

@app.route('/ping', methods=['POST'])
def ping():
    return "Pong"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)