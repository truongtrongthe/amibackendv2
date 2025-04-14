#!/usr/bin/env python
"""
Debug Server Startup

This script starts the server with enhanced debugging options
to help diagnose WebSocket session issues.
"""

import os
import sys
import logging
from main import app, socketio

# Setup enhanced logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('websocket_debug.log')
    ]
)

# Set module-specific log levels
logging.getLogger('socket').setLevel(logging.DEBUG)
logging.getLogger('engineio').setLevel(logging.DEBUG)
logging.getLogger('socketio').setLevel(logging.DEBUG)
logging.getLogger('eventlet').setLevel(logging.DEBUG)

if __name__ == "__main__":
    # Get port from environment or use default
    port = int(os.environ.get('PORT', 5001))
    host = os.environ.get('HOST', '0.0.0.0')
    
    print(f"Starting SocketIO server with DEBUG logging on {host}:{port}")
    print(f"Log file: websocket_debug.log")
    
    # Start server with debug mode
    socketio.run(app, host=host, port=port, debug=True, log_output=True) 