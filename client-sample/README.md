# WebSocket Session Monitoring

This directory contains client-side utilities for reliable WebSocket session management with the AMI backend.

## Purpose

These utilities help solve issues with WebSocket connections by:

1. **Session Persistence** - Maintaining active sessions with the server through periodic pings
2. **Transport Monitoring** - Logging the transport type (polling or WebSocket) being used
3. **Connection Recovery** - Automatically reconnecting and re-registering sessions when disconnected
4. **Activity Tracking** - Keeping the server updated with client activity to prevent session cleanup

## Files

- `session_monitor.js` - Core utility class for monitoring WebSocket sessions
- `example-usage.js` - Example implementation showing how to use the SessionMonitor

## How to Use

### 1. Setup Socket.IO Connection

First, initialize a Socket.IO connection with the proper configuration:

```javascript
const socket = io('http://your-server:5001', {
  transports: ['polling', 'websocket'], // Allow polling, then upgrade
  reconnectionAttempts: 5,
  reconnectionDelay: 1000,
  timeout: 10000,
  forceNew: true,
  path: '/socket.io'
});
```

### 2. Initialize Session Monitor

Create a session monitor instance to keep the session active:

```javascript
const monitor = new SessionMonitor(socket, {
  thread_id: 'your_conversation_thread_id', // Optional, can be set later
  pingInterval: 15000,  // Send ping every 15 seconds
  debug: true           // Enable debug logging
});
```

### 3. Start Monitoring

Start monitoring when your conversation begins:

```javascript
// With thread ID
monitor.start('your_conversation_thread_id');

// Or if thread_id was already set in constructor
monitor.start();
```

### 4. Send Messages with WebSocket Support

When sending messages to the server, enable WebSocket support:

```javascript
// Using fetch with the /havefun endpoint
fetch('/havefun', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    user_input: 'Your message',
    thread_id: 'your_conversation_thread_id',
    use_websocket: true  // Enable WebSocket for this conversation
  })
});
```

### 5. Cleanup When Done

When the conversation ends, stop monitoring and disconnect:

```javascript
// Stop monitoring
monitor.stop();

// Disconnect the socket if needed
socket.disconnect();
```

## Troubleshooting

If you're experiencing issues with WebSocket connections:

1. Check the browser console for connection and transport logs
2. Verify that the server is receiving pings (check server logs with [SESSION_TRACE])
3. Use the `/debug/thread-sessions?thread_id=your_thread_id` endpoint to verify active sessions
4. Ensure your client is properly configured and using modern WebSocket support

For detailed session debugging, visit:
- `/debug/websocket-sessions?api_key=debug_websocket_key`
- `/debug/all-sessions`
- `/debug/thread-sessions?thread_id=your_thread_id` 