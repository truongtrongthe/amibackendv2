// Example usage of SessionMonitor for WebSocket connection management

// Import socket.io client (in browser this would be from a CDN)
// const io = require('socket.io-client');
// const { SessionMonitor } = require('./session_monitor');

/**
 * Initialize Socket.IO connection with the server
 * @param {string} serverUrl - WebSocket server URL
 * @returns {Object} - Socket.IO connection object
 */
function initializeSocketConnection(serverUrl = 'http://localhost:5001') {
  const socket = io(serverUrl, {
    transports: ['polling', 'websocket'], // Allow polling, then upgrade to WebSocket
    reconnectionAttempts: 5,              // Number of reconnection attempts
    reconnectionDelay: 1000,              // Delay between reconnection attempts (ms)
    timeout: 10000,                       // Connection timeout (ms)
    forceNew: true,                       // Force a new connection
    path: '/socket.io'                    // Socket.IO path
  });

  // Log connection events
  socket.on('connect', () => {
    console.log('Socket connected successfully');
    console.log('Socket ID:', socket.id);
    console.log('Transport used:', socket.io.engine.transport.name);
  });

  socket.on('connect_error', (error) => {
    console.error('Connection error:', error);
  });

  socket.on('disconnect', (reason) => {
    console.log('Socket disconnected:', reason);
  });

  socket.on('reconnect_attempt', (attemptNumber) => {
    console.log(`Reconnection attempt ${attemptNumber}`);
  });

  socket.on('error', (error) => {
    console.error('Socket error:', error);
  });

  socket.on('reconnect', (attemptNumber) => {
    console.log(`Reconnected after ${attemptNumber} attempts`);
    
    // After reconnection, request any missed messages
    if (threadId) {
      console.log(`Requesting missed messages for thread ${threadId}`);
      socket.emit('request_missed_messages', { thread_id: threadId });
    }
  });

  // Handle missed messages from the server
  socket.on('missed_messages', (data) => {
    console.log(`Received ${data.messages.length} missed messages for thread ${data.thread_id}`);
    
    // Process each missed message
    data.messages.forEach(message => {
      console.log('Processing missed message:', message);
      // Process as if it was a regular analysis_update
      // This would call the same handler used for real-time updates
    });
  });

  // Handle analysis updates
  socket.on('analysis_update', (data) => {
    console.log('Received analysis update:', data);
    // Update UI or process data here
  });

  return socket;
}

/**
 * Example function to initialize a WebSocket connection and session monitoring
 */
function setupLiveAnalysisConnection(threadId) {
  // Create a WebSocket connection
  const socket = initializeSocketConnection();
  
  // Initialize session monitor
  const monitor = new SessionMonitor(socket, {
    thread_id: threadId,        // The conversation thread ID
    pingInterval: 15000,        // Send ping every 15 seconds (ms)
    debug: true                 // Enable debug logging
  });
  
  // Start monitoring
  monitor.start();
  
  // Return socket and monitor for later use (e.g., cleanup)
  return { socket, monitor };
}

/**
 * Example function to send a message via the /havefun endpoint
 * with WebSocket support
 */
async function sendMessageWithWebSocket(message, threadId) {
  const response = await fetch('/havefun', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      user_input: message,
      thread_id: threadId,
      use_websocket: true  // Enable WebSocket for this conversation
    })
  });
  
  // Process streaming response
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    const chunk = decoder.decode(value);
    console.log('Received chunk:', chunk);
    // Process chunk...
  }
}

/**
 * Example usage in a web application
 */
async function startConversation() {
  const threadId = `thread_${Date.now()}`;  // Generate a unique thread ID
  
  // Set up WebSocket connection and monitoring
  const { socket, monitor } = setupLiveAnalysisConnection(threadId);
  
  // Example of sending messages
  await sendMessageWithWebSocket('Hello, this is a test message', threadId);
  
  // Later, when the conversation ends or component unmounts
  function cleanup() {
    // Stop the session monitor
    monitor.stop();
    
    // Disconnect the socket
    socket.disconnect();
  }
  
  // Return cleanup function (e.g., for React useEffect)
  return cleanup;
}

/*
// In a React component, you might use it like this:
useEffect(() => {
  let cleanupFn;
  
  async function init() {
    cleanupFn = await startConversation();
  }
  
  init();
  
  // Return cleanup function
  return () => {
    if (cleanupFn) cleanupFn();
  };
}, []);
*/ 