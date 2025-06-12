# /conversation/learning Endpoint Documentation

## Overview

The `/conversation/learning` endpoint is a learning-based conversation system that provides AI-powered responses with active learning capabilities. It supports both HTTP Server-Sent Events (SSE) streaming and WebSocket real-time communication.

## Endpoint Details

- **URL**: `/conversation/learning`
- **Method**: `POST`
- **Content-Type**: `application/json`
- **Response**: `text/event-stream` (SSE format)

## Request Format

### Request Model (`ConversationLearningRequest`)

```json
{
  "user_input": "string",           // Required: The user's message/question
  "user_id": "string",              // Optional: User identifier (default: "learner")
  "thread_id": "string",            // Optional: Conversation thread ID (default: "learning_thread")
  "graph_version_id": "string",     // Optional: Knowledge graph version (default: "")
  "use_websocket": boolean          // Optional: Enable WebSocket events (default: false)
}
```

### Example Request

```json
{
  "user_input": "Explain machine learning concepts",
  "user_id": "student_123",
  "thread_id": "ml_learning_session",
  "graph_version_id": "v2.1",
  "use_websocket": true
}
```

## Response Format

The endpoint returns Server-Sent Events (SSE) stream with the following characteristics:

### SSE Response Structure

Each event follows the SSE format:
```
data: {"key": "value"}

```

### Response Types

#### 1. Processing Status (WebSocket Mode)
```json
{
  "status": "processing",
  "message": "Request is being processed and results will be sent via WebSocket",
  "thread_id": "learning_thread",
  "process_id": "uuid-string"
}
```

#### 2. Learning Response Data
```json
{
  "message": "AI response content",
  "thread_id": "learning_thread",
  "timestamp": "2024-01-01T12:00:00Z",
  "learning_metadata": {
    "similarity_score": 0.85,
    "knowledge_updated": true,
    "learning_intent": "concept_explanation"
  }
}
```

#### 3. Error Response
```json
{
  "error": "Error message description",
  "thread_id": "learning_thread",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

## Frontend Integration

### 1. Basic SSE Implementation

```javascript
async function startLearningConversation(userInput, options = {}) {
  const requestBody = {
    user_input: userInput,
    user_id: options.userId || "learner",
    thread_id: options.threadId || "learning_thread",
    graph_version_id: options.graphVersionId || "",
    use_websocket: options.useWebSocket || false
  };

  try {
    const response = await fetch('/conversation/learning', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'text/event-stream'
      },
      body: JSON.stringify(requestBody)
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value);
      const lines = chunk.split('\n');

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const jsonStr = line.slice(6); // Remove 'data: ' prefix
          try {
            const data = JSON.parse(jsonStr);
            handleLearningResponse(data);
          } catch (e) {
            console.error('Failed to parse SSE data:', e);
          }
        }
      }
    }
  } catch (error) {
    console.error('Learning conversation error:', error);
    handleError(error);
  }
}

function handleLearningResponse(data) {
  if (data.error) {
    console.error('Learning error:', data.error);
    displayError(data.error);
  } else if (data.status === 'processing') {
    showProcessingIndicator(data.message);
  } else if (data.message) {
    displayAIResponse(data.message, data.learning_metadata);
  }
}
```

### 2. React Hook Implementation

```jsx
import { useState, useEffect, useCallback } from 'react';

export const useLearningConversation = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [responses, setResponses] = useState([]);
  const [error, setError] = useState(null);

  const sendMessage = useCallback(async (userInput, options = {}) => {
    setIsLoading(true);
    setError(null);

    const requestBody = {
      user_input: userInput,
      user_id: options.userId || "learner",
      thread_id: options.threadId || `learning_${Date.now()}`,
      graph_version_id: options.graphVersionId || "",
      use_websocket: options.useWebSocket || false
    };

    try {
      const response = await fetch('/conversation/learning', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'text/event-stream'
        },
        body: JSON.stringify(requestBody)
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              
              if (data.error) {
                setError(data.error);
              } else if (data.message) {
                setResponses(prev => [...prev, {
                  id: Date.now(),
                  content: data.message,
                  timestamp: new Date(),
                  metadata: data.learning_metadata
                }]);
              }
            } catch (e) {
              console.error('Parse error:', e);
            }
          }
        }
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  }, []);

  return {
    sendMessage,
    responses,
    isLoading,
    error,
    clearResponses: () => setResponses([])
  };
};
```

### 3. WebSocket Integration

When `use_websocket: true`, the endpoint coordinates with WebSocket events:

```javascript
// Initialize WebSocket connection first
const socket = io('your-websocket-server');

socket.on('connect', () => {
  // Register for learning events
  socket.emit('register_session', {
    thread_id: 'learning_thread',
    user_id: 'learner'
  });
});

// Listen for learning-specific events
socket.on('learning_intent_event', (data) => {
  console.log('Learning intent detected:', data);
  displayLearningIntent(data);
});

socket.on('learning_knowledge_event', (data) => {
  console.log('Knowledge update:', data);
  updateKnowledgeDisplay(data);
});

// Then make the HTTP request with WebSocket enabled
async function sendLearningMessage(userInput) {
  const response = await fetch('/conversation/learning', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      user_input: userInput,
      thread_id: 'learning_thread',
      use_websocket: true  // Enable WebSocket coordination
    })
  });

  // Handle the initial HTTP response
  const reader = response.body.getReader();
  // ... SSE handling code
}
```

## Advanced Features

### Thread Management

The endpoint uses thread-based isolation with asyncio locks:

- Each `thread_id` gets its own processing lock
- Multiple threads can run in parallel
- 180-second timeout prevents hanging requests
- Automatic lock cleanup for unused threads

### Error Handling

The endpoint provides comprehensive error handling:

```javascript
function handleStreamError(error) {
  if (error.message.includes('timeout')) {
    displayMessage('Server is busy, please try again');
  } else if (error.message.includes('HTTP 5')) {
    displayMessage('Server error, please contact support');
  } else {
    displayMessage(`Connection error: ${error.message}`);
  }
}
```

### Request Options

```javascript
const advancedOptions = {
  userId: "student_123",
  threadId: `learning_session_${Date.now()}`,
  graphVersionId: "v2.1",
  useWebSocket: true,
  timeout: 180000  // 3 minutes
};
```

## Best Practices

1. **Thread ID Management**: Use unique thread IDs for different conversation sessions
2. **Error Handling**: Always implement proper error handling for network issues
3. **Loading States**: Show loading indicators during processing
4. **WebSocket Coordination**: Use WebSockets for real-time learning events
5. **Resource Cleanup**: Close SSE streams and WebSocket connections when done
6. **Timeout Handling**: Implement client-side timeouts (3+ minutes recommended)

## Example Complete Implementation

```html
<!DOCTYPE html>
<html>
<head>
    <title>Learning Conversation</title>
</head>
<body>
    <div id="chat-container">
        <div id="messages"></div>
        <input type="text" id="user-input" placeholder="Ask a learning question...">
        <button onclick="sendMessage()">Send</button>
        <div id="loading" style="display: none;">Processing...</div>
    </div>

    <script>
        let currentThreadId = `learning_${Date.now()}`;

        async function sendMessage() {
            const input = document.getElementById('user-input');
            const userInput = input.value.trim();
            if (!userInput) return;

            // Clear input and show loading
            input.value = '';
            document.getElementById('loading').style.display = 'block';

            // Display user message
            addMessage('user', userInput);

            try {
                const response = await fetch('/conversation/learning', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        user_input: userInput,
                        thread_id: currentThreadId,
                        user_id: 'web_user',
                        use_websocket: false
                    })
                });

                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let aiMessage = '';

                while (true) {
                    const { value, done } = await reader.read();
                    if (done) break;

                    const chunk = decoder.decode(value);
                    const lines = chunk.split('\n');

                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            try {
                                const data = JSON.parse(line.slice(6));
                                if (data.error) {
                                    addMessage('error', data.error);
                                } else if (data.message) {
                                    aiMessage += data.message;
                                    updateLastMessage('ai', aiMessage);
                                }
                            } catch (e) {
                                console.error('Parse error:', e);
                            }
                        }
                    }
                }
            } catch (error) {
                addMessage('error', `Connection error: ${error.message}`);
            } finally {
                document.getElementById('loading').style.display = 'none';
            }
        }

        function addMessage(type, content) {
            const messages = document.getElementById('messages');
            const div = document.createElement('div');
            div.className = `message ${type}`;
            div.textContent = content;
            messages.appendChild(div);
            messages.scrollTop = messages.scrollHeight;
        }

        function updateLastMessage(type, content) {
            const messages = document.getElementById('messages');
            let lastAiMessage = messages.querySelector(`.message.${type}:last-child`);
            
            if (!lastAiMessage) {
                addMessage(type, content);
            } else {
                lastAiMessage.textContent = content;
            }
        }

        // Allow Enter key to send message
        document.getElementById('user-input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendMessage();
        });
    </script>
</body>
</html>
```

This documentation provides a comprehensive guide for integrating with the `/conversation/learning` endpoint, covering both basic usage and advanced features like WebSocket integration and proper error handling. 