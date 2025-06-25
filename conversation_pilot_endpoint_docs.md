# /conversation/pilot Endpoint Documentation

## Overview

The `/conversation/pilot` endpoint provides a conversational AI system using the full AVA (Active Learning Assistant) framework but operating in pure conversation mode. Unlike the `/conversation/learning` endpoint, this pilot version uses AVA's conversational capabilities while completely bypassing all knowledge management, teaching intent detection, and knowledge saving features.

## Endpoint Details

- **URL**: `/conversation/pilot`
- **Method**: `POST`
- **Content-Type**: `application/json`
- **Response**: `text/event-stream` (SSE format)

## Key Differences from `/conversation/learning`

- ❌ **NO Knowledge Saving**: Does not save any conversation content to knowledge base
- ❌ **NO Teaching Intent Detection**: Skips teaching intent analysis and processing
- ❌ **NO Similarity Analysis**: Skips knowledge similarity checks and exploration
- ❌ **NO UPDATE vs CREATE Decisions**: No knowledge management decisions required
- ✅ **Uses AVA System**: Utilizes the full AVA conversational AI system
- ✅ **Pure Conversation Mode**: AVA operates in conversation-only mode
- ✅ **Conversation History**: Maintains conversation context within the session
- ✅ **Real-time Streaming**: Provides real-time response streaming via AVA
- ✅ **WebSocket Support**: Supports WebSocket coordination (optional)

## Request Format

### Request Model (`ConversationPilotRequest`)

```json
{
  "user_input": "string",           // Required: The user's message/question
  "user_id": "string",              // Optional: User identifier (default: "pilot_user")
  "thread_id": "string",            // Optional: Conversation thread ID (default: "pilot_thread")
  "graph_version_id": "string",     // Optional: Graph version (used for context only)
  "use_websocket": boolean,         // Optional: Enable WebSocket events (default: false)
  "org_id": "string"               // Optional: Organization identifier (default: "unknown")
}
```

### Example Request

```json
{
  "user_input": "Hello! Can you help me understand machine learning?",
  "user_id": "user_123",
  "thread_id": "pilot_session_001",
  "use_websocket": false
}
```

## Response Format

The endpoint returns Server-Sent Events (SSE) stream with the following format:

### SSE Response Structure

#### 1. Streaming Response Chunks
```json
{
  "status": "streaming",
  "content": "chunk of response text",
  "complete": false
}
```

#### 2. Complete Response
```json
{
  "type": "response_complete",
  "status": "success",
  "message": "Complete AI response text",
  "complete": true,
  "metadata": {
    "timestamp": "2024-01-01T12:00:00Z",
    "user_id": "user_123",
    "thread_id": "pilot_session_001",
    "mode": "pilot",
    "knowledge_saving": false,
    "org_id": "unknown"
  }
}
```

#### 3. Error Response
```json
{
  "error": "Error message description",
  "mode": "pilot"
}
```

## Frontend Integration

### 1. Basic JavaScript Implementation

```javascript
async function startPilotConversation(userInput, options = {}) {
  const requestBody = {
    user_input: userInput,
    user_id: options.userId || "pilot_user",
    thread_id: options.threadId || "pilot_thread",
    graph_version_id: options.graphVersionId || "",
    use_websocket: options.useWebSocket || false,
    org_id: options.orgId || "unknown"
  };

  try {
    const response = await fetch('/conversation/pilot', {
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
    let accumulatedResponse = '';

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value);
      const lines = chunk.split('\n');

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const jsonStr = line.slice(6);
          try {
            const data = JSON.parse(jsonStr);
            
            if (data.status === 'streaming') {
              // Handle streaming chunks
              accumulatedResponse += data.content;
              displayStreamingResponse(accumulatedResponse);
            } else if (data.type === 'response_complete') {
              // Handle complete response
              displayFinalResponse(data.message);
            } else if (data.error) {
              displayError(data.error);
            }
          } catch (e) {
            console.error('Failed to parse SSE data:', e);
          }
        }
      }
    }
  } catch (error) {
    console.error('Pilot conversation error:', error);
    displayError(error.message);
  }
}

function displayStreamingResponse(text) {
  document.getElementById('ai-response').textContent = text;
}

function displayFinalResponse(text) {
  document.getElementById('ai-response').textContent = text;
  document.getElementById('loading').style.display = 'none';
}

function displayError(error) {
  document.getElementById('error').textContent = `Error: ${error}`;
  document.getElementById('loading').style.display = 'none';
}
```

### 2. React Hook Implementation

```jsx
import { useState, useCallback } from 'react';

export const usePilotConversation = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [responses, setResponses] = useState([]);
  const [error, setError] = useState(null);

  const sendMessage = useCallback(async (userInput, options = {}) => {
    setIsLoading(true);
    setError(null);

    const requestBody = {
      user_input: userInput,
      user_id: options.userId || "pilot_user",
      thread_id: options.threadId || `pilot_${Date.now()}`,
      graph_version_id: options.graphVersionId || "",
      use_websocket: options.useWebSocket || false,
      org_id: options.orgId || "unknown"
    };

    try {
      const response = await fetch('/conversation/pilot', {
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
      let accumulatedResponse = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              
              if (data.status === 'streaming') {
                accumulatedResponse += data.content;
              } else if (data.type === 'response_complete') {
                setResponses(prev => [...prev, {
                  id: Date.now(),
                  content: data.message,
                  timestamp: new Date(),
                  mode: 'pilot'
                }]);
              } else if (data.error) {
                setError(data.error);
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

### 3. Complete HTML Example

```html
<!DOCTYPE html>
<html>
<head>
    <title>Pilot Conversation</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        #chat-container { max-width: 800px; margin: 0 auto; }
        #messages { height: 400px; overflow-y: auto; border: 1px solid #ccc; padding: 10px; margin-bottom: 10px; }
        .message { margin: 10px 0; padding: 8px; border-radius: 5px; }
        .user { background-color: #e3f2fd; text-align: right; }
        .ai { background-color: #f3e5f5; }
        .error { background-color: #ffebee; color: #c62828; }
        #input-container { display: flex; gap: 10px; }
        #user-input { flex: 1; padding: 10px; border: 1px solid #ccc; border-radius: 4px; }
        button { padding: 10px 20px; background-color: #2196F3; color: white; border: none; border-radius: 4px; cursor: pointer; }
        button:disabled { background-color: #ccc; cursor: not-allowed; }
        #loading { display: none; color: #666; font-style: italic; }
    </style>
</head>
<body>
    <div id="chat-container">
        <h1>Pilot Conversation</h1>
        <div id="messages"></div>
        <div id="loading">Processing...</div>
        <div id="input-container">
            <input type="text" id="user-input" placeholder="Type your message..." maxlength="1000">
            <button onclick="sendMessage()" id="send-btn">Send</button>
        </div>
    </div>

    <script>
        let currentThreadId = `pilot_${Date.now()}`;
        let messageId = 0;

        async function sendMessage() {
            const input = document.getElementById('user-input');
            const sendBtn = document.getElementById('send-btn');
            const loadingDiv = document.getElementById('loading');
            
            const userInput = input.value.trim();
            if (!userInput) return;

            // Disable input and show loading
            input.disabled = true;
            sendBtn.disabled = true;
            loadingDiv.style.display = 'block';

            // Display user message
            addMessage('user', userInput);

            // Clear input
            input.value = '';

            try {
                const response = await fetch('/conversation/pilot', {
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
                let aiResponse = '';
                let aiMessageElement = null;

                while (true) {
                    const { value, done } = await reader.read();
                    if (done) break;

                    const chunk = decoder.decode(value);
                    const lines = chunk.split('\n');

                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            try {
                                const data = JSON.parse(line.slice(6));
                                
                                if (data.status === 'streaming') {
                                    aiResponse += data.content;
                                    if (!aiMessageElement) {
                                        aiMessageElement = addMessage('ai', aiResponse);
                                    } else {
                                        aiMessageElement.textContent = aiResponse;
                                    }
                                } else if (data.type === 'response_complete') {
                                    if (aiMessageElement) {
                                        aiMessageElement.textContent = data.message;
                                    } else {
                                        addMessage('ai', data.message);
                                    }
                                } else if (data.error) {
                                    addMessage('error', `Error: ${data.error}`);
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
                // Re-enable input
                input.disabled = false;
                sendBtn.disabled = false;
                loadingDiv.style.display = 'none';
                input.focus();
            }
        }

        function addMessage(type, content) {
            const messages = document.getElementById('messages');
            const div = document.createElement('div');
            div.className = `message ${type}`;
            div.textContent = content;
            div.id = `message-${++messageId}`;
            messages.appendChild(div);
            messages.scrollTop = messages.scrollHeight;
            return div;
        }

        // Allow Enter key to send message
        document.getElementById('user-input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });

        // Focus on input when page loads
        document.getElementById('user-input').focus();
    </script>
</body>
</html>
```

## Use Cases

The `/conversation/pilot` endpoint is ideal for:

1. **AVA-Powered Chat Applications**: When you need advanced conversational AI without knowledge management
2. **Prototyping**: Testing AVA's conversation flows and response strategies without knowledge operations
3. **Customer Support**: Advanced customer service with AVA's conversation intelligence but no data persistence
4. **Personal Assistants**: Sophisticated helper applications with AVA's capabilities but no learning/memory
5. **Educational Tools**: Interactive learning with AVA's advanced responses but no persistent storage
6. **Demos**: Showcasing AVA's full conversational capabilities without backend complexity
7. **Production Chat**: When you need AVA's advanced conversation handling but want to control knowledge saving separately

## Performance Characteristics

- **Faster Response Times**: Uses AVA but skips knowledge lookup and similarity analysis
- **Lower Resource Usage**: Full AVA processing but without vector operations and knowledge management
- **Cleaner Logs**: AVA logging but reduced complexity without knowledge saving flows
- **AVA Benefits**: Maintains AVA's advanced conversation flow detection and response strategies

## Error Handling

The endpoint provides simple error handling:

```javascript
// Handle different error scenarios
function handleError(error) {
  if (error.message.includes('timeout')) {
    displayMessage('Request timed out, please try again');
  } else if (error.message.includes('HTTP 5')) {
    displayMessage('Server error, please contact support');
  } else {
    displayMessage(`Error: ${error.message}`);
  }
}
```

## Comparison with `/conversation/learning`

| Feature | `/conversation/pilot` | `/conversation/learning` |
|---------|----------------------|-------------------------|
| Uses AVA Framework | ✅ Yes (pilot mode) | ✅ Yes (full mode) |
| Knowledge Saving | ❌ No | ✅ Yes |
| Teaching Intent Detection | ❌ Disabled | ✅ Yes |
| Similarity Analysis | ❌ Skipped | ✅ Yes |
| UPDATE vs CREATE Decisions | ❌ No | ✅ Yes |
| Conversation Flow Detection | ✅ Yes (AVA) | ✅ Yes (AVA) |
| Advanced Response Strategies | ✅ Yes (AVA) | ✅ Yes (AVA) |
| Conversation History | ✅ Yes | ✅ Yes |
| Real-time Streaming | ✅ Yes | ✅ Yes |
| WebSocket Support | ✅ Yes | ✅ Yes |
| Response Time | ✅ Faster | ⚠️ Slower |
| Use Case | AVA Chat/Demo | AVA Learning/Teaching |

## Best Practices

1. **Thread Management**: Use unique thread IDs for different conversations
2. **Error Handling**: Implement proper error handling for network issues
3. **Resource Management**: Close SSE streams when done
4. **User Experience**: Show loading states during processing
5. **Input Validation**: Validate user input before sending

This pilot endpoint provides a powerful AVA-based conversational interface perfect for applications that need advanced AI chat capabilities with sophisticated conversation handling, but without the complexity of knowledge management systems. It gives you the full power of AVA's conversational intelligence while maintaining complete control over knowledge persistence. 