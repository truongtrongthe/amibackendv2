# /conversation/pilot Endpoint - React Integration Guide

## Quick Start

**Endpoint:** `POST /conversation/pilot`  
**Response:** Server-Sent Events (SSE) streaming

## Basic React Hook

```jsx
import { useState, useCallback } from 'react';

export const usePilotChat = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [messages, setMessages] = useState([]);
  const [error, setError] = useState(null);

  const sendMessage = useCallback(async (userInput, threadId = `pilot_${Date.now()}`) => {
    setIsLoading(true);
    setError(null);

    // Add user message immediately
    const userMessage = { id: Date.now(), type: 'user', content: userInput };
    setMessages(prev => [...prev, userMessage]);

    try {
      const response = await fetch('/conversation/pilot', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_input: userInput,
          thread_id: threadId,
          user_id: 'web_user'
        })
      });

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let aiResponse = '';

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
                // Update AI message in real-time
                setMessages(prev => {
                  const newMessages = [...prev];
                  const lastMessage = newMessages[newMessages.length - 1];
                  if (lastMessage?.type === 'ai') {
                    lastMessage.content = aiResponse;
                  } else {
                    newMessages.push({ id: Date.now(), type: 'ai', content: aiResponse });
                  }
                  return newMessages;
                });
              } else if (data.type === 'response_complete') {
                // Final response
                setMessages(prev => {
                  const newMessages = [...prev];
                  const lastMessage = newMessages[newMessages.length - 1];
                  if (lastMessage?.type === 'ai') {
                    lastMessage.content = data.message;
                    lastMessage.complete = true;
                  }
                  return newMessages;
                });
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

  return { sendMessage, messages, isLoading, error, clearMessages: () => setMessages([]) };
};
```

## Basic Chat Component

```jsx
import React, { useState } from 'react';
import { usePilotChat } from './usePilotChat';

const PilotChat = () => {
  const [input, setInput] = useState('');
  const { sendMessage, messages, isLoading, error } = usePilotChat();

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;
    
    sendMessage(input);
    setInput('');
  };

  return (
    <div className="chat-container">
      {/* Messages */}
      <div className="messages">
        {messages.map((msg) => (
          <div key={msg.id} className={`message ${msg.type}`}>
            {msg.content}
          </div>
        ))}
        {error && <div className="error">Error: {error}</div>}
      </div>

      {/* Input */}
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type your message..."
          disabled={isLoading}
        />
        <button type="submit" disabled={isLoading || !input.trim()}>
          {isLoading ? 'Sending...' : 'Send'}
        </button>
      </form>
    </div>
  );
};

export default PilotChat;
```

## Request Format

```javascript
{
  "user_input": "Hello! How are you?",           // Required
  "user_id": "web_user",                        // Optional
  "thread_id": "pilot_session_123",             // Optional  
  "use_websocket": false                        // Optional
}
```

## Response Format

**Streaming chunks:**
```javascript
{ "status": "streaming", "content": "Hello! I'm", "complete": false }
{ "status": "streaming", "content": " doing great,", "complete": false }
```

**Final response:**
```javascript
{
  "type": "response_complete",
  "status": "success", 
  "message": "Hello! I'm doing great, thanks for asking!",
  "complete": true,
  "metadata": {
    "pilot_mode": true,
    "knowledge_saving": false
  }
}
```

## Simple CSS

```css
.chat-container {
  max-width: 600px;
  margin: 0 auto;
  border: 1px solid #ddd;
  border-radius: 8px;
  overflow: hidden;
}

.messages {
  height: 400px;
  overflow-y: auto;
  padding: 16px;
  background: #f9f9f9;
}

.message {
  margin-bottom: 12px;
  padding: 8px 12px;
  border-radius: 16px;
  max-width: 80%;
}

.message.user {
  background: #007bff;
  color: white;
  margin-left: auto;
  text-align: right;
}

.message.ai {
  background: white;
  border: 1px solid #eee;
}

.error {
  background: #ffebee;
  color: #c62828;
  padding: 8px;
  border-radius: 4px;
}

form {
  display: flex;
  padding: 16px;
  background: white;
}

input {
  flex: 1;
  padding: 12px;
  border: 1px solid #ddd;
  border-radius: 4px;
  margin-right: 8px;
}

button {
  padding: 12px 20px;
  background: #007bff;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}

button:disabled {
  background: #ccc;
  cursor: not-allowed;
}
```

## That's it! 

Just use the `PilotChat` component in your app and you'll have a working chat interface with the pilot endpoint. The responses stream in real-time and the pilot mode means no knowledge is saved. 