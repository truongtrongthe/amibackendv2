# Frontend SSE Integration Guide
## Streaming LLM Endpoint `/tool/llm`

### 📋 **Quick Reference**

**Endpoint:** `POST /tool/llm`  
**Response Type:** `text/event-stream` (Server-Sent Events)  
**Content-Type:** `application/json`

---

## 🚀 **Basic Request Example**

```javascript
const response = await fetch('http://localhost:5001/tool/llm', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Accept': 'text/event-stream',
  },
  body: JSON.stringify({
    llm_provider: 'openai',
    user_query: 'What is the capital of France?',
    system_prompt: 'You are a helpful assistant.',
    model_params: {
      temperature: 0.7,
      max_tokens: 1000
    },
    enable_tools: false,  // ✅ Use this parameter (not enable_search)
    org_id: 'your-org-id',
    user_id: 'user-123'
  })
});
```

---

## 📡 **SSE Event Handling**

### **Event Types (Always the Same Structure):**

```javascript
// All events follow this exact format:
{
  "type": "event_type",        // string
  "content": "message",        // string  
  "complete": false,           // boolean
  "provider": "openai",        // string (optional)
  "success": true,             // boolean (optional)
  "execution_time": 1.23,      // number (optional)
  "metadata": {...}            // object (optional)
}
```

### **Event Types to Handle:**

| Event Type | Description | Action |
|------------|-------------|---------|
| `status` | Processing started | Show loading indicator |
| `response_chunk` | Real-time content | Append to response |
| `response_complete` | Full response ready | Complete response |
| `complete` | Execution finished | Hide loading, show metadata |
| `error` | Error occurred | Show error message |

---

## 💻 **Complete Frontend Implementation**

```javascript
class StreamingLLMClient {
  constructor(baseUrl = 'http://localhost:5001') {
    this.baseUrl = baseUrl;
  }

  async streamLLMResponse(requestData, onChunk, onComplete, onError) {
    try {
      const response = await fetch(`${this.baseUrl}/tool/llm`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'text/event-stream',
        },
        body: JSON.stringify(requestData)
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        
        // Process complete lines
        const lines = buffer.split('\n');
        buffer = lines.pop(); // Keep incomplete line in buffer

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6); // Remove 'data: ' prefix
            
            if (data === '[DONE]') {
              onComplete?.();
              return;
            }

            try {
              const event = JSON.parse(data);
              this.handleSSEEvent(event, onChunk, onComplete, onError);
            } catch (parseError) {
              console.warn('Failed to parse SSE data:', data);
            }
          }
        }
      }
    } catch (error) {
      onError?.(error.message);
    }
  }

  handleSSEEvent(event, onChunk, onComplete, onError) {
    switch (event.type) {
      case 'status':
        console.log('Status:', event.content);
        onChunk?.({
          type: 'status',
          content: event.content,
          isLoading: true
        });
        break;

      case 'response_chunk':
        onChunk?.({
          type: 'content',
          content: event.content,
          isLoading: true
        });
        break;

      case 'response_complete':
        onChunk?.({
          type: 'complete',
          content: event.content,
          isLoading: false
        });
        break;

      case 'complete':
        onComplete?.({
          executionTime: event.execution_time,
          provider: event.provider,
          model: event.model_used,
          metadata: event.metadata
        });
        break;

      case 'error':
        onError?.(event.content);
        break;

      default:
        console.warn('Unknown event type:', event.type);
    }
  }
}
```

---

## 🎯 **Usage Example**

```javascript
const client = new StreamingLLMClient();
let fullResponse = '';

// Request configuration
const requestData = {
  llm_provider: 'openai',
  user_query: 'Explain quantum computing',
  enable_tools: false,  // ✅ Disable search for faster response
  system_prompt: 'You are a helpful AI assistant.',
  model_params: {
    temperature: 0.7,
    max_tokens: 1000
  }
};

// Event handlers
const onChunk = (chunk) => {
  switch (chunk.type) {
    case 'status':
      updateUI('status', chunk.content);
      break;
    case 'content':
      fullResponse += chunk.content;
      updateUI('content', fullResponse);
      break;
    case 'complete':
      updateUI('complete', fullResponse);
      break;
  }
};

const onComplete = (metadata) => {
  console.log('Stream completed:', metadata);
  hideLoadingIndicator();
};

const onError = (error) => {
  console.error('Stream error:', error);
  showErrorMessage(error);
};

// Start streaming
client.streamLLMResponse(requestData, onChunk, onComplete, onError);
```

---

## ⚙️ **Tool Control Parameters**

```javascript
// ✅ CORRECT - Use enable_tools
{
  "enable_tools": false,     // Disable all tools (faster, knowledge-based)
  "force_tools": false,      // Don't force tool usage
  "tools_whitelist": null    // Allow all tools (when enabled)
}

// ❌ DEPRECATED - Don't use enable_search (legacy support only)
{
  "enable_search": false     // Still works but prefer enable_tools
}
```

### **Tool Control Options:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_tools` | `true` | Master switch for all tools |
| `force_tools` | `false` | Force LLM to use tools |
| `tools_whitelist` | `null` | Restrict specific tools |

---

## 🚨 **Common Issues & Solutions**

### **Issue 1: No Response Received**
```javascript
// ✅ Solution: Check response status
if (!response.ok) {
  throw new Error(`HTTP ${response.status}`);
}
```

### **Issue 2: Parsing Errors**
```javascript
// ✅ Solution: Handle malformed JSON gracefully
try {
  const event = JSON.parse(data);
} catch (error) {
  console.warn('Parse error:', data);
  continue; // Skip malformed data
}
```

### **Issue 3: Connection Drops**
```javascript
// ✅ Solution: Implement retry logic
const retryStream = async (retries = 3) => {
  for (let i = 0; i < retries; i++) {
    try {
      await client.streamLLMResponse(...);
      break;
    } catch (error) {
      if (i === retries - 1) throw error;
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
  }
};
```

### **Issue 4: Memory Leaks**
```javascript
// ✅ Solution: Clean up resources
const controller = new AbortController();

fetch(url, { 
  signal: controller.signal,
  // ... other options
});

// Later: controller.abort(); // Cancel request
```

---

## 🔧 **React Hook Example**

```javascript
import { useState, useCallback } from 'react';

const useStreamingLLM = () => {
  const [response, setResponse] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [metadata, setMetadata] = useState(null);

  const streamRequest = useCallback(async (requestData) => {
    setResponse('');
    setIsLoading(true);
    setError(null);
    setMetadata(null);

    const client = new StreamingLLMClient();
    
    try {
      await client.streamLLMResponse(
        requestData,
        (chunk) => {
          if (chunk.type === 'content') {
            setResponse(prev => prev + chunk.content);
          }
        },
        (meta) => {
          setIsLoading(false);
          setMetadata(meta);
        },
        (err) => {
          setError(err);
          setIsLoading(false);
        }
      );
    } catch (err) {
      setError(err.message);
      setIsLoading(false);
    }
  }, []);

  return { response, isLoading, error, metadata, streamRequest };
};
```

---

## 📊 **Performance Tips**

1. **Disable tools for simple questions** (`enable_tools: false`)
2. **Use appropriate timeouts** (30-60 seconds)
3. **Implement response chunking** for large responses
4. **Cache responses** for repeated queries
5. **Show progress indicators** during streaming

---

## ✅ **Testing Your Integration**

```bash
# Test with tools disabled (fast)
curl -X POST "http://localhost:5001/tool/llm" \
  -H "Content-Type: application/json" \
  -d '{"llm_provider": "openai", "user_query": "What is 2+2?", "enable_tools": false}'

# Test with tools enabled (slower, with search)
curl -X POST "http://localhost:5001/tool/llm" \
  -H "Content-Type: application/json" \
  -d '{"llm_provider": "openai", "user_query": "Latest AI news", "enable_tools": true}'
```

**Expected Response:** SSE stream with identical format for both tests.

---

## 🎯 **Key Takeaways for Frontend Team**

1. **SSE format is ALWAYS identical** - same parsing logic works for all cases
2. **Use `enable_tools` parameter** - not `enable_search`
3. **Handle all 5 event types** - status, response_chunk, response_complete, complete, error
4. **Implement proper error handling** - network issues, parsing errors
5. **Show loading states** - users expect real-time feedback
6. **Test both tool modes** - disabled (fast) vs enabled (slower with search) 