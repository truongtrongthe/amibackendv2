# LLM Tool Execution API Documentation

## Overview

The `/api/llm/execute` endpoint allows you to execute LLM tool calling with dynamic system prompts and parameters. It supports both Anthropic Claude and OpenAI GPT-4 with customizable settings.

## Endpoint

```
POST /api/llm/execute
```

## Request Body

```json
{
  "llm_provider": "openai",          // Required: "anthropic" or "openai"
  "user_query": "Your question here", // Required: The user's input query
  "system_prompt": "Custom prompt",   // Optional: Custom system prompt
  "model_params": {                   // Optional: Model-specific parameters
    "temperature": 0.7,
    "max_tokens": 500,
    "top_p": 1.0
  },
  "org_id": "your_org",              // Optional: Organization ID (default: "default")
  "user_id": "your_user"             // Optional: User ID (default: "anonymous")
}
```

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `llm_provider` | string | Yes | LLM provider: `"anthropic"` or `"openai"` |
| `user_query` | string | Yes | The user's input query |
| `system_prompt` | string | No | Custom system prompt to override defaults |
| `model_params` | object | No | Model-specific parameters |
| `org_id` | string | No | Organization identifier |
| `user_id` | string | No | User identifier |

### Model Parameters

For **OpenAI**:
- `temperature`: Number (0.0-2.0) - Controls randomness
- `max_tokens`: Integer - Maximum tokens in response
- `top_p`: Number (0.0-1.0) - Controls diversity

For **Anthropic**:
- System prompts are handled differently (prepended to user query)
- Model parameters are not currently customizable

## Response Format

### Success Response (200)

```json
{
  "success": true,
  "result": "The LLM's response text here...",
  "provider": "openai",
  "model_used": "gpt-4-1106-preview",
  "execution_time": 2.34,
  "request_id": "abc12345",
  "total_elapsed_time": 2.45,
  "metadata": {
    "org_id": "your_org",
    "user_id": "your_user",
    "tools_used": ["search"]
  },
  "error": null
}
```

### Error Response (400/500)

```json
{
  "success": false,
  "result": "",
  "provider": "openai",
  "model_used": "gpt-4-1106-preview",
  "execution_time": 0.5,
  "request_id": "req_123456",
  "total_elapsed_time": 0.7,
  "error": "Error message here",
  "metadata": null
}
```

## Brain Vectors API

The `/brain-vectors` endpoint allows you to fetch all AI synthesis vectors from the knowledge base for display in the frontend.

### Endpoint

```
POST /brain-vectors
```

### Request Body

```json
{
  "namespace": "conversation",      // Optional: Pinecone namespace (default: "conversation")
  "max_vectors": 1000,             // Optional: Maximum number of vectors to fetch (default: 1000)
  "batch_size": 100,               // Optional: Batch size for fetching (default: 100)
  "include_metadata": true,        // Optional: Include full metadata (default: true)
  "include_content": true,         // Optional: Include content preview (default: true)
  "content_preview_length": 300,   // Optional: Length of content preview (default: 300)
  "org_id": "your_org_id"          // Required: Organization ID
}
```

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `namespace` | string | No | Pinecone namespace to search (default: "conversation") |
| `max_vectors` | integer | No | Maximum number of vectors to return (default: 1000) |
| `batch_size` | integer | No | Batch size for fetching vectors (default: 100) |
| `include_metadata` | boolean | No | Whether to include full metadata (default: true) |
| `include_content` | boolean | No | Whether to include content preview (default: true) |
| `content_preview_length` | integer | No | Length of content preview in characters (default: 300) |
| `org_id` | string | Yes | Organization identifier |

### Response Format

#### Success Response (200)

```json
{
  "success": true,
  "request_id": "vectors_20231215_143022",
  "namespace": "conversation",
  "generated_at": "2023-12-15T14:30:22.123456",
  "processing_time_seconds": 2.45,
  "vectors": {
    "total_count": 150,
    "returned_count": 150,
    "data": [
      {
        "id": "vector_123",
        "title": "User Learning Summary",
        "created_at": "2023-12-15T10:00:00Z",
        "confidence": 0.85,
        "source": "conversation",
        "user_id": "user123",
        "thread_id": "thread456",
        "topic": "machine_learning",
        "categories": ["ai_synthesis", "technical", "learning"],
        "score": 0.92,
        "content_preview": "This is a comprehensive summary of machine learning concepts discussed...",
        "content_truncated": true,
        "content_length": 1250,
        "metadata": {
          "raw": "Full content here...",
          "expires_at": "2024-12-15T10:00:00Z",
          "additional_metadata": "..."
        }
      }
    ]
  },
  "statistics": {
    "total_vectors": 150,
    "categories": {
      "unique_count": 12,
      "top_10": {
        "ai_synthesis": 150,
        "technical": 89,
        "learning": 45
      }
    },
    "topics": {
      "unique_count": 25,
      "top_10": {
        "machine_learning": 35,
        "programming": 28,
        "data_science": 22
      }
    },
    "content": {
      "average_length": 850.5,
      "min_length": 100,
      "max_length": 3000
    },
    "confidence": {
      "average": 0.78,
      "min": 0.2,
      "max": 0.95
    }
  },
  "query_params": {
    "namespace": "conversation",
    "max_vectors": 1000,
    "batch_size": 100,
    "include_metadata": true,
    "include_content": true,
    "content_preview_length": 300
  }
}
```

#### Error Response (400/500)

```json
{
  "success": false,
  "request_id": "vectors_20231215_143022",
  "error": "Failed to fetch AI synthesis vectors: Connection timeout",
  "processing_time_seconds": 30.0
}
```

### Frontend Integration Example

```javascript
// Fetch brain vectors for display
async function fetchBrainVectors(orgId, options = {}) {
  const requestBody = {
    org_id: orgId,
    namespace: options.namespace || "conversation",
    max_vectors: options.maxVectors || 1000,
    include_content: options.includeContent !== false,
    include_metadata: options.includeMetadata !== false,
    content_preview_length: options.previewLength || 300
  };
  
  try {
    const response = await fetch('/brain-vectors', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestBody)
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const data = await response.json();
    
    if (data.success) {
      return {
        vectors: data.vectors.data,
        statistics: data.statistics,
        totalCount: data.vectors.total_count
      };
    } else {
      throw new Error(data.error || 'Failed to fetch brain vectors');
    }
  } catch (error) {
    console.error('Error fetching brain vectors:', error);
    throw error;
  }
}

// Usage example
fetchBrainVectors('your_org_id', {
  maxVectors: 500,
  includeContent: true,
  previewLength: 200
}).then(result => {
  console.log(`Fetched ${result.totalCount} vectors`);
  console.log('Statistics:', result.statistics);
  
  // Display vectors in your UI
  result.vectors.forEach(vector => {
    console.log(`Vector: ${vector.title}`);
    console.log(`Content: ${vector.content_preview}`);
    console.log(`Categories: ${vector.categories.join(', ')}`);
  });
}).catch(error => {
  console.error('Failed to fetch brain vectors:', error);
});
```

### Notes

- The endpoint fetches vectors with the "ai_synthesis" category, which represents AI-generated summaries and insights
- Content preview is automatically truncated if it exceeds the specified length
- The `content_truncated` field indicates whether the content was truncated
- Statistics provide insights into the knowledge base composition
- All vectors include creation timestamps and confidence scores
- The endpoint supports CORS for frontend integration

## Examples

### Basic OpenAI Request

```bash
curl -X POST http://localhost:8000/api/llm/execute \
  -H "Content-Type: application/json" \
  -d '{
    "llm_provider": "openai",
    "user_query": "What is machine learning?",
    "system_prompt": "You are a helpful AI tutor."
  }'
```

### OpenAI with Custom Parameters

```bash
curl -X POST http://localhost:8000/api/llm/execute \
  -H "Content-Type: application/json" \
  -d '{
    "llm_provider": "openai",
    "user_query": "Tell me a creative story",
    "system_prompt": "You are a creative storyteller.",
    "model_params": {
      "temperature": 0.8,
      "max_tokens": 300
    }
  }'
```

### Anthropic Claude Request

```bash
curl -X POST http://localhost:8000/api/llm/execute \
  -H "Content-Type: application/json" \
  -d '{
    "llm_provider": "anthropic",
    "user_query": "Explain quantum computing",
    "system_prompt": "You are a science educator."
  }'
```

### Python Example

```python
import requests

response = requests.post(
    "http://localhost:8000/api/llm/execute",
    json={
        "llm_provider": "openai",
        "user_query": "What are the latest AI trends?",
        "system_prompt": "You are a tech analyst. Provide current insights.",
        "model_params": {
            "temperature": 0.7,
            "max_tokens": 400
        },
        "org_id": "my_org",
        "user_id": "john_doe"
    }
)

if response.status_code == 200:
    result = response.json()
    print(f"Success: {result['result']}")
else:
    print(f"Error: {response.json()}")
```

## Features

### 🔍 **Tool Integration**
- Automatic Google Search integration
- Tools are triggered based on query content
- Search results are incorporated into responses

### 🎯 **Dynamic System Prompts**
- Customize AI behavior per request
- Override default prompts
- Provider-specific handling

### ⚙️ **Model Configuration**
- Adjust temperature, max_tokens, top_p
- Provider-specific parameter support
- Fine-tune response characteristics

### 📊 **Comprehensive Logging**
- Request tracking with unique IDs
- Execution time monitoring
- Detailed error reporting

### 🔄 **Multi-Provider Support**
- Anthropic Claude Sonnet
- OpenAI GPT-4 Turbo
- Consistent API interface

## Environment Requirements

Make sure these environment variables are set:

```bash
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
SERPAPI_API_KEY=your_serpapi_key  # For search functionality
```

## Testing

Run the test script to verify the API:

```bash
python test_llm_api.py
```

This will run various test scenarios demonstrating different features and configurations.

## Error Handling

The API provides detailed error information:

- **400**: Client errors (invalid provider, missing parameters)
- **500**: Server errors (API failures, internal errors)
- Structured error responses with request tracking
- Comprehensive logging for debugging

## Rate Limiting

- Depends on your LLM provider limits
- Consider implementing client-side rate limiting
- Monitor execution times for optimization

## Best Practices

1. **System Prompts**: Be specific and clear about desired behavior
2. **Model Parameters**: Start with default values, then fine-tune
3. **Error Handling**: Always check the `success` field in responses
4. **Logging**: Use `request_id` for tracking and debugging
5. **Timeouts**: Set appropriate timeouts for your use case 