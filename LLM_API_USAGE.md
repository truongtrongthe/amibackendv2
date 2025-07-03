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
  "model_used": "unknown",
  "execution_time": 0,
  "request_id": "abc12345",
  "total_elapsed_time": 1.23,
  "error": "Error message here",
  "metadata": null
}
```

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