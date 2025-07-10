# Streaming LLM Endpoint Guide

## Overview
The `/tool/llm` endpoint provides real-time streaming responses from LLM providers (OpenAI and Anthropic) with configurable tool usage. This endpoint supports Server-Sent Events (SSE) for streaming responses and provides fine-grained control over when and how tools are used.

## New Tool Control Features 🆕

### Tool Control Parameters

The endpoint now supports several parameters to control when and how tools are used:

```json
{
  "llm_provider": "openai",  // Required: 'openai' or 'anthropic'
  "user_query": "What is the capital of France?",  // Required: User's question
  "system_prompt": "You are a helpful assistant.",  // Optional: Custom system prompt
  "model_params": {  // Optional: Model-specific parameters
    "temperature": 0.7,
    "max_tokens": 1000
  },
  "org_id": "your_org",  // Optional: Organization ID
  "user_id": "user123",  // Optional: User ID
  
  // NEW TOOL CONTROL PARAMETERS
  "enable_tools": true,  // Optional: Enable/disable all tools (default: true)
  "force_tools": false,  // Optional: Force tool usage (default: false)
  "tools_whitelist": null  // Optional: Array of allowed tools (default: null = all)
}
```

### Tool Control Options

#### 1. `enable_tools` (boolean)
- **Default**: `true`
- **Purpose**: Master switch for tool availability
- **When `false`**: No tools will be available to the LLM
- **When `true`**: Tools are available (subject to other constraints)

#### 2. `force_tools` (boolean)  
- **Default**: `false`
- **Purpose**: Force the LLM to use tools
- **When `true`**: LLM will be required to use available tools
- **When `false`**: LLM decides whether to use tools (OpenAI only)
- **Note**: Anthropic doesn't support forced tool usage directly

#### 3. `tools_whitelist` (array)
- **Default**: `null` (all tools allowed)
- **Purpose**: Restrict which tools can be used
- **Example**: `["search"]` - only allows the search tool
- **When `null`**: All available tools are allowed

### System Prompt Behavior

The system prompt now automatically adapts based on tool availability:

- **With tools enabled**: Uses prompts that mention tool capabilities
- **With tools disabled**: Uses prompts that focus on knowledge-based responses
- **Custom prompts**: Override the automatic behavior

## Usage Examples

### Example 1: Default Behavior (Tools Enabled)
```json
{
  "llm_provider": "openai",
  "user_query": "What is the current population of Tokyo?",
  "enable_tools": true
}
```
**Result**: LLM may choose to search for current data if it thinks it's needed.

### Example 2: Disable Tools (Knowledge-Only Response)
```json
{
  "llm_provider": "openai",
  "user_query": "What is the capital of France?",
  "enable_tools": false
}
```
**Result**: LLM will answer from training data without searching.

### Example 3: Force Tool Usage
```json
{
  "llm_provider": "openai",
  "user_query": "What is 2 + 2?",
  "enable_tools": true,
  "force_tools": true
}
```
**Result**: LLM will be forced to use search even for simple questions.

### Example 4: Custom System Prompt to Discourage Search
```json
{
  "llm_provider": "openai",
  "user_query": "What is the largest planet?",
  "system_prompt": "Answer directly from your training data without searching.",
  "enable_tools": true
}
```
**Result**: Tools are available but prompt discourages their use.

### Example 5: Whitelist Specific Tools
```json
{
  "llm_provider": "openai",
  "user_query": "What's the weather like?",
  "enable_tools": true,
  "tools_whitelist": ["search"]
}
```
**Result**: Only the search tool is available (useful for future expansion).

## Why Control Tool Usage?

### Common Use Cases:

1. **Performance Optimization**: Disable tools for simple questions to get faster responses
2. **Cost Management**: Avoid unnecessary API calls for factual questions
3. **User Experience**: Provide immediate answers from training data when appropriate
4. **Testing**: Force tool usage to test search functionality
5. **Compliance**: Restrict tool usage based on user permissions or data policies

### Performance Impact:

- **Tools Disabled**: ~0.5-1.0s response time
- **Tools Enabled**: ~2-8s response time (depending on search complexity)
- **Streaming Advantage**: Users see first content in ~0.5s regardless of total time

## Endpoint Details

### URL
```
POST /tool/llm
```

### Headers
```
Content-Type: application/json
Accept: text/event-stream
```

### Request Body
```json
{
  "llm_provider": "openai",
  "user_query": "Your question here",
  "system_prompt": "Optional custom system prompt",
  "model_params": {
    "temperature": 0.7,
    "max_tokens": 1000
  },
  "org_id": "your_org",
  "user_id": "user123",
  "enable_tools": true,
  "force_tools": false,
  "tools_whitelist": null
}
```

### Response Format (SSE)
```
data: {"type": "status", "content": "Starting LLM execution...", "status": "processing"}

data: {"type": "response_chunk", "content": "The capital", "complete": false}

data: {"type": "response_chunk", "content": " of France", "complete": false}

data: {"type": "response_chunk", "content": " is Paris.", "complete": false}

data: {"type": "response_complete", "content": "The capital of France is Paris.", "complete": true}

data: {"type": "complete", "content": "Execution completed", "execution_time": 1.23, "success": true}
```

## Event Types

- `status`: Processing status updates
- `response_chunk`: Real-time response content
- `response_complete`: Full response assembled
- `complete`: Execution finished with metadata
- `error`: Error occurred during processing

## Testing

Use the provided test script to see all configurations in action:

```bash
python test_streaming_endpoint.py
```

This will run 6 different test scenarios:
1. Default with tools enabled
2. Tools disabled
3. Force tools usage
4. Anthropic with tools disabled
5. Custom prompt discouraging search
6. Tools whitelist

## Comparison with Non-Streaming Endpoint

| Feature | `/tool/llm` (Streaming) | `/api/llm/execute` (Non-Streaming) |
|---------|-------------------------|-----------------------------------|
| Response Type | SSE Stream | JSON Response |
| First Content | ~0.5-1.0s | ~2-8s |
| Tool Control | ✅ Full control | ❌ Always enabled |
| System Prompt | ✅ Adaptive | ❌ Fixed |
| User Experience | ✅ Real-time | ❌ Wait for complete |
| Performance | ✅ Perceived faster | ❌ Slower perceived |

## Future Enhancements

- Support for more tool types (weather, calendar, etc.)
- User-specific tool permissions
- Tool usage analytics
- Rate limiting per tool type
- Custom tool configurations 