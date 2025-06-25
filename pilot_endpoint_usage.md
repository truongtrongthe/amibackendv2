# /conversation/pilot Endpoint Usage

## Endpoint
```
POST /conversation/pilot
Content-Type: application/json
```

## Request Example
```javascript
const response = await fetch('/conversation/pilot', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    "user_input": "Hello! How are you?",
    "user_id": "web_user",
    "thread_id": "pilot_session_123"
  })
});
```

## Response (Server-Sent Events)
The response streams in real-time:

```javascript
// Streaming chunks
data: {"status": "streaming", "content": "Hello! I'm", "complete": false}
data: {"status": "streaming", "content": " doing great,", "complete": false}
data: {"status": "streaming", "content": " thanks for asking!", "complete": false}

// Final response
data: {"type": "response_complete", "status": "success", "message": "Hello! I'm doing great, thanks for asking!", "complete": true, "metadata": {"pilot_mode": true, "knowledge_saving": false}}
```

## cURL Example
```bash
curl -X POST http://localhost:5001/conversation/pilot \
  -H "Content-Type: application/json" \
  -d '{
    "user_input": "Hello! How are you?",
    "user_id": "web_user",
    "thread_id": "pilot_session_123"
  }'
```

## Required Fields
- `user_input` (string) - The user's message

## Optional Fields
- `user_id` (string) - Default: "pilot_user"
- `thread_id` (string) - Default: "pilot_thread"
- `use_websocket` (boolean) - Default: false
- `org_id` (string) - Default: "unknown" 