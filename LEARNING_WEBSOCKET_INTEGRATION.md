# Learning WebSocket Integration Documentation

## Overview
This document describes the WebSocket integration for AVA's learning events, enabling real-time communication of learning progress to connected clients.

## Architecture

### WebSocket Events
Two new WebSocket events have been implemented:

1. **`learning_intent`** - Emitted when AVA understands human intent
2. **`learning_knowledge`** - Emitted when AVA finds relevant knowledge during active learning

### Event Flow
```
User Message → AVA Processing → Learning Events → WebSocket Emission → Client Notification
```

## Implementation Details

### 1. WebSocket Emission Functions (`socketio_manager_async.py`)

#### `emit_learning_intent_event(thread_id: str, data: Dict[str, Any]) -> bool`
- Emits learning intent events to all clients in a thread room
- Event name: `learning_intent`
- Returns `True` if delivered to active sessions, `False` otherwise

#### `emit_learning_knowledge_event(thread_id: str, data: Dict[str, Any]) -> bool`
- Emits learning knowledge discovery events
- Event name: `learning_knowledge`
- Returns `True` if delivered to active sessions, `False` otherwise

### 2. AVA Integration (`ava.py`)

#### Updated Function Signatures
```python
async def read_human_input(
    self, 
    message: str, 
    conversation_context: str, 
    user_id: str, 
    thread_id: Optional[str] = None,
    use_websocket: bool = False,  # NEW
    thread_id_for_analysis: Optional[str] = None  # NEW
) -> AsyncGenerator[Union[str, Dict], None]:
```

```python
async def _active_learning_streaming(
    self,
    message: Union[str, List],
    conversation_context: str = "",
    analysis_knowledge: Dict = None,
    user_id: str = "unknown",
    prior_data: Dict = None,
    use_websocket: bool = False,  # NEW
    thread_id_for_analysis: Optional[str] = None  # NEW
) -> AsyncGenerator[Union[str, Dict], None]:
```

#### Emission Points

##### Point 1: Knowledge Discovery
```python
# Location: read_human_input() after knowledge_explorer.explore()
learning_knowledge_event = {
    "type": "learning_knowledge",
    "thread_id": thread_id_for_analysis,
    "timestamp": datetime.now().isoformat(),
    "content": {
        "message": "Found relevant knowledge for learning",
        "similarity_score": analysis_knowledge.get("similarity", 0.0),
        "knowledge_count": len(analysis_knowledge.get("query_results", [])),
        "queries": analysis_knowledge.get("queries", []),
        "complete": False
    }
}
```

##### Point 2: Intent Understanding
```python
# Location: read_human_input() after LLM evaluation
learning_intent_event = {
    "type": "learning_intent",
    "thread_id": thread_id_for_analysis,
    "timestamp": datetime.now().isoformat(),
    "content": {
        "message": "Understanding human intent",
        "intent_type": intent_type,
        "has_teaching_intent": has_teaching_intent,
        "is_priority_topic": is_priority_topic,
        "priority_topic_name": priority_topic_name,
        "should_save_knowledge": should_save_knowledge,
        "complete": True
    }
}
```

### 3. AMI Integration (`ami.py`)

#### Updated Imports
```python
from socketio_manager_async import (
    emit_analysis_event, 
    emit_knowledge_event, 
    emit_next_action_event, 
    emit_learning_intent_event,  # NEW
    emit_learning_knowledge_event  # NEW
)
```

#### Updated Function Call
```python
async for chunk in ava.read_human_input(
    user_input,
    conversation_context,
    user_id=user_id,
    thread_id=thread_id,
    use_websocket=use_websocket,  # NEW
    thread_id_for_analysis=thread_id_for_analysis  # NEW
):
```

### 4. Main API Integration (`main.py`)

#### New Emit Functions
```python
async def emit_learning_intent_event(thread_id: str, data: Dict[str, Any]) -> bool:
    """Emit a learning intent event to all clients in a thread room"""
    if "thread_id" not in data:
        data["thread_id"] = thread_id
    return await socketio_manager_async.emit_learning_intent_event(thread_id, data)

async def emit_learning_knowledge_event(thread_id: str, data: Dict[str, Any]) -> bool:
    """Emit a learning knowledge event to all clients in a thread room"""
    if "thread_id" not in data:
        data["thread_id"] = thread_id
    return await socketio_manager_async.emit_learning_knowledge_event(thread_id, data)
```

## Event Structures

### Learning Intent Event
```json
{
  "type": "learning_intent",
  "thread_id": "thread_123",
  "timestamp": "2024-01-15T10:30:00.000Z",
  "content": {
    "message": "Understanding human intent",
    "intent_type": "teaching",
    "has_teaching_intent": true,
    "is_priority_topic": false,
    "priority_topic_name": "machine_learning",
    "should_save_knowledge": true,
    "complete": true
  }
}
```

### Learning Knowledge Event
```json
{
  "type": "learning_knowledge",
  "thread_id": "thread_123",
  "timestamp": "2024-01-15T10:30:00.000Z",
  "content": {
    "message": "Found relevant knowledge for learning",
    "similarity_score": 0.75,
    "knowledge_count": 3,
    "queries": ["machine learning", "artificial intelligence"],
    "complete": false
  }
}
```

## Usage

### Backend API Call
```python
# Enable WebSocket learning events
response = await convo_stream_learning(
    user_input="Tôi muốn dạy bạn về AI",
    user_id="user123",
    thread_id="thread456",
    use_websocket=True,  # Enable WebSocket
    thread_id_for_analysis="thread456"  # WebSocket room
)
```

### Frontend WebSocket Listeners
```javascript
// Listen for learning intent events
socket.on('learning_intent', (data) => {
    console.log('Learning Intent:', data.content);
    // Update UI to show intent understanding
    updateLearningStatus(data.content.intent_type);
});

// Listen for learning knowledge events
socket.on('learning_knowledge', (data) => {
    console.log('Knowledge Found:', data.content);
    // Update UI to show knowledge discovery
    updateKnowledgeStatus(data.content.similarity_score);
});
```

## Error Handling

### WebSocket Connection Issues
- Events are stored in `undelivered_messages` if no active sessions
- Failed emissions are logged but don't interrupt processing
- Graceful fallback to HTTP streaming if WebSocket unavailable

### Import Failures
```python
# Safe import pattern used throughout
try:
    from socketio_manager_async import emit_learning_intent_event
    socket_imports_success = True
except Exception as e:
    socket_imports_success = False
    logger.error(f"Failed to import WebSocket functions: {str(e)}")
```

## Testing

### Test Script
Run `python test_learning_websocket.py` to verify:
- Event structure validation
- Learning process simulation
- WebSocket emission points
- Error handling

### Integration Testing
1. Start the backend server
2. Connect WebSocket client to `/`
3. Register session with thread_id
4. Send learning request via API
5. Verify events are received in real-time

## Monitoring

### Log Messages
- `[WS_EMISSION]` - WebSocket emission events
- `Emitted learning_intent_event` - Successful intent event emission
- `Emitted learning_knowledge_event` - Successful knowledge event emission
- `Error emitting learning_*_event` - Emission failures

### Metrics to Monitor
- Learning event delivery rates
- WebSocket session counts per thread
- Event emission latency
- Failed emission counts

## Frontend Integration Requirements

### Event Handlers
Clients must implement handlers for:
- `learning_intent` events
- `learning_knowledge` events

### UI Updates
Recommended UI updates:
- Show "Understanding intent..." when learning_intent received
- Show "Finding knowledge..." when learning_knowledge received
- Display intent type and confidence levels
- Show knowledge similarity scores

### Session Management
- Ensure WebSocket sessions are properly registered with thread_id
- Handle reconnection scenarios
- Request missed events on reconnection

## Performance Considerations

### Event Frequency
- `learning_knowledge`: 1 event per knowledge search (typically 1-2 per request)
- `learning_intent`: 1 event per request after LLM evaluation

### Payload Size
- Events are lightweight (< 1KB typical)
- Knowledge events include minimal metadata only
- Full knowledge content not included in events

### Scalability
- Events delivered to thread-specific rooms only
- No global broadcasts
- Undelivered message storage is bounded (50 messages max)

## Security

### Thread Isolation
- Events only delivered to sessions in the same thread room
- Thread IDs act as authorization tokens
- No cross-thread event leakage

### Data Privacy
- Events contain minimal sensitive information
- Full conversation content not included in events
- Intent types and similarity scores are non-sensitive

## Future Enhancements

### Possible Additions
1. `learning_progress` events for long-running processes
2. `learning_error` events for error conditions
3. `learning_complete` events for process completion
4. Configurable event verbosity levels
5. Event filtering based on client preferences 