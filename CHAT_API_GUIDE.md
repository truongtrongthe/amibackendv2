# Chat API Guide for Frontend Integration

## Overview

This guide provides comprehensive documentation for integrating with the AMI Backend Chat API. The chat system is designed to work similarly to Cursor's chat interface, allowing users to have conversations with AMI while maintaining chat history.

## Base Configuration

- **Base URL**: `http://localhost:5001`
- **API Prefix**: `/api/chats`
- **Full Base URL**: `http://localhost:5001/api/chats`

## Authentication

All endpoints require proper user authentication. Include the user_id in requests for access control.

## Response Format

All endpoints return a standardized response format:

```json
{
  "success": true,
  "message": "Operation completed successfully",
  "data": { /* response data */ },
  "error": null
}
```

## Chat Management Endpoints

### 1. Create New Chat

**Endpoint**: `POST /api/chats/`

**Purpose**: Create a new chat session

**Request Body**:
```json
{
  "user_id": "user123",
  "title": "My Chat with AMI",
  "org_id": "org456",
  "chat_type": "conversation",
  "metadata": {
    "source": "web_app",
    "version": "1.0"
  }
}
```

**Response**:
```json
{
  "success": true,
  "message": "Chat created successfully",
  "data": {
    "id": "chat-uuid-123",
    "user_id": "user123",
    "title": "My Chat with AMI",
    "status": "active",
    "chat_type": "conversation",
    "org_id": "org456",
    "metadata": {
      "source": "web_app",
      "version": "1.0"
    },
    "created_at": "2025-01-15T10:30:00Z",
    "updated_at": "2025-01-15T10:30:00Z"
  }
}
```

**Frontend Example**:
```javascript
async function createChat(userId, title, orgId = null) {
  const response = await fetch('http://localhost:5001/api/chats/', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      user_id: userId,
      title: title,
      org_id: orgId,
      chat_type: 'conversation'
    })
  });
  
  const result = await response.json();
  return result.success ? result.data : null;
}
```

### 2. Get User's Chats

**Endpoint**: `GET /api/chats/`

**Purpose**: Retrieve all chats for a user with pagination and filtering

**Query Parameters**:
- `user_id` (required): User identifier
- `limit` (optional): Number of chats to return (default: 50, max: 100)
- `offset` (optional): Number of chats to skip (default: 0)
- `status` (optional): Filter by status (`active`, `archived`, `deleted`)
- `org_id` (optional): Filter by organization

**Request**:
```
GET /api/chats/?user_id=user123&limit=20&offset=0&status=active
```

**Response**:
```json
{
  "success": true,
  "message": "Retrieved 5 chats",
  "data": {
    "chats": [
      {
        "id": "chat-uuid-123",
        "user_id": "user123",
        "title": "My Chat with AMI",
        "status": "active",
        "chat_type": "conversation",
        "org_id": "org456",
        "metadata": {},
        "created_at": "2025-01-15T10:30:00Z",
        "updated_at": "2025-01-15T10:30:00Z"
      }
    ],
    "total": 5,
    "limit": 20,
    "offset": 0
  }
}
```

**Frontend Example**:
```javascript
async function getUserChats(userId, limit = 50, offset = 0, status = null) {
  const params = new URLSearchParams({
    user_id: userId,
    limit: limit.toString(),
    offset: offset.toString()
  });
  
  if (status) {
    params.append('status', status);
  }
  
  const response = await fetch(`http://localhost:5001/api/chats/?${params}`);
  const result = await response.json();
  return result.success ? result.data.chats : [];
}
```

### 3. Get Specific Chat

**Endpoint**: `GET /api/chats/{chat_id}`

**Purpose**: Retrieve a specific chat by ID

**Request**:
```
GET /api/chats/chat-uuid-123?user_id=user123
```

**Response**:
```json
{
  "success": true,
  "message": "Chat retrieved successfully",
  "data": {
    "id": "chat-uuid-123",
    "user_id": "user123",
    "title": "My Chat with AMI",
    "status": "active",
    "chat_type": "conversation",
    "org_id": "org456",
    "metadata": {},
    "created_at": "2025-01-15T10:30:00Z",
    "updated_at": "2025-01-15T10:30:00Z"
  }
}
```

### 4. Get Chat with Messages

**Endpoint**: `GET /api/chats/{chat_id}/with-messages`

**Purpose**: Retrieve a chat with all its messages (most commonly used)

**Request**:
```
GET /api/chats/chat-uuid-123/with-messages?user_id=user123
```

**Response**:
```json
{
  "success": true,
  "message": "Chat with messages retrieved successfully",
  "data": {
    "id": "chat-uuid-123",
    "user_id": "user123",
    "title": "My Chat with AMI",
    "status": "active",
    "chat_type": "conversation",
    "org_id": "org456",
    "metadata": {},
    "created_at": "2025-01-15T10:30:00Z",
    "updated_at": "2025-01-15T10:30:00Z",
    "messages": [
      {
        "id": "msg-uuid-456",
        "chat_id": "chat-uuid-123",
        "user_id": "user123",
        "role": "user",
        "content": "Hello AMI, how are you?",
        "message_type": "text",
        "metadata": {},
        "thread_id": "thread_789",
        "parent_message_id": null,
        "token_count": 5,
        "created_at": "2025-01-15T10:31:00Z",
        "updated_at": "2025-01-15T10:31:00Z"
      },
      {
        "id": "msg-uuid-789",
        "chat_id": "chat-uuid-123",
        "user_id": "user123",
        "role": "assistant",
        "content": "Hello! I'm doing well, thank you for asking. How can I help you today?",
        "message_type": "text",
        "metadata": {},
        "thread_id": "thread_789",
        "parent_message_id": "msg-uuid-456",
        "token_count": 15,
        "created_at": "2025-01-15T10:31:30Z",
        "updated_at": "2025-01-15T10:31:30Z"
      }
    ]
  }
}
```

**Frontend Example**:
```javascript
async function getChatWithMessages(chatId, userId) {
  const response = await fetch(`http://localhost:5001/api/chats/${chatId}/with-messages?user_id=${userId}`);
  const result = await response.json();
  return result.success ? result.data : null;
}
```

### 5. Update Chat

**Endpoint**: `PUT /api/chats/{chat_id}`

**Purpose**: Update chat properties (title, status, metadata)

**Request Body**:
```json
{
  "chat_id": "chat-uuid-123",
  "user_id": "user123",
  "title": "Updated Chat Title",
  "status": "active",
  "metadata": {
    "updated_reason": "user_request"
  }
}
```

**Frontend Example**:
```javascript
async function updateChat(chatId, userId, updates) {
  const response = await fetch(`http://localhost:5001/api/chats/${chatId}`, {
    method: 'PUT',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      chat_id: chatId,
      user_id: userId,
      ...updates
    })
  });
  
  const result = await response.json();
  return result.success ? result.data : null;
}
```

### 6. Delete Chat

**Endpoint**: `DELETE /api/chats/{chat_id}`

**Purpose**: Delete a chat and all its messages

**Request**:
```
DELETE /api/chats/chat-uuid-123?user_id=user123
```

**Frontend Example**:
```javascript
async function deleteChat(chatId, userId) {
  const response = await fetch(`http://localhost:5001/api/chats/${chatId}?user_id=${userId}`, {
    method: 'DELETE'
  });
  
  const result = await response.json();
  return result.success;
}
```

## Message Management Endpoints

### 1. Create Message

**Endpoint**: `POST /api/chats/{chat_id}/messages`

**Purpose**: Add a new message to a chat

**Request Body**:
```json
{
  "chat_id": "chat-uuid-123",
  "user_id": "user123",
  "role": "user",
  "content": "What can you help me with?",
  "message_type": "text",
  "metadata": {
    "client_timestamp": "2025-01-15T10:32:00Z"
  },
  "thread_id": "thread_789",
  "parent_message_id": null,
  "token_count": 6
}
```

**Response**:
```json
{
  "success": true,
  "message": "Message created successfully",
  "data": {
    "id": "msg-uuid-101",
    "chat_id": "chat-uuid-123",
    "user_id": "user123",
    "role": "user",
    "content": "What can you help me with?",
    "message_type": "text",
    "metadata": {
      "client_timestamp": "2025-01-15T10:32:00Z"
    },
    "thread_id": "thread_789",
    "parent_message_id": null,
    "token_count": 6,
    "created_at": "2025-01-15T10:32:00Z",
    "updated_at": "2025-01-15T10:32:00Z"
  }
}
```

**Frontend Example**:
```javascript
async function sendMessage(chatId, userId, content, role = 'user', threadId = null) {
  const response = await fetch(`http://localhost:5001/api/chats/${chatId}/messages`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      chat_id: chatId,
      user_id: userId,
      role: role,
      content: content,
      message_type: 'text',
      thread_id: threadId
    })
  });
  
  const result = await response.json();
  return result.success ? result.data : null;
}
```

### 2. Get Chat Messages

**Endpoint**: `GET /api/chats/{chat_id}/messages`

**Purpose**: Retrieve messages for a specific chat with pagination

**Query Parameters**:
- `limit` (optional): Number of messages to return (default: 100, max: 200)
- `offset` (optional): Number of messages to skip (default: 0)
- `role` (optional): Filter by role (`user`, `assistant`, `system`)
- `since` (optional): ISO timestamp to get messages after this time

**Request**:
```
GET /api/chats/chat-uuid-123/messages?limit=50&offset=0&role=user
```

**Frontend Example**:
```javascript
async function getChatMessages(chatId, limit = 100, offset = 0, role = null) {
  const params = new URLSearchParams({
    limit: limit.toString(),
    offset: offset.toString()
  });
  
  if (role) {
    params.append('role', role);
  }
  
  const response = await fetch(`http://localhost:5001/api/chats/${chatId}/messages?${params}`);
  const result = await response.json();
  return result.success ? result.data.messages : [];
}
```

### 3. Update Message

**Endpoint**: `PUT /api/chats/messages/{message_id}`

**Purpose**: Update a message's content or metadata

**Request Body**:
```json
{
  "message_id": "msg-uuid-101",
  "content": "Updated message content",
  "metadata": {
    "edited": true,
    "edit_timestamp": "2025-01-15T10:35:00Z"
  }
}
```

### 4. Delete Message

**Endpoint**: `DELETE /api/chats/messages/{message_id}`

**Purpose**: Delete a specific message

**Request**:
```
DELETE /api/chats/messages/msg-uuid-101
```

## Quick Start Endpoints

### 1. Create Chat with First Message

**Endpoint**: `POST /api/chats/create-with-message`

**Purpose**: Create a new chat and add the first message in one request

**Request Body**:
```json
{
  "user_id": "user123",
  "content": "Hello AMI, I need help with my project",
  "title": "Project Help Chat",
  "org_id": "org456",
  "chat_type": "conversation",
  "thread_id": "thread_789"
}
```

**Response**:
```json
{
  "success": true,
  "message": "Chat created with first message successfully",
  "data": {
    "chat": {
      "id": "chat-uuid-123",
      "user_id": "user123",
      "title": "Project Help Chat",
      "status": "active",
      "chat_type": "conversation",
      "org_id": "org456",
      "metadata": {},
      "created_at": "2025-01-15T10:30:00Z",
      "updated_at": "2025-01-15T10:30:00Z"
    },
    "message": {
      "id": "msg-uuid-456",
      "chat_id": "chat-uuid-123",
      "user_id": "user123",
      "role": "user",
      "content": "Hello AMI, I need help with my project",
      "message_type": "text",
      "metadata": {},
      "thread_id": "thread_789",
      "parent_message_id": null,
      "token_count": 8,
      "created_at": "2025-01-15T10:30:30Z",
      "updated_at": "2025-01-15T10:30:30Z"
    }
  }
}
```

**Frontend Example**:
```javascript
async function createChatWithMessage(userId, content, title, orgId = null) {
  const response = await fetch('http://localhost:5001/api/chats/create-with-message', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      user_id: userId,
      content: content,
      title: title,
      org_id: orgId,
      chat_type: 'conversation'
    })
  });
  
  const result = await response.json();
  return result.success ? result.data : null;
}
```

## Utility Endpoints

### 1. Get Chat Statistics

**Endpoint**: `GET /api/chats/statistics`

**Purpose**: Get usage statistics for chats and messages

**Query Parameters**:
- `user_id` (optional): Filter statistics by user
- `org_id` (optional): Filter statistics by organization

**Request**:
```
GET /api/chats/statistics?user_id=user123
```

**Response**:
```json
{
  "success": true,
  "message": "Chat statistics retrieved successfully",
  "data": {
    "total_chats": 15,
    "active_chats": 12,
    "total_messages": 234,
    "average_messages_per_chat": 15.6
  }
}
```

### 2. Health Check

**Endpoint**: `GET /api/chats/health`

**Purpose**: Check if the chat service is running

**Response**:
```json
{
  "success": true,
  "message": "Chat service is healthy",
  "data": {
    "status": "healthy",
    "timestamp": "2025-01-15T10:30:00Z"
  }
}
```

## Integration Patterns

### Complete Chat Interface Implementation

Here's a complete example of how to implement a chat interface:

```javascript
class ChatManager {
  constructor(baseUrl = 'http://localhost:5001') {
    this.baseUrl = baseUrl;
    this.apiUrl = `${baseUrl}/api/chats`;
  }

  // Create a new chat with initial message
  async startNewChat(userId, initialMessage, title = null, orgId = null) {
    try {
      const response = await fetch(`${this.apiUrl}/create-with-message`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          user_id: userId,
          content: initialMessage,
          title: title || `Chat ${new Date().toLocaleString()}`,
          org_id: orgId,
          chat_type: 'conversation'
        })
      });
      
      const result = await response.json();
      if (result.success) {
        return result.data;
      } else {
        throw new Error(result.error || 'Failed to create chat');
      }
    } catch (error) {
      console.error('Error creating chat:', error);
      throw error;
    }
  }

  // Load chat history
  async loadChatHistory(chatId, userId) {
    try {
      const response = await fetch(`${this.apiUrl}/${chatId}/with-messages?user_id=${userId}`);
      const result = await response.json();
      
      if (result.success) {
        return result.data;
      } else {
        throw new Error(result.error || 'Failed to load chat');
      }
    } catch (error) {
      console.error('Error loading chat:', error);
      throw error;
    }
  }

  // Send a message
  async sendMessage(chatId, userId, content, threadId = null) {
    try {
      const response = await fetch(`${this.apiUrl}/${chatId}/messages`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          chat_id: chatId,
          user_id: userId,
          role: 'user',
          content: content,
          message_type: 'text',
          thread_id: threadId
        })
      });
      
      const result = await response.json();
      if (result.success) {
        return result.data;
      } else {
        throw new Error(result.error || 'Failed to send message');
      }
    } catch (error) {
      console.error('Error sending message:', error);
      throw error;
    }
  }

  // Get user's chat list
  async getUserChats(userId, limit = 50, offset = 0) {
    try {
      const params = new URLSearchParams({
        user_id: userId,
        limit: limit.toString(),
        offset: offset.toString(),
        status: 'active'
      });
      
      const response = await fetch(`${this.apiUrl}/?${params}`);
      const result = await response.json();
      
      if (result.success) {
        return result.data.chats;
      } else {
        throw new Error(result.error || 'Failed to load chats');
      }
    } catch (error) {
      console.error('Error loading chats:', error);
      throw error;
    }
  }

  // Update chat title
  async updateChatTitle(chatId, userId, newTitle) {
    try {
      const response = await fetch(`${this.apiUrl}/${chatId}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          chat_id: chatId,
          user_id: userId,
          title: newTitle
        })
      });
      
      const result = await response.json();
      if (result.success) {
        return result.data;
      } else {
        throw new Error(result.error || 'Failed to update chat');
      }
    } catch (error) {
      console.error('Error updating chat:', error);
      throw error;
    }
  }

  // Delete a chat
  async deleteChat(chatId, userId) {
    try {
      const response = await fetch(`${this.apiUrl}/${chatId}?user_id=${userId}`, {
        method: 'DELETE'
      });
      
      const result = await response.json();
      return result.success;
    } catch (error) {
      console.error('Error deleting chat:', error);
      throw error;
    }
  }
}

// Usage example
const chatManager = new ChatManager();

// Start a new conversation
async function startConversation(userId, initialMessage) {
  try {
    const chatData = await chatManager.startNewChat(userId, initialMessage);
    console.log('New chat created:', chatData);
    return chatData;
  } catch (error) {
    console.error('Failed to start conversation:', error);
  }
}

// Load existing conversation
async function loadConversation(chatId, userId) {
  try {
    const chatData = await chatManager.loadChatHistory(chatId, userId);
    console.log('Chat loaded:', chatData);
    return chatData;
  } catch (error) {
    console.error('Failed to load conversation:', error);
  }
}
```

### Integration with Existing AMI Endpoints

You can integrate the chat system with existing AMI conversation endpoints:

```javascript
// Function to handle AMI conversation and store results
async function handleAMIConversation(chatId, userId, userInput, threadId) {
  try {
    // 1. Store user message
    await chatManager.sendMessage(chatId, userId, userInput, threadId);
    
    // 2. Call existing AMI endpoint
    const amiResponse = await fetch('http://localhost:5001/havefun', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        user_input: userInput,
        user_id: userId,
        thread_id: threadId,
        use_websocket: false
      })
    });
    
    // 3. Process AMI response (streaming)
    const reader = amiResponse.body.getReader();
    let amiReply = '';
    
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      
      const chunk = new TextDecoder().decode(value);
      // Process SSE data
      const lines = chunk.split('\n');
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = JSON.parse(line.substring(6));
          if (data.message) {
            amiReply += data.message;
          }
        }
      }
    }
    
    // 4. Store AMI response
    if (amiReply) {
      await chatManager.sendMessage(chatId, userId, amiReply, threadId, 'assistant');
    }
    
    return amiReply;
  } catch (error) {
    console.error('Error in AMI conversation:', error);
    throw error;
  }
}
```

## Error Handling

All endpoints return consistent error responses:

```json
{
  "success": false,
  "message": "Error description",
  "data": null,
  "error": "Detailed error message"
}
```

Common HTTP status codes:
- `200`: Success
- `400`: Bad Request (invalid parameters)
- `404`: Not Found (chat/message doesn't exist)
- `500`: Internal Server Error

## Data Types Reference

### Chat Types
- `conversation`: General conversation
- `support`: Support/help chat
- `training`: Training/learning chat
- `pilot`: Pilot/experimental chat

### Message Roles
- `user`: Message from the user
- `assistant`: Message from AMI
- `system`: System-generated message

### Message Types
- `text`: Regular text message
- `image`: Image message
- `file`: File attachment
- `system`: System message

### Chat Statuses
- `active`: Active chat
- `archived`: Archived chat
- `deleted`: Deleted chat

## Performance Recommendations

1. **Pagination**: Always use pagination for large datasets
2. **Caching**: Cache frequently accessed chats on the frontend
3. **Batch Operations**: Use `create-with-message` for new chats
4. **Lazy Loading**: Load messages on-demand for better performance
5. **Debouncing**: Debounce update operations to avoid excessive API calls

This guide provides everything you need to integrate the chat system into your frontend application. The system is designed to be flexible and scalable while maintaining compatibility with your existing AMI conversation endpoints. 