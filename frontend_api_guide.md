# API Documentation for Frontend Team

## Overview

This document provides comprehensive documentation for the backend APIs available for frontend integration, focusing on three key areas:

1. **Organization Management**: Endpoints for creating and managing organizations
2. **AIA (AI Assistant) Management**: Endpoints for creating, updating, and managing AI assistants
3. **Facebook Messenger Integration**: How the system handles Facebook messages and user profiles

## Organization Management APIs

Organizations are the top-level entities that own AIAs and brains.

### 1. Get Organization Details

Retrieves detailed information about a specific organization.

- **URL**: `/get-org-detail/{orgid}`
- **Method**: `GET`
- **URL Parameters**:
  - `orgid` - UUID of the organization
- **Response**:
  ```json
  {
    "organization": {
      "id": "uuid-string",
      "org_id": 123,
      "name": "Organization Name",
      "description": "Description of the organization",
      "email": "contact@organization.com",
      "phone": "+1-555-123-4567",
      "address": "123 Main St, City, State, ZIP",
      "created_date": "2023-06-01T10:15:23Z"
    }
  }
  ```

### 2. Create Organization

Creates a new organization with optional contact information.

- **URL**: `/create-organization`
- **Method**: `POST`
- **Request Body**:
  ```json
  {
    "name": "New Organization",
    "description": "Optional description of the organization",
    "email": "contact@organization.com",
    "phone": "+1-555-123-4567",
    "address": "123 Main St, City, State, ZIP"
  }
  ```
- **Response**:
  ```json
  {
    "message": "Organization created successfully",
    "organization": {
      "id": "uuid-string",
      "org_id": 123,
      "name": "New Organization",
      "description": "Optional description of the organization",
      "email": "contact@organization.com",
      "phone": "+1-555-123-4567",
      "address": "123 Main St, City, State, ZIP",
      "created_date": "2023-06-01T10:15:23Z"
    }
  }
  ```

### 3. Update Organization

Updates an existing organization's information.

- **URL**: `/update-organization`
- **Method**: `POST`
- **Request Body**:
  ```json
  {
    "id": "uuid-string",
    "name": "Updated Organization Name",
    "description": "Updated description",
    "email": "new-email@organization.com",
    "phone": "+1-555-987-6543",
    "address": "456 New St, City, State, ZIP"
  }
  ```
- **Response**:
  ```json
  {
    "message": "Organization updated successfully",
    "organization": {
      "id": "uuid-string",
      "org_id": 123,
      "name": "Updated Organization Name",
      "description": "Updated description",
      "email": "new-email@organization.com",
      "phone": "+1-555-987-6543",
      "address": "456 New St, City, State, ZIP",
      "created_date": "2023-06-01T10:15:23Z"
    }
  }
  ```

## AIA Management APIs

AIAs (AI Assistants) are managed through the following endpoints:

### 1. Get All AIAs

Retrieves all AI assistants for a specific organization.

- **URL**: `/aias`
- **Method**: `GET`
- **Query Parameters**:
  - `org_id` - UUID of the organization
- **Response**:
  ```json
  {
    "aias": [
      {
        "id": "uuid-string",
        "aia_id": 123,
        "org_id": "org-uuid-string",
        "task_type": "Chat",
        "name": "Assistant Name",
        "brain_ids": [1, 2, 3],
        "delivery_method_ids": [1, 2],
        "created_date": "2023-06-01T10:15:23Z"
      }
    ]
  }
  ```

### 2. Get AIA Details

Retrieves details of a specific AI assistant.

- **URL**: `/aia-details`
- **Method**: `GET`
- **Query Parameters**:
  - `aia_id` - UUID of the AIA
- **Response**:
  ```json
  {
    "aia": {
      "id": "uuid-string",
      "aia_id": 123,
      "org_id": "org-uuid-string",
      "task_type": "Chat",
      "name": "Assistant Name",
      "brain_ids": [1, 2, 3],
      "delivery_method_ids": [1, 2],
      "created_date": "2023-06-01T10:15:23Z"
    }
  }
  ```

### 3. Create AIA

Creates a new AI assistant.

- **URL**: `/create-aia`
- **Method**: `POST`
- **Request Body**:
  ```json
  {
    "org_id": "org-uuid-string",
    "task_type": "Chat",
    "name": "New Assistant",
    "brain_ids": [1, 2, 3],
    "delivery_method_ids": [1, 2]
  }
  ```
- **Response**:
  ```json
  {
    "message": "AIA created successfully",
    "aia": {
      "id": "uuid-string",
      "aia_id": 123,
      "org_id": "org-uuid-string",
      "task_type": "Chat",
      "name": "New Assistant",
      "brain_ids": [1, 2, 3],
      "delivery_method_ids": [1, 2],
      "created_date": "2023-06-01T10:15:23Z"
    }
  }
  ```

### 4. Update AIA

Updates an existing AI assistant.

- **URL**: `/update-aia`
- **Method**: `POST`
- **Request Body**:
  ```json
  {
    "id": "uuid-string",
    "task_type": "Work", // optional
    "name": "Updated Name", // optional
    "brain_ids": [1, 2, 3, 4], // optional
    "delivery_method_ids": [1, 3] // optional
  }
  ```
- **Response**:
  ```json
  {
    "message": "AIA updated successfully",
    "aia": {
      "id": "uuid-string",
      "aia_id": 123,
      "org_id": "org-uuid-string",
      "task_type": "Work",
      "name": "Updated Name",
      "brain_ids": [1, 2, 3, 4],
      "delivery_method_ids": [1, 3],
      "created_date": "2023-06-01T10:15:23Z"
    }
  }
  ```

### 5. Delete AIA

Deletes an AI assistant.

- **URL**: `/delete-aia`
- **Method**: `POST`
- **Request Body**:
  ```json
  {
    "id": "uuid-string"
  }
  ```
- **Response**:
  ```json
  {
    "message": "AIA with id uuid-string deleted successfully"
  }
  ```

## Facebook Messenger Integration

The backend includes a comprehensive Facebook Messenger integration that:

1. Automatically handles webhook events from Facebook
2. Creates user profiles from Facebook data
3. Manages conversations with users
4. Sends responses back to users

### Key Features

#### 1. User Profile Creation

When a user messages your page for the first time, the system:

- Automatically retrieves their profile information from Facebook (name, profile picture)
- Creates a contact record in the database with this information
- Attempts to fetch additional data (email, location, etc.) when available
- Creates a rich contact profile to use in conversations

#### 2. Conversation Management

All messages are saved into a structured conversation system that:

- Preserves the full message history for each user
- Tracks both incoming and outgoing messages
- Supports various message types (text, images, attachments)
- Associates conversations with contact records

#### 3. Message Handling

For testing purposes, the system:

- Only sends actual messages to a designated test user ID (29495554333369135)
- For all other users, simulates sending but doesn't make actual API calls
- Maintains complete conversation records regardless of whether messages are actually sent

### Relevant Endpoints

#### 1. Get Contact Conversations

Retrieves conversation history for a specific contact.

- **URL**: `/contact-conversations`
- **Method**: `GET`
- **Query Parameters**:
  - `contact_id` - ID of the contact
  - `limit` - (Optional) Number of conversations to return
  - `offset` - (Optional) Pagination offset
- **Response**:
  ```json
  {
    "conversations": [
      {
        "id": 123,
        "contact_id": 456,
        "platform": "facebook",
        "platform_conversation_id": "29495554333369135",
        "last_message_at": "2023-06-01T10:15:23Z",
        "conversation_data": {
          "messages": [
            {
              "id": "msg_123",
              "content": "Hello there!",
              "content_type": "text",
              "sender_id": "29495554333369135",
              "sender_type": "contact",
              "timestamp": "2023-06-01T10:15:23Z"
            },
            {
              "id": "msg_124",
              "content": "Thank you for your message!",
              "content_type": "text",
              "sender_id": "system",
              "sender_type": "system",
              "timestamp": "2023-06-01T10:16:00Z"
            }
          ]
        }
      }
    ]
  }
  ```

#### 2. Get Conversation Details

Retrieves detailed information about a specific conversation.

- **URL**: `/conversation`
- **Method**: `GET`
- **Query Parameters**:
  - `conversation_id` - ID of the conversation
  - `read` - (Optional) Set to 'true' to mark the conversation as read
- **Response**:
  ```json
  {
    "conversation": {
      "id": 123,
      "contact_id": 456,
      "platform": "facebook",
      "platform_conversation_id": "29495554333369135",
      "last_message_at": "2023-06-01T10:15:23Z",
      "is_read": true,
      "conversation_data": {
        "messages": [
          // Array of message objects
        ]
      }
    }
  }
  ```

#### 3. Get Contact Details

Retrieves contact information including profile data.

- **URL**: `/contact-details`
- **Method**: `GET`
- **Query Parameters**:
  - `contact_id` - ID of the contact
- **Response**:
  ```json
  {
    "contact": {
      "id": 456,
      "type": "customer",
      "first_name": "John",
      "last_name": "Doe",
      "email": "john@example.com",
      "facebook_id": "29495554333369135",
      "profile_picture_url": "https://platform-lookaside.fbsbx.com/platform/profilepic/?eai=...",
      "created_at": "2023-06-01T10:15:23Z",
      "profiles": {
        "id": 789,
        "profile_summary": "Facebook user John Doe from California.",
        "general_info": "{\"source\":\"Facebook Messenger\",\"location\":\"California\"}",
        "personality": "John is a Facebook Messenger user.",
        "social_media_urls": [{"platform": "facebook", "url": "https://facebook.com/29495554333369135"}],
        "best_goals": [{"goal": "Get information or assistance", "importance": "High"}]
      }
    }
  }
  ```

## Brain Graph Endpoints

### 1. Create Brain Graph
**Endpoint**: `/create-brain-graph`  
**Method**: POST  
**Description**: Creates a new brain graph for an organization.

**Request Body**:
```json
{
    "org_id": "string (required)",
    "name": "string (required)",
    "description": "string (optional)"
}
```

**Response**:
```json
{
    "message": "Brain graph created successfully",
    "brain_graph": {
        "id": "uuid",
        "graph_id": "string",
        "org_id": "string",
        "name": "string",
        "description": "string",
        "created_date": "ISO datetime string"
    }
}
```

### 2. Get Organization's Brain Graph
**Endpoint**: `/get-org-brain-graph`  
**Method**: GET  
**Description**: Retrieves the brain graph and its latest version for an organization.

**Query Parameters**:
- `org_id` (required): Organization ID

**Response**:
```json
{
    "brain_graph": {
        "id": "uuid",
        "graph_id": "string",
        "org_id": "string",
        "name": "string",
        "description": "string",
        "created_date": "ISO datetime string",
        "latest_version": {
            "version_number": "integer",
            "brain_ids": ["array of brain UUIDs"],
            "status": "string",
            "released_date": "ISO datetime string"
        }
    }
}
```

### 3. Create Graph Version
**Endpoint**: `/create-graph-version`  
**Method**: POST  
**Description**: Creates a new version of a brain graph.

**Request Body**:
```json
{
    "graph_id": "string (required)",
    "brain_ids": ["array of brain UUIDs (optional)"]
}
```

**Response**:
```json
{
    "message": "Graph version created successfully",
    "version": {
        "id": "uuid",
        "version_id": "string",
        "version_number": "integer",
        "brain_ids": ["array of brain UUIDs"],
        "status": "string",
        "released_date": "ISO datetime string"
    }
}
```

### 4. Release Graph Version
**Endpoint**: `/release-graph-version`  
**Method**: POST  
**Description**: Publishes a graph version, making it active.

**Request Body**:
```json
{
    "version_id": "string (required)"
}
```

**Response**:
```json
{
    "message": "Graph version published successfully",
    "version": {
        "id": "uuid",
        "version_number": "integer",
        "brain_ids": ["array of brain UUIDs"],
        "status": "string",
        "released_date": "ISO datetime string"
    }
}
```

### 5. Update Graph Version Brains
**Endpoint**: `/update-graph-version-brains`  
**Method**: POST  
**Description**: Adds or removes brains from a graph version.

**Request Body**:
```json
{
    "version_id": "string (required)",
    "action": "string (required, either 'add' or 'remove')",
    "brain_ids": ["array of brain UUIDs (required)"]
}
```

**Response**:
```json
{
    "message": "Brain IDs added/removed successfully",
    "version": {
        "id": "uuid",
        "version_number": "integer",
        "brain_ids": ["array of brain UUIDs"],
        "status": "string",
        "released_date": "ISO datetime string"
    }
}
```

### 6. Get Version Brains
**Endpoint**: `/get-version-brains`  
**Method**: GET  
**Description**: Retrieves all brains associated with a graph version.

**Query Parameters**:
- `version_id` (required): Version ID

**Response**:
```json
{
    "version_id": "string",
    "brains": [
        {
            "id": "integer",
            "brain_id": "uuid",
            "org_id": "string",
            "name": "string",
            "status": "string",
            "bank_name": "string",
            "summary": "string"
        }
    ]
}
```

## Updated Havefun Endpoint

### Chat with AI
**Endpoint**: `/havefun`  
**Method**: POST  
**Description**: Initiates or continues a conversation with the AI, now supporting brain graph versioning for knowledge retrieval.

**Request Body**:
```json
{
    "user_input": "string (required)",
    "user_id": "string (optional, defaults to 'thefusionlab')",
    "thread_id": "string (optional)",
    "bank_name": "string (optional)",
    "brain_uuid": "string (optional)",
    "graph_version_id": "string (optional)"
}
```

**Key Parameters**:
- `graph_version_id`: UUID of the brain graph version to use for knowledge retrieval
- `brain_uuid`: UUID of the brain to log conversations to
- `bank_name`: Legacy parameter, maintained for backward compatibility

**Response**:
Server-Sent Events (SSE) stream with the following format for each chunk:
```json
{
    "message": "string (AI response chunk)"
}
```

**Notes**:
1. The endpoint now uses `graph_version_id` to query knowledge across multiple brains in the specified graph version
2. Conversation logging is done to the specified `brain_uuid` if provided
3. The response is streamed in chunks for real-time interaction
4. The conversation state is maintained across multiple requests using the same `thread_id`

## Implementation Notes

### Testing Facebook Integration

1. The system is currently configured to only send real messages to the test user ID: `29495554333369135`
2. When testing, send messages to your Facebook page from this account to receive actual responses
3. Messages to/from other users are tracked in the database but no actual Facebook API calls are made

### Error Handling

All endpoints return appropriate HTTP status codes:

- `200/201`: Success
- `400`: Bad request (missing parameters, invalid data)
- `404`: Resource not found
- `500`: Server error

Most error responses include an `error` field with a description of what went wrong.
