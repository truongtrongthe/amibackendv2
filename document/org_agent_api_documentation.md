# Organization Agent API Documentation

## Overview

The Organization Agent API provides endpoints for managing AI agents within organizations. All endpoints require authentication via JWT Bearer token and enforce organization-based permissions.

**Base URL:** `/org-agents`

## Authentication

All requests must include a valid JWT Bearer token in the Authorization header:

```
Authorization: Bearer <your-jwt-token>
```

## Data Models

### Agent Response Model
```typescript
interface AgentResponse {
  id: string;
  agent_id: number;
  org_id: string;
  name: string;
  description?: string;
  system_prompt: Record<string, any>;
  tools_list: string[];
  knowledge_list: string[];
  status: 'active' | 'deactive' | 'delete';
  created_by: string;
  created_date: string; // ISO datetime
  updated_date: string; // ISO datetime
}
```

### Create Agent Request
```typescript
interface CreateAgentRequest {
  name: string; // Required, 1-255 characters
  description?: string;
  system_prompt?: Record<string, any>; // JSON object
  tools_list?: string[]; // Array of tool names/IDs
  knowledge_list?: string[]; // Array of knowledge base names/IDs
}
```

### Update Agent Request
```typescript
interface UpdateAgentRequest {
  name?: string; // 1-255 characters
  description?: string;
  system_prompt?: Record<string, any>;
  tools_list?: string[];
  knowledge_list?: string[];
  status?: 'active' | 'deactive' | 'delete';
}
```

### Search Agents Request
```typescript
interface SearchAgentsRequest {
  query: string; // Required search term
  limit?: number; // Optional, 1-100, default: 10
}
```

## Endpoints

### 1. Create Agent

**POST** `/org-agents/`

Creates a new agent for the current user's organization.

**Permissions:** Organization members (owner, admin, member)

**Request Body:**
```json
{
  "name": "Customer Support Agent",
  "description": "AI agent for handling customer inquiries",
  "system_prompt": {
    "role": "customer_support",
    "tone": "friendly"
  },
  "tools_list": ["email_tool", "calendar_tool"],
  "knowledge_list": ["faq_database", "product_manual"]
}
```

**Response:** `AgentResponse`

**Error Codes:**
- `401` - Invalid or missing token
- `403` - User not a member of an organization
- `500` - Failed to create agent

---

### 2. Get All Agents

**GET** `/org-agents/`

Retrieves all agents for the current user's organization.

**Permissions:** Organization members

**Query Parameters:**
- `status` (optional): Filter by status - `active`, `deactive`, or `delete`

**Example:**
```
GET /org-agents/?status=active
```

**Response:** `AgentResponse[]`

**Error Codes:**
- `401` - Invalid or missing token
- `403` - User not a member of an organization
- `400` - Invalid status parameter
- `500` - Failed to get agents

---

### 3. Get Specific Agent

**GET** `/org-agents/{agent_id}`

Retrieves a specific agent by ID.

**Permissions:** Organization members

**Path Parameters:**
- `agent_id` (string): The agent's unique identifier

**Example:**
```
GET /org-agents/123e4567-e89b-12d3-a456-426614174000
```

**Response:** `AgentResponse`

**Error Codes:**
- `401` - Invalid or missing token
- `403` - User not a member of the organization
- `404` - Agent not found
- `500` - Failed to get agent

---

### 4. Update Agent

**PUT** `/org-agents/{agent_id}`

Updates an agent's information.

**Permissions:** Organization owners and admins only

**Path Parameters:**
- `agent_id` (string): The agent's unique identifier

**Request Body:**
```json
{
  "name": "Updated Agent Name",
  "description": "Updated description",
  "system_prompt": {
    "role": "updated_role",
    "tone": "professional"
  },
  "tools_list": ["new_tool_1", "new_tool_2"],
  "knowledge_list": ["updated_knowledge"],
  "status": "active"
}
```

**Response:** `AgentResponse`

**Error Codes:**
- `401` - Invalid or missing token
- `403` - Insufficient permissions or not a member
- `404` - Agent not found
- `500` - Failed to update agent

---

### 5. Delete Agent

**DELETE** `/org-agents/{agent_id}`

Soft deletes an agent by setting status to 'delete'.

**Permissions:** Organization owners and admins only

**Path Parameters:**
- `agent_id` (string): The agent's unique identifier

**Response:**
```json
{
  "message": "Agent deleted successfully"
}
```

**Error Codes:**
- `401` - Invalid or missing token
- `403` - Insufficient permissions or not a member
- `404` - Agent not found
- `500` - Failed to delete agent

---

### 6. Search Agents

**POST** `/org-agents/search`

Searches agents by name within the current user's organization.

**Permissions:** Organization members

**Request Body:**
```json
{
  "query": "support",
  "limit": 20
}
```

**Response:** `AgentResponse[]`

**Error Codes:**
- `401` - Invalid or missing token
- `403` - User not a member of an organization
- `500` - Failed to search agents

---

### 7. Activate Agent

**POST** `/org-agents/{agent_id}/activate`

Activates a deactivated agent.

**Permissions:** Organization owners and admins only

**Path Parameters:**
- `agent_id` (string): The agent's unique identifier

**Response:**
```json
{
  "message": "Agent activated successfully"
}
```

**Error Codes:**
- `401` - Invalid or missing token
- `403` - Insufficient permissions or not a member
- `404` - Agent not found
- `500` - Failed to activate agent

---

### 8. Deactivate Agent

**POST** `/org-agents/{agent_id}/deactivate`

Deactivates an active agent.

**Permissions:** Organization owners and admins only

**Path Parameters:**
- `agent_id` (string): The agent's unique identifier

**Response:**
```json
{
  "message": "Agent deactivated successfully"
}
```

**Error Codes:**
- `401` - Invalid or missing token
- `403` - Insufficient permissions or not a member
- `404` - Agent not found
- `500` - Failed to deactivate agent

---

## Permission Levels

### Organization Roles
- **Owner**: Full access to all agent operations
- **Admin**: Full access to all agent operations
- **Member**: Can view and create agents, but cannot update/delete/activate/deactivate

### Operation Permissions

| Operation | Owner | Admin | Member |
|-----------|-------|-------|--------|
| Create Agent | ✅ | ✅ | ✅ |
| View Agents | ✅ | ✅ | ✅ |
| Update Agent | ✅ | ✅ | ❌ |
| Delete Agent | ✅ | ✅ | ❌ |
| Activate Agent | ✅ | ✅ | ❌ |
| Deactivate Agent | ✅ | ✅ | ❌ |
| Search Agents | ✅ | ✅ | ✅ |

## Status Values

- `active`: Agent is active and available for use
- `deactive`: Agent is inactive but can be reactivated
- `delete`: Agent is soft-deleted (not permanently removed)

## Frontend Implementation Examples

### JavaScript/TypeScript with Fetch

```typescript
class AgentAPI {
  private baseURL: string;
  private token: string;

  constructor(baseURL: string, token: string) {
    this.baseURL = baseURL;
    this.token = token;
  }

  private async request(endpoint: string, options: RequestInit = {}) {
    const response = await fetch(`${this.baseURL}/org-agents${endpoint}`, {
      ...options,
      headers: {
        'Authorization': `Bearer ${this.token}`,
        'Content-Type': 'application/json',
        ...options.headers,
      },
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'API request failed');
    }

    return response.json();
  }

  // Create a new agent
  async createAgent(data: CreateAgentRequest): Promise<AgentResponse> {
    return this.request('/', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  // Get all agents
  async getAgents(status?: string): Promise<AgentResponse[]> {
    const params = status ? `?status=${status}` : '';
    return this.request(params);
  }

  // Get specific agent
  async getAgent(agentId: string): Promise<AgentResponse> {
    return this.request(`/${agentId}`);
  }

  // Update agent
  async updateAgent(agentId: string, data: UpdateAgentRequest): Promise<AgentResponse> {
    return this.request(`/${agentId}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  }

  // Delete agent
  async deleteAgent(agentId: string): Promise<{ message: string }> {
    return this.request(`/${agentId}`, {
      method: 'DELETE',
    });
  }

  // Search agents
  async searchAgents(query: string, limit?: number): Promise<AgentResponse[]> {
    return this.request('/search', {
      method: 'POST',
      body: JSON.stringify({ query, limit }),
    });
  }

  // Activate agent
  async activateAgent(agentId: string): Promise<{ message: string }> {
    return this.request(`/${agentId}/activate`, {
      method: 'POST',
    });
  }

  // Deactivate agent
  async deactivateAgent(agentId: string): Promise<{ message: string }> {
    return this.request(`/${agentId}/deactivate`, {
      method: 'POST',
    });
  }
}

// Usage example
const agentAPI = new AgentAPI('https://api.example.com', 'your-jwt-token');

// Create an agent
const newAgent = await agentAPI.createAgent({
  name: 'Support Bot',
  description: 'Customer support assistant',
  system_prompt: { role: 'support' },
  tools_list: ['email', 'chat'],
  knowledge_list: ['faq']
});

// Get active agents
const activeAgents = await agentAPI.getAgents('active');

// Update an agent
await agentAPI.updateAgent(newAgent.id, {
  name: 'Enhanced Support Bot',
  status: 'active'
});
```

### React Hook Example

```typescript
import { useState, useEffect } from 'react';

interface UseAgentsReturn {
  agents: AgentResponse[];
  loading: boolean;
  error: string | null;
  createAgent: (data: CreateAgentRequest) => Promise<void>;
  updateAgent: (id: string, data: UpdateAgentRequest) => Promise<void>;
  deleteAgent: (id: string) => Promise<void>;
  activateAgent: (id: string) => Promise<void>;
  deactivateAgent: (id: string) => Promise<void>;
}

export function useAgents(token: string): UseAgentsReturn {
  const [agents, setAgents] = useState<AgentResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const api = new AgentAPI('https://api.example.com', token);

  const fetchAgents = async () => {
    try {
      setLoading(true);
      const data = await api.getAgents();
      setAgents(data);
      setError(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const createAgent = async (data: CreateAgentRequest) => {
    try {
      const newAgent = await api.createAgent(data);
      setAgents(prev => [...prev, newAgent]);
    } catch (err) {
      setError(err.message);
      throw err;
    }
  };

  const updateAgent = async (id: string, data: UpdateAgentRequest) => {
    try {
      const updatedAgent = await api.updateAgent(id, data);
      setAgents(prev => prev.map(agent => 
        agent.id === id ? updatedAgent : agent
      ));
    } catch (err) {
      setError(err.message);
      throw err;
    }
  };

  const deleteAgent = async (id: string) => {
    try {
      await api.deleteAgent(id);
      setAgents(prev => prev.filter(agent => agent.id !== id));
    } catch (err) {
      setError(err.message);
      throw err;
    }
  };

  const activateAgent = async (id: string) => {
    try {
      await api.activateAgent(id);
      setAgents(prev => prev.map(agent => 
        agent.id === id ? { ...agent, status: 'active' } : agent
      ));
    } catch (err) {
      setError(err.message);
      throw err;
    }
  };

  const deactivateAgent = async (id: string) => {
    try {
      await api.deactivateAgent(id);
      setAgents(prev => prev.map(agent => 
        agent.id === id ? { ...agent, status: 'deactive' } : agent
      ));
    } catch (err) {
      setError(err.message);
      throw err;
    }
  };

  useEffect(() => {
    fetchAgents();
  }, []);

  return {
    agents,
    loading,
    error,
    createAgent,
    updateAgent,
    deleteAgent,
    activateAgent,
    deactivateAgent,
  };
}
```

## Error Handling

All endpoints return appropriate HTTP status codes and error messages. Common error responses:

```json
{
  "detail": "Error message describing what went wrong"
}
```

### Common Error Scenarios

1. **Authentication Errors (401)**
   - Invalid or expired JWT token
   - Missing Authorization header

2. **Permission Errors (403)**
   - User not a member of the organization
   - Insufficient permissions for the operation

3. **Not Found Errors (404)**
   - Agent ID doesn't exist

4. **Validation Errors (400)**
   - Invalid status parameter
   - Missing required fields
   - Invalid data format

5. **Server Errors (500)**
   - Database connection issues
   - Internal server errors

## Best Practices

1. **Always handle errors gracefully** - Check for error responses and display appropriate messages to users
2. **Implement loading states** - Show loading indicators during API calls
3. **Validate input data** - Ensure required fields are provided and data formats are correct
4. **Cache responses** - Consider caching agent lists to reduce API calls
5. **Implement retry logic** - For network failures, implement exponential backoff
6. **Check permissions** - Verify user roles before showing action buttons
7. **Use optimistic updates** - Update UI immediately, then sync with server response 