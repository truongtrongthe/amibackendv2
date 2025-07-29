# Frontend Implementation Guide: Using Ami.py to Build AI Agents

## Table of Contents
- [Overview](#overview)
- [Collaborative Approach (Recommended)](#collaborative-approach-recommended)
- [Direct Approach (Legacy)](#direct-approach-legacy)
- [Frontend Implementation Examples](#frontend-implementation-examples)
- [Error Handling](#error-handling)
- [Best Practices](#best-practices)
- [Quick Start Checklist](#quick-start-checklist)

## Overview

Ami operates as a **Chief Product Officer** for AI agents, offering two approaches:
- **🤝 Collaborative Approach** (Recommended) - Multi-step conversation with human approval
- **⚡ Direct Approach** (Legacy) - One-shot agent creation

**Base URL**: `http://localhost:5001`

## Collaborative Approach (Recommended)

### Conversation Flow Overview

The collaborative approach follows a multi-step conversation flow where Ami acts as your **Chief Product Officer**, helping refine agent ideas through iterative feedback. Here's how it works:

**🔄 Complete Flow Example:**
1. **User Request**: "Build me an agent to read google drive folder then spot out sale opportunities from spreadsheets"
2. **Ami Analyzes & Proposes**: Creates detailed agent skeleton with capabilities, tools, and specifications
3. **User Feedback**: "I love it but make it focus more on B2B opportunities and add email integration" 
4. **Ami Refines**: Updates the skeleton based on feedback, keeps what you liked
5. **User Approval**: "Perfect! Build it!" or "Love it, build this agent!"
6. **Ami Builds**: Generates final system prompt and saves agent to database

**🎯 Key Conversation States:**
- **`initial_idea`** - Ami analyzes your initial request and proposes an agent skeleton
- **`skeleton_review`** - You can provide feedback and iterate (this can loop multiple times!)
- **`completed`** - Agent is built and ready to use

**💡 Important Notes for Frontend:**
- The `skeleton_review` state can repeat indefinitely until user approves
- Users can request multiple changes: "Make it more focused on X", "Add tool Y", "Change the name"
- Ami remembers what users like and only changes what they request
- Final agent creation only happens after explicit approval ("Build it!", "Perfect!", "Love it!")
- Each conversation has a unique `conversation_id` that persists the entire flow

### Step 1: Start the Conversation

**Endpoint**: `POST /ami/collaborate`

```javascript
const startCollaboration = async (userIdea) => {
    const response = await fetch('http://localhost:5001/ami/collaborate', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${userToken}`
        },
        body: JSON.stringify({
            user_input: userIdea,
            current_state: "initial_idea",
            llm_provider: "anthropic" // or "openai"
        })
    });
    
    return await response.json();
};

// Example usage
const result = await startCollaboration("I want a sales agent for my Vietnamese restaurant");
```

**Response Format**:
```javascript
{
    "success": true,
    "conversation_id": "conv-abc123",
    "current_state": "skeleton_review",
    "ami_message": "I understand you want to create an agent for your Vietnamese restaurant business! Here's what I'm thinking...",
    "data": {
        "understanding": "You want to streamline restaurant operations...",
        "clarifying_questions": [
            "Will this agent primarily handle customer orders and reservations?",
            "Do you need Vietnamese language support for local customers?",
            "Should it manage inventory or staff scheduling?"
        ],
        "agent_skeleton": {
            "agent_name": "Vietnamese Restaurant Assistant",
            "agent_purpose": "Specialized AI assistant for Vietnamese restaurant operations",
            "target_users": "Restaurant staff, managers, and customers",
            "use_cases": ["Order taking", "Reservation management", "Menu recommendations"],
            "agent_type": "support",
            "language": "vietnamese",
            "key_capabilities": ["Vietnamese/English bilingual", "Food service expertise"],
            "required_tools": ["search", "context", "business_logic"],
            "success_criteria": ["Improved customer satisfaction", "Reduced order errors"]
        }
    },
    "next_actions": [
        "Approve this plan to build the agent",
        "Request changes to any part of the plan",
        "Ask for clarification on capabilities"
    ]
}
```

### Step 2: Review and Provide Feedback

```javascript
const provideFeedback = async (conversationId, feedback) => {
    const response = await fetch('http://localhost:5001/ami/collaborate', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${userToken}`
        },
        body: JSON.stringify({
            user_input: feedback,
            conversation_id: conversationId,
            current_state: "skeleton_review",
            llm_provider: "anthropic"
        })
    });
    
    return await response.json();
};

// Example feedback
const feedback = await provideFeedback(
    "conv-abc123", 
    "I love it! But also add inventory management and make it more focused on takeout orders"
);
```

### Step 3: Approve and Build

```javascript
const approveAgent = async (conversationId) => {
    const response = await fetch('http://localhost:5001/ami/collaborate', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${userToken}`
        },
        body: JSON.stringify({
            user_input: "Perfect! Build it!",
            conversation_id: conversationId,
            current_state: "skeleton_review",
            llm_provider: "anthropic"
        })
    });
    
    return await response.json();
};

// Final approval
const finalResult = await approveAgent("conv-abc123");
```

**Final Success Response**:
```javascript
{
    "success": true,
    "conversation_id": "conv-abc123",
    "current_state": "completed",
    "ami_message": "🎉 Perfect! I've successfully created 'Vietnamese Takeout Operations Assistant'!",
    "data": {
        "agent_id": "agent-xyz789",
        "agent_name": "Vietnamese Takeout Operations Assistant",
        "agent_config": {
            "capabilities": ["Takeout optimization", "Inventory management"],
            "tools": ["search", "context", "business_logic", "file_access"],
            "language": "vietnamese"
        }
    },
    "next_actions": [
        "Start using 'Vietnamese Takeout Operations Assistant' for your tasks",
        "Create another agent",
        "Test the agent with a sample task"
    ]
}
```

### Detailed Flow Example: Google Drive Sales Agent

Here's a complete example showing the exact API flow for creating a Google Drive sales opportunity agent:

**Step 1: Initial Request**
```javascript
// Frontend sends
const response = await fetch('http://localhost:5001/ami/collaborate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token}` },
    body: JSON.stringify({
        user_input: "Build me an agent to read google drive folder then spot out sale opportunities from spreadsheets",
        current_state: "initial_idea",
        llm_provider: "anthropic"
    })
});

// Ami responds with detailed proposal
{
    "success": true,
    "conversation_id": "conv-sales-123",
    "current_state": "skeleton_review",
    "ami_message": "I understand you want an agent to analyze Google Drive spreadsheets for sales opportunities! Here's what I'm thinking...",
    "data": {
        "understanding": "You need automated sales opportunity detection from your Google Drive spreadsheets...",
        "clarifying_questions": [
            "Should this focus on B2B or B2C opportunities?",
            "Do you want automatic notifications when opportunities are found?",
            "Which spreadsheet formats should it prioritize?"
        ],
        "agent_skeleton": {
            "agent_name": "Google Drive Sales Opportunity Analyzer",
            "agent_purpose": "Automatically scan Google Drive spreadsheets to identify and highlight potential sales opportunities",
            "target_users": "Sales teams, business analysts, sales managers",
            "use_cases": ["Spreadsheet analysis", "Lead identification", "Sales opportunity scoring"],
            "agent_type": "analytics",
            "language": "english",
            "key_capabilities": ["Google Drive integration", "Spreadsheet parsing", "Sales pattern recognition"],
            "required_tools": ["google_drive", "file_access", "business_logic", "context"],
            "success_criteria": ["Accurate opportunity identification", "Time saved on manual review"]
        }
    },
    "next_actions": [
        "Approve this plan to build the agent",
        "Request changes to any part of the plan",
        "Ask for clarification on capabilities"
    ]
}
```

**Step 2: User Provides Feedback**
```javascript
// Frontend sends refinement request
{
    user_input: "I love it but make it focus more on B2B opportunities and add email integration to notify the team",
    conversation_id: "conv-sales-123",
    current_state: "skeleton_review",
    llm_provider: "anthropic"
}

// Ami refines the proposal
{
    "success": true,
    "conversation_id": "conv-sales-123",
    "current_state": "skeleton_review", // Still in review!
    "ami_message": "Perfect feedback! I've refined the agent to focus specifically on B2B opportunities and added email notifications...",
    "data": {
        "agent_skeleton": {
            "agent_name": "B2B Sales Opportunity Analyzer with Notifications",
            "agent_purpose": "Scan Google Drive spreadsheets to identify B2B sales opportunities and notify team members",
            "target_users": "B2B sales teams, account managers, sales directors",
            "use_cases": ["B2B lead identification", "Opportunity scoring", "Team notifications"],
            "agent_type": "analytics",
            "language": "english",
            "key_capabilities": ["B2B opportunity scoring", "Email notifications", "Team alerts", "Google Drive integration"],
            "required_tools": ["google_drive", "file_access", "business_logic", "context", "email_notifications"],
            "success_criteria": ["High-quality B2B leads identified", "Timely team notifications", "Reduced manual review time"]
        }
    },
    "next_actions": [
        "Approve this refined plan",
        "Request additional changes",
        "Ask questions about the B2B focus"
    ]
}
```

**Step 3: Final Approval & Build**
```javascript
// Frontend sends approval
{
    user_input: "Perfect! Build it!",
    conversation_id: "conv-sales-123", 
    current_state: "skeleton_review",
    llm_provider: "anthropic"
}

// Ami builds and saves the agent
{
    "success": true,
    "conversation_id": "conv-sales-123",
    "current_state": "completed",
    "ami_message": "🎉 Perfect! I've successfully created 'B2B Sales Opportunity Analyzer with Notifications'!",
    "data": {
        "agent_id": "agent-b2b-sales-analyzer-456",
        "agent_name": "B2B Sales Opportunity Analyzer with Notifications",
        "agent_config": {
            "capabilities": ["Google Drive integration", "B2B opportunity detection", "Email notifications"],
            "tools": ["google_drive", "file_access", "business_logic", "context", "email_notifications"],
            "agent_type": "analytics",
            "language": "english"
        }
    },
    "next_actions": [
        "Start using 'B2B Sales Opportunity Analyzer with Notifications'",
        "Test it with your Google Drive folder",
        "Create another specialized agent"
    ]
}
```

**Frontend Implementation Notes:**
- The `current_state` determines which UI to show (input form vs skeleton review vs success)
- Users can iterate in `skeleton_review` state multiple times before approving
- Always check for `agent_skeleton` in the response data to display the proposed plan
- The `conversation_id` must be passed in subsequent requests to maintain context
- Look for approval keywords like "build it", "perfect", "love it" to trigger final creation

## Direct Approach (Legacy)

### One-Shot Agent Creation

**Endpoint**: `POST /ami/create-agent`

```javascript
const createAgentDirect = async (description) => {
    const response = await fetch('http://localhost:5001/ami/create-agent', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${userToken}`
        },
        body: JSON.stringify({
            user_request: description,
            llm_provider: "anthropic", // or "openai"
            model: "claude-3-5-sonnet-20241022" // optional
        })
    });
    
    return await response.json();
};

// Example usage
const result = await createAgentDirect(
    "Create a Vietnamese sales agent that can analyze documents and help with customer service"
);
```

**Response Format**:
```javascript
{
    "success": true,
    "agent_id": "agent-direct123",
    "agent_name": "Vietnamese Sales Document Specialist",
    "message": "✅ Created 'Vietnamese Sales Document Specialist' successfully!",
    "agent_config": {
        "name": "Vietnamese Sales Document Specialist",
        "description": "Analyzes Vietnamese sales documents and provides customer service",
        "agent_type": "sales",
        "language": "vietnamese",
        "tools": ["search", "context", "business_logic"],
        "knowledge": ["sales_techniques", "product_information"]
    }
}
```

## Frontend Implementation Examples

### React Component for Collaborative Agent Creation

```jsx
import React, { useState } from 'react';

const CollaborativeAgentBuilder = () => {
    const [conversationId, setConversationId] = useState(null);
    const [currentState, setCurrentState] = useState('initial_idea');
    const [userInput, setUserInput] = useState('');
    const [amiResponse, setAmiResponse] = useState(null);
    const [loading, setLoading] = useState(false);

    const sendToAmi = async () => {
        setLoading(true);
        
        try {
            const response = await fetch('http://localhost:5001/ami/collaborate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${localStorage.getItem('token')}`
                },
                body: JSON.stringify({
                    user_input: userInput,
                    conversation_id: conversationId,
                    current_state: currentState,
                    llm_provider: "anthropic"
                })
            });

            const result = await response.json();
            
            if (result.success) {
                setConversationId(result.conversation_id);
                setCurrentState(result.current_state);
                setAmiResponse(result);
                setUserInput('');
            } else {
                console.error('Error:', result.error);
            }
        } catch (error) {
            console.error('Request failed:', error);
        } finally {
            setLoading(false);
        }
    };

    const quickApprove = () => {
        setUserInput("Perfect! Build it!");
        sendToAmi();
    };

    return (
        <div className="collaborative-builder">
            <h2>🤝 Create Agent with Ami</h2>
            
            {/* Ami's Response */}
            {amiResponse && (
                <div className="ami-response">
                    <div className="ami-message">
                        <strong>Ami:</strong> {amiResponse.ami_message}
                    </div>
                    
                    {/* Show Agent Skeleton if available */}
                    {amiResponse.data?.agent_skeleton && (
                        <div className="agent-skeleton">
                            <h3>📋 Proposed Agent Plan:</h3>
                            <div className="skeleton-details">
                                <p><strong>Name:</strong> {amiResponse.data.agent_skeleton.agent_name}</p>
                                <p><strong>Purpose:</strong> {amiResponse.data.agent_skeleton.agent_purpose}</p>
                                <p><strong>Type:</strong> {amiResponse.data.agent_skeleton.agent_type}</p>
                                <p><strong>Language:</strong> {amiResponse.data.agent_skeleton.language}</p>
                                <p><strong>Tools:</strong> {amiResponse.data.agent_skeleton.required_tools.join(', ')}</p>
                            </div>
                        </div>
                    )}
                    
                    {/* Show Next Actions */}
                    {amiResponse.next_actions && (
                        <div className="next-actions">
                            <h4>What you can do next:</h4>
                            <ul>
                                {amiResponse.next_actions.map((action, index) => (
                                    <li key={index}>{action}</li>
                                ))}
                            </ul>
                        </div>
                    )}
                </div>
            )}

            {/* Input Area */}
            <div className="input-area">
                <textarea
                    value={userInput}
                    onChange={(e) => setUserInput(e.target.value)}
                    placeholder={
                        currentState === 'initial_idea' 
                            ? "Describe what kind of agent you want to create..."
                            : "Provide feedback or say 'Perfect! Build it!' to approve..."
                    }
                    rows={4}
                />
                
                <div className="action-buttons">
                    <button 
                        onClick={sendToAmi} 
                        disabled={loading || !userInput.trim()}
                    >
                        {loading ? 'Sending...' : 'Send to Ami'}
                    </button>
                    
                    {currentState === 'skeleton_review' && (
                        <button 
                            onClick={quickApprove}
                            className="approve-button"
                            disabled={loading}
                        >
                            ✅ Quick Approve
                        </button>
                    )}
                </div>
            </div>

            {/* Show Success */}
            {currentState === 'completed' && amiResponse?.data?.agent_id && (
                <div className="success-message">
                    <h3>🎉 Agent Created Successfully!</h3>
                    <p><strong>Agent ID:</strong> {amiResponse.data.agent_id}</p>
                    <p><strong>Name:</strong> {amiResponse.data.agent_name}</p>
                    <button onClick={() => window.location.href = `/agents/${amiResponse.data.agent_id}`}>
                        Start Using Agent
                    </button>
                </div>
            )}
        </div>
    );
};

export default CollaborativeAgentBuilder;
```

### Simple Direct Creation Component

```jsx
import React, { useState } from 'react';

const DirectAgentBuilder = () => {
    const [description, setDescription] = useState('');
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);

    const createAgent = async () => {
        setLoading(true);
        
        try {
            const response = await fetch('http://localhost:5001/ami/create-agent', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${localStorage.getItem('token')}`
                },
                body: JSON.stringify({
                    user_request: description,
                    llm_provider: "anthropic"
                })
            });

            const result = await response.json();
            setResult(result);
        } catch (error) {
            console.error('Creation failed:', error);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="direct-builder">
            <h2>⚡ Quick Agent Creation</h2>
            
            <textarea
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="Describe your agent (e.g., 'Create a Vietnamese customer service agent that can handle orders and complaints')"
                rows={4}
            />
            
            <button 
                onClick={createAgent}
                disabled={loading || !description.trim()}
            >
                {loading ? 'Creating...' : 'Create Agent'}
            </button>

            {result && (
                <div className={`result ${result.success ? 'success' : 'error'}`}>
                    <p>{result.message}</p>
                    {result.success && (
                        <div>
                            <p><strong>Agent ID:</strong> {result.agent_id}</p>
                            <p><strong>Name:</strong> {result.agent_name}</p>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};

export default DirectAgentBuilder;
```

### Vue.js Implementation

```vue
<template>
  <div class="collaborative-builder">
    <h2>🤝 Create Agent with Ami</h2>
    
    <!-- Ami's Response -->
    <div v-if="amiResponse" class="ami-response">
      <div class="ami-message">
        <strong>Ami:</strong> {{ amiResponse.ami_message }}
      </div>
      
      <!-- Agent Skeleton -->
      <div v-if="amiResponse.data?.agent_skeleton" class="agent-skeleton">
        <h3>📋 Proposed Agent Plan:</h3>
        <div class="skeleton-details">
          <p><strong>Name:</strong> {{ amiResponse.data.agent_skeleton.agent_name }}</p>
          <p><strong>Purpose:</strong> {{ amiResponse.data.agent_skeleton.agent_purpose }}</p>
          <p><strong>Type:</strong> {{ amiResponse.data.agent_skeleton.agent_type }}</p>
          <p><strong>Language:</strong> {{ amiResponse.data.agent_skeleton.language }}</p>
          <p><strong>Tools:</strong> {{ amiResponse.data.agent_skeleton.required_tools.join(', ') }}</p>
        </div>
      </div>
      
      <!-- Next Actions -->
      <div v-if="amiResponse.next_actions" class="next-actions">
        <h4>What you can do next:</h4>
        <ul>
          <li v-for="(action, index) in amiResponse.next_actions" :key="index">
            {{ action }}
          </li>
        </ul>
      </div>
    </div>

    <!-- Input Area -->
    <div class="input-area">
      <textarea
        v-model="userInput"
        :placeholder="inputPlaceholder"
        rows="4"
      ></textarea>
      
      <div class="action-buttons">
        <button 
          @click="sendToAmi" 
          :disabled="loading || !userInput.trim()"
        >
          {{ loading ? 'Sending...' : 'Send to Ami' }}
        </button>
        
        <button 
          v-if="currentState === 'skeleton_review'"
          @click="quickApprove"
          class="approve-button"
          :disabled="loading"
        >
          ✅ Quick Approve
        </button>
      </div>
    </div>

    <!-- Success Message -->
    <div v-if="currentState === 'completed' && amiResponse?.data?.agent_id" class="success-message">
      <h3>🎉 Agent Created Successfully!</h3>
      <p><strong>Agent ID:</strong> {{ amiResponse.data.agent_id }}</p>
      <p><strong>Name:</strong> {{ amiResponse.data.agent_name }}</p>
      <button @click="goToAgent">Start Using Agent</button>
    </div>
  </div>
</template>

<script>
export default {
  name: 'CollaborativeAgentBuilder',
  data() {
    return {
      conversationId: null,
      currentState: 'initial_idea',
      userInput: '',
      amiResponse: null,
      loading: false
    }
  },
  computed: {
    inputPlaceholder() {
      return this.currentState === 'initial_idea' 
        ? "Describe what kind of agent you want to create..."
        : "Provide feedback or say 'Perfect! Build it!' to approve...";
    }
  },
  methods: {
    async sendToAmi() {
      this.loading = true;
      
      try {
        const response = await fetch('http://localhost:5001/ami/collaborate', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${localStorage.getItem('token')}`
          },
          body: JSON.stringify({
            user_input: this.userInput,
            conversation_id: this.conversationId,
            current_state: this.currentState,
            llm_provider: "anthropic"
          })
        });

        const result = await response.json();
        
        if (result.success) {
          this.conversationId = result.conversation_id;
          this.currentState = result.current_state;
          this.amiResponse = result;
          this.userInput = '';
        } else {
          console.error('Error:', result.error);
        }
      } catch (error) {
        console.error('Request failed:', error);
      } finally {
        this.loading = false;
      }
    },
    
    quickApprove() {
      this.userInput = "Perfect! Build it!";
      this.sendToAmi();
    },
    
    goToAgent() {
      this.$router.push(`/agents/${this.amiResponse.data.agent_id}`);
    }
  }
}
</script>
```

## Using Created Agents

Once an agent is created, you can use it through the agent execution endpoints:

### Execute Agent (Single Response)

**Endpoint**: `POST /agent/execute`

```javascript
const useAgent = async (agentId, userMessage, mode = "execute") => {
    const response = await fetch('http://localhost:5001/agent/execute', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${userToken}`
        },
        body: JSON.stringify({
            agent_identifier: agentId,
            user_request: userMessage,
            agent_mode: mode, // "execute" or "collaborate"
            llm_provider: "anthropic"
        })
    });
    
    return await response.json();
};

// Example usage
const result = await useAgent(
    "agent-xyz789", 
    "Help me process this customer order: 2 pho bo, 1 spring rolls",
    "execute"
);
```

### Stream Agent Response (Real-time)

**Endpoint**: `POST /agent/stream`

```javascript
const streamAgent = async (agentId, userMessage, onChunk, mode = "execute") => {
    const response = await fetch('http://localhost:5001/agent/stream', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${userToken}`
        },
        body: JSON.stringify({
            agent_identifier: agentId,
            user_request: userMessage,
            agent_mode: mode,
            llm_provider: "anthropic"
        })
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');
        
        for (const line of lines) {
            if (line.startsWith('data: ')) {
                const data = line.slice(6);
                if (data !== '[DONE]') {
                    try {
                        const parsed = JSON.parse(data);
                        onChunk(parsed);
                    } catch (e) {
                        console.error('Failed to parse chunk:', data);
                    }
                }
            }
        }
    }
};

// Example usage
await streamAgent(
    "agent-xyz789",
    "What are today's specials?",
    (chunk) => {
        console.log('Received chunk:', chunk);
        // Update UI with streaming response
    },
    "collaborate"
);
```

### Agent Modes

**Execute Mode** (Default):
- Task-focused and efficient
- Direct problem-solving
- Minimal back-and-forth
- Best for: Specific tasks, data processing, quick answers

**Collaborate Mode**:
- Discussion-oriented
- Asks clarifying questions
- Interactive and exploratory
- Best for: Planning, brainstorming, complex problem-solving

```javascript
// Execute mode - Direct task completion
const executeResult = await useAgent(
    "sales-agent-123",
    "Process this sales report and give me the key metrics",
    "execute"
);

// Collaborate mode - Interactive discussion
const collaborateResult = await useAgent(
    "sales-agent-123",
    "I want to improve our sales process",
    "collaborate"
);
```

## Error Handling

### Common Error Responses

```javascript
// Authentication Error
{
    "detail": "Not authenticated"
}

// Invalid Input
{
    "success": false,
    "error": "user_input field is required",
    "message": "❌ Agent creation failed"
}

// Conversation Lost
{
    "success": false,
    "conversation_id": "unknown",
    "ami_message": "I lost track of our conversation. Let's start fresh!",
    "error": "Conversation state not found"
}

// Agent Not Found
{
    "success": false,
    "error": "Agent not found or access denied",
    "message": "❌ Could not find agent with identifier: invalid-agent-id"
}
```

### Error Handling Implementation

```javascript
const handleApiCall = async (url, data) => {
    try {
        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${localStorage.getItem('token')}`
            },
            body: JSON.stringify(data)
        });

        if (!response.ok) {
            if (response.status === 401) {
                // Redirect to login
                window.location.href = '/login';
                return;
            }
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const result = await response.json();
        
        if (!result.success) {
            throw new Error(result.error || result.message || 'Unknown error');
        }
        
        return result;
        
    } catch (error) {
        console.error('API call failed:', error);
        throw error;
    }
};

// Usage with error handling
const createAgentSafely = async (description) => {
    try {
        return await handleApiCall('http://localhost:5001/ami/create-agent', {
            user_request: description,
            llm_provider: "anthropic"
        });
    } catch (error) {
        // Show user-friendly error message
        showErrorNotification(`Failed to create agent: ${error.message}`);
        return null;
    }
};
```

## Best Practices

### 1. State Management

```javascript
// Store conversation state in localStorage or state management
const saveConversationState = (conversationId, state) => {
    localStorage.setItem('ami_conversation', JSON.stringify({
        id: conversationId,
        state: state,
        timestamp: Date.now()
    }));
};

const loadConversationState = () => {
    const saved = localStorage.getItem('ami_conversation');
    if (saved) {
        const parsed = JSON.parse(saved);
        // Check if conversation is recent (within 1 hour)
        if (Date.now() - parsed.timestamp < 3600000) {
            return parsed;
        }
    }
    return null;
};

// Redux/Zustand store example
const useAgentStore = create((set, get) => ({
    conversations: {},
    activeConversation: null,
    
    startConversation: (id, state) => set((prev) => ({
        conversations: {
            ...prev.conversations,
            [id]: { id, state, timestamp: Date.now() }
        },
        activeConversation: id
    })),
    
    updateConversation: (id, updates) => set((prev) => ({
        conversations: {
            ...prev.conversations,
            [id]: { ...prev.conversations[id], ...updates }
        }
    }))
}));
```

### 2. User Experience Enhancements

```javascript
// Debounced auto-save for drafts
import { debounce } from 'lodash';

const saveDraft = debounce((text) => {
    localStorage.setItem('ami_draft', text);
}, 500);

// Typing indicator
const useTypingIndicator = () => {
    const [isTyping, setIsTyping] = useState(false);
    
    const showTyping = () => {
        setIsTyping(true);
        setTimeout(() => setIsTyping(false), 2000);
    };
    
    return { isTyping, showTyping };
};

// Progress indicator for multi-step process
const ProgressIndicator = ({ currentStep, steps }) => (
    <div className="progress-bar">
        {steps.map((step, index) => (
            <div 
                key={step.key}
                className={`step ${index <= currentStep ? 'completed' : 'pending'}`}
            >
                <div className="step-icon">{step.icon}</div>
                <div className="step-label">{step.label}</div>
            </div>
        ))}
    </div>
);
```

### 3. Conversation Flow UI

```jsx
const ConversationFlow = ({ currentState }) => {
    const steps = [
        { key: 'initial_idea', label: 'Share Idea', icon: '💡' },
        { key: 'skeleton_review', label: 'Review Plan', icon: '📋' },
        { key: 'approved', label: 'Approved', icon: '✅' },
        { key: 'completed', label: 'Agent Built', icon: '🎉' }
    ];

    const currentIndex = steps.findIndex(step => step.key === currentState);

    return (
        <div className="conversation-flow">
            {steps.map((step, index) => (
                <div 
                    key={step.key}
                    className={`step ${
                        currentState === step.key ? 'active' : 
                        index < currentIndex ? 'completed' : 'pending'
                    }`}
                >
                    <span className="icon">{step.icon}</span>
                    <span className="label">{step.label}</span>
                    {index < steps.length - 1 && (
                        <div className={`connector ${index < currentIndex ? 'completed' : 'pending'}`}></div>
                    )}
                </div>
            ))}
        </div>
    );
};
```

### 4. Responsive Design Considerations

```css
/* Mobile-first responsive design */
.collaborative-builder {
    padding: 1rem;
    max-width: 800px;
    margin: 0 auto;
}

.input-area textarea {
    width: 100%;
    min-height: 120px;
    padding: 1rem;
    border: 2px solid #e1e5e9;
    border-radius: 8px;
    font-family: inherit;
    resize: vertical;
}

.action-buttons {
    display: flex;
    gap: 1rem;
    margin-top: 1rem;
    flex-wrap: wrap;
}

.action-buttons button {
    flex: 1;
    min-width: 120px;
    padding: 0.75rem 1.5rem;
    border-radius: 6px;
    font-weight: 500;
    transition: all 0.2s;
}

/* Tablet and desktop */
@media (min-width: 768px) {
    .collaborative-builder {
        padding: 2rem;
    }
    
    .action-buttons {
        flex-wrap: nowrap;
    }
    
    .action-buttons button {
        flex: none;
    }
}
```

### 5. Performance Optimization

```javascript
// Memoize expensive components
const MemoizedAgentSkeleton = React.memo(({ skeleton }) => (
    <div className="agent-skeleton">
        <h3>📋 Proposed Agent Plan:</h3>
        <div className="skeleton-details">
            <p><strong>Name:</strong> {skeleton.agent_name}</p>
            <p><strong>Purpose:</strong> {skeleton.agent_purpose}</p>
            <p><strong>Type:</strong> {skeleton.agent_type}</p>
            <p><strong>Language:</strong> {skeleton.language}</p>
            <p><strong>Tools:</strong> {skeleton.required_tools.join(', ')}</p>
        </div>
    </div>
));

// Use React Query for caching API calls
import { useQuery, useMutation } from 'react-query';

const useCreateAgent = () => {
    return useMutation(
        (agentData) => handleApiCall('http://localhost:5001/ami/create-agent', agentData),
        {
            onSuccess: (data) => {
                // Invalidate and refetch agent list
                queryClient.invalidateQueries('agents');
            },
            onError: (error) => {
                console.error('Agent creation failed:', error);
            }
        }
    );
};

// Lazy load components
const LazyDirectAgentBuilder = React.lazy(() => import('./DirectAgentBuilder'));
```

### 6. Accessibility

```jsx
// Add proper ARIA labels and keyboard navigation
const AccessibleAgentBuilder = () => {
    const [announcements, setAnnouncements] = useState('');
    
    const announce = (message) => {
        setAnnouncements(message);
        setTimeout(() => setAnnouncements(''), 1000);
    };

    return (
        <div className="collaborative-builder" role="main">
            {/* Screen reader announcements */}
            <div 
                aria-live="polite" 
                aria-atomic="true" 
                className="sr-only"
            >
                {announcements}
            </div>
            
            <h2 id="builder-title">Create Agent with Ami</h2>
            
            <textarea
                aria-labelledby="builder-title"
                aria-describedby="input-help"
                onFocus={() => announce('Agent description input focused')}
            />
            
            <div id="input-help" className="help-text">
                Describe what kind of agent you want to create
            </div>
            
            <button
                onClick={sendToAmi}
                aria-describedby="send-help"
                disabled={loading}
            >
                {loading ? 'Sending...' : 'Send to Ami'}
            </button>
        </div>
    );
};
```

## Testing Your Implementation

### Unit Tests

```javascript
// Jest tests for API functions
describe('Ami API Functions', () => {
    beforeEach(() => {
        fetch.resetMocks();
    });

    test('startCollaboration sends correct request', async () => {
        fetch.mockResponseOnce(JSON.stringify({
            success: true,
            conversation_id: 'test-conv-123',
            current_state: 'skeleton_review'
        }));

        const result = await startCollaboration('Create a test agent');
        
        expect(fetch).toHaveBeenCalledWith(
            'http://localhost:5001/ami/collaborate',
            expect.objectContaining({
                method: 'POST',
                headers: expect.objectContaining({
                    'Content-Type': 'application/json'
                }),
                body: JSON.stringify({
                    user_input: 'Create a test agent',
                    current_state: 'initial_idea',
                    llm_provider: 'anthropic'
                })
            })
        );

        expect(result.success).toBe(true);
        expect(result.conversation_id).toBe('test-conv-123');
    });

    test('handles API errors gracefully', async () => {
        fetch.mockRejectOnce(new Error('Network error'));

        await expect(startCollaboration('Test')).rejects.toThrow('Network error');
    });
});
```

### Integration Tests

```javascript
// Cypress end-to-end tests
describe('Agent Creation Flow', () => {
    it('completes collaborative agent creation', () => {
        cy.visit('/create-agent');
        
        // Start conversation
        cy.get('[data-testid="agent-description"]')
            .type('Create a customer service agent for my restaurant');
        
        cy.get('[data-testid="send-to-ami"]').click();
        
        // Wait for Ami's response
        cy.get('[data-testid="ami-response"]', { timeout: 10000 })
            .should('be.visible');
        
        // Approve the plan
        cy.get('[data-testid="quick-approve"]').click();
        
        // Verify success
        cy.get('[data-testid="success-message"]')
            .should('contain', 'Agent Created Successfully!');
    });
});
```

## Quick Start Checklist

- [ ] **Authentication Setup**: Ensure user token is available and valid
- [ ] **Base URL Configuration**: Set to `http://localhost:5001`
- [ ] **Error Handling**: Implement proper error catching and user feedback
- [ ] **State Management**: Track conversation ID and current state
- [ ] **UI Feedback**: Show loading states, progress indicators, and success messages
- [ ] **Responsive Design**: Ensure components work on different screen sizes
- [ ] **Accessibility**: Add proper ARIA labels and keyboard navigation
- [ ] **Performance**: Implement memoization and lazy loading where appropriate
- [ ] **Testing**: Write unit and integration tests for critical flows
- [ ] **Documentation**: Document component props and API integration

## API Reference Summary

| Endpoint | Method | Purpose | Key Parameters |
|----------|--------|---------|----------------|
| `/ami/collaborate` | POST | Multi-step agent creation | `user_input`, `current_state`, `conversation_id` |
| `/ami/create-agent` | POST | Direct agent creation | `user_request`, `llm_provider` |
| `/agent/execute` | POST | Execute agent (single response) | `agent_identifier`, `user_request`, `agent_mode` |
| `/agent/stream` | POST | Execute agent (streaming) | `agent_identifier`, `user_request`, `agent_mode` |
| `/ami/chat` | POST | Simple chat with Ami | `user_message`, `llm_provider` |

## Conclusion

Your frontend now has everything needed to create powerful AI agents using Ami's Chief Product Officer approach. Users can choose between:

- **🤝 Collaborative experience** - Thorough requirement gathering with human approval
- **⚡ Quick creation** - Fast agent generation for simple needs  
- **🎯 Dual-mode execution** - Agents that can collaborate or execute tasks based on context

The implementation provides a robust, user-friendly interface for the complete agent lifecycle: creation, configuration, and execution.

**Happy building!** 🚀 