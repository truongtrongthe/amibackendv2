# Frontend Guide: How to Use Agent.py

## Table of Contents
- [Overview](#overview)
- [Available API Endpoints](#available-api-endpoints)
- [Request Model](#request-model)
- [Frontend Implementation Examples](#frontend-implementation-examples)
- [Agent Modes Explained](#agent-modes-explained)
- [Advanced Usage Examples](#advanced-usage-examples)
- [Error Handling](#error-handling)
- [Best Practices](#best-practices)
- [Quick Reference](#quick-reference)

## Overview

Agent.py provides a dynamic, database-driven system for executing specialized AI agents. Unlike hardcoded systems, agents load their configuration (system prompts, tools, knowledge) dynamically from the database, making them highly flexible and customizable.

**Key Features:**
- **Dynamic Configuration**: Agents load prompts and tools from database
- **Dual Modes**: "collaborate" (interactive) and "execute" (task-focused)
- **Real-time Streaming**: Server-sent events for live responses
- **Google Drive Integration**: Automatic document processing
- **Authentication**: JWT-based user authentication
- **Organization Support**: Multi-tenant agent access

**Base URL**: `http://localhost:5001`

## Available API Endpoints

### 1. Execute Agent (Single Response)
**Endpoint**: `POST /agent/execute`

- **Purpose**: Execute an agent and get complete response
- **Authentication**: Required (`Bearer` token)
- **Use case**: When you need a full response at once
- **Response Type**: JSON

### 2. Stream Agent (Real-time Response)
**Endpoint**: `POST /agent/stream`

- **Purpose**: Stream agent execution with real-time updates
- **Authentication**: Required (`Bearer` token)
- **Use case**: For interactive chat-like experiences
- **Response Type**: Server-Sent Events (SSE)

### 3. Legacy Agent Endpoint
**Endpoint**: `POST /api/tool/agent`

- **Purpose**: Backward compatibility (avoid using for new development)
- **Authentication**: Not required
- **Status**: Deprecated

## Request Model

### AgentAPIRequest Schema

```typescript
interface AgentAPIRequest {
    // Required fields
    agent_id: string;              // Agent ID or name
    user_request: string;          // Task for the agent to perform
    
    // Optional configuration
    agent_mode?: "execute" | "collaborate";  // Default: "execute"
    llm_provider?: "anthropic" | "openai";   // Default: "anthropic"
    model?: string;                          // Specific model (e.g., "claude-3-5-sonnet-20241022")
    agent_type?: string;                     // Agent type hint
    specialized_knowledge_domains?: string[]; // Knowledge areas
    conversation_history?: ConversationMessage[]; // Previous messages
}

interface ConversationMessage {
    role: "user" | "agent";
    content: string;
    timestamp: string;
}
```

### Example Request Payload

```json
{
    "agent_id": "B2B Sales Analyzer",
    "user_request": "Analyze this Google Drive folder for sales opportunities",
    "agent_mode": "execute",
    "llm_provider": "anthropic",
    "model": "claude-3-5-sonnet-20241022",
    "specialized_knowledge_domains": ["sales", "b2b", "crm"]
}
```

## Frontend Implementation Examples

### JavaScript/Fetch API

#### Execute Agent (Single Response)

```javascript
const executeAgent = async (agentId, userRequest, mode = "execute") => {
    try {
        const response = await fetch('http://localhost:5001/agent/execute', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${localStorage.getItem('token')}`
            },
            body: JSON.stringify({
                agent_id: agentId,
                user_request: userRequest,
                agent_mode: mode,
                llm_provider: "anthropic"
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const result = await response.json();
        return result;
        
    } catch (error) {
        console.error('Agent execution failed:', error);
        throw error;
    }
};

// Usage examples
const result1 = await executeAgent(
    "B2B Sales Analyzer", 
    "Analyze this Google Drive folder for sales opportunities",
    "execute"
);

const result2 = await executeAgent(
    "customer-service-bot",
    "I need help understanding our return policy",
    "collaborate"
);
```

#### Stream Agent (Real-time)

```javascript
const streamAgent = async (agentId, userRequest, onChunk, mode = "execute") => {
    try {
        const response = await fetch('http://localhost:5001/agent/stream', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${localStorage.getItem('token')}`
            },
            body: JSON.stringify({
                agent_id: agentId,
                user_request: userRequest,
                agent_mode: mode,
                llm_provider: "anthropic"
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

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
    } catch (error) {
        console.error('Agent streaming failed:', error);
        throw error;
    }
};

// Usage example
await streamAgent(
    "content-writer",
    "Write a blog post about AI agents",
    (chunk) => {
        console.log('Chunk type:', chunk.type);
        console.log('Content:', chunk.content);
        
        // Handle different chunk types
        switch(chunk.type) {
            case 'status':
                showStatus(chunk.content);
                break;
            case 'thinking':
                showThinking(chunk.content);
                break;
            case 'response_chunk':
                appendToResponse(chunk.content);
                break;
            case 'complete':
                showComplete();
                break;
            case 'error':
                showError(chunk.content);
                break;
        }
    },
    "collaborate"
);
```

### React Implementation

#### React Hook for Agent Execution

```jsx
import { useState, useCallback } from 'react';

const useAgent = () => {
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [result, setResult] = useState(null);

    const executeAgent = useCallback(async (agentId, userRequest, mode = "execute") => {
        setLoading(true);
        setError(null);
        setResult(null);

        try {
            const response = await fetch('http://localhost:5001/agent/execute', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${localStorage.getItem('token')}`
                },
                body: JSON.stringify({
                    agent_id: agentId,
                    user_request: userRequest,
                    agent_mode: mode,
                    llm_provider: "anthropic"
                })
            });

            if (!response.ok) {
                throw new Error(`Agent execution failed: ${response.statusText}`);
            }

            const data = await response.json();
            
            if (data.success) {
                setResult(data);
            } else {
                throw new Error(data.error || 'Agent execution failed');
            }
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    }, []);

    return { executeAgent, loading, error, result };
};

// Usage in component
const AgentChat = () => {
    const { executeAgent, loading, error, result } = useAgent();
    const [input, setInput] = useState('');

    const handleSubmit = async (e) => {
        e.preventDefault();
        await executeAgent("sales-assistant", input, "collaborate");
    };

    return (
        <div className="agent-chat">
            <form onSubmit={handleSubmit}>
                <input 
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    placeholder="Ask your agent..."
                    disabled={loading}
                />
                <button type="submit" disabled={loading || !input.trim()}>
                    {loading ? 'Processing...' : 'Send'}
                </button>
            </form>

            {error && (
                <div className="error">
                    ❌ Error: {error}
                </div>
            )}

            {result && (
                <div className="result">
                    <h3>Agent: {result.agent_id}</h3>
                    <div className="response">
                        {result.result}
                    </div>
                    <div className="metadata">
                        <small>
                            Execution time: {result.execution_time?.toFixed(2)}s | 
                            Tasks completed: {result.tasks_completed}
                        </small>
                    </div>
                </div>
            )}
        </div>
    );
};

export default AgentChat;
```

#### React Streaming Component

```jsx
import { useState, useCallback } from 'react';

const StreamingAgentChat = () => {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');
    const [streaming, setStreaming] = useState(false);
    const [currentResponse, setCurrentResponse] = useState('');

    const streamAgent = useCallback(async (agentId, userRequest, mode = "execute") => {
        setStreaming(true);
        setCurrentResponse('');
        
        // Add user message
        setMessages(prev => [...prev, {
            role: 'user',
            content: userRequest,
            timestamp: new Date()
        }]);

        try {
            const response = await fetch('http://localhost:5001/agent/stream', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${localStorage.getItem('token')}`
                },
                body: JSON.stringify({
                    agent_id: agentId,
                    user_request: userRequest,
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
                                
                                if (parsed.type === 'response_chunk') {
                                    setCurrentResponse(prev => prev + parsed.content);
                                } else if (parsed.type === 'complete') {
                                    // Finalize the response
                                    setMessages(prev => [...prev, {
                                        role: 'agent', 
                                        content: currentResponse,
                                        timestamp: new Date(),
                                        agentId: parsed.agent_id
                                    }]);
                                    setCurrentResponse('');
                                }
                            } catch (e) {
                                console.error('Parse error:', e);
                            }
                        }
                    }
                }
            }
        } catch (error) {
            console.error('Streaming error:', error);
            setMessages(prev => [...prev, {
                role: 'error',
                content: `Error: ${error.message}`,
                timestamp: new Date()
            }]);
        } finally {
            setStreaming(false);
        }
    }, [currentResponse]);

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!input.trim() || streaming) return;
        
        const userInput = input;
        setInput('');
        await streamAgent("general-assistant", userInput, "collaborate");
    };

    return (
        <div className="streaming-chat">
            <div className="messages">
                {messages.map((msg, idx) => (
                    <div key={idx} className={`message ${msg.role}`}>
                        <div className="content">{msg.content}</div>
                        <div className="timestamp">
                            {msg.timestamp.toLocaleTimeString()}
                            {msg.agentId && <span> • {msg.agentId}</span>}
                        </div>
                    </div>
                ))}
                
                {/* Show current streaming response */}
                {currentResponse && (
                    <div className="message agent streaming">
                        <div className="content">{currentResponse}</div>
                        <div className="typing-indicator">✍️ Agent is typing...</div>
                    </div>
                )}
            </div>

            <form onSubmit={handleSubmit} className="input-form">
                <input
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    placeholder="Chat with your agent..."
                    disabled={streaming}
                />
                <button type="submit" disabled={streaming || !input.trim()}>
                    {streaming ? '⏳' : '➤'}
                </button>
            </form>
        </div>
    );
};

export default StreamingAgentChat;
```

### Vue.js Implementation

```vue
<template>
  <div class="agent-interface">
    <div class="agent-selector">
      <select v-model="selectedAgent">
        <option value="">Select an agent...</option>
        <option value="sales-assistant">Sales Assistant</option>
        <option value="document-analyzer">Document Analyzer</option>
        <option value="customer-service">Customer Service</option>
      </select>
      
      <div class="mode-selector">
        <label>
          <input type="radio" v-model="agentMode" value="execute" />
          Execute Mode
        </label>
        <label>
          <input type="radio" v-model="agentMode" value="collaborate" />
          Collaborate Mode
        </label>
      </div>
    </div>

    <div class="chat-area">
      <div v-for="message in messages" :key="message.id" :class="`message ${message.role}`">
        <div class="content">{{ message.content }}</div>
        <div class="timestamp">{{ formatTime(message.timestamp) }}</div>
      </div>
      
      <div v-if="loading" class="loading">
        Agent is thinking...
      </div>
    </div>

    <form @submit.prevent="sendMessage" class="input-form">
      <textarea 
        v-model="userInput"
        placeholder="Type your message..."
        :disabled="loading || !selectedAgent"
        rows="3"
      ></textarea>
      <button 
        type="submit" 
        :disabled="loading || !selectedAgent || !userInput.trim()"
      >
        {{ loading ? 'Sending...' : 'Send' }}
      </button>
    </form>
  </div>
</template>

<script>
export default {
  name: 'AgentInterface',
  data() {
    return {
      selectedAgent: '',
      agentMode: 'execute',
      userInput: '',
      messages: [],
      loading: false
    }
  },
  methods: {
    async sendMessage() {
      if (!this.userInput.trim() || !this.selectedAgent) return;
      
      const userMessage = {
        id: Date.now(),
        role: 'user',
        content: this.userInput,
        timestamp: new Date()
      };
      
      this.messages.push(userMessage);
      const currentInput = this.userInput;
      this.userInput = '';
      this.loading = true;

      try {
        const response = await fetch('http://localhost:5001/agent/execute', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${localStorage.getItem('token')}`
          },
          body: JSON.stringify({
            agent_id: this.selectedAgent,
            user_request: currentInput,
            agent_mode: this.agentMode,
            llm_provider: 'anthropic'
          })
        });

        const result = await response.json();
        
        if (result.success) {
          this.messages.push({
            id: Date.now() + 1,
            role: 'agent',
            content: result.result,
            timestamp: new Date(),
            agentId: result.agent_id
          });
        } else {
          throw new Error(result.error || 'Agent execution failed');
        }
      } catch (error) {
        this.messages.push({
          id: Date.now() + 1,
          role: 'error',
          content: `Error: ${error.message}`,
          timestamp: new Date()
        });
      } finally {
        this.loading = false;
      }
    },
    
    formatTime(timestamp) {
      return timestamp.toLocaleTimeString();
    }
  }
}
</script>

<style scoped>
.agent-interface {
  max-width: 800px;
  margin: 0 auto;
  padding: 1rem;
}

.agent-selector {
  margin-bottom: 1rem;
  display: flex;
  gap: 1rem;
  align-items: center;
}

.mode-selector label {
  margin-left: 1rem;
  cursor: pointer;
}

.chat-area {
  height: 400px;
  overflow-y: auto;
  border: 1px solid #ddd;
  padding: 1rem;
  margin-bottom: 1rem;
}

.message {
  margin-bottom: 1rem;
  padding: 0.5rem;
  border-radius: 8px;
}

.message.user {
  background-color: #e3f2fd;
  margin-left: 2rem;
}

.message.agent {
  background-color: #f3e5f5;
  margin-right: 2rem;
}

.message.error {
  background-color: #ffebee;
  color: #c62828;
}

.input-form {
  display: flex;
  gap: 0.5rem;
}

.input-form textarea {
  flex: 1;
  padding: 0.5rem;
  border: 1px solid #ddd;
  border-radius: 4px;
  resize: vertical;
}

.input-form button {
  padding: 0.5rem 1rem;
  background-color: #1976d2;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}

.input-form button:disabled {
  background-color: #ccc;
  cursor: not-allowed;
}
</style>
```

## Agent Modes Explained

### Execute Mode (Default)

**Purpose**: Task-focused, efficient completion

```javascript
// Example: Generate a report
await executeAgent(
    "data-analyst", 
    "Generate a Q4 sales report from this Google Drive folder",
    "execute"
);
```

**Characteristics:**
- Direct problem-solving approach
- Minimal back-and-forth questions
- Uses tools efficiently to gather information
- Provides comprehensive, actionable results
- Optimized for specific task completion

**Best for:**
- Data analysis and reporting
- Document processing
- Information retrieval
- Automated workflows
- Quick answers to specific questions

### Collaborate Mode

**Purpose**: Interactive, discussion-focused assistance

```javascript
// Example: Strategic planning
await executeAgent(
    "business-consultant",
    "I want to improve our customer retention strategy", 
    "collaborate"
);
```

**Characteristics:**
- Asks clarifying questions to understand needs
- Discusses options and alternatives
- Explains reasoning and approach
- Seeks feedback before taking major steps
- Encourages iterative problem-solving

**Best for:**
- Strategic planning and brainstorming
- Complex problem-solving
- Learning and education
- Creative projects
- Situations requiring human input and feedback

### Mode Selection Guide

```javascript
// Use EXECUTE mode when:
const salesReport = await executeAgent("analyst", "Generate monthly sales report", "execute");
const documentSummary = await executeAgent("summarizer", "Summarize this 50-page contract", "execute");
const dataAnalysis = await executeAgent("data-scientist", "Find trends in customer behavior data", "execute");

// Use COLLABORATE mode when:
const strategy = await executeAgent("consultant", "Help me develop a marketing strategy", "collaborate");
const brainstorm = await executeAgent("creative-assistant", "I need ideas for our product launch", "collaborate");
const learning = await executeAgent("tutor", "Teach me about machine learning concepts", "collaborate");
```

## Advanced Usage Examples

### Google Drive Document Analysis

```javascript
const analyzeGoogleDriveDoc = async (driveLink) => {
    const result = await executeAgent(
        "document-analyzer",
        `Analyze this Google Drive document for key insights: ${driveLink}`,
        "execute"
    );
    return result;
};

// Usage examples
const contractAnalysis = await analyzeGoogleDriveDoc(
    "https://docs.google.com/document/d/1ABC123xyz/edit"
);

const folderAnalysis = await executeAgent(
    "sales-analyzer",
    "Analyze all documents in the 'Q4 Sales Reports' Google Drive folder for opportunities and trends",
    "execute"
);
```

### Multi-Agent Workflow

```javascript
const processBusinessPlan = async (planText) => {
    // Step 1: Financial analysis
    const financialAnalysis = await executeAgent(
        "financial-analyst",
        `Analyze the financial projections in this business plan: ${planText}`,
        "execute"
    );
    
    // Step 2: Market analysis  
    const marketAnalysis = await executeAgent(
        "market-researcher", 
        `Analyze the market opportunity described in: ${planText}`,
        "execute"
    );
    
    // Step 3: Strategic recommendations
    const recommendations = await executeAgent(
        "business-strategist",
        `Based on this financial analysis: ${financialAnalysis.result} and market analysis: ${marketAnalysis.result}, provide strategic recommendations`,
        "collaborate"
    );
    
    return {
        financial: financialAnalysis,
        market: marketAnalysis, 
        strategy: recommendations
    };
};

// Usage
const businessPlanText = "..."; // Your business plan content
const analysis = await processBusinessPlan(businessPlanText);
console.log('Complete business plan analysis:', analysis);
```

### Conversation History Management

```javascript
class AgentConversation {
    constructor(agentId, maxHistory = 20) {
        this.agentId = agentId;
        this.history = [];
        this.maxHistory = maxHistory;
    }
    
    async sendMessage(message, mode = "collaborate") {
        // Add user message to history
        const userMessage = {
            role: 'user',
            content: message,
            timestamp: new Date().toISOString()
        };
        this.history.push(userMessage);
        
        // Execute agent with conversation history
        const result = await executeAgent(
            this.agentId,
            message,
            mode,
            this.history.slice(-this.maxHistory) // Keep recent history
        );
        
        // Add agent response to history
        if (result.success) {
            const agentMessage = {
                role: 'agent',
                content: result.result,
                timestamp: new Date().toISOString()
            };
            this.history.push(agentMessage);
        }
        
        // Trim history if too long
        if (this.history.length > this.maxHistory) {
            this.history = this.history.slice(-this.maxHistory);
        }
        
        return result;
    }
    
    getHistory() {
        return this.history;
    }
    
    clearHistory() {
        this.history = [];
    }
}

// Usage
const conversation = new AgentConversation("customer-support-bot");

await conversation.sendMessage("I have a problem with my order", "collaborate");
await conversation.sendMessage("It was supposed to arrive yesterday", "collaborate");
await conversation.sendMessage("Order number is #12345", "collaborate");

console.log('Conversation history:', conversation.getHistory());
```

### Specialized Agent Configurations

```javascript
// Create specialized configurations for different use cases
const AgentConfigs = {
    sales: {
        agent_id: "sales-assistant",
        mode: "collaborate",
        knowledge_domains: ["sales", "crm", "lead_qualification"]
    },
    
    analytics: {
        agent_id: "data-analyst", 
        mode: "execute",
        knowledge_domains: ["data_analysis", "reporting", "business_intelligence"]
    },
    
    support: {
        agent_id: "customer-support",
        mode: "collaborate", 
        knowledge_domains: ["product_knowledge", "troubleshooting", "policies"]
    }
};

// Specialized execution functions
const executeSalesAgent = (request) => executeAgent(
    AgentConfigs.sales.agent_id,
    request,
    AgentConfigs.sales.mode
);

const executeAnalyticsAgent = (request) => executeAgent(
    AgentConfigs.analytics.agent_id,
    request,
    AgentConfigs.analytics.mode
);

const executeSupportAgent = (request) => executeAgent(
    AgentConfigs.support.agent_id,
    request,
    AgentConfigs.support.mode
);

// Usage
const salesLead = await executeSalesAgent("Qualify this lead: John Smith, CEO of TechCorp, interested in our enterprise solution");
const monthlyReport = await executeAnalyticsAgent("Generate monthly performance report for all departments");
const customerHelp = await executeSupportAgent("Customer can't access their account after password reset");
```

## Error Handling

### Response Structure

#### Success Response
```json
{
    "success": true,
    "result": "Agent response content here...",
    "agent_id": "resolved-agent-id",
    "agent_type": "analytics", 
    "execution_time": 2.34,
    "tasks_completed": 1,
    "metadata": {
        "org_id": "org-123",
        "user_id": "user-456",
        "tools_used": ["search", "file_access", "business_logic"],
        "specialized_domains": ["sales", "b2b"]
    }
}
```

#### Error Response
```json
{
    "success": false,
    "result": "",
    "agent_id": "invalid-agent",
    "agent_type": "",
    "execution_time": 0.12,
    "tasks_completed": 0,
    "error": "Agent not found or access denied",
    "metadata": null
}
```

### Comprehensive Error Handling

```javascript
class AgentError extends Error {
    constructor(message, code, details = {}) {
        super(message);
        this.name = 'AgentError';
        this.code = code;
        this.details = details;
    }
}

const handleAgentCall = async (agentId, request, mode = "execute") => {
    try {
        const result = await executeAgent(agentId, request, mode);
        
        if (!result.success) {
            // Handle specific error cases
            if (result.error.includes("Agent not found")) {
                throw new AgentError(
                    `Agent '${agentId}' not found. Please check the agent name or create it first.`,
                    'AGENT_NOT_FOUND',
                    { agentId, availableAgents: await getAvailableAgents() }
                );
            } else if (result.error.includes("access denied")) {
                throw new AgentError(
                    `You don't have permission to use agent '${agentId}'.`,
                    'ACCESS_DENIED',
                    { agentId, userPermissions: await getUserPermissions() }
                );
            } else if (result.error.includes("configuration")) {
                throw new AgentError(
                    `Agent '${agentId}' has configuration issues.`,
                    'CONFIG_ERROR',
                    { agentId, error: result.error }
                );
            } else {
                throw new AgentError(
                    result.error,
                    'EXECUTION_ERROR',
                    { agentId, rawError: result.error }
                );
            }
        }
        
        return result;
        
    } catch (error) {
        if (error instanceof AgentError) {
            throw error;
        }
        
        // Handle HTTP and network errors
        if (error.message.includes('401')) {
            throw new AgentError(
                'Please log in to use agents.',
                'AUTHENTICATION_REQUIRED'
            );
        } else if (error.message.includes('403')) {
            throw new AgentError(
                'You don\'t have permission to access this agent.',
                'FORBIDDEN'
            );
        } else if (error.message.includes('429')) {
            throw new AgentError(
                'Too many requests. Please wait a moment and try again.',
                'RATE_LIMITED'
            );
        } else if (error.message.includes('500')) {
            throw new AgentError(
                'Server error. Please try again later.',
                'SERVER_ERROR'
            );
        } else if (error.message.includes('NetworkError') || error.message.includes('fetch')) {
            throw new AgentError(
                'Network connection failed. Please check your internet connection.',
                'NETWORK_ERROR'
            );
        } else {
            throw new AgentError(
                `Unexpected error: ${error.message}`,
                'UNKNOWN_ERROR',
                { originalError: error.message }
            );
        }
    }
};

// Usage with comprehensive error handling
const executeWithErrorHandling = async (agentId, request, mode) => {
    try {
        const result = await handleAgentCall(agentId, request, mode);
        return result;
    } catch (error) {
        console.error(`Agent Error [${error.code}]:`, error.message, error.details);
        
        // Show user-friendly error messages based on error type
        switch (error.code) {
            case 'AGENT_NOT_FOUND':
                showErrorNotification(`Agent not found. Did you mean: ${error.details.availableAgents?.slice(0, 3).join(', ')}?`);
                break;
            case 'ACCESS_DENIED':
                showErrorNotification('You need permission to use this agent. Contact your administrator.');
                break;
            case 'AUTHENTICATION_REQUIRED':
                redirectToLogin();
                break;
            case 'RATE_LIMITED':
                showErrorNotification('Too many requests. Please wait 30 seconds before trying again.');
                break;
            case 'NETWORK_ERROR':
                showErrorNotification('Connection failed. Please check your internet and try again.');
                break;
            default:
                showErrorNotification(`Something went wrong: ${error.message}`);
        }
        
        throw error;
    }
};
```

### Retry Logic

```javascript
const executeAgentWithRetry = async (agentId, request, mode = "execute", maxRetries = 3) => {
    let lastError;
    
    for (let attempt = 1; attempt <= maxRetries; attempt++) {
        try {
            return await executeAgent(agentId, request, mode);
        } catch (error) {
            lastError = error;
            
            // Don't retry on certain error types
            if (error.message.includes('401') || 
                error.message.includes('403') ||
                error.message.includes('Agent not found')) {
                throw error;
            }
            
            // Don't retry on last attempt
            if (attempt === maxRetries) {
                break;
            }
            
            // Exponential backoff
            const delay = Math.pow(2, attempt) * 1000;
            console.log(`Attempt ${attempt} failed, retrying in ${delay}ms...`);
            await new Promise(resolve => setTimeout(resolve, delay));
        }
    }
    
    throw new Error(`Agent execution failed after ${maxRetries} attempts: ${lastError.message}`);
};
```

## Best Practices

### 1. Agent Identification

```javascript
// ✅ Good: Use descriptive agent names or IDs
await executeAgent("B2B Sales Opportunity Analyzer", request);
await executeAgent("customer-service-bot", request);
await executeAgent("document-legal-reviewer", request);

// ❌ Avoid: Generic or unclear identifiers
await executeAgent("agent-123", request);
await executeAgent("bot", request);
await executeAgent("ai", request);
```

### 2. Request Optimization

```javascript
// ✅ Good: Clear, specific requests with context
await executeAgent(
    "document-reviewer",
    "Review this contract for potential legal issues and compliance concerns: [document content]",
    "execute"
);

await executeAgent(
    "sales-analyzer", 
    "Analyze Q4 sales data focusing on customer acquisition cost and lifetime value trends",
    "execute"
);

// ❌ Avoid: Vague or ambiguous requests
await executeAgent("lawyer-bot", "check this", "execute");
await executeAgent("analyst", "look at data", "execute");
```

### 3. Mode Selection Guidelines

```javascript
// Use "execute" mode for:
// - Specific tasks with clear outcomes
// - Data analysis and processing
// - Document analysis and summarization
// - Information retrieval
const report = await executeAgent("analyst", "Generate Q4 sales report", "execute");
const summary = await executeAgent("summarizer", "Summarize this research paper", "execute");

// Use "collaborate" mode for:
// - Open-ended problem solving
// - Strategy development
// - Learning and education
// - Creative projects
const strategy = await executeAgent("consultant", "Help me improve customer retention", "collaborate");
const brainstorm = await executeAgent("creative", "I need marketing campaign ideas", "collaborate");
```

### 4. Performance Optimization

```javascript
// Cache frequently used agents and responses
class AgentCache {
    constructor(ttl = 300000) { // 5 minutes default
        this.cache = new Map();
        this.ttl = ttl;
    }
    
    generateKey(agentId, request, mode) {
        return `${agentId}:${mode}:${this.hashString(request)}`;
    }
    
    hashString(str) {
        let hash = 0;
        for (let i = 0; i < str.length; i++) {
            const char = str.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash; // Convert to 32-bit integer
        }
        return hash;
    }
    
    get(agentId, request, mode) {
        const key = this.generateKey(agentId, request, mode);
        const cached = this.cache.get(key);
        
        if (cached && Date.now() - cached.timestamp < this.ttl) {
            return cached.data;
        }
        
        return null;
    }
    
    set(agentId, request, mode, data) {
        const key = this.generateKey(agentId, request, mode);
        this.cache.set(key, {
            data,
            timestamp: Date.now()
        });
    }
    
    clear() {
        this.cache.clear();
    }
}

const agentCache = new AgentCache();

const getCachedAgentResponse = async (agentId, request, mode) => {
    // Check cache first
    const cached = agentCache.get(agentId, request, mode);
    if (cached) {
        console.log('Using cached response');
        return cached;
    }
    
    // Execute agent if not cached
    const result = await executeAgent(agentId, request, mode);
    
    // Cache successful results
    if (result.success) {
        agentCache.set(agentId, request, mode, result);
    }
    
    return result;
};
```

### 5. Concurrent Agent Execution

```javascript
// Execute multiple agents concurrently for complex workflows
const analyzeBusinessComprehensively = async (businessData) => {
    const agents = [
        { id: "financial-analyst", task: `Analyze financial health: ${businessData.financial}`, mode: "execute" },
        { id: "market-researcher", task: `Research market position: ${businessData.market}`, mode: "execute" },
        { id: "operations-specialist", task: `Review operational efficiency: ${businessData.operations}`, mode: "execute" },
        { id: "risk-assessor", task: `Assess business risks: ${businessData.risks}`, mode: "execute" }
    ];
    
    // Execute all agents concurrently
    const results = await Promise.allSettled(
        agents.map(agent => executeAgent(agent.id, agent.task, agent.mode))
    );
    
    // Process results
    const analysis = {
        financial: null,
        market: null,
        operations: null,
        risks: null,
        errors: []
    };
    
    results.forEach((result, index) => {
        const agentType = agents[index].id.split('-')[0];
        
        if (result.status === 'fulfilled' && result.value.success) {
            analysis[agentType] = result.value.result;
        } else {
            analysis.errors.push({
                agent: agents[index].id,
                error: result.reason?.message || 'Unknown error'
            });
        }
    });
    
    return analysis;
};
```

### 6. Progressive Enhancement

```javascript
// Start with basic functionality, then enhance based on agent capabilities
const createProgressiveAgentInterface = (agentId) => {
    return {
        // Basic execution
        async execute(request) {
            return await executeAgent(agentId, request, "execute");
        },
        
        // Enhanced collaboration if supported
        async collaborate(request) {
            try {
                return await executeAgent(agentId, request, "collaborate");
            } catch (error) {
                // Fallback to execute mode if collaborate is not supported
                console.log('Collaborate mode not available, falling back to execute mode');
                return await executeAgent(agentId, request, "execute");
            }
        },
        
        // Streaming if supported by client
        async stream(request, onChunk, mode = "execute") {
            if (typeof ReadableStream !== 'undefined') {
                return await streamAgent(agentId, request, onChunk, mode);
            } else {
                // Fallback to regular execution for older browsers
                const result = await executeAgent(agentId, request, mode);
                onChunk({ type: 'response_chunk', content: result.result });
                onChunk({ type: 'complete', success: result.success });
                return result;
            }
        }
    };
};

// Usage
const salesAgent = createProgressiveAgentInterface("sales-assistant");
const result = await salesAgent.collaborate("Help me qualify this lead");
```

### 7. Monitoring and Analytics

```javascript
// Track agent usage and performance
class AgentAnalytics {
    constructor() {
        this.metrics = {
            calls: 0,
            successes: 0,
            failures: 0,
            totalExecutionTime: 0,
            agentUsage: new Map(),
            modeUsage: { execute: 0, collaborate: 0 }
        };
    }
    
    recordCall(agentId, mode, executionTime, success) {
        this.metrics.calls++;
        this.metrics.totalExecutionTime += executionTime;
        
        if (success) {
            this.metrics.successes++;
        } else {
            this.metrics.failures++;
        }
        
        // Track agent usage
        const currentUsage = this.metrics.agentUsage.get(agentId) || 0;
        this.metrics.agentUsage.set(agentId, currentUsage + 1);
        
        // Track mode usage
        this.metrics.modeUsage[mode]++;
    }
    
    getStats() {
        return {
            successRate: (this.metrics.successes / this.metrics.calls * 100).toFixed(2) + '%',
            averageExecutionTime: (this.metrics.totalExecutionTime / this.metrics.calls).toFixed(2) + 's',
            totalCalls: this.metrics.calls,
            mostUsedAgent: this.getMostUsedAgent(),
            modeDistribution: this.metrics.modeUsage
        };
    }
    
    getMostUsedAgent() {
        let maxUsage = 0;
        let mostUsed = null;
        
        for (const [agentId, usage] of this.metrics.agentUsage) {
            if (usage > maxUsage) {
                maxUsage = usage;
                mostUsed = agentId;
            }
        }
        
        return { agentId: mostUsed, usage: maxUsage };
    }
}

const analytics = new AgentAnalytics();

// Wrap executeAgent with analytics
const executeAgentWithAnalytics = async (agentId, request, mode = "execute") => {
    const startTime = Date.now();
    
    try {
        const result = await executeAgent(agentId, request, mode);
        const executionTime = (Date.now() - startTime) / 1000;
        
        analytics.recordCall(agentId, mode, executionTime, result.success);
        
        return result;
    } catch (error) {
        const executionTime = (Date.now() - startTime) / 1000;
        analytics.recordCall(agentId, mode, executionTime, false);
        throw error;
    }
};

// View analytics
console.log('Agent Analytics:', analytics.getStats());
```

## Quick Reference

### API Endpoints
| Endpoint | Method | Purpose | Response Type |
|----------|--------|---------|---------------|
| `/agent/execute` | POST | Single agent execution | JSON |
| `/agent/stream` | POST | Streaming agent execution | Server-Sent Events |

### Request Parameters
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `agent_id` | string | ✅ | - | Agent ID or name |
| `user_request` | string | ✅ | - | Task for agent |
| `agent_mode` | string | ❌ | "execute" | "collaborate" or "execute" |
| `llm_provider` | string | ❌ | "anthropic" | "anthropic" or "openai" |
| `model` | string | ❌ | null | Specific model name |

### Response Structure
```typescript
interface AgentResponse {
    success: boolean;
    result: string;
    agent_id: string;
    agent_type: string;
    execution_time: number;
    tasks_completed: number;
    error?: string;
    metadata?: {
        org_id: string;
        user_id: string;
        tools_used: string[];
        specialized_domains: string[];
    };
}
```

### Streaming Chunk Types
| Type | Description | Content |
|------|-------------|---------|
| `status` | System status updates | Loading messages |
| `thinking` | Agent reasoning process | Thought process |
| `response_chunk` | Partial response content | Incremental text |
| `complete` | Execution finished | Success confirmation |
| `error` | Error occurred | Error message |

### Common Agent Types
- `sales-assistant` - Sales support and lead qualification
- `document-analyzer` - Document processing and analysis
- `customer-service` - Customer support and troubleshooting
- `data-analyst` - Data analysis and reporting
- `content-writer` - Content creation and editing
- `business-consultant` - Strategic planning and advice

### Authentication
All agent endpoints require Bearer token authentication:
```javascript
headers: {
    'Authorization': `Bearer ${localStorage.getItem('token')}`
}
```

### Error Codes
| HTTP Code | Description | Action |
|-----------|-------------|---------|
| 401 | Authentication required | Redirect to login |
| 403 | Access denied | Check permissions |
| 404 | Agent not found | Verify agent ID |
| 429 | Rate limited | Wait and retry |
| 500 | Server error | Try again later |

---

## Conclusion

The Agent.py system provides a powerful, flexible foundation for building AI-powered applications. With dynamic configuration loading, dual operational modes, and comprehensive tooling support, agents can be tailored to specific use cases while maintaining consistent, reliable performance.

Key benefits:
- **Dynamic Configuration**: No hardcoded prompts or tools
- **Dual Modes**: Adaptable interaction styles
- **Real-time Streaming**: Responsive user experience
- **Comprehensive Error Handling**: Robust production readiness
- **Google Drive Integration**: Seamless document processing
- **Multi-tenant Support**: Organization-based access control

Start with the basic examples and gradually incorporate advanced patterns as your application grows. The system is designed to scale from simple agent interactions to complex multi-agent workflows.

**Happy building!** 🚀 