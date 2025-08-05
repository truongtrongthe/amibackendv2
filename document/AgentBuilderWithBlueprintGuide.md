# 🏗️ Agent Builder - Conversation-First Frontend Integration Guide

## 📋 **Overview**

This guide provides frontend integration documentation for the **Conversation-First Agent Builder**. The system enables users to create production-ready AI agents through an intelligent, guided conversation that only creates agents when users are truly ready.

## 🚀 **NEW: Conversation-First Architecture**

**✨ NEW PARADIGM:** No more database pollution! AMI now has three intelligent modes:

### **🧠 Three-Mode System:**

#### **1. Conversation Mode** (No agent created)
- Explore ideas and ask questions
- AMI guides and educates without creating anything
- Messages saved to chat sessions for context

#### **2. Creation Mode** (LLM detects approval)
- Smart LLM detects when user is ready to create
- Creates agent from full conversation context
- Returns agent_id and blueprint_id for refinement

#### **3. Refinement Mode** (Has existing agent)
- Standard blueprint refinement with context awareness
- Modify existing agent capabilities

### **🎯 Key Benefits:**

1. **🗣️ Natural Conversation**: Users explore ideas before committing to creation
2. **🧠 LLM Intelligence**: Smart approval detection prevents database pollution  
3. **📱 Chat Integration**: Messages saved to chat sessions for context
4. **🎯 Context-Aware Creation**: Agents built from full conversation understanding
5. **⚡ Seamless Transitions**: Conversation → Creation → Refinement

### **Complete User Journey:**
```
User: "I need a sales agent"
AMI: "What tasks? What integrations?" (CONVERSATION MODE)
User: "Gmail integration for Vietnamese market"  
AMI: "What about CRM? Language preferences?" (STILL CONVERSATION)
User: "Let's build this!"
AMI: Creates agent from conversation → Returns agent_id (CREATION MODE)
User: "Add WhatsApp integration"  
AMI: Refines existing agent (REFINEMENT MODE)
```

---

## 💻 **Frontend Integration - Conversation-First Flow**

### **🚀 Core Component Structure**

```jsx
const ConversationFirstAMI = () => {
  const [collaborationState, setCollaborationState] = useState({
    mode: 'conversation',        // 'conversation' | 'created' | 'refinement'
    chatId: null,
    agentId: null,
    blueprintId: null,
    messages: []
  });
  
  const sendMessage = async (userInput) => {
    // 1. Save user message to chat first
    await saveUserMessageToChat(userInput);
    
    // 2. Send to AMI collaborate endpoint  
    const response = await fetch('/ami/collaborate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        user_input: userInput,
        conversation_id: collaborationState.chatId,     // For chat context
        agent_id: collaborationState.agentId,           // Only for refinement
        blueprint_id: collaborationState.blueprintId,   // Only for refinement
        org_id: currentOrgId,
        user_id: currentUserId
      })
    });
    
    const result = await response.json();
    
    // 3. Handle different response modes
    handleModeResponse(result);
  };
  
  const handleModeResponse = (result) => {
    const mode = result.data?.mode || 'conversation';
    
    switch (mode) {
      case 'conversation':
        // Still exploring ideas - show suggestions
        setCollaborationState(prev => ({
          ...prev,
          messages: [...prev.messages, { sender: 'ami', content: result.ami_message }]
        }));
        showSuggestions(result.data.suggestions);
        break;
        
      case 'created':
        // Agent was created! Switch to refinement mode
        setCollaborationState(prev => ({
          ...prev,
          mode: 'refinement',
          agentId: result.agent_id,
          blueprintId: result.blueprint_id,
          messages: [...prev.messages, { sender: 'ami', content: result.ami_message }]
        }));
        showAgentCreatedSuccess(result.data.agent_name);
        break;
        
      case 'refinement':
        // Refining existing agent
        setCollaborationState(prev => ({
          ...prev,
          messages: [...prev.messages, { sender: 'ami', content: result.ami_message }]
        }));
        break;
    }
  };
  
  return (
    <div className="conversation-first-ami">
      <ConversationModeIndicator mode={collaborationState.mode} />
      <MessageHistory messages={collaborationState.messages} />
      <MessageInput onSend={sendMessage} />
    </div>
  );
};
```

---

## 🎨 **Recommended UX Approach: Conversation-First**

### **Core User Experience**
**Natural conversation → Smart approval detection → Context-rich agent creation**

This approach prevents database pollution by:
- ✅ **No premature creation** - Agents only created when user approves
- ✅ **LLM intelligence** - Smart detection of user intent and readiness
- ✅ **Rich context** - Agents built from full conversation understanding  
- ✅ **Chat persistence** - All conversations saved for context

---

## 🚀 **Simple Landing Page Implementation**

```jsx
const ConversationFirstLanding = () => {
  const [userInput, setUserInput] = useState('');
  
  const startConversation = async () => {
    const chatId = await createNewChatSession();
    await saveUserMessageToChat(userInput, chatId);
    
    // Navigate to conversation view
    navigate('/agent-builder/collaborate', { 
      state: { chatId, initialMessage: userInput }
    });
  };
  
  return (
    <div className="conversation-landing">
      <div className="hero">
        <h1>🤖 Let's Build Your AI Agent</h1>
        <p>Describe what you need and I'll help you explore the possibilities</p>
        
        <textarea
          value={userInput}
          onChange={(e) => setUserInput(e.target.value)}
          placeholder="I need help with..."
          className="idea-input"
        />
        
        <button onClick={startConversation} disabled={!userInput.trim()}>
          Start Conversation with AMI
        </button>
      </div>
    </div>
  );
};
```

---

## 🎯 **API Integration Guide**

### **Main Collaboration Endpoint**

**Endpoint:** `POST /ami/collaborate`

**Request Structure:**
```javascript
const collaborateWithAMI = async (userInput, chatId, agentId = null, blueprintId = null) => {
  const response = await fetch('/ami/collaborate', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${userToken}`
    },
    body: JSON.stringify({
      user_input: userInput,
      conversation_id: chatId,           // For chat context
      agent_id: agentId,                 // Only for refinement mode
      blueprint_id: blueprintId,         // Only for refinement mode
      org_id: currentOrgId,
      user_id: currentUserId,
      llm_provider: "anthropic"
    })
  });
  
  return response.json();
};
```

### **Response Types by Mode**

#### **Conversation Mode Response**
```json
{
  "success": true,
  "conversation_id": "chat-session-id",
  "current_state": "initial_idea",
  "ami_message": "Great idea! What specific tasks should this agent handle?",
  "data": {
    "mode": "conversation",
    "suggestions": ["What problem should it solve?", "What tools needed?"],
    "agent_concept": "Sales automation concept"
  },
  "next_actions": ["Answer AMI's questions", "Provide more details"]
}
```

#### **Creation Mode Response**
```json
{
  "success": true,
  "conversation_id": "chat-session-id",
  "current_state": "skeleton_review", 
  "ami_message": "Perfect! I've created 'Sales Agent' for you.",
  "agent_id": "newly-created-agent-id",
  "blueprint_id": "newly-created-blueprint-id",
  "data": {
    "mode": "created",
    "agent_name": "Sales Agent",
    "next_phase": "refinement",
    "agent_blueprint": {
      "agent_name": "Vietnamese Sales Assistant",
      "agent_purpose": "Sales agent for Vietnamese market with email integration",
      "target_users": "Sales team targeting Vietnamese clients",
      "agent_type": "sales",
      "language": "vietnamese",
      "meet_me": {
        "introduction": "Hi, I'm Vietnamese Sales Assistant! I help boost your sales with smart email outreach.",
        "value_proposition": "Think of me as your sales automation expert who speaks Vietnamese."
      },
      "what_i_do": {
        "primary_tasks": [
          {"task": "Email automation", "description": "I send personalized sales emails in Vietnamese"},
          {"task": "Lead qualification", "description": "I qualify leads based on your criteria"}
        ],
        "personality": {
          "tone": "friendly",
          "style": "professional",
          "analogy": "like a dedicated sales assistant who never sleeps"
        },
        "sample_conversation": {
          "user_question": "Can you send follow-up emails to potential clients?",
          "agent_response": "Absolutely! I'll send personalized follow-up emails in Vietnamese, tracking engagement and scheduling follow-ups based on responses."
        }
      },
      "knowledge_sources": [
        {
          "source": "Customer Database",
          "type": "database",
          "update_frequency": "real-time",
          "content_examples": ["customer profiles", "interaction history"]
        }
      ],
      "integrations": [
        {
          "app_name": "Gmail",
          "trigger": "When sending sales emails",
          "action": "I compose and send personalized emails with tracking"
        }
      ],
      "monitoring": {
        "reporting_method": "Weekly sales performance reports",
        "metrics_tracked": ["email open rates", "response rates", "conversion rates"],
        "fallback_response": "Let me check our CRM for the latest information",
        "escalation_method": "I'll notify the sales manager for complex requests"
      },
      "test_scenarios": [
        {
          "question": "Send a follow-up email to John Doe about our product demo",
          "expected_response": "I'll send a personalized follow-up email to John Doe in Vietnamese, referencing the product demo and suggesting next steps."
        }
      ],
      "workflow_steps": [
        "You request email outreach or lead qualification",
        "I analyze the customer data and context",
        "I compose personalized messages in Vietnamese",
        "I send emails and track engagement",
        "I provide reports and recommendations"
      ],
      "visual_flow": "Request → Data Analysis → Message Composition → Email Sending → Performance Tracking"
    }
  },
  "next_actions": ["Refine capabilities", "Add integrations", "Approve design"]
}
```

### **Chat Integration**

```javascript
// Create new chat session
const createNewChatSession = async () => {
  const response = await fetch('/api/chats/', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${userToken}`
    },
    body: JSON.stringify({
      title: 'Agent Creation Chat',
      user_id: currentUserId,
      org_id: currentOrgId
    })
  });
  
  const chat = await response.json();
  return chat.id;
};

// Save user message to chat
const saveUserMessageToChat = async (message, chatId) => {
  await fetch('/api/chats/messages', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${userToken}`
    },
    body: JSON.stringify({
      chat_id: chatId,
      sender: 'user',
      content: message,
      message_type: 'text'
    })
  });
};
```

#### **Refinement Mode Response**
```json
{
  "success": true,
  "conversation_id": "chat-session-id",
  "current_state": "skeleton_review",
  "ami_message": "I've enhanced your Sales Agent with WhatsApp integration. The agent can now send follow-ups via both email and WhatsApp.",
  "agent_id": "existing-agent-id",
  "blueprint_id": "existing-blueprint-id", 
  "data": {
    "mode": "refinement",
    "changes_made": ["Added WhatsApp integration", "Enhanced multi-channel communication"],
    "updated_blueprint": {
      // ... complete updated blueprint structure
    },
    "context": {
      "agent_identity": {
        "name": "Vietnamese Sales Assistant",
        "purpose": "Multi-channel sales agent with email and WhatsApp",
        "completeness_score": 85
      },
      "current_capabilities": {
        "task_count": 3,
        "knowledge_count": 1, 
        "integration_count": 2
      }
    }
  },
  "next_actions": ["Add more integrations", "Define success metrics", "Approve for compilation"]
}
```

---

## 🏗️ **Agent Blueprint Structure**

### **Complete Blueprint Format**
Every agent created has a comprehensive blueprint with these sections:

```json
{
  "agent_name": "Agent Name",
  "agent_purpose": "Clear description of what the agent does",
  "target_users": "Who will use this agent",
  "agent_type": "Type category (sales, support, analysis, etc.)",
  "language": "Primary language for communication",
  
  "meet_me": {
    "introduction": "First-person introduction from the agent",
    "value_proposition": "What value the agent provides"
  },
  
  "what_i_do": {
    "primary_tasks": [
      {
        "task": "Task name",
        "description": "Detailed description of what this task involves"
      }
    ],
    "personality": {
      "tone": "Communication tone (friendly, professional, etc.)",
      "style": "Communication style (concise, detailed, etc.)",
      "analogy": "How the agent describes itself"
    },
    "sample_conversation": {
      "user_question": "Example user question",
      "agent_response": "How the agent would respond"
    }
  },
  
  "knowledge_sources": [
    {
      "source": "Data source name",
      "type": "Type of source (database, api, file, etc.)",
      "update_frequency": "How often data is updated",
      "content_examples": ["List of content types"]
    }
  ],
  
  "integrations": [
    {
      "app_name": "Integration name (Gmail, Slack, etc.)",
      "trigger": "When this integration is used", 
      "action": "What the agent does with this integration"
    }
  ],
  
  "monitoring": {
    "reporting_method": "How the agent reports its activities",
    "metrics_tracked": ["List of metrics the agent tracks"],
    "fallback_response": "What agent says when it can't help",
    "escalation_method": "How agent escalates complex issues"
  },
  
  "test_scenarios": [
    {
      "question": "Test question for the agent",
      "expected_response": "Expected agent response"
    }
  ],
  
  "workflow_steps": [
    "Step 1: First thing agent does",
    "Step 2: Second thing agent does",
    "etc..."
  ],
  
  "visual_flow": "Simple description of the agent's workflow"
}
```

---

## 🚀 **Complete Implementation Example**

```javascript
// Complete conversation-first flow example
class AgentCollaborationManager {
  constructor() {
    this.chatId = null;
    this.agentId = null;
    this.blueprintId = null;
    this.mode = 'conversation';
  }

  async startConversation(userInput) {
    // Create chat session
    this.chatId = await createNewChatSession();
    
    // Save user message
    await saveUserMessageToChat(userInput, this.chatId);
    
    // Start collaboration
    return await this.sendToAMI(userInput);
  }

  async sendToAMI(userInput) {
    const response = await collaborateWithAMI(
      userInput, 
      this.chatId, 
      this.agentId, 
      this.blueprintId
    );
    
    // Handle mode changes
    if (response.data?.mode === 'created') {
      this.agentId = response.agent_id;
      this.blueprintId = response.blueprint_id;
      this.mode = 'refinement';
    }
    
    return response;
  }
}

// Usage
const manager = new AgentCollaborationManager();
const result = await manager.startConversation("I need a sales agent");
```

---

## 🎨 **Frontend UI Components**

### **Mode Indicator Component**
```jsx
const ConversationModeIndicator = ({ mode, agentName }) => {
  const modeConfig = {
    conversation: { icon: '🗣️', label: 'Exploring Ideas', color: 'blue' },
    created: { icon: '🎉', label: 'Agent Created!', color: 'green' },
    refinement: { icon: '🔧', label: `Refining ${agentName}`, color: 'orange' }
  };
  
  const config = modeConfig[mode] || modeConfig.conversation;
  
  return (
    <div className={`mode-indicator ${config.color}`}>
      <span className="mode-icon">{config.icon}</span>
      <span className="mode-label">{config.label}</span>
    </div>
  );
};
```

### **Message Display Component**
```jsx
const MessageHistory = ({ messages }) => (
  <div className="message-history">
    {messages.map((message, index) => (
      <div key={index} className={`message ${message.sender}`}>
        <div className="message-header">
          <span className="sender">
            {message.sender === 'ami' ? '🤖 AMI' : '👤 You'}
          </span>
        </div>
        <div className="message-content">{message.content}</div>
      </div>
    ))}
  </div>
);
```

---

## 🛠️ **Essential CSS Styling**

```css
.conversation-first-ami {
  max-width: 800px;
  margin: 0 auto;
  padding: 1rem;
}

.mode-indicator {
  padding: 0.5rem 1rem;
  border-radius: 8px;
  margin-bottom: 1rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.mode-indicator.blue { background: #dbeafe; color: #1e40af; }
.mode-indicator.green { background: #dcfce7; color: #166534; }
.mode-indicator.orange { background: #fed7aa; color: #c2410c; }

.message-history {
  max-height: 400px;
  overflow-y: auto;
  margin-bottom: 1rem;
}

.message {
  margin-bottom: 1rem;
  padding: 1rem;
  border-radius: 8px;
}

.message.user { 
  background: #f3f4f6; 
  margin-left: 2rem;
}

.message.ami { 
  background: #dbeafe; 
  margin-right: 2rem;
}

.conversation-landing textarea {
  width: 100%;
  min-height: 120px;
  padding: 1rem;
  border: 2px solid #e5e7eb;
  border-radius: 8px;
  resize: vertical;
}

.conversation-landing button {
  background: #2563eb;
  color: white;
  padding: 0.75rem 1.5rem;
  border: none;
  border-radius: 8px;
  font-weight: 600;
  cursor: pointer;
  margin-top: 1rem;
}

.conversation-landing button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
```

---

## ⚡ **Quick Start Checklist**

### **Frontend Implementation Checklist**
- [ ] **Landing Page**: Simple textarea + "Start Conversation" button
- [ ] **Mode Indicator**: Show current conversation mode (conversation/created/refinement)
- [ ] **Message History**: Display chat messages with AMI
- [ ] **Chat Integration**: Create chat sessions and save user messages
- [ ] **Response Handling**: Handle different response modes appropriately
- [ ] **Agent Creation Success**: Show celebration when agent is created
- [ ] **Context Switching**: Seamlessly move between conversation and refinement modes

### **API Integration Checklist**
- [ ] **Main Endpoint**: `POST /ami/collaborate` with conversation-first flow
- [ ] **Chat Endpoints**: Create chat sessions and save messages
- [ ] **Response Parsing**: Handle conversation, creation, and refinement responses
- [ ] **Error Handling**: Graceful fallbacks for API failures
- [ ] **Loading States**: Show loading during AMI processing

### **Success Metrics**
- **Conversation Engagement**: Average messages before agent creation
- **Creation Success Rate**: % of conversations that result in agent creation
- **Database Hygiene**: Reduction in draft/unused agents
- **User Experience**: Time to functional agent, user satisfaction scores

---

## 🎯 **Summary**

The **Conversation-First Agent Builder** revolutionizes agent creation by:

✨ **Preventing Database Pollution** - No agents created until user approval  
🧠 **Smart LLM Detection** - Intelligently detects when users are ready  
💬 **Natural Conversations** - Users explore ideas before committing  
📱 **Chat Integration** - All messages preserved for context  
🎯 **Context-Rich Creation** - Agents built from full conversation understanding

### **The Result**

Users now experience a natural, conversational flow that feels like talking to an expert colleague who understands their needs before building anything. No more database clutter, no more premature agent creation - just intelligent, contextual agent building.

---

## 📋 **Todo System & Agent Implementation**

### **When Agent is Approved**
After the user approves the agent blueprint, the system generates implementation todos:

```json
{
  "success": true,
  "state": "building",
  "ami_message": "🎉 Perfect! I've created your agent. Now let's set up the integrations.",
  "data": {
    "agent_id": "agent_123",
    "blueprint_id": "blueprint_456",
    "todos_generated": 3,
    "implementation_todos": [
      {
        "id": "todo_1",
        "title": "Configure Gmail Integration",
        "description": "Set up Gmail API credentials for email sending",
        "category": "integration_setup",
        "priority": "high",
        "status": "pending",
        "input_required": {
          "type": "oauth_credentials",
          "fields": [
            {
              "name": "gmail_api_key",
              "type": "string", 
              "required": true,
              "description": "Gmail API key from Google Console"
            },
            {
              "name": "client_secret",
              "type": "password",
              "required": true,
              "description": "OAuth 2.0 Client Secret"
            }
          ]
        },
        "tool_instructions": {
          "tool_name": "Gmail API",
          "how_to_call": "Use Gmail API v1 with OAuth2",
          "when_to_use": "For sending emails and managing contacts",
          "expected_output": "Successful email delivery confirmation"
        }
      }
    ]
  }
}
```

### **Todo Management Endpoints**

#### **Get Todos**
```javascript
const getTodos = async (agentId, blueprintId) => {
  const response = await fetch(`/org-agents/${agentId}/blueprints/${blueprintId}/todos`, {
    headers: { 'Authorization': `Bearer ${userToken}` }
  });
  return response.json();
};
```

#### **Submit Todo Inputs**
```javascript
const submitTodoInputs = async (agentId, blueprintId, todoId, inputs) => {
  const response = await fetch(
    `/org-agents/${agentId}/blueprints/${blueprintId}/todos/${todoId}/collect-inputs`,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${userToken}`
      },
      body: JSON.stringify({ collected_inputs: inputs })
    }
  );
  return response.json();
};
```

### **Todo UI Implementation**
```jsx
const TodoCard = ({ todo, onComplete }) => {
  const [inputs, setInputs] = useState({});
  
  return (
    <div className={`todo-card ${todo.status}`}>
      <h4>{todo.title}</h4>
      <p>{todo.description}</p>
      
      {todo.input_required && (
        <div className="input-form">
          {todo.input_required.fields.map(field => (
            <div key={field.name} className="form-field">
              <label>{field.description}</label>
              <input
                type={field.type === 'password' ? 'password' : 'text'}
                value={inputs[field.name] || ''}
                onChange={(e) => setInputs(prev => ({
                  ...prev,
                  [field.name]: e.target.value
                }))}
                required={field.required}
              />
            </div>
          ))}
          
          <button onClick={() => onComplete(inputs)}>
            Submit Configuration
          </button>
        </div>
      )}
    </div>
  );
};
```

---

## ⚙️ **Agent Compilation & Production**

### **Compilation Process**
Once all todos are completed, compile the agent:

```javascript
const compileAgent = async (agentId, blueprintId) => {
  const response = await fetch(
    `/org-agents/${agentId}/blueprints/${blueprintId}/compile`,
    {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${userToken}` }
    }
  );
  return response.json();
};

// Compilation Response
{
  "success": true,
  "blueprint": {
    "id": "blueprint_456",
    "compilation_status": "compiled",
    "compiled_at": "2024-01-15T14:30:00Z",
    "compiled_system_prompt": "You are Vietnamese Sales Assistant...\n\n# CONFIGURATIONS\n- Gmail: [CONFIGURED]\n- API Keys: [SECURE]\n\n# INSTRUCTIONS\n## Email Sending\n**When to use:** For sales outreach\n**How to use:** Use Gmail API with provided credentials..."
  },
  "message": "Agent compiled and ready for production use!"
}
```

### **Production Agent Access**
```javascript
const getProductionAgent = async (agentId) => {
  const response = await fetch(`/org-agents/${agentId}`, {
    headers: { 'Authorization': `Bearer ${userToken}` }
  });
  return response.json();
};

// Agent is now ready with:
// ✅ Complete system prompt with configurations
// ✅ Working tool integrations with credentials  
// ✅ Cultural context and language preferences
// ✅ Monitoring and escalation procedures
```

---

## 🎯 **Complete Frontend Workflow**

### **1. Conversation Phase**
```javascript
// Start conversation
const result = await collaborateWithAMI("I need a sales agent", chatId);
// Handle conversation responses, show suggestions
```

### **2. Creation Phase**  
```javascript
// When LLM detects approval
if (result.data.mode === 'created') {
  // Show agent created success
  // Display blueprint preview
  // Store agent_id and blueprint_id for refinement
}
```

### **3. Refinement Phase**
```javascript
// Continue refining with agent context
const refinement = await collaborateWithAMI(
  "Add WhatsApp integration", 
  chatId, 
  agentId, 
  blueprintId
);
```

### **4. Implementation Phase**
```javascript
// Get and display todos
const todos = await getTodos(agentId, blueprintId);
// User completes todos
await submitTodoInputs(agentId, blueprintId, todoId, inputs);
```

### **5. Compilation Phase**
```javascript
// When all todos complete
await compileAgent(agentId, blueprintId);
// Agent is now production-ready!
```

**Ready to implement the future of AI agent creation!** 🚀