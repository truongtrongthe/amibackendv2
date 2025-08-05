# Frontend Guide: Conversation-First Agent Creation

## 🎯 **NEW WORKFLOW: Conversation → Creation → Refinement**

The `/ami/collaborate` endpoint now supports **conversation-first flow** with LLM-based approval detection.

---

## 🚀 **Three Modes of Operation**

### **1. Conversation Mode** (No agent created yet)
- User explores ideas with AMI
- AMI asks clarifying questions  
- Messages saved to chat sessions
- **NO agent creation until approval**

### **2. Creation Mode** (LLM detects approval)
- LLM detects user approval intent
- Creates agent from conversation context
- Returns `agent_id` and `blueprint_id`

### **3. Refinement Mode** (Has agent/blueprint)
- Standard blueprint refinement
- Modify existing agent capabilities

---

## 💻 **Frontend Implementation**

### **Request Structure** (Updated)
```javascript
// Same endpoint for all modes
const collaborateRequest = {
  user_input: "User's message",                    // ✅ Always required
  conversation_id: "chat-session-id",             // ✅ For chat integration
  org_id: "organization-id",                      // ✅ Required  
  user_id: "user-id",                             // ✅ Required
  llm_provider: "anthropic",                      // ✅ Optional
  
  // ✅ NEW: Only include these for refinement mode
  agent_id: "existing-agent-id",                  // Only when refining
  blueprint_id: "existing-blueprint-id"           // Only when refining
};
```

### **Response Modes**

#### **Conversation Mode Response**
```json
{
  "success": true,
  "conversation_id": "chat-session-id",
  "current_state": "initial_idea",
  "ami_message": "Great idea! What specific tasks should this agent handle? Who will be using it?",
  "data": {
    "mode": "conversation",
    "suggestions": [
      "What problem should this agent solve?",
      "What tools should it integrate with?"
    ],
    "agent_concept": "Sales automation agent concept",
    "context": "exploring_ideas"
  },
  "next_actions": [
    "Answer AMI's questions",
    "Provide more details about your needs",
    "Say 'let's build this' when ready to create"
  ]
}
```

#### **Creation Mode Response** (When approved)
```json
{
  "success": true,
  "conversation_id": "chat-session-id", 
  "current_state": "skeleton_review",
  "ami_message": "Perfect! I've created 'Sales Assistant Agent' for you. Let me know if you'd like to refine any aspects.",
  "agent_id": "newly-created-agent-id",           // ✨ NEW: For follow-ups
  "blueprint_id": "newly-created-blueprint-id",   // ✨ NEW: For follow-ups
  "data": {
    "mode": "created",
    "agent_name": "Sales Assistant Agent",
    "agent_concept": "AI assistant for Vietnamese sales processes",
    "next_phase": "refinement"
  },
  "next_actions": [
    "Refine the agent's capabilities",
    "Add specific integrations",
    "Approve the agent for use"
  ]
}
```

---

## 🔄 **Complete Frontend Flow**

### **Chat Integration Pattern**
```javascript
class AgentCollaborationManager {
  constructor() {
    this.chatId = null;
    this.agentId = null;
    this.blueprintId = null;
    this.conversationMode = 'conversation'; // 'conversation', 'created', 'refinement'
  }

  async sendMessage(userInput) {
    // 1. Save user message to chat first
    if (this.chatId) {
      await this.saveUserMessageToChat(userInput);
    } else {
      this.chatId = await this.createNewChatSession();
      await this.saveUserMessageToChat(userInput);
    }

    // 2. Send to AMI collaborate endpoint
    const response = await fetch('/ami/collaborate', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${userToken}`
      },
      body: JSON.stringify({
        user_input: userInput,
        conversation_id: this.chatId,           // ✨ Link to chat session
        agent_id: this.agentId,                 // Only when refining
        blueprint_id: this.blueprintId,         // Only when refining
        org_id: currentOrgId,
        user_id: currentUserId,
        llm_provider: "anthropic"
      })
    });

    const result = await response.json();
    
    // 3. Handle different modes
    if (result.success) {
      await this.handleAMIResponse(result);
    }
  }

  async handleAMIResponse(result) {
    const mode = result.data?.mode || 'conversation';
    
    switch (mode) {
      case 'conversation':
        // Still exploring ideas
        this.displayConversationMode(result);
        break;
        
      case 'created':
        // Agent was created!
        this.agentId = result.agent_id;
        this.blueprintId = result.blueprint_id;
        this.conversationMode = 'refinement';
        this.displayAgentCreated(result);
        break;
        
      case 'refinement':
        // Refining existing agent
        this.displayRefinementMode(result);
        break;
    }
  }

  displayConversationMode(result) {
    // Show AMI's questions and suggestions
    this.addMessageToUI('ami', result.ami_message);
    
    if (result.data.suggestions) {
      this.showSuggestions(result.data.suggestions);
    }
    
    // Show current agent concept if available
    if (result.data.agent_concept) {
      this.showAgentConcept(result.data.agent_concept);
    }
  }

  displayAgentCreated(result) {
    // Celebrate! Agent was created
    this.addMessageToUI('ami', result.ami_message);
    this.showAgentCreatedSuccess(result.data.agent_name);
    
    // Now switch to refinement mode UI
    this.switchToRefinementMode();
  }

  displayRefinementMode(result) {
    // Standard blueprint refinement UI
    this.addMessageToUI('ami', result.ami_message);
    this.showBlueprintRefinementOptions(result.data);
  }

  async saveUserMessageToChat(message) {
    await fetch('/api/chats/messages', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${userToken}`
      },
      body: JSON.stringify({
        chat_id: this.chatId,
        sender: 'user',
        content: message,
        message_type: 'text'
      })
    });
  }

  async createNewChatSession() {
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
  }
}
```

### **UI State Management**
```javascript
const [collaborationState, setCollaborationState] = useState({
  mode: 'conversation',        // 'conversation' | 'created' | 'refinement'
  chatId: null,
  agentId: null,
  blueprintId: null,
  messages: [],
  agentConcept: null,
  suggestions: []
});

const handleUserMessage = async (userInput) => {
  // Add user message to UI immediately
  setCollaborationState(prev => ({
    ...prev,
    messages: [...prev.messages, { sender: 'user', content: userInput }]
  }));

  // Send to collaboration endpoint
  const manager = new AgentCollaborationManager();
  manager.chatId = collaborationState.chatId;
  manager.agentId = collaborationState.agentId;
  manager.blueprintId = collaborationState.blueprintId;
  
  await manager.sendMessage(userInput);
};
```

---

## 🚨 **Key Changes for Frontend**

### ✅ **What Works Now**
1. **Conversation Mode**: Ask questions, explore ideas (no agent created)
2. **LLM Approval Detection**: AI detects when user is ready to create
3. **Chat Integration**: Messages saved to chat sessions automatically  
4. **Seamless Transitions**: Conversation → Creation → Refinement

### 🔄 **What Changed**
1. **`conversation_id`**: Now links to chat sessions for context
2. **Response Modes**: Different responses based on conversation phase
3. **Agent Creation Timing**: Only creates agent when user approves
4. **Context Awareness**: LLM uses full conversation for better agents

### 📋 **Frontend Checklist**
- [ ] Update request structure to include `conversation_id`
- [ ] Handle different response modes (`conversation`, `created`, `refinement`)
- [ ] Integrate with existing chat system
- [ ] Save user messages to chat before sending to AMI
- [ ] Show different UI based on collaboration mode
- [ ] Store `agent_id` and `blueprint_id` when agent is created

---

## 💡 **Example Complete Flow**

```javascript
// 1. User starts conversation
await manager.sendMessage("I need a sales agent for Vietnamese market");
// → Conversation mode: AMI asks questions

// 2. User provides details  
await manager.sendMessage("It should handle lead qualification and use WhatsApp");
// → Still conversation mode: AMI asks more questions

// 3. User gives approval
await manager.sendMessage("Yes, let's build this agent!");
// → Creation mode: Agent created, returns agent_id/blueprint_id

// 4. User refines agent
await manager.sendMessage("Add CRM integration");
// → Refinement mode: Modifies existing agent blueprint
```

**The system now feels natural and conversational while preventing database pollution!** 🎉